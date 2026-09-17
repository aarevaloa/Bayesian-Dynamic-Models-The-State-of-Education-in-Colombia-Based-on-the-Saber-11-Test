# -*- coding: utf-8 -*-
import json, os, time
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================== CONFIGURACION ================================
RUTA_BASE   = "C:/Users/USUARIO/OneDrive - Universidad Nacional de Colombia/Tesis/Escritorio/Códigos/Modelos/Modelo 4"                                  # carpeta del Modelo 4
RUTA_EXPORT = os.path.join(RUTA_BASE, "export_python")
DIR_SALIDA  = os.path.join(RUTA_BASE, "diagnosticos_M4")

N_REP_PPC   = 200      # replicas predictivas (como n_rep del modulo R)
S_MAX_LIK   = 800      # draws (adelgazados) para WAIC/DIC
S_R2        = 200      # draws para el R2 bayesiano
N_KDE_MAX   = 60000    # submuestra de obs para las densidades del PPC
DPI         = 200
SEMILLA     = 2026

# Umbrales del veredicto (identicos a 10_diagnostico_M4.R)
U_RHAT_VEREDICTO = 1.10
U_ESS_VEREDICTO  = 100
U_TASA_MH        = (0.15, 0.60)
U_PPC_EXTREMOS   = 2
U_RHAT_AVISO     = 1.05    # conteos informativos del modulo R
U_ESS_AVISO      = 400

rng = np.random.default_rng(SEMILLA)
os.makedirs(DIR_SALIDA, exist_ok=True)
t_ini = time.time()

# ============================== CARGA DEL EXPORT =============================
def _leer(f, dtype="<f8"):
    return np.fromfile(os.path.join(RUTA_EXPORT, f), dtype=dtype)

with open(os.path.join(RUTA_EXPORT, "meta.json"), encoding="utf-8") as fh:
    meta = json.load(fh)

p, Tt, m, S = meta["p"], meta["Tt"], meta["m"], meta["S"]
n_k     = np.asarray(meta["n_k"], int)
N_t     = np.asarray(meta["N_t"], int)
idx_mu  = meta["idx_mu"] - 1                       # a base 0
bloques = {B: np.asarray(meta[f"idx_{B}"], int) - 1 for B in ("E", "M", "D")}
labels  = meta["theta_labels"]
deptos  = [str(d) for d in meta["depto_ids"]]
tiene_datos = bool(meta["tiene_datos"])

theta   = _leer("theta.bin").reshape((p, Tt + 1, S), order="F")
nu      = _leer("nu.bin")
phi2    = _leer("phi2.bin").reshape((S, 3), order="F")      # E, M, D
lambda2 = _leer("lambda2.bin").reshape((S, 3), order="F")
v_mean  = _leer("v_mean.bin").reshape((int(n_k.max()), m), order="F")
phi_mean = _leer("phi_mean.bin").reshape((m, Tt), order="F")
COLB = {"E": 0, "M": 1, "D": 2}

print(f"Export cargado: p={p}, Tt={Tt}, m={m}, S={S}, tiene_datos={tiene_datos}")

if tiene_datos:
    N = meta["N"]
    y = _leer("y.bin")
    F = np.memmap(os.path.join(RUTA_EXPORT, "F.bin"), dtype="<f8",
                  mode="r", shape=(N, p), order="F")
    t_obs = _leer("t_obs.bin", "<i4")              # 1..Tt (apilado en orden t)
    k_obs = _leer("k_obs.bin", "<i4")
    j_obs = _leer("j_obs.bin", "<i4")
    ini_t = np.searchsorted(t_obs, np.arange(1, Tt + 2))    # cortes por periodo
    kap2_obs = 1.0 / (v_mean[j_obs - 1, k_obs - 1] *
                      phi_mean[k_obs - 1, t_obs - 1])
    sd_obs = np.sqrt(kap2_obs)

# ====================== UTILIDADES DE CONVERGENCIA ===========================
# (implementaciones espejo de 10_diagnostico_M4.R)

def _autocorr_fft(x):
    n = len(x); a = x - x.mean()
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(a, nfft)
    ac = np.fft.irfft(f * np.conj(f))[:n] / n
    if ac[0] <= 0:
        return None
    return ac / ac[0]

def ess_geyer(x):
    """ESS por secuencia monotona inicial positiva (Geyer 1992)."""
    n = len(x)
    if n < 8 or np.std(x) == 0:
        return np.nan
    rho = _autocorr_fft(x)
    if rho is None:
        return np.nan
    rho = rho[1:]
    suma, k = 0.0, 0
    while 2 * k + 1 < len(rho):
        par = rho[2 * k] + rho[2 * k + 1]
        if par <= 0:
            break
        suma += par; k += 1
    return min(n / max(1.0 + 2.0 * suma, 1e-8), n)

def split_rhat(x, n_seg=4):
    n_tot = len(x); seg = n_tot // n_seg
    if seg < 20:
        return np.nan
    M = x[: seg * n_seg].reshape(n_seg, seg).T     # col = segmento
    medias = M.mean(0); varis = M.var(0, ddof=1)
    W = varis.mean(); B = seg * medias.var(ddof=1)
    if W <= 0:
        return np.nan
    var_post = (seg - 1) / seg * W + B / seg
    return float(np.sqrt(var_post / W))

def geweke_z(x, frac_a=0.1, frac_b=0.5):
    n = len(x); na, nb = int(frac_a * n), int(frac_b * n)
    if na < 10 or nb < 10:
        return np.nan
    xa, xb = x[:na], x[n - nb:]
    def var_sp(z):
        nz = len(z); b = max(1, int(np.sqrt(nz))); nl = nz // b
        if nl < 2:
            return z.var(ddof=1) / nz
        ml = z[: nl * b].reshape(nl, b).mean(1)
        return b * ml.var(ddof=1) / nz
    se = np.sqrt(var_sp(xa) + var_sp(xb))
    return np.nan if se <= 0 else float((xa.mean() - xb.mean()) / se)

def bh_ajuste(pvals):
    p_ = np.asarray(pvals, float); nn = np.sum(~np.isnan(p_))
    q = np.full_like(p_, np.nan)
    idx = np.argsort(np.where(np.isnan(p_), np.inf, p_))
    orden = p_[idx][:nn] * nn / np.arange(1, nn + 1)
    orden = np.minimum.accumulate(orden[::-1])[::-1]
    q[idx[:nn]] = np.clip(orden, 0, 1)
    return q

def resumen_cadena(x, nombre):
    ess = ess_geyer(x)
    return dict(parametro=nombre, media=x.mean(), sd=x.std(ddof=1),
                q2_5=np.quantile(x, .025), mediana=np.quantile(x, .5),
                q97_5=np.quantile(x, .975), ess=ess, rhat=split_rhat(x),
                geweke_z=geweke_z(x),
                mcse=x.std(ddof=1) / np.sqrt(ess) if ess == ess else np.nan)

# ================== CONVERGENCIA: monitoreados + malla completa ==============
print("Convergencia: ESS / split-R-hat / Geweke ...")

filas, series_mon = [], {}
series_mon["nu"] = nu
for B in ("E", "M", "D"):
    series_mon[f"phi2_{B}"] = phi2[:, COLB[B]]
    series_mon[f"lambda2_{B}"] = lambda2[:, COLB[B]]
for tt in range(Tt + 1):
    series_mon[f"mu_t{tt}"] = theta[idx_mu, tt, :]
for nom, x in series_mon.items():
    filas.append(resumen_cadena(x, nom))
tab_mon = pd.DataFrame(filas)

# malla completa theta_{l,t}
ess_g  = np.full((p, Tt + 1), np.nan)
rhat_g = np.full((p, Tt + 1), np.nan)
gew_g  = np.full((p, Tt + 1), np.nan)
for l in range(p):
    for tt in range(Tt + 1):
        x = theta[l, tt, :]
        ess_g[l, tt] = ess_geyer(x)
        rhat_g[l, tt] = split_rhat(x)
        gew_g[l, tt] = geweke_z(x)
pd.DataFrame({
    "coeficiente": np.repeat(labels, Tt + 1),
    "t": np.tile(np.arange(Tt + 1), p),
    "ess": ess_g.ravel(), "rhat": rhat_g.ravel(), "geweke_z": gew_g.ravel(),
}).to_csv(os.path.join(DIR_SALIDA, "tabla_convergencia_theta.csv"), index=False)

# conjunto de PARIDAD con el modulo R (theta en periodos representativos)
per_R = sorted({0, int(np.ceil(Tt / 4)), int(np.ceil(Tt / 2)),
                int(np.ceil(3 * Tt / 4)), Tt})
ess_par  = np.concatenate([[ess_geyer(nu)],
                           [ess_geyer(phi2[:, j]) for j in range(3)],
                           [ess_geyer(lambda2[:, j]) for j in range(3)],
                           ess_g[:, per_R].ravel()])
rhat_par = np.concatenate([[split_rhat(nu)],
                           [split_rhat(phi2[:, j]) for j in range(3)],
                           [split_rhat(lambda2[:, j]) for j in range(3)],
                           rhat_g[:, per_R].ravel()])
gew_par  = np.concatenate([[geweke_z(nu)],
                           [geweke_z(phi2[:, j]) for j in range(3)],
                           [geweke_z(lambda2[:, j]) for j in range(3)],
                           gew_g[:, per_R].ravel()])
p_gew = 2 * stats.norm.sf(np.abs(gew_par))
n_gew_fdr   = int(np.nansum(bh_ajuste(p_gew) < 0.05))
ess_min     = float(np.nanmin(ess_par));  rhat_max = float(np.nanmax(rhat_par))
n_ess_bajo  = int(np.nansum(ess_par < U_ESS_AVISO))
n_rhat_alto = int(np.nansum(rhat_par > U_RHAT_AVISO))
n_par       = int(np.sum(~np.isnan(gew_par)))

tab_mon.to_csv(os.path.join(DIR_SALIDA, "tabla_convergencia_monitoreados.csv"),
               index=False)

# ============================ ESTILO GRAFICO =================================
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "figure.titlesize": 13, "figure.titleweight": "bold",
})
C  = {"E": "#1f77b4", "M": "#e07b39", "D": "#2f9e69",
      "nu": "#7a4dab", "neu": "#5b5b5b", "acc": "#8c2d43", "gris": "#bdbdbd"}
FIGS = []
def guardar(fig, nombre):
    ruta = os.path.join(DIR_SALIDA, nombre)
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    FIGS.append((nombre, fig))
    print("   ->", nombre)

def _traza(ax, x, color, titulo):
    it = np.arange(1, len(x) + 1)
    ax.plot(it, x, lw=0.4, color=color, alpha=0.75)
    ax.plot(it, np.cumsum(x) / it, lw=1.4, color="k")
    e, r = ess_geyer(x), split_rhat(x)
    ax.set_title(f"{titulo}   (ESS={e:.0f}, R\u0302={r:.3f})")
    ax.set_xlabel("iteración guardada")

# ---- fig01: trazas globales -------------------------------------------------
fig, axs = plt.subplots(4, 2, figsize=(10.5, 11))
_traza(axs[0, 0], nu, C["nu"], "ν  (grados de libertad)")
axs[0, 1].axis("off")
axs[0, 1].text(0.02, 0.80, "Trazas posteriores + media acumulada (línea negra)\n"
               "Cadena post burn-in del muestreador híbrido M4.",
               fontsize=9.5, va="top")
axs[0, 1].text(0.02, 0.42,
               f"S = {S} guardadas (burn-in={meta['burn_in']}, thin={meta['thin']})\n"
               f"Aceptación MH de ν = {meta['tasa_MH']:.3f}", fontsize=9.5, va="top")
for i, B in enumerate(("E", "M", "D")):
    _traza(axs[i + 1, 0], phi2[:, COLB[B]], C[B], f"φ²_{B}  (escala global LASSO)")
    _traza(axs[i + 1, 1], lambda2[:, COLB[B]], C[B], f"λ²_{B}  (regularización)")
fig.suptitle("M4 — Convergencia I: trazas de los parámetros globales")
fig.tight_layout(rect=(0, 0, 1, 0.97))
guardar(fig, "fig01_trazas_globales.png")

# ---- fig02: trazas de mu_t --------------------------------------------------
sel_t = sorted(set(np.linspace(0, Tt, 6).round().astype(int)))
fig, axs = plt.subplots(3, 2, figsize=(10.5, 8.5))
for ax, tt in zip(axs.ravel(), sel_t):
    _traza(ax, theta[idx_mu, tt, :], C["neu"], f"μ_t,  t={tt}")
fig.suptitle("M4 — Convergencia II: trazas del intercepto dinámico μ_t")
fig.tight_layout(rect=(0, 0, 1, 0.96))
guardar(fig, "fig02_trazas_mu.png")

# ---- fig03: autocorrelaciones ----------------------------------------------
def _acf_ax(ax, x, color, titulo, lmax=60):
    rho = _autocorr_fft(x)[: lmax + 1]
    ax.bar(np.arange(lmax + 1), rho, width=0.9, color=color)
    b = 1.96 / np.sqrt(len(x))
    ax.axhline(b, ls="--", lw=0.8, c="k"); ax.axhline(-b, ls="--", lw=0.8, c="k")
    ax.set_ylim(-0.35, 1.02); ax.set_title(titulo); ax.set_xlabel("rezago")

fig, axs = plt.subplots(4, 2, figsize=(10.5, 11))
_acf_ax(axs[0, 0], nu, C["nu"], "ACF ν")
_acf_ax(axs[0, 1], theta[idx_mu, Tt, :], C["neu"], f"ACF μ_t (t={Tt})")
for i, B in enumerate(("E", "M", "D")):
    _acf_ax(axs[i + 1, 0], phi2[:, COLB[B]], C[B], f"ACF φ²_{B}")
    _acf_ax(axs[i + 1, 1], lambda2[:, COLB[B]], C[B], f"ACF λ²_{B}")
fig.suptitle("M4 — Convergencia III: autocorrelación de las cadenas")
fig.tight_layout(rect=(0, 0, 1, 0.97))
guardar(fig, "fig03_acf.png")

# ---- fig04: heatmaps ESS y R-hat sobre toda la malla theta ------------------
fig, axs = plt.subplots(1, 2, figsize=(12.5, min(0.32 * p + 2.6, 22)))
ext = (-0.5, Tt + 0.5, p - 0.5, -0.5)
im0 = axs[0].imshow(ess_g, aspect="auto", cmap="viridis", extent=ext,
                    vmin=0, vmax=S)
axs[0].set_title(f"ESS por coeficiente y periodo (mín={np.nanmin(ess_g):.0f})")
im1 = axs[1].imshow(rhat_g, aspect="auto", cmap="RdYlGn_r", extent=ext,
                    vmin=1.0, vmax=1.2)
axs[1].set_title(f"split-R̂ (máx={np.nanmax(rhat_g):.3f})")
for ax, im in ((axs[0], im0), (axs[1], im1)):
    ax.set_xlabel("periodo t"); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    for B in ("E", "M", "D"):
        ax.axhline(bloques[B][0] - 0.5, color="w", lw=1.2)
axs[0].set_yticks(range(p)); axs[0].set_yticklabels(labels, fontsize=6.5)
axs[1].set_yticks([])
fig.suptitle("M4 — Convergencia IV: malla completa de θ (todos los coeficientes × t)")
fig.tight_layout(rect=(0, 0, 1, 0.94))
guardar(fig, "fig04_ess_rhat_heatmap.png")

# ---- fig05: Geweke ----------------------------------------------------------
noms = tab_mon["parametro"].tolist()
zs   = tab_mon["geweke_z"].to_numpy()
fig, ax = plt.subplots(figsize=(10.5, 4.2))
xs = np.arange(len(noms))
cols = [C["acc"] if abs(z) > 1.96 else C["neu"] for z in zs]
ax.vlines(xs, 0, zs, color=cols, lw=2); ax.scatter(xs, zs, s=14, c=cols)
ax.axhline(1.96, ls="--", c="k", lw=0.8); ax.axhline(-1.96, ls="--", c="k", lw=0.8)
ax.axhline(0, c="k", lw=0.6)
ax.set_xticks(xs); ax.set_xticklabels(noms, rotation=90, fontsize=7)
ax.set_ylabel("z de Geweke")
ax.set_title(f"M4 — Convergencia V: Geweke (10% vs 50%) · "
             f"{n_gew_fdr}/{n_par} significativos tras FDR en el conjunto de paridad")
fig.tight_layout()
guardar(fig, "fig05_geweke.png")

# =================== POSTERIOR: nu, mu_t, LASSO, escalas =====================
print("Posterior y LASSO dinámico ...")
nu_med = float(np.median(nu)); nu_ic = np.quantile(nu, [.025, .975])
p_nu10, p_nu30 = float(np.mean(nu < 10)), float(np.mean(nu < 30))
rho_nu = _autocorr_fft(nu)

# ---- fig06: posterior de nu -------------------------------------------------
fig = plt.figure(figsize=(10.5, 5.2))
gs = fig.add_gridspec(2, 2, width_ratios=(2.1, 1), hspace=0.45)
ax = fig.add_subplot(gs[:, 0])
kde = stats.gaussian_kde(nu); xx = np.linspace(nu.min(), nu.max(), 400)
ax.hist(nu, bins=45, density=True, color=C["nu"], alpha=0.35)
ax.plot(xx, kde(xx), color=C["nu"], lw=2)
ax.axvspan(nu_ic[0], nu_ic[1], color=C["nu"], alpha=0.12)
ax.axvline(nu_med, color="k", lw=1.2)
ax.set_xlabel("ν"); ax.set_ylabel("densidad")
ax.set_title("Posterior de ν (grados de libertad de la t)")
ax.text(0.97, 0.95, (f"mediana = {nu_med:.2f}\nIC95 = [{nu_ic[0]:.2f}, {nu_ic[1]:.2f}]\n"
        f"P(ν<10) = {p_nu10:.3f}\nP(ν<30) = {p_nu30:.3f}\n"
        f"acept. MH = {meta['tasa_MH']:.3f}\nACF(1)={rho_nu[1]:.2f}  ACF(10)={rho_nu[10]:.2f}"),
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(fc="white", ec="0.7"))
ax2 = fig.add_subplot(gs[0, 1]); ax2.plot(nu, lw=0.4, color=C["nu"])
ax2.set_title("traza de ν", fontsize=9)
ax3 = fig.add_subplot(gs[1, 1]); _acf_ax(ax3, nu, C["nu"], "ACF ν", 40)
fig.suptitle("M4 — Evidencia de colas pesadas (justificación de la verosimilitud t)")
guardar(fig, "fig06_nu_posterior.png")

# ---- fig07: trayectoria de mu_t ---------------------------------------------
mu_dr = theta[idx_mu, :, :]                            # (Tt+1) x S
qs = np.quantile(mu_dr, [.025, .25, .5, .75, .975], axis=1)
fig, ax = plt.subplots(figsize=(9.5, 4.6))
tx = np.arange(Tt + 1)
ax.fill_between(tx, qs[0], qs[4], color=C["neu"], alpha=0.18, label="IC 95%")
ax.fill_between(tx, qs[1], qs[3], color=C["neu"], alpha=0.30, label="IC 50%")
ax.plot(tx, qs[2], "-o", color="k", ms=4, lw=1.6, label="mediana")
ax.set_xlabel("periodo t (t=0 es el estado inicial)"); ax.set_ylabel("μ_t")
ax.legend(frameon=False)
ax.set_title("M4 — Trayectoria del intercepto dinámico μ_t (nivel nacional)")
fig.tight_layout()
guardar(fig, "fig07_mu_trayectoria.png")

# ---- fig08: sendas de los coeficientes (LASSO dinámico) ---------------------
th_mean = theta.mean(axis=2)                           # p x (Tt+1)
th_lo   = np.quantile(theta, .05, axis=2)
th_hi   = np.quantile(theta, .95, axis=2)
act_medio = np.abs(th_mean[:, 1:]).mean(axis=1)        # media_t |beta|
fig, axs = plt.subplots(1, 3, figsize=(13, 4.6), sharex=True)
for ax, B in zip(axs, ("E", "M", "D")):
    for l in bloques[B]:
        activo = act_medio[l] > 0.1
        cc = C[B] if activo else C["gris"]
        ax.plot(tx[1:], th_mean[l, 1:], lw=1.8 if activo else 0.9, color=cc,
                alpha=1.0 if activo else 0.6)
        if activo:
            ax.fill_between(tx[1:], th_lo[l, 1:], th_hi[l, 1:], color=cc, alpha=0.12)
            ax.annotate(labels[l], (Tt, th_mean[l, Tt]), xytext=(4, 0),
                        textcoords="offset points", fontsize=7, color=cc,
                        va="center")
    ax.axhline(0, c="k", lw=0.7)
    n_act = int(np.sum(act_medio[bloques[B]] > 0.1))
    ax.set_title(f"Bloque {B}: {n_act}/{len(bloques[B])} activos (|β̄|>0.1)")
    ax.set_xlabel("periodo t")
axs[0].set_ylabel("β (media posterior, banda 90% en activos)")
fig.suptitle("M4 — Sendas dinámicas de los coeficientes por bloque")
fig.tight_layout(rect=(0, 0, 1, 0.94))
guardar(fig, "fig08_coeficientes_paths.png")

# ---- fig09: shrinkage de innovaciones (métrica del módulo R) ----------------
P_innov = np.full((p, Tt), np.nan)
for l in range(p):
    escala = theta[l].std(ddof=1)                      # sd sobre (t, draws)
    umbral = 0.1 * escala
    dif = np.abs(theta[l, 1:, :] - theta[l, :-1, :])   # Tt x S
    P_innov[l] = (dif > umbral).mean(axis=1)
fig = plt.figure(figsize=(9.5, min(0.42 * p + 3.2, 20)))
gs9 = fig.add_gridspec(3, 1, hspace=0.38,
                       height_ratios=[len(bloques[B]) for B in ("E", "M", "D")])
axs = [fig.add_subplot(gs9[i, 0]) for i in range(3)]
for ax, B in zip(axs, ("E", "M", "D")):
    idx = bloques[B]
    im = ax.imshow(P_innov[idx], aspect="auto", cmap="magma", vmin=0, vmax=1,
                   extent=(0.5, Tt + 0.5, len(idx) - 0.5, -0.5))
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([labels[l] for l in idx], fontsize=7)
    n_act = int((P_innov[idx] > 0.5).sum()); ax.grid(False)
    ax.set_title(f"Bloque {B}: {n_act}/{idx.size * Tt} innovaciones activas (P>0.5) · "
                 f"φ²={np.median(phi2[:, COLB[B]]):.3f}  λ²={np.median(lambda2[:, COLB[B]]):.3f}",
                 fontsize=9)
axs[-1].set_xlabel("periodo t")
fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02,
             label="P(|Δθ| > 0.1·sd(θ))")
fig.suptitle("M4 — Shrinkage del LASSO dinámico sobre las innovaciones ω")
guardar(fig, "fig09_shrinkage.png")

# ---- fig15 (escalas; se genera aquí, se ordena al final del PDF) -------------
s_kt = 1.0 / phi_mean
fig = plt.figure(figsize=(12.5, 4.4))
gs = fig.add_gridspec(1, 3, width_ratios=(1.4, 1, 1))
ax = fig.add_subplot(gs[0, 0]); ax.grid(False)
im = ax.imshow(s_kt, aspect="auto", cmap="cividis",
               extent=(0.5, Tt + 0.5, m - 0.5, -0.5))
ax.set_yticks(range(m)); ax.set_yticklabels(deptos, fontsize=8)
ax.set_xlabel("periodo t"); ax.set_ylabel("departamento (código)")
ax.set_title("s_kt = 1/φ̄  (escala departamental)")
fig.colorbar(im, ax=ax, fraction=0.045)
ax = fig.add_subplot(gs[0, 1])
vv = v_mean[np.isfinite(v_mean) & (v_mean > 0)]
ax.hist(vv, bins=25, color=C["M"], alpha=0.8)
ax.axvline(1, c="k", ls="--", lw=1)
ax.set_xlabel("v̄_jk"); ax.set_title("Precisiones municipales (media 1 por identif.)")
ax = fig.add_subplot(gs[0, 2])
if tiene_datos:
    ax.hist(kap2_obs, bins=40, color=C["neu"], alpha=0.8)
    ax.set_xlabel("κ² = 1/(v̄·φ̄) por observación")
else:
    ax.axis("off"); ax.text(.5, .5, "sin datos exportados", ha="center")
ax.set_title("Escala base de la t por observación")
fig.suptitle("M4 — Heterogeneidad de las escalas observacionales")
fig.tight_layout(rect=(0, 0, 1, 0.93))
guardar(fig, "fig15_escalas.png")

# ======================= AJUSTE Y VALIDEZ PREDICTIVA =========================
res = {}          # métricas para el resumen
if tiene_datos:
    print("Ajuste (residuos, R², QQ) ...")

    def ajustados(th_s):
        """valores ajustados F_i' theta_{t_i} para un draw (o media) th_s: p x (Tt+1)."""
        out = np.empty(N)
        for tt in range(1, Tt + 1):
            a, b = ini_t[tt - 1], ini_t[tt]
            out[a:b] = F[a:b] @ th_s[:, tt]
        return out

    fit_mean = ajustados(th_mean)
    resid    = y - fit_mean
    r_std    = resid / sd_obs
    rmse, mae = float(np.sqrt(np.mean(resid ** 2))), float(np.mean(np.abs(resid)))
    r2_cl = float(1 - resid.var() / y.var())

    # R2 bayesiano (Gelman et al. 2019, versión con residuos por draw)
    idx_r2 = np.unique(np.linspace(0, S - 1, min(S_R2, S)).round().astype(int))
    r2_dr = np.empty(len(idx_r2))
    for i, s in enumerate(idx_r2):
        f_s = ajustados(theta[:, :, s])
        r2_dr[i] = f_s.var() / (f_s.var() + (y - f_s).var())
    r2_b = np.quantile(r2_dr, [.025, .5, .975])

    # ---- fig10: observado vs ajustado + residuos ----
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8))
    hb = axs[0].hexbin(fit_mean, y, gridsize=60, cmap="Blues", mincnt=1)
    lims = [min(fit_mean.min(), y.min()), max(fit_mean.max(), y.max())]
    axs[0].plot(lims, lims, "k--", lw=1)
    axs[0].set_xlabel("ŷ (media posterior)"); axs[0].set_ylabel("y observado")
    axs[0].set_title(f"Observado vs ajustado · R²={r2_cl:.3f} · "
                     f"R²_bayes={r2_b[1]:.3f} [{r2_b[0]:.3f}, {r2_b[2]:.3f}]")
    fig.colorbar(hb, ax=axs[0], fraction=0.04)
    hb2 = axs[1].hexbin(fit_mean, r_std, gridsize=60, cmap="Greys", mincnt=1)
    med_bin = stats.binned_statistic(fit_mean, r_std, "median", bins=25)
    ctr = 0.5 * (med_bin.bin_edges[1:] + med_bin.bin_edges[:-1])
    axs[1].plot(ctr, med_bin.statistic, color=C["acc"], lw=1.8)
    axs[1].axhline(0, c="k", lw=0.8)
    axs[1].set_xlabel("ŷ"); axs[1].set_ylabel("residuo estandarizado")
    axs[1].set_title(f"Residuos vs ajustados · RMSE={rmse:.2f}  MAE={mae:.2f}")
    fig.suptitle("M4 — Ajuste I: capacidad explicativa y estructura de residuos")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    guardar(fig, "fig10_ajuste_obs_vs_pred.png")

    # ---- fig11: QQ del residuo estandarizado, t vs Normal ----
    n_q = min(4000, N)
    pr = (np.arange(1, n_q + 1) - 0.5) / n_q
    emp = np.quantile(r_std, pr)
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, dist, nom in ((axs[0], stats.t(df=nu_med), f"t con ν̂={nu_med:.1f}"),
                          (axs[1], stats.norm(0, r_std.std(ddof=1)), "Normal")):
        te = dist.ppf(pr)
        ax.plot(te, emp, ".", ms=2.5,
                color=C["nu"] if ax is axs[0] else C["acc"])
        li = [min(te.min(), emp.min()), max(te.max(), emp.max())]
        ax.plot(li, li, "k--", lw=1)
        ax.set_xlabel(f"cuantiles teóricos {nom}")
        ax.set_title(f"QQ vs {nom}")
    axs[0].set_ylabel("cuantiles empíricos del residuo estandarizado")
    fig.suptitle("M4 — Ajuste II: la t captura las colas que la Normal no "
                 "(evidencia de la especificación robusta)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    guardar(fig, "fig11_residuos_qq.png")

    # ---- fig12: estructura de residuos por periodo y departamento ----
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axs[0].boxplot([r_std[t_obs == tt] for tt in range(1, Tt + 1)],
                   showfliers=False)
    axs[0].set_xticklabels([str(tt) for tt in range(1, Tt + 1)])
    axs[0].axhline(0, c="k", lw=0.8)
    axs[0].set_xlabel("periodo t"); axs[0].set_ylabel("residuo estandarizado")
    axs[0].set_title("Por periodo")
    axs[1].boxplot([r_std[k_obs == kk] for kk in range(1, m + 1)],
                   showfliers=False)
    axs[1].set_xticklabels(deptos)
    axs[1].axhline(0, c="k", lw=0.8)
    axs[1].set_xlabel("departamento (código)"); axs[1].set_title("Por departamento")
    fig.suptitle("M4 — Ajuste III: residuos estandarizados sin estructura remanente")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    guardar(fig, "fig12_residuos_estructura.png")

    # -------------------- PPC con réplicas t (espejo del módulo R) -----------
    print(f"PPC: {N_REP_PPC} réplicas t ...")
    def skew(z): return float(np.mean((z - z.mean()) ** 3) / z.std(ddof=1) ** 3)
    def kurt(z): return float(np.mean((z - z.mean()) ** 4) / z.std(ddof=1) ** 4 - 3)
    est_nombres = ["media", "sd", "q05", "q95", "asimetria", "curtosis"]
    def estad(z):
        return np.array([z.mean(), z.std(ddof=1), np.quantile(z, .05),
                         np.quantile(z, .95), skew(z), kurt(z)])
    est_obs = estad(y)
    idx_rep = np.unique(np.linspace(0, S - 1, min(N_REP_PPC, S)).round().astype(int))
    n_rep = len(idx_rep)
    est_rep = np.empty((n_rep, 6))
    pit_acu = np.zeros(N)
    sub = rng.choice(N, size=min(N_KDE_MAX, N), replace=False)
    dens_rep = []
    xx_y = np.linspace(np.quantile(y, .001), np.quantile(y, .999), 300)
    for r, s in enumerate(idx_rep):
        if r % max(1, n_rep // 8) == 0:
            print(f"     replica {r + 1}/{n_rep}", flush=True)
        f_s = ajustados(theta[:, :, s])
        yrep = f_s + sd_obs * rng.standard_t(df=nu[s], size=N)
        est_rep[r] = estad(yrep)
        pit_acu += (yrep <= y)
        if r < 25:
            dens_rep.append(stats.gaussian_kde(yrep[sub])(xx_y))
    pval = (est_rep >= est_obs).mean(axis=0)
    n_extremo = int(np.sum((pval < 0.05) | (pval > 0.95)))
    pit = pit_acu / n_rep
    jit = rng.uniform(-0.5 / n_rep, 0.5 / n_rep, N)
    ks_pit = float(stats.kstest(np.clip(pit + jit, 0, 1), "uniform").pvalue)
    cob = {niv: float(np.mean((pit > (1 - niv) / 2) & (pit < (1 + niv) / 2)))
           for niv in (0.50, 0.80, 0.95)}
    cob_t = {niv: [float(np.mean((pit[t_obs == tt] > (1 - niv) / 2) &
                                 (pit[t_obs == tt] < (1 + niv) / 2)))
                   for tt in range(1, Tt + 1)] for niv in (0.50, 0.80, 0.95)}

    # ---- fig13: densidades y estadísticos PPC ----
    fig = plt.figure(figsize=(11.5, 7.2))
    gs = fig.add_gridspec(2, 4, height_ratios=(1.35, 1), hspace=0.42)
    ax = fig.add_subplot(gs[0, :])
    for d in dens_rep:
        ax.plot(xx_y, d, color=C["gris"], lw=0.7, alpha=0.7)
    ax.plot(xx_y, stats.gaussian_kde(y[sub])(xx_y), color="k", lw=2,
            label="y observado")
    ax.plot([], [], color=C["gris"], label="réplicas $y^{rep}$ (25)")
    ax.legend(frameon=False); ax.set_xlabel("puntaje")
    ax.set_title("Densidad observada vs réplicas posteriores t")
    for i, (jj, nom) in enumerate([(0, "media"), (1, "sd"),
                                   (4, "asimetría"), (5, "curtosis")]):
        ax = fig.add_subplot(gs[1, i])
        ax.hist(est_rep[:, jj], bins=25, color=C["E"], alpha=0.7)
        ax.axvline(est_obs[jj], color=C["acc"], lw=2)
        ax.set_title(f"{nom}\np-val bayes = {pval[jj]:.3f}", fontsize=9)
        ax.set_yticks([])
    fig.suptitle("M4 — Validez predictiva I: posterior predictive checks")
    guardar(fig, "fig13_ppc_densidad.png")

    # ---- fig14: PIT y cobertura ----
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))
    nb = 20
    axs[0].hist(pit, bins=nb, density=True, color=C["E"], alpha=0.75)
    band = 1.96 * np.sqrt((1 / nb) * (1 - 1 / nb) / N) * nb
    axs[0].axhline(1, c="k", lw=1)
    axs[0].axhspan(1 - band, 1 + band, color="k", alpha=0.08)
    axs[0].set_xlabel("PIT"); axs[0].set_ylabel("densidad")
    axs[0].set_title(f"PIT (calibración) · KS p = {ks_pit:.3f}")
    nivs = [0.50, 0.80, 0.95]
    xsb = np.arange(3)
    axs[1].bar(xsb - 0.18, nivs, width=0.36, color=C["gris"], label="nominal")
    axs[1].bar(xsb + 0.18, [cob[n] for n in nivs], width=0.36, color=C["D"],
               label="empírica")
    for i, n in enumerate(nivs):
        axs[1].text(i + 0.18, cob[n] + 0.015, f"{cob[n]:.3f}", ha="center",
                    fontsize=8.5)
    axs[1].set_xticks(xsb); axs[1].set_xticklabels(["50%", "80%", "95%"])
    axs[1].set_ylim(0, 1.05); axs[1].legend(frameon=False)
    axs[1].set_title("Cobertura de los intervalos predictivos")
    fig.suptitle("M4 — Validez predictiva II: calibración")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    guardar(fig, "fig14_pit_cobertura.png")

    # -------------------------- WAIC / DIC (streaming) -----------------------
    print(f"WAIC/DIC con {min(S_MAX_LIK, S)} draws ...")
    idx_lik = np.unique(np.linspace(0, S - 1, min(S_MAX_LIK, S)).round().astype(int))
    Slik = len(idx_lik)
    M_run  = np.full(N, -np.inf); A_run = np.zeros(N)      # logsumexp streaming
    mean_w = np.zeros(N); M2_w = np.zeros(N)               # Welford
    dev_sum = 0.0
    for cnt, s in enumerate(idx_lik, start=1):
        if (cnt - 1) % max(1, Slik // 8) == 0:
            print(f"     draw {cnt}/{Slik}", flush=True)
        f_s = ajustados(theta[:, :, s])
        ll = stats.t.logpdf(y, df=nu[s], loc=f_s, scale=sd_obs)
        dev_sum += -2.0 * ll.sum()
        Mn = np.maximum(M_run, ll)
        A_run = A_run * np.exp(M_run - Mn) + np.exp(ll - Mn)
        M_run = Mn
        d = ll - mean_w; mean_w += d / cnt; M2_w += d * (ll - mean_w)
    lme    = M_run + np.log(A_run) - np.log(Slik)          # log E[p(y_i|.)]
    var_ll = M2_w / (Slik - 1)
    elpd_i = lme - var_ll
    lppd, p_waic = float(lme.sum()), float(var_ll.sum())
    waic = float(-2 * elpd_i.sum())
    se_waic = float(2 * np.sqrt(N * elpd_i.var(ddof=1)))
    ll_hat = stats.t.logpdf(y, df=float(nu.mean()), loc=fit_mean, scale=sd_obs)
    D_hat, D_bar = float(-2 * ll_hat.sum()), dev_sum / Slik
    p_dic = D_bar - D_hat; dic = D_hat + 2 * p_dic

    res.update(rmse=rmse, mae=mae, r2=r2_cl, r2_bayes=r2_b, pval=pval,
               n_extremo=n_extremo, ks_pit=ks_pit, cob=cob,
               waic=waic, se_waic=se_waic, lppd=lppd, p_waic=p_waic,
               dic=dic, p_dic=p_dic, elpd=float(elpd_i.sum()))

    pd.DataFrame({"estadistico": est_nombres, "observado": est_obs,
                  "media_rep": est_rep.mean(0), "sd_rep": est_rep.std(0, ddof=1),
                  "p_value_bayes": pval}).to_csv(
        os.path.join(DIR_SALIDA, "tabla_ppc.csv"), index=False)
    pd.DataFrame({"periodo": np.tile(np.arange(1, Tt + 1), 3),
                  "nivel": np.repeat([0.50, 0.80, 0.95], Tt),
                  "cobertura": np.concatenate([cob_t[n] for n in (0.50, 0.80, 0.95)])
                  }).to_csv(os.path.join(DIR_SALIDA, "tabla_cobertura.csv"), index=False)
    pd.DataFrame([dict(WAIC=waic, se_WAIC=se_waic, elpd=res["elpd"], lppd=lppd,
                       p_WAIC=p_waic, DIC=dic, p_DIC=p_dic, RMSE=rmse, MAE=mae,
                       R2=r2_cl, R2_bayes_mediana=r2_b[1])]).to_csv(
        os.path.join(DIR_SALIDA, "tabla_ajuste_waic.csv"), index=False)

# ====================== RESUMEN, VEREDICTO Y PDF =============================
conv_ok = (rhat_max < U_RHAT_VEREDICTO) and (ess_min > U_ESS_VEREDICTO)
nu_ess, nu_rhat = ess_geyer(nu), split_rhat(nu)
tasa_ok = U_TASA_MH[0] < meta["tasa_MH"] < U_TASA_MH[1]
nu_ok = tasa_ok and (nu_ess > U_ESS_VEREDICTO)
if tiene_datos:
    ppc_ok = res["n_extremo"] <= U_PPC_EXTREMOS
    veredicto = conv_ok and nu_ok and ppc_ok
else:
    ppc_ok = None
    veredicto = conv_ok and nu_ok

L = []
L.append("==== DIAGNOSTICO DEL MODELO 4 (t-Student dinamico robusto) ====")
L.append(f"Fecha: {time.strftime('%Y-%m-%d %H:%M')}   |   S={S} guardadas "
         f"(burn-in={meta['burn_in']}, thin={meta['thin']})")
L.append("")
L.append("-- Convergencia (conjunto de paridad con 10_diagnostico_M4.R) --")
L.append(f"  ESS   : min={ess_min:.0f} | nu={nu_ess:.0f} | {n_ess_bajo} de {n_par} con ESS<{U_ESS_AVISO}")
L.append(f"  R-hat : max={rhat_max:.3f} | nu={nu_rhat:.3f} | {n_rhat_alto} con R-hat>{U_RHAT_AVISO}")
L.append(f"  Geweke: {n_gew_fdr}/{n_par} significativos tras FDR (BH)")
L.append(f"  Malla completa theta: ESS min={np.nanmin(ess_g):.0f}, "
         f"R-hat max={np.nanmax(rhat_g):.3f}  ({p}x{Tt + 1} cadenas)")
L.append("")
L.append("-- Bloque nu (Metropolis-Hastings) --")
L.append(f"  posterior: mediana={nu_med:.3f}  IC95=[{nu_ic[0]:.3f}, {nu_ic[1]:.3f}]")
L.append(f"  P(nu<10)={p_nu10:.3f}  P(nu<30)={p_nu30:.3f}   "
         f"(colas pesadas -> justificacion de la t)")
L.append(f"  acept.MH={meta['tasa_MH']:.3f} {'(en rango)' if tasa_ok else '(FUERA de rango)'}"
         f"  ESS={nu_ess:.0f}  R-hat={nu_rhat:.3f}")
L.append("")
if tiene_datos:
    L.append("-- Ajuste --")
    L.append(f"  RMSE={res['rmse']:.3f}  MAE={res['mae']:.3f}  R2={res['r2']:.3f}  "
             f"R2_bayes={res['r2_bayes'][1]:.3f} [{res['r2_bayes'][0]:.3f}, {res['r2_bayes'][2]:.3f}]")
    L.append(f"  WAIC={res['waic']:.1f} (se={res['se_waic']:.1f})  p_WAIC={res['p_waic']:.1f}  "
             f"elpd={res['elpd']:.1f}")
    L.append(f"  DIC ={res['dic']:.1f}  p_DIC={res['p_dic']:.1f}   "
             f"(comparables contra el Modelo 1/otros con los mismos datos)")
    L.append("")
    L.append("-- Validez predictiva --")
    pv = res["pval"]
    L.append(f"  p-values bayes: media={pv[0]:.3f} sd={pv[1]:.3f} q05={pv[2]:.3f} "
             f"q95={pv[3]:.3f} asim={pv[4]:.3f} curt={pv[5]:.3f}")
    L.append(f"  {res['n_extremo']}/6 estadisticos con p-value extremo (<0.05 o >0.95)")
    L.append(f"  PIT: KS p={res['ks_pit']:.3f}   Cobertura 50/80/95: "
             f"{res['cob'][0.50]:.3f} / {res['cob'][0.80]:.3f} / {res['cob'][0.95]:.3f}")
    L.append("")
else:
    L.append("-- Ajuste y validez predictiva: OMITIDOS (export sin datos) --")
    L.append("")
L.append("-- VEREDICTO (umbrales del modulo R) --")
L.append(f"  convergencia : {'OK' if conv_ok else 'REVISAR'} "
         f"(R-hat max<{U_RHAT_VEREDICTO} y ESS min>{U_ESS_VEREDICTO})")
L.append(f"  bloque nu    : {'OK' if nu_ok else 'REVISAR'} "
         f"(tasa MH en {U_TASA_MH} y ESS>{U_ESS_VEREDICTO})")
if ppc_ok is not None:
    L.append(f"  predictivo   : {'OK' if ppc_ok else 'REVISAR'} "
             f"(<= {U_PPC_EXTREMOS} p-values extremos)")
L.append(f"  GLOBAL       : {'DIAGNOSTICOS SATISFACTORIOS' if veredicto else 'REVISAR ADVERTENCIAS'}")
texto = "\n".join(L)
with open(os.path.join(DIR_SALIDA, "resumen_diagnostico_M4.txt"), "w",
          encoding="utf-8") as fh:
    fh.write(texto + "\n")
print("\n" + texto)

with PdfPages(os.path.join(DIR_SALIDA, "diagnosticos_M4.pdf")) as pdf:
    for nom, fg in sorted(FIGS, key=lambda t: t[0]):
        pdf.savefig(fg)
print(f"\nLISTO en {time.time() - t_ini:.1f}s  ->  {DIR_SALIDA}/ "
      f"({len(FIGS)} figuras + tablas + resumen + PDF)")
