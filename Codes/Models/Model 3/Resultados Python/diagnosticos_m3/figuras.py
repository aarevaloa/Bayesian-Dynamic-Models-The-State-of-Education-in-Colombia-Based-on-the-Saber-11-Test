"""
figuras.py — Generación de figuras del diagnóstico M3
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib import font_manager

def _elegir_fuente():
    preferidas = ["Source Sans 3", "Source Sans Pro", "Inter", "Lato",
                  "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]
    disponibles = {f.name for f in font_manager.fontManager.ttflist}
    for f in preferidas:
        if f in disponibles:
            return f
    return "DejaVu Sans"

_FUENTE = _elegir_fuente()

# --- Neutros ---
TINTA       = "#1a1a1a"   # texto principal / línea protagonista (negro suave)
TINTA_SUAVE = "#555555"   # texto secundario, anotaciones
PAPEL       = "#ffffff"   # fondo blanco puro
REJILLA     = "#e6e6e6"   # rejilla muy tenue
EJE         = "#333333"   # spines definidos

# --- Trío cualitativo vivo (como figR3: bloques/series distintas) ---
VINO    = "#8b1a3f"   # vino/granate saturado
NARANJA = "#e07b1a"   # naranja cálido
INDIGO  = "#5b4bbd"   # azul-violeta
VERDE_V = "#2e8b6b"   # verde esmeralda (cuarto color si hace falta)

# Compatibilidad: los tres bloques E/M/D usan el trío vivo
AZUL   = VINO         # E  (protagonista cálido)
TERRA  = NARANJA      # M
VERDE  = INDIGO       # D
CIRUELA = INDIGO
AMBAR  = NARANJA
CARMIN = "#b0182f"    # línea de referencia / cero / observado (rojo vivo)

C_MEDIANA = TINTA
C_BANDA95 = VINO
C_BANDA90 = VINO
C_CERO    = CARMIN
COL_BLOQUE = {"E": VINO, "M": NARANJA, "D": INDIGO}

# --- Rampa secuencial cálida (como figR1: categorías ordenadas) ---
# durazno claro -> coral -> rojo -> vino profundo
RAMPA_CALIDA = ["#f2b48c", "#e8804f", "#d1442f", "#a01e30", "#6b1029"]

def rampa_calida(n):
    """Devuelve n colores interpolados de la rampa cálida (durazno->vino)."""
    cmap = LinearSegmentedColormap.from_list("calida_seq", RAMPA_CALIDA)
    if n == 1:
        return [RAMPA_CALIDA[2]]
    return [cmap(i / (n - 1)) for i in range(n)]

# --- Mapas de color secuenciales/divergentes a juego ---
# Divergente para shrinkage: índigo -> blanco -> naranja/vino
CMAP_SHRINK = LinearSegmentedColormap.from_list(
    "shrink_v", ["#3a2f8f", "#8b86c9", "#ffffff", "#e8a05a", "#a01e30"])
# ESS: rojo vivo (malo) -> blanco -> verde (bueno)
CMAP_ESS = LinearSegmentedColormap.from_list(
    "ess_v", ["#b0182f", "#e89aa0", "#ffffff", "#8fcbb5", "#2e8b6b"])
# Calor secuencial (actividad / CRPS): blanco -> durazno -> vino profundo
CMAP_CALOR = LinearSegmentedColormap.from_list(
    "calor_v", ["#fdf3ec", "#f2b48c", "#e8804f", "#c0362c", "#6b1029"])
CMAP_SEC = CMAP_CALOR

# ---------------------------------------------------------------------------
# Estilo global (rcParams)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.facecolor": PAPEL,
    "axes.facecolor": PAPEL,

    "font.family": _FUENTE,
    "font.size": 11,
    "text.color": TINTA,

    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlecolor": TINTA,
    "axes.titlepad": 10,
    "axes.labelsize": 10.5,
    "axes.labelcolor": TINTA_SUAVE,
    "axes.labelweight": "regular",

    "axes.edgecolor": EJE,
    "axes.linewidth": 1.1,
    "axes.spines.top": False,
    "axes.spines.right": False,

    "axes.grid": True,
    "axes.grid.axis": "both",
    "grid.color": REJILLA,
    "grid.alpha": 0.9,
    "grid.linewidth": 0.7,

    "xtick.color": TINTA_SUAVE,
    "ytick.color": TINTA_SUAVE,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.size": 0,
    "ytick.major.size": 0,

    "legend.frameon": False,
    "legend.fontsize": 9.5,

    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "figure.autolayout": False,
})


def _labels_tiempo(meta):
    """Etiquetas del eje temporal (0..Tt), usando tiempo_ids si están."""
    Tt = meta.Tt
    x = np.arange(Tt + 1)
    tid = meta.tiempo_ids
    if len(tid) == Tt:
        return x, ["inicial"] + [str(v) for v in tid]
    if len(tid) == Tt + 1:
        return x, [str(v) for v in tid]
    return x, [str(v) for v in x]


def _ralear_ticks_tiempo(ax, x, xlabs, cada=None, rot=0):
    """
    Muestra un subconjunto de etiquetas de tiempo para evitar amontonamiento.
    Con muchos periodos (p.ej. 17), rota etiquetas y muestra 1 de cada `cada`.
    """
    n = len(x)
    if cada is None:
        cada = 1 if n <= 8 else (2 if n <= 14 else 3)
    idx_vis = list(range(0, n, cada))
    if (n - 1) not in idx_vis:      # asegurar que el último periodo se vea
        idx_vis.append(n - 1)
    ax.set_xticks([x[i] for i in idx_vis])
    # rotar sólo si hay etiquetas largas (años); si son 0..Tt cortas, sin rotar
    largas = any(len(str(xlabs[i])) > 2 for i in idx_vis)
    rotacion = rot if rot else (30 if largas else 0)
    ax.set_xticklabels([xlabs[i] for i in idx_vis], rotation=rotacion,
                       ha="right" if rotacion else "center", fontsize=7.5)


def _estilizar_ejes(ax):
    """Retoques finos: rejilla por debajo, ticks discretos."""
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _guardar(fig, ruta):
    fig.savefig(ruta)
    plt.close(fig)


# ===========================================================================
# BLOQUE DINÁMICO
# ===========================================================================

def fig_evolucion_intercepto(muestras, evol_mu, dir_out):
    """Intercepto dinámico mu_t con bandas 90% y 95%."""
    meta = muestras.meta
    x, xlabs = _labels_tiempo(meta)
    q = evol_mu   # (Tt+1, 5): q025,q05,q50,q95,q975

    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    ax.fill_between(x, q[:, 0], q[:, 4], color=VINO, alpha=0.13,
                    lw=0, label="IC 95%")
    ax.fill_between(x, q[:, 1], q[:, 3], color=VINO, alpha=0.24,
                    lw=0, label="IC 90%")
    ax.plot(x, q[:, 2], color=VINO, lw=2.8, label="Mediana",
            solid_capstyle="round")
    _ralear_ticks_tiempo(ax, x, xlabs)
    ax.set_xlabel("Periodo")
    ax.set_ylabel(r"$\mu_t$")
    ax.set_title("Evolución del intercepto dinámico", loc="left")
    ax.legend(loc="best", handlelength=1.6)
    _estilizar_ejes(ax)
    _guardar(fig, os.path.join(dir_out, "din_evolucion_intercepto.pdf"))


def fig_evolucion_bloque(muestras, B, evol, top_idx, dir_out):
    """Panel de los coeficientes más dinámicos de un bloque con bandas."""
    meta = muestras.meta
    x, xlabs = _labels_tiempo(meta)
    Q = evol["quantiles"]
    labels = evol["labels"]
    col = COL_BLOQUE[B]
    n = len(top_idx)
    nc = int(np.ceil(np.sqrt(n)))
    nr = int(np.ceil(n / nc))

    fig, axes = plt.subplots(nr, nc, figsize=(3.9 * nc, 3.2 * nr),
                             squeeze=False, constrained_layout=True)
    for ax in axes.ravel():
        ax.set_visible(False)
    for i, l in enumerate(top_idx):
        ax = axes.ravel()[i]
        ax.set_visible(True)
        q = Q[l]
        ax.fill_between(x, q[:, 0], q[:, 4], color=col, alpha=0.12, lw=0)
        ax.fill_between(x, q[:, 1], q[:, 3], color=col, alpha=0.24, lw=0)
        ax.plot(x, q[:, 2], color=col, lw=2.2, solid_capstyle="round")
        ax.axhline(0, ls=(0, (3, 3)), color=TINTA_SUAVE, lw=0.9, alpha=0.7)
        ax.set_title(labels[l], fontsize=10, loc="left", color=TINTA)
        # ticks raleados y en horizontal (clave para no amontonar 17 años)
        _ralear_ticks_tiempo(ax, x, xlabs, cada=max(3, (len(x) + 3) // 5))
        ax.tick_params(axis="y", labelsize=7.5)
        _estilizar_ejes(ax)
    nombre = meta.bloques[B]["nombre"]
    fig.suptitle(f"Bloque {B} ({nombre}) · Top {n} coeficientes más dinámicos",
                 fontsize=13.5, fontweight="bold")
    _guardar(fig, os.path.join(dir_out, f"din_evolucion_bloque_{B}.pdf"))


def fig_lasso_actividad(muestras, B, innov, dir_out):
    """Mapa de calor P(|Δβ| > 0.1·SD) por (coeficiente × periodo)."""
    meta = muestras.meta
    prob = innov["prob_cambio"]        # (p_B, Tt)
    labels = innov["labels"]
    p_B, Tt = prob.shape

    fig, ax = plt.subplots(figsize=(max(6.5, 0.52 * Tt + 3.2),
                                    max(3.2, 0.30 * p_B + 1.6)),
                           constrained_layout=True)
    im = ax.imshow(prob, aspect="auto", cmap=CMAP_CALOR, vmin=0, vmax=1,
                   interpolation="nearest")
    _ticks_periodo_heatmap(ax, Tt)
    ax.set_yticks(np.arange(p_B))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Periodo")
    ax.grid(False)
    ax.set_title(f"Bloque {B} · actividad del LASSO dinámico   "
                 r"$P(|\Delta\beta| > 0.1\,SD)$", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("P(cambio)")
    cb.outline.set_visible(False)
    _guardar(fig, os.path.join(dir_out, f"din_lasso_actividad_{B}.pdf"))


def fig_lasso_qq(B, q_th, q_emp, dir_out):
    """QQ-plot de innovaciones contra Laplace."""
    if q_th is None:
        return
    col = COL_BLOQUE[B]
    fig, ax = plt.subplots(figsize=(5.6, 5.6), constrained_layout=True)
    ax.scatter(q_th, q_emp, s=10, alpha=0.45, color=col, edgecolors="none")
    lim = [min(q_th.min(), q_emp.min()), max(q_th.max(), q_emp.max())]
    ax.plot(lim, lim, color=CARMIN, lw=1.8, ls=(0, (4, 3)))
    ax.set_xlabel("Cuantil teórico (Laplace)")
    ax.set_ylabel(r"Cuantil empírico ($\omega$)")
    ax.set_title(f"Bloque {B} · innovaciones vs Laplace", loc="left")
    ax.set_aspect("equal", adjustable="datalim")
    _estilizar_ejes(ax)
    _guardar(fig, os.path.join(dir_out, f"din_lasso_qq_{B}.pdf"))


def fig_varianza_departamental(muestras, dvar, dir_out):
    """Triple heatmap de s_{k,t}: q05 / mediana / q95."""
    meta = muestras.meta
    s05, smed, s95 = dvar["s_q05"], dvar["s_median"], dvar["s_q95"]
    m, Tt = smed.shape
    dep = dvar["depto_ids"] if dvar["depto_ids"] else [f"k={i+1}" for i in range(m)]

    vmin = np.nanmin(s05)
    vmax = np.nanmax(s95)
    mid = float(np.median(smed))
    norm = TwoSlopeNorm(vmin=min(vmin, mid - 1e-6), vcenter=mid,
                        vmax=max(vmax, mid + 1e-6))

    fig, axes = plt.subplots(1, 3, figsize=(15, max(5, 0.30 * m + 1.6)),
                             sharey=True, constrained_layout=True)
    for ax, M, tit in zip(axes, (s05, smed, s95),
                          ("Cuantil 5%", "Mediana", "Cuantil 95%")):
        im = ax.imshow(M, aspect="auto", cmap=CMAP_SHRINK, norm=norm,
                       interpolation="nearest")
        ax.set_title(tit, fontsize=11.5, loc="left")
        _ticks_periodo_heatmap(ax, Tt)
        ax.set_xlabel("Periodo")
        ax.grid(False)
    axes[0].set_yticks(np.arange(m))
    axes[0].set_yticklabels(dep, fontsize=7.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label(r"$\tilde{s}_{k,t}$")
    cb.outline.set_visible(False)
    fig.suptitle(r"Varianza departamental dinámica  $s_{k,t}$",
                 fontweight="bold")
    _guardar(fig, os.path.join(dir_out, "din_varianza_departamental.pdf"))


def fig_shrinkage_temporal(muestras, dir_out, shrink_por_bloque):
    """Trayectoria del shrinkage local (mediana log tau2) por periodo y bloque."""
    Tt = muestras.meta.Tt
    x = np.arange(1, Tt + 1)
    fig, ax = plt.subplots(figsize=(9.2, 5), constrained_layout=True)
    for B in ("E", "M", "D"):
        d = shrink_por_bloque[B]
        c = COL_BLOQUE[B]
        ax.plot(x, d["median"], "-o", color=c, lw=2, ms=4.5,
                label=f"Bloque {B}", markeredgecolor=PAPEL,
                markeredgewidth=0.6)
        ax.fill_between(x, d["q25"], d["q75"], color=c, alpha=0.12, lw=0)
    ax.set_xlabel("Periodo")
    ax.set_ylabel(r"mediana $\log\tau^2$")
    ax.set_title("Evolución temporal del shrinkage local por bloque",
                 loc="left")
    ax.text(0.0, 1.015,
            "más negativo → innovaciones más comprimidas hacia 0",
            transform=ax.transAxes, fontsize=8.5, color=TINTA_SUAVE)
    ax.legend(handlelength=1.4)
    _estilizar_ejes(ax)
    _guardar(fig, os.path.join(dir_out, "din_shrinkage_temporal.pdf"))


def _ticks_periodo_heatmap(ax, Tt, cada=None):
    """Ticks de periodo raleados para heatmaps (eje X = 0..Tt-1, etiqueta 1..Tt)."""
    if cada is None:
        cada = 1 if Tt <= 12 else 2
    idx = list(range(0, Tt, cada))
    if (Tt - 1) not in idx:
        idx.append(Tt - 1)
    ax.set_xticks(idx)
    ax.set_xticklabels([str(i + 1) for i in idx], fontsize=8)


# ===========================================================================
# CONVERGENCIA
# ===========================================================================

def fig_traceplots(muestras, dir_out):
    """Traceplots + ACF + running mean del intercepto y lambda2."""
    meta = muestras.meta
    Tt = meta.Tt
    mu = muestras.theta[0, :, :]            # (Tt+1, S)
    idx_t = [0, int(np.ceil(Tt / 2)), Tt]

    # ---- intercepto: trace / acf / running ----
    fig, axes = plt.subplots(3, 3, figsize=(13, 8.4), constrained_layout=True)
    for r, t in enumerate(idx_t):
        x = mu[t, :]
        _traza_una(axes[r, 0], x, AZUL, rf"traza  $\mu_{{t={t}}}$")
        _acf_plot(axes[r, 1], x, 50)
        axes[r, 1].set_title(rf"ACF  $\mu_{{t={t}}}$", fontsize=10, loc="left")
        _media_acum(axes[r, 2], x, AZUL, rf"media acumulada  $\mu_{{t={t}}}$")
    fig.suptitle("Diagnóstico de trazas · intercepto dinámico",
                 fontweight="bold")
    _guardar(fig, os.path.join(dir_out, "conv_traceplots_intercepto.pdf"))

    # ---- lambda2 ----
    lam = muestras.lambda2                  # (S, 3)
    nombres = ["E", "M", "D"]
    cols = [AZUL, TERRA, VERDE]
    fig, axes = plt.subplots(3, 3, figsize=(13, 8.4), constrained_layout=True)
    for r in range(3):
        x = lam[:, r]
        c = cols[r]
        _traza_una(axes[r, 0], x, c, rf"traza  $\lambda^2_{nombres[r]}$")
        _acf_plot(axes[r, 1], x, 50)
        axes[r, 1].set_title(rf"ACF  $\lambda^2_{nombres[r]}$", fontsize=10,
                             loc="left")
        _media_acum(axes[r, 2], x, c,
                    rf"media acumulada  $\lambda^2_{nombres[r]}$")
    fig.suptitle(r"Diagnóstico de trazas · $\lambda^2$ por bloque",
                 fontweight="bold")
    _guardar(fig, os.path.join(dir_out, "conv_traceplots_lambda2.pdf"))


def _traza_una(ax, x, color, titulo):
    ax.plot(x, color=color, lw=0.5, alpha=0.85)
    ax.axhline(x.mean(), color=CARMIN, ls=(0, (4, 3)), lw=1.1)
    ax.set_title(titulo, fontsize=10, loc="left")
    ax.margins(x=0.01)
    _estilizar_ejes(ax)


def _media_acum(ax, x, color, titulo):
    rm = np.cumsum(x) / np.arange(1, x.size + 1)
    ax.plot(rm, color=color, lw=1.4)
    ax.axhline(x.mean(), color=CARMIN, ls=(0, (4, 3)), lw=1.1)
    ax.set_title(titulo, fontsize=10, loc="left")
    ax.margins(x=0.01)
    _estilizar_ejes(ax)


def _acf_plot(ax, x, lag_max):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    ac = np.correlate(x, x, mode="full")[n - 1:] / (np.arange(n, 0, -1) * x.var())
    lags = np.arange(min(lag_max + 1, n))
    ax.bar(lags, ac[:lags.size], width=0.85, color=AZUL, alpha=0.8)
    ax.axhline(0, color=TINTA_SUAVE, lw=0.7)
    ci = 1.96 / np.sqrt(n)
    ax.axhline(ci, color=CARMIN, ls=(0, (1, 2)), lw=0.9)
    ax.axhline(-ci, color=CARMIN, ls=(0, (1, 2)), lw=0.9)
    ax.set_ylim(-0.2, 1.05)
    ax.margins(x=0.01)
    _estilizar_ejes(ax)


def fig_ess_resumen(diag_conv, dir_out):
    """Histograma/resumen de ESS por grupo de parámetros."""
    grupos = {
        r"$\mu_t$": diag_conv.ess_mu,
        r"$\lambda^2$": diag_conv.ess_lambda,
        r"$s_{k,t}$": diag_conv.ess_s,
    }
    # agregar theta de todos los periodos
    theta_all = np.concatenate([np.asarray(v).ravel()
                                for v in diag_conv.ess_theta.values()])
    grupos[r"$\theta$ (multi-t)"] = theta_all

    fig, ax = plt.subplots(figsize=(8.8, 5), constrained_layout=True)
    data = [np.asarray(v)[np.isfinite(v)] for v in grupos.values()]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True,
                    medianprops=dict(color=TINTA, lw=1.6),
                    whiskerprops=dict(color=TINTA_SUAVE, lw=1),
                    capprops=dict(color=TINTA_SUAVE, lw=1),
                    boxprops=dict(lw=0))
    ax.set_xticks(range(1, len(grupos) + 1))
    ax.set_xticklabels(list(grupos.keys()))
    for patch, col in zip(bp["boxes"], [AZUL, TERRA, VERDE, CIRUELA]):
        patch.set_facecolor(col)
        patch.set_alpha(0.62)
    ax.axhline(400, color=CARMIN, ls=(0, (4, 3)), lw=1.2, label="umbral 400")
    ax.axhline(1000, color=VERDE, ls=(0, (1, 2)), lw=1.3, label="ideal 1000")
    ax.set_ylabel("ESS")
    ax.set_title("Tamaño de muestra efectivo por grupo de parámetros",
                 loc="left")
    ax.legend(handlelength=1.4)
    _estilizar_ejes(ax)
    _guardar(fig, os.path.join(dir_out, "conv_ess_resumen.pdf"))


# ===========================================================================
# tau2
# ===========================================================================

def fig_tau2_ess_heatmap(diag_tau2, dir_out):
    B = diag_tau2["B"]
    ess = diag_tau2["ess_mat"]
    labels = diag_tau2["labels"]
    p_B, Tt = ess.shape

    fig, ax = plt.subplots(figsize=(max(6.5, 0.52 * Tt + 3.2),
                                    max(3.2, 0.30 * p_B + 1.6)),
                           constrained_layout=True)
    norm = TwoSlopeNorm(vmin=0, vcenter=400, vmax=max(800, np.nanmax(ess)))
    im = ax.imshow(ess, aspect="auto", cmap=CMAP_ESS, norm=norm,
                   interpolation="nearest")
    _ticks_periodo_heatmap(ax, Tt)
    ax.set_yticks(np.arange(p_B)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Periodo"); ax.grid(False)
    ax.set_title(rf"Bloque {B} · ESS$(\log\tau^2)$   "
                 "rojo = mezcla mal (<400), verde = OK", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("ESS")
    cb.outline.set_visible(False)
    _guardar(fig, os.path.join(dir_out, f"tau2_{B}_ess_heatmap.pdf"))


def fig_tau2_magnitud(diag_tau2, dir_out):
    B = diag_tau2["B"]
    med = diag_tau2["med_log_tau2"]
    labels = diag_tau2["labels"]
    p_B, Tt = med.shape
    fig, ax = plt.subplots(figsize=(max(6.5, 0.52 * Tt + 3.2),
                                    max(3.2, 0.30 * p_B + 1.6)),
                           constrained_layout=True)
    lim = np.nanmax(np.abs(med))
    im = ax.imshow(med, aspect="auto", cmap=CMAP_SHRINK,
                   norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
                   interpolation="nearest")
    _ticks_periodo_heatmap(ax, Tt)
    ax.set_yticks(np.arange(p_B)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Periodo"); ax.grid(False)
    ax.set_title(rf"Bloque {B} · mediana posterior de $\log\tau^2$   "
                 "frío = shrinkage fuerte, cálido = innovación libre", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(r"$\log\tau^2$"); cb.outline.set_visible(False)
    _guardar(fig, os.path.join(dir_out, f"tau2_{B}_magnitud.pdf"))


def fig_tau2_comparacion(comp, dir_out):
    fig, ax = plt.subplots(figsize=(8.2, 5), constrained_layout=True)
    data = [comp[B]["log_tau2"] for B in ("E", "M", "D")]
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for pc, B in zip(parts["bodies"], ("E", "M", "D")):
        pc.set_facecolor(COL_BLOQUE[B]); pc.set_alpha(0.55)
        pc.set_edgecolor(TINTA_SUAVE); pc.set_linewidth(0.6)
    if "cmedians" in parts:
        parts["cmedians"].set_color(TINTA); parts["cmedians"].set_linewidth(1.4)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["E", "M", "D"])
    ax.set_ylabel(r"$\log\tau^2$")
    ax.set_title(r"Distribución posterior de $\log\tau^2$ por bloque", loc="left")
    ax.text(0.0, 1.015, "valores menores → shrinkage más fuerte (menos señal)",
            transform=ax.transAxes, fontsize=8.5, color=TINTA_SUAVE)
    _estilizar_ejes(ax)
    _guardar(fig, os.path.join(dir_out, "tau2_comparacion_bloques.pdf"))


def fig_tau2_acoplamiento(acop, dir_out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, B in zip(axes, ("E", "M", "D")):
        d = acop[B]
        x = d["log_abs_omega"]; y = d["log_tau2_med"]
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], s=10, alpha=0.35, color=COL_BLOQUE[B],
                   edgecolors="none")
        if np.isfinite(d["pendiente"]) and mask.sum() > 2:
            xs = np.array([x[mask].min(), x[mask].max()])
            b, a = np.polyfit(x[mask], y[mask], 1)
            ax.plot(xs, a + b * xs, color=CARMIN, lw=2, ls=(0, (4, 3)))
        ax.set_xlabel(r"$\log|\omega|$ posterior")
        ax.set_ylabel(r"$\log\tau^2$ posterior")
        ax.set_title(rf"Bloque {B} · pendiente = {d['pendiente']:.3f}",
                     fontsize=10.5, loc="left")
        _estilizar_ejes(ax)
    fig.suptitle(r"Acoplamiento $\omega$–$\tau^2$ (coherencia del muestreador)",
                 fontweight="bold")
    _guardar(fig, os.path.join(dir_out, "tau2_acoplamiento_omega.pdf"))


# ===========================================================================
# PREDICTIVO
# ===========================================================================

def fig_ppc_estadisticos(ppc, dir_out):
    stats_rep = ppc["stats_rep"]; obs = ppc["obs"]; pval = ppc["pval"]
    orden = [("media", "Media"), ("sd", "Desviación estándar"),
             ("q05", "Cuantil 5%"), ("q25", "Cuantil 25%"),
             ("q75", "Cuantil 75%"), ("q95", "Cuantil 95%"),
             ("skew", "Asimetría"), ("kurt", "Curtosis (exceso)")]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10.4), constrained_layout=True)
    for i, (k, tit) in enumerate(orden):
        ax = axes.ravel()[i]
        rep = stats_rep[k]; ov = obs[k]; p = pval[k]
        extremo = (p < 0.05 or p > 0.95)
        barras = TERRA if extremo else AZUL
        ax.hist(rep, bins=30, color=barras, alpha=0.55, edgecolor=PAPEL,
                linewidth=0.4)
        ax.axvline(ov, color=CARMIN, lw=2.4)
        flag = "  ▲" if extremo else ""
        col = CARMIN if extremo else TINTA
        # título en dos líneas: nombre (arriba) y p-valor (abajo, más chico)
        ax.set_title(f"{tit}\n$p_B = {p:.3f}${flag}", fontsize=10, color=col,
                     loc="left")
        ax.set_xlabel(r"$T(y^{\mathrm{rep}})$")
        ax.set_yticks([])
        _estilizar_ejes(ax)
    axes.ravel()[8].axis("off")
    axes.ravel()[8].text(
        0.5, 0.5, "línea vertical = $T(y_{\\mathrm{obs}})$\n\n"
        "barra terracota:\n$p_B$ extremo (mal ajuste)",
        ha="center", va="center", color=TINTA_SUAVE, fontsize=10.5)
    fig.suptitle("Chequeos predictivos posteriores · estadísticos",
                 fontweight="bold")
    _guardar(fig, os.path.join(dir_out, "pred_ppc_estadisticos.pdf"))


def fig_ppc_temporal(muestras, ppc, dir_out):
    """PPC de media y SD por periodo: observado vs banda replicada."""
    meta = muestras.meta
    Tt = meta.Tt
    x = np.arange(1, Tt + 1)
    xt, xlabs = _labels_tiempo(meta)
    mean_rep_t = ppc["mean_rep_t"]; mean_obs_t = ppc["mean_obs_t"]
    sd_rep_t = ppc["sd_rep_t"]; sd_obs_t = ppc["sd_obs_t"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), constrained_layout=True)
    for ax, rep, ob, tit in zip(
            axes, (mean_rep_t, sd_rep_t), (mean_obs_t, sd_obs_t),
            ("Media por periodo", "Desviación estándar por periodo")):
        lo = np.nanpercentile(rep, 5, axis=0)
        hi = np.nanpercentile(rep, 95, axis=0)
        md = np.nanpercentile(rep, 50, axis=0)
        ax.fill_between(x, lo, hi, color=AZUL, alpha=0.18, lw=0,
                        label="IC 90% replicado")
        ax.plot(x, md, color=AZUL, lw=1.6, label="Mediana replicada")
        ax.plot(x, ob, "o-", color=CARMIN, lw=2, ms=5, label="Observado",
                markeredgecolor=PAPEL, markeredgewidth=0.6)
        # eje X con años raleados (x va 1..Tt -> mapear a etiquetas 1..Tt)
        lab_periodo = xlabs[1:] if len(xlabs) == Tt + 1 else [str(i) for i in x]
        _ralear_ticks_tiempo(ax, x, lab_periodo,
                             cada=max(2, (Tt + 2) // 6))
        ax.set_xlabel("Periodo"); ax.set_title(tit, loc="left")
        ax.legend(fontsize=8.5, handlelength=1.4)
        _estilizar_ejes(ax)
    fig.suptitle("Chequeo predictivo por periodo · ajuste temporal",
                 fontweight="bold")
    _guardar(fig, os.path.join(dir_out, "pred_ppc_temporal.pdf"))


def fig_pit(pit_vals, k_all, dir_out, ks_p=None):
    """Histograma PIT global + boxplot por departamento."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    axes[0].hist(pit_vals, bins=20, density=True, color=AZUL, alpha=0.55,
                 edgecolor=PAPEL, linewidth=0.5)
    axes[0].axhline(1, color=CARMIN, ls=(0, (4, 3)), lw=1.8,
                    label="uniforme ideal")
    tit = "PIT · mezcla predictiva exacta"
    if ks_p is not None:
        tit += f"   (KS vs U(0,1): p = {ks_p:.3f})"
    axes[0].set_title(tit, loc="left")
    axes[0].set_xlabel("PIT"); axes[0].set_ylabel("densidad")
    axes[0].set_ylim(0, 2); axes[0].legend(handlelength=1.4)
    _estilizar_ejes(axes[0])

    # boxplot por depto
    m = int(k_all.max())
    data = [pit_vals[k_all == kk] for kk in range(1, m + 1)]
    bp = axes[1].boxplot(data, showfliers=False, patch_artist=True,
                         medianprops=dict(color=TINTA, lw=1.2),
                         whiskerprops=dict(color=TINTA_SUAVE, lw=0.8),
                         capprops=dict(color=TINTA_SUAVE, lw=0.8),
                         boxprops=dict(lw=0))
    for patch in bp["boxes"]:
        patch.set_facecolor(AZUL); patch.set_alpha(0.5)
    axes[1].axhline(0.5, color=CARMIN, ls=(0, (4, 3)), lw=1)
    axes[1].set_title("PIT por departamento", loc="left")
    axes[1].set_xlabel("departamento (k)"); axes[1].set_ylabel("PIT")
    _ticks_indices(axes[1], m)
    _estilizar_ejes(axes[1])
    _guardar(fig, os.path.join(dir_out, "pred_pit.pdf"))


def fig_residuos(res, k_all, t_all, dir_out, seed=0):
    """Panel de diagnóstico de residuos estandarizados."""
    from scipy import stats as _st
    rstd = res["resid_std"]; resid = res["resid"]; yhat = res["yhat"]
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    # hist + normal
    axes[0, 0].hist(rstd, bins=60, density=True, color=AZUL, alpha=0.5,
                    edgecolor=PAPEL, linewidth=0.3)
    xs = np.linspace(rstd.min(), rstd.max(), 200)
    axes[0, 0].plot(xs, _st.norm.pdf(xs), color=CARMIN, lw=2)
    axes[0, 0].set_title("Residuos estandarizados", loc="left")
    axes[0, 0].set_yticks([]); _estilizar_ejes(axes[0, 0])

    # QQ normal (submuestra)
    n_sub = min(20000, rstd.size)
    sub = rng.choice(rstd, n_sub, replace=False)
    _st.probplot(sub, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("QQ-plot vs Normal", loc="left")
    axes[0, 1].get_lines()[0].set_markersize(2)
    axes[0, 1].get_lines()[0].set_color(AZUL)
    axes[0, 1].get_lines()[0].set_alpha(0.5)
    axes[0, 1].get_lines()[1].set_color(CARMIN)
    axes[0, 1].get_lines()[1].set_linewidth(1.8)
    axes[0, 1].set_xlabel("cuantiles teóricos")
    axes[0, 1].set_ylabel("cuantiles observados")
    _estilizar_ejes(axes[0, 1])

    # ACF
    _acf_plot(axes[0, 2], rstd[:min(5000, rstd.size)], 30)
    axes[0, 2].set_title("ACF de residuos", loc="left")

    # resid vs ajustados
    n_sub2 = min(5000, resid.size)
    idx = rng.choice(resid.size, n_sub2, replace=False)
    axes[1, 0].scatter(yhat[idx], resid[idx], s=3, alpha=0.25, color=AZUL,
                       edgecolors="none")
    axes[1, 0].axhline(0, color=CARMIN, ls=(0, (4, 3)), lw=1.2)
    axes[1, 0].set_title("Residuos vs ajustados", loc="left")
    axes[1, 0].set_xlabel(r"$\hat{y}$"); axes[1, 0].set_ylabel("residuo")
    _estilizar_ejes(axes[1, 0])

    # boxplot por periodo
    Tt = int(t_all.max())
    _boxplot_estilo(axes[1, 1], [rstd[t_all == tt] for tt in range(1, Tt + 1)],
                    AZUL)
    axes[1, 1].axhline(0, color=CARMIN, ls=(0, (4, 3)), lw=1)
    axes[1, 1].set_title("Residuos std por periodo", loc="left")
    axes[1, 1].set_xlabel("periodo (t)")
    _ticks_indices(axes[1, 1], Tt, cada=1 if Tt <= 12 else 2)
    _estilizar_ejes(axes[1, 1])

    # boxplot por depto
    m = int(k_all.max())
    _boxplot_estilo(axes[1, 2], [rstd[k_all == kk] for kk in range(1, m + 1)],
                    TERRA)
    axes[1, 2].axhline(0, color=CARMIN, ls=(0, (4, 3)), lw=1)
    axes[1, 2].set_title("Residuos std por departamento", loc="left")
    axes[1, 2].set_xlabel("departamento (k)")
    _ticks_indices(axes[1, 2], m)
    _estilizar_ejes(axes[1, 2])

    fig.suptitle("Diagnóstico de residuos estructurales", fontweight="bold")
    _guardar(fig, os.path.join(dir_out, "pred_residuos.pdf"))


def fig_crps_heatmap(crps, k_all, t_all, meta, dir_out):
    """CRPS medio por (departamento × periodo)."""
    m, Tt = meta.m, meta.Tt
    M = np.full((m, Tt), np.nan)
    for kk in range(1, m + 1):
        for tt in range(1, Tt + 1):
            mask = (k_all == kk) & (t_all == tt)
            if mask.sum() > 0:
                M[kk - 1, tt - 1] = crps[mask].mean()
    dep = meta.depto_ids if meta.depto_ids else [f"k={i+1}" for i in range(m)]
    fig, ax = plt.subplots(figsize=(max(6.5, 0.52 * Tt + 3.2),
                                    max(4, 0.30 * m + 1.6)),
                           constrained_layout=True)
    im = ax.imshow(M, aspect="auto", cmap=CMAP_CALOR, interpolation="nearest")
    _ticks_periodo_heatmap(ax, Tt)
    ax.set_yticks(np.arange(m)); ax.set_yticklabels(dep, fontsize=7.5)
    ax.set_xlabel("Periodo"); ax.grid(False)
    ax.set_title("CRPS por departamento × periodo   "
                 "(más oscuro = peor predicción)", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("CRPS")
    cb.outline.set_visible(False)
    _guardar(fig, os.path.join(dir_out, "pred_crps_heatmap.pdf"))


def _boxplot_estilo(ax, data, color):
    bp = ax.boxplot(data, showfliers=False, patch_artist=True,
                    medianprops=dict(color=TINTA, lw=1.1),
                    whiskerprops=dict(color=TINTA_SUAVE, lw=0.8),
                    capprops=dict(color=TINTA_SUAVE, lw=0.8),
                    boxprops=dict(lw=0))
    for patch in bp["boxes"]:
        patch.set_facecolor(color); patch.set_alpha(0.5)
    return bp


def _ticks_indices(ax, n, cada=None):
    """Etiquetas de índice 1..n raleadas (para ejes de departamento/periodo)."""
    if cada is None:
        cada = 1 if n <= 12 else (2 if n <= 24 else 4)
    pos = list(range(1, n + 1, cada))
    if n not in pos:
        pos.append(n)
    ax.set_xticks(pos)
    ax.set_xticklabels([str(p) for p in pos], fontsize=7.5)


def fig_lambda2_comparacion(muestras, dir_out):
    lam = muestras.lambda2
    fig, ax = plt.subplots(figsize=(7.6, 5), constrained_layout=True)
    parts = ax.violinplot([lam[:, 0], lam[:, 1], lam[:, 2]],
                          showmedians=True, showextrema=False)
    for pc, B in zip(parts["bodies"], ("E", "M", "D")):
        pc.set_facecolor(COL_BLOQUE[B]); pc.set_alpha(0.55)
        pc.set_edgecolor(TINTA_SUAVE); pc.set_linewidth(0.6)
    if "cmedians" in parts:
        parts["cmedians"].set_color(TINTA); parts["cmedians"].set_linewidth(1.4)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["E", "M", "D"])
    ax.set_ylabel(r"$\lambda^2$")
    ax.set_title(r"Posterior de $\lambda^2$ por bloque", loc="left")
    ax.text(0.0, 1.015, r"mayor $\lambda^2$ → shrinkage global más fuerte",
            transform=ax.transAxes, fontsize=8.5, color=TINTA_SUAVE)
    _estilizar_ejes(ax)
    _guardar(fig, os.path.join(dir_out, "pred_lambda2_comparacion.pdf"))
