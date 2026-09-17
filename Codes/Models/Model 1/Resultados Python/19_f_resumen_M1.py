# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# @  MODELO 1 - NORMAL JERARQUICO ESTATICO CON COVARIABLES Y LASSO BAYESIANO
# _____________________________________________________________________________

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

_AQUI = Path(__file__).resolve().parent


def _importar(nombre_archivo, nombre_modulo):
    spec = importlib.util.spec_from_file_location(nombre_modulo, _AQUI / nombre_archivo)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


u = _importar("19_0_utilidades_M1.py", "util_m1")
m_fit = _importar("19_c_ajuste_M1.py", "ajuste_m1")     # calcular_waic / _loo / _dic
m_ppc = _importar("19_d_ppc_M1.py", "ppc_m1")           # correr_ppc / tablas
plt = u.plt


# =============================================================================
# 0. DIAGNOSTICOS Y RESUMENES
# =============================================================================
BLOQUES_DIAG = ["mu", "sigma2_mu", "beta_E", "beta_M", "beta_D",
                "tau2_E", "tau2_M", "tau2_D", "lambda2", "kappa2_k", "kappa2_jk"]


def diagnosticos_todos(cad):
    """R-hat / ESS de todos los parametros monitoreados -> DataFrame."""
    filas = []
    for b in BLOQUES_DIAG:
        if not cad.disponible(b):
            continue
        X = u.preparar_bloque(cad, b)
        d = u.diagnosticos_mcmc(X)
        filas.append(pd.DataFrame({"bloque": b, "rhat": d["rhat"],
                                   "ess_bulk": d["ess_bulk"],
                                   "ess_tail": d["ess_tail"]}))
    return pd.concat(filas, ignore_index=True)


def resumen_coeficientes(sal):
    """Media/IC95/senal por coeficiente y nivel (E, M, D)."""
    filas = []
    for niv in "EMD":
        b = f"beta_{niv}"
        if not sal.disponible(b):
            continue
        X = sal.bloque(b)
        if X.ndim == 1:
            X = X[:, None]
        q = np.quantile(X, [0.025, 0.5, 0.975], axis=0)
        senal = (q[0] > 0) | (q[2] < 0)
        filas.append(pd.DataFrame({
            "nivel": niv, "parametro": sal.etiquetas(b),
            "mediana": q[1], "q025": q[0], "q975": q[2],
            "media": X.mean(axis=0), "senal": senal}))
    return pd.concat(filas, ignore_index=True)


# =============================================================================
# 1. COLORES / ESTILO LOCAL
# =============================================================================
def _anota(ax, texto, loc="upper left", color="#000000"):
    """Caja de anotacion con la conclusion, en una esquina del eje."""
    xy = {"upper left": (0.03, 0.97, "left", "top"),
          "upper right": (0.97, 0.97, "right", "top"),
          "lower right": (0.97, 0.05, "right", "bottom"),
          "lower left": (0.03, 0.05, "left", "bottom")}[loc]
    ax.text(xy[0], xy[1], texto, transform=ax.transAxes, ha=xy[2], va=xy[3],
            fontsize=9.5, color=color,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color,
                      alpha=0.9, lw=1.0))


# =============================================================================
# 2. FIGURA 1 - CONVERGENCIA
# =============================================================================
def fig_convergencia(cad, diag, ll_tot, dir_diag, pdf):
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.9))

    # (a) R-hat
    rh = diag["rhat"].dropna().to_numpy()
    n_ok = int((rh <= u.UMBRAL_RHAT).sum())
    axs[0].hist(rh, bins=35, color=u.PALETA["azul"], edgecolor="white", lw=0.3)
    axs[0].axvline(u.UMBRAL_RHAT, color=u.PALETA["rojo"], ls="--", lw=1.4)
    axs[0].set_title(r"Factor de reduccion de escala $\widehat{R}$")
    axs[0].set_xlabel(r"$\widehat{R}$ (rango-normalizado y plegado)")
    axs[0].set_ylabel("n.o de parametros")
    _anota(axs[0], f"{n_ok}/{len(rh)} con $\\widehat{{R}}<{u.UMBRAL_RHAT}$\n"
                   f"max = {rh.max():.3f}",
           "upper right", u.PALETA["azul"])

    # (b) ESS bulk & tail
    eb = diag["ess_bulk"].dropna().to_numpy()
    et = diag["ess_tail"].dropna().to_numpy()
    axs[1].hist(eb, bins=30, color=u.PALETA["verde"], alpha=0.75, label="ESS bulk")
    axs[1].hist(et, bins=30, color=u.PALETA["naranja"], alpha=0.6, label="ESS cola")
    axs[1].axvline(u.UMBRAL_ESS, color=u.PALETA["rojo"], ls="--", lw=1.4)
    axs[1].set_title("Tamano de muestra efectivo (ESS)")
    axs[1].set_xlabel("muestras efectivas"); axs[1].set_ylabel("n.o de parametros")
    axs[1].legend(fontsize=8.5, loc="upper right")
    _anota(axs[1], f"ESS bulk min = {eb.min():.0f}\nESS cola min = {et.min():.0f}",
           "upper left", u.PALETA["verde"])

    # (c) traza estacionaria (log-verosimilitud total, o mu)
    if ll_tot is not None:
        X = ll_tot; etiq = "log-verosimilitud total"
    else:
        X = u.preparar_bloque(cad, "mu")[0]; etiq = r"$\mu$"
    m, n = X.shape
    col = [u.PALETA[c] for c in ("azul", "naranja", "verde", "morado")]
    for c in range(m):
        axs[2].plot(X[c], lw=0.5, alpha=0.7, color=col[c % len(col)],
                    label=f"cadena {c+1}" if m > 1 else None)
    comb = X.mean(axis=0)
    axs[2].plot(np.cumsum(comb) / np.arange(1, n + 1), color="black", lw=1.6,
                ls="--", label="media acumulada")
    axs[2].set_title(f"Traza de {etiq} (estacionariedad)")
    axs[2].set_xlabel("muestra guardada")
    if m > 1:
        axs[2].legend(fontsize=8.5, loc="lower right", ncols=2)

    ok = (n_ok == len(rh)) and (eb.min() >= u.UMBRAL_ESS) and (et.min() >= u.UMBRAL_ESS)
    verd = ("Convergencia satisfactoria: las cadenas son validas para inferencia"
            if ok else
            "Revisar: hay parametros fuera de umbral (alargar la corrida o anadir cadenas)")
    fig.suptitle(f"Convergencia del muestreador  —  {verd}", y=1.02,
                 color=(u.PALETA["verde"] if ok else u.PALETA["rojo"]))
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_resumen_1_convergencia", pdf)
    return ok


# =============================================================================
# 3. FIGURA 2 - COEFICIENTES POR NIVEL
# =============================================================================
def fig_coeficientes(tab, dir_diag, pdf, max_sin_filtrar=22):
    niveles = [n for n in "EMD" if (tab["nivel"] == n).any()]
    fig, axs = plt.subplots(1, len(niveles), figsize=(5.2 * len(niveles), 6.6),
                            squeeze=False)
    axs = axs[0]
    for ax, niv in zip(axs, niveles):
        g = tab[tab["nivel"] == niv].copy()
        n_total = len(g)
        n_senal = int(g["senal"].sum())
        # con muchas covariables, mostrar solo las de senal (LASSO encogio el resto)
        filtrado = n_total > max_sin_filtrar
        if filtrado:
            gm = g[g["senal"]].copy()
            if len(gm) == 0:                       # ninguna con senal: top-10 |media|
                gm = g.reindex(g["media"].abs().sort_values(ascending=False).index).head(10)
        else:
            gm = g
        gm = gm.sort_values("mediana").reset_index(drop=True)
        yy = np.arange(len(gm))
        color = np.where(gm["senal"], u.NIVEL_COLOR[niv], u.PALETA["gris"])
        for i in yy:
            ax.plot([gm["q025"][i], gm["q975"][i]], [i, i],
                    color=color[i], lw=2.2 if gm["senal"][i] else 1.4,
                    alpha=0.95 if gm["senal"][i] else 0.6, solid_capstyle="round")
        ax.scatter(gm["mediana"], yy, s=np.where(gm["senal"], 46, 26),
                   color=color, zorder=3,
                   edgecolor="white", linewidth=0.6)
        ax.axvline(0, color="black", ls="--", lw=1.0, alpha=0.7)
        ax.set_yticks(yy)
        ax.set_yticklabels([p[:30] for p in gm["parametro"]], fontsize=8.2)
        ax.set_xlabel("coeficiente estandarizado (mediana e IC 95%)")
        ax.set_title(u.NIVEL_NOMBRE[niv], color=u.NIVEL_COLOR[niv])
        nota = f"{n_senal} de {n_total} con senal (IC 95% no cruza 0)"
        if filtrado:
            nota += f"\n(se muestran solo las de senal;\n {n_total - n_senal} encogidas a ~0)"
        _anota(ax, nota, "lower right", u.NIVEL_COLOR[niv])
    fig.suptitle("Efecto de las covariables sobre el puntaje global "
                 "(color = senal; gris = encogida por el LASSO)", y=1.01)
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_resumen_2_coeficientes", pdf)


# =============================================================================
# 4. FIGURA 3 - HETEROGENEIDAD RESIDUAL
# =============================================================================
def fig_heterogeneidad(sal, dir_diag, pdf):
    # --- departamental: kappa_k = sqrt(kappa2_k) ---
    K2k = sal.bloque("kappa2_k")
    if K2k.ndim == 1:
        K2k = K2k[:, None]
    kap = np.sqrt(K2k)
    med = np.median(kap, axis=0)
    q = np.quantile(kap, [0.025, 0.975], axis=0)
    dep = sal.nombres.get("deptos")
    etiqs = (dep["depto_id"].astype(str).tolist() if dep is not None
             else [f"D{i+1}" for i in range(len(med))])
    orden = np.argsort(med)

    # panel municipal solo si hay n_jk y kappa2_jk
    cel = sal.nombres.get("celdas")
    hay_muni = (sal.disponible("kappa2_jk") and cel is not None
                and "n_jk" in cel.columns and cel["n_jk"].notna().any())

    if hay_muni:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.5, 6.2),
                                       gridspec_kw={"width_ratios": [1.05, 1]})
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(8.5, 6.2))

    yy = np.arange(len(med))
    ax0.hlines(yy, q[0][orden], q[1][orden], color=u.PALETA["naranja"], lw=2.0)
    ax0.scatter(med[orden], yy, color=u.PALETA["rojo"], s=42, zorder=3,
                edgecolor="white", lw=0.6)
    kap_global = med.mean()
    ax0.axvline(kap_global, color="black", ls="--", lw=1.1)
    ax0.set_yticks(yy); ax0.set_yticklabels([etiqs[i] for i in orden], fontsize=8.6)
    ax0.set_xlabel(r"desviacion residual por departamento  $\kappa_k=\sqrt{\kappa^2_k}$  (puntos)")
    ax0.set_title("Heterogeneidad residual entre departamentos")
    _anota(ax0, f"rango: {med.min():.1f} – {med.max():.1f} pts\n"
                f"promedio $\\approx$ {kap_global:.1f} pts",
           "lower right", u.PALETA["rojo"])

    if hay_muni:
        K2jk = sal.bloque("kappa2_jk")
        kap_m = np.sqrt(np.median(K2jk, axis=0))
        njk = cel["n_jk"].to_numpy(dtype=float)
        ok = np.isfinite(njk) & (njk > 0)
        ax1.scatter(njk[ok], kap_m[ok], s=18, alpha=0.55, color=u.PALETA["verde"],
                    edgecolor="white", lw=0.3)
        ax1.axhline(kap_global, color="black", ls="--", lw=1.1,
                    label=r"$\kappa$ global")
        ax1.set_xscale("log")
        ax1.set_xlabel(r"n.o de estudiantes en el municipio  $n_{jk}$  (log)")
        ax1.set_ylabel(r"$\kappa_{jk}=\sqrt{\kappa^2_{jk}}$  (mediana posterior)")
        ax1.set_title("Encogimiento de la varianza municipal")
        ax1.legend(fontsize=9, loc="upper right")
        _anota(ax1, "municipios pequenos se contraen\nhacia la escala global "
                    "(estabilidad)", "lower left", u.PALETA["verde"])

    fig.suptitle("Estructura jerarquica de la varianza residual (niveles D y M)",
                 y=1.01)
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_resumen_3_heterogeneidad", pdf)


# =============================================================================
# 5. FIGURA 4 - ADECUACION DEL MODELO NORMAL
# =============================================================================
def fig_adecuacion(P, dir_diag, pdf):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # (a) densidad observada vs replicada
    rej, y = P["rej"], P["y"]
    dens_obs = stats.gaussian_kde(y)(rej)
    banda = np.quantile(P["dens_rep"], [0.025, 0.975], axis=0)
    ax0.fill_between(rej, banda[0], banda[1], color=u.PALETA["celeste"],
                     alpha=0.45, label="replicado (banda 95%)")
    ax0.plot(rej, P["dens_rep"].mean(axis=0), color=u.PALETA["azul"], lw=1.2,
             label="replicado (media)")
    ax0.plot(rej, dens_obs, color=u.PALETA["rojo"], lw=2.4, label="observado")
    ax0.set_xlabel("puntaje global"); ax0.set_yticks([])
    ax0.set_title("Densidad observada vs. replicada")
    ax0.legend(fontsize=9, loc="upper right")
    cob95 = P["cobertura"]["95"].mean() * 100
    _anota(ax0, f"cobertura IP 95% = {cob95:.1f}%\n"
                f"$R^2$ bayes $\\approx$ {P['r2'].mean():.2f}",
           "upper left", u.PALETA["azul"])

    # (b) QQ de normalidad de los residuos via PIT
    pit = np.clip(P["pit"], 1e-6, 1 - 1e-6)
    z = norm.ppf(pit)
    z.sort()
    N = z.size
    teor = norm.ppf((np.arange(1, N + 1) - 0.5) / N)
    ax1.scatter(teor, z, s=10, alpha=0.5, color=u.PALETA["azul"],
                edgecolor="none")
    lim = [min(teor[0], z[0]) - 0.2, max(teor[-1], z[-1]) + 0.2]
    ax1.plot(lim, lim, color=u.PALETA["rojo"], ls="--", lw=1.4)
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel("cuantiles teoricos  N(0,1)")
    ax1.set_ylabel(r"cuantiles de  $\Phi^{-1}(\mathrm{PIT})$")
    curt = stats.kurtosis(z)   # exceso
    ax1.set_title("Normalidad de los residuos (QQ vía PIT)")
    diag_colas = ("colas mas pesadas que Normal\n-> justifica el Modelo 2 (t-Student)"
                  if curt > 0.3 else
                  "colas compatibles con Normal" if abs(curt) <= 0.3 else
                  "colas mas livianas que Normal")
    col = u.PALETA["rojo"] if curt > 0.3 else u.PALETA["verde"]
    _anota(ax1, f"curtosis en exceso = {curt:+.2f}\n{diag_colas}",
           "lower right", col)

    fig.suptitle("Adecuacion del modelo Normal — chequeo predictivo posterior",
                 y=1.02)
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_resumen_4_adecuacion", pdf)
    return dict(cob95=cob95, curt=curt, r2=float(P["r2"].mean()))


# =============================================================================
# 6. FIGURA 5 - COMPARACION M1 vs M2
# =============================================================================
def fig_comparacion(sal_A, sal_B, nombre_A, nombre_B, dir_diag, pdf):
    if not (sal_A.disponible("loglik") and sal_B.disponible("loglik")):
        print("  (comparacion omitida: falta loglik.bin en algun modelo)")
        return None
    mmA = sal_A.bloque("loglik", memmap=True)
    mmB = sal_B.bloque("loglik", memmap=True)
    if mmA.shape[1] != mmB.shape[1]:
        print(f"  (comparacion omitida: N distinto {mmA.shape[1]} vs {mmB.shape[1]})")
        return None
    lA = m_fit.calcular_loo(mmA)
    lB = m_fit.calcular_loo(mmB)
    N = mmA.shape[1]
    d_i = lA["elpd_i"] - lB["elpd_i"]
    delta = d_i.sum(); se = np.sqrt(N) * d_i.std(ddof=1)
    mejor = nombre_A if delta > 0 else nombre_B
    creible = abs(delta) > 2 * se

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.0, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    # (a) elpd por modelo (dumbbell)
    vals = [lA["elpd_loo"], lB["elpd_loo"]]
    nombres = [nombre_A, nombre_B]
    colores = [u.PALETA["azul"], u.PALETA["naranja"]]
    yy = [1, 0]
    ax0.hlines(yy, [min(vals)] * 2, vals, color=u.PALETA["gris"], lw=1.0, ls=":")
    ax0.scatter(vals, yy, s=140, color=colores, zorder=3, edgecolor="white", lw=1)
    for v, y0, nm in zip(vals, yy, nombres):
        ax0.text(v, y0 + 0.12, f"{v:.0f}", ha="center", fontsize=10, weight="bold")
    ax0.set_yticks(yy); ax0.set_yticklabels(nombres)
    ax0.set_xlabel("elpd LOO  (mayor = mejor ajuste predictivo)")
    ax0.set_ylim(-0.5, 1.5)
    ax0.set_title("Capacidad predictiva (LOO)")

    # (b) diferencia puntual acumulada
    orden = np.argsort(d_i)
    ax1.plot(np.arange(N), np.cumsum(d_i[orden]), color=u.PALETA["morado"], lw=1.6)
    ax1.axhline(0, color="black", ls="--", lw=1.0)
    ax1.set_xlabel("observaciones (ordenadas por diferencia)")
    ax1.set_ylabel(f"suma acumulada  elpd({nombre_A[:12]}) − elpd({nombre_B[:12]})")
    ax1.set_title("Origen de la diferencia por observacion")
    signo = "+" if delta >= 0 else ""
    _anota(ax1, f"$\\Delta$elpd = {signo}{delta:.1f}  (SE {se:.1f})\n"
                f"{'diferencia creible (>2 SE)' if creible else 'no concluyente (<2 SE)'}",
           "upper left", u.PALETA["morado"])

    fig.suptitle(f"Comparacion de modelos — preferido por LOO: {mejor}"
                 + ("" if creible else "  (diferencia no concluyente)"),
                 y=1.02, color=(u.PALETA["verde"] if creible else u.PALETA["gris"]))
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_resumen_5_comparacion", pdf)
    return dict(elpd_A=lA["elpd_loo"], elpd_B=lB["elpd_loo"],
                delta=delta, se=se, mejor=mejor, creible=creible)


# =============================================================================
# 7. ORQUESTADOR
# =============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="19f: figuras minimas de resultados para el informe")
    ap.add_argument("dirs", nargs="+", help="carpeta(s) de correr_gibbs_disco (M1)")
    ap.add_argument("--diag", default="figuras_informe", help="carpeta de salida")
    ap.add_argument("--comparar", default=None,
                    help="carpeta de otro modelo (M2) para la figura de comparacion")
    ap.add_argument("--nombres", nargs=2, default=["Normal (M1)", "t-Student (M2)"])
    ap.add_argument("--draws", type=int, default=400,
                    help="draws del posterior para el chequeo predictivo")
    ap.add_argument("--pdf", action="store_true", help="guardar tambien PDF vectorial")
    args = ap.parse_args()

    u.estilo()
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12.5,
                         "axes.labelsize": 11})
    dir_diag = Path(args.diag); dir_diag.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("FIGURAS DE RESULTADOS PARA EL INFORME - Modelo 1 (19f)")
    print("=" * 72)
    cad = u.CadenasM1(args.dirs)
    sal = u.SalidaPool(args.dirs)
    b = cad.base
    print(f"Cadenas: {cad.m} | muestras: {sal.n_save} | "
          f"N={b.meta.get('N','?')}, K={b.meta.get('K','?')}, "
          f"n_cells={b.meta.get('n_cells','?')} | "
          f"p_E={b.meta.get('p_E',0)}, p_M={b.meta.get('p_M',0)}, "
          f"p_D={b.meta.get('p_D',0)}\n")

    resumen = {}

    # ---- Figura 1: convergencia ----
    print("[1/4] Convergencia...")
    diag = diagnosticos_todos(cad)
    ll_tot = None
    if all(s.disponible("loglik") for s in cad.salidas):
        series = [u.serie_loglik_total(s)[: cad.n_min] for s in cad.salidas]
        ll_tot = np.stack(series, axis=0)
    ok_conv = fig_convergencia(cad, diag, ll_tot, dir_diag, args.pdf)
    resumen.update(rhat_max=float(np.nanmax(diag["rhat"])),
                   ess_bulk_min=float(np.nanmin(diag["ess_bulk"])),
                   ess_tail_min=float(np.nanmin(diag["ess_tail"])),
                   convergencia_ok=bool(ok_conv))

    # ---- Figura 2: coeficientes ----
    print("[2/4] Coeficientes por nivel...")
    tab = resumen_coeficientes(sal)
    fig_coeficientes(tab, dir_diag, args.pdf)
    for niv in "EMD":
        g = tab[tab["nivel"] == niv]
        if len(g):
            resumen[f"senal_{niv}"] = f"{int(g['senal'].sum())}/{len(g)}"

    # ---- Figura 3: heterogeneidad ----
    print("[3/4] Heterogeneidad residual...")
    fig_heterogeneidad(sal, dir_diag, args.pdf)
    kapk = np.sqrt(np.median(sal.bloque("kappa2_k"), axis=0))
    resumen.update(kappa_k_min=float(kapk.min()), kappa_k_max=float(kapk.max()))

    # ---- Figura 4: adecuacion (PPC) ----
    print("[4/4] Adecuacion (chequeo predictivo)...")
    if sal.datos_ppc() is not None:
        P = m_ppc.correr_ppc(sal, n_draws=args.draws)
        info = fig_adecuacion(P, dir_diag, args.pdf)
        resumen.update(cobertura_95=info["cob95"], curtosis_pit=info["curt"],
                       r2_bayes=info["r2"])
    else:
        print("  OMITIDA: faltan insumos del PPC (y, X_E, Z, W). "
              "Re-exportar con la Etapa 18 usando datos=.")

    # ---- WAIC/LOO/DIC para el resumen (una cadena base) ----
    if b.disponible("loglik"):
        mm = b.bloque("loglik", memmap=True)
        w = m_fit.calcular_waic(mm)
        lo = m_fit.calcular_loo(mm)
        di = m_fit.calcular_dic(b, mm)
        resumen.update(WAIC=float(w["waic"]), LOOIC=float(lo["looic"]),
                       DIC=float(di["dic"]),
                       khat_max=float(np.nanmax(lo["khat"])))

    # ---- Figura 5: comparacion (opcional) ----
    comp = None
    if args.comparar:
        print("[+]  Comparacion con", args.comparar, "...")
        salB = u.SalidaPool([args.comparar])
        comp = fig_comparacion(sal, salB, args.nombres[0], args.nombres[1],
                               dir_diag, args.pdf)
        if comp:
            resumen.update(comparacion_delta_elpd=comp["delta"],
                           comparacion_se=comp["se"],
                           comparacion_preferido=comp["mejor"])

    # ---- resumen ejecutivo ----
    df_res = pd.DataFrame({"indicador": list(resumen.keys()),
                           "valor": [str(v) for v in resumen.values()]})
    df_res.to_csv(dir_diag / "resumen_ejecutivo_M1.csv", index=False,
                  encoding="utf-8")

    print("\n" + "=" * 72)
    print("RESUMEN EJECUTIVO")
    print("=" * 72)
    print(f"  Convergencia : R-hat max = {resumen['rhat_max']:.3f} | "
          f"ESS bulk min = {resumen['ess_bulk_min']:.0f} | "
          f"ESS cola min = {resumen['ess_tail_min']:.0f}  "
          f"-> {'OK' if resumen['convergencia_ok'] else 'REVISAR'}")
    print("  Coeficientes : " + " | ".join(
        f"{niv}: {resumen.get('senal_'+niv,'-')} con senal" for niv in "EMD"))
    print(f"  Heterogen.   : kappa_k entre {resumen['kappa_k_min']:.1f} y "
          f"{resumen['kappa_k_max']:.1f} pts")
    if "cobertura_95" in resumen:
        print(f"  Adecuacion   : cobertura IP95 = {resumen['cobertura_95']:.1f}% | "
              f"curtosis PIT = {resumen['curtosis_pit']:+.2f} | "
              f"R2 bayes = {resumen['r2_bayes']:.2f}")
    if "WAIC" in resumen:
        print(f"  Ajuste rel.  : WAIC={resumen['WAIC']:.0f} | "
              f"LOOIC={resumen['LOOIC']:.0f} | DIC={resumen['DIC']:.0f} | "
              f"khat max={resumen['khat_max']:.2f}")
    if comp:
        print(f"  Comparacion  : Delta_elpd={comp['delta']:+.1f} (SE {comp['se']:.1f})"
              f" -> preferido {comp['mejor']}"
              f"{' (creible)' if comp['creible'] else ' (no concluyente)'}")
    print("-" * 72)
    print(f"  Figuras y resumen_ejecutivo_M1.csv en: {dir_diag}/")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
