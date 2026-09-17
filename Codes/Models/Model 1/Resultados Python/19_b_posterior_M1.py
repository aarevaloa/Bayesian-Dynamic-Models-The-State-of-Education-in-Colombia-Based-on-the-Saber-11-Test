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

_AQUI = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("util_m1", _AQUI / "19_0_utilidades_M1.py")
u = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(u)
plt = u.plt


# =============================================================================
# 1. TABLAS
# =============================================================================
def tabla_coeficientes(sal: "u.SalidaM1", conv_csv=None):
    """Resumen por coeficiente (media, sd, IC95, pd, prob. de direccion) por
    nivel. Si se pasa el CSV de 19a, fusiona ESS bulk/tail y Rhat."""
    filas = []
    for niv in "EMD":
        b = f"beta_{niv}"
        if not sal.disponible(b):
            continue
        X = sal.bloque(b)                          # (n_save, p)
        etq = sal.etiquetas(b)
        med = X.mean(axis=0)
        q = np.quantile(X, [0.025, 0.5, 0.975], axis=0)
        p_pos = (X > 0).mean(axis=0)
        pd_dir = np.maximum(p_pos, 1 - p_pos)      # prob. de la direccion dominante
        df = pd.DataFrame({
            "nivel": niv, "parametro": etq,
            "media": med, "sd": X.std(axis=0, ddof=1),
            "q025": q[0], "mediana": q[1], "q975": q[2],
            "p_pos": p_pos, "pd": pd_dir,
            "senal_ic95": np.where((q[0] > 0) | (q[2] < 0), "*", ""),
            "senal_pd95": np.where(pd_dir > 0.95, "*", ""),
        })
        filas.append(df)
    tab = pd.concat(filas, ignore_index=True)

    if conv_csv and Path(conv_csv).exists():
        conv = pd.read_csv(conv_csv)[
            ["parametro", "ess_bulk", "ess_tail", "rhat"]]
        tab = tab.merge(conv, on="parametro", how="left")
    return tab


def tablas_kappa(sal: "u.SalidaM1"):
    """Varianzas por departamento (kappa2_k) y por municipio (kappa2_jk),
    con IC y desviacion en escala de puntaje. Devuelve (df_depto, df_muni)."""
    k2k = sal.bloque("kappa2_k")                   # (n_save, K)
    dep = sal.nombres.get("deptos")
    depto_id = dep["depto_id"].tolist() if dep is not None else \
        [f"D{k+1}" for k in range(k2k.shape[1])]
    q = np.quantile(k2k, [0.025, 0.5, 0.975], axis=0)
    df_depto = pd.DataFrame({
        "departamento": depto_id,
        "kappa2_k_media": k2k.mean(axis=0),
        "kappa2_k_q025": q[0], "kappa2_k_mediana": q[1], "kappa2_k_q975": q[2],
        "kappa_k_media": np.sqrt(k2k).mean(axis=0),   # desviacion (escala puntaje)
    })

    df_muni = None
    if sal.disponible("kappa2_jk"):
        k2jk = sal.bloque("kappa2_jk")             # (n_save, n_cells)
        cel = sal.nombres.get("celdas")
        qj = np.quantile(k2jk, [0.025, 0.5, 0.975], axis=0)
        df_muni = pd.DataFrame({
            "celda": np.arange(1, k2jk.shape[1] + 1),
            "kappa2_jk_media": k2jk.mean(axis=0),
            "kappa2_jk_q025": qj[0], "kappa2_jk_mediana": qj[1],
            "kappa2_jk_q975": qj[2],
        })
        if cel is not None and len(cel) == k2jk.shape[1]:
            df_muni.insert(1, "departamento", cel["depto_id"].values)
            df_muni.insert(2, "municipio", cel["muni_id"].values)
            if "n_jk" in cel:
                df_muni["n_jk"] = cel["n_jk"].values
    return df_depto, df_muni


# =============================================================================
# 2. FIGURAS
# =============================================================================
def fig_forest_por_nivel(sal, tab, dir_diag, pdf):
    """Forest plot (mediana + IC95%) por nivel. E puede tener muchos coefs -> su
    propio panel alto; M y D comparten un panel."""
    niveles = [n for n in "EMD" if (tab["nivel"] == n).any()]
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.4], width_ratios=[1, 1],
                          hspace=0.28, wspace=0.32)
    ax_E = fig.add_subplot(gs[:, 0])
    ax_M = fig.add_subplot(gs[0, 1])
    ax_D = fig.add_subplot(gs[1, 1])
    ejes = {"E": ax_E, "M": ax_M, "D": ax_D}

    for niv in niveles:
        ax = ejes[niv]
        g = tab[tab["nivel"] == niv].copy().sort_values("media")
        y = np.arange(len(g))
        senal = g["senal_ic95"].values == "*"
        colores = np.where(senal, u.NIVEL_COLOR[niv], u.PALETA["gris"])
        ax.hlines(y, g["q025"], g["q975"], color=colores, lw=2.0, alpha=0.85)
        ax.scatter(g["mediana"], y, color=colores, s=22, zorder=3,
                   edgecolor="white", linewidth=0.5)
        ax.axvline(0, color=u.PALETA["negro"], lw=1.0, ls="--", alpha=0.7)
        ax.set_yticks(y)
        etiquetas = [p[:34] for p in g["parametro"]]
        ax.set_yticklabels(etiquetas, fontsize=7.5 if niv == "E" else 8.5)
        ax.set_ylim(-0.8, len(g) - 0.2)
        n_sig = int(senal.sum())
        ax.set_title(f"{u.NIVEL_NOMBRE[niv]} - {n_sig}/{len(g)} con IC95% != 0",
                     color=u.NIVEL_COLOR[niv])
        ax.set_xlabel("coeficiente (mediana e IC95%)")

    fig.suptitle("Modelo 1 - coeficientes LASSO por nivel jerarquico "
                 "(color = senal; gris = encogido)", y=0.995)
    u.guardar_fig(fig, dir_diag, "fig_19b_01_forest_por_nivel", pdf)


def fig_encogimiento(sal, dir_diag, pdf):
    """Perfil de encogimiento: |media beta| vs escala local media tau2, por
    nivel, en escala log-x. Coeficientes senal (IC95 != 0) resaltados."""
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, niv in zip(axs, "EMD"):
        b, t = f"beta_{niv}", f"tau2_{niv}"
        if not (sal.disponible(b) and sal.disponible(t)):
            ax.axis("off"); continue
        B = sal.bloque(b); T = sal.bloque(t)
        med = np.abs(B.mean(axis=0))
        tau_med = T.mean(axis=0)
        q = np.quantile(B, [0.025, 0.975], axis=0)
        senal = (q[0] > 0) | (q[1] < 0)
        ax.scatter(tau_med[~senal], med[~senal], s=45, color=u.PALETA["gris"],
                   alpha=0.7, label="encogido", edgecolor="white", linewidth=0.5)
        ax.scatter(tau_med[senal], med[senal], s=55, color=u.NIVEL_COLOR[niv],
                   alpha=0.9, label="senal (IC95 != 0)", edgecolor="white",
                   linewidth=0.5)
        etq = sal.etiquetas(b)
        for i in np.where(senal)[0]:
            ax.annotate(etq[i][:16], (tau_med[i], med[i]), fontsize=6.5,
                        xytext=(4, 3), textcoords="offset points", alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"escala local media $E[\tau^2]$ (log)")
        if niv == "E":
            ax.set_ylabel(r"|media posterior de $\beta$|")
        ax.set_title(u.NIVEL_NOMBRE[niv], color=u.NIVEL_COLOR[niv])
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Perfil de encogimiento del LASSO bayesiano "
                 r"(tau$^2$ chico + |$\beta$| chico = variable descartada)")
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_19b_02_encogimiento", pdf)


def fig_lambda(sal, dir_diag, pdf):
    """Densidades posteriores de lambda2 por nivel (penalizacion global)."""
    L = sal.bloque("lambda2")                      # (n_save, 3)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, niv in enumerate("EMD"):
        x = L[:, i]
        kde = u.stats.gaussian_kde(x)
        rej = np.linspace(x.min(), np.quantile(x, 0.995), 300)
        ax.plot(rej, kde(rej), color=u.NIVEL_COLOR[niv], lw=1.8,
                label=rf"$\lambda^2_{niv}$  (mediana={np.median(x):.2f})")
        ax.fill_between(rej, kde(rej), color=u.NIVEL_COLOR[niv], alpha=0.15)
    ax.set_xlabel(r"$\lambda^2$ (penalizacion global del nivel)")
    ax.set_yticks([]); ax.set_title(
        r"Penalizacion LASSO por nivel (mayor $\lambda^2$ = mas encogimiento)")
    ax.legend()
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_19b_03_lambda2", pdf)


def fig_caterpillar_depto(df_depto, dir_diag, pdf):
    """kappa_k = raiz(kappa2_k) por departamento, ordenada, con IC en escala
    de puntaje (interpretable)."""
    g = df_depto.copy()
    g["lo"] = np.sqrt(g["kappa2_k_q025"]); g["hi"] = np.sqrt(g["kappa2_k_q975"])
    g["mid"] = np.sqrt(g["kappa2_k_mediana"])
    g = g.sort_values("mid")
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.32 * len(g))))
    ax.hlines(y, g["lo"], g["hi"], color=u.PALETA["naranja"], lw=2.2, alpha=0.85)
    ax.scatter(g["mid"], y, color=u.PALETA["rojo"], s=32, zorder=3,
               edgecolor="white", linewidth=0.5)
    med_global = np.sqrt((g["kappa2_k_media"]).mean())
    ax.axvline(med_global, color=u.PALETA["negro"], lw=1.0, ls="--",
               label=f"media global ~ {med_global:.1f} pts")
    ax.set_yticks(y); ax.set_yticklabels(g["departamento"], fontsize=8)
    ax.set_xlabel(r"desviacion residual por depto  $\kappa_k=\sqrt{\kappa^2_k}$ (puntos)")
    ax.set_title("Heterogeneidad residual entre departamentos")
    ax.legend(fontsize=8)
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_19b_04_kappa_depto", pdf)


def fig_kappa_muni(df_muni, dir_diag, pdf):
    """kappa2_jk: boxplot por depto + dispersion mediana vs n_jk (encogimiento
    de celdas pequenas hacia la varianza departamental)."""
    if df_muni is None or "departamento" not in df_muni:
        return
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6),
                            gridspec_kw={"width_ratios": [1.3, 1]})

    # (a) boxplot de la mediana de kappa2_jk por departamento
    grupos = [g["kappa2_jk_mediana"].values for _, g in
              df_muni.groupby("departamento")]
    labs = [k for k, _ in df_muni.groupby("departamento")]
    try:
        bp = axs[0].boxplot(grupos, tick_labels=labs, patch_artist=True,
                            showfliers=True)
    except TypeError:                       # matplotlib < 3.9
        bp = axs[0].boxplot(grupos, labels=labs, patch_artist=True,
                            showfliers=True)
    for caja in bp["boxes"]:
        caja.set(facecolor=u.PALETA["celeste"], alpha=0.6)
    for med in bp["medians"]:
        med.set(color=u.PALETA["rojo"], linewidth=1.4)
    axs[0].set_xticklabels(labs, rotation=60, ha="right", fontsize=7.5)
    axs[0].set_ylabel(r"$\kappa^2_{jk}$ (mediana posterior por municipio)")
    axs[0].set_title("Varianza municipal dentro de cada departamento")

    # (b) mediana kappa2_jk vs n_jk (si hay n_jk)
    if "n_jk" in df_muni:
        deptos = sorted(df_muni["departamento"].unique())
        cmap = plt.get_cmap("tab10")
        for i, d in enumerate(deptos):
            g = df_muni[df_muni["departamento"] == d]
            axs[1].scatter(g["n_jk"], g["kappa2_jk_mediana"], s=28,
                           color=cmap(i % 10), alpha=0.75, label=str(d))
        axs[1].set_xscale("log")
        axs[1].set_xlabel(r"tamano muestral de la celda $n_{jk}$ (log)")
        axs[1].set_ylabel(r"$\kappa^2_{jk}$ (mediana)")
        axs[1].set_title("Encogimiento: celdas pequenas -> hacia su depto")
        axs[1].legend(fontsize=6.5, ncols=2, title="depto", title_fontsize=7)
    else:
        axs[1].axis("off")
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_19b_05_kappa_muni", pdf)


def fig_correlacion(sal, dir_diag, pdf, umbral=0.8):
    """Heatmap de correlacion posterior entre TODOS los coeficientes."""
    partes, etq = [], []
    for niv in "EMD":
        b = f"beta_{niv}"
        if sal.disponible(b):
            partes.append(sal.bloque(b))
            etq += [f"{niv}:{n[:14]}" for n in sal.etiquetas(b)]
    if not partes:
        return
    B = np.column_stack(partes)
    C = np.corrcoef(B, rowvar=False)
    p = C.shape[0]
    fig, ax = plt.subplots(figsize=(max(7, 0.28 * p), max(6, 0.28 * p)))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(p)); ax.set_yticks(range(p))
    ax.set_xticklabels(etq, rotation=90, fontsize=5.5)
    ax.set_yticklabels(etq, fontsize=5.5)
    # resaltar pares |r|>umbral (fuera de la diagonal)
    Cx = C.copy(); np.fill_diagonal(Cx, 0.0)
    for i, j in zip(*np.where(np.abs(np.triu(Cx, 1)) > umbral)):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="black", lw=1.4))
    n_altos = int((np.abs(np.triu(Cx, 1)) > umbral).sum())
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="correlacion posterior")
    ax.set_title(f"Correlacion posterior entre coeficientes "
                 f"({n_altos} pares con |r|>{umbral})")
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_19b_06_correlacion", pdf)
    return C


# =============================================================================
# 3. ORQUESTADOR
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="19b: analisis posterior del Modelo 1")
    ap.add_argument("dirs", nargs="+", help="carpeta(s) de correr_gibbs_disco")
    ap.add_argument("--diag", default="diag_m1_py")
    ap.add_argument("--conv", default=None,
                    help="CSV de 19a para fusionar ESS/Rhat en la tabla de coefs")
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    u.estilo()
    dir_diag = Path(args.diag); dir_diag.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("ANALISIS POSTERIOR - Modelo 1 (19b, Python)")
    print("=" * 72)
    sal = u.SalidaPool(args.dirs)
    if sal.m > 1:
        print(f"Posterior conjunta de {sal.m} cadenas "
              f"({sal.n_min} draws c/u -> {sal.n_save} en total)")
    print(f"Muestras: {sal.n_save} | N={sal.meta.get('N','?')}, "
          f"K={sal.meta.get('K','?')}, n_cells={sal.meta.get('n_cells','?')} | "
          f"p_E={sal.meta.get('p_E',0)}, p_M={sal.meta.get('p_M',0)}, "
          f"p_D={sal.meta.get('p_D',0)}")

    # -- inferir CSV de convergencia por defecto --
    conv = args.conv
    if conv is None:
        cand = dir_diag / "resumen_convergencia_M1.csv"
        conv = str(cand) if cand.exists() else None

    # -- tablas --
    print("\n== Tablas ==")
    tab = tabla_coeficientes(sal, conv)
    tab.to_csv(dir_diag / "resumen_coeficientes_M1.csv", index=False,
               encoding="utf-8")
    n_sig = int((tab["senal_ic95"] == "*").sum())
    print(f"  resumen_coeficientes_M1.csv: {len(tab)} coeficientes, "
          f"{n_sig} con IC95% != 0")
    for niv in "EMD":
        g = tab[tab["nivel"] == niv]
        if len(g):
            ns = int((g["senal_ic95"] == "*").sum())
            print(f"    nivel {niv}: {ns}/{len(g)} con senal")

    df_depto, df_muni = tablas_kappa(sal)
    df_depto.to_csv(dir_diag / "kappa2_k_por_depto.csv", index=False,
                    encoding="utf-8")
    print(f"  kappa2_k_por_depto.csv: {len(df_depto)} departamentos "
          f"(kappa_k global ~ {np.sqrt(df_depto['kappa2_k_media'].mean()):.1f} pts)")
    if df_muni is not None:
        df_muni.to_csv(dir_diag / "kappa2_jk_por_muni.csv", index=False,
                       encoding="utf-8")
        print(f"  kappa2_jk_por_muni.csv: {len(df_muni)} municipios")

    # -- figuras --
    print("\n== Figuras ==")
    fig_forest_por_nivel(sal, tab, dir_diag, args.pdf)
    fig_encogimiento(sal, dir_diag, args.pdf)
    fig_lambda(sal, dir_diag, args.pdf)
    fig_caterpillar_depto(df_depto, dir_diag, args.pdf)
    fig_kappa_muni(df_muni, dir_diag, args.pdf)
    fig_correlacion(sal, dir_diag, args.pdf)

    # -- top coeficientes en consola --
    print("\n== Top 8 coeficientes por |mediana| ==")
    top = tab.reindex(tab["media"].abs().sort_values(ascending=False).index).head(8)
    for _, r in top.iterrows():
        print(f"    [{r['nivel']}] {r['parametro'][:32]:<32} "
              f"med={r['mediana']:+.3f}  IC95=[{r['q025']:+.3f},{r['q975']:+.3f}] "
              f"{r['senal_ic95']}")
    print("\n" + "=" * 72)
    print(f"19b COMPLETADO. Resultados en {dir_diag}/")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
