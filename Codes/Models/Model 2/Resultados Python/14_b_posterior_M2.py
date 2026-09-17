# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# @  MODELO 2 - t-STUDENT JERARQUICO ESTATICO CON LASSO BAYESIANO
# @  ANALISIS POSTERIOR Y EFECTOS JERARQUICOS 
# _____________________________________________________________________________
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_AQUI = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("util_m2", _AQUI / "14_0_utilidades_M2.py")
u = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(u)
plt = u.plt


# =============================================================================
# 1. TABLAS
# =============================================================================
def tabla_coeficientes(sal: "u.SalidaPool", conv_csv=None):
    """Resumen por coeficiente (media, sd, IC95, prob. de direccion) por
    nivel. Si se pasa el CSV de 14a, fusiona ESS bulk/tail y Rhat."""
    filas = []
    for niv in "EMD":
        p_L = {"E": sal.p_E, "M": sal.p_M, "D": sal.p_D}[niv]
        if p_L == 0:
            continue
        X = sal.beta_nivel(niv)                    # (n_save, p_L)
        etq = sal.etiquetas_nivel(niv)
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


def tablas_kappa(sal: "u.SalidaPool"):
    """Varianzas por departamento (kappa2_k) y por celda municipio-depto
    (kappa2_jk), con IC y desviacion en escala de puntaje."""
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
# 2. ROBUSTEZ VIA xi (Rao-Blackwell)
# =============================================================================
def calcular_robustez_xi(sal: "u.SalidaPool", max_draws=400):
    """E[xi_i | y] Rao-Blackwellizado + residuo bajo la media posterior.

    Devuelve dict con e_xi (N,), resid (N,) bajo E[beta], rho de Spearman,
    correlacion con xi.bin (si existe) y el numero de draws usados; o None si
    el export de la Etapa 13 no esta disponible."""
    datos = sal.datos_ppc()
    if datos is None:
        return None
    y, X, row_cell = datos["y"], datos["X_star"], datos["row_cell"]
    S = sal.n_save
    idx = np.unique(np.linspace(0, S - 1, min(max_draws, S)).astype(int))
    B = sal.bloque("beta_star")[idx]               # (s, pp1)
    K2 = sal.bloque("kappa2_jk")[idx]              # (s, n_cells)
    NU = np.atleast_1d(sal.bloque("nu"))[idx]      # (s,)

    e_xi = np.zeros(len(y))
    for s in range(len(idx)):
        eps = y - X @ B[s]
        e_xi += u.e_xi_condicional(eps, K2[s][row_cell], NU[s])
    e_xi /= len(idx)

    resid = y - X @ B.mean(axis=0)
    rho = u.stats.spearmanr(np.abs(resid), e_xi).statistic

    corr_xi_bin = None
    if sal.disponible("xi"):
        mm = sal.base.bloque("xi", memmap=True)
        xi_emp = u.media_columnas_memmap(mm)
        corr_xi_bin = float(np.corrcoef(e_xi, xi_emp)[0, 1])
    return {"e_xi": e_xi, "resid": resid, "rho_spearman": float(rho),
            "corr_xi_bin": corr_xi_bin, "n_draws": len(idx),
            "row_cell": row_cell, "row_k": datos["row_k"], "y": y}


def tabla_atipicos(sal, rob, n_top=200):
    """Top n_top observaciones mas descontadas (E[xi] menor)."""
    orden = np.argsort(rob["e_xi"])[:n_top]
    cel = sal.nombres.get("celdas")
    dep = sal.nombres.get("deptos")
    df = pd.DataFrame({
        "obs": orden + 1,
        "y": rob["y"][orden],
        "residuo": rob["resid"][orden],
        "E_xi": rob["e_xi"][orden],
    })
    if cel is not None:
        df["departamento"] = cel["depto_id"].values[rob["row_cell"][orden]]
        df["municipio"] = cel["muni_id"].values[rob["row_cell"][orden]]
    elif dep is not None:
        df["departamento"] = dep["depto_id"].values[rob["row_k"][orden]]
    return df


# =============================================================================
# 3. FIGURAS
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

    fig.suptitle("Modelo 2 - coeficientes LASSO por nivel jerarquico "
                 "(color = senal; gris = encogido)", y=0.995)
    u.guardar_fig(fig, dir_diag, "fig_14b_01_forest_por_nivel", pdf)


def fig_encogimiento(sal, dir_diag, pdf):
    """Perfil de encogimiento: |media beta| vs escala local media tau2, por
    nivel, en escala log-x. Coeficientes senal (IC95 != 0) resaltados."""
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, niv in zip(axs, "EMD"):
        t = f"tau2_{niv}"
        p_L = {"E": sal.p_E, "M": sal.p_M, "D": sal.p_D}[niv]
        if p_L == 0 or not sal.disponible(t):
            ax.axis("off"); continue
        B = sal.beta_nivel(niv); T = sal.bloque(t)
        med = np.abs(B.mean(axis=0))
        tau_med = T.mean(axis=0)
        q = np.quantile(B, [0.025, 0.975], axis=0)
        senal = (q[0] > 0) | (q[1] < 0)
        ax.scatter(tau_med[~senal], med[~senal], s=45, color=u.PALETA["gris"],
                   alpha=0.7, label="encogido", edgecolor="white", linewidth=0.5)
        ax.scatter(tau_med[senal], med[senal], s=55, color=u.NIVEL_COLOR[niv],
                   alpha=0.9, label="senal (IC95 != 0)", edgecolor="white",
                   linewidth=0.5)
        etq = sal.etiquetas_nivel(niv)
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
    u.guardar_fig(fig, dir_diag, "fig_14b_02_encogimiento", pdf)


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
    u.guardar_fig(fig, dir_diag, "fig_14b_03_lambda2", pdf)


def fig_caterpillar_depto(df_depto, dir_diag, pdf):
    """kappa_k = raiz(kappa2_k) por departamento, ordenada, con IC en escala
    de puntaje (interpretable). En M2 kappa2 es la ESCALA de la t: la
    desviacion residual real es raiz(kappa2 * nu/(nu-2))."""
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
    ax.set_xlabel(r"escala residual por depto  $\kappa_k=\sqrt{\kappa^2_k}$ (puntos)")
    ax.set_title("Heterogeneidad residual entre departamentos (escala de la t)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_14b_04_kappa_depto", pdf)


def fig_kappa_muni(df_muni, dir_diag, pdf):
    """kappa2_jk: boxplot por depto + dispersion mediana vs n_jk (encogimiento
    de celdas pequenas hacia la varianza departamental)."""
    if df_muni is None or "departamento" not in df_muni:
        return
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6),
                            gridspec_kw={"width_ratios": [1.3, 1]})

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
    axs[0].set_title("Escala municipal dentro de cada departamento")

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
    u.guardar_fig(fig, dir_diag, "fig_14b_05_kappa_muni", pdf)


def fig_correlacion(sal, dir_diag, pdf, umbral=0.8):
    """Heatmap de correlacion posterior entre TODOS los coeficientes."""
    partes, etq = [], []
    for niv in "EMD":
        p_L = {"E": sal.p_E, "M": sal.p_M, "D": sal.p_D}[niv]
        if p_L:
            partes.append(sal.beta_nivel(niv))
            etq += [f"{niv}:{n[:14]}" for n in sal.etiquetas_nivel(niv)]
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
    Cx = C.copy(); np.fill_diagonal(Cx, 0.0)
    for i, j in zip(*np.where(np.abs(np.triu(Cx, 1)) > umbral)):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="black", lw=1.4))
    n_altos = int((np.abs(np.triu(Cx, 1)) > umbral).sum())
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="correlacion posterior")
    ax.set_title(f"Correlacion posterior entre coeficientes "
                 f"({n_altos} pares con |r|>{umbral})")
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_14b_06_correlacion", pdf)
    return C


def fig_robustez_xi(rob, dir_diag, pdf):
    """(a) |residuo| vs E[xi] con la firma de la robustez (monotonia
    decreciente); (b) histograma de E[xi] con la cota de descuento."""
    if rob is None:
        return
    e_xi, resid = rob["e_xi"], np.abs(rob["resid"])
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.0),
                            gridspec_kw={"width_ratios": [1.35, 1]})
    n = len(e_xi)
    idx = np.random.default_rng(2026).choice(n, size=min(n, 20000), replace=False)
    axs[0].scatter(resid[idx], e_xi[idx], s=6, alpha=0.35,
                   color=u.PALETA["azul"], edgecolor="none")
    axs[0].axhline(1.0, color=u.PALETA["negro"], lw=1.0, ls="--",
                   label=r"$E[\xi]=1$ (peso pleno)")
    axs[0].axhline(0.5, color=u.PALETA["rojo"], lw=1.0, ls=":",
                   label=r"$E[\xi]=0.5$ (descuento fuerte)")
    axs[0].set_xlabel("|residuo| bajo la media posterior")
    axs[0].set_ylabel(r"$E[\xi_i \mid y]$ (Rao-Blackwell)")
    axs[0].set_title(f"Descuento de atipicos: Spearman = "
                     f"{rob['rho_spearman']:+.3f}  ({rob['n_draws']} draws)")
    axs[0].legend(fontsize=8, loc="upper right")

    axs[1].hist(e_xi, bins=60, color=u.PALETA["celeste"])
    axs[1].axvline(1.0, color=u.PALETA["negro"], lw=1.0, ls="--")
    axs[1].axvline(0.5, color=u.PALETA["rojo"], lw=1.0, ls=":")
    n_05 = int((e_xi < 0.5).sum())
    axs[1].set_xlabel(r"$E[\xi_i \mid y]$")
    axs[1].set_ylabel("observaciones")
    axs[1].set_title(f"{n_05} obs. con descuento fuerte "
                     f"({100 * n_05 / n:.2f}%)")
    extra = (f" | corr con xi.bin = {rob['corr_xi_bin']:.3f}"
             if rob["corr_xi_bin"] is not None else "")
    fig.suptitle("Pesos latentes de la mezcla Normal-Gamma"
                 r" ($\xi_i$ chico = observacion descontada)" + extra)
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_14b_07_robustez_xi", pdf)


def fig_funnel(sal, dir_diag, pdf):
    """Embudo beta vs log(tau2): por nivel, el coeficiente MAS encogido
    (menor E[tau2]) y el de mayor E[tau2] como contraste."""
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0))
    algo = False
    for ax, niv in zip(axs, "EMD"):
        t = f"tau2_{niv}"
        p_L = {"E": sal.p_E, "M": sal.p_M, "D": sal.p_D}[niv]
        if p_L == 0 or not sal.disponible(t):
            ax.axis("off"); continue
        algo = True
        B = sal.beta_nivel(niv); T = sal.bloque(t)
        etq = sal.etiquetas_nivel(niv)
        orden = np.argsort(T.mean(axis=0))
        j_min, j_max = orden[0], orden[-1]
        ax.scatter(np.log(T[:, j_max]), B[:, j_max], s=7, alpha=0.35,
                   color=u.PALETA["gris"], edgecolor="none",
                   label=f"mayor E[tau2]: {etq[j_max][:14]}")
        ax.scatter(np.log(T[:, j_min]), B[:, j_min], s=7, alpha=0.45,
                   color=u.NIVEL_COLOR[niv], edgecolor="none",
                   label=f"mas encogido: {etq[j_min][:14]}")
        ax.axhline(0, color=u.PALETA["negro"], lw=0.8, ls="--", alpha=0.6)
        ax.set_xlabel(r"$\log(\tau^2)$"); ax.set_title(u.NIVEL_NOMBRE[niv],
                                                       color=u.NIVEL_COLOR[niv])
        if niv == "E":
            ax.set_ylabel(r"$\beta$ (draws)")
        ax.legend(fontsize=7.5, loc="upper left")
    if not algo:
        plt.close(fig); return
    fig.suptitle("Geometria de embudo de la jerarquia local del LASSO "
                 r"($\tau^2 \to 0$ estrecha $\beta$ hacia 0)")
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_14b_08_funnel", pdf)


# =============================================================================
# 4. ORQUESTADOR
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="14b: analisis posterior del Modelo 2")
    ap.add_argument("dirs", nargs="+", help="carpeta(s) de correr_gibbs_m2")
    ap.add_argument("--diag", default="diag_m2_py")
    ap.add_argument("--conv", default=None,
                    help="CSV de 14a para fusionar ESS/Rhat en la tabla de coefs")
    ap.add_argument("--max-draws-xi", type=int, default=400,
                    help="draws (equiespaciados) para el E[xi] Rao-Blackwell")
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args(argv)

    u.estilo()
    dir_diag = Path(args.diag); dir_diag.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("ANALISIS POSTERIOR - Modelo 2 (14b, Python)")
    print("=" * 72)
    sal = u.SalidaPool(args.dirs)
    if sal.m > 1:
        print(f"Posterior conjunta de {sal.m} cadenas "
              f"({sal.n_min} draws c/u -> {sal.n_save} en total)")
    print(f"Muestras: {sal.n_save} | N={sal.meta.get('N','?')}, "
          f"K={sal.meta.get('K','?')}, n_cells={sal.meta.get('n_cells','?')} | "
          f"p_E={sal.p_E}, p_M={sal.p_M}, p_D={sal.p_D}")

    conv = args.conv
    if conv is None:
        cand = dir_diag / "resumen_convergencia_M2.csv"
        conv = str(cand) if cand.exists() else None

    # -- tablas --
    print("\n== Tablas ==")
    tab = tabla_coeficientes(sal, conv)
    tab.to_csv(dir_diag / "resumen_coeficientes_M2.csv", index=False,
               encoding="utf-8")
    n_sig = int((tab["senal_ic95"] == "*").sum())
    print(f"  resumen_coeficientes_M2.csv: {len(tab)} coeficientes, "
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
        print(f"  kappa2_jk_por_muni.csv: {len(df_muni)} celdas municipio-depto")

    # -- robustez xi --
    print("\n== Robustez via xi (Rao-Blackwell) ==")
    rob = calcular_robustez_xi(sal, max_draws=args.max_draws_xi)
    if rob is None:
        print("  export_python/ sin datos a nivel observacion: se omite "
              "(correr la Etapa 13 con exportar_datos=TRUE).")
    else:
        print(f"  E[xi] con {rob['n_draws']} draws | "
              f"Spearman(|resid|, E[xi]) = {rob['rho_spearman']:+.3f} "
              f"{'OK (robustez activa)' if rob['rho_spearman'] < -0.3 else ''}")
        if rob["corr_xi_bin"] is not None:
            print(f"  contraste con xi.bin (media por columna): "
                  f"corr = {rob['corr_xi_bin']:.4f}")
        n_05 = int((rob["e_xi"] < 0.5).sum())
        print(f"  obs. con descuento fuerte E[xi]<0.5: {n_05} "
              f"({100 * n_05 / len(rob['e_xi']):.2f}%)")
        tabla_atipicos(sal, rob).to_csv(dir_diag / "obs_atipicas_M2.csv",
                                        index=False, encoding="utf-8")
        print(f"  obs_atipicas_M2.csv: top 200 por E[xi] ascendente")

    # -- figuras --
    print("\n== Figuras ==")
    fig_forest_por_nivel(sal, tab, dir_diag, args.pdf)
    fig_encogimiento(sal, dir_diag, args.pdf)
    fig_lambda(sal, dir_diag, args.pdf)
    fig_caterpillar_depto(df_depto, dir_diag, args.pdf)
    fig_kappa_muni(df_muni, dir_diag, args.pdf)
    fig_correlacion(sal, dir_diag, args.pdf)
    fig_robustez_xi(rob, dir_diag, args.pdf)
    fig_funnel(sal, dir_diag, args.pdf)

    # -- top coeficientes en consola --
    print("\n== Top 8 coeficientes por |mediana| ==")
    top = tab.reindex(tab["media"].abs().sort_values(ascending=False).index).head(8)
    for _, r in top.iterrows():
        print(f"    [{r['nivel']}] {r['parametro'][:32]:<32} "
              f"med={r['mediana']:+.3f}  IC95=[{r['q025']:+.3f},{r['q975']:+.3f}] "
              f"{r['senal_ic95']}")

    # -- resumen legible por 14_f --
    import json
    resumen = {
        "n_coeficientes": int(len(tab)),
        "n_senal_ic95": int((tab["senal_ic95"] == "*").sum()),
        "senal_por_nivel": {niv: [int((tab.loc[tab["nivel"] == niv,
                                               "senal_ic95"] == "*").sum()),
                                  int((tab["nivel"] == niv).sum())]
                            for niv in "EMD" if (tab["nivel"] == niv).any()},
        "top_coefs": [{"nivel": r["nivel"], "parametro": r["parametro"],
                       "mediana": float(r["mediana"]),
                       "q025": float(r["q025"]), "q975": float(r["q975"]),
                       "senal": r["senal_ic95"] == "*"}
                      for _, r in top.head(5).iterrows()],
        "kappa_k_global": float(np.sqrt(df_depto["kappa2_k_media"].mean())),
    }
    if rob is not None:
        resumen.update({
            "xi_spearman": rob["rho_spearman"],
            "xi_corr_xi_bin": rob["corr_xi_bin"],
            "xi_n_draws": rob["n_draws"],
            "xi_n_descuento_fuerte": int((rob["e_xi"] < 0.5).sum()),
            "xi_pct_descuento_fuerte":
                float(100 * (rob["e_xi"] < 0.5).mean()),
        })
    with open(dir_diag / "posterior_resumen_M2.json", "w",
              encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)
    print(f"  [json] {dir_diag}/posterior_resumen_M2.json")

    print("\n" + "=" * 72)
    print(f"14b COMPLETADO. Resultados en {dir_diag}/")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
