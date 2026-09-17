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

_AQUI = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("util_m1", _AQUI / "19_0_utilidades_M1.py")
u = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(u)
plt = u.plt


# =============================================================================
# 1. NUCLEO DEL PPC
# =============================================================================
ESTADISTICOS = ["media", "sd", "asimetria", "curtosis", "min", "max",
                "q05", "q25", "q50", "q75", "q95"]


def _estadisticos(y):
    return np.array([
        y.mean(), y.std(ddof=1), stats.skew(y), stats.kurtosis(y),  # curtosis EXCESO
        y.min(), y.max(),
        *np.quantile(y, [0.05, 0.25, 0.50, 0.75, 0.95])])


def correr_ppc(sal: "u.SalidaM1", n_draws=400, seed=0):
    """Recorre S_ppc draws, genera y_rep y acumula todos los diagnosticos sin
    materializar la matriz y_rep completa."""
    datos = sal.datos_ppc()
    if datos is None:
        raise RuntimeError(
            "No hay insumos del PPC. Re-exportar con la Etapa 18 usando datos=: "
            "exportar_para_python_m1(dir, datos=datos).")
    y = datos["y"]; N = y.size
    row_cell, row_k = datos["row_cell"], datos["row_k"]
    XE, Z, W = datos["X_E"], datos["Z"], datos["W"]

    S = sal.n_save
    idx = np.linspace(0, S - 1, min(n_draws, S)).astype(int)
    S_ppc = len(idx)
    rng = np.random.default_rng(seed)

    mu = sal.bloque("mu")
    bE = sal.bloque("beta_E") if sal.disponible("beta_E") else None
    bM = sal.bloque("beta_M") if sal.disponible("beta_M") else None
    bD = sal.bloque("beta_D") if sal.disponible("beta_D") else None
    k2 = sal.bloque("kappa2_jk")                   # (S, n_cells)

    # observados
    T_obs = _estadisticos(y)
    dep = sal.nombres.get("deptos")
    K = W.shape[0]
    media_obs_dep = np.array([y[row_k == k].mean() if (row_k == k).any() else np.nan
                              for k in range(K)])

    # acumuladores
    T_rep = np.empty((S_ppc, len(ESTADISTICOS)))
    pit = np.zeros(N)                              # P(y_rep <= y_obs) promediado
    dentro50 = np.zeros(N); dentro80 = np.zeros(N); dentro95 = np.zeros(N)
    media_rep_dep = np.zeros((S_ppc, K))
    r2 = np.empty(S_ppc); rmse = np.empty(S_ppc); mae = np.empty(S_ppc)
    z95 = stats.norm.ppf([0.025, 0.975]); z80 = stats.norm.ppf([0.10, 0.90])
    z50 = stats.norm.ppf([0.25, 0.75])
    # rejilla comun para densidades (banda de replicadas)
    lo, hi = np.quantile(y, [0.001, 0.999])
    rej = np.linspace(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo), 200)
    dens_rep = np.empty((S_ppc, rej.size))

    for t, s in enumerate(idx):
        zeta = mu[s] + (XE @ bE[s] if bE is not None else 0.0)
        if bM is not None:
            zeta = zeta + Z[row_cell] @ bM[s]
        if bD is not None:
            zeta = zeta + W[row_k] @ bD[s]
        sd = np.sqrt(k2[s][row_cell])
        y_rep = rng.normal(zeta, sd)

        T_rep[t] = _estadisticos(y_rep)
        # PIT bajo la predictiva de este draw (Normal): Phi((y-zeta)/sd)
        pit += stats.norm.cdf((y - zeta) / sd)
        # cobertura (intervalos predictivos de este draw)
        dentro50 += ((y >= zeta + z50[0] * sd) & (y <= zeta + z50[1] * sd))
        dentro80 += ((y >= zeta + z80[0] * sd) & (y <= zeta + z80[1] * sd))
        dentro95 += ((y >= zeta + z95[0] * sd) & (y <= zeta + z95[1] * sd))
        # medias por depto (replicadas)
        for k in range(K):
            sel = row_k == k
            if sel.any():
                media_rep_dep[t, k] = y_rep[sel].mean()
        # R2 bayesiano (Gelman et al. 2019): var(zeta)/(var(zeta)+var residual)
        var_fit = np.var(zeta, ddof=1)
        var_res = np.mean(k2[s][row_cell])
        r2[t] = var_fit / (var_fit + var_res)
        resid = y - zeta
        rmse[t] = np.sqrt(np.mean(resid ** 2)); mae[t] = np.mean(np.abs(resid))
        dens_rep[t] = stats.gaussian_kde(y_rep)(rej)

    pit /= S_ppc
    cob = {"50": dentro50 / S_ppc, "80": dentro80 / S_ppc, "95": dentro95 / S_ppc}

    return dict(y=y, N=N, S_ppc=S_ppc, row_k=row_k, K=K, dep=dep,
                T_obs=T_obs, T_rep=T_rep, pit=pit, cobertura=cob,
                media_obs_dep=media_obs_dep, media_rep_dep=media_rep_dep,
                r2=r2, rmse=rmse, mae=mae, rej=rej, dens_rep=dens_rep,
                zeta_ultimo=zeta)


# =============================================================================
# 2. RESUMENES
# =============================================================================
def tabla_estadisticos(P):
    p_B = np.array([(P["T_rep"][:, j] >= P["T_obs"][j]).mean()
                    for j in range(len(ESTADISTICOS))])
    return pd.DataFrame({
        "estadistico": ESTADISTICOS,
        "T_obs": P["T_obs"],
        "T_rep_medio": P["T_rep"].mean(axis=0),
        "T_rep_q025": np.quantile(P["T_rep"], 0.025, axis=0),
        "T_rep_q975": np.quantile(P["T_rep"], 0.975, axis=0),
        "p_bayes": p_B,
        "bandera": np.where((p_B < 0.025) | (p_B > 0.975), "!!",
                            np.where((p_B < 0.05) | (p_B > 0.95), "!", "")),
    })


def tabla_cobertura(P):
    filas = [{"nivel": "GLOBAL", "n_obs": P["N"],
              "cob_50": P["cobertura"]["50"].mean(),
              "cob_80": P["cobertura"]["80"].mean(),
              "cob_95": P["cobertura"]["95"].mean()}]
    if P["dep"] is not None:
        for k in range(P["K"]):
            sel = P["row_k"] == k
            if sel.any():
                filas.append({"nivel": P["dep"]["depto_id"].iloc[k],
                              "n_obs": int(sel.sum()),
                              "cob_50": P["cobertura"]["50"][sel].mean(),
                              "cob_80": P["cobertura"]["80"][sel].mean(),
                              "cob_95": P["cobertura"]["95"][sel].mean()})
    return pd.DataFrame(filas)


# =============================================================================
# 3. FIGURAS
# =============================================================================
def fig_densidad_overlay(P, dir_diag, pdf):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    lo = np.quantile(P["dens_rep"], 0.025, axis=0)
    hi = np.quantile(P["dens_rep"], 0.975, axis=0)
    med = P["dens_rep"].mean(axis=0)
    ax.fill_between(P["rej"], lo, hi, color=u.PALETA["celeste"], alpha=0.35,
                    label="y_rep (banda 95%)")
    ax.plot(P["rej"], med, color=u.PALETA["azul"], lw=1.2, label="y_rep (media)")
    kde_obs = stats.gaussian_kde(P["y"])(P["rej"])
    ax.plot(P["rej"], kde_obs, color=u.PALETA["rojo"], lw=2.0, label="y observado")
    ax.set_xlabel("puntaje global"); ax.set_yticks([])
    ax.set_title("Densidad observada vs. replicada por el Modelo 1")
    ax.legend()
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_19d_01_densidad_overlay", pdf)


def fig_pit(P, dir_diag, pdf):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    n_bins = 20
    axs[0].hist(P["pit"], bins=n_bins, density=True, color=u.PALETA["verde"],
                alpha=0.8, edgecolor="white")
    axs[0].axhline(1.0, color=u.PALETA["negro"], lw=1.1, ls="--",
                   label="uniforme (ideal)")
    esp = P["N"] / n_bins
    banda = 1.96 * np.sqrt(esp * (1 - 1 / n_bins)) / esp
    axs[0].axhspan(1 - banda, 1 + banda, color=u.PALETA["gris"], alpha=0.2,
                   label="banda 95%")
    axs[0].set_xlabel("PIT"); axs[0].set_ylabel("densidad")
    axs[0].set_title("Histograma PIT (U-forma = colas mal modeladas)")
    axs[0].legend(fontsize=8)

    # ECDF del PIT vs diagonal
    xs = np.sort(P["pit"]); ecdf = np.arange(1, P["N"] + 1) / P["N"]
    axs[1].plot(xs, ecdf, color=u.PALETA["azul"], lw=1.5)
    axs[1].plot([0, 1], [0, 1], color=u.PALETA["negro"], lw=1.0, ls="--")
    axs[1].set_xlabel("PIT"); axs[1].set_ylabel("ECDF")
    axs[1].set_title("ECDF del PIT vs. uniforme")
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_19d_02_pit", pdf)


def fig_qq(P, dir_diag, pdf):
    """QQ de Phi^{-1}(PIT) vs Normal(0,1): la prueba directa de normalidad."""
    z = stats.norm.ppf(np.clip(P["pit"], 1e-6, 1 - 1e-6))
    z = np.sort(z[np.isfinite(z)])
    n = z.size
    teor = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(teor, z, s=6, alpha=0.4, color=u.PALETA["azul"])
    lim = [min(teor[0], z[0]) - 0.3, max(teor[-1], z[-1]) + 0.3]
    ax.plot(lim, lim, color=u.PALETA["rojo"], lw=1.4, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("cuantiles teoricos Normal(0,1)")
    ax.set_ylabel(r"cuantiles de $\Phi^{-1}$(PIT)")
    curt = stats.kurtosis(z)
    ax.set_title(f"QQ de normalidad de los residuos\n"
                 f"(curtosis en exceso = {curt:+.2f}; "
                 f"{'colas pesadas -> t-Student motivada' if curt > 0.3 else 'compatible con Normal'})")
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_19d_03_qq_normalidad", pdf)


def fig_estadisticos(P, tab, dir_diag, pdf):
    """Histograma de T(y_rep) con T(y_obs) marcado, para los estadisticos clave."""
    claves = ["media", "sd", "asimetria", "curtosis", "min", "max"]
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, est in zip(axs.ravel(), claves):
        j = ESTADISTICOS.index(est)
        ax.hist(P["T_rep"][:, j], bins=30, color=u.PALETA["celeste"], alpha=0.85)
        ax.axvline(P["T_obs"][j], color=u.PALETA["rojo"], lw=2.0,
                   label="observado")
        pB = tab.loc[tab["estadistico"] == est, "p_bayes"].squeeze()
        band = tab.loc[tab["estadistico"] == est, "bandera"].squeeze()
        ax.set_title(f"{est}  (p_B={pB:.3f}) {band}",
                     color=u.PALETA["rojo"] if band else u.PALETA["negro"])
        ax.legend(fontsize=8)
    fig.suptitle("Estadisticos de test: replicado (barras) vs. observado (linea)\n"
                 "curtosis/min/max lejos del centro => la Normal no captura las colas")
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_19d_04_estadisticos", pdf)


def fig_cobertura(cob_df, dir_diag, pdf):
    dep = cob_df[cob_df["nivel"] != "GLOBAL"]
    if len(dep) == 0:
        dep = cob_df
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(dep))
    for col, nom, c, obj in [("cob_50", "50%", u.PALETA["verde"], 0.50),
                             ("cob_80", "80%", u.PALETA["azul"], 0.80),
                             ("cob_95", "95%", u.PALETA["naranja"], 0.95)]:
        ax.scatter(x, dep[col], s=30, color=c, label=f"IP {nom}", zorder=3)
        ax.axhline(obj, color=c, lw=1.0, ls="--", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(dep["nivel"], rotation=60, ha="right",
                                         fontsize=7.5)
    ax.set_ylabel("cobertura empirica")
    ax.set_title("Cobertura de intervalos predictivos por departamento\n"
                 "(puntos deben caer sobre las lineas nominales)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_19d_05_cobertura", pdf)


def fig_medias_depto(P, dir_diag, pdf):
    if P["dep"] is None:
        return
    med_rep = P["media_rep_dep"].mean(axis=0)
    lo = np.quantile(P["media_rep_dep"], 0.025, axis=0)
    hi = np.quantile(P["media_rep_dep"], 0.975, axis=0)
    obs = P["media_obs_dep"]
    orden = np.argsort(obs)
    y = np.arange(P["K"])
    fig, ax = plt.subplots(figsize=(8.5, max(3.4, 0.32 * P["K"])))
    ax.hlines(y, lo[orden], hi[orden], color=u.PALETA["celeste"], lw=2.2,
              alpha=0.8, label="y_rep (IC95%)")
    ax.scatter(med_rep[orden], y, color=u.PALETA["azul"], s=26,
               label="y_rep (media)", zorder=3)
    ax.scatter(obs[orden], y, color=u.PALETA["rojo"], s=42, marker="D",
               label="y observado", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(P["dep"]["depto_id"].iloc[orden].values, fontsize=8)
    ax.set_xlabel("puntaje medio")
    ax.set_title("Media observada vs. replicada por departamento")
    ax.legend(fontsize=8)
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_19d_06_medias_depto", pdf)


def fig_dispersion(P, dir_diag, pdf):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4),
                            gridspec_kw={"width_ratios": [1.3, 1]})
    hb = axs[0].hexbin(P["zeta_ultimo"], P["y"], gridsize=40, cmap="Blues",
                       mincnt=1)
    lim = [min(P["zeta_ultimo"].min(), P["y"].min()),
           max(P["zeta_ultimo"].max(), P["y"].max())]
    axs[0].plot(lim, lim, color=u.PALETA["rojo"], lw=1.4, ls="--")
    axs[0].set_xlabel(r"valor ajustado $\hat\zeta$ (ultimo draw)")
    axs[0].set_ylabel("y observado")
    axs[0].set_title("Observado vs. ajustado")
    fig.colorbar(hb, ax=axs[0], label="n obs")

    axs[1].hist(P["r2"], bins=30, color=u.PALETA["verde"], alpha=0.85)
    axs[1].axvline(P["r2"].mean(), color=u.PALETA["rojo"], lw=1.6,
                   label=f"media={P['r2'].mean():.3f}")
    axs[1].set_xlabel(r"$R^2$ bayesiano por draw"); axs[1].set_yticks([])
    axs[1].set_title(r"$R^2$ bayesiano (Gelman et al. 2019)")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_19d_07_dispersion", pdf)


# =============================================================================
# 4. ORQUESTADOR
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="19d: chequeo predictivo posterior")
    ap.add_argument("dirs", nargs="+", help="carpeta(s) de correr_gibbs_disco")
    ap.add_argument("--diag", default="diag_m1_py")
    ap.add_argument("--draws", type=int, default=400,
                    help="numero de draws del posterior a usar (equiespaciados)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    u.estilo()
    dir_diag = Path(args.diag); dir_diag.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("CHEQUEO PREDICTIVO POSTERIOR - Modelo 1 (19d)")
    print("=" * 72)
    sal = u.SalidaPool(args.dirs)
    if sal.m > 1:
        print(f"Posterior conjunta de {sal.m} cadenas -> {sal.n_save} draws")
    if sal.datos_ppc() is None:
        print("ERROR: faltan los insumos del PPC (y, X_E, Z, W, row_cell, row_k).")
        print("       Re-exportar con la Etapa 18 pasando datos=:")
        print("       exportar_para_python_m1(dir_salida, datos = datos)")
        return 1

    print(f"Generando y_rep sobre {min(args.draws, sal.n_save)} draws "
          f"(de {sal.n_save} guardados)...")
    P = correr_ppc(sal, n_draws=args.draws, seed=args.seed)

    # -- tablas --
    tab = tabla_estadisticos(P)
    tab.to_csv(dir_diag / "ppc_estadisticos_M1.csv", index=False, encoding="utf-8")
    cob = tabla_cobertura(P)
    cob.to_csv(dir_diag / "ppc_cobertura_M1.csv", index=False, encoding="utf-8")

    print("\n== Estadisticos de test (p_bayes cerca de 0 o 1 = mal ajuste) ==")
    for _, r in tab.iterrows():
        print(f"    {r['estadistico']:<10} obs={r['T_obs']:>10.3f}  "
              f"rep={r['T_rep_medio']:>10.3f}  p_B={r['p_bayes']:.3f} {r['bandera']}")

    g = cob[cob["nivel"] == "GLOBAL"].iloc[0]
    print("\n== Cobertura global de intervalos predictivos ==")
    print(f"    IP 50%: {g['cob_50']*100:5.1f}% (nominal 50)  |  "
          f"IP 80%: {g['cob_80']*100:5.1f}% (nominal 80)  |  "
          f"IP 95%: {g['cob_95']*100:5.1f}% (nominal 95)")

    z = stats.norm.ppf(np.clip(P["pit"], 1e-6, 1 - 1e-6))
    curt_pit = stats.kurtosis(z[np.isfinite(z)])
    print(f"\n== Normalidad de los residuos (via PIT) ==")
    print(f"    curtosis en exceso de Phi^-1(PIT) = {curt_pit:+.3f}")
    print(f"    R2 bayesiano medio = {P['r2'].mean():.3f} | "
          f"RMSE medio = {P['rmse'].mean():.2f} | MAE medio = {P['mae'].mean():.2f}")

    # -- figuras --
    print("\n== Figuras ==")
    fig_densidad_overlay(P, dir_diag, args.pdf)
    fig_pit(P, dir_diag, args.pdf)
    fig_qq(P, dir_diag, args.pdf)
    fig_estadisticos(P, tab, dir_diag, args.pdf)
    fig_cobertura(cob, dir_diag, args.pdf)
    fig_medias_depto(P, dir_diag, args.pdf)
    fig_dispersion(P, dir_diag, args.pdf)

    # -- veredicto --
    n_flag = int((tab["bandera"] == "!!").sum())
    cob95_ok = abs(g["cob_95"] - 0.95) < 0.03
    colas_pesadas = curt_pit > 0.3 or \
        tab.loc[tab["estadistico"] == "curtosis", "bandera"].squeeze() == "!!"
    print("\n" + "=" * 72)
    print("VEREDICTO DEL CHEQUEO PREDICTIVO (ajuste absoluto)")
    print("=" * 72)
    print(f"  Estadisticos con p_B extremo (<0.025 o >0.975): {n_flag}/{len(tab)}")
    txt_cob = "(OK)" if cob95_ok else "(desviada del 95% nominal)"
    txt_colas = ("PESADAS -> el Modelo 2 (t-Student) esta motivado por los datos"
                 if colas_pesadas else "compatibles con Normal")
    print(f"  Cobertura del IP 95%: {g['cob_95']*100:.1f}%  {txt_cob}")
    print(f"  Colas de los residuos: {txt_colas}")
    print("-" * 72)
    if n_flag == 0 and cob95_ok and not colas_pesadas:
        print("  CONCLUSION: el Modelo 1 (Normal) describe bien los datos; no hay")
        print("  evidencia fuerte para preferir la t-Student por ajuste absoluto.")
    else:
        print("  CONCLUSION: el Modelo 1 muestra desajustes (arriba). Documentar")
        print("  cuales y usar 19c para verificar si el Modelo 2 los corrige.")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
