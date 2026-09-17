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
from scipy.stats import norm

_AQUI = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("util_m1", _AQUI / "19_0_utilidades_M1.py")
u = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(u)
plt = u.plt


# =============================================================================
# 1. RECORRIDO DE BLOQUES Y TABLA DE DIAGNOSTICOS
# =============================================================================
BLOQUES_DIAG = ["mu", "sigma2_mu", "beta_E", "beta_M", "beta_D",
                "tau2_E", "tau2_M", "tau2_D", "lambda2", "kappa2_k", "kappa2_jk"]


def tabla_diagnosticos(cad: "u.CadenasM1", incluir_kappa2_jk=True):
    """DataFrame con resumen posterior + diagnosticos para cada parametro."""
    filas = []
    for b in BLOQUES_DIAG:
        if b == "kappa2_jk" and not incluir_kappa2_jk:
            continue
        if not cad.disponible(b):
            continue
        X = u.preparar_bloque(cad, b)                 # (P, m, n)
        d = u.diagnosticos_mcmc(X)
        res = u.resumen_posterior(X)
        res.insert(0, "parametro", cad.etiquetas(b))
        res.insert(0, "bloque", b)
        for k, v in d.items():
            res[k] = v
        filas.append(res)
        print(f"  bloque {b:<10} : {X.shape[0]:4d} parametros diagnosticados")
    df = pd.concat(filas, ignore_index=True)
    df["geweke_p"] = 2 * norm.sf(np.abs(df["geweke_z"]))
    df["geweke_p_fdr"] = u.bh_fdr(df["geweke_p"].to_numpy())
    return df


# =============================================================================
# 2. FIGURAS
# =============================================================================
COL_CAD = [u.PALETA[c] for c in ("azul", "naranja", "verde", "morado",
                                 "celeste", "rojo", "amarillo", "gris")]


def _traza(ax, X, titulo, ylab=""):
    """X: (m, n). Traza por cadena + media movil de la cadena combinada."""
    m, n = X.shape
    for c in range(m):
        ax.plot(X[c], lw=0.5, alpha=0.75, color=COL_CAD[c % len(COL_CAD)],
                label=f"cadena {c+1}" if m > 1 else None)
    comb = X.mean(axis=0)
    ax.plot(np.cumsum(comb) / np.arange(1, n + 1), color=u.PALETA["negro"],
            lw=1.4, ls="--", label="media movil")
    ax.set_title(titulo); ax.set_xlabel("muestra guardada"); ax.set_ylabel(ylab)


def _densidad(ax, X, titulo):
    m, n = X.shape
    plano = X.reshape(-1)
    lo, hi = np.quantile(plano, [0.001, 0.999])
    rejilla = np.linspace(lo, hi, 300)
    for c in range(m):
        kde = u.stats.gaussian_kde(X[c])
        ax.plot(rejilla, kde(rejilla), lw=1.0, alpha=0.8,
                color=COL_CAD[c % len(COL_CAD)])
    kde = u.stats.gaussian_kde(plano)
    ax.fill_between(rejilla, kde(rejilla), color=u.PALETA["gris"], alpha=0.25)
    q = np.quantile(plano, [0.025, 0.5, 0.975])
    for x0, ls in zip(q, (":", "--", ":")):
        ax.axvline(x0, color=u.PALETA["negro"], lw=0.9, ls=ls, alpha=0.7)
    ax.set_title(titulo); ax.set_yticks([])


def fig_trazas_globales(cad, ll_tot, dir_diag, pdf):
    """Traza + densidad de los escalares estructurales y la log-verosimilitud."""
    piezas = [("mu", r"$\mu$ (intercepto global)"),
              ("sigma2_mu", r"$\sigma^2_\mu$"),]
    lam = u.preparar_bloque(cad, "lambda2")
    filas = [(u.preparar_bloque(cad, b)[0], t) for b, t in piezas]
    for i, niv in enumerate("EMD"):
        filas.append((lam[i], rf"$\lambda^2_{{{niv}}}$ (penalizacion LASSO nivel {niv})"))
    if ll_tot is not None:
        filas.append((ll_tot, "log-verosimilitud total (suma sobre obs.)"))
    nf = len(filas)
    fig, axs = plt.subplots(nf, 2, figsize=(11, 2.1 * nf),
                            gridspec_kw={"width_ratios": [2.2, 1]})
    for i, (X, t) in enumerate(filas):
        _traza(axs[i, 0], X, t)
        _densidad(axs[i, 1], X, "densidad posterior")
        if i < nf - 1:
            axs[i, 0].set_xlabel("")
    if cad.m > 1:
        axs[0, 0].legend(loc="upper right", ncols=min(cad.m + 1, 4), fontsize=8)
    fig.suptitle("Modelo 1 - trazas y densidades de los parametros estructurales",
                 y=1.001)
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_19a_01_trazas_globales", pdf)


def fig_acf(cad, df, dir_diag, pdf, lag_max=60):
    """ACF (promedio entre cadenas) de escalares + el beta de peor ESS."""
    objetivos = [("mu", 0, r"$\mu$"), ("sigma2_mu", 0, r"$\sigma^2_\mu$"),
                 ("lambda2", 0, r"$\lambda^2_E$"), ("kappa2_k", 0, r"$\kappa^2_k[1]$")]
    betas = df[df["bloque"].str.startswith("beta")]
    if len(betas):
        peor = betas.loc[betas["ess_bulk"].idxmin()]
        idx = cad.etiquetas(peor["bloque"]).index(peor["parametro"])
        objetivos.append((peor["bloque"], idx,
                          f"peor ESS: {peor['parametro'][:28]}"))
    fig, axs = plt.subplots(1, len(objetivos), figsize=(2.9 * len(objetivos), 2.9),
                            sharey=True)
    for ax, (b, j, t) in zip(np.atleast_1d(axs), objetivos):
        X = u.preparar_bloque(cad, b)[j][None, :, :]     # (1, m, n)
        ac = u._autocov_fft(X)                            # (1, m, n)
        rho = (ac / ac[:, :, :1]).mean(axis=1)[0][:lag_max + 1]
        ax.bar(np.arange(lag_max + 1), rho, width=0.9, color=u.PALETA["azul"])
        n_ef = X.shape[1] * X.shape[2]
        ax.axhline(0, color="k", lw=0.8)
        for s in (1, -1):
            ax.axhline(s * 1.96 / np.sqrt(n_ef), color=u.PALETA["rojo"],
                       lw=0.8, ls="--")
        ax.set_title(t, fontsize=9.5); ax.set_xlabel("rezago")
    np.atleast_1d(axs)[0].set_ylabel("autocorrelacion")
    fig.suptitle("Funciones de autocorrelacion (banda: +/-1.96/raiz(S))")
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_19a_02_acf", pdf)


def fig_rhat_ess(df, dir_diag, pdf):
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.4))
    rh = df["rhat"].dropna()
    axs[0].hist(rh, bins=40, color=u.PALETA["azul"])
    axs[0].axvline(u.UMBRAL_RHAT, color=u.PALETA["rojo"], ls="--", lw=1.2,
                   label=f"umbral {u.UMBRAL_RHAT}")
    axs[0].set_title(r"$\widehat{R}$ (rango-normalizado y plegado)")
    axs[0].set_xlabel(r"$\widehat{R}$"); axs[0].legend()

    eb = df["ess_bulk"].dropna()
    axs[1].hist(eb, bins=40, color=u.PALETA["verde"])
    axs[1].axvline(u.UMBRAL_ESS, color=u.PALETA["rojo"], ls="--", lw=1.2,
                   label="umbral 400")
    axs[1].set_title("ESS bulk"); axs[1].set_xlabel("Muestras Efectivas")
    axs[1].legend()

    for b, g in df.groupby("bloque"):
        base = b.split("_")[0]
        color = {"beta": u.PALETA["azul"], "tau2": u.PALETA["celeste"],
                 "kappa2": u.PALETA["naranja"], "lambda2": u.PALETA["verde"],
                 "mu": u.PALETA["negro"], "sigma2": u.PALETA["morado"]}.get(
                     base, u.PALETA["gris"])
        axs[2].scatter(g["ess_bulk"], g["ess_tail"], s=12, alpha=0.6,
                       color=color, label=b)
    lim = max(df["ess_bulk"].max(), df["ess_tail"].max()) * 1.05
    axs[2].plot([0, lim], [0, lim], color=u.PALETA["gris"], lw=0.8, ls=":")
    axs[2].axvline(u.UMBRAL_ESS, color=u.PALETA["rojo"], ls="--", lw=0.9)
    axs[2].axhline(u.UMBRAL_ESS, color=u.PALETA["rojo"], ls="--", lw=0.9)
    axs[2].set_xlabel("ESS bulk"); axs[2].set_ylabel("ESS tail")
    axs[2].set_title("bulk vs tail por bloque")
    axs[2].legend(fontsize=7, ncols=2, markerscale=1.2)
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_19a_03_rhat_ess", pdf)


def fig_peores_trazas(cad, df, dir_diag, pdf, n_peores=6):
    """El test de estres: trazas de los parametros con peor Rhat / menor ESS."""
    orden = df.sort_values(["rhat", "ess_bulk"], ascending=[False, True]).head(n_peores)
    fig, axs = plt.subplots(int(np.ceil(n_peores / 2)), 2,
                            figsize=(11, 1.9 * np.ceil(n_peores / 2)))
    axs = np.atleast_1d(axs).ravel()
    for ax, (_, fila) in zip(axs, orden.iterrows()):
        idx = cad.etiquetas(fila["bloque"]).index(fila["parametro"])
        X = u.preparar_bloque(cad, fila["bloque"])[idx]
        _traza(ax, X, f"{fila['parametro'][:34]}  "
                      f"(Rhat={fila['rhat']:.3f}, ESSb={fila['ess_bulk']:.0f})")
        ax.set_xlabel("")
    for ax in axs[len(orden):]:
        ax.axis("off")
    fig.suptitle("Parametros con peor diagnostico")
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_19a_04_peores_trazas", pdf)


def fig_rangos(cad, df, dir_diag, pdf, n_bins=20):
    """Graficos de rangos (Vehtari et al. 2021): cada mitad de cadena deberia
    tener rangos ~uniformes sobre el conjunto combinado."""
    objetivos = [("mu", 0, r"$\mu$"), ("sigma2_mu", 0, r"$\sigma^2_\mu$"),
                 ("lambda2", 0, r"$\lambda^2_E$")]
    betas = df[df["bloque"].str.startswith("beta")]
    if len(betas):
        peor = betas.loc[betas["rhat"].idxmax()]
        idx = cad.etiquetas(peor["bloque"]).index(peor["parametro"])
        objetivos.append((peor["bloque"], idx, f"peor Rhat: {peor['parametro'][:24]}"))
    fig, axs = plt.subplots(1, len(objetivos), figsize=(3.0 * len(objetivos), 2.9),
                            sharey=True)
    for ax, (b, j, t) in zip(np.atleast_1d(axs), objetivos):
        X = u.preparar_bloque(cad, b)[j][None]            # (1, m, n)
        Xs = u._dividir(X)[0]                             # (2m, n2)
        M, n2 = Xs.shape
        rangos = u.stats.rankdata(Xs.reshape(-1)).reshape(M, n2)
        esperado = n2 / n_bins
        banda = 1.96 * np.sqrt(esperado * (1 - 1 / n_bins))
        bordes = np.linspace(0, M * n2, n_bins + 1)
        for c in range(M):
            h, _ = np.histogram(rangos[c], bins=bordes)
            ax.step(np.arange(n_bins), h, where="mid", lw=1.0,
                    color=COL_CAD[c % len(COL_CAD)], alpha=0.85)
        ax.axhline(esperado, color=u.PALETA["negro"], lw=1.0, ls="--")
        ax.axhspan(esperado - banda, esperado + banda,
                   color=u.PALETA["gris"], alpha=0.2)
        ax.set_title(t, fontsize=9.5); ax.set_xlabel("bin de rango")
    np.atleast_1d(axs)[0].set_ylabel("frecuencia")
    fig.suptitle("Graficos de rangos por mitad de cadena (uniforme = buena mezcla)")
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_19a_05_rangos", pdf)


def fig_geweke(df, dir_diag, pdf):
    fig, ax = plt.subplots(figsize=(11, 3.4))
    x0 = 0
    for b, g in df.groupby("bloque", sort=False):
        xs = np.arange(x0, x0 + len(g))
        ax.scatter(xs, g["geweke_z"], s=10, alpha=0.7, label=b)
        x0 += len(g)
    ax.axhline(1.96, color=u.PALETA["rojo"], ls="--", lw=1.0)
    ax.axhline(-1.96, color=u.PALETA["rojo"], ls="--", lw=1.0)
    ax.axhline(0, color=u.PALETA["negro"], lw=0.8)
    ax.set_xlabel("parametro (agrupado por bloque)")
    ax.set_ylabel("z de Geweke (max entre cadenas)")
    ax.set_title("Geweke: 10% inicial vs 50% final de cada cadena")
    ax.legend(fontsize=8, ncols=6)
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_19a_06_geweke", pdf)


# =============================================================================
# 3. ORQUESTADOR
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="19a: convergencia del Modelo 1")
    ap.add_argument("dirs", nargs="+", help="carpeta(s) de correr_gibbs_disco")
    ap.add_argument("--diag", default="diag_m1_py", help="carpeta de salida")
    ap.add_argument("--sin-kappa2-jk", action="store_true",
                    help="excluir kappa2_jk del diagnostico (si n_cells es enorme)")
    ap.add_argument("--pdf", action="store_true", help="guardar tambien PDF")
    args = ap.parse_args()

    u.estilo()
    dir_diag = Path(args.diag); dir_diag.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("DIAGNOSTICO DE CONVERGENCIA - Modelo 1 (19a, Python)")
    print("=" * 72)
    cad = u.CadenasM1(args.dirs)
    b = cad.base
    print(f"Cadenas: {cad.m} | muestras/cadena (min): {cad.n_min} | "
          f"N={b.meta.get('N','?')}, K={b.meta.get('K','?')}, "
          f"n_cells={b.meta.get('n_cells','?')} | "
          f"p_E={b.meta.get('p_E',0)}, p_M={b.meta.get('p_M',0)}, "
          f"p_D={b.meta.get('p_D',0)}")
    if cad.m == 1:
        print("NOTA: con 1 sola cadena, Rhat es el split-Rhat. Para el "
              "Gelman-Rubin real, correr 2-4 cadenas con semillas distintas\n"
              "      (correr_gibbs_disco con dir_salida y semilla diferentes) "
              "y pasarlas todas a este script.")

    print("\n== Calculo de diagnosticos ==")
    df = tabla_diagnosticos(cad, incluir_kappa2_jk=not args.sin_kappa2_jk)

    # -- log-verosimilitud total (opcional) --
    ll_tot = None
    if all(s.disponible("loglik") for s in cad.salidas):
        print("  serie de log-verosimilitud total (streaming sobre loglik.bin)...")
        series = [u.serie_loglik_total(s)[: cad.n_min] for s in cad.salidas]
        ll_tot = np.stack(series, axis=0)
        d_ll = u.diagnosticos_mcmc(ll_tot[None])
        fila = pd.DataFrame({"bloque": ["loglik_total"],
                             "parametro": ["log p(y|theta) total"]})
        for k, v in d_ll.items():
            fila[k] = v
        res_ll = u.resumen_posterior(ll_tot[None]); fila = pd.concat(
            [fila, res_ll], axis=1)
        fila["geweke_p"] = 2 * norm.sf(np.abs(fila["geweke_z"]))
        fila["geweke_p_fdr"] = np.nan
        df = pd.concat([df, fila], ignore_index=True)

    ruta_csv = dir_diag / "resumen_convergencia_M1.csv"
    df.to_csv(ruta_csv, index=False, encoding="utf-8")
    print(f"\n  [csv] {ruta_csv}  ({len(df)} parametros)")

    # -- resumen en consola --
    print("\n== Resumen global ==")
    print(f"  Rhat   : max={np.nanmax(df['rhat']):.4f} "
          f"(parametro: {df.loc[df['rhat'].idxmax(),'parametro']})")
    print(f"  ESSbulk: min={np.nanmin(df['ess_bulk']):.0f} "
          f"| mediana={np.nanmedian(df['ess_bulk']):.0f}")
    print(f"  ESStail: min={np.nanmin(df['ess_tail']):.0f} "
          f"| mediana={np.nanmedian(df['ess_tail']):.0f}")
    print(f"  MCSE(mu)={df.loc[df['parametro']=='mu','mcse'].squeeze():.4f} "
          f"(error Monte Carlo de la media posterior de mu)")

    # -- figuras --
    print("\n== Figuras ==")
    fig_trazas_globales(cad, ll_tot, dir_diag, args.pdf)
    fig_acf(cad, df, dir_diag, args.pdf)
    fig_rhat_ess(df, dir_diag, args.pdf)
    fig_peores_trazas(cad, df, dir_diag, args.pdf)
    fig_rangos(cad, df, dir_diag, args.pdf)
    fig_geweke(df, dir_diag, args.pdf)

    u.veredicto_convergencia(df)


if __name__ == "__main__":
    sys.exit(main())
