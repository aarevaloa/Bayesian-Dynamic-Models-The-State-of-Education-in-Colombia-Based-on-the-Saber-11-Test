# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# @  MODELO 2 - t-STUDENT JERARQUICO ESTATICO CON LASSO BAYESIANO
# @  DIAGNOSTICO DE CONVERGENCIA
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
_spec = importlib.util.spec_from_file_location("util_m2", _AQUI / "14_0_utilidades_M2.py")
u = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(u)
plt = u.plt


# =============================================================================
# 1. RECORRIDO DE BLOQUES Y TABLA DE DIAGNOSTICOS
# =============================================================================
def tabla_diagnosticos(cad: "u.CadenasM2", incluir_kappa2_jk=True):
    """DataFrame con resumen posterior + diagnosticos para cada parametro.

    beta_star se reporta particionado (mu, beta_E, beta_M, beta_D); los
    tau2 se reportan en escala log (bloques log_tau2_*)."""
    filas = []

    def _agregar(X, bloque, etiquetas):
        d = u.diagnosticos_mcmc(X)
        res = u.resumen_posterior(X)
        res.insert(0, "parametro", etiquetas)
        res.insert(0, "bloque", bloque)
        for k, v in d.items():
            res[k] = v
        filas.append(res)
        print(f"  bloque {bloque:<11}: {X.shape[0]:4d} parametros diagnosticados")

    # -- beta_star particionado por nivel --
    Xb = u.preparar_bloque(cad, "beta_star")            # (pp1, m, n)
    et = cad.etiquetas("beta_star")
    b0 = cad.base
    particion = [("mu", [b0.idx_mu]), ("beta_E", list(b0.idx_E)),
                 ("beta_M", list(b0.idx_M)), ("beta_D", list(b0.idx_D))]
    for nombre, idx in particion:
        if len(idx):
            _agregar(np.ascontiguousarray(Xb[idx]), nombre,
                     [et[i] for i in idx])

    # -- escalares y bloques directos --
    for b in ("sigma2_mu", "nu", "lambda2", "kappa2_k", "kappa2_jk"):
        if b == "kappa2_jk" and not incluir_kappa2_jk:
            continue
        if cad.disponible(b):
            _agregar(u.preparar_bloque(cad, b), b, cad.etiquetas(b))

    # -- tau2 en escala log (cola pesada; ver encabezado) --
    for b in ("tau2_E", "tau2_M", "tau2_D"):
        if cad.disponible(b):
            X = np.log(u.preparar_bloque(cad, b))
            _agregar(X, "log_" + b, ["log " + e for e in cad.etiquetas(b)])

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
    if np.ptp(plano) <= 0:
        ax.hist(plano, bins=10, color=u.PALETA["gris"])
        ax.set_title(titulo); ax.set_yticks([])
        return
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
    b0 = cad.base
    Xb = u.preparar_bloque(cad, "beta_star")
    filas = [(Xb[b0.idx_mu], r"$\mu$ (intercepto global)"),
             (u.preparar_bloque(cad, "nu")[0],
              r"$\nu$ (grados de libertad de la t)"),
             (u.preparar_bloque(cad, "sigma2_mu")[0], r"$\sigma^2_\mu$")]
    lam = u.preparar_bloque(cad, "lambda2")
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
    fig.suptitle("Modelo 2 - trazas y densidades de los parametros estructurales",
                 y=1.001)
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_14a_01_trazas_globales", pdf)


def fig_nu(cad, df, dir_diag, pdf, lag_max=60):
    """Panel dedicado a nu: traza, ACF y densidad con el regimen de colas.

    Es la figura clave del capitulo de M2: si la masa posterior de nu esta en
    la zona 'casi normal', la robustez t no encuentra nada que corregir y el
    supuesto gaussiano de M1 queda respaldado; si esta bajo 10, hay colas
    pesadas y M1 esta mal especificado en colas."""
    X = u.preparar_bloque(cad, "nu")[0]               # (m, n)
    plano = X.reshape(-1)
    fila = df[df["bloque"] == "nu"].iloc[0]
    etiqueta, _, probs = u.interpretar_regimen_nu(plano)

    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.4))
    _traza(axs[0], X, "traza")
    axs[0].set_ylabel(r"$\nu$")

    ac = u._autocov_fft(X[None])                      # (1, m, n)
    rho = (ac / ac[:, :, :1]).mean(axis=1)[0][:lag_max + 1]
    axs[1].bar(np.arange(lag_max + 1), rho, width=0.9, color=u.PALETA["azul"])
    n_ef = X.shape[0] * X.shape[1]
    axs[1].axhline(0, color="k", lw=0.8)
    for s in (1, -1):
        axs[1].axhline(s * 1.96 / np.sqrt(n_ef), color=u.PALETA["rojo"],
                       lw=0.8, ls="--")
    axs[1].set_title("autocorrelacion (slice sampling en " + r"$\log(\nu-2)$)")
    axs[1].set_xlabel("rezago")

    lo, hi = np.quantile(plano, [0.001, 0.999])
    hi = max(hi, 32.0)                                # que se vean las 3 zonas
    zonas = [(2.0, 10.0, "naranja", "colas pesadas"),
             (10.0, 30.0, "amarillo", "intermedio"),
             (30.0, hi, "verde", "casi normal")]
    for a, b, c, _t in zonas:
        if b > lo and a < hi:
            axs[2].axvspan(max(a, lo), min(b, hi), color=u.PALETA[c], alpha=0.15)
    if np.ptp(plano) > 0:
        rejilla = np.linspace(lo, hi, 400)
        kde = u.stats.gaussian_kde(plano)
        axs[2].fill_between(rejilla, kde(rejilla), color=u.PALETA["azul"],
                            alpha=0.55)
    else:
        axs[2].hist(plano, bins=10, color=u.PALETA["azul"])
    axs[2].axvline(np.median(plano), color=u.PALETA["negro"], lw=1.1, ls="--")
    axs[2].set_title(f"posterior de " + r"$\nu$" + f"  -  regimen: {etiqueta}")
    axs[2].set_yticks([]); axs[2].set_xlabel(r"$\nu$")
    axs[2].text(0.98, 0.95,
                (f"P($\\nu$<10) = {probs['P(nu<10)']:.3f}\n"
                 f"P($\\nu$>30) = {probs['P(nu>30)']:.3f}\n"
                 f"P($\\nu$>100) = {probs['P(nu>100)']:.3f}"),
                transform=axs[2].transAxes, ha="right", va="top", fontsize=8.5,
                bbox=dict(boxstyle="round", fc="white", ec=u.PALETA["gris"]))

    fig.suptitle(rf"$\nu$: Rhat = {fila['rhat']:.3f}, "
                 rf"ESS bulk = {fila['ess_bulk']:.0f}, "
                 rf"ESS tail = {fila['ess_tail']:.0f}, "
                 rf"MCSE = {fila['mcse']:.3f}")
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_14a_02_nu", pdf)


def fig_acf(cad, df, dir_diag, pdf, lag_max=60):
    """ACF (promedio entre cadenas) de escalares + el beta de peor ESS."""
    b0 = cad.base
    Xb = u.preparar_bloque(cad, "beta_star")
    objetivos = [(Xb[b0.idx_mu], r"$\mu$"),
                 (u.preparar_bloque(cad, "sigma2_mu")[0], r"$\sigma^2_\mu$"),
                 (u.preparar_bloque(cad, "lambda2")[0], r"$\lambda^2_E$"),
                 (u.preparar_bloque(cad, "kappa2_k")[0], r"$\kappa^2_k[1]$")]
    betas = df[df["bloque"].str.startswith("beta")]
    if len(betas):
        peor = betas.loc[betas["ess_bulk"].idxmin()]
        pos = et_pos(cad, peor)
        objetivos.append((Xb[pos], f"peor ESS: {peor['parametro'][:28]}"))
    fig, axs = plt.subplots(1, len(objetivos), figsize=(2.9 * len(objetivos), 2.9),
                            sharey=True)
    for ax, (X, t) in zip(np.atleast_1d(axs), objetivos):
        ac = u._autocov_fft(X[None])
        rho = (ac / ac[:, :, :1]).mean(axis=1)[0][:lag_max + 1]
        ax.bar(np.arange(lag_max + 1), rho, width=0.9, color=u.PALETA["azul"])
        n_ef = X.shape[0] * X.shape[1]
        ax.axhline(0, color="k", lw=0.8)
        for s in (1, -1):
            ax.axhline(s * 1.96 / np.sqrt(n_ef), color=u.PALETA["rojo"],
                       lw=0.8, ls="--")
        ax.set_title(t, fontsize=9.5); ax.set_xlabel("rezago")
    np.atleast_1d(axs)[0].set_ylabel("autocorrelacion")
    fig.suptitle("Funciones de autocorrelacion (banda: +/-1.96/raiz(S))")
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_14a_03_acf", pdf)


def et_pos(cad, fila):
    """Columna (0-based) de un parametro de beta_star a partir de una fila de
    la tabla (bloques mu/beta_E/beta_M/beta_D usan etiquetas de beta_star)."""
    return cad.etiquetas("beta_star").index(fila["parametro"])


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
        base = b.split("_")[0].replace("log", "tau2")
        color = {"beta": u.PALETA["azul"], "tau2": u.PALETA["celeste"],
                 "kappa2": u.PALETA["naranja"], "lambda2": u.PALETA["verde"],
                 "mu": u.PALETA["negro"], "sigma2": u.PALETA["morado"],
                 "nu": u.PALETA["rojo"]}.get(base, u.PALETA["gris"])
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
    u.guardar_fig(fig, dir_diag, "fig_14a_04_rhat_ess", pdf)


def fig_peores_trazas(cad, df, dir_diag, pdf, n_peores=6):
    """El test de estres: trazas de los parametros con peor Rhat / menor ESS."""
    orden = df.sort_values(["rhat", "ess_bulk"], ascending=[False, True]).head(n_peores)
    fig, axs = plt.subplots(int(np.ceil(n_peores / 2)), 2,
                            figsize=(11, 1.9 * np.ceil(n_peores / 2)))
    axs = np.atleast_1d(axs).ravel()
    for ax, (_, fila) in zip(axs, orden.iterrows()):
        X = _extraer_param(cad, fila)
        _traza(ax, X, f"{fila['parametro'][:34]}  "
                      f"(Rhat={fila['rhat']:.3f}, ESSb={fila['ess_bulk']:.0f})")
        ax.set_xlabel("")
    for ax in axs[len(orden):]:
        ax.axis("off")
    fig.suptitle("Parametros con peor diagnostico")
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_14a_05_peores_trazas", pdf)


def _extraer_param(cad, fila):
    """(m, n) del parametro descrito por una fila de la tabla, respetando la
    particion de beta_star y la escala log de los tau2."""
    b = fila["bloque"]
    if b in ("mu", "beta_E", "beta_M", "beta_D"):
        return u.preparar_bloque(cad, "beta_star")[et_pos(cad, fila)]
    if b.startswith("log_tau2"):
        real = b[4:]
        idx = cad.etiquetas(real).index(fila["parametro"][4:])
        return np.log(u.preparar_bloque(cad, real)[idx])
    idx = cad.etiquetas(b).index(fila["parametro"])
    return u.preparar_bloque(cad, b)[idx]


def fig_rangos(cad, df, dir_diag, pdf, n_bins=20):
    """Graficos de rangos (Vehtari et al. 2021): cada mitad de cadena deberia
    tener rangos ~uniformes sobre el conjunto combinado."""
    b0 = cad.base
    Xb = u.preparar_bloque(cad, "beta_star")
    objetivos = [(Xb[b0.idx_mu], r"$\mu$"),
                 (u.preparar_bloque(cad, "nu")[0], r"$\nu$"),
                 (u.preparar_bloque(cad, "sigma2_mu")[0], r"$\sigma^2_\mu$")]
    betas = df[df["bloque"].str.startswith("beta")]
    if len(betas):
        peor = betas.loc[betas["rhat"].idxmax()]
        objetivos.append((Xb[et_pos(cad, peor)],
                          f"peor Rhat: {peor['parametro'][:24]}"))
    fig, axs = plt.subplots(1, len(objetivos), figsize=(3.0 * len(objetivos), 2.9),
                            sharey=True)
    for ax, (X, t) in zip(np.atleast_1d(axs), objetivos):
        Xs = u._dividir(X[None])[0]                   # (2m, n2)
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
    u.guardar_fig(fig, dir_diag, "fig_14a_06_rangos", pdf)


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
    u.guardar_fig(fig, dir_diag, "fig_14a_07_geweke", pdf)


# =============================================================================
# 3. ORQUESTADOR
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="14a: convergencia del Modelo 2")
    ap.add_argument("dirs", nargs="+", help="carpeta(s) de correr_gibbs_m2")
    ap.add_argument("--diag", default="diag_m2_py", help="carpeta de salida")
    ap.add_argument("--sin-kappa2-jk", action="store_true",
                    help="excluir kappa2_jk del diagnostico (si n_cells es enorme)")
    ap.add_argument("--pdf", action="store_true", help="guardar tambien PDF")
    args = ap.parse_args(argv)

    u.estilo()
    dir_diag = Path(args.diag); dir_diag.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("DIAGNOSTICO DE CONVERGENCIA - Modelo 2 (14a, Python)")
    print("=" * 72)
    cad = u.CadenasM2(args.dirs)
    b = cad.base
    print(f"Cadenas: {cad.m} | muestras/cadena (min): {cad.n_min} | "
          f"N={b.meta.get('N','?')}, K={b.meta.get('K','?')}, "
          f"n_cells={b.meta.get('n_cells','?')} | "
          f"p_E={b.p_E}, p_M={b.p_M}, p_D={b.p_D}")
    if cad.m == 1:
        print("NOTA: con 1 sola cadena, Rhat es el split-Rhat. Para el "
              "Gelman-Rubin real, correr 2-4 cadenas con semillas distintas\n"
              "      (correr_gibbs_m2 con dir_salida y semilla diferentes) "
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
        res_ll = u.resumen_posterior(ll_tot[None])
        fila = pd.concat([fila, res_ll], axis=1)
        fila["geweke_p"] = 2 * norm.sf(np.abs(fila["geweke_z"]))
        fila["geweke_p_fdr"] = np.nan
        df = pd.concat([df, fila], ignore_index=True)

    ruta_csv = dir_diag / "resumen_convergencia_M2.csv"
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
    f_mu = df[df["bloque"] == "mu"]
    f_nu = df[df["bloque"] == "nu"]
    if len(f_mu):
        print(f"  MCSE(mu)={f_mu['mcse'].iloc[0]:.4f} "
              f"(error Monte Carlo de la media posterior de mu)")
    if len(f_nu):
        nu_plano = u.preparar_bloque(cad, "nu")[0].reshape(-1)
        etq, texto, _ = u.interpretar_regimen_nu(nu_plano)
        print(f"  MCSE(nu)={f_nu['mcse'].iloc[0]:.4f} | regimen de nu: {etq}")
        print(f"    -> {texto}")

    # -- figuras --
    print("\n== Figuras ==")
    fig_trazas_globales(cad, ll_tot, dir_diag, args.pdf)
    fig_nu(cad, df, dir_diag, args.pdf)
    fig_acf(cad, df, dir_diag, args.pdf)
    fig_rhat_ess(df, dir_diag, args.pdf)
    fig_peores_trazas(cad, df, dir_diag, args.pdf)
    fig_rangos(cad, df, dir_diag, args.pdf)
    fig_geweke(df, dir_diag, args.pdf)

    ok = u.veredicto_convergencia(df)

    # -- resumen legible por 14_f --
    import json
    nu_plano = u.preparar_bloque(cad, "nu")[0].reshape(-1)
    etq, txt, probs = u.interpretar_regimen_nu(nu_plano)
    resumen = {
        "n_parametros": int(len(df)),
        "n_cadenas": int(cad.m),
        "n_save_min": int(cad.n_min),
        "rhat_max": float(np.nanmax(df["rhat"])),
        "param_rhat_max": str(df.loc[df["rhat"].idxmax(), "parametro"]),
        "ess_bulk_min": float(np.nanmin(df["ess_bulk"])),
        "ess_tail_min": float(np.nanmin(df["ess_tail"])),
        "n_rhat_altos": int(np.nansum(df["rhat"].to_numpy() > u.UMBRAL_RHAT)),
        "n_ess_bulk_bajos": int(np.nansum(df["ess_bulk"].to_numpy() < u.UMBRAL_ESS)),
        "n_ess_tail_bajos": int(np.nansum(df["ess_tail"].to_numpy() < u.UMBRAL_ESS)),
        "n_geweke_fdr": int(np.nansum(df["geweke_p_fdr"].to_numpy() < 0.05)),
        "mcse_mu": float(f_mu["mcse"].iloc[0]) if len(f_mu) else None,
        "mcse_nu": float(f_nu["mcse"].iloc[0]) if len(f_nu) else None,
        "nu_media": float(nu_plano.mean()),
        "nu_mediana": float(np.median(nu_plano)),
        "nu_regimen": etq, "nu_texto": txt, "nu_probs": probs,
        "convergencia_ok": bool(ok),
    }
    with open(dir_diag / "convergencia_resumen_M2.json", "w",
              encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)
    print(f"  [json] {dir_diag}/convergencia_resumen_M2.json")


if __name__ == "__main__":
    sys.exit(main())
