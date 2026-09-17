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
logsumexp = u.logsumexp


# =============================================================================
# 1. WAIC out-of-core (por bloques de columnas)
# =============================================================================
def _iter_bloques_col(mm, ancho):
    """Genera (col0, submatriz S x ancho) recorriendo el memmap por columnas."""
    S, N = mm.shape
    for c0 in range(0, N, ancho):
        c1 = min(N, c0 + ancho)
        yield c0, np.asarray(mm[:, c0:c1], dtype=np.float64)


def calcular_waic(mm, ancho=2048):
    """lppd, p_waic, WAIC y elpd_i puntual, en streaming por columnas.

    lppd_i = logsumexp_s(ll[s,i]) - log(S);  p_waic_i = var_s(ll[s,i])."""
    S, N = mm.shape
    lppd_i = np.empty(N); pwaic_i = np.empty(N)
    for c0, B in _iter_bloques_col(mm, ancho):
        lppd_i[c0:c0 + B.shape[1]] = logsumexp(B, axis=0) - np.log(S)
        pwaic_i[c0:c0 + B.shape[1]] = B.var(axis=0, ddof=1)
    lppd = lppd_i.sum(); p_waic = pwaic_i.sum()
    elpd = lppd - p_waic
    return dict(waic=-2 * elpd, elpd_waic=elpd, lppd=lppd, p_waic=p_waic,
                elpd_i=lppd_i - pwaic_i, lppd_i=lppd_i, pwaic_i=pwaic_i)


# =============================================================================
# 2. PSIS-LOO out-of-core con GPD de Zhang-Stephens vectorizada
# =============================================================================
def _gpd_fit_zhang(x):
    """Ajuste GPD (forma k, escala sigma) por verosimilitud perfilada de
    Zhang-Stephens (2009). x = excedencias positivas y ORDENADAS ascendente."""
    n = x.size
    if n < 5:
        return np.nan, np.nan
    x_max = x[-1]
    x_q = x[int(n / 4 + 0.5)]                       # cuartil inferior
    mgrid = 30 + int(np.sqrt(n))
    j = np.arange(1, mgrid + 1)
    theta = 1.0 / x_max + (1.0 - np.sqrt(mgrid / (j - 0.5))) / (3.0 * x_q)
    # log-verosimilitud perfilada l(theta) para cada nodo (vectorizada en x)
    #   k(theta) = -mean(log(1 - theta*x));  l = n*(log(theta/k) + k - 1)
    kth = -np.mean(np.log1p(-np.outer(theta, x)), axis=1)   # (mgrid,)
    with np.errstate(divide="ignore", invalid="ignore"):
        lth = n * (np.log(theta / kth) + kth - 1.0)
    lth[~np.isfinite(lth)] = -np.inf
    w = np.exp(lth - lth.max()); w /= w.sum()
    theta_hat = np.sum(theta * w)
    k = float(np.mean(np.log1p(-theta_hat * x)))
    sigma = float(-k / theta_hat)
    return k, sigma


def _psis_una_obs(lw, M):
    """Suaviza la cola superior de un vector de log-pesos y devuelve
    (lw_suavizado_y_normalizado, khat)."""
    S = lw.size
    orden = np.argsort(lw)
    idx_cola = orden[S - M:]
    lw_cola = lw[idx_cola]
    umbral = lw[orden[S - M - 1]]                   # log-peso justo bajo la cola
    exced = np.exp(lw_cola) - np.exp(umbral)        # excedencias (positivas)
    exced = np.sort(exced[exced > 0])
    khat = np.nan
    if exced.size >= 5:
        k, sigma = _gpd_fit_zhang(exced)
        if np.isfinite(k) and np.isfinite(sigma) and sigma > 0:
            khat = k
            m = lw_cola.size
            p = (np.arange(1, m + 1) - 0.5) / m
            # cuantiles esperados de la GPD (en la escala de excedencias)
            if abs(k) > 1e-8:
                q = sigma / k * ((1 - p) ** (-k) - 1)
            else:
                q = -sigma * np.log1p(-p)
            nuevos = np.log(np.exp(umbral) + q)
            # reasignar respetando el orden ascendente de la cola
            lw_cola_ord = np.argsort(lw_cola)
            lw[idx_cola[lw_cola_ord]] = np.sort(nuevos)
    # truncar pesos en S^(3/4) (Vehtari et al. 2021) y normalizar
    lw = lw - logsumexp(lw)
    tope = 0.75 * np.log(S) - logsumexp(lw)         # equivalente a max_w = S^{3/4}/sum
    lw = np.minimum(lw, tope)
    lw = lw - logsumexp(lw)
    # regularizacion de khat (Vehtari et al. 2021)
    if np.isfinite(khat):
        khat = (M * khat + 5.0) / (M + 10.0)
    return lw, khat


def calcular_loo(mm, ancho=1024):
    """PSIS-LOO en streaming por columnas. Devuelve elpd_i, khat_i, p_loo."""
    S, N = mm.shape
    M = int(min(0.2 * S, np.ceil(3 * np.sqrt(S))))
    M = max(M, 5)
    elpd_i = np.empty(N); khat_i = np.empty(N); lppd_i = np.empty(N)
    for c0, B in _iter_bloques_col(mm, ancho):
        for j in range(B.shape[1]):
            ll = B[:, j]
            lppd_i[c0 + j] = logsumexp(ll) - np.log(S)
            lw, kh = _psis_una_obs(-ll.copy(), M)  # log-pesos LOO crudos = -ll
            elpd_i[c0 + j] = logsumexp(lw + ll)
            khat_i[c0 + j] = kh
    elpd = elpd_i.sum()
    p_loo = lppd_i.sum() - elpd
    return dict(elpd_loo=elpd, looic=-2 * elpd, p_loo=p_loo,
                elpd_i=elpd_i, khat=khat_i, M=M)


# =============================================================================
# 3. DIC (usa la media posterior; requiere export de la Etapa 18)
# =============================================================================
def calcular_dic(sal: "u.SalidaM1", mm, ancho=4096):
    """DIC = 2*Dbar - D(theta_bar).

    Dbar = -2 * mean_s sum_i ll[s,i]  (streaming por filas via suma por columnas)
    D(theta_bar) = -2 * sum_i log p(y_i | theta_medio), evaluado en la media
    posterior de los parametros (reconstruida desde el export de datos)."""
    S, N = mm.shape
    # ---- Dbar: media sobre iteraciones de la desvianza total ----
    suma_por_iter = np.zeros(S)
    for _, B in _iter_bloques_col(mm, ancho):
        suma_por_iter += B.sum(axis=1)
    Dbar = -2.0 * suma_por_iter.mean()

    # ---- D(theta_bar): log-verosimilitud en la media posterior ----
    datos = sal.datos_ppc()
    if datos is None:
        return dict(dic=np.nan, Dbar=Dbar, D_thetabar=np.nan, p_dic=np.nan,
                    nota="sin export de datos (Etapa 18 con datos=): DIC omitido")
    mu_b = sal.bloque("mu").mean()
    bE = sal.bloque("beta_E").mean(axis=0) if sal.disponible("beta_E") else np.zeros(0)
    bM = sal.bloque("beta_M").mean(axis=0) if sal.disponible("beta_M") else np.zeros(0)
    bD = sal.bloque("beta_D").mean(axis=0) if sal.disponible("beta_D") else np.zeros(0)
    k2 = sal.bloque("kappa2_jk").mean(axis=0)      # media posterior por celda
    zeta = (mu_b
            + (datos["X_E"] @ bE if bE.size else 0.0)
            + (datos["Z"][datos["row_cell"]] @ bM if bM.size else 0.0)
            + (datos["W"][datos["row_k"]] @ bD if bD.size else 0.0))
    kap = k2[datos["row_cell"]]
    ll_bar = -0.5 * np.log(2 * np.pi * kap) - 0.5 * (datos["y"] - zeta) ** 2 / kap
    D_thetabar = -2.0 * ll_bar.sum()
    p_dic = Dbar - D_thetabar
    return dict(dic=2 * Dbar - D_thetabar, Dbar=Dbar, D_thetabar=D_thetabar,
                p_dic=p_dic, nota="")


# =============================================================================
# 4. FIGURAS
# =============================================================================
def fig_khat(loo, dir_diag, pdf):
    kh = loo["khat"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4),
                            gridspec_kw={"width_ratios": [2, 1]})
    axs[0].scatter(np.arange(len(kh)), kh, s=8, alpha=0.5,
                   color=u.PALETA["azul"])
    for y0, c, t in [(0.5, u.PALETA["naranja"], "0.5 (bueno)"),
                     (0.7, u.PALETA["rojo"], "0.7 (limite)")]:
        axs[0].axhline(y0, color=c, ls="--", lw=1.1, label=t)
    axs[0].set_xlabel("observacion"); axs[0].set_ylabel(r"$\hat{k}$ de Pareto")
    axs[0].set_title("Diagnostico PSIS por observacion"); axs[0].legend(fontsize=8)

    axs[1].hist(kh[np.isfinite(kh)], bins=40, color=u.PALETA["celeste"],
                orientation="horizontal")
    axs[1].axhline(0.5, color=u.PALETA["naranja"], ls="--", lw=1.1)
    axs[1].axhline(0.7, color=u.PALETA["rojo"], ls="--", lw=1.1)
    axs[1].set_xlabel("frecuencia"); axs[1].set_ylabel(r"$\hat{k}$")
    n_mal = int(np.nansum(kh > 0.7))
    axs[1].set_title(f"{n_mal} obs con k>0.7")
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_19c_01_khat", pdf)


def fig_elpd_depto(df_dep, dir_diag, pdf):
    if df_dep is None:
        return
    g = df_dep.sort_values("elpd_medio")
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.32 * len(g))))
    colores = np.where(g["elpd_medio"] < g["elpd_medio"].median(),
                       u.PALETA["rojo"], u.PALETA["verde"])
    ax.barh(y, g["elpd_medio"], color=colores, alpha=0.8)
    ax.set_yticks(y); ax.set_yticklabels(g["departamento"], fontsize=8)
    ax.set_xlabel("elpd LOO medio por observacion (mayor = mejor ajuste local)")
    ax.set_title("Ajuste predictivo por departamento")
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_19c_02_elpd_depto", pdf)


def fig_comparacion(comp, dir_diag, pdf):
    d = comp["d_loo_i"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].hist(d, bins=60, color=u.PALETA["morado"], alpha=0.8)
    axs[0].axvline(0, color=u.PALETA["negro"], lw=1.0, ls="--")
    axs[0].set_xlabel(f"elpd_i({comp['nombre_A']}) - elpd_i({comp['nombre_B']})")
    axs[0].set_title("Diferencia puntual de ajuste")

    orden = np.argsort(d)
    axs[1].plot(np.cumsum(d[orden]), color=u.PALETA["azul"], lw=1.4)
    axs[1].axhline(0, color=u.PALETA["negro"], lw=0.8, ls="--")
    axs[1].set_xlabel("observaciones (ordenadas por diferencia)")
    axs[1].set_ylabel("suma acumulada de la diferencia")
    axs[1].set_title(f"Delta_elpd = {comp['delta_loo']:+.1f} "
                     f"(SE {comp['se_loo']:.1f})")
    fig.tight_layout()
    u.guardar_fig(fig, dir_diag, "fig_19c_03_comparacion", pdf)


# =============================================================================
# 5. COMPARACION DE MODELOS
# =============================================================================
def comparar_modelos(sal_A, sal_B, nombre_A, nombre_B, dir_diag, pdf):
    mmA = sal_A.bloque("loglik", memmap=True)
    mmB = sal_B.bloque("loglik", memmap=True)
    if mmA.shape[1] != mmB.shape[1]:
        raise ValueError(f"N distinto: {nombre_A} tiene {mmA.shape[1]} obs, "
                         f"{nombre_B} tiene {mmB.shape[1]}. No comparables.")
    N = mmA.shape[1]
    print(f"\n== Comparacion: {nombre_A} vs {nombre_B}  (N={N}) ==")
    looA = calcular_loo(mmA); looB = calcular_loo(mmB)
    d_loo_i = looA["elpd_i"] - looB["elpd_i"]
    delta = d_loo_i.sum(); se = np.sqrt(N) * d_loo_i.std(ddof=1)
    waicA = calcular_waic(mmA); waicB = calcular_waic(mmB)
    mejor = nombre_A if delta > 0 else nombre_B
    creible = abs(delta) > 2 * se
    print(f"  elpd_LOO {nombre_A} = {looA['elpd_loo']:.1f}  |  "
          f"{nombre_B} = {looB['elpd_loo']:.1f}")
    print(f"  Delta_elpd({nombre_A}-{nombre_B}) = {delta:+.1f}  (SE {se:.1f})  "
          f"-> {abs(delta)/se:.1f} SE")
    print(f"  Preferido por LOO: {mejor}  "
          f"({'diferencia creible (>2 SE)' if creible else 'NO concluyente (<2 SE)'})")
    khA = np.nanmax(looA["khat"]); khB = np.nanmax(looB["khat"])
    if max(khA, khB) > 0.7:
        print(f"  /!\\ khat max: {nombre_A}={khA:.2f}, {nombre_B}={khB:.2f}: "
              "interpretar LOO con cautela.")
    comp = dict(nombre_A=nombre_A, nombre_B=nombre_B, delta_loo=delta,
                se_loo=se, d_loo_i=d_loo_i, mejor=mejor, creible=creible)
    pd.DataFrame({
        "modelo": [nombre_A, nombre_B],
        "elpd_loo": [looA["elpd_loo"], looB["elpd_loo"]],
        "looic": [looA["looic"], looB["looic"]],
        "p_loo": [looA["p_loo"], looB["p_loo"]],
        "waic": [waicA["waic"], waicB["waic"]],
        "p_waic": [waicA["p_waic"], waicB["p_waic"]],
        "khat_max": [khA, khB],
    }).to_csv(dir_diag / "comparacion_modelos.csv", index=False, encoding="utf-8")
    print(f"  [csv] {dir_diag}/comparacion_modelos.csv")
    fig_comparacion(comp, dir_diag, pdf)
    return comp


# =============================================================================
# 6. ORQUESTADOR
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="19c: ajuste (WAIC/LOO/DIC) del Modelo 1")
    ap.add_argument("dir", help="carpeta de correr_gibbs_disco (M1)")
    ap.add_argument("--diag", default="diag_m1_py")
    ap.add_argument("--comparar", default=None,
                    help="carpeta de otro modelo (p.ej. M2) para comparar por LOO")
    ap.add_argument("--nombres", nargs=2, default=["Normal (M1)", "t-Student (M2)"])
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    u.estilo()
    dir_diag = Path(args.diag); dir_diag.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("AJUSTE RELATIVO - Modelo 1 (19c: WAIC, PSIS-LOO, DIC)")
    print("=" * 72)
    sal = u.SalidaM1(args.dir)
    if not sal.disponible("loglik"):
        print("ERROR: loglik.bin no disponible. Re-correr el Gibbs con "
              "guardar_loglik=TRUE."); return 1
    mm = sal.bloque("loglik", memmap=True)
    S, N = mm.shape
    print(f"loglik: {S} muestras x {N} observaciones "
          f"({mm.nbytes/1e9:.2f} GB en disco, procesado out-of-core)")

    # -- WAIC --
    print("\n== WAIC (streaming) ==")
    waic = calcular_waic(mm)
    print(f"  WAIC = {waic['waic']:.1f}  (lppd={waic['lppd']:.1f}, "
          f"p_WAIC={waic['p_waic']:.1f}, elpd={waic['elpd_waic']:.1f})")

    # -- LOO --
    print("\n== PSIS-LOO (streaming, GPD Zhang-Stephens) ==")
    loo = calcular_loo(mm)
    kh = loo["khat"]
    print(f"  LOOIC = {loo['looic']:.1f}  (elpd_loo={loo['elpd_loo']:.1f}, "
          f"p_loo={loo['p_loo']:.1f}, M cola={loo['M']})")
    n_ok = int(np.nansum(kh <= 0.5)); n_med = int(np.nansum((kh > 0.5) & (kh <= 0.7)))
    n_mal = int(np.nansum(kh > 0.7))
    print(f"  khat: <=0.5 (bueno) {n_ok} | 0.5-0.7 {n_med} | >0.7 (poco fiable) "
          f"{n_mal}  {'(OK)' if n_mal == 0 else '<- revisar esas obs'}")

    # -- DIC --
    print("\n== DIC ==")
    dic = calcular_dic(sal, mm)
    if dic["nota"]:
        print(f"  {dic['nota']}")
    else:
        print(f"  DIC = {dic['dic']:.1f}  (Dbar={dic['Dbar']:.1f}, "
              f"D(theta_bar)={dic['D_thetabar']:.1f}, p_DIC={dic['p_dic']:.1f})")

    # -- guardar resumen y khat --
    pd.DataFrame([{
        "criterio_WAIC": waic["waic"], "p_WAIC": waic["p_waic"],
        "elpd_WAIC": waic["elpd_waic"], "criterio_LOOIC": loo["looic"],
        "elpd_LOO": loo["elpd_loo"], "p_LOO": loo["p_loo"],
        "khat_max": np.nanmax(kh), "prop_khat_gt_0.7": np.nanmean(kh > 0.7),
        "DIC": dic["dic"], "p_DIC": dic["p_dic"], "N": N, "n_save": S,
    }]).to_csv(dir_diag / "ajuste_M1.csv", index=False, encoding="utf-8")
    pd.DataFrame({"observacion": np.arange(1, N + 1),
                  "elpd_loo_i": loo["elpd_i"], "khat_i": kh,
                  "elpd_waic_i": waic["elpd_i"]}).to_csv(
        dir_diag / "khat_M1.csv", index=False, encoding="utf-8")
    print(f"\n  [csv] {dir_diag}/ajuste_M1.csv  y  khat_M1.csv")

    # -- elpd por departamento (si hay mapeo) --
    df_dep = None
    datos = sal.datos_ppc()
    if datos is not None and sal.nombres.get("deptos") is not None:
        dep = sal.nombres["deptos"]
        elpd_i = loo["elpd_i"]; rk = datos["row_k"]
        filas = []
        for k in range(len(dep)):
            sel = rk == k
            if sel.any():
                filas.append({"departamento": dep["depto_id"].iloc[k],
                              "n_obs": int(sel.sum()),
                              "elpd_medio": float(elpd_i[sel].mean()),
                              "elpd_total": float(elpd_i[sel].sum())})
        df_dep = pd.DataFrame(filas)
        df_dep.to_csv(dir_diag / "elpd_por_depto_M1.csv", index=False,
                      encoding="utf-8")
        print(f"  [csv] {dir_diag}/elpd_por_depto_M1.csv")

    # -- figuras --
    print("\n== Figuras ==")
    fig_khat(loo, dir_diag, args.pdf)
    fig_elpd_depto(df_dep, dir_diag, args.pdf)

    # -- comparacion opcional --
    if args.comparar:
        sal_B = u.SalidaM1(args.comparar)
        comparar_modelos(sal, sal_B, args.nombres[0], args.nombres[1],
                         dir_diag, args.pdf)

    print("\n" + "=" * 72)
    print("INTERPRETACION: WAIC/LOO/DIC son criterios RELATIVOS (menor = mejor).")
    print("  Sirven para elegir entre M1 y M2, no para juzgar si M1 es 'bueno'")
    print("  en terminos absolutos: eso lo decide el chequeo predictivo (19d).")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
