# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# @  MODELO 2 - t-STUDENT JERARQUICO ESTATICO CON COVARIABLES Y LASSO BAYESIANO
# @  FIXTURE SINTETICO PARA PROBAR LA SUITE 14_*
# _____________________________________________________________________________
#

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260715)

# ---- dimensiones del fixture -------------------------------------------------
K = 6                          # departamentos
J_POR_K = [4, 6, 5, 3, 7, 5]   # municipios por depto  -> n_cells = 30
N_CELLS = sum(J_POR_K)
P_E, P_M, P_D = 8, 4, 3
PP1 = 1 + P_E + P_M + P_D
N_SAVE = 500
N_OBS_POR_CELDA = (30, 180)    # rango de tamano muestral por celda
NU_TRUE = 6.0                  # colas pesadas: la robustez t debe activarse
N_TEST_POR_CELDA = (5, 20)     # hold-out sintetico (Capa B)


def _tofile_c(arr, ruta):
    """Escribe (n_save, dim) en orden C (fila=iteracion), como writeBin append."""
    np.ascontiguousarray(arr, dtype="<f8").tofile(ruta)


def _tofile_colmajor(mat, ruta):
    """Escribe una matriz en orden col-major, como writeBin(as.double(M)) en R."""
    np.asarray(mat, dtype="<f8").reshape(-1, order="F").tofile(ruta)


def _ar1(n, media, sd, rho, rng):
    """Serie AR(1) estacionaria alrededor de `media` (autocorrelacion realista)."""
    x = np.empty(n)
    x[0] = media + rng.normal(0, sd)
    inn = sd * np.sqrt(1 - rho**2)
    for t in range(1, n):
        x[t] = media + rho * (x[t - 1] - media) + rng.normal(0, inn)
    return x


def _disenio(n_por_celda, rng):
    """Matrices de disenio X_E/X_M/X_D a nivel observacion para las celdas dadas
    (X_M constante dentro de la celda, X_D constante dentro del depto)."""
    cell_k = np.repeat(np.arange(K), J_POR_K)
    n_jk = np.asarray(n_por_celda)
    N = int(n_jk.sum())
    row_cell = np.repeat(np.arange(N_CELLS), n_jk)
    row_k = cell_k[row_cell]
    X_E = rng.normal(0, 1, size=(N, P_E))
    X_E[:, 4] = rng.integers(0, 2, N)                          # una dummy
    Z_cell = rng.normal(0, 1, size=(N_CELLS, P_M))             # contextual muni
    W_dep = rng.normal(0, 1, size=(K, P_D))                    # contextual depto
    X_star = np.column_stack([np.ones(N), X_E,
                              Z_cell[row_cell], W_dep[row_k]])
    return dict(n_jk=n_jk, N=N, row_cell=row_cell, row_k=row_k,
                cell_k=cell_k, X_star=X_star)


def construir_verdad():
    """Parametros 'verdaderos' y datos observados, compartidos por las cadenas."""
    cell_k = np.repeat(np.arange(K), J_POR_K)
    cell_j = np.concatenate([np.arange(j) for j in J_POR_K])
    n_jk = RNG.integers(*N_OBS_POR_CELDA, size=N_CELLS)
    D = _disenio(n_jk, RNG)

    # coeficientes verdaderos (algunos exactamente 0 -> encogidos por el LASSO)
    beta_E = np.array([2.5, -1.8, 0.0, 0.9, 1.2, 0.0, -0.4, 0.0])
    beta_M = np.array([1.5, 0.0, -0.7, 0.0])
    beta_D = np.array([-1.1, 0.6, 0.0])
    mu_true = 250.0
    beta_star = np.concatenate([[mu_true], beta_E, beta_M, beta_D])

    kappa2_k_true = np.array([120., 180., 90., 200., 150., 110.])   # var depto
    kappa2_jk_true = np.array([
        max(20.0, RNG.normal(kappa2_k_true[cell_k[c]],
                             0.25 * kappa2_k_true[cell_k[c]]))
        for c in range(N_CELLS)])

    zeta = D["X_star"] @ beta_star
    # y de la t MARGINAL: zeta + sqrt(kappa2) * t_nu  (colas pesadas reales)
    y = zeta + np.sqrt(kappa2_jk_true[D["row_cell"]]) * RNG.standard_t(NU_TRUE, D["N"])

    # hold-out sintetico con el mismo proceso generador
    n_test = RNG.integers(*N_TEST_POR_CELDA, size=N_CELLS)
    T = _disenio(n_test, RNG)
    zeta_test = T["X_star"] @ beta_star
    y_test = (zeta_test + np.sqrt(kappa2_jk_true[T["row_cell"]])
              * RNG.standard_t(NU_TRUE, T["N"]))

    return dict(cell_k=cell_k, cell_j=cell_j, n_jk=D["n_jk"], N=D["N"],
                row_cell=D["row_cell"], row_k=D["row_k"], X_star=D["X_star"],
                beta_star=beta_star, mu=mu_true, nu=NU_TRUE,
                kappa2_k=kappa2_k_true, kappa2_jk=kappa2_jk_true,
                y=y, zeta=zeta,
                test=dict(n_jk=n_test, N=T["N"], row_cell=T["row_cell"],
                          row_k=T["row_k"], X_star=T["X_star"], y=y_test))


def escribir_export(dir_salida, V, con_datos, semilla):
    """Escribe export_python/ imitando exactamente a exportar_para_python_m2."""
    exp = Path(dir_salida) / "export_python"
    exp.mkdir(parents=True, exist_ok=True)

    sizes = {"beta_star": PP1, "sigma2_mu": 1, "nu": 1, "lambda2": 3,
             "kappa2_k": K, "kappa2_jk": N_CELLS,
             "tau2_E": P_E, "tau2_M": P_M, "tau2_D": P_D,
             "loglik": V["N"], "xi": V["N"]}
    pd.DataFrame({"bloque": list(sizes), "dim": list(sizes.values())}).to_csv(
        exp / "sizes.csv", index=False)

    meta = {"n_save": N_SAVE, "n_iter": 5000 + N_SAVE * 5, "burn_in": 5000,
            "thin": 5, "N": V["N"], "K": K, "n_cells": N_CELLS,
            "p": PP1 - 1, "p_E": P_E, "p_M": P_M, "p_D": P_D, "pp1": PP1,
            "guardar_loglik": 1, "guardar_xi": 1,
            "semilla": semilla, "timestamp": "fixture"}
    pd.DataFrame({"clave": list(meta), "valor": [str(v) for v in meta.values()]}
                 ).to_csv(exp / "meta.csv", index=False)

    depto_ids = [f"D{5000 + k}" for k in range(K)]
    pd.DataFrame({"k": np.arange(1, K + 1), "depto_id": depto_ids}).to_csv(
        exp / "deptos.csv", index=False)

    nombres_E = [f"cov_E_{i+1}" for i in range(P_E)]
    nombres_M = ["TASA_HOMICIDIOS", "PROP_NBI", "SGP_EDUC_PC", "LOG_POBLACION"]
    nombres_D = ["PIB_CONSTANTE", "TASA_DESEMPLEO", "IPM"]
    (exp / "nombres_E.txt").write_text("\n".join(nombres_E) + "\n", encoding="utf-8")
    (exp / "nombres_M.txt").write_text("\n".join(nombres_M) + "\n", encoding="utf-8")
    (exp / "nombres_D.txt").write_text("\n".join(nombres_D) + "\n", encoding="utf-8")
    (exp / "theta_labels.txt").write_text(
        "\n".join(["intercepto"] + nombres_E + nombres_M + nombres_D) + "\n",
        encoding="utf-8")

    muni_ids = [f"M{8000 + c}" for c in range(N_CELLS)]
    pd.DataFrame({"celda": np.arange(1, N_CELLS + 1),
                  "k": V["cell_k"] + 1, "j": V["cell_j"] + 1,
                  "depto_id": [depto_ids[k] for k in V["cell_k"]],
                  "muni_id": muni_ids, "n_jk": V["n_jk"]}).to_csv(
        exp / "celdas.csv", index=False)

    np.array([np.pi, float(N_SAVE)], dtype="<f8").tofile(exp / "control.bin")

    if con_datos:
        V["y"].astype("<f8").tofile(exp / "y_vec.bin")
        (V["row_cell"] + 1).astype("<f8").tofile(exp / "row_cell.bin")
        (V["row_k"] + 1).astype("<f8").tofile(exp / "row_k.bin")
        _tofile_colmajor(V["X_star"], exp / "X_star.bin")
        pd.DataFrame({"objeto": ["X_star"], "filas": [V["N"]],
                      "columnas": [PP1], "orden": "col-major"}).to_csv(
            exp / "dims.csv", index=False)
        # ---- hold-out (Capa B) ----
        h = exp / "holdout"; h.mkdir(exist_ok=True)
        T = V["test"]
        T["y"].astype("<f8").tofile(h / "y_test.bin")
        (T["row_cell"] + 1).astype("<f8").tofile(h / "row_cell_test.bin")
        (T["row_k"] + 1).astype("<f8").tofile(h / "row_k_test.bin")
        _tofile_colmajor(T["X_star"], h / "X_star_test.bin")
        pd.DataFrame({"objeto": ["X_star_test"], "filas": [T["N"]],
                      "columnas": [PP1], "orden": "col-major"}).to_csv(
            h / "dims.csv", index=False)


def generar_cadena(dir_salida, V, semilla, degradar_loglik=0.0):
    """Genera los .bin de una cadena con draws AR(1) alrededor de la verdad.

    degradar_loglik>0 desplaza la log-verosimilitud hacia peor (para simular un
    modelo rival con peor ajuste y probar la comparacion Delta_elpd de 14_c)."""
    rng = np.random.default_rng(semilla)
    dir_salida = Path(dir_salida); dir_salida.mkdir(parents=True, exist_ok=True)

    def draws_vec(verdad, sd_frac, rho, positivo=False, piso=1e-6):
        cols = []
        for v in np.atleast_1d(verdad):
            sd = max(abs(v) * sd_frac, sd_frac)
            s = _ar1(N_SAVE, v, sd, rho, rng)
            if positivo:
                s = np.abs(s) + piso
            cols.append(s)
        return np.column_stack(cols)

    beta_star = draws_vec(V["beta_star"], 0.04, 0.4)
    beta_star[:, 0] = _ar1(N_SAVE, V["mu"], 0.6, 0.35, rng)   # mu con su propia sd
    sigma2_mu = draws_vec(50.0, 0.15, 0.45, positivo=True)
    # nu: AR(1) alrededor de la verdad, con piso > 2 (soporte del prior)
    nu = np.maximum(_ar1(N_SAVE, V["nu"], 0.6, 0.5, rng), 2.05)[:, None]
    lambda2 = draws_vec(np.array([1.0, 1.5, 2.0]), 0.15, 0.5, positivo=True)
    kappa2_k = draws_vec(V["kappa2_k"], 0.08, 0.5, positivo=True)
    kappa2_jk = draws_vec(V["kappa2_jk"], 0.12, 0.45, positivo=True)
    bE = np.abs(V["beta_star"][1:1 + P_E])
    bM = np.abs(V["beta_star"][1 + P_E:1 + P_E + P_M])
    bD = np.abs(V["beta_star"][1 + P_E + P_M:])
    tau2_E = draws_vec(bE + 0.5, 0.2, 0.5, positivo=True)
    tau2_M = draws_vec(bM + 0.5, 0.2, 0.5, positivo=True)
    tau2_D = draws_vec(bD + 0.5, 0.2, 0.5, positivo=True)

    # log-verosimilitud t MARGINAL coherente con los draws, por iteracion
    from scipy.special import gammaln
    N = V["N"]
    loglik = np.empty((N_SAVE, N), dtype="<f8")
    xi = np.empty((N_SAVE, N), dtype="<f8")
    Xb = V["X_star"] @ beta_star.T                # (N, n_save)
    for s in range(N_SAVE):
        zeta_s = Xb[:, s]
        kap_s = kappa2_jk[s][V["row_cell"]]
        nu_s = nu[s, 0]
        eps2 = (V["y"] - zeta_s) ** 2
        loglik[s] = (gammaln((nu_s + 1) / 2) - gammaln(nu_s / 2)
                     - 0.5 * np.log(nu_s * np.pi * kap_s)
                     - ((nu_s + 1) / 2) * np.log1p(eps2 / (nu_s * kap_s))
                     - degradar_loglik)
        # xi ~ condicional completa G((nu+1)/2, rate=(nu + eps^2/kappa2)/2),
        # exactamente lo que produce el paso de Gibbs de la Etapa 2
        rate = 0.5 * (nu_s + eps2 / kap_s)
        xi[s] = rng.gamma(shape=(nu_s + 1) / 2, scale=1.0 / rate)
    loglik = np.ascontiguousarray(loglik)
    xi = np.ascontiguousarray(xi)

    _tofile_c(beta_star, dir_salida / "beta_star.bin")
    _tofile_c(sigma2_mu, dir_salida / "sigma2_mu.bin")
    _tofile_c(nu, dir_salida / "nu.bin")
    _tofile_c(lambda2, dir_salida / "lambda2.bin")
    _tofile_c(kappa2_k, dir_salida / "kappa2_k.bin")
    _tofile_c(kappa2_jk, dir_salida / "kappa2_jk.bin")
    _tofile_c(tau2_E, dir_salida / "tau2_E.bin")
    _tofile_c(tau2_M, dir_salida / "tau2_M.bin")
    _tofile_c(tau2_D, dir_salida / "tau2_D.bin")
    _tofile_c(loglik, dir_salida / "loglik.bin")
    _tofile_c(xi, dir_salida / "xi.bin")


def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/claude/fixture_m2")
    V = construir_verdad()
    print(f"Fixture M2: N={V['N']} (test={V['test']['N']}), K={K}, "
          f"n_cells={N_CELLS}, p=(E{P_E},M{P_M},D{P_D}), pp1={PP1}, "
          f"n_save={N_SAVE}, nu_true={NU_TRUE}")

    # Modelo 2: dos cadenas con semillas distintas
    for i, sem in enumerate([101, 202], start=1):
        d = base / "m2" / f"cad{i}"
        generar_cadena(d, V, semilla=sem)
        escribir_export(d, V, con_datos=(i == 1), semilla=sem)
        print(f"  escrito {d}")

    # Modelo rival falso: misma verdad, log-verosimilitud peor -> Delta_elpd>0
    d2 = base / "rival_peor" / "cad1"
    generar_cadena(d2, V, semilla=303, degradar_loglik=0.15)
    escribir_export(d2, V, con_datos=True, semilla=303)
    print(f"  escrito {d2} (loglik degradada para probar la comparacion)")

    print("Listo.")


if __name__ == "__main__":
    main()
