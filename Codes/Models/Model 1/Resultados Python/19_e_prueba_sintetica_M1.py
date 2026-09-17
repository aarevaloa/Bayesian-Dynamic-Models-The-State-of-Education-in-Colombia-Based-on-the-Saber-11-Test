# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# @  MODELO 1 - ETAPA 19e: FIXTURE SINTETICO PARA PROBAR LA SUITE 19_*
# _____________________________________________________________________________

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260703)

# ---- dimensiones del fixture -------------------------------------------------
K = 6                 # departamentos
J_POR_K = [4, 6, 5, 3, 7, 5]   # municipios por depto  -> n_cells = 30
N_CELLS = sum(J_POR_K)
P_E, P_M, P_D = 8, 4, 3
N_SAVE = 500
N_OBS_POR_CELDA = (30, 180)    # rango de tamano muestral por celda


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


def construir_verdad():
    """Parametros 'verdaderos' y datos observados, compartidos por las cadenas."""
    cell_k = np.repeat(np.arange(K), J_POR_K)                 # depto de cada celda
    cell_j = np.concatenate([np.arange(j) for j in J_POR_K])  # muni dentro del depto
    n_jk = RNG.integers(*N_OBS_POR_CELDA, size=N_CELLS)
    N = int(n_jk.sum())

    row_cell = np.repeat(np.arange(N_CELLS), n_jk)            # celda de cada obs
    row_k = cell_k[row_cell]                                  # depto de cada obs

    # matrices de diseno
    X_E = RNG.normal(0, 1, size=(N, P_E))
    X_E[:, 4] = RNG.integers(0, 2, N)                         # una dummy
    Z = RNG.normal(0, 1, size=(N_CELLS, P_M))
    W = RNG.normal(0, 1, size=(K, P_D))

    # coeficientes verdaderos (algunos exactamente 0 -> encogidos por el LASSO)
    beta_E = np.array([2.5, -1.8, 0.0, 0.9, 1.2, 0.0, -0.4, 0.0])
    beta_M = np.array([1.5, 0.0, -0.7, 0.0])
    beta_D = np.array([-1.1, 0.6, 0.0])
    mu_true = 250.0

    kappa2_k_true = np.array([120., 180., 90., 200., 150., 110.])   # var depto
    # var por celda alrededor de la de su depto (encogimiento jerarquico)
    kappa2_jk_true = np.array([
        max(20.0, RNG.normal(kappa2_k_true[cell_k[c]], 0.25 * kappa2_k_true[cell_k[c]]))
        for c in range(N_CELLS)])

    zeta = (mu_true + X_E @ beta_E + Z[row_cell] @ beta_M + W[row_k] @ beta_D)
    y = RNG.normal(zeta, np.sqrt(kappa2_jk_true[row_cell]))

    return dict(cell_k=cell_k, cell_j=cell_j, n_jk=n_jk, N=N,
                row_cell=row_cell, row_k=row_k, X_E=X_E, Z=Z, W=W,
                beta_E=beta_E, beta_M=beta_M, beta_D=beta_D, mu=mu_true,
                kappa2_k=kappa2_k_true, kappa2_jk=kappa2_jk_true, y=y, zeta=zeta)


def escribir_export(dir_salida, V, con_datos, semilla):
    """Escribe export_python/ imitando exactamente a exportar_para_python_m1."""
    exp = Path(dir_salida) / "export_python"
    exp.mkdir(parents=True, exist_ok=True)

    sizes = {"mu": 1, "sigma2_mu": 1, "beta_E": P_E, "beta_M": P_M, "beta_D": P_D,
             "tau2_E": P_E, "tau2_M": P_M, "tau2_D": P_D, "lambda2": 3,
             "kappa2_k": K, "kappa2_jk": N_CELLS, "loglik": V["N"]}
    pd.DataFrame({"bloque": list(sizes), "dim": list(sizes.values())}).to_csv(
        exp / "sizes.csv", index=False)

    meta = {"n_save": N_SAVE, "n_iter": N_SAVE, "burnin": 5000, "thin": 5,
            "total_iter": 5000 + N_SAVE * 5, "N": V["N"], "K": K,
            "n_cells": N_CELLS, "p": P_E + P_M + P_D, "p_E": P_E, "p_M": P_M,
            "p_D": P_D, "guardar_kappa2_jk": 1, "guardar_loglik": 1,
            "semilla": semilla, "timestamp": "fixture"}
    pd.DataFrame({"clave": list(meta), "valor": [str(v) for v in meta.values()]}
                 ).to_csv(exp / "meta.csv", index=False)

    depto_ids = [f"D{5000+k}" for k in range(K)]
    pd.DataFrame({"k": np.arange(1, K + 1), "depto_id": depto_ids}).to_csv(
        exp / "deptos.csv", index=False)

    (exp / "nombres_E.txt").write_text(
        "\n".join(f"cov_E_{i+1}" for i in range(P_E)) + "\n", encoding="utf-8")
    (exp / "nombres_M.txt").write_text(
        "\n".join(["TASA_HOMICIDIOS", "PROP_NBI", "SGP_EDUC_PC", "LOG_POBLACION"]) + "\n",
        encoding="utf-8")
    (exp / "nombres_D.txt").write_text(
        "\n".join(["PIB_CONSTANTE", "TASA_DESEMPLEO", "IPM"]) + "\n", encoding="utf-8")

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
        _tofile_colmajor(V["X_E"], exp / "X_E.bin")
        _tofile_colmajor(V["Z"], exp / "Z.bin")
        _tofile_colmajor(V["W"], exp / "W.bin")
        pd.DataFrame({"objeto": ["X_E", "Z", "W"],
                      "filas": [V["N"], N_CELLS, K],
                      "columnas": [P_E, P_M, P_D],
                      "orden": "col-major"}).to_csv(exp / "dims.csv", index=False)


def generar_cadena(dir_salida, V, semilla, degradar_loglik=0.0):
    """Genera .bin de una cadena con draws AR(1) alrededor de la verdad.

    degradar_loglik>0 desplaza la log-verosimilitud hacia peor (para simular un
    'modelo 2' con peor ajuste y probar la comparacion de 19_c)."""
    rng = np.random.default_rng(semilla)
    dir_salida = Path(dir_salida); dir_salida.mkdir(parents=True, exist_ok=True)

    def draws_vec(verdad, sd_frac, rho, positivo=False):
        cols = []
        for v in np.atleast_1d(verdad):
            sd = max(abs(v) * sd_frac, sd_frac)
            s = _ar1(N_SAVE, v, sd, rho, rng)
            if positivo:
                s = np.abs(s) + 1e-6
            cols.append(s)
        return np.column_stack(cols)

    mu = draws_vec(V["mu"], 0.02, 0.35)
    sigma2_mu = draws_vec(50.0, 0.15, 0.45, positivo=True)
    beta_E = draws_vec(V["beta_E"], 0.05, 0.4)
    beta_M = draws_vec(V["beta_M"], 0.06, 0.4)
    beta_D = draws_vec(V["beta_D"], 0.06, 0.4)
    tau2_E = draws_vec(np.abs(V["beta_E"]) + 0.5, 0.2, 0.5, positivo=True)
    tau2_M = draws_vec(np.abs(V["beta_M"]) + 0.5, 0.2, 0.5, positivo=True)
    tau2_D = draws_vec(np.abs(V["beta_D"]) + 0.5, 0.2, 0.5, positivo=True)
    lambda2 = draws_vec(np.array([1.0, 1.5, 2.0]), 0.15, 0.5, positivo=True)
    kappa2_k = draws_vec(V["kappa2_k"], 0.08, 0.5, positivo=True)
    kappa2_jk = draws_vec(V["kappa2_jk"], 0.12, 0.45, positivo=True)

    # log-verosimilitud coherente con los draws, por iteracion
    N = V["N"]
    loglik = np.empty((N_SAVE, N), dtype="<f8")
    Xb_E = V["X_E"] @ beta_E.T                 # (N, n_save)
    Zb_M = (V["Z"] @ beta_M.T)[V["row_cell"]]  # (N, n_save)
    Wb_D = (V["W"] @ beta_D.T)[V["row_k"]]     # (N, n_save)
    for s in range(N_SAVE):
        zeta_s = mu[s, 0] + Xb_E[:, s] + Zb_M[:, s] + Wb_D[:, s]
        kap_s = kappa2_jk[s][V["row_cell"]]
        ll = -0.5 * np.log(2 * np.pi * kap_s) - 0.5 * (V["y"] - zeta_s) ** 2 / kap_s
        loglik[s] = ll - degradar_loglik      # desplazamiento opcional
    loglik = np.ascontiguousarray(loglik)

    _tofile_c(mu, dir_salida / "mu.bin")
    _tofile_c(sigma2_mu, dir_salida / "sigma2_mu.bin")
    _tofile_c(beta_E, dir_salida / "beta_E.bin")
    _tofile_c(beta_M, dir_salida / "beta_M.bin")
    _tofile_c(beta_D, dir_salida / "beta_D.bin")
    _tofile_c(tau2_E, dir_salida / "tau2_E.bin")
    _tofile_c(tau2_M, dir_salida / "tau2_M.bin")
    _tofile_c(tau2_D, dir_salida / "tau2_D.bin")
    _tofile_c(lambda2, dir_salida / "lambda2.bin")
    _tofile_c(kappa2_k, dir_salida / "kappa2_k.bin")
    _tofile_c(kappa2_jk, dir_salida / "kappa2_jk.bin")
    _tofile_c(loglik, dir_salida / "loglik.bin")


def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/claude/fixture")
    V = construir_verdad()
    print(f"Fixture: N={V['N']}, K={K}, n_cells={N_CELLS}, "
          f"p=(E{P_E},M{P_M},D{P_D}), n_save={N_SAVE}")

    # Modelo 1: dos cadenas con semillas distintas
    for i, sem in enumerate([101, 202], start=1):
        d = base / "m1" / f"cad{i}"
        generar_cadena(d, V, semilla=sem)
        escribir_export(d, V, con_datos=(i == 1), semilla=sem)
        print(f"  escrito {d}")

    # 'Modelo 2' falso: misma verdad, log-verosimilitud peor -> comparar en 19_c
    d2 = base / "m2_falso" / "cad1"
    generar_cadena(d2, V, semilla=303, degradar_loglik=0.15)
    escribir_export(d2, V, con_datos=True, semilla=303)
    print(f"  escrito {d2} (loglik degradada para probar comparacion)")

    print("Listo.")


if __name__ == "__main__":
    main()
