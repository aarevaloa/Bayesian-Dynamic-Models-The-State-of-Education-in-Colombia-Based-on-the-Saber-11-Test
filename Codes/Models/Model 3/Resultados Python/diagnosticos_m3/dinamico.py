"""
dinamico.py — Análisis y presentación de los resultados DINÁMICOS del Modelo 3.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ===========================================================================
# Evolución de estados: cuantiles por periodo
# ===========================================================================

def cuantiles_estado(theta_coef, probs=(0.025, 0.05, 0.5, 0.95, 0.975)):
    """
    theta_coef : (Tt+1, S) cadena de un coeficiente en todos los periodos.
    Devuelve (Tt+1, len(probs)) con los cuantiles por periodo.
    """
    return np.quantile(theta_coef, probs, axis=1).T


def evolucion_intercepto(muestras, probs=(0.025, 0.05, 0.5, 0.95, 0.975)):
    """Cuantiles del intercepto mu_t (fila 0 de theta) por periodo."""
    mu = muestras.theta[0, :, :]        # (Tt+1, S)
    return cuantiles_estado(mu, probs)


def evolucion_bloque(muestras, B, probs=(0.025, 0.05, 0.5, 0.95, 0.975)):
    """
    Cuantiles de todos los coeficientes de un bloque, por periodo.
    Devuelve dict: labels, quantiles (n_coef, Tt+1, n_probs), variabilidad.
    """
    idx = muestras.meta.bloques[B]["idx"]
    labels = muestras.meta.etiquetas_bloque(B)
    n_coef = len(idx)
    Tt1 = muestras.theta.shape[1]

    Q = np.empty((n_coef, Tt1, len(probs)))
    variabilidad = np.empty(n_coef)
    for l in range(n_coef):
        chain = muestras.theta[idx[l], :, :]     # (Tt+1, S)
        Q[l] = cuantiles_estado(chain, probs)
        # variabilidad temporal = sd de la mediana a lo largo de t
        variabilidad[l] = np.std(np.median(chain, axis=1))
    return {"labels": labels, "quantiles": Q, "variabilidad": variabilidad,
            "idx": idx, "probs": probs}


def top_dinamicos(evol_bloque, n=9):
    """Índices de los n coeficientes con mayor variabilidad temporal."""
    var = evol_bloque["variabilidad"]
    return np.argsort(var)[::-1][:min(n, var.size)]


# ===========================================================================
# LASSO dinámico: innovaciones omega y probabilidad de cambio
# ===========================================================================

def innovaciones_bloque(muestras, B, umbral_frac=0.1):
    """
    Para cada coeficiente l y periodo t (1..Tt):
        omega_{l,t}^(s) = theta_{l,t}^(s) - theta_{l,t-1}^(s)
        omega_median[l,t]  = mediana posterior de omega
        prob_cambio[l,t]   = P(|omega| > umbral_frac * SD_posterior(theta_l))

    Devuelve dict con matrices (n_coef, Tt), etiquetas y las innovaciones
    "pooled" para el QQ contra Laplace.
    """
    idx = muestras.meta.bloques[B]["idx"]
    labels = muestras.meta.etiquetas_bloque(B)
    n_coef = len(idx)
    Tt = muestras.meta.Tt

    omega_med = np.full((n_coef, Tt), np.nan)
    prob_cambio = np.full((n_coef, Tt), np.nan)

    for l in range(n_coef):
        chain = muestras.theta[idx[l], :, :]     # (Tt+1, S)
        escala = np.std(chain)                    # SD posterior agregada
        umbral = umbral_frac * escala
        for t in range(1, Tt + 1):
            omega = chain[t, :] - chain[t - 1, :]
            omega_med[l, t - 1] = np.median(omega)
            prob_cambio[l, t - 1] = np.mean(np.abs(omega) > umbral)

    omega_pool = omega_med[np.isfinite(omega_med)].ravel()
    return {"labels": labels, "omega_med": omega_med,
            "prob_cambio": prob_cambio, "omega_pool": omega_pool,
            "n_cambios_altos": int(np.sum(prob_cambio > 0.5)),
            "n_total": n_coef * Tt}


def qq_laplace(omega_pool):
    """
    Cuantiles teóricos Laplace ajustados vs empíricos de omega. La prior LASSO
    implica omega ~ Laplace marginal; desviación en las colas indica que la
    posterior tiene colas más pesadas que Laplace.
    Devuelve (q_teorico, q_empirico, b_hat).
    """
    x = np.asarray(omega_pool)
    x = x[np.isfinite(x)]
    if x.size < 50:
        return None, None, None
    med = np.median(x)
    b_hat = np.mean(np.abs(x - med))          # MLE escala Laplace
    q_emp = np.sort(x)
    probs = (np.arange(x.size) + 0.5) / x.size
    q_th = med + b_hat * np.sign(probs - 0.5) * np.log1p(-2 * np.abs(probs - 0.5))
    return q_th, q_emp, b_hat


# ===========================================================================
# Varianza departamental dinámica s_{k,t}
# ===========================================================================

def dinamica_varianza(muestras, probs=(0.05, 0.5, 0.95)):
    """
    Cuantiles de s_{k,t} para cada (k, t).
    Devuelve dict con matrices (m, Tt) para q05, mediana, q95 y ranking de
    departamentos por varianza promedio.
    """
    s = muestras.s                    # (m, Tt, S)
    m, Tt, _ = s.shape
    q = np.quantile(s, probs, axis=2)   # (n_probs, m, Tt)
    s_q05, s_med, s_q95 = q[0], q[1], q[2]
    s_prom = s_med.mean(axis=1)
    orden = np.argsort(s_prom)[::-1]
    return {"s_q05": s_q05, "s_median": s_med, "s_q95": s_q95,
            "s_prom": s_prom, "orden_desc": orden,
            "depto_ids": muestras.meta.depto_ids}


# ===========================================================================
# Trayectoria global de shrinkage
# ===========================================================================

def shrinkage_medio_por_periodo(muestras, B):
    """
    Traza cómo el shrinkage local del bloque B evoluciona en el tiempo:
    mediana sobre coeficientes y muestras de log(tau2_{l,t,B}) para cada t.
    Un valor más negativo = innovaciones más comprimidas hacia 0 en ese
    periodo (el modelo "congela" los coeficientes).
    Devuelve (Tt,) mediana y bandas.
    """
    tau2 = muestras.tau2(B)           # (p_B, Tt, S)
    log_tau2 = np.log(tau2)
    # mediana sobre (coef, muestra) por periodo
    med = np.median(log_tau2, axis=(0, 2))     # (Tt,)
    q25 = np.quantile(log_tau2, 0.25, axis=(0, 2))
    q75 = np.quantile(log_tau2, 0.75, axis=(0, 2))
    return {"median": med, "q25": q25, "q75": q75}


# ===========================================================================
# Resumen tidy de todos los parámetros
# ===========================================================================

def tabla_resumen(muestras):
    """
    Tabla larga parametro x periodo con media, mediana, sd, IC95, P(>0) y
    marcas de significancia (IC no cruza 0; P(dirección)>0.95).
    Devuelve lista de dicts (para volcar a CSV) — no usa pandas para mantener
    dependencias mínimas.
    """
    theta = muestras.theta
    p, Tt1, S = theta.shape
    labels = muestras.meta.theta_labels
    tiempo = muestras.meta.tiempo_ids
    t_lab = tiempo if len(tiempo) == Tt1 else (
        ["inicial"] + list(tiempo) if len(tiempo) == Tt1 - 1
        else [str(i) for i in range(Tt1)])

    filas = []
    for j in range(p):
        for t in range(Tt1):
            x = theta[j, t, :]
            media = float(x.mean())
            mediana = float(np.median(x))
            sd = float(x.std(ddof=1))
            q025 = float(np.quantile(x, 0.025))
            q975 = float(np.quantile(x, 0.975))
            p_pos = float(np.mean(x > 0))
            ic95 = "*" if (q025 > 0 or q975 < 0) else ""
            pd = "*" if max(p_pos, 1 - p_pos) > 0.95 else ""
            filas.append({
                "parametro": labels[j],
                "periodo": t_lab[t] if t < len(t_lab) else str(t),
                "t_idx": t,
                "media": media, "mediana": mediana, "sd": sd,
                "q025": q025, "q975": q975, "p_positivo": p_pos,
                "ic95_no_cero": ic95, "pd_alta": pd,
            })
    return filas
