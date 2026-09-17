"""
tau2_diag.py — Diagnóstico de los tau2 del LASSO bayesiano dinámico.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .convergencia import ess_geyer, ess_arviz, geweke_z, geweke_fdr, _HAS_ARVIZ


def diagnosticar_bloque(muestras, B, usar_arviz_ess=True):
    """
    Calcula todas las métricas de tau2 para el bloque B.
    Devuelve dict con matrices (p_B, Tt) y resúmenes.
    """
    tau2 = muestras.tau2(B)              # (p_B, Tt, S)
    labels = muestras.meta.etiquetas_bloque(B)
    p_B, Tt, S = tau2.shape
    log_tau2 = np.log(tau2)

    ess_mat = np.full((p_B, Tt), np.nan)
    gew_mat = np.full((p_B, Tt), np.nan)
    ess_fn = ess_arviz if (usar_arviz_ess and _HAS_ARVIZ) else ess_geyer

    for l in range(p_B):
        for t in range(Tt):
            x = log_tau2[l, t, :]
            if np.unique(x).size < 5:
                continue
            ess_mat[l, t] = ess_fn(x)
            gew_mat[l, t] = geweke_z(x)

    gew_pfdr = geweke_fdr(gew_mat.ravel()).reshape(gew_mat.shape)
    med_log_tau2 = np.median(log_tau2, axis=2)   # (p_B, Tt)

    n_low = int(np.sum(ess_mat < 200))
    n_fail_fdr = int(np.sum(gew_pfdr < 0.05))

    return {
        "B": B, "labels": labels,
        "ess_mat": ess_mat, "gew_mat": gew_mat, "gew_pfdr": gew_pfdr,
        "med_log_tau2": med_log_tau2, "log_tau2": log_tau2,
        "n_low_ess": n_low, "n_fail_fdr": n_fail_fdr,
        "p_B": p_B, "Tt": Tt,
        "ess_min": float(np.nanmin(ess_mat)) if np.any(np.isfinite(ess_mat)) else np.nan,
        "ess_med": float(np.nanmedian(ess_mat)) if np.any(np.isfinite(ess_mat)) else np.nan,
        "ess_max": float(np.nanmax(ess_mat)) if np.any(np.isfinite(ess_mat)) else np.nan,
    }


def comparar_bloques(muestras):
    """log(tau2) agregado por bloque para violines comparativos."""
    out = {}
    for B in ("E", "M", "D"):
        tau2 = muestras.tau2(B)
        lt = np.log(tau2).ravel()
        out[B] = {
            "log_tau2": lt,
            "mediana": float(np.median(lt)),
            "q25": float(np.quantile(lt, 0.25)),
            "q75": float(np.quantile(lt, 0.75)),
        }
    return out


def sanity_lambda_tau(muestras):
    """
    E[tau2|lambda2] = 2/lambda2 bajo la prior LASSO clásica. Si la mediana
    posterior de tau2 está muy lejos de 2/mediana(lambda2), hay problema de
    identificación o de implementación.
    """
    cols = {"E": 0, "M": 1, "D": 2}
    res = {}
    for B, c in cols.items():
        tau2 = muestras.tau2(B)
        lam2 = muestras.lambda2[:, c]
        esperado = 2.0 / np.median(lam2)
        observado = float(np.median(tau2))
        ratio = observado / esperado
        res[B] = {"esperado": esperado, "observado": observado, "ratio": ratio,
                  "ok": 0.5 <= ratio <= 2.0}
    return res


def acoplamiento_omega_tau(muestras, diag_bloque):
    """
    Regresión log(tau2_mediano) ~ log|omega_mediano|. Pendiente ~ 0 indica que
    el muestreador no acopla la innovación con su varianza local (stickiness).
    Devuelve dict por bloque con (log_abs_omega, log_tau2_med, pendiente).
    """
    res = {}
    for B in ("E", "M", "D"):
        idx = muestras.meta.bloques[B]["idx"]
        theta = muestras.theta
        Tt = muestras.meta.Tt
        # omega medianos por (l,t): theta[idx, t+1] - theta[idx, t]
        sub = theta[idx, :, :]                     # (p_B, Tt+1, S)
        omega = sub[:, 1:, :] - sub[:, :-1, :]     # (p_B, Tt, S)
        omega_med = np.median(omega, axis=2)       # (p_B, Tt)
        log_abs_omega = np.log(np.abs(omega_med.ravel()) + 1e-8)
        log_tau2_med = diag_bloque[B]["med_log_tau2"].ravel()

        mask = np.isfinite(log_abs_omega) & np.isfinite(log_tau2_med)
        if mask.sum() > 2:
            slope, intercept, r, p, se = stats.linregress(
                log_abs_omega[mask], log_tau2_med[mask])
        else:
            slope = np.nan
        res[B] = {"log_abs_omega": log_abs_omega, "log_tau2_med": log_tau2_med,
                  "pendiente": float(slope) if np.isfinite(slope) else np.nan}
    return res


def detectar_bimodalidad(muestras, B, threshold=0.05):
    """
    Cuenta celdas (l,t) cuya densidad de log(tau2) tiene >=2 modas con la
    segunda moda > threshold * primera. La bimodalidad hace engañosos los
    resúmenes por mediana.
    """
    tau2 = muestras.tau2(B)
    p_B, Tt, S = tau2.shape
    bimod = np.zeros((p_B, Tt), dtype=bool)
    for l in range(p_B):
        for t in range(Tt):
            x = np.log(tau2[l, t, :])
            if np.unique(x).size < 10:
                continue
            try:
                kde = stats.gaussian_kde(x)
                grid = np.linspace(x.min(), x.max(), 200)
                dens = kde(grid)
                # picos = máximos locales
                d = np.diff(np.sign(np.diff(dens)))
                peaks = np.where(d == -2)[0] + 1
                if peaks.size >= 2:
                    h = np.sort(dens[peaks])[::-1]
                    if h[1] / h[0] > threshold:
                        bimod[l, t] = True
            except Exception:
                continue
    n_bi = int(bimod.sum())
    return {"bimod": bimod, "n_bimodal": n_bi, "n_total": p_B * Tt,
            "frac": n_bi / (p_B * Tt)}
