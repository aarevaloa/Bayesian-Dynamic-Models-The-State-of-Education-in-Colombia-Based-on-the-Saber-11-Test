"""
predictivo.py — Diagnóstico predictivo posterior del Modelo 3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

try:
    import arviz as az
    _HAS_ARVIZ = True
except Exception:
    _HAS_ARVIZ = False


# ===========================================================================
# Predicción por muestra: media mu_i^(s) y varianza sigma2_i^(s)
# ===========================================================================

def _mu_por_muestra(F, theta_s, t_all):
    """
    mu_i^(s) = F_i · theta_{t_i}^(s)   para todas las obs a la vez.

    F        : (N, p)
    theta_s  : (p, Tt+1)  una muestra
    t_all    : (N,) periodo 1-based -> columna t (theta col 0 es t=0 inicial)
    """
    col = theta_s[:, t_all]          # (p, N): estado de cada obs
    return np.einsum("np,pn->n", F, col)


def _sigma2_por_muestra(s_s, v_s, k_all, j_all):
    """sigma2_i^(s) = s_{k_i,t_i}? No: s es (m,Tt); v es (max_nk,m)."""
    raise NotImplementedError  # reemplazado por versión con t_all abajo


def sigma2_obs(s_s, v_s, k_all, t_all, j_all):
    """
    sigma2_i = s_{k_i, t_i} / v_{j_i, k_i}

    s_s : (m, Tt)     una muestra de varianza departamental
    v_s : (max_nk, m) una muestra de precisión municipal
    índices 1-based en k_all/t_all/j_all.
    """
    s_i = s_s[k_all - 1, t_all - 1]        # (N,)
    v_i = v_s[j_all - 1, k_all - 1]        # (N,)
    return s_i / v_i


# ===========================================================================
# Log-verosimilitud puntual (N x S) — base de WAIC y LOO
# ===========================================================================

def log_lik_pointwise(datos, muestras, idx_muestras=None,
                      bloque_obs=200_000, verbose=True):
    """
    Calcula la matriz log_lik de forma (S, N) (convención ArviZ: draws x obs).

    Para N grande (~500K) y S grande esto puede ser enorme; se procesa por
    bloques de observaciones y se devuelve como float32 para ahorrar memoria.
    Si idx_muestras se da, usa ese subconjunto de muestras.

    datos    : DatosObservados
    muestras : MuestrasM3
    """
    y = datos.y
    F = datos.F
    k_all, t_all, j_all = datos.k_all, datos.t_all, datos.j_all
    N = datos.N_total

    theta = muestras.theta       # (p, Tt+1, S)
    s = muestras.s               # (m, Tt, S)
    v = muestras.v               # (max_nk, m, S)
    S_tot = theta.shape[2]

    if idx_muestras is None:
        idx_muestras = np.arange(S_tot)
    S = len(idx_muestras)

    log_lik = np.empty((S, N), dtype=np.float32)
    log_2pi = np.log(2.0 * np.pi)

    for si, s_idx in enumerate(idx_muestras):
        theta_s = theta[:, :, s_idx]
        s_s = s[:, :, s_idx]
        v_s = v[:, :, s_idx]
        # procesar por bloques de observaciones
        for a in range(0, N, bloque_obs):
            b = min(a + bloque_obs, N)
            sl = slice(a, b)
            mu = _mu_por_muestra(F[sl], theta_s, t_all[sl])
            sig2 = sigma2_obs(s_s, v_s, k_all[sl], t_all[sl], j_all[sl])
            resid = y[sl] - mu
            log_lik[si, sl] = (-0.5 * log_2pi - 0.5 * np.log(sig2)
                               - 0.5 * resid * resid / sig2).astype(np.float32)
        if verbose and (si + 1) % max(1, S // 10) == 0:
            print(f"    log_lik: {si + 1}/{S} muestras")

    return log_lik


# ===========================================================================
# WAIC vectorizado
# ===========================================================================

def waic(log_lik):
    """
    WAIC a partir de log_lik (S, N).

    lppd_i = logsumexp_s(ll) - log S
    p_waic_i = var_s(ll)
    elpd_waic = sum_i (lppd_i - p_waic_i)
    Devuelve dict con elpd_waic, p_waic, waic (=-2*elpd), y ee (error estándar).
    """
    ll = np.asarray(log_lik, dtype=np.float64)
    S, N = ll.shape
    lppd_i = _logsumexp(ll, axis=0) - np.log(S)
    p_waic_i = ll.var(axis=0, ddof=1)
    elpd_i = lppd_i - p_waic_i
    elpd = elpd_i.sum()
    p = p_waic_i.sum()
    se = np.sqrt(N * elpd_i.var(ddof=1))
    return {
        "elpd_waic": float(elpd),
        "p_waic": float(p),
        "waic": float(-2 * elpd),
        "se": float(se),
        "elpd_i": elpd_i,
        "p_waic_i": p_waic_i,
        "n_p_waic_alto": int(np.sum(p_waic_i > 0.4)),
    }


def _logsumexp(a, axis=None):
    amax = np.max(a, axis=axis, keepdims=True)
    out = np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True)) + amax
    return np.squeeze(out, axis=axis)


# ===========================================================================
# PSIS-LOO (Pareto-smoothed importance sampling)
# ===========================================================================

def psis_loo(log_lik, usar_arviz=True):
    """
    PSIS-LOO. Por defecto delega en ArviZ (validado a ~1e-13 en el proyecto).
    Si usar_arviz=False o ArviZ no está, usa una implementación propia de PSIS.

    Devuelve dict: elpd_loo, p_loo, loo (=-2*elpd), se, pareto_k (N,),
    n_k_altos (k>0.7).
    """
    ll = np.asarray(log_lik, dtype=np.float64)
    S, N = ll.shape

    if usar_arviz and _HAS_ARVIZ:
        try:
            import warnings
            # reff (relative effective sample size) de las muestras. Con una
            # sola cadena Gibbs se estima como ESS/S promedio; si no se puede,
            # reff=1.0 es la convención conservadora.
            reff = _reff_estimado(ll)
            dt = az.from_dict({"log_likelihood": {"y": ll[np.newaxis, :, :]}})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = az.loo(dt, pointwise=True, reff=reff)
            khat = _extraer_khat(res)
            elpd = float(res.elpd)
            return {
                "elpd_loo": elpd,
                "p_loo": float(res.p),
                "loo": float(-2 * elpd),
                "se": float(res.se),
                "pareto_k": khat,
                "n_k_altos": int(np.sum(khat > 0.7)) if khat is not None else -1,
                "reff": float(reff),
                "motor": "arviz",
            }
        except Exception as e:
            print(f"    (ArviZ loo falló: {e}; usando PSIS propio)")

    # PSIS propio
    khat = np.empty(N)
    lw = np.empty((S, N))
    for i in range(N):
        # pesos LOO no normalizados: -log_lik (importance ratios)
        lw_i, k_i = _psis_smooth(-ll[:, i])
        lw[:, i] = lw_i
        khat[i] = k_i
    # elpd_loo_i = logsumexp(lw_i + ll_i) - logsumexp(lw_i)
    elpd_i = _logsumexp(lw + ll, axis=0) - _logsumexp(lw, axis=0)
    elpd = elpd_i.sum()
    lppd_i = _logsumexp(ll, axis=0) - np.log(S)
    p_loo = float((lppd_i - elpd_i).sum())
    se = np.sqrt(N * elpd_i.var(ddof=1))
    return {
        "elpd_loo": float(elpd),
        "p_loo": p_loo,
        "loo": float(-2 * elpd),
        "se": float(se),
        "pareto_k": khat,
        "n_k_altos": int(np.sum(khat > 0.7)),
        "motor": "propio",
    }


def _psis_smooth(lw, tail_frac=0.2):
    """Suavizado de Pareto de la cola de pesos log (single obs)."""
    lw = lw - np.max(lw)
    S = lw.size
    M = int(min(0.2 * S, 3 * np.sqrt(S)))
    if M < 5:
        return lw - _logsumexp(lw), np.nan
    idx_sort = np.argsort(lw)
    tail_idx = idx_sort[-M:]
    tail = lw[tail_idx]
    cutoff = lw[idx_sort[-M - 1]]
    exp_tail = np.exp(tail) - np.exp(cutoff)
    exp_tail = np.clip(exp_tail, 1e-300, None)
    k, sigma = _gpd_fit(exp_tail)
    if np.isfinite(k):
        ranks = (np.arange(M) + 0.5) / M
        smoothed = _gpd_q(ranks, k, sigma) + np.exp(cutoff)
        lw[tail_idx] = np.log(np.clip(smoothed, 1e-300, None))
    lw = np.minimum(lw, 0.0)
    return lw - _logsumexp(lw), k


def _gpd_fit(x):
    """Ajuste GPD por el método de Zhang & Stephens (2009)."""
    x = np.sort(x)
    n = x.size
    if n < 5:
        return np.nan, np.nan
    m = 30 + int(np.sqrt(n))
    theta = 1.0 / x[-1] + (1 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))) \
        / (3.0 * x[int(n / 4 + 0.5) - 1])
    k_theta = np.array([np.mean(np.log1p(-t * x)) for t in theta])
    l_theta = n * (np.log(-theta / k_theta) - k_theta - 1)
    w = 1.0 / np.array([np.sum(np.exp(l_theta - lt)) for lt in l_theta])
    theta_hat = np.sum(theta * w)
    k = np.mean(np.log1p(-theta_hat * x))
    sigma = -k / theta_hat
    # corrección de sesgo para muestras pequeñas
    k = k * n / (n + 10) + 0.5 * 10 / (n + 10)
    return k, sigma


def _gpd_q(p, k, sigma):
    """Cuantil de la GPD."""
    if abs(k) < 1e-8:
        return -sigma * np.log1p(-p)
    return sigma * (np.power(1 - p, -k) - 1) / k


def _reff_estimado(ll):
    """
    reff = (ESS relativo) para PSIS-LOO. Con una sola cadena Gibbs, estima
    ESS/S promediado sobre observaciones. Cae a 1.0 si algo falla.
    """
    try:
        from .convergencia import ess_geyer
        S = ll.shape[0]
        # muestrear algunas columnas para no recorrer las N obs
        cols = np.linspace(0, ll.shape[1] - 1, min(50, ll.shape[1])).astype(int)
        ratios = []
        for c in cols:
            e = ess_geyer(ll[:, c], n_split=1)
            if np.isfinite(e) and e > 0:
                ratios.append(e / S)
        r = float(np.mean(ratios)) if ratios else 1.0
        return max(min(r, 1.0), 1e-3)
    except Exception:
        return 1.0


def _extraer_khat(res):
    """Extrae pareto_k de un ELPDData de ArviZ 1.2 (DataArray)."""
    for attr in ("pareto_k", "pareto_khat", "khat"):
        if hasattr(res, attr):
            v = getattr(res, attr)
            try:
                return np.asarray(v.values).ravel()
            except Exception:
                return np.asarray(v).ravel()
    return None


# ===========================================================================
# PIT exacto de la mezcla predictiva
# ===========================================================================

def pit_mezcla(datos, muestras, idx_muestras=None, bloque_obs=200_000,
               verbose=True):
    """
    PIT_i = (1/S) sum_s Phi( (y_i - mu_i^(s)) / sigma_i^(s) )

    Es el PIT exacto de la distribución predictiva (mezcla de Normales sobre
    las S muestras), no el PIT empírico por conteo de y_rep <= y_obs. Elimina
    el ruido Monte Carlo del conteo y da una calibración mucho más nítida.
    """
    y = datos.y
    F = datos.F
    k_all, t_all, j_all = datos.k_all, datos.t_all, datos.j_all
    N = datos.N_total

    theta = muestras.theta
    s = muestras.s
    v = muestras.v
    S_tot = theta.shape[2]

    if idx_muestras is None:
        idx_muestras = np.arange(S_tot)
    S = len(idx_muestras)

    pit = np.zeros(N, dtype=np.float64)
    for si, s_idx in enumerate(idx_muestras):
        theta_s = theta[:, :, s_idx]
        s_s = s[:, :, s_idx]
        v_s = v[:, :, s_idx]
        for a in range(0, N, bloque_obs):
            b = min(a + bloque_obs, N)
            sl = slice(a, b)
            mu = _mu_por_muestra(F[sl], theta_s, t_all[sl])
            sig2 = sigma2_obs(s_s, v_s, k_all[sl], t_all[sl], j_all[sl])
            pit[sl] += stats.norm.cdf((y[sl] - mu) / np.sqrt(sig2))
        if verbose and (si + 1) % max(1, S // 10) == 0:
            print(f"    PIT: {si + 1}/{S} muestras")
    pit /= S
    return pit


# ===========================================================================
# fair-CRPS (estimador insesgado)
# ===========================================================================

def fair_crps_normal(datos, muestras, idx_muestras=None, bloque_obs=100_000,
                     verbose=True):
    """
    CRPS con forma cerrada Normal por muestra, promediado. Para una predictiva
    Normal(mu, sigma), CRPS tiene forma analítica:

        CRPS(N(mu,sigma), y) = sigma * [ w(2 Phi(w) - 1) + 2 phi(w) - 1/sqrt(pi) ]
        con w = (y - mu)/sigma

    Se promedia sobre las S muestras la CRPS de cada Normal componente. Esto es
    el "fair" en el sentido de usar la forma cerrada por componente (sin ruido
    de muestreo de y_rep) y luego promediar; devuelve CRPS por obs y por (k,t).
    """
    y = datos.y
    F = datos.F
    k_all, t_all, j_all = datos.k_all, datos.t_all, datos.j_all
    N = datos.N_total

    theta = muestras.theta
    s = muestras.s
    v = muestras.v
    S_tot = theta.shape[2]
    if idx_muestras is None:
        idx_muestras = np.arange(S_tot)
    S = len(idx_muestras)

    inv_sqrt_pi = 1.0 / np.sqrt(np.pi)
    crps = np.zeros(N, dtype=np.float64)

    for si, s_idx in enumerate(idx_muestras):
        theta_s = theta[:, :, s_idx]
        s_s = s[:, :, s_idx]
        v_s = v[:, :, s_idx]
        for a in range(0, N, bloque_obs):
            b = min(a + bloque_obs, N)
            sl = slice(a, b)
            mu = _mu_por_muestra(F[sl], theta_s, t_all[sl])
            sig = np.sqrt(sigma2_obs(s_s, v_s, k_all[sl], t_all[sl], j_all[sl]))
            w = (y[sl] - mu) / sig
            Phi = stats.norm.cdf(w)
            phi = stats.norm.pdf(w)
            crps[sl] += sig * (w * (2 * Phi - 1) + 2 * phi - inv_sqrt_pi)
        if verbose and (si + 1) % max(1, S // 10) == 0:
            print(f"    CRPS: {si + 1}/{S} muestras")
    crps /= S
    return crps


# ===========================================================================
# PPC de estadísticos con cuantiles robustos
# ===========================================================================

def _skew(x):
    m = x.mean(); sd = x.std()
    return 0.0 if sd == 0 else np.mean((x - m) ** 3) / sd ** 3


def _kurt(x):
    m = x.mean(); sd = x.std()
    return 0.0 if sd == 0 else np.mean((x - m) ** 4) / sd ** 4 - 3.0


def ppc_estadisticos(datos, muestras, n_rep=200, seed=0, verbose=True):
    """
    Simula y_rep bajo el modelo para n_rep muestras y compara estadísticos
    (media, sd, cuantiles 5/25/75/95, asimetría, curtosis exceso) contra los
    observados. Devuelve estadísticos rep, observados, p-values bayesianos y
    estratificación por k y por t.
    """
    y = datos.y
    F = datos.F
    k_all, t_all, j_all = datos.k_all, datos.t_all, datos.j_all
    N = datos.N_total
    m = muestras.meta.m
    Tt = muestras.meta.Tt

    theta = muestras.theta
    s = muestras.s
    v = muestras.v
    S_tot = theta.shape[2]
    idx = np.linspace(0, S_tot - 1, min(n_rep, S_tot)).round().astype(int)
    R = len(idx)
    rng = np.random.default_rng(seed)

    stats_rep = {k: np.empty(R) for k in
                 ["media", "sd", "q05", "q25", "q75", "q95", "skew", "kurt"]}
    mean_rep_k = np.full((R, m), np.nan)
    sd_rep_k = np.full((R, m), np.nan)
    mean_rep_t = np.full((R, Tt), np.nan)
    sd_rep_t = np.full((R, Tt), np.nan)

    for r, s_idx in enumerate(idx):
        theta_s = theta[:, :, s_idx]
        s_s = s[:, :, s_idx]
        v_s = v[:, :, s_idx]
        mu = _mu_por_muestra(F, theta_s, t_all)
        sig = np.sqrt(sigma2_obs(s_s, v_s, k_all, t_all, j_all))
        yrep = rng.normal(mu, sig)

        stats_rep["media"][r] = yrep.mean()
        stats_rep["sd"][r] = yrep.std()
        stats_rep["q05"][r] = np.quantile(yrep, 0.05)
        stats_rep["q25"][r] = np.quantile(yrep, 0.25)
        stats_rep["q75"][r] = np.quantile(yrep, 0.75)
        stats_rep["q95"][r] = np.quantile(yrep, 0.95)
        stats_rep["skew"][r] = _skew(yrep)
        stats_rep["kurt"][r] = _kurt(yrep)

        for kk in range(1, m + 1):
            mask = k_all == kk
            if mask.sum() > 1:
                mean_rep_k[r, kk - 1] = yrep[mask].mean()
                sd_rep_k[r, kk - 1] = yrep[mask].std()
        for tt in range(1, Tt + 1):
            mask = t_all == tt
            if mask.sum() > 1:
                mean_rep_t[r, tt - 1] = yrep[mask].mean()
                sd_rep_t[r, tt - 1] = yrep[mask].std()
        if verbose and (r + 1) % max(1, R // 5) == 0:
            print(f"    PPC y_rep: {r + 1}/{R}")

    obs = {
        "media": y.mean(), "sd": y.std(),
        "q05": np.quantile(y, 0.05), "q25": np.quantile(y, 0.25),
        "q75": np.quantile(y, 0.75), "q95": np.quantile(y, 0.95),
        "skew": _skew(y), "kurt": _kurt(y),
    }
    pval = {k: float(np.mean(stats_rep[k] >= obs[k])) for k in obs}

    # observados por estrato
    mean_obs_k = np.array([y[k_all == kk].mean() if np.any(k_all == kk) else np.nan
                           for kk in range(1, m + 1)])
    sd_obs_k = np.array([y[k_all == kk].std() if np.any(k_all == kk) else np.nan
                         for kk in range(1, m + 1)])
    mean_obs_t = np.array([y[t_all == tt].mean() if np.any(t_all == tt) else np.nan
                           for tt in range(1, Tt + 1)])
    sd_obs_t = np.array([y[t_all == tt].std() if np.any(t_all == tt) else np.nan
                         for tt in range(1, Tt + 1)])

    pval_mean_k = np.array([
        np.mean(mean_rep_k[:, i] >= mean_obs_k[i])
        if np.all(np.isfinite(mean_rep_k[:, i])) else np.nan
        for i in range(m)])
    pval_mean_t = np.array([
        np.mean(mean_rep_t[:, i] >= mean_obs_t[i])
        if np.all(np.isfinite(mean_rep_t[:, i])) else np.nan
        for i in range(Tt)])

    return {
        "stats_rep": stats_rep, "obs": obs, "pval": pval,
        "mean_rep_k": mean_rep_k, "mean_obs_k": mean_obs_k,
        "mean_rep_t": mean_rep_t, "mean_obs_t": mean_obs_t,
        "sd_rep_t": sd_rep_t, "sd_obs_t": sd_obs_t,
        "pval_mean_k": pval_mean_k, "pval_mean_t": pval_mean_t,
    }


# ===========================================================================
# Residuos estructurales (con posterior medio)
# ===========================================================================

def residuos_estructurales(datos, muestras):
    """
    Residuos con el posterior medio de theta, s, v (idéntico a §6.2 de R):
        yhat_i = F_i · E[theta_{t_i}] ,  sigma2_i = E[s]/E[v]
        resid_std_i = (y_i - yhat_i) / sqrt(sigma2_i)
    Devuelve dict con residuos crudos y estandarizados, yhat, sigma2 y tests.
    """
    y = datos.y
    F = datos.F
    k_all, t_all, j_all = datos.k_all, datos.t_all, datos.j_all
    N = datos.N_total

    theta_mean = muestras.theta.mean(axis=2)   # (p, Tt+1)
    s_mean = muestras.s.mean(axis=2)           # (m, Tt)
    v_mean = muestras.v.mean(axis=2)           # (max_nk, m)

    yhat = _mu_por_muestra(F, theta_mean, t_all)
    sig2 = sigma2_obs(s_mean, v_mean, k_all, t_all, j_all)
    resid = y - yhat
    resid_std = resid / np.sqrt(sig2)

    # tests estructurales
    # tendencia temporal de residuo medio
    rt = np.array([resid[t_all == tt].mean() for tt in range(1, muestras.meta.Tt + 1)])
    xt = np.arange(rt.size)
    slope_t, _, _, p_trend, _ = stats.linregress(xt, rt)
    # heteroscedasticidad |resid| ~ yhat
    slope_h, _, _, p_het, _ = stats.linregress(yhat, np.abs(resid))

    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))

    return {
        "resid": resid, "resid_std": resid_std, "yhat": yhat, "sigma2": sig2,
        "rmse": rmse, "mae": mae,
        "p_trend_t": float(p_trend), "p_het": float(p_het),
        "resid_mean_t": rt,
    }
