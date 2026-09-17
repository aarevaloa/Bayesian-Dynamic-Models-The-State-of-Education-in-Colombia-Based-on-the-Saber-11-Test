"""
convergencia.py — Diagnóstico de convergencia MCMC para el Modelo 3.
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
# R-hat rank-normalizado (Vehtari 2021)
# ===========================================================================

def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """Rank-normalización: rango fraccional -> cuantil normal (Blom)."""
    shape = x.shape
    flat = x.ravel()
    ranks = stats.rankdata(flat, method="average")
    n = flat.size
    z = stats.norm.ppf((ranks - 3.0 / 8.0) / (n - 2.0 * 3.0 / 8.0 + 1.0))
    return z.reshape(shape)


def _rhat_bruto(chains: np.ndarray) -> float:
    """R̂ clásico (Gelman-Rubin) sobre matriz (n_chains, n_draws)."""
    m, n = chains.shape
    if n < 2:
        return np.nan
    medias = chains.mean(axis=1)
    var_within = chains.var(axis=1, ddof=1).mean()
    var_between = n * medias.var(ddof=1)
    var_hat = (n - 1) / n * var_within + var_between / n
    if var_within <= 0:
        return np.nan
    return float(np.sqrt(var_hat / var_within))


def split_rhat(chain: np.ndarray, n_split: int = 4) -> float:
    """
    split-R̂ rank-normalizado para UNA cadena (vector 1D) dividida en n_split
    segmentos. Réplica de la intención de split_rhat() de R pero con la
    corrección rank-normal de Vehtari (más robusta a colas pesadas, justo lo
    que ocurre con τ²).
    """
    chain = np.asarray(chain, dtype=float).ravel()
    n = chain.size
    seg = n // n_split
    if seg < 50:
        return np.nan
    segs = np.stack([chain[i * seg:(i + 1) * seg] for i in range(n_split)])
    # rank-normalizar conjuntamente sobre todos los segmentos
    z = _rank_normalize(segs)
    return _rhat_bruto(z)


def rhat_arviz(chain: np.ndarray, n_split: int = 4) -> float:
    """R̂ vía ArviZ para validación (mismos segmentos como cadenas)."""
    if not _HAS_ARVIZ:
        return np.nan
    chain = np.asarray(chain, dtype=float).ravel()
    n = chain.size
    seg = n // n_split
    if seg < 50:
        return np.nan
    segs = np.stack([chain[i * seg:(i + 1) * seg] for i in range(n_split)])
    # ArviZ espera (chain, draw); ya no hace split adicional si pasamos >1 cadena
    try:
        return float(az.rhat(az.convert_to_dataset(segs[np.newaxis, ...]
                                                    if segs.ndim == 1 else segs)))
    except Exception:
        # API 1.x: pasar ndarray (chain, draw) directo
        try:
            return float(az.rhat(segs))
        except Exception:
            return np.nan


# ===========================================================================
# ESS (Effective Sample Size)
# ===========================================================================

def _autocov(x: np.ndarray) -> np.ndarray:
    """Autocovarianza vía FFT (método de Geyer), sesgo 1/n."""
    x = np.asarray(x, dtype=float)
    n = x.size
    x = x - x.mean()
    # zero-pad a potencia de 2 >= 2n para FFT lineal
    m2 = 1
    while m2 < 2 * n:
        m2 *= 2
    f = np.fft.rfft(x, n=m2)
    acov = np.fft.irfft(f * np.conjugate(f), n=m2)[:n]
    acov /= n
    return acov


def ess_geyer(chain: np.ndarray, n_split: int = 4) -> float:
    """
    ESS con split + suma de pares monótona de Geyer (initial monotone
    sequence estimator). Coherente con el ESS de ArviZ (que usa este
    estimador sobre cadenas split).
    """
    chain = np.asarray(chain, dtype=float).ravel()
    n = chain.size
    seg = n // n_split
    if seg < 8:
        return np.nan
    segs = [chain[i * seg:(i + 1) * seg] for i in range(n_split)]
    m = n_split
    N = seg

    # autocovarianza promedio entre cadenas + varianza between
    acov = np.zeros(N)
    medias = np.array([s.mean() for s in segs])
    for s in segs:
        acov += _autocov(s)
    acov /= m
    var_within = acov[0] * N / (N - 1) if N > 1 else acov[0]
    var_between = medias.var(ddof=1) if m > 1 else 0.0
    var_plus = (N - 1) / N * var_within + var_between
    if var_plus <= 0:
        return np.nan

    rho = 1.0 - (var_within - acov) / var_plus  # rho_t
    # suma monótona de Geyer sobre pares
    rho_sum = 0.0
    t = 1
    while t + 1 < N:
        par = rho[t] + rho[t + 1]
        if par < 0:
            break
        rho_sum += par
        t += 2
    tau = 1.0 + 2.0 * rho_sum
    if tau < 1.0:
        tau = 1.0
    return float(m * N / tau)


def ess_arviz(chain: np.ndarray, n_split: int = 4, kind: str = "bulk") -> float:
    """ESS vía ArviZ para validación."""
    if not _HAS_ARVIZ:
        return np.nan
    chain = np.asarray(chain, dtype=float).ravel()
    n = chain.size
    seg = n // n_split
    if seg < 8:
        return np.nan
    segs = np.stack([chain[i * seg:(i + 1) * seg] for i in range(n_split)])
    try:
        return float(az.ess(segs, method=kind))
    except Exception:
        try:
            return float(az.ess(az.convert_to_dataset(segs), method=kind))
        except Exception:
            return np.nan


def ess_por_columnas(mat: np.ndarray, usar_arviz: bool = False) -> np.ndarray:
    """ESS para cada columna de una matriz (n_draws, n_param)."""
    mat = np.atleast_2d(mat)
    if mat.shape[0] < mat.shape[1] and mat.ndim == 2:
        # heurística: si viene (param, draws) transponer no; asumimos (draws,param)
        pass
    out = np.empty(mat.shape[1])
    fn = ess_arviz if (usar_arviz and _HAS_ARVIZ) else ess_geyer
    for j in range(mat.shape[1]):
        out[j] = fn(mat[:, j])
    return out


# ===========================================================================
# Geweke con corrección FDR
# ===========================================================================

def geweke_z(chain: np.ndarray, frac1: float = 0.1, frac2: float = 0.5) -> float:
    """
    Estadístico z de Geweke: compara media del primer frac1 y último frac2 de
    la cadena, normalizando por el error estándar espectral (densidad espectral
    en frecuencia 0) de cada tramo.
    """
    chain = np.asarray(chain, dtype=float).ravel()
    n = chain.size
    n1 = int(np.floor(frac1 * n))
    n2 = int(np.floor(frac2 * n))
    if n1 < 5 or n2 < 5:
        return np.nan
    x1 = chain[:n1]
    x2 = chain[n - n2:]
    m1, m2 = x1.mean(), x2.mean()
    s1 = _spectrum0_bm(x1)
    s2 = _spectrum0_bm(x2)
    denom = np.sqrt(s1 / n1 + s2 / n2)
    if denom <= 0:
        return np.nan
    return float((m1 - m2) / denom)


def geweke_fdr(z_vals: np.ndarray) -> np.ndarray:
    """p-values ajustados (Benjamini-Hochberg) a partir de estadísticos z."""
    z = np.asarray(z_vals, dtype=float)
    finite = np.isfinite(z)
    p = np.full(z.shape, np.nan)
    p[finite] = 2.0 * stats.norm.cdf(-np.abs(z[finite]))
    p_adj = np.full(z.shape, np.nan)
    if finite.sum() > 0:
        p_adj[finite] = _bh_adjust(p[finite])
    return p_adj


def _bh_adjust(p: np.ndarray) -> np.ndarray:
    """Corrección Benjamini-Hochberg (idéntica a p.adjust method='BH')."""
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    # imponer monotonía desde el final
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty_like(adj)
    out[order] = adj
    return out


# ===========================================================================
# Estimador espectral en frecuencia 0: batch means ("bm")
# ===========================================================================

def _spectrum0_bm(x: np.ndarray) -> float:
    """
    Estimador de la densidad espectral en 0 vía batch means (no solapados).
    Este es el estimador "bm": divide la cadena en b lotes de tamaño a,
    la varianza de las medias de lote (escalada) estima S0. A diferencia del
    estimador de Geyer/ventana, batch-means NO absorbe la deriva en S0, de
    modo que conserva potencia contra no-estacionariedad — requisito explícito
    para Heidelberger-Welch en este proyecto.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 10:
        return np.var(x, ddof=1) if n > 1 else 0.0
    a = max(1, int(np.floor(np.sqrt(n))))   # tamaño de lote ~ sqrt(n)
    b = n // a                               # número de lotes
    if b < 2:
        return np.var(x, ddof=1)
    x_trim = x[:a * b]
    lotes = x_trim.reshape(b, a)
    medias_lote = lotes.mean(axis=1)
    gran_media = x_trim.mean()
    # var de batch means -> S0 = a * var(medias_lote) * b/(b-1)  (escala espectral)
    s0 = a * np.sum((medias_lote - gran_media) ** 2) / (b - 1)
    return float(s0)


# ===========================================================================
# Heidelberger-Welch (estacionariedad) con batch-means
# ===========================================================================

def heidel_welch(chain: np.ndarray, alpha: float = 0.05,
                 eps: float = 0.1) -> dict:
    """
    Test de estacionariedad de Heidelberger-Welch usando el estadístico de
    Cramér-von Mises sobre el proceso de sumas parciales, con S0 estimado por
    batch-means. Descarta iterativamente el 10% inicial hasta pasar (o hasta
    descartar el 50%).

    Devuelve dict con: stest (1 pasa, 0 falla), start (fracción descartada),
    pvalue, halfwidth_test, halfwidth_ratio.
    """
    x0 = np.asarray(chain, dtype=float).ravel()
    n0 = x0.size

    for frac_desc in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        ini = int(np.floor(frac_desc * n0))
        x = x0[ini:]
        n = x.size
        if n < 20:
            break
        s0 = _spectrum0_bm(x)
        if s0 <= 0:
            continue
        # Proceso de Brownian bridge sobre sumas parciales estandarizadas
        media = x.mean()
        T = np.cumsum(x - media)
        # estadístico Cramér-von Mises
        t_grid = np.arange(1, n + 1)
        B = T / np.sqrt(n * s0)
        cvm = np.sum(B ** 2) / n
        pval = _cvm_pvalue(cvm)
        if pval > alpha:
            # pasa estacionariedad; test de half-width
            s0_hw = _spectrum0_bm(x)
            hw = 1.96 * np.sqrt(s0_hw / n)
            ratio = hw / abs(media) if media != 0 else np.inf
            return {
                "stest": 1,
                "start": frac_desc,
                "pvalue": pval,
                "halfwidth_test": int(ratio < eps),
                "halfwidth_ratio": ratio,
            }
    return {"stest": 0, "start": np.nan, "pvalue": np.nan,
            "halfwidth_test": 0, "halfwidth_ratio": np.nan}


def _cvm_pvalue(stat: float) -> float:
    """
    p-value aproximado del estadístico Cramér-von Mises para el puente
    browniano (distribución asintótica de Anderson-Darling/CvM). Se usa la
    aproximación de la serie para la distribución CvM de Heidelberger-Welch.
    """
    # Aproximación práctica: la distribución límite del estadístico de HW es la
    # de la integral del cuadrado de un puente browniano, cuyos cuantiles
    # tabulados dan ~0.461 (95%). Usamos una interpolación monótona de la CDF.
    # Tabla de (stat, p) de Heidelberger & Welch (1983).
    tab_stat = np.array([0.00, 0.02, 0.05, 0.12, 0.18, 0.24, 0.35,
                         0.46, 0.61, 0.74, 1.00, 2.00])
    tab_p = np.array([1.00, 0.99, 0.95, 0.80, 0.65, 0.50, 0.25,
                      0.05, 0.025, 0.01, 0.001, 1e-6])
    if stat <= tab_stat[0]:
        return 1.0
    if stat >= tab_stat[-1]:
        return float(tab_p[-1])
    return float(np.interp(stat, tab_stat, tab_p))


# ===========================================================================
# Raftery-Lewis (con fórmula corregida para el caso iid)
# ===========================================================================

def raftery_lewis(chain: np.ndarray, q: float = 0.025, r: float = 0.005,
                  s: float = 0.95) -> dict:
    """
    Diagnóstico de Raftery-Lewis: número de iteraciones para estimar el
    cuantil q con precisión ±r y probabilidad s.

    Corrección importante: para una cadena SIN dependencia (iid), la cadena
    binaria Z_t = 1{x_t <= cuantil_q} tiene dependencia nula y el factor de
    dependencia I debe tender a 1, de modo que N -> N_min. Implementaciones
    ingenuas devuelven un N inflado por mal manejo del caso de transición
    degenerada; aquí se maneja explícitamente.

    Devuelve dict: N_min, N, burn_in (M), thin (k), I (dependence factor).
    """
    x = np.asarray(chain, dtype=float).ravel()
    n = x.size
    phi = stats.norm.ppf(0.5 * (1 + s))
    # N mínimo (bajo independencia)
    N_min = int(np.ceil(phi ** 2 * q * (1 - q) / r ** 2))

    cuantil = np.quantile(x, q)
    z = (x <= cuantil).astype(int)

    # Buscar el thinning k tal que la cadena k-thinned sea Markov-1 (test BIC
    # de orden 1 vs orden 2). Empezamos en k=1.
    k = 1
    while True:
        zk = z[::k]
        if zk.size < 10:
            k = max(1, k - 1)
            zk = z[::k]
            break
        # transiciones de orden 1 y 2
        if _es_markov1(zk) or k > 20:
            break
        k += 1

    zk = z[::k]
    # matriz de transición 2x2
    t00 = np.sum((zk[:-1] == 0) & (zk[1:] == 0))
    t01 = np.sum((zk[:-1] == 0) & (zk[1:] == 1))
    t10 = np.sum((zk[:-1] == 1) & (zk[1:] == 0))
    t11 = np.sum((zk[:-1] == 1) & (zk[1:] == 1))
    n0 = t00 + t01
    n1 = t10 + t11
    if n0 == 0 or n1 == 0:
        # cadena binaria degenerada: comportamiento iid, I=1
        return {"N_min": N_min, "N": N_min, "burn_in": k, "thin": k, "I": 1.0}

    alpha = t01 / n0   # P(0->1)
    beta = t10 / n1    # P(1->0)
    suma = alpha + beta
    if suma <= 0 or suma >= 2:
        return {"N_min": N_min, "N": N_min, "burn_in": k, "thin": k, "I": 1.0}

    # Burn-in M: iteraciones para que |P^m - pi| < eps
    eps_conv = 0.001
    m_val = np.log(eps_conv * suma / max(alpha, beta)) / np.log(abs(1 - suma)) \
        if abs(1 - suma) > 0 else 0.0
    M = int(np.ceil(max(m_val, 0.0))) * k

    # N: longitud requerida
    numer = (2 - suma) * alpha * beta
    denom = suma ** 3
    N_val = numer / denom * (phi / r) ** 2
    N = int(np.ceil(N_val)) * k

    I = (M + N) / N_min if N_min > 0 else np.nan
    # cota inferior: N no puede ser menor que N_min
    N = max(N, N_min)
    return {"N_min": N_min, "N": N, "burn_in": M, "thin": k, "I": float(I)}


def _es_markov1(z: np.ndarray) -> bool:
    """Test BIC: ¿la cadena binaria es Markov de orden 1 vs orden 2?"""
    z = np.asarray(z, dtype=int)
    if z.size < 20:
        return True
    # conteos de tripletas
    cnt = {}
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                cnt[(a, b, c)] = np.sum(
                    (z[:-2] == a) & (z[1:-1] == b) & (z[2:] == c))
    # log-verosimilitud orden 2
    G2 = 0.0
    for a in (0, 1):
        for b in (0, 1):
            nab = cnt[(a, b, 0)] + cnt[(a, b, 1)]
            if nab == 0:
                continue
            for c in (0, 1):
                nabc = cnt[(a, b, c)]
                if nabc == 0:
                    continue
                # bajo orden 1: P(c|b) ignora a
                nb0 = sum(cnt[(aa, b, 0)] for aa in (0, 1))
                nb1 = sum(cnt[(aa, b, 1)] for aa in (0, 1))
                nb = nb0 + nb1
                p1 = (nb0 if c == 0 else nb1) / nb if nb > 0 else 0
                p2 = nabc / nab
                if p1 > 0 and p2 > 0:
                    G2 += 2 * nabc * np.log(p2 / p1)
    bic = G2 - 2 * np.log(z.size - 2)   # df = 2 para orden1 vs orden2
    return bic < 0   # BIC favorece orden 1 (más simple)


# ===========================================================================
# Contenedor y función de alto nivel
# ===========================================================================

@dataclass
class ResumenConvergencia:
    ess_mu: np.ndarray
    ess_lambda: np.ndarray
    ess_theta: dict          # {periodo: array}
    ess_s: np.ndarray
    geweke_z: np.ndarray
    geweke_pfdr: np.ndarray
    rhat_mu: np.ndarray
    rhat_lambda: np.ndarray
    rhat_theta: dict
    heidel_mu: list
    raftery: dict
    periodos_diag: list = field(default_factory=list)
    validacion_arviz: dict = field(default_factory=dict)


def periodos_diagnostico(Tt: int) -> list:
    """t = 0, T/4, T/2, 3T/4, T (0-based en el array theta de dim Tt+1)."""
    cand = sorted(set([
        0,
        int(np.ceil(Tt / 4)),
        int(np.ceil(Tt / 2)),
        int(np.ceil(3 * Tt / 4)),
        Tt,
    ]))
    return cand
