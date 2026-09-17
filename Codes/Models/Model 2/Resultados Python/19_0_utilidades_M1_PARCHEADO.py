# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ndtri, logsumexp  # noqa: F401 (logsumexp lo usa 19_c)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# =============================================================================
# 1. ESTILO GRAFICO
# =============================================================================
PALETA = {
    "azul":     "#0072B2",
    "naranja":  "#E69F00",
    "verde":    "#009E73",
    "rojo":     "#D55E00",
    "morado":   "#CC79A7",
    "celeste":  "#56B4E9",
    "amarillo": "#F0E442",
    "gris":     "#7F7F7F",
    "negro":    "#000000",
}
NIVEL_COLOR = {"E": PALETA["azul"], "M": PALETA["verde"], "D": PALETA["naranja"]}
NIVEL_NOMBRE = {"E": "Estudiante (E)", "M": "Municipio (M)", "D": "Departamento (D)"}


def estilo():
    """Fija el estilo comun de todas las figuras de la suite 19_*."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.titlesize": 11.5, "axes.titleweight": "semibold",
        "axes.labelsize": 10.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.autolayout": False,
    })


def guardar_fig(fig, dir_diag, nombre, tambien_pdf=False):
    """Guarda la figura en PNG 300 dpi (y opcionalmente PDF vectorial)."""
    dir_diag = Path(dir_diag); dir_diag.mkdir(parents=True, exist_ok=True)
    ruta = dir_diag / f"{nombre}.png"
    fig.savefig(ruta)
    if tambien_pdf:
        fig.savefig(dir_diag / f"{nombre}.pdf")
    plt.close(fig)
    print(f"  [fig] {ruta}")
    return ruta


# =============================================================================
# 2. LECTOR DE SALIDAS DEL GIBBS
# =============================================================================
BLOQUES = ["mu", "sigma2_mu", "beta_E", "beta_M", "beta_D",
           "tau2_E", "tau2_M", "tau2_D", "lambda2", "kappa2_k",
           "kappa2_jk", "loglik"]


class SalidaM1:
    """Una carpeta de correr_gibbs_disco (una cadena).

    Atributos principales
      dir          : Path de la carpeta con los .bin
      n_save       : numero de muestras guardadas (inferido de mu.bin)
      sizes        : dict bloque -> dimension
      nombres      : dict con nombres_E/M/D, depto_ids, celdas (DataFrame)
      meta         : dict con escalares (N, K, n_cells, burnin, thin, ...)
    """

    def __init__(self, dir_salida):
        self.dir = Path(dir_salida)
        if not self.dir.exists():
            raise FileNotFoundError(f"No existe la carpeta {self.dir}")
        self.export = self.dir / "export_python"
        self.meta, self.sizes, self.nombres = {}, {}, {}
        if (self.export / "meta.csv").exists():
            self._leer_export()
        else:
            warnings.warn(
                f"{self.dir}: no hay export_python/ (Etapa 18). Se infieren "
                "dimensiones de los .bin y se usan etiquetas genericas.")
        self._inferir_desde_bin()   # n_save y sizes faltantes SIEMPRE desde los .bin
        self._verificar_control()

    # ---- lectura de export_python -----------------------------------------
    def _leer_export(self):
        m = pd.read_csv(self.export / "meta.csv", dtype=str)
        for _, fila in m.iterrows():
            v = fila["valor"]
            try:
                v = int(float(v))
            except (TypeError, ValueError):
                pass
            self.meta[fila["clave"]] = v
        s = pd.read_csv(self.export / "sizes.csv")
        self.sizes = {str(r["bloque"]): int(r["dim"]) for _, r in s.iterrows()}
        for niv in "EMD":
            f = self.export / f"nombres_{niv}.txt"
            if f.exists():
                self.nombres[f"nombres_{niv}"] = f.read_text(encoding="utf-8").splitlines()
        f = self.export / "deptos.csv"
        if f.exists():
            self.nombres["deptos"] = pd.read_csv(f, dtype={"depto_id": str})
        f = self.export / "celdas.csv"
        if f.exists():
            self.nombres["celdas"] = pd.read_csv(
                f, dtype={"depto_id": str, "muni_id": str})

    # ---- inferencia de n_save/dimensiones desde los tamanos de archivo -----
    def _inferir_desde_bin(self):
        f_mu = self.dir / "mu.bin"
        if not f_mu.exists():
            raise FileNotFoundError(f"Falta {f_mu}: no parece una salida del Gibbs.")
        self.n_save = f_mu.stat().st_size // 8
        for b in BLOQUES:
            f = self.dir / f"{b}.bin"
            if not f.exists():
                continue
            nbytes = f.stat().st_size
            if nbytes % (8 * self.n_save) != 0:
                warnings.warn(f"{f.name}: {nbytes} bytes no es multiplo de "
                              f"n_save*8 ({self.n_save}*8). Corrida mezclada?")
                continue
            dim = nbytes // (8 * self.n_save)
            if b in self.sizes and self.sizes[b] not in (0, dim):
                warnings.warn(f"{f.name}: dim inferida {dim} != metadatos "
                              f"{self.sizes[b]}. Se usa la inferida.")
            self.sizes[b] = int(dim)
        # escalares minimos si no vinieron del export
        self.meta.setdefault("N", self.sizes.get("loglik", 0))
        self.meta.setdefault("K", self.sizes.get("kappa2_k", 0))
        self.meta.setdefault("n_cells", self.sizes.get("kappa2_jk", 0))
        for niv in "EMD":
            self.meta.setdefault(f"p_{niv}", self.sizes.get(f"beta_{niv}", 0))

    def _verificar_control(self):
        f = self.export / "control.bin"
        if not f.exists():
            return
        c = np.fromfile(f, dtype="<f8")
        if len(c) != 2 or abs(c[0] - np.pi) > 1e-9:
            warnings.warn(f"{f}: fallo la verificacion binaria (endianness?). "
                          "Revisar plataforma donde se corrio el Gibbs.")
        elif int(c[1]) != self.n_save:
            warnings.warn(f"control.bin dice n_save={int(c[1])} pero mu.bin "
                          f"implica {self.n_save}: los .bin cambiaron tras la Etapa 18.")

    # ---- acceso a bloques ---------------------------------------------------
    def disponible(self, bloque):
        return (self.dir / f"{bloque}.bin").exists() and self.sizes.get(bloque, 0) > 0

    def bloque(self, nombre, desde=0, hasta=None, memmap=False):
        """Devuelve las muestras del bloque como (n, dim) o (n,) si dim==1.

        desde/hasta son indices 0-based sobre las muestras guardadas;
        memmap=True devuelve un np.memmap (sin cargar a RAM: usarlo para
        loglik con N grande)."""
        if not self.disponible(nombre):
            raise KeyError(f"Bloque '{nombre}' no disponible en {self.dir}")
        dim = self.sizes[nombre]
        mm = np.memmap(self.dir / f"{nombre}.bin", dtype="<f8", mode="r",
                       shape=(self.n_save, dim))
        hasta = self.n_save if hasta is None else hasta
        out = mm[desde:hasta]
        if not memmap:
            out = np.array(out)
        if dim == 1 and not memmap:
            out = out[:, 0]
        return out

    def etiquetas(self, bloque):
        """Nombres de columna informativos para un bloque (lista de str)."""
        dim = self.sizes.get(bloque, 0)
        if bloque in ("beta_E", "beta_M", "beta_D", "tau2_E", "tau2_M", "tau2_D"):
            niv = bloque[-1]
            base = self.nombres.get(f"nombres_{niv}")
            if base and len(base) == dim:
                pref = "" if bloque.startswith("beta") else "tau2:"
                return [pref + n for n in base]
        if bloque == "lambda2" and dim == 3:
            return ["lambda2_E", "lambda2_M", "lambda2_D"]
        if bloque == "kappa2_k":
            dep = self.nombres.get("deptos")
            if dep is not None and len(dep) == dim:
                return [f"kappa2_k[{d}]" for d in dep["depto_id"]]
        if bloque == "kappa2_jk":
            cel = self.nombres.get("celdas")
            if cel is not None and len(cel) == dim:
                return [f"kappa2_jk[{d}:{m}]" for d, m
                        in zip(cel["depto_id"], cel["muni_id"])]
        if dim == 1:
            return [bloque]
        return [f"{bloque}[{i+1}]" for i in range(dim)]

    # ---- datos del PPC --------------------------------
    def datos_ppc(self):
        """Carga y_vec, row_cell, row_k, X_E, Z, W desde export_python.

        Devuelve dict o None si el export completo no se hizo."""
        f = self.export / "y_vec.bin"
        if not f.exists():
            return None
        d = {"y": np.fromfile(f, dtype="<f8")}
        d["row_cell"] = np.fromfile(self.export / "row_cell.bin", dtype="<f8").astype(np.int64) - 1
        d["row_k"] = np.fromfile(self.export / "row_k.bin", dtype="<f8").astype(np.int64) - 1
        dims = pd.read_csv(self.export / "dims.csv").set_index("objeto")

        def _mat(nombre):
            fila = dims.loc[nombre]
            nf, nc = int(fila["filas"]), int(fila["columnas"])
            if nc == 0:
                return np.zeros((nf, 0))
            x = np.fromfile(self.export / f"{nombre}.bin", dtype="<f8")
            return x.reshape((nf, nc), order="F")   # col-major: orden de R

        d["X_E"], d["Z"], d["W"] = _mat("X_E"), _mat("Z"), _mat("W")
        n = len(d["y"])
        assert d["X_E"].shape[0] == n and len(d["row_cell"]) == n
        return d


class CadenasM1:
    """Una o varias carpetas de salida (= cadenas con semillas distintas).

    Con 1 carpeta los diagnosticos usan split-Rhat; con >=2, Rhat multi-cadena
    real. Las cadenas se recortan a la longitud comun minima."""

    def __init__(self, dirs):
        if isinstance(dirs, (str, os.PathLike)):
            dirs = [dirs]
        self.salidas = [SalidaM1(d) for d in dirs]
        self.m = len(self.salidas)
        self.n_min = min(s.n_save for s in self.salidas)
        self.base = self.salidas[0]           # metadatos/etiquetas de referencia
        for s in self.salidas[1:]:
            for b, d in self.base.sizes.items():
                if s.sizes.get(b, d) != d:
                    raise ValueError(f"Dimensiones incompatibles entre cadenas en '{b}'")

    def disponible(self, bloque):
        return all(s.disponible(bloque) for s in self.salidas)

    def apilar(self, bloque):
        """(m_cadenas, n_min, dim) con las cadenas recortadas a n_min."""
        xs = []
        for s in self.salidas:
            x = s.bloque(bloque, hasta=self.n_min)
            if x.ndim == 1:
                x = x[:, None]
            xs.append(x)
        return np.stack(xs, axis=0)

    def etiquetas(self, bloque):
        return self.base.etiquetas(bloque)


class SalidaPool:
    """Adaptador que expone la MISMA interfaz que SalidaM1 pero agrupando los
    draws de varias cadenas (concatenados a lo largo de las muestras).

    Se usa en 19_b/19_c/19_d cuando se pasan varias carpetas: la posterior
    conjunta usa m*n_min draws, con lo que los forest plots y los cuantiles
    quedan mas suaves. Con una sola carpeta es equivalente a SalidaM1.
    Metadatos, etiquetas y datos del PPC se delegan a la primera cadena."""

    def __init__(self, dirs):
        if isinstance(dirs, (str, os.PathLike)):
            dirs = [dirs]
        self.salidas = [SalidaM1(d) for d in dirs]
        self.base = self.salidas[0]
        self.n_min = min(s.n_save for s in self.salidas)
        self.m = len(self.salidas)
        self.n_save = self.n_min * self.m
        self.meta, self.sizes, self.nombres = self.base.meta, self.base.sizes, self.base.nombres
        self.export, self.dir = self.base.export, self.base.dir
        for s in self.salidas[1:]:
            for b, d in self.base.sizes.items():
                if s.sizes.get(b, d) != d:
                    raise ValueError(f"Dimensiones incompatibles entre cadenas en '{b}'")

    def disponible(self, bloque):
        return all(s.disponible(bloque) for s in self.salidas)

    def bloque(self, nombre, desde=0, hasta=None, memmap=False):
        if memmap or self.m == 1:
            return self.base.bloque(nombre, desde, hasta, memmap)
        partes = [s.bloque(nombre, hasta=self.n_min) for s in self.salidas]
        return np.concatenate(partes, axis=0)   # (m*n_min, dim) o (m*n_min,)

    def etiquetas(self, bloque):
        return self.base.etiquetas(bloque)

    def datos_ppc(self):
        return self.base.datos_ppc()


def serie_loglik_total(salida: "SalidaM1", bloque_filas=256):
    """Traza S x 1 de la log-verosimilitud TOTAL (suma sobre observaciones),
    calculada por bloques de filas sobre el memmap (no carga S x N a RAM)."""
    if not salida.disponible("loglik"):
        return None
    mm = salida.bloque("loglik", memmap=True)
    S = mm.shape[0]
    out = np.empty(S)
    for i0 in range(0, S, bloque_filas):
        i1 = min(S, i0 + bloque_filas)
        out[i0:i1] = np.asarray(mm[i0:i1]).sum(axis=1)
    return out


# =============================================================================
# 3. DIAGNOSTICOS MCMC (Vehtari et al. 2021), vectorizados por parametro
# =============================================================================
# Convencion de formas: X tiene forma (P, m, n) = (parametros, cadenas, draws).

def _dividir(X):
    """Split de cada cadena en dos mitades: (P, m, n) -> (P, 2m, n//2)."""
    P, m, n = X.shape
    n2 = n // 2
    return X[:, :, : 2 * n2].reshape(P, m, 2, n2).reshape(P, 2 * m, n2)


def _autocov_fft(X):
    """Autocovarianza sesgada (divide entre n) por cadena via FFT.
    X: (P, M, n) -> (P, M, n)."""
    P, M, n = X.shape
    Xc = X - X.mean(axis=2, keepdims=True)
    n2 = 1
    while n2 < 2 * n:
        n2 <<= 1
    F = np.fft.rfft(Xc, n=n2, axis=2)
    ac = np.fft.irfft(F * np.conj(F), n=n2, axis=2)[:, :, :n].real
    return ac / n


def _rhat_nucleo(X):
    """R-hat de Gelman-Rubin sobre cadenas ya divididas. X: (P, M, n) -> (P,)."""
    P, M, n = X.shape
    medias = X.mean(axis=2)
    W = X.var(axis=2, ddof=1).mean(axis=1)
    B_n = medias.var(axis=1, ddof=1)             # = B/n
    var_plus = W * (n - 1) / n + B_n
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.sqrt(var_plus / W)
    r[W <= 0] = np.nan
    return r


def _ess_nucleo(X):
    """ESS multi-cadena (Geyer 1992 via Vehtari et al. 2021). X: (P, M, n) -> (P,).

    Replica EXACTAMENTE el algoritmo de referencia (Stan / ArviZ `_ess`):
      - autocovarianza sesgada por cadena via FFT, promediada entre cadenas;
      - rho_0 fijado en 1; rho_t = 1 - (W - acov_t)/var_plus para t>=1;
      - secuencia inicial POSITIVA de Geyer sobre pares (rho_{2t}+rho_{2t+1}),
        con un termino impar final adicional si el elemento par sobrante es >0;
      - secuencia inicial MONOTONA de Geyer (promediando dentro del par);
      - tau = -1 + 2*sum(rho_0..rho_maxt) + rho_{maxt+1}, con piso
        1/log10(M*n); ESS = M*n/tau (SIN tope: puede exceder M*n si las
        cadenas son antiteticas).
    Validado digito a digito contra arviz 0.22 (ver 14_e / bitacora)."""
    P, M, n = X.shape
    if n < 8:
        return np.full(P, np.nan)
    ac = _autocov_fft(X).mean(axis=1)                            # (P, n) sesgada
    mean_var = ac[:, 0] * n / (n - 1.0)                          # W insesgada
    B = X.mean(axis=2).var(axis=1, ddof=1) if M > 1 else np.zeros(P)
    var_plus = mean_var * (n - 1.0) / n + B
    ess = np.full(P, np.nan)
    validos = np.isfinite(var_plus) & (var_plus > 0)
    for p in np.where(validos)[0]:
        rho = np.zeros(n)
        rho[0] = 1.0
        rho_par = 1.0
        rho_imp = 1.0 - (mean_var[p] - ac[p, 1]) / var_plus[p]
        rho[1] = rho_imp
        t = 1
        while t < (n - 3) and (rho_par + rho_imp) > 0.0:
            rho_par = 1.0 - (mean_var[p] - ac[p, t + 1]) / var_plus[p]
            rho_imp = 1.0 - (mean_var[p] - ac[p, t + 2]) / var_plus[p]
            if (rho_par + rho_imp) >= 0:
                rho[t + 1] = rho_par
                rho[t + 2] = rho_imp
            t += 2
        max_t = t - 2
        if rho_par > 0:                       # termino par sobrante
            rho[max_t + 1] = rho_par
        # secuencia inicial monotona (promedia dentro del par si sube)
        t = 1
        while t <= max_t - 2:
            if (rho[t + 1] + rho[t + 2]) > (rho[t - 1] + rho[t]):
                rho[t + 1] = (rho[t - 1] + rho[t]) / 2.0
                rho[t + 2] = rho[t + 1]
            t += 2
        mn = M * n
        tau = -1.0 + 2.0 * rho[: max_t + 1].sum() + rho[max_t + 1: max_t + 2].sum()
        tau = max(tau, 1.0 / np.log10(mn))
        ess[p] = mn / tau
    return ess


def _rango_normalizar(X):
    """Rangos fraccionales -> cuantiles normales (Vehtari et al. 2021, ec. 14).
    X: (P, M, n); los rangos se calculan JUNTANDO todas las cadenas."""
    P, M, n = X.shape
    plano = X.reshape(P, M * n)
    r = stats.rankdata(plano, method="average", axis=1)
    z = ndtri((r - 0.375) / (M * n + 0.25))
    return z.reshape(P, M, n)


def diagnosticos_mcmc(X):
    """Diagnosticos completos por parametro.

    X : array (P, m, n) o (m, n, P) NO -- usar (m, n, dim) via preparar().
        Aqui se espera (P, m, n).
    Devuelve dict de arrays (P,):
      rhat (max de bulk y plegado, rango-normalizados), ess_bulk, ess_tail,
      mcse (sd/sqrt(ess_bulk)), geweke_z (max |z| entre cadenas).
    """
    P, m, n = X.shape
    Xs = _dividir(X)                                             # (P, 2m, n2)

    # --- Rhat rango-normalizado (bulk) y plegado (colas/varianza) ---
    Zb = _rango_normalizar(Xs)
    r_bulk = _rhat_nucleo(Zb)
    med = np.median(X.reshape(P, -1), axis=1)
    Xf = np.abs(X - med[:, None, None])
    Zf = _rango_normalizar(_dividir(Xf))
    r_fold = _rhat_nucleo(Zf)
    rhat = np.fmax(r_bulk, r_fold)

    # --- ESS bulk (sobre rango-normalizados) ---
    ess_bulk = _ess_nucleo(Zb)

    # --- ESS tail: min ESS de los indicadores de los cuantiles 5% y 95% ---
    q05 = np.quantile(X.reshape(P, -1), 0.05, axis=1)
    q95 = np.quantile(X.reshape(P, -1), 0.95, axis=1)
    I05 = _dividir((X <= q05[:, None, None]).astype(np.float64))
    I95 = _dividir((X <= q95[:, None, None]).astype(np.float64))
    ess_tail = np.fmin(_ess_nucleo(I05), _ess_nucleo(I95))

    # --- MCSE de la media ---
    sd = X.reshape(P, -1).std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mcse = sd / np.sqrt(ess_bulk)

    # --- Geweke por cadena (10% inicial vs 50% final), max |z| ---
    gz = np.full((P, m), np.nan)
    n1, n2 = max(int(0.1 * n), 5), max(int(0.5 * n), 10)
    if n1 + n2 < n:
        A = X[:, :, :n1]
        B = X[:, :, n - n2:]
        eA = _ess_nucleo(_dividir(A.reshape(P * m, 1, n1))).reshape(P, m)
        eB = _ess_nucleo(_dividir(B.reshape(P * m, 1, n2))).reshape(P, m)
        vA = A.var(axis=2, ddof=1) / np.fmax(eA, 1.0)
        vB = B.var(axis=2, ddof=1) / np.fmax(eB, 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            gz = (A.mean(axis=2) - B.mean(axis=2)) / np.sqrt(vA + vB)
    geweke = np.nanmax(np.abs(gz), axis=1)

    return dict(rhat=rhat, ess_bulk=ess_bulk, ess_tail=ess_tail,
                mcse=mcse, geweke_z=geweke)


def bh_fdr(p):
    """p-valores ajustados por Benjamini-Hochberg (sin dependencias)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    nq = len(q)
    if nq == 0:
        return out
    orden = np.argsort(q)
    escal = q[orden] * nq / np.arange(1, nq + 1)
    adj = np.minimum.accumulate(escal[::-1])[::-1]
    res = np.empty(nq)
    res[orden] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def resumen_posterior(X):
    """Resumen por parametro. X: (P, m, n) -> DataFrame sin etiquetas."""
    plano = X.reshape(X.shape[0], -1)
    qs = np.quantile(plano, [0.025, 0.25, 0.5, 0.75, 0.975], axis=1)
    return pd.DataFrame({
        "media": plano.mean(axis=1), "sd": plano.std(axis=1, ddof=1),
        "q025": qs[0], "q25": qs[1], "mediana": qs[2], "q75": qs[3], "q975": qs[4],
        "p_pos": (plano > 0).mean(axis=1),
    })


def preparar_bloque(cad: "CadenasM1", bloque):
    """apilar() + transponer a la convencion (P, m, n) de los diagnosticos."""
    X = cad.apilar(bloque)                    # (m, n, dim)
    return np.ascontiguousarray(np.moveaxis(X, 2, 0))   # (dim, m, n)


# =============================================================================
# 4. AUXILIARES DE INTERPRETACION
# =============================================================================
UMBRAL_RHAT, UMBRAL_ESS = 1.01, 400.0


def veredicto_convergencia(df):
    """Imprime el checklist de convergencia a partir del resumen (DataFrame
    con columnas rhat, ess_bulk, ess_tail, geweke_p_fdr). Devuelve bool."""
    n = len(df)
    rh = df["rhat"].to_numpy()
    n_rhat = int(np.nansum(rh > UMBRAL_RHAT))
    n_bulk = int(np.nansum(df["ess_bulk"].to_numpy() < UMBRAL_ESS))
    n_tail = int(np.nansum(df["ess_tail"].to_numpy() < UMBRAL_ESS))
    n_gew = int(np.nansum(df["geweke_p_fdr"].to_numpy() < 0.05))
    print("\n" + "=" * 72)
    print("VEREDICTO DE CONVERGENCIA (umbrales de Vehtari et al. 2021)")
    print("=" * 72)
    print(f"  Parametros monitoreados          : {n}")
    print(f"  Rhat  > {UMBRAL_RHAT}                     : {n_rhat:5d}  "
          f"{'OK' if n_rhat == 0 else '<- REVISAR (correr mas / mas cadenas)'}")
    print(f"  ESS bulk < {UMBRAL_ESS:.0f}                  : {n_bulk:5d}  "
          f"{'OK' if n_bulk == 0 else '<- precision baja en media/mediana'}")
    print(f"  ESS tail < {UMBRAL_ESS:.0f}                  : {n_tail:5d}  "
          f"{'OK' if n_tail == 0 else '<- precision baja en colas/IC'}")
    print(f"  Geweke significativo (FDR<0.05)  : {n_gew:5d}  "
          f"{'OK' if n_gew == 0 else '<- posible no-estacionariedad'}")
    ok = (n_rhat == 0) and (n_bulk == 0) and (n_tail == 0)
    print("-" * 72)
    print("  CONCLUSION: " + ("las cadenas son utilizables para inferencia."
          if ok else "NO concluir todavia: alargar la corrida, subir thin o "
          "anadir cadenas con otras semillas."))
    print("=" * 72)
    return ok
