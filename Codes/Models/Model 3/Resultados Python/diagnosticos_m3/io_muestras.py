"""
io_muestras.py — Lectura de muestras MCMC y datos observados del Modelo 3.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Lectura de metadatos.rds
# ---------------------------------------------------------------------------


def _leer_rds_si_posible(ruta_rds: str) -> Optional[dict]:
    """Intenta leer metadatos.rds con pyreadr. Devuelve None si no se puede."""
    try:
        import pyreadr  # type: ignore
    except Exception:
        return None
    try:
        res = pyreadr.read_r(ruta_rds)
        if None in res:
            return None
    except Exception:
        return None
    return None  # las listas RDS complejas no se mapean bien; usar JSON


@dataclass
class MetaM3:
    """Metadatos dimensionales del Modelo 3."""
    p: int
    Tt: int
    m: int
    n_k: np.ndarray
    max_nk: int
    n_save: int
    p_E: int
    p_M: int
    p_D: int
    theta_labels: list
    depto_ids: list
    tiempo_ids: list
    # índices 0-based de cada bloque dentro del vector theta (p,)
    idx_E: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    idx_M: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    idx_D: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    def __post_init__(self):
        # Reconstruir índices de bloque a partir de p_E/p_M/p_D.
        # theta = [intercepto, E(p_E), M(p_M), D(p_D)]  -> 0-based
        self.idx_E = np.arange(1, 1 + self.p_E)
        self.idx_M = np.arange(1 + self.p_E, 1 + self.p_E + self.p_M)
        self.idx_D = np.arange(1 + self.p_E + self.p_M, self.p)

    @property
    def bloques(self):
        """Diccionario homólogo a `bloques` de R (idx 0-based)."""
        return {
            "E": {"nombre": "Individual",    "dim": self.p_E, "idx": self.idx_E},
            "M": {"nombre": "Municipal",     "dim": self.p_M, "idx": self.idx_M},
            "D": {"nombre": "Departamental", "dim": self.p_D, "idx": self.idx_D},
        }

    def etiquetas_bloque(self, B: str) -> list:
        idx = self.bloques[B]["idx"]
        return [self.theta_labels[i] for i in idx]


def cargar_metadatos(dir_salida: str) -> MetaM3:
    """
    Carga metadatos desde (en orden de preferencia):
      1. metadatos.json  (volcado explícito desde R)
      2. datos_export/export_meta.json
    Lanza FileNotFoundError con instrucciones si no encuentra ninguno.
    """
    ruta_json = os.path.join(dir_salida, "metadatos.json")
    ruta_export = os.path.join(dir_salida, "datos_export", "export_meta.json")

    src = None
    if os.path.exists(ruta_json):
        with open(ruta_json, "r", encoding="utf-8") as fh:
            src = json.load(fh)
    elif os.path.exists(ruta_export):
        with open(ruta_export, "r", encoding="utf-8") as fh:
            src = json.load(fh)
    else:
        raise FileNotFoundError(
            "No se encontró metadatos.json ni datos_export/export_meta.json.\n"
            "Exporte los metadatos desde R con:\n"
            "  jsonlite::write_json(metadatos, "
            "file.path(dir_salida, 'metadatos.json'), auto_unbox=TRUE, digits=12)\n"
            "o ejecute el exportador 10_exportar_datos_para_python.R."
        )

    def _get(k, default=None):
        v = src.get(k, default)
        return v

    _faltan = [k for k in ("p", "Tt", "m", "n_k", "p_E", "p_M", "p_D",
                           "theta_labels") if src.get(k) is None]
    _sin_nsave = (src.get("save_idx") is None) and (src.get("n_save") is None)
    if _faltan or _sin_nsave:
        cuales = ", ".join(_faltan + (["n_save/save_idx"] if _sin_nsave else []))
        raise ValueError(
            f"Los metadatos cargados no incluyen: {cuales}.\n"
            "Probablemente se leyó datos_export/export_meta.json, que NO trae "
            "las dimensiones de bloque.\n"
            "Genera metadatos.json completo desde metadatos.rds (una vez, en R):\n"
            "  jsonlite::write_json(readRDS('metadatos.rds'), 'metadatos.json', "
            "auto_unbox=TRUE, digits=12)\n"
            "o usa el módulo preparar_metadatos.asegurar_metadatos_json()."
        )

    # n_save puede llamarse save_idx en metadatos.rds
    n_save = _get("save_idx", _get("n_save", None))
    if isinstance(n_save, list):
        n_save = int(n_save[0])

    n_k = np.asarray(_get("n_k"), dtype=int).ravel()
    max_nk = int(_get("max_nk", n_k.max()))

    return MetaM3(
        p=int(_scalar(_get("p"))),
        Tt=int(_scalar(_get("Tt"))),
        m=int(_scalar(_get("m"))),
        n_k=n_k,
        max_nk=max_nk,
        n_save=int(n_save) if n_save is not None else 0,
        p_E=int(_scalar(_get("p_E"))),
        p_M=int(_scalar(_get("p_M"))),
        p_D=int(_scalar(_get("p_D"))),
        theta_labels=list(_get("theta_labels")),
        depto_ids=[str(x) for x in _get("depto_ids", [])],
        tiempo_ids=[str(x) for x in _get("tiempo_ids", [])],
    )


def _scalar(v):
    """R serializa escalares a veces como lista de longitud 1."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return v[0]
    return v


# ---------------------------------------------------------------------------
# Lectura de los .bin de muestras
# ---------------------------------------------------------------------------

def _leer_bin(ruta: str, n_por_muestra: int, desde: int, hasta: int) -> np.ndarray:
    """Lee [desde, hasta] (1-based inclusivo) muestras de un .bin de doubles."""
    n_leer = hasta - desde + 1
    offset_bytes = (desde - 1) * n_por_muestra * 8
    count = n_leer * n_por_muestra
    with open(ruta, "rb") as fh:
        fh.seek(offset_bytes)
        raw = np.frombuffer(fh.read(count * 8), dtype="<f8", count=count)
    if raw.size != count:
        raise IOError(
            f"{os.path.basename(ruta)}: se esperaban {count} doubles, "
            f"se leyeron {raw.size}. ¿n_save incorrecto o archivo truncado?"
        )
    return raw


def leer_muestras(dir_salida: str, meta: MetaM3, que: str = "theta",
                  desde: int = 1, hasta: Optional[int] = None) -> np.ndarray:
    """
    Réplica de leer_muestras() de R. Devuelve arrays con la muestra como
    ÚLTIMA dimensión (excepto lambda2, que sale como (n_save, 3) igual que R).

    Parámetros
    ----------
    que : 'theta' | 'lambda2' | 'v' | 's' | 'tau2_E' | 'tau2_M' | 'tau2_D'
    desde, hasta : rango 1-based inclusivo. hasta=None -> meta.n_save
    """
    if hasta is None:
        hasta = meta.n_save
    n_leer = hasta - desde + 1

    if que == "theta":
        npm = meta.p * (meta.Tt + 1)
        raw = _leer_bin(os.path.join(dir_salida, "theta.bin"), npm, desde, hasta)
        # R: array(raw, dim=c(p, Tt+1, n_leer)) -> column-major
        arr = raw.reshape((meta.p, meta.Tt + 1, n_leer), order="F")
        return arr

    if que == "lambda2":
        npm = 3
        raw = _leer_bin(os.path.join(dir_salida, "lambda2.bin"), npm, desde, hasta)
        # R: matrix(raw, nrow=n_leer, ncol=3, byrow=TRUE) -> cada muestra es
        # un bloque contiguo de 3; reshape C-order da (n_leer, 3) directo.
        mat = raw.reshape((n_leer, 3), order="C")
        return mat  # columnas: E, M, D

    if que == "v":
        npm = meta.max_nk * meta.m
        raw = _leer_bin(os.path.join(dir_salida, "v.bin"), npm, desde, hasta)
        return raw.reshape((meta.max_nk, meta.m, n_leer), order="F")

    if que == "s":
        npm = meta.m * meta.Tt
        raw = _leer_bin(os.path.join(dir_salida, "s.bin"), npm, desde, hasta)
        return raw.reshape((meta.m, meta.Tt, n_leer), order="F")

    if que.startswith("tau2_"):
        B = que.split("_", 1)[1]
        p_B = meta.bloques[B]["dim"]
        npm = p_B * meta.Tt
        raw = _leer_bin(os.path.join(dir_salida, f"tau2_{B}.bin"),
                        npm, desde, hasta)
        return raw.reshape((p_B, meta.Tt, n_leer), order="F")

    raise ValueError(
        f"Tipo '{que}' no reconocido. Use: theta, lambda2, v, s, tau2_E/M/D"
    )


# ---------------------------------------------------------------------------
# Contenedor de todas las muestras cargadas
# ---------------------------------------------------------------------------

@dataclass
class MuestrasM3:
    meta: MetaM3
    theta: np.ndarray      # (p, Tt+1, S)
    lambda2: np.ndarray    # (S, 3)
    s: np.ndarray          # (m, Tt, S)
    v: np.ndarray          # (max_nk, m, S)
    tau2_E: np.ndarray     # (p_E, Tt, S)
    tau2_M: np.ndarray     # (p_M, Tt, S)
    tau2_D: np.ndarray     # (p_D, Tt, S)
    n_save: int

    def tau2(self, B: str) -> np.ndarray:
        return getattr(self, f"tau2_{B}")


def cargar_todo(dir_salida: str, n_cargar: Optional[int] = 2000,
                verbose: bool = True) -> MuestrasM3:
    """
    Carga las últimas `n_cargar` muestras de todos los bloques (o todas si
    n_cargar=None). Homólogo al bloque 0 del script 9_diagnosticos_M3.R.
    """
    meta = cargar_metadatos(dir_salida)
    if meta.n_save <= 0:
        raise ValueError("meta.n_save == 0. ¿Metadatos correctos?")

    if n_cargar is None:
        n_cargar = meta.n_save
    n_cargar = min(int(n_cargar), meta.n_save)
    desde = meta.n_save - n_cargar + 1

    if verbose:
        print(f"Cargando muestras {desde} a {meta.n_save} "
              f"({n_cargar} muestras) desde {os.path.basename(dir_salida)}/")

    theta = leer_muestras(dir_salida, meta, "theta", desde=desde)
    lambda2 = leer_muestras(dir_salida, meta, "lambda2", desde=desde)
    s = leer_muestras(dir_salida, meta, "s", desde=desde)
    v = leer_muestras(dir_salida, meta, "v", desde=desde)
    tau2_E = leer_muestras(dir_salida, meta, "tau2_E", desde=desde)
    tau2_M = leer_muestras(dir_salida, meta, "tau2_M", desde=desde)
    tau2_D = leer_muestras(dir_salida, meta, "tau2_D", desde=desde)

    if verbose:
        nbytes = sum(a.nbytes for a in
                     (theta, lambda2, s, v, tau2_E, tau2_M, tau2_D))
        print(f"RAM usada por muestras: {nbytes / 1e6:.0f} MB")
        print(f"  theta {theta.shape}  s {s.shape}  v {v.shape}")
        print(f"  tau2_E {tau2_E.shape}  tau2_M {tau2_M.shape}  "
              f"tau2_D {tau2_D.shape}")

    return MuestrasM3(meta=meta, theta=theta, lambda2=lambda2, s=s, v=v,
                      tau2_E=tau2_E, tau2_M=tau2_M, tau2_D=tau2_D,
                      n_save=n_cargar)


# ---------------------------------------------------------------------------
# Lectura de datos observados exportados (para PPC / residuos / WAIC)
# ---------------------------------------------------------------------------

@dataclass
class DatosObservados:
    """Datos observados aplanados en orden t->k->j (idéntico a R)."""
    y: np.ndarray            # (N_total,)
    F: np.ndarray            # (N_total, p)  matriz de diseño apilada
    k_all: np.ndarray        # (N_total,) departamento 1-based
    t_all: np.ndarray        # (N_total,) periodo 1-based
    j_all: np.ndarray        # (N_total,) municipio 1-based (dentro de depto)
    cell_offset: np.ndarray  # inicio de cada celda en el flujo aplanado
    cell_t: np.ndarray
    cell_k: np.ndarray
    cell_j: np.ndarray
    cell_n: np.ndarray
    N_total: int
    p: int


def cargar_datos_observados(dir_salida: str,
                            meta: MetaM3) -> Optional[DatosObservados]:
    """
    Carga y_obs.bin + F_obs.bin + export_meta.json. Devuelve None si no
    existen (el pipeline entonces omite las secciones predictivas).
    """
    d = os.path.join(dir_salida, "datos_export")
    ruta_meta = os.path.join(d, "export_meta.json")
    ruta_y = os.path.join(d, "y_obs.bin")
    ruta_F = os.path.join(d, "F_obs.bin")

    if not (os.path.exists(ruta_meta) and os.path.exists(ruta_y)
            and os.path.exists(ruta_F)):
        return None

    with open(ruta_meta, "r", encoding="utf-8") as fh:
        em = json.load(fh)

    N_total = int(_scalar(em["N_total"]))
    p = int(_scalar(em["p"]))

    y = np.fromfile(ruta_y, dtype="<f8")
    if y.size != N_total:
        raise IOError(f"y_obs.bin: {y.size} != N_total {N_total}")

    F = np.fromfile(ruta_F, dtype="<f8")
    if F.size != N_total * p:
        raise IOError(f"F_obs.bin: {F.size} != N_total*p {N_total * p}")
    # exportado row-major por celda; concatenado da (N_total, p) en C-order
    F = F.reshape((N_total, p), order="C")

    cell_t = np.asarray(em["cell_t"], dtype=int).ravel()
    cell_k = np.asarray(em["cell_k"], dtype=int).ravel()
    cell_j = np.asarray(em["cell_j"], dtype=int).ravel()
    cell_n = np.asarray(em["cell_n"], dtype=int).ravel()
    cell_offset = np.asarray(em["cell_offset"], dtype=int).ravel()

    # Expandir a vectores por-observación (repeat según cell_n)
    k_all = np.repeat(cell_k, cell_n)
    t_all = np.repeat(cell_t, cell_n)
    j_all = np.repeat(cell_j, cell_n)

    return DatosObservados(
        y=y, F=F, k_all=k_all, t_all=t_all, j_all=j_all,
        cell_offset=cell_offset, cell_t=cell_t, cell_k=cell_k,
        cell_j=cell_j, cell_n=cell_n, N_total=N_total, p=p,
    )
