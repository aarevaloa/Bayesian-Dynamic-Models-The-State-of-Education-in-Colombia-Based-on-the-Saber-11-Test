# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# @  MODELO 4 - ESTILO, PALETA Y UTILIDADES COMPARTIDAS
# _____________________________________________________________________________

import json, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib import cm as mcm

# ============================== CONFIGURACION ================================
# Unico lugar donde se declaran rutas. Modificar aqui, no en los modulos.
RAIZ = os.environ.get(
    "M4_RAIZ",
    "C:/Users/USUARIO/OneDrive - Universidad Nacional de Colombia/Tesis/"
    "Escritorio/Códigos/Modelos/Modelo 4")

RUTA_EXPORT  = os.path.join(RAIZ, "export_python")
RUTA_GEOJSON = os.path.join(RAIZ, "colombia_departamentos.geojson")
RUTA_GEOJSON_MUNI = os.path.join(RAIZ, "colombia_municipios.geojson")   # pendiente
DIR_SALIDA   = os.path.join(RAIZ, "resultados_M4")
DIR_CACHE    = os.path.join(RAIZ, "cache_M4")

DPI = 200
os.makedirs(DIR_SALIDA, exist_ok=True)
os.makedirs(DIR_CACHE, exist_ok=True)

# ============================== PALETA =======================================
# Colores por bloque jerarquico (identicos al script 15)
C = {"E": "#1f77b4", "M": "#e07b39", "D": "#2f9e69"}

# Rampas tematicas ya usadas
C_TRABAJO   = ["#f4a582", "#d6604d", "#b2182b", "#67001f"]   # rojos: intensidad laboral
C_VIOLENCIA = ["#8c2d43", "#c4562a", "#5e3c99"]
C_COLEGIO   = "#4a6fa5"
# cm.<nombre> desaparece en matplotlib >= 3.9; este shim mantiene el script 15
# funcionando tal cual y protege el codigo nuevo ante una actualizacion.
try:
    from matplotlib import colormaps as _cmaps
    def cmap(nombre):
        return _cmaps[nombre]
except ImportError:                                   # matplotlib < 3.5
    def cmap(nombre):
        return getattr(mcm, nombre)

def azules(n):
    """Rampa Blues usada en la escalera de educacion de la madre."""
    return cmap("Blues")(np.linspace(0.35, 0.95, n))

# Mapas de color: niveles secuenciales vs. cambios divergentes
CMAP_NIVEL   = cmap("magma_r")   # dispersion / nivel
CMAP_CAMBIO  = cmap("RdBu_r")    # Delta (rojo = mas, azul = menos)
CMAP_RESIDUO = cmap("PuOr_r")    # residuo territorial (evita chocar con CAMBIO)
GRIS_SIN_DATO = "#d4d4d4"

# Semestres: los periodos impares y pares son poblaciones distintas (ver 16_)
C_SEM = {"impar": "#9d7fb5", "par": "#1f4e6b"}
ET_SEM = {"impar": "Primer semestre (calendario B y rezagados)",
          "par":   "Segundo semestre (cohorte principal)"}

# ============================== RCPARAMS =====================================
def aplicar_estilo():
    plt.rcParams.update({"font.size": 9.5, "figure.facecolor": "white",
                         "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 10.5, "axes.titleweight": "bold",
                         "figure.titlesize": 13, "figure.titleweight": "bold"})

aplicar_estilo()

FIGS = []
def guardar(fig, nombre, cerrar=False):
    """Guarda en DIR_SALIDA y acumula en FIGS para el PDF consolidado."""
    fig.savefig(os.path.join(DIR_SALIDA, nombre), dpi=DPI, bbox_inches="tight")
    print("   ->", nombre)
    if cerrar:
        plt.close(fig)          # cerrada: no se acumula, no sirve para un PDF
    else:
        FIGS.append(fig)

# ============================== BLOQUES ======================================
LM = ["TASA_HOMICIDIOS", "PROP_HOMICIDIOS_URBANO", "TASA_VICTIMAS",
      "TASA_MATRICULA_ES", "IES_CON_OFERTA", "SGP_EDUC_PC", "PROP_NBI",
      "COMP_INASISTENCIA_ESCOLAR", "PROP_URBANA", "LOG_POBLACION",
      "SGP_EDUC_PC_MISSING"]
LD = ["PIB_CONSTANTE", "CREC_PIB_REAL", "TASA_DESEMPLEO", "TASA_SUBOCUPACION",
      "IPM", "CREC_PIB_REAL_MISSING", "TASA_DESEMPLEO_MISSING",
      "TASA_SUBOCUPACION_MISSING"]

_BLOQUE_DE = {}

def bloque(nom):
    if _BLOQUE_DE:
        return _BLOQUE_DE.get(nom, "E")
    return "M" if nom in LM else ("D" if nom in LD else "E")

# ====================== POSTERIOR DEL ESTADO DINAMICO ========================
class Posterior:
    """Acceso a theta.bin: resumenes, sendas y contrastes en unidades de puntaje.

    theta tiene forma (p, Tt+1, S); el indice temporal 0 es el estado inicial y
    NO se usa en los resumenes sustantivos (solo t = 1..Tt).
    """

    def __init__(self, theta, labels, meta):
        self.theta = theta
        self.labels = list(labels)
        self.meta = meta
        self.p, self.Tt1, self.S = theta.shape
        self.Tt = self.Tt1 - 1
        self.ix = {n: i for i, n in enumerate(self.labels)}
        self.tx = np.arange(1, self.Tt + 1)
        self.idx_par = self.tx[self.tx % 2 == 0]
        self.bar_par = theta[:, self.idx_par, :].mean(axis=1)     # (p, S)
        self.bar_16 = theta[:, 1:, :].mean(axis=1)
        self.bar = self.bar_par            # el default es la cohorte principal

    def _bar(self, pares=True):
        return self.bar_par if pares else self.bar_16

    # ---- efecto promedio temporal -----------------------------------------
    def resumen(self, nom, pares=True):
        b = self._bar(pares)[self.ix[nom]]
        return dict(nombre=nom, mediana=float(np.median(b)),
                    media=float(b.mean()), sd=float(b.std(ddof=1)),
                    lo=float(np.quantile(b, .025)), hi=float(np.quantile(b, .975)),
                    p_neg=float(np.mean(b < 0)), p_pos=float(np.mean(b > 0)))

    # ---- senda dinamica ----------------------------------------------------
    def senda(self, nom, qq=(.05, .5, .95)):
        return np.quantile(self.theta[self.ix[nom], 1:, :], qq, axis=1)  # 3 x Tt

    def draws(self, nom, pares=True):
        """Draws del efecto promedio temporal (S,). Por defecto solo pares."""
        return self._bar(pares)[self.ix[nom]]

    def draws_t(self, nom):
        """Draws de la senda completa (Tt, S)."""
        return self.theta[self.ix[nom], 1:, :]

    # ---- contrastes posteriores -------------------------------------------
    def contraste(self, a, b, nombre=None, pares=True):
        """Posterior de theta_a - theta_b (promedio temporal). Si b es None,
        contrasta contra la categoria de referencia (efecto crudo)."""
        B = self._bar(pares)
        d = B[self.ix[a]] - (0.0 if b is None else B[self.ix[b]])
        return dict(nombre=nombre or f"{a} - {b}", mediana=float(np.median(d)),
                    lo=float(np.quantile(d, .025)), hi=float(np.quantile(d, .975)),
                    p_neg=float(np.mean(d < 0)), p_pos=float(np.mean(d > 0)))

    # ---- cambio temporal del efecto ---------------------------------------
    def cambio(self, nom, t0=None, t1=None, pares=True):
        """Posterior de theta_{l,t1} - theta_{l,t0}. Por defecto compara el
        primer y el ultimo periodo PAR, para no mezclar semestres."""
        idx = np.arange(1, self.Tt + 1)
        if pares:
            idx = idx[idx % 2 == 0]
        t0 = idx[0] if t0 is None else t0
        t1 = idx[-1] if t1 is None else t1
        d = self.theta[self.ix[nom], t1, :] - self.theta[self.ix[nom], t0, :]
        return dict(nombre=nom, t0=int(t0), t1=int(t1),
                    mediana=float(np.median(d)),
                    lo=float(np.quantile(d, .025)), hi=float(np.quantile(d, .975)),
                    p_neg=float(np.mean(d < 0)), p_pos=float(np.mean(d > 0)))

    # ---- pendiente posterior de la senda -----------------------------------
    def pendiente(self, nom, pares=True):
        """Pendiente OLS de la senda por draw (puntos por periodo par)."""
        idx = np.arange(1, self.Tt + 1)
        if pares:
            idx = idx[idx % 2 == 0]
        Y = self.theta[self.ix[nom], idx, :]                    # n_idx x S
        # la abscisa es el ORDINAL del periodo dentro de la serie usada, de modo
        # que la pendiente sale por periodo par (no por unidad del indice t, que
        # avanza de dos en dos y daria la mitad)
        x = np.arange(len(idx), dtype=float); x -= x.mean()
        return (x[:, None] * Y).sum(axis=0) / (x ** 2).sum()


_POST = None

def cargar_posterior(ruta_export=None):
    """Lee theta.bin + meta.json y devuelve un objeto Posterior (y lo fija
    como posterior por defecto de forest()/sendas())."""
    global _POST
    ruta_export = ruta_export or RUTA_EXPORT
    with open(os.path.join(ruta_export, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    p, Tt, S = meta["p"], meta["Tt"], meta["S"]
    th = np.fromfile(os.path.join(ruta_export, "theta.bin")).reshape(
        (p, Tt + 1, S), order="F")
    _POST = Posterior(th, meta["theta_labels"], meta)
    global _BLOQUE_DE
    _BLOQUE_DE = {}
    for b in ("E", "M", "D"):
        for i in meta.get(f"idx_{b}", []):
            _BLOQUE_DE[meta["theta_labels"][i - 1]] = b      # meta va en base 1
    return _POST

def posterior():
    if _POST is None:
        raise RuntimeError("Llama primero a estilo_M4.cargar_posterior()")
    return _POST

# ============================== FOREST / SENDAS ==============================
def forest(ax, noms, etiquetas, colores, titulo, anot_p="neg", post=None):
    """Forest plot del efecto promedio temporal. Firma identica al script 15."""
    P = post or posterior()
    rs = [P.resumen(n) for n in noms]
    ys = np.arange(len(rs))[::-1]
    for y, r, c in zip(ys, rs, colores):
        ax.plot([r["lo"], r["hi"]], [y, y], color=c, lw=2.2, solid_capstyle="round")
        ax.plot(r["mediana"], y, "o", color=c, ms=6)
        pp = r["p_neg"] if anot_p == "neg" else r["p_pos"]
        ax.annotate(f'{r["mediana"]:+.1f}  [P({"β<0" if anot_p=="neg" else "β>0"})={pp:.2f}]',
                    (r["hi"], y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color="0.25")
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(ys); ax.set_yticklabels(etiquetas, fontsize=9)
    ax.set_xlabel("efecto sobre el puntaje global (puntos) — promedio temporal, IC 95%")
    ax.set_title(titulo)
    lo = min(r["lo"] for r in rs); hi = max(r["hi"] for r in rs)
    ax.set_xlim(lo - 0.08 * (hi - lo), hi + 0.42 * (hi - lo))
    return rs


def forest_dict(ax, rs, etiquetas, colores, titulo, anot_p="neg", xlab=None):
    """Igual que forest() pero recibe una lista de dicts ya calculados
    (contrastes, cambios, indices...). Misma estetica."""
    ys = np.arange(len(rs))[::-1]
    for y, r, c in zip(ys, rs, colores):
        ax.plot([r["lo"], r["hi"]], [y, y], color=c, lw=2.2, solid_capstyle="round")
        ax.plot(r["mediana"], y, "o", color=c, ms=6)
        pp = r["p_neg"] if anot_p == "neg" else r["p_pos"]
        ax.annotate(f'{r["mediana"]:+.1f}  [P({"<0" if anot_p=="neg" else ">0"})={pp:.2f}]',
                    (r["hi"], y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color="0.25")
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(ys); ax.set_yticklabels(etiquetas, fontsize=9)
    ax.set_xlabel(xlab or "puntos del puntaje global, IC 95%")
    ax.set_title(titulo)
    lo = min(r["lo"] for r in rs); hi = max(r["hi"] for r in rs)
    ax.set_xlim(lo - 0.08 * (hi - lo), hi + 0.42 * (hi - lo))
    return rs


def sendas(ax, noms, etiquetas, colores, titulo, post=None, banda=(.05, .5, .95)):
    """Sendas dinamicas con banda de credibilidad. Firma identica al script 15."""
    P = post or posterior()
    for n, e, c in zip(noms, etiquetas, colores):
        q = P.senda(n, banda)
        ax.fill_between(P.tx, q[0], q[2], color=c, alpha=0.14)
        ax.plot(P.tx, q[1], color=c, lw=2, label=e)
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xlabel("periodo t"); ax.set_ylabel("efecto (puntos)")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title(titulo)


def marcar_semestres(ax, Tt=16, alpha=0.06):
    """Sombrea los periodos impares (primer semestre) para recordar que la
    serie alterna entre dos poblaciones distintas."""
    for t in range(1, Tt + 1, 2):
        ax.axvspan(t - 0.5, t + 0.5, color=C_SEM["impar"], alpha=alpha, lw=0, zorder=0)

# ============================== MAPAS ========================================
class Geo:
    """Poligonos y centroides listos para PolyCollection."""

    def __init__(self, polis, cods, centro, xl, yl, orden):
        self.POLIS, self.POLI_COD, self.CENTRO = polis, cods, centro
        self.XL, self.YL = xl, yl
        self.K_DE_COD = {c: i for i, c in enumerate(orden)}
        self.orden = list(orden)


def cargar_geojson(ruta=None, campo="DPTO_CCDGO", orden=None, clave=int):
    """Lee un geojson de poligonos y devuelve un objeto Geo.
    `orden` es la lista de codigos en el orden de las filas de los datos
    (depto_ids para departamentos; codigos DIVIPOLA para municipios).
    `clave` normaliza los codigos antes de comparar: int para departamentos,
    str para municipios ('05001' del MGN frente a 5001 del modelo)."""
    ruta = ruta or RUTA_GEOJSON
    with open(ruta, encoding="utf-8") as fh:
        gj = json.load(fh)
    POLIS, POLI_COD, CENTRO = [], [], {}

    def _interior(an):
        ys = np.unique(an[:, 1])
        if ys.size < 2:
            return an.mean(axis=0)
        k2 = ys.size // 2
        y = float((ys[k2 - 1] + ys[k2]) / 2.0)
        x0, y0 = an[:-1, 0], an[:-1, 1]
        x1, y1 = an[1:, 0], an[1:, 1]
        cruza = (y0 <= y) != (y1 <= y)
        if not cruza.any():
            return an.mean(axis=0)
        xs = np.sort(x0[cruza] + (y - y0[cruza]) *
                     (x1[cruza] - x0[cruza]) / (y1[cruza] - y0[cruza]))
        if len(xs) < 2:
            return an.mean(axis=0)
        a, b = xs[0::2], xs[1::2]                 # tramos interiores
        n = min(len(a), len(b))
        k = int(np.argmax(b[:n] - a[:n]))         # el tramo interior mas ancho
        return np.array([(a[k] + b[k]) / 2.0, y])
    for f in gj["features"]:
        cod = clave(f["properties"][campo])
        partes = ([f["geometry"]["coordinates"]]
                  if f["geometry"]["type"] == "Polygon"
                  else f["geometry"]["coordinates"])
        mayor, amax = None, -1
        for parte in partes:
            an = np.asarray(parte[0], float)
            POLIS.append(an); POLI_COD.append(cod)
            xs, ys = an[:, 0], an[:, 1]
            a = 0.5 * abs(float(np.dot(xs, np.roll(ys, -1)) -
                                np.dot(ys, np.roll(xs, -1))))
            if a > amax:
                mayor, amax = an, a
        CENTRO[cod] = _interior(mayor)
    POLI_COD = np.asarray(POLI_COD)
    tt = np.vstack(POLIS)
    XL = (tt[:, 0].min() - .3, tt[:, 0].max() + .3)
    YL = (tt[:, 1].min() - .3, tt[:, 1].max() + .3)
    return Geo(POLIS, POLI_COD, CENTRO, XL, YL, [clave(c) for c in orden])


def colores_mapa(vals, cmap, norm, geo):
    """Color por poligono; gris si el codigo no esta en los datos o es NaN."""
    out = []
    for c in geo.POLI_COD:
        i = geo.K_DE_COD.get(c.item() if hasattr(c, "item") else c)
        v = np.nan if i is None else vals[i]
        out.append(GRIS_SIN_DATO if (i is None or not np.isfinite(v))
                   else cmap(norm(v)))
    return out


def eje_mapa(ax, vals, cmap, norm, geo=None, etiq=True, lw=0.4):
    """Dibuja un coropletico. Firma compatible con el script 15."""
    geo = geo or _GEO_DEF
    pc = PolyCollection(geo.POLIS, facecolors=colores_mapa(vals, cmap, norm, geo),
                        edgecolors="white", linewidths=lw)
    ax.add_collection(pc)
    ax.set_xlim(*geo.XL); ax.set_ylim(*geo.YL)
    ax.set_aspect("equal"); ax.axis("off"); ax.grid(False)
    if etiq:
        for cod, (cx, cy) in geo.CENTRO.items():
            ax.text(cx, cy, str(cod), ha="center", va="center",
                    fontsize=5.5, color="0.15")
    return pc


_GEO_DEF = None
def fijar_geo(geo):
    global _GEO_DEF
    _GEO_DEF = geo
    return geo


def norma_divergente(vals):
    """TwoSlopeNorm simetrica alrededor de 0, robusta a NaN."""
    a = float(np.nanmax(np.abs(vals)))
    return TwoSlopeNorm(0.0, -a, a)


def norma_robusta(vals, pmin=2, pmax=98):
    lo, hi = np.nanpercentile(vals, [pmin, pmax])
    return Normalize(lo, hi)

# ============================== CACHE ========================================
def cargar_cache(dir_cache=None, con_varianzas=True):
    """Contenido de prep_M4.npz, nivel_M4.npz y, si existe, varianzas_M4.npz.

    OJO con las dos versiones de la dispersion departamental:
      sd_kt_plugin  (prep)      = (1/E[phi_kt])^1/2, estimacion de enchufe SIN
                                  incertidumbre. Por Jensen no es la mediana de
                                  la posterior. No usar en figuras.
      sd_kt_med/lo/hi (varianzas) = cuantiles de la posterior de phi^-1/2
                                  reconstruida por el modulo 17. Es la buena.
    """
    dir_cache = dir_cache or DIR_CACHE
    D, origen = {}, {}

    def _absorber(ruta, f):
        with np.load(ruta, allow_pickle=True) as z:
            for k in z.files:
                if k in origen:
                    raise KeyError(
                        f"clave '{k}' repetida: la produce {origen[k]} y tambien "
                        f"{f}. Al fusionar, la segunda pisa a la primera y el "
                        f"consumidor recibe el objeto equivocado. Renombrala en "
                        f"el modulo que la genera.")
                D[k] = z[k]
                origen[k] = f

    for f in ("prep_M4.npz", "nivel_M4.npz"):
        ruta = os.path.join(dir_cache, f)
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"Falta {ruta}. Ejecuta 16_preparacion_M4.py")
        _absorber(ruta, f)
    if con_varianzas:
        ruta = os.path.join(dir_cache, "varianzas_M4.npz")
        if os.path.exists(ruta):
            _absorber(ruta, "varianzas_M4.npz")
        else:
            print("  AVISO: falta varianzas_M4.npz (modulo 17). La dispersion\n"
                  "         departamental quedara sin intervalos de credibilidad.")
    return D


def nivel_esperado(Xbar, theta_t, bloques=None, idx_bloque=None):
    """Posterior del nivel esperado por celda territorial.

    Xbar      : (p, n_terr, Tt) medias de covariables (NaN en celdas vacias)
    theta_t   : (p, Tt, S) estado dinamico en t = 1..Tt
    bloques   : None (todo) o clave de idx_bloque
    Devuelve  : (n_terr, Tt, S)
    """
    p, nter, Tt = Xbar.shape
    S = theta_t.shape[2]
    ii = slice(None) if bloques is None else idx_bloque[bloques]
    X = np.nan_to_num(Xbar[ii], nan=0.0)
    th = theta_t[ii]
    out = np.empty((nter, Tt, S), dtype=np.float64)
    for t in range(Tt):
        out[:, t, :] = X[:, :, t].T @ th[:, t, :]
    return out


MULT_MUNI = [
    ("TASA_HOMICIDIOS",           10,        "+10 homicidios por 100 mil hab."),
    ("TASA_VICTIMAS",             10000,     "+10 pp de la población como víctimas"),
    ("PROP_HOMICIDIOS_URBANO",    0.10,      "+10 pp de homicidios urbanos"),
    ("PROP_NBI",                  10,        "+10 pp de necesidades básicas insatisfechas"),
    ("COMP_INASISTENCIA_ESCOLAR", 1,         "+1 punto de inasistencia escolar"),
    ("PROP_URBANA",               0.10,      "+10 pp de población urbana"),
    ("SGP_EDUC_PC",               100000,    "+100 mil pesos de SGP educativo per cápita"),
    ("TASA_MATRICULA_ES",         10,        "+10 puntos de matrícula en educación superior"),
    ("IES_CON_OFERTA",            10,        "+10 instituciones con oferta"),
    ("LOG_POBLACION",             np.log(2), "duplicar la población"),
]
MULT_DEPTO = [
    ("IPM",               10,    "+10 puntos del índice de pobreza multidimensional"),
    ("TASA_DESEMPLEO",    1,     "+1 pp de desempleo"),
    ("TASA_SUBOCUPACION", 1,     "+1 pp de subocupación"),
    ("CREC_PIB_REAL",     1,     "+1 pp de crecimiento del PIB"),
    ("PIB_CONSTANTE",     10000, "+10 billones de PIB departamental"),
]
