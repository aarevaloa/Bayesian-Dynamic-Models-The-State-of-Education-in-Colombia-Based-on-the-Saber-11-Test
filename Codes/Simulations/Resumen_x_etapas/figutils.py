"""
@author: aarevaloa
de la variabilidad residual.
figutils.py  —  Utilidades comunes para las figuras del capitulo de resultados
================================================================================
Paleta oficial, estilo de ejes, carga de la tabla y guardado en PNG+PDF.
Importado por cada parteN_*.py y por el script maestro.

Paleta (logica frio=normal / calido=t, intensidad=estatico/dinamico):
    M1  nautico/petroleo  (normal estatico)
    M2  terracota          (t estatico)
    M3  nautico oscuro     (normal dinamico)
    M4  terracota oscuro   (t dinamico)
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")          # backend sin ventana, para guardar a disco
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
#  Paleta y colores de apoyo
# -----------------------------------------------------------------------------
PAL = {
    "M1": "#1F4E5F",   # nautico/petroleo  (normal estatico)
    "M2": "#A8552E",   # terracota          (t estatico)
    "M3": "#13323D",   # nautico oscuro     (normal dinamico)
    "M4": "#73371B",   # terracota oscuro   (t dinamico)
}
COLOR_REF = "#9aa0a6"   # lineas de referencia (0.95, 1.0)
BANDA_A   = "#f4f6f7"   # sombreado de familia (claro)
BANDA_B   = "#e9eef0"   # sombreado de familia (alterno)
CAJA_FC   = "#f7f6f4"   # relleno de cajas de texto
CAJA_EC   = "#d8cfc8"   # borde de cajas de texto
GRIS_TXT  = "#555555"
GRIS_SUB  = "#777777"

# etiquetas estandar de modelo
ETQ_MODELO = {"M1": "M₁ (normal)", "M2": "M₂ (t)",
              "M3": "M₃ (normal, din.)", "M4": "M₄ (t, din.)"}
ETQ_MODELO_DIN = {"M1": "M₁ (est., normal)", "M2": "M₂ (est., t)",
                  "M3": "M₃ (din., normal)", "M4": "M₄ (din., t)"}


# -----------------------------------------------------------------------------
#  Estilo global (se aplica una vez al importar)
# -----------------------------------------------------------------------------
def aplicar_estilo():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#333333",
        "axes.edgecolor": "#cccccc",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

aplicar_estilo()


def limpiar_ejes(ax):
    """Quita los spines superior y derecho; suaviza los restantes."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cccccc")
    ax.set_axisbelow(True)


def grid_y(ax, which="major", alpha=0.18):
    ax.grid(axis="y", which=which, alpha=alpha)
    ax.set_axisbelow(True)


# -----------------------------------------------------------------------------
#  Carga de la tabla de resultados
# -----------------------------------------------------------------------------
def cargar_tabla(ruta_csv):
    """Lee el CSV y devuelve un dict indexado por (escenario, modelo) -> fila."""
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(
            f"No encuentro el CSV de resultados en: {ruta_csv}\n"
            f"Pasa la ruta correcta a las funciones de figura.")
    filas = []
    with open(ruta_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            filas.append(row)
    D = {(r["escenario"], r["modelo"]): r for r in filas}
    return D


def val(D, esc, mod, col):
    """Acceso seguro a una celda; devuelve float o None (para 'NA')."""
    r = D.get((esc, mod))
    if r is None:
        return None
    v = r.get(col, "NA")
    return None if v in ("NA", "", None) else float(v)


# -----------------------------------------------------------------------------
#  Guardado en PNG (alta resolucion) + PDF (vectorial)
# -----------------------------------------------------------------------------
def guardar(fig, nombre, dir_figuras="figuras", formatos=("png", "pdf"),
            dpi=300, verbose=True):
    os.makedirs(dir_figuras, exist_ok=True)
    rutas = []
    for fmt in formatos:
        ruta = os.path.join(dir_figuras, f"{nombre}.{fmt}")
        fig.savefig(ruta, dpi=dpi, bbox_inches="tight")
        rutas.append(ruta)
    if verbose:
        print(f"[figutils] guardada: {nombre}  ({', '.join(formatos)})")
    plt.close(fig)
    return rutas
