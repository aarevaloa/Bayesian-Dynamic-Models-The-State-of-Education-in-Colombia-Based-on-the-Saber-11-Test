"""
@author: aarevaloa
parte1_estructurales.py  —  Parte 1: Familias estructurales
============================================================
Panel de 2 sub-figuras principales (RMSE, cobertura) + 1 figura independiente
(cociente de varianza) sobre los escenarios estructurales S01-S10.

Uso:
    from parte1_estructurales import figura_parte1
    figura_parte1("resultados_estudio/tabla_resumen.csv") 
    # -> guarda figuras/p1_panel.* y figuras/p1_cociente_varianza.*
"""

import numpy as np
import figutils as fu


ESCEN = ["S01", "S02", "S04", "S05", "S06", "S07", "S08", "S09", "S10"]
BANDAS = [(-0.5, 1.5, fu.BANDA_A), (1.5, 3.5, fu.BANDA_B),
          (3.5, 6.5, fu.BANDA_A), (6.5, 8.5, fu.BANDA_B)]
ETIQ_FAM = [(0.5, "Tamaño\nmuestral"), (2.5, "Desbalance\npoblacional"),
            (5.0, "Niveles de\nvarianza"), (7.5, "Colinealidad")]


def _stripes(ax):
    for lo, hi, col in BANDAS:
        ax.axvspan(lo, hi, color=col, zorder=0)


def _subplot(ax, D, col, titulo, ylab, ref=None, ylim=None, ref_lab=None):
    x = np.arange(len(ESCEN)); w = 0.38
    v1 = [fu.val(D, e, "M1", col) for e in ESCEN]
    v2 = [fu.val(D, e, "M2", col) for e in ESCEN]
    _stripes(ax)
    ax.bar(x - w/2, v1, w, color=fu.PAL["M1"], zorder=3,
           edgecolor="white", linewidth=0.6, label=fu.ETQ_MODELO["M1"])
    ax.bar(x + w/2, v2, w, color=fu.PAL["M2"], zorder=3,
           edgecolor="white", linewidth=0.6, label=fu.ETQ_MODELO["M2"])
    if ref is not None:
        ax.axhline(ref, ls=(0, (5, 3)), color=fu.COLOR_REF, lw=1.2, zorder=4)
        if ref_lab:
            ax.text(len(ESCEN) - 0.5, ref * 1.002, ref_lab,
                    color=fu.COLOR_REF, fontsize=9, va="bottom", ha="right")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(titulo, fontsize=12, color="#2b2b2b")
    ax.set_ylabel(ylab, fontsize=10.5)
    ax.set_xticks(x); ax.set_xticklabels(ESCEN, fontsize=9)
    fu.grid_y(ax); fu.limpiar_ejes(ax)


def figura_parte1(ruta_csv, dir_figuras="figuras", formatos=("png", "pdf"),
                  guardar=True):
    D = fu.cargar_tabla(ruta_csv)
    
    # -------------------------------------------------------------------------
    # 1. Panel principal con las primeras 2 sub-figuras (RMSE y Cobertura)
    # -------------------------------------------------------------------------
    fig1, axes1 = fu.plt.subplots(1, 2, figsize=(10.5, 5.2))

    _subplot(axes1[0], D, "rmse", "RMSE de coeficientes",
             "RMSE  (menor es mejor)")
    axes1[0].legend(frameon=False, fontsize=10.5)

    _subplot(axes1[1], D, "cobertura95", "Cobertura IC 95%",
             "Cobertura  (referencia 0.95)", ref=0.95, ylim=(0.85, 1.0),
             ref_lab="0.95")

    fig1.suptitle("Parte 1 · Familias estructurales (errores gaussianos): "
                  "recuperación paramétrica",
                  fontsize=14, fontweight="bold", y=1.00, color="#1a1a1a")
    fig1.text(0.5, 0.004,
              "M₁ y M₂ son prácticamente indistinguibles porque la verosimilitud "
              "no está en juego — esta es la línea base del estudio",
              ha="center", fontsize=10, color=fu.GRIS_SUB, style="italic")
    fig1.tight_layout(rect=[0, 0.025, 1, 0.965])

    # -------------------------------------------------------------------------
    # 2. Figura independiente para el gráfico de Cociente de varianza
    # -------------------------------------------------------------------------
    fig2, ax2 = fu.plt.subplots(1, 1, figsize=(6, 5.2))

    _subplot(ax2, D, "var_ratio", "Cociente de varianza (post/verdadera)",
             "Cociente  (ideal = 1)", ref=1.0, ylim=(0.9, 1.2), ref_lab="1.0")
    
    fig2.tight_layout()

    # -------------------------------------------------------------------------
    # Guardar o retornar ambas figuras
    # -------------------------------------------------------------------------
    if guardar:
        res1 = fu.guardar(fig1, "p1_panel", dir_figuras, formatos)
        res2 = fu.guardar(fig2, "p1_cociente_varianza", dir_figuras, formatos)
        return res1, res2
    
    return fig1, fig2


if __name__ == "__main__":
    figura_parte1("tabla_resumen.csv")