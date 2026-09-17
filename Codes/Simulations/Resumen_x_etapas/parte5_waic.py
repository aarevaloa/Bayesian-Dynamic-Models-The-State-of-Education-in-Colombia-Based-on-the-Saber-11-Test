"""
@author: aarevaloa
parte5_waic.py  —  Parte 5: El WAIC como criterio de selección
===============================================================
Módulos independientes de figuras:
  (1) Mapa de los 29 escenarios coloreado por tipo de selección del WAIC:
      - Acierto directo (21): elige el etiquetado correcto
      - Discrepancia por contaminación (5): la t es mejor en la dist. efectiva
      - Indistinguibles (3): M3/M4 idénticos, margen <1 por mil
      -> Reclasificando, el WAIC elige el modelo adecuado en los 29.
  (2) Márgenes crecientes en la familia de colas (S17/S18/S19).
  (3) Número efectivo de parámetros (p_waic) en D09: los normales consumen más
      complejidad acomodando atípicos -> esto favorece a los robustos.

Uso:
    from parte5_waic import figura_parte5
"""

import numpy as np
from matplotlib.patches import Patch
import figutils as fu


# Colores de categoría de acierto
C_ACIERTO = "#2E7D52"   # verde
C_CONTAM  = "#C5832B"   # ámbar
C_INDIST  = "#7596A0"   # gris-azul


def _clasifica(D, e):
    g = lambda m, c: fu.val(D, e, m, c)
    modelos = [m for m in ("M1", "M2", "M3", "M4") if (e, m) in D]
    waics = {m: g(m, "waic") for m in modelos}
    elige = min(waics, key=waics.get)
    correctos = [m for m in modelos if D[(e, m)]["especificacion"] == "correcta"]
    if elige in correctos:
        cat = "A"
    elif e in ("S11", "S12", "S13", "S14", "D05"):
        cat = "B"
    elif e in ("D02", "D03", "D04"):
        cat = "C"
    else:
        cat = "X"
    return cat, elige


def figura_parte5(ruta_csv, dir_figuras="figuras", formatos=("png", "pdf"),
                  guardar=True):
    D = fu.cargar_tabla(ruta_csv)
    g = lambda e, m, c: fu.val(D, e, m, c)

    # -------------------------------------------------------------------------
    # Figura 1: Mapa de selección del WAIC en los 29 escenarios
    # -------------------------------------------------------------------------
    fig1, ax1 = fu.plt.subplots(figsize=(8.0, 6.5))

    orden = ["S01", "S02", "S04", "S05", "S06", "S07", "S08", "S09", "S10",
             "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19",
             "S20", "S21", "D01", "D02", "D03", "D04", "D05", "D06", "D07",
             "D08", "D09"]
    ncol = 5
    catcolor = {"A": C_ACIERTO, "B": C_CONTAM, "C": C_INDIST, "X": "#999999"}

    for i, e in enumerate(orden):
        r, c = i // ncol, i % ncol
        cat, elige = _clasifica(D, e)
        y = (len(orden) // ncol) - r
        ax1.add_patch(fu.plt.Rectangle((c, y), 0.92, 0.92, facecolor=catcolor[cat],
                                      edgecolor="white", linewidth=1.5, zorder=2))
        ax1.text(c + 0.46, y + 0.58, e, ha="center", va="center", color="white",
                 fontsize=9, fontweight="bold", zorder=3)
        ax1.text(c + 0.46, y + 0.30, f"→{elige}", ha="center", va="center",
                 color="white", fontsize=7.5, zorder=3)

    ax1.set_xlim(-0.15, ncol + 0.05)
    ax1.set_ylim(-0.15, (len(orden) // ncol) + 1.1)
    ax1.axis("off")
    ax1.set_title("Selección del WAIC en los 29 escenarios", fontsize=12.5,
                  color="#2b2b2b", pad=12)

    leg = [
        Patch(facecolor=C_ACIERTO, label="Acierto directo (21): elige el etiquetado correcto"),
        Patch(facecolor=C_CONTAM, label="Discrepancia por contaminación (5): la t es mejor\nen la distribución efectiva"),
        Patch(facecolor=C_INDIST, label="Indistinguibles (3): M₃/M₄ idénticos, margen <1‰")
    ]
    ax1.legend(handles=leg, frameon=False, fontsize=8.5, loc="upper center",
               bbox_to_anchor=(0.5, -0.02), handlelength=1.3, labelspacing=0.9)

    ax1.text(0.5, -0.28, "Reclasificando las discrepancias interpretables, el "
             "WAIC elige el modelo adecuado en los 29 escenarios",
             transform=ax1.transAxes, ha="center", fontsize=8.8, color=C_ACIERTO,
             style="italic", fontweight="bold")

    fig1.tight_layout()

    # -------------------------------------------------------------------------
    # Figura 2: Márgenes crecientes en la familia de colas (S17, S18, S19)
    # -------------------------------------------------------------------------
    fig2, ax2 = fu.plt.subplots(figsize=(6.0, 5.5))

    colas = ["S17", "S18", "S19"]
    lab_colas = ["S17\nν=3", "S18\nν=5", "S19\nmezcla"]
    margenes = [g(e, "M1", "waic") - g(e, "M2", "waic") for e in colas]
    x2 = np.arange(len(colas))

    ax2.bar(x2, margenes, 0.55, color=fu.PAL["M2"], zorder=3, edgecolor="white",
            linewidth=0.7)

    for xi, v in zip(x2, margenes):
        ax2.text(xi, v + 40, f"{v:.0f}", ha="center", fontsize=10,
                 fontweight="bold", color=fu.PAL["M2"])

    ax2.set_xticks(x2)
    ax2.set_xticklabels(lab_colas, fontsize=9.5)
    ax2.set_ylabel("Margen de WAIC a favor de M₂ (correcto)", fontsize=10)
    ax2.set_title("El margen crece con la\npesadez de las colas", fontsize=12,
                  color="#2b2b2b")
    ax2.set_ylim(0, 2400)
    fu.grid_y(ax2); fu.limpiar_ejes(ax2)

    ax2.annotate("colas más pesadas →\ndiscriminación más clara",
                 xy=(0, margenes[0]), xytext=(1.3, 2050), fontsize=8.2,
                 color="#666", ha="center",
                 arrowprops=dict(arrowstyle="->", color="#888", lw=0.9))

    ax2.text(0.5, -0.20, "El WAIC selecciona M₂ en los tres casos, con márgenes "
             "que aumentan\ncuanto más se desvía la verosimilitud incorrecta de la "
             "verdadera.", transform=ax2.transAxes, ha="center", fontsize=8.0,
             color="#666", style="italic", va="top")

    fig2.tight_layout()

    # -------------------------------------------------------------------------
    # Figura 3: Número efectivo de parámetros (p_waic) en D09
    # -------------------------------------------------------------------------
    fig3, ax3 = fu.plt.subplots(figsize=(6.0, 5.5))

    modelos = ["M1", "M3", "M2", "M4"]
    pwaic = [g("D09", m, "waic_pwaic") for m in modelos]
    x3 = np.arange(len(modelos))

    ax3.bar(x3, pwaic, 0.58, color=[fu.PAL[m] for m in modelos], zorder=3,
            edgecolor="white", linewidth=0.7)

    for xi, m, v in zip(x3, modelos, pwaic):
        ax3.text(xi, v + 15, f"{v:.0f}", ha="center", fontsize=10,
                 fontweight="bold", color=fu.PAL[m])

    ax3.set_xticks(x3)
    ax3.set_xticklabels([f'{m}\n({"normal" if m in ["M1","M3"] else "t"})'
                         for m in modelos], fontsize=9)
    ax3.set_ylabel("Número efectivo de parámetros (p$_{WAIC}$)", fontsize=10)
    ax3.set_title("Por qué el WAIC penaliza a los\nnormales: complejidad en D09",
                  fontsize=12, color="#2b2b2b")
    ax3.set_ylim(0, 950)
    fu.grid_y(ax3); fu.limpiar_ejes(ax3)

    ax3.annotate("", xy=(1.4, 860), xytext=(-0.4, 860),
                 arrowprops=dict(arrowstyle="-", color=fu.PAL["M1"], lw=1.5))
    ax3.text(0.5, 890, "normales: 698–793", ha="center", fontsize=8.2,
             color=fu.PAL["M1"], fontweight="bold")

    ax3.annotate("", xy=(3.4, 400), xytext=(1.6, 400),
                 arrowprops=dict(arrowstyle="-", color=fu.PAL["M4"], lw=1.5))
    ax3.text(2.5, 430, "robustos: 194–230", ha="center", fontsize=8.2,
             color=fu.PAL["M4"], fontweight="bold")

    ax3.text(0.5, -0.20, "Los modelos normales consumen complejidad acomodando "
             "los atípicos\nen la varianza residual; esta penalización "
             "diferencial favorece a los robustos.",
             transform=ax3.transAxes, ha="center", fontsize=8.0, color="#666",
             style="italic", va="top")

    fig3.tight_layout()

    # -------------------------------------------------------------------------
    # Guardado de archivos
    # -------------------------------------------------------------------------
    if guardar:
        res1 = fu.guardar(fig1, "p5_waic_mapa", dir_figuras, formatos)
        res2 = fu.guardar(fig2, "p5_waic_margenes_colas", dir_figuras, formatos)
        res3 = fu.guardar(fig3, "p5_waic_pwaic_d09", dir_figuras, formatos)
        return res1, res2, res3

    return fig1, fig2, fig3


if __name__ == "__main__":
    figura_parte5("tabla_resumen.csv")