"""
@author: aarevaloa
parte4_limites.py  —  Parte 4: Mala especificación estructural y encogimiento
==============================================================================
Módulos independientes de figuras:
  (1) S20 mala especificación de la media: todo se degrada severamente frente a
      S02, y M1 y M2 se degradan IDENTICAMENTE (la robustez no protege la media).
  (2) S21 encogimiento extremo: el pooling parcial homogeneiza las varianzas
      (cociente 1.44), lo cual es CORRECTO con información mínima.

Uso:
    from parte4_limites import figura_parte4
"""

import numpy as np
import figutils as fu


def figura_parte4(ruta_csv, dir_figuras="figuras", formatos=("png", "pdf"),
                  guardar=True):
    D = fu.cargar_tabla(ruta_csv)
    g = lambda e, m, c: fu.val(D, e, m, c)

    def vv(esc, m, met):
        return abs(g(esc, m, "sesgo")) if met == "sesgo_abs" else g(esc, m, met)

    # -------------------------------------------------------------------------
    # Figura 1: (1) S20 Mala especificación de la media
    # -------------------------------------------------------------------------
    fig1, ax1 = fu.plt.subplots(figsize=(8.5, 5.8))

    mets = ["cobertura95", "var_ratio", "rmse", "sesgo_abs"]
    f1, f2 = [], []
    for met in mets:
        if met == "cobertura95":
            # cobertura cae: usar S02/S20 para que >1 = peor
            f1.append(vv("S02", "M1", met) / vv("S20", "M1", met))
            f2.append(vv("S02", "M2", met) / vv("S20", "M2", met))
        else:
            f1.append(vv("S20", "M1", met) / vv("S02", "M1", met))
            f2.append(vv("S20", "M2", met) / vv("S02", "M2", met))

    x = np.arange(len(mets)); w = 0.34
    ax1.bar(x - w/2, f1, w, color=fu.PAL["M1"], zorder=3, edgecolor="white",
            linewidth=0.6, label="M₁ (normal)")
    ax1.bar(x + w/2, f2, w, color=fu.PAL["M2"], zorder=3, edgecolor="white",
            linewidth=0.6, label="M₂ (t)")
    ax1.set_yscale("log"); ax1.set_ylim(0.8, 80)
    ax1.axhline(1.0, ls=(0, (5, 3)), color=fu.COLOR_REF, lw=1.2, zorder=2)
    ax1.text(3.5, 1.05, "sin cambio", color=fu.COLOR_REF, fontsize=8,
             va="bottom", ha="right")

    for xi, v in zip(x - w/2, f1):
        ax1.text(xi, v * 1.12, f"×{v:.0f}" if v >= 2 else f"×{v:.1f}",
                 ha="center", fontsize=8.5, color=fu.PAL["M1"], fontweight="bold")
    for xi, v in zip(x + w/2, f2):
        ax1.text(xi, v * 1.12, f"×{v:.0f}" if v >= 2 else f"×{v:.1f}",
                 ha="center", fontsize=8.5, color=fu.PAL["M2"], fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(["Caída de\ncobertura", "Inflación de\nvarianza",
                         "Aumento\nde RMSE", "Aumento\nde |sesgo|"], fontsize=9)
    ax1.set_ylabel("Factor de empeoramiento S20 vs S02  (log)", fontsize=10)
    ax1.set_title("S20 · Mala especificación de la media:\nM₁ y M₂ se degradan "
                  "IDÉNTICAMENTE", fontsize=12, color="#2b2b2b")
    ax1.legend(frameon=False, fontsize=9.5, loc="upper left")
    fu.grid_y(ax1, which="both"); fu.limpiar_ejes(ax1)

    fig1.text(0.5, 0.02, "Las barras para M₁ y M₂ son idénticas: la robustez de la t "
              "NO protege contra la omisión de una componente estructural de la media.",
              transform=fig1.transFigure, fontsize=8.5, color="#666", ha="center", style="italic")
    fig1.tight_layout(rect=[0, 0.06, 1, 0.96])

    # -------------------------------------------------------------------------
    # Figura 2: (2) S21 Encogimiento extremo (Pooling parcial)
    # -------------------------------------------------------------------------
    fig2, ax2 = fu.plt.subplots(figsize=(6.5, 5.8))

    modelos = ["M1", "M2"]; x2 = np.arange(len(modelos)); ww2 = 0.5
    vr = [g("S21", m, "var_ratio") for m in modelos]
    cob = [g("S21", m, "cobertura95") for m in modelos]
    ax2.bar(x2, vr, ww2, color=[fu.PAL["M1"], fu.PAL["M2"]], zorder=3,
            edgecolor="white", linewidth=0.7)

    for xi, v in zip(x2, vr):
        ax2.text(xi, v + 0.04, f"{v:.2f}", ha="center", fontsize=10,
                 fontweight="bold", color="#2b2b2b")
    ax2.axhline(1.0, ls=(0, (5, 3)), color=fu.COLOR_REF, lw=1.2, zorder=2)
    ax2.set_ylim(0, 1.9)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["M₁ (normal)\n◄ correcto", "M₂ (t)"], fontsize=9.5)
    ax2.set_ylabel("Cociente de varianza", fontsize=10)
    ax2.set_title("S21 · Encogimiento extremo:\nel pooling homogeneiza (y eso es "
                  "correcto)", fontsize=12, color="#2b2b2b")
    fu.grid_y(ax2); fu.limpiar_ejes(ax2)

    # txt = ("3 individuos/municipio: imposible estimar\ncada varianza municipal "
    #        "por separado.\nEl pooling parcial arrastra las varianzas\nhacia el "
    #        "nivel departamental (cociente 1.44).\nNo es un defecto: con tan poca "
    #        "información,\nencoger es la respuesta sensata.\n\n"
    #        f"Pese a todo: cobertura {cob[0]:.2f}, RMSE {g('S21','M1','rmse'):.3f}\n"
    #        "(finito y estable). WAIC elige M₁ (normal).")
    # ax2.text(0.5, -0.22, txt, transform=ax2.transAxes, fontsize=7.6, color="#555",
    #          ha="center", va="top",
    #          bbox=dict(boxstyle="round,pad=0.45", fc=fu.CAJA_FC, ec=fu.CAJA_EC,
    #                    alpha=0.9))

    fig2.tight_layout()

    # -------------------------------------------------------------------------
    # Guardado de archivos
    # -------------------------------------------------------------------------
    if guardar:
        res1 = fu.guardar(fig1, "p4_s20_mala_especificacion", dir_figuras, formatos)
        res2 = fu.guardar(fig2, "p4_s21_encogimiento", dir_figuras, formatos)
        return res1, res2

    return fig1, fig2


if __name__ == "__main__":
    figura_parte4("tabla_resumen.csv")