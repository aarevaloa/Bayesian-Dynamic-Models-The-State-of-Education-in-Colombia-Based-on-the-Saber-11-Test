"""
@author: aarevaloa
parte3_dinamica.py  —  Parte 3: Dimension temporal y aditividad
================================================================
Panel de 2 sub-figuras principales (cociente de varianza):
  (1) regimen temporal D01-D04: el cociente de varianza de los ESTATICOS crece
      con la velocidad de la variacion temporal; los dinamicos quedan en ~1.
  (2) interaccion D05 vs D06: en contaminacion solo la t controla (la dinamica
      no protege); en colas t cada eje aporta y solo M4 acierta -> aditividad.

Mas la figura independiente de comparacion global (WAIC):
  (3) prueba de estres D09: los 4 modelos se separan en 2 grupos por
      verosimilitud (~93k de WAIC entre grupos vs 3059 dentro del robusto).

Mas la mini-figura de varianzas dinamicas (D07/D08): el componente de varianza
dinamica es robusto a la forma de la evolucion.

Uso:
    from parte3_dinamica import figura_parte3, minifigura_d07d08
"""

import numpy as np
import figutils as fu


def figura_parte3(ruta_csv, dir_figuras="figuras", formatos=("png", "pdf"),
                  guardar=True):
    D = fu.cargar_tabla(ruta_csv)
    g = lambda e, m, c: fu.val(D, e, m, c)
    etq4 = fu.ETQ_MODELO_DIN

    # -------------------------------------------------------------------------
    # 1. Panel principal: (1) Régimen temporal y (2) Interacción / Aditividad
    # -------------------------------------------------------------------------
    fig1, axes1 = fu.plt.subplots(1, 2, figsize=(12.5, 5.8))

    # ---- (1) régimen temporal ----
    ax = axes1[0]
    reg = ["D01", "D02", "D03", "D04"]
    lab_reg = ["Sin\ndinámica", "Camin.\nlenta", "Camin.\nrápida", "Ruptura"]
    x = np.arange(len(reg)); w = 0.20
    for j, m in enumerate(["M1", "M2", "M3", "M4"]):
        vals = [g(e, m, "var_ratio") for e in reg]
        ax.bar(x + (j - 1.5) * w, vals, w, color=fu.PAL[m], zorder=3,
               edgecolor="white", linewidth=0.5, label=etq4[m])
    ax.axhline(1.0, ls=(0, (5, 3)), color=fu.COLOR_REF, lw=1.2, zorder=2)
    ax.set_title("Régimen temporal: el cociente de varianza\nde los estáticos "
                 "crece con la dinámica", fontsize=11.5, color="#2b2b2b")
    ax.set_ylabel("Cociente de varianza (ideal = 1)", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(lab_reg, fontsize=9)
    ax.legend(frameon=False, fontsize=8.2, loc="upper left")
    fu.grid_y(ax); fu.limpiar_ejes(ax); ax.set_ylim(0, 4.6)
    ax.text(3.45, 1.06, "ideal = 1", color=fu.COLOR_REF, fontsize=8.5,
            va="bottom", ha="right")
    ax.annotate("los estáticos (M₁,M₂)\nse inflan: 1.01 → 3.85",
                xy=(2 - 1.5 * w, g("D03", "M1", "var_ratio")), xytext=(2.35, 3.9),
                fontsize=8, color=fu.PAL["M1"], ha="center",
                arrowprops=dict(arrowstyle="->", color="#888", lw=0.8))

    # ---- (2) interacción D05 vs D06 ----
    ax = axes1[1]
    casos = ["D05", "D06"]
    lab_casos = ["D05\nCamin.+contam.10%", "D06\nCamin.+colas t"]
    x = np.arange(len(casos)); w = 0.20
    for j, m in enumerate(["M1", "M2", "M3", "M4"]):
        vals = [g(e, m, "var_ratio") for e in casos]
        ax.bar(x + (j - 1.5) * w, vals, w, color=fu.PAL[m], zorder=3,
               edgecolor="white", linewidth=0.5, label=etq4[m])
    ax.set_yscale("log"); ax.set_ylim(0.7, 1500)
    ax.axhline(1.0, ls=(0, (5, 3)), color=fu.COLOR_REF, lw=1.2, zorder=2)
    ax.text(1.42, 1.04, "ideal = 1", color=fu.COLOR_REF, fontsize=8,
            va="bottom", ha="right")
    ax.set_title("La pregunta central: ¿son aditivos\nrobustez y dinámica?",
                 fontsize=11.5, color="#2b2b2b")
    ax.set_ylabel("Cociente de varianza (log, ideal = 1)", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(lab_casos, fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.2, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 0.86))
    fu.grid_y(ax, which="both"); fu.limpiar_ejes(ax)
    # ax.text(0, 950, "contaminación:\nsolo la t controls;\nla dinámica no protege",
    #         ha="center", va="top", fontsize=7.6, color="#444",
    #         bbox=dict(boxstyle="round,pad=0.35", fc=fu.CAJA_FC, ec=fu.CAJA_EC,
    #                   alpha=0.85))
    # ax.text(1, 60, "colas t:\ncada eje aporta;\nsolo M₄ (ambos) acierta",
    #         ha="center", va="top", fontsize=7.6, color="#444",
    #         bbox=dict(boxstyle="round,pad=0.35", fc=fu.CAJA_FC, ec=fu.CAJA_EC,
    #                   alpha=0.85))

    fig1.suptitle("Parte 3 · Dimensión temporal y aditividad: evaluación del cociente de varianza",
                  fontsize=13.5, fontweight="bold", y=1.00, color="#1a1a1a")
    fig1.text(0.5, 0.004,
              "La dinámica importa cuando hay variación temporal genuina (izq.), "
              "pero no protege frente a contaminación (der.); solo el modelo que reúne ambos ejes (M₄) acierta.",
              ha="center", fontsize=9.3, color=fu.GRIS_SUB, style="italic")
    fig1.tight_layout(rect=[0, 0.03, 1, 0.96])

    # -------------------------------------------------------------------------
    # 2. Figura independiente: (3) Prueba de estrés D09 (Ajuste global WAIC)
    # -------------------------------------------------------------------------
    fig2, ax = fu.plt.subplots(1, 1, figsize=(6.5, 5.8))

    modelos = ["M4", "M2", "M3", "M1"]
    vals = [g("D09", m, "waic") for m in modelos]
    y = np.arange(len(modelos))[::-1]
    ax.barh(y, vals, 0.6, color=[fu.PAL[m] for m in modelos], zorder=3,
            edgecolor="white", linewidth=0.7)
    for yi, m, v in zip(y, modelos, vals):
        if v > 250000:
            ax.text(v - 8000, yi, f"{v/1000:.0f}k", va="center", ha="right",
                    fontsize=10, fontweight="bold", color="white")
        else:
            ax.text(v + 5000, yi, f"{v/1000:.0f}k", va="center", fontsize=10,
                    fontweight="bold", color=fu.PAL[m])
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f'{m} ({"t" if m in ["M2","M4"] else "normal"}, '
         f'{"din." if m in ["M3","M4"] else "est."})' for m in modelos],
        fontsize=9)
    ax.set_xlim(0, 380000)
    ax.set_xlabel("WAIC  (menor es mejor)", fontsize=10)
    ax.set_title("Prueba de estrés D09: dos grupos\nseparados por la verosimilitud",
                 fontsize=11.5, color="#2b2b2b")
    ax.grid(axis="x", alpha=0.18); fu.limpiar_ejes(ax)
    ax.annotate("", xy=(355000, 3.45), xytext=(355000, 2.55),
                arrowprops=dict(arrowstyle="-", color=fu.PAL["M4"], lw=2.5))
    ax.text(360000, 3.0, "t robustos\n≈200k", va="center", fontsize=8.5,
            color=fu.PAL["M4"], fontweight="bold")
    ax.annotate("", xy=(355000, 1.45), xytext=(355000, 0.55),
                arrowprops=dict(arrowstyle="-", color=fu.PAL["M1"], lw=2.5))
    ax.text(360000, 1.0, "normales\n≈295k", va="center", fontsize=8.5,
            color=fu.PAL["M1"], fontweight="bold")
    ax.text(0.5, -0.15, "Separación entre grupos: ~93 000 unidades · dentro del "
            "grupo robusto, M₄ aventaja a M₂ por 3 059", transform=ax.transAxes,
            fontsize=8, color="#666", ha="center", style="italic")

    fig2.tight_layout()

    # -------------------------------------------------------------------------
    # Guardado de archivos
    # -------------------------------------------------------------------------
    if guardar:
        res1 = fu.guardar(fig1, "p3_panel", dir_figuras, formatos)
        res2 = fu.guardar(fig2, "p3_waic_d09", dir_figuras, formatos)
        return res1, res2

    return fig1, fig2


def minifigura_d07d08(ruta_csv, dir_figuras="figuras", formatos=("png", "pdf"),
                      guardar=True):
    D = fu.cargar_tabla(ruta_csv)
    g = lambda e, m, c: fu.val(D, e, m, c)
    fig, ax = fu.plt.subplots(figsize=(11, 5.6))

    casos = ["D07", "D08"]
    sub = ["Mecanismo de descuento\n(coincide con el supuesto del modelo)",
           "Mecanismo alternativo\n(ajeno a la especificación)"]
    x = np.arange(len(casos)); w = 0.20
    etq4 = {"M1": "M₁ (est., normal)", "M2": "M₂ (est., t)",
            "M3": "M₃ (din., normal) ◄ correcto", "M4": "M₄ (din., t)"}

    ax.axhspan(0.95, 1.15, color="#eaf1ec", zorder=0)
    ax.text(1.45, 1.145, "zona ideal (≈1)", color="#5a8a6a", fontsize=8.5,
            va="top", ha="right", style="italic")
    for j, m in enumerate(["M1", "M2", "M3", "M4"]):
        vals = [g(e, m, "var_ratio") for e in casos]
        ax.bar(x + (j - 1.5) * w, vals, w, color=fu.PAL[m], zorder=3,
               edgecolor="white", linewidth=0.6, label=etq4[m])
        if m in ("M3", "M4"):
            for xi, v in zip(x + (j - 1.5) * w, vals):
                ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5,
                        color=fu.PAL[m], fontweight="bold")
    ax.axhline(1.0, ls=(0, (5, 3)), color=fu.COLOR_REF, lw=1.2, zorder=2)
    ax.set_ylim(0, 1.32)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{s}" for c, s in zip(casos, sub)], fontsize=9.5)
    ax.set_ylabel("Cociente de varianza (ideal = 1)", fontsize=11)
    ax.set_title("Mini-figura · Varianzas dinámicas (D07/D08): el componente de "
                 "varianza dinámica\nes robusto a la forma de la evolución",
                 fontsize=12.5, color="#1a1a1a")
    ax.legend(frameon=False, fontsize=9, loc="lower center", ncol=4,
              bbox_to_anchor=(0.5, -0.30))
    fu.grid_y(ax); fu.limpiar_ejes(ax)
    ax.annotate("", xy=(1 - 1.5 * w, g("D08", "M3", "var_ratio") + 0.05),
                xytext=(0 - 1.5 * w, g("D07", "M3", "var_ratio") + 0.05),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.0,
                                connectionstyle="arc3,rad=-0.2"))
    ax.text(0.5, 1.20, "el modelo correcto (M₃) apenas se degrada:\n"
            "1.03 → 1.11 aunque el mecanismo cambie", ha="center", va="center",
            fontsize=8.3, color="#444",
            bbox=dict(boxstyle="round,pad=0.4", fc=fu.CAJA_FC, ec=fu.CAJA_EC,
                      alpha=0.9))

    fig.text(0.5, -0.02,
             "Cuando la varianza evoluciona según el mecanismo que el modelo "
             "asume (D07) la recuperación es casi perfecta (1.03); cuando "
             "evoluciona según un mecanismo\najeno (D08) se degrada solo "
             "levemente (1.11). El componente recupera la varianza incluso si el "
             "mecanismo generador no coincide con el supuesto.",
             ha="center", fontsize=8.5, color=fu.GRIS_SUB, style="italic")
    fig.subplots_adjust(bottom=0.26, top=0.86)

    if guardar:
        return fu.guardar(fig, "p3_minifig_D07D08", dir_figuras, formatos)
    return fig


if __name__ == "__main__":
    figura_parte3("tabla_resumen.csv")
    minifigura_d07d08("tabla_resumen.csv")