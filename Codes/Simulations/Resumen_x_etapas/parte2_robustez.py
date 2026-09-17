"""
@author: aarevaloa
parte2_robustez.py  —  Parte 2: Robustez en el marco estatico
==============================================================
Panel de 2 sub-figuras principales sobre contaminación (S11-S16):
  (1) cociente de varianza bajo contaminacion, escala log, BARRAS
  (2) RMSE de coeficientes bajo contaminacion, escala log, PUNTOS

Más la figura independiente sobre colas intrínsecas (S17/S18):
  (3) recuperacion de nu: estimado vs verdadero

Más la mini-figura de contaminacion contextual (S15): media vs mediana.

Uso:
    from parte2_robustez import figura_parte2, minifigura_s15
    figura_parte2("ruta/tabla_resumen.csv")     # -> figuras/p2_panel.* y figuras/p2_recuperacion_nu.*
    minifigura_s15("ruta/tabla_resumen.csv")    # -> figuras/p2_minifig_S15.*
"""

import numpy as np
import figutils as fu


CONTA = ["S11", "S12", "S13", "S14", "S15", "S16"]
LAB_CONTA = ["5%", "10%", "20%", "Asim.", "Context.", "Covar."]


def figura_parte2(ruta_csv, dir_figuras="figuras", formatos=("png", "pdf"),
                  guardar=True):
    D = fu.cargar_tabla(ruta_csv)
    g = lambda e, m, c: fu.val(D, e, m, c)
    x = np.arange(len(CONTA))

    # -------------------------------------------------------------------------
    # 1. Panel principal: (1) Cociente de varianza y (2) RMSE bajo contaminación
    # -------------------------------------------------------------------------
    fig1, axes1 = fu.plt.subplots(1, 2, figsize=(11.5, 5.4))

    # ---- (1) cociente de varianza: BARRAS ----
    ax = axes1[0]
    vr1 = [g(e, "M1", "var_ratio") for e in CONTA]
    vr2 = [g(e, "M2", "var_ratio") for e in CONTA]
    w = 0.38
    ax.axvspan(-0.5, 2.5, color=fu.BANDA_A, zorder=0)
    ax.axvspan(2.5, 5.5, color=fu.BANDA_B, zorder=0)
    ax.bar(x - w/2, vr1, w, color=fu.PAL["M1"], zorder=3, edgecolor="white",
           linewidth=0.6, label="M₁ (normal)")
    ax.bar(x + w/2, vr2, w, color=fu.PAL["M2"], zorder=3, edgecolor="white",
           linewidth=0.6, label="M₂ (t)")
    ax.set_yscale("log"); ax.set_ylim(0.4, 700)
    ax.axhline(1.0, ls=(0, (5, 3)), color=fu.COLOR_REF, lw=1.2, zorder=1)
    ax.text(5.4, 1.12, "ideal = 1", color=fu.COLOR_REF, fontsize=8.5,
            va="bottom", ha="right")
    ax.set_title("Cociente de varianza bajo contaminación", fontsize=11.5,
                 color="#2b2b2b")
    ax.set_ylabel("Cociente post/verdadera  (log, ideal = 1)", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(LAB_CONTA, fontsize=9)
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    fu.grid_y(ax, which="both"); fu.limpiar_ejes(ax)
    for i, e in enumerate(["S11", "S12", "S13"]):
        ax.text(i - w/2, vr1[i] * 1.18, f"{vr1[i]:.0f}", ha="center",
                fontsize=7.5, color=fu.PAL["M1"], fontweight="bold")
    ax.text(1, 480, "gradiente", ha="center", fontsize=8.5, color="#888",
            style="italic")
    ax.text(4, 480, "modalidades", ha="center", fontsize=8.5, color="#888",
            style="italic")

    # ---- (2) RMSE: PUNTOS ----
    ax = axes1[1]
    rmse1 = [g(e, "M1", "rmse") for e in CONTA]
    rmse2 = [g(e, "M2", "rmse") for e in CONTA]
    ax.axvspan(-0.5, 2.5, color=fu.BANDA_A, zorder=0)
    ax.axvspan(2.5, 5.5, color=fu.BANDA_B, zorder=0)
    for xi, v1, v2 in zip(x, rmse1, rmse2):
        ax.plot([xi, xi], [v2, v1], color="#bbbbbb", lw=1.0, zorder=2)
    ax.scatter(x, rmse1, s=70, color=fu.PAL["M1"], zorder=4,
                edgecolor="white", linewidth=0.8, label="M₁ (normal)")
    ax.scatter(x, rmse2, s=70, color=fu.PAL["M2"], zorder=4,
                edgecolor="white", linewidth=0.8, label="M₂ (t)")
    ax.set_yscale("log"); ax.set_ylim(0.008, 1.5)
    ax.set_title("RMSE de coeficientes bajo contaminación", fontsize=11.5,
                 color="#2b2b2b")
    ax.set_ylabel("RMSE  (log, menor es mejor)", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(LAB_CONTA, fontsize=9)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    fu.grid_y(ax, which="both"); fu.limpiar_ejes(ax)
    ax.text(3, rmse1[3] * 1.3, "0.87", ha="center", fontsize=7.5,
            color=fu.PAL["M1"], fontweight="bold")
    ax.annotate("S14: M₁ pierde\nla señal (r=0.55)", xy=(3.1, rmse1[3] * 0.85),
                xytext=(3.6, 0.4), fontsize=7.5, color="#666", ha="left",
                arrowprops=dict(arrowstyle="->", color="#999", lw=0.8))
    ax.annotate("S15,S16: empate\n(t no protege)", xy=(4.5, rmse1[4]),
                xytext=(3.7, 0.025), fontsize=7.5, color="#666", ha="center",
                arrowprops=dict(arrowstyle="->", color="#999", lw=0.8))

    fig1.suptitle("Parte 2 · Robustez en el marco estático: verosimilitud t bajo contaminación",
                  fontsize=13.5, fontweight="bold", y=1.00, color="#1a1a1a")
    fig1.text(0.5, 0.004,
              "Bajo contaminación, M₁ infla la varianza hasta dos órdenes de "
              "magnitud mientras M₂ la mantiene en ~1. La protección es nula en covariables (S16).",
              ha="center", fontsize=9.5, color=fu.GRIS_SUB, style="italic")
    fig1.tight_layout(rect=[0, 0.025, 1, 0.96])

    # -------------------------------------------------------------------------
    # 2. Figura independiente: (3) Recuperación de nu (colas intrínsecas)
    # -------------------------------------------------------------------------
    fig2, ax_nu = fu.plt.subplots(1, 1, figsize=(5.8, 5.4))

    nu_est = [g("S17", "M2", "nu_post"), g("S18", "M2", "nu_post")]
    nu_tru = [3, 5]
    ax_nu.plot([2, 6], [2, 6], ls=(0, (4, 3)), color=fu.COLOR_REF, lw=1, zorder=1)
    for nt, ne in zip(nu_tru, nu_est):
        ax_nu.scatter(nt, ne, s=150, color=fu.PAL["M2"], zorder=4,
                      edgecolor="white", linewidth=1.3)
    ax_nu.annotate(f"S17: ν̂={nu_est[0]:.2f}\n(verdad 3)", (3, nu_est[0]),
                   textcoords="offset points", xytext=(12, 8), fontsize=8.5,
                   color="#444")
    ax_nu.annotate(f"S18: ν̂={nu_est[1]:.2f}\n(verdad 5)", (5, nu_est[1]),
                   textcoords="offset points", xytext=(12, -2), fontsize=8.5,
                   color="#444")
    ax_nu.set_xlim(2, 6); ax_nu.set_ylim(2, 6)
    ax_nu.set_title("Recuperación de ν (colas intrínsecas)", fontsize=11.5,
                    color="#2b2b2b")
    ax_nu.set_xlabel("ν verdadero", fontsize=10)
    ax_nu.set_ylabel("ν estimado (posterior de M₂)", fontsize=10)
    ax_nu.grid(alpha=0.18); fu.limpiar_ejes(ax_nu)
    ax_nu.text(0.96, 0.06, "la diagonal = recuperación perfecta",
               transform=ax_nu.transAxes, fontsize=8, color="#888", style="italic",
               ha="right")
    txt = ("M₁ infla la varianza (teoría ν/(ν−2)):\n"
           f"  ν=3 → teórico 3.00 · obs. {g('S17','M1','var_ratio'):.2f}\n"
           f"  ν=5 → teórico 1.67 · obs. {g('S18','M1','var_ratio'):.2f}")
    ax_nu.text(0.04, 0.97, txt, transform=ax_nu.transAxes, fontsize=7.8,
               color="#1F4E5F", va="top",
               bbox=dict(boxstyle="round,pad=0.45", fc="#eef3f4",
                         ec="#1F4E5F", alpha=0.85))

    fig2.tight_layout()

    # -------------------------------------------------------------------------
    # Guardado de archivos
    # -------------------------------------------------------------------------
    if guardar:
        res1 = fu.guardar(fig1, "p2_panel", dir_figuras, formatos)
        res2 = fu.guardar(fig2, "p2_recuperacion_nu", dir_figuras, formatos)
        return res1, res2

    return fig1, fig2


def minifigura_s15(ruta_csv, dir_figuras="figuras", formatos=("png", "pdf"),
                   guardar=True):
    D = fu.cargar_tabla(ruta_csv)
    g = lambda e, m, c: fu.val(D, e, m, c)
    C_MEDIA, C_MEDI = "#5a7d88", "#c98a6a"
    fig, (ax1, ax2) = fu.plt.subplots(1, 2, figsize=(13.5, 5.4),
                                      gridspec_kw={"width_ratios": [1.25, 1]})

    # IZQUIERDA — BARRAS media vs mediana (valores grandes -> lectura segura)
    modelos = ["M1", "M2"]; x = np.arange(len(modelos)); w = 0.36
    media = [g("S15", m, "var_ratio") for m in modelos]
    medi = [g("S15", m, "var_err_mediano") for m in modelos]
    ax1.bar(x - w/2, media, w, color=C_MEDIA, zorder=3, edgecolor="white",
            linewidth=0.6, label="Cociente de varianza (media)")
    ax1.bar(x + w/2, medi, w, color=C_MEDI, zorder=3, edgecolor="white",
            linewidth=0.6, label="Error relativo mediano")
    ax1.axhline(1.0, ls=(0, (5, 3)), color=fu.COLOR_REF, lw=1.2, zorder=2)
    ax1.text(1.62, 1.1, "ideal = 1", color=fu.COLOR_REF, fontsize=8.5,
             va="bottom", ha="right")
    for xi, v in zip(x - w/2, media):
        ax1.text(xi, v * 1.2, f"{v:.0f}", ha="center", fontsize=10,
                 color="#3c5a64", fontweight="bold")
    for xi, v in zip(x + w/2, medi):
        ax1.text(xi, v * 1.2, f"{v:.2f}", ha="center", fontsize=10,
                 color="#a8552e", fontweight="bold")
    ax1.set_yscale("log"); ax1.set_ylim(0.05, 900)
    ax1.set_xticks(x); ax1.set_xticklabels(["M₁ (normal)", "M₂ (t)"], fontsize=10.5)
    ax1.set_ylabel("Métrica de varianza  (escala log)", fontsize=10.5)
    ax1.set_title("S15 contextual: media vs mediana de la varianza",
                  fontsize=12, color="#2b2b2b")
    ax1.legend(frameon=False, fontsize=9.5, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02))
    fu.grid_y(ax1, which="both"); fu.limpiar_ejes(ax1)

    # DERECHA — BARRAS S12 vs S15
    casos = ["S12\ndispersa", "S15\ncontextual"]; xc = np.arange(len(casos))
    vr = [g("S12", "M2", "var_ratio"), g("S15", "M2", "var_ratio")]
    cols = [fu.PAL["M2"], C_MEDI]
    ax2.bar(xc, vr, 0.5, color=cols, zorder=3, edgecolor="white", linewidth=0.7)
    ax2.axhline(1.0, ls=(0, (5, 3)), color=fu.COLOR_REF, lw=1.2, zorder=2)
    for xi, v in zip(xc, vr):
        ax2.text(xi, v * 1.25, f"{v:.2f}" if v < 10 else f"{v:.0f}", ha="center",
                 fontsize=11, color="#a8552e", fontweight="bold")
    ax2.set_yscale("log"); ax2.set_ylim(0.5, 900)
    ax2.set_xticks(xc); ax2.set_xticklabels(casos, fontsize=10)
    ax2.set_ylabel("Cociente de varianza de M₂  (log)", fontsize=10.5)
    ax2.set_title("Por qué la t no mitiga aquí\n(M₂, ambos al 10%)",
                  fontsize=12, color="#2b2b2b")
    fu.grid_y(ax2, which="both"); fu.limpiar_ejes(ax2)

    fig.suptitle("Mini-figura · Contaminación contextual (S15): dos métricas de "
                 "varianza, una lectura coherente",
                 fontsize=13, fontweight="bold", y=1.02, color="#1a1a1a")
    fig.text(0.5, -0.04,
             "El cociente medio es alto porque unas pocas celdas —municipios "
             "contaminados completos— tienen varianza genuinamente elevada;\n"
             "la mediana, baja, refleja la mayoría de celdas estimadas con "
             "precisión. Ambos modelos recuperan los coeficientes (RMSE 0.012, r=0.9999).",
             ha="center", fontsize=8.5, color=fu.GRIS_SUB, style="italic")
    fig.subplots_adjust(bottom=0.16, top=0.88, wspace=0.28)

    if guardar:
        return fu.guardar(fig, "p2_minifig_S15", dir_figuras, formatos)
    return fig


if __name__ == "__main__":
    figura_parte2("tabla_resumen.csv")
    minifigura_s15("tabla_resumen.csv")