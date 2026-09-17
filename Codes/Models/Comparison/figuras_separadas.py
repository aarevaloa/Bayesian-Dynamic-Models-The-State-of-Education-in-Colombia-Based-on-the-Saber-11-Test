
ANCHO_TOTAL = {1: 6.6, 2: 9.4}
ALTO_FIG = 4.5


GRUPOS_FIG = {
    "comp_02_convergencia": [
        (["rhat"], "comp_02a_rhat",
         "Convergencia MCMC: $\\hat{R}$ máximo por modelo"),
        (["ess_min", "ess_lt400"], "comp_02b_ess",
         "Convergencia MCMC: tamaño efectivo de muestra"),
        (["geweke"], "comp_02c_geweke",
         "Convergencia MCMC: diagnóstico de Geweke"),
    ],
    "comp_03_ajuste": [
        (["r2_bayes", "r2"], "comp_03a_r2",
         "Ajuste dentro de muestra: coeficiente de determinación"),
        (["rmse", "mae"], "comp_03b_error",
         "Ajuste dentro de muestra: errores de predicción"),
    ],
    "comp_04_informacion": [
        # el forest de Delta-elpd es el criterio formal de selección: va solo
        (["delta"], "comp_04a_delta",
         "Comparación formal: diferencias pareadas de elpd"),
        (["elpd", "peff"], "comp_04b_elpd_peff",
         "Criterios de información: elpd y complejidad efectiva"),
    ],
    "comp_05_calibracion": [
        (["ppc_z"], "comp_05a_ppc",
         "Verificación predictiva posterior por estadístico"),
        (["cobertura", "pit_ks"], "comp_05b_cobertura_pit",
         "Calibración de la distribución predictiva"),
    ],
}


def _barras(ax, presentes, vals, titulo, mejor, fmt="{:.3f}", log=False,
            lineas=(), etiquetas_lineas=()):
    """Idéntica a la original salvo por la colocación de las etiquetas de las
    líneas de referencia: antes iban fuera del eje (x = -0.46 en coordenadas
    de datos), lo que en paneles estrechos las hacía chocar con los rótulos
    del eje y, en escala logarítmica, superponerse a los 10^k. Ahora se
    dibujan dentro del área de ejes, pegadas al borde derecho.
    """
    x = np.arange(len(presentes))
    v = np.array([vals.get(M, np.nan) for M in presentes], float)
    cols = [MODELOS[M]["color"] for M in presentes]
    disp = np.where(np.isfinite(v), v, 0)
    ax.bar(x, disp, color=cols, alpha=0.92, width=0.66,
           edgecolor="white", linewidth=0.8, zorder=3)
    vmax = float(np.nanmax(disp)) if np.any(np.isfinite(disp)) else 1.0
    incl_lineas = [lv for lv in lineas if np.isfinite(lv)]
    tope_ref = max([vmax] + incl_lineas) if incl_lineas else vmax
    hay_datos = bool(np.any(np.isfinite(v)))
    if not hay_datos:
        ax.set_ylim(0, 1)
        ax.set_xticks(x); ax.set_xticklabels(presentes)
        ax.text(0.5, 0.5, "sin datos", transform=ax.transAxes, ha="center",
                va="center", fontsize=9, color=GRIS, style="italic")
        ax.set_title(titulo, fontsize=11, color=TINTA, pad=8)
        ax.text(0.5, 1.015, mejor, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=8.3, color=GRIS, style="italic")
        return
    if not log:
        ax.set_ylim(0, tope_ref * 1.20 if tope_ref > 0 else 1.0)
    umbral_dentro = tope_ref * 0.86
    for i, M in enumerate(presentes):
        if not np.isfinite(v[i]):
            ax.text(i, 0.04, "n/d", ha="center", va="bottom",
                    transform=ax.get_xaxis_transform(),
                    fontsize=8.5, color=GRIS, style="italic")
            continue
        txt = fmt.format(v[i])
        if (not log) and disp[i] > umbral_dentro:
            ax.text(i, disp[i] - tope_ref * 0.015, txt, ha="center",
                    va="top", fontsize=9, color="white", fontweight="medium",
                    zorder=4)
        else:
            off = (disp[i] * 1.04 if log else disp[i] + tope_ref * 0.02)
            ax.text(i, off, txt, ha="center", va="bottom", fontsize=9,
                    color=TINTA, zorder=4)
    for lv, lab in zip(lineas, etiquetas_lineas):
        ax.axhline(lv, color=GRIS, lw=1, ls=(0, (5, 3)), zorder=1, alpha=0.7)
        # --- CAMBIO: etiqueta dentro del eje, pegada al borde derecho ---
        ax.text(0.995, lv, lab, transform=ax.get_yaxis_transform(),
                va="bottom", ha="right", fontsize=7.8, color=GRIS,
                style="italic", zorder=4)
    if log and np.nanmin(disp[disp > 0], initial=1) > 0:
        ax.set_yscale("log")
        pos = disp[disp > 0]
        if pos.size:
            lo = min(pos.min(), min(incl_lineas) if incl_lineas else pos.min())
            hi = max(pos.max(), max(incl_lineas) if incl_lineas else pos.max())
            ax.set_ylim(lo * 0.55, hi * 1.9)
    ax.set_xticks(x); ax.set_xticklabels(presentes)
    ax.set_title(titulo, fontsize=11, color=TINTA, pad=8)
    ax.text(0.5, 1.015, mejor, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=8.3, color=GRIS, style="italic")


def _emitir_grupos(base, paneles, dir_diag, pdf, nota=""):
    """Dibuja y guarda una imagen por grupo definido en GRUPOS_FIG[base].

    `paneles` es un dict clave -> {"dibujar": f(ax), "w": peso_de_ancho}.
    Las claves ausentes se ignoran, de modo que un panel sin datos no rompe
    la emision del resto.
    """
    grupos = GRUPOS_FIG.get(base) or [(list(paneles), base, "")]
    emitidos = []
    for claves, nombre, titulo in grupos:
        claves = [k for k in claves if k in paneles]
        if not claves:
            continue
        n = len(claves)
        pesos = [paneles[k].get("w", 1.0) for k in claves]
        fig, axs = plt.subplots(
            1, n, figsize=(ANCHO_TOTAL.get(n, 4.7 * n), ALTO_FIG),
            gridspec_kw={"width_ratios": pesos})
        axs = np.atleast_1d(axs)
        for ax, k in zip(axs, claves):
            paneles[k]["dibujar"](ax)
        sup = titulo + (("\n" + nota) if nota else "")
        # subplots_adjust en lugar de tight_layout + fig.text: con
        # savefig.bbox='tight' las coordenadas absolutas expanden el lienzo.
        fig.subplots_adjust(left=0.115 if n == 1 else 0.075,
                            right=0.975,
                            top=0.76 if nota else 0.83,
                            bottom=0.135, wspace=0.30)
        fig.suptitle(sup, y=0.985, fontsize=12.5, fontweight="bold",
                     va="top", linespacing=1.6)
        guardar_fig(fig, dir_diag, nombre, pdf)
        emitidos.append(nombre)
    return emitidos


# =============================================================================
def fig_convergencia(presentes, metricas, dir_diag, pdf):
    vis = {M: _valores_tablero(metricas[M]) for M in presentes}
    rh = {M: vis[M]["rhat_max_vis"] for M in presentes}

    def p_rhat(ax):
        _barras(ax, presentes, rh, r"$\hat{R}$ máximo",
                "más bajo = mejor · umbral 1.01",
                lineas=(1.01, 1.05), etiquetas_lineas=("1.01", "1.05"))
        ax.set_title(r"$\hat{R}$ máximo", fontsize=11, pad=15)
        rh_max = max([x for x in rh.values() if np.isfinite(x)] + [1.05])
        ax.set_ylim(0.995, rh_max * 1.03 + 0.006)

    def p_ess_min(ax):
        _barras(ax, presentes,
                {M: vis[M]["ess_bulk_min_vis"] for M in presentes},
                "ESS mínimo", "más alto = mejor; umbral 400", fmt="{:,.0f}",
                log=True, lineas=(100, 400), etiquetas_lineas=("100", "400"))
        ax.set_title("ESS mínimo", fontsize=11, pad=15)

    def p_ess_lt400(ax):
        _barras(ax, presentes,
                {M: 100 * metricas[M].get("prop_ess_lt_400", np.nan)
                 for M in presentes},
                "% de parámetros con ESS<400", "más bajo = mejor",
                fmt="{:.1f}%")
        ax.set_title("% de parámetros con ESS<400", fontsize=11, pad=15)

    def p_geweke(ax):
        _barras(ax, presentes,
                {M: 100 * metricas[M].get("geweke_frac_fdr", np.nan)
                 for M in presentes},
                "% Geweke significativo (FDR)", "más bajo = mejor; ref. 5%",
                fmt="{:.1f}%", lineas=(5,), etiquetas_lineas=("5%",))
        ax.set_title("% Geweke significativo (FDR)", fontsize=11, pad=15)

    dag = any(vis[M]["_dag"] for M in presentes)
    nota = ("(†) M4 sin ν: la no-identificación de ν es un hallazgo, "
            "no un fallo del muestreador") if dag else ""
    paneles = {
        "rhat":      {"dibujar": p_rhat,      "w": 1.0},
        "ess_min":   {"dibujar": p_ess_min,   "w": 1.0},
        "ess_lt400": {"dibujar": p_ess_lt400, "w": 1.0},
        "geweke":    {"dibujar": p_geweke,    "w": 1.0},
    }
    return _emitir_grupos("comp_02_convergencia", paneles, dir_diag, pdf, nota)


# =============================================================================
def fig_ajuste(presentes, metricas, dir_diag, pdf):
    def _p(clave, titulo, sentido, fmt="{:.3f}"):
        def dibujar(ax):
            _barras(ax, presentes,
                    {M: metricas[M].get(clave, np.nan) for M in presentes},
                    titulo, sentido, fmt=fmt)
            ax.set_title(titulo, fontsize=11, pad=15)
        return dibujar

    paneles = {
        "r2":       {"dibujar": _p("r2", "R² (media posterior)",
                                   "más alto = mejor"), "w": 1.0},
        "r2_bayes": {"dibujar": _p("r2_bayes",
                                   "R² bayesiano (Gelman et al. 2019)",
                                   "más alto = mejor"), "w": 1.0},
        "rmse":     {"dibujar": _p("rmse", "RMSE", "más bajo = mejor",
                                   "{:.2f}"), "w": 1.0},
        "mae":      {"dibujar": _p("mae", "MAE", "más bajo = mejor",
                                   "{:.2f}"), "w": 1.0},
    }
    nota = ("definiciones idénticas entre modelos bajo la capa uniforme")
    return _emitir_grupos("comp_03_ajuste", paneles, dir_diag, pdf, nota)


# =============================================================================
def fig_informacion(presentes, metricas, df_delta, dir_diag, pdf,
                    mismos_datos=True):
    def p_elpd(ax):
        for i, M in enumerate(presentes):
            for dx, (k, sk, mk) in enumerate(
                    [("elpd_loo", "se_elpd_loo", "o"),
                     ("elpd_waic", "se_elpd_waic", "s")]):
                v = metricas[M].get(k, np.nan)
                se = metricas[M].get(sk, np.nan)
                if np.isfinite(v):
                    ax.errorbar(i + (dx - 0.5) * 0.22, v,
                                yerr=(se if np.isfinite(se) else None),
                                fmt=mk, color=MODELOS[M]["color"], capsize=3,
                                ms=6.5)
        ax.plot([], [], "o", color=TINTA, label="elpd LOO")
        ax.plot([], [], "s", color=TINTA, label="elpd WAIC")
        ax.legend(fontsize=8.5, loc="best")
        ax.set_xticks(range(len(presentes)))
        ax.set_xticklabels(presentes)
        ax.set_title("elpd ± SE", fontsize=11, pad=15)
        ax.text(0.5, 1.015, "más alto = mejor predicción",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.3, color=GRIS, style="italic")
        ax.ticklabel_format(axis="y", style="plain")
        if not mismos_datos:
            ax.text(0.5, 0.02, "AVISO: N distinto entre modelos;\nestos elpd "
                    "NO son comparables entre sí", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=8.5,
                    color=MODELOS["M4"]["oscuro"], fontweight="bold")

    def p_delta(ax):
        tiene_loo = all(np.isfinite(metricas[M].get("elpd_loo", np.nan))
                        for M in presentes)
        clave_g = "elpd_loo" if tiene_loo else "elpd_waic"
        vals = {M: metricas[M].get(clave_g, np.nan) for M in presentes}
        fin = [M for M in presentes if np.isfinite(vals[M])]
        if not fin:
            return
        mejor = max(fin, key=lambda M: vals[M])
        otros = [M for M in presentes if M != mejor]
        ax.axvline(0, color=TINTA, lw=1.2, zorder=2)
        yy = np.arange(len(otros))
        datos = []
        for iy, M in enumerate(otros):
            par1, par2 = f"{mejor} vs {M}", f"{M} vs {mejor}"
            fila = None
            if len(df_delta):
                sel = df_delta[df_delta["par"].isin([par1, par2])]
                for cri in ("LOO", "WAIC"):
                    cand = sel[sel["criterio"] == cri]
                    if len(cand):
                        if np.isfinite(cand.iloc[0]["delta_elpd"]):
                            fila = cand.iloc[0]
                            break
                        fila = fila if fila is not None else cand.iloc[0]
            if fila is not None and np.isfinite(fila["delta_elpd"]):
                d = (fila["delta_elpd"] if fila["par"] == par1
                     else -fila["delta_elpd"])
                se = fila["se"]
                met_lbl = f"{fila['criterio']} · {fila['metodo']}"
            elif fila is not None:
                d, se, met_lbl = np.nan, np.nan, str(fila["metodo"])
            else:
                vo = metricas[M].get(clave_g, np.nan)
                if np.isfinite(vo):
                    d = vals[mejor] - vo
                    se = float(np.hypot(
                        metricas[mejor].get(f"se_{clave_g}", np.nan),
                        metricas[M].get(f"se_{clave_g}", np.nan)))
                    met_lbl = (("LOO" if clave_g == "elpd_loo" else "WAIC")
                               + " · no pareado (SE cuadrática)")
                else:
                    d, se = np.nan, np.nan
                    met_lbl = "sin criterio común"
            datos.append((iy, d, se, met_lbl, MODELOS[M]["color"],
                          np.isfinite(d)))
        xs = [d for _, d, _, _, _, ok in datos if ok]
        ses = [se for _, d, se, _, _, ok in datos if ok and np.isfinite(se)]
        if xs:
            span = max(xs + [0]) - min(xs + [0])
            span = span if span > 0 else (abs(xs[0]) or 1.0)
            pad = 0.30 * span + (2 * max(ses) if ses else 0)
            ax.set_xlim(min(xs + [0]) - pad, max(xs + [0]) + pad)
        ax.set_ylim(-0.6, len(otros) - 0.4)
        for iy, d, se, met_lbl, col, ok in datos:
            if ok:
                ax.errorbar(d, iy, xerr=(2 * se if np.isfinite(se) else None),
                            fmt="o", color=col, capsize=4, ms=8,
                            elinewidth=1.6, zorder=3)
                txt = (f"{d:,.0f} ± {se:,.0f}" if np.isfinite(se)
                       else f"{d:,.0f}")
                ax.annotate(txt, (d, iy), textcoords="offset points",
                            xytext=(0, 11), ha="center", fontsize=8.8,
                            color=TINTA, fontweight="medium")
                ax.annotate(met_lbl, (d, iy), textcoords="offset points",
                            xytext=(0, -15), ha="center", fontsize=7.3,
                            color=GRIS, style="italic")
            else:
                ax.annotate("no comparable · " + met_lbl, (0, iy),
                            textcoords="offset points", xytext=(8, 0),
                            ha="left", va="center", fontsize=8,
                            color=MODELOS["M4"]["oscuro"])
        ax.set_yticks(yy)
        ax.set_yticklabels([f"{mejor} − {M}" for M in otros])
        ax.set_xlabel(f"Δelpd   ( >0 favorece a {mejor} )")
        ax.set_title(f"Δelpd contra el mejor modelo ({mejor})", fontsize=11,
                     pad=15)
        ax.text(0.5, 1.015, "barras = ±2 SE · si cruza 0, no concluyente",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.3, color=GRIS, style="italic")

    def p_peff(ax):
        _barras(ax, presentes,
                {M: _valores_tablero(metricas[M])["p_eff"]
                 for M in presentes},
                "p_eff (complejidad efectiva)", "informativo", fmt="{:,.0f}")
        ax.set_title("p_eff (complejidad efectiva)", fontsize=11, pad=15)

    paneles = {
        "elpd":  {"dibujar": p_elpd,  "w": 1.15},
        "delta": {"dibujar": p_delta, "w": 1.0},
        "peff":  {"dibujar": p_peff,  "w": 0.85},
    }
    return _emitir_grupos("comp_04_informacion", paneles, dir_diag, pdf)


# =============================================================================
def fig_calibracion(presentes, metricas, dir_diag, pdf):
    def p_cobertura(ax):
        nom = np.array([0.50, 0.80, 0.95])
        ax.axhspan(-0.03, 0.03, color=AMARILLO, alpha=0.30, lw=0)
        ax.axhspan(-0.01, 0.01, color=VERDE, alpha=0.50, lw=0)
        ax.axhline(0, color=TINTA, lw=1)
        for M in presentes:
            v = np.array([metricas[M].get(k, np.nan)
                          for k in ("cob_50", "cob_80", "cob_95")], float)
            if np.any(np.isfinite(v)):
                ax.plot(nom, v - nom, "o-", color=MODELOS[M]["color"],
                        label=M, ms=6)
        ax.set_xticks(nom)
        ax.set_xticklabels(["50%", "80%", "95%"])
        ax.set_xlabel("cobertura nominal")
        ax.set_ylabel("cobertura empírica − nominal")
        ax.legend(fontsize=8.5)
        ax.set_title("Error de cobertura de los IP\n(0 = perfecta; verde "
                     "±1 pt, amarillo ±3 pt)", fontsize=10.5)

    def p_ppc_z(ax):
        xs = np.arange(len(ESTAD_PPC))
        etiq = ["media", "sd", "asimetría", "curtosis", "q05", "q95"]
        hay_z = any(metricas[M].get("ppc_z") for M in presentes)
        if hay_z:
            ax.axhspan(-2, 2, color=VERDE, alpha=0.35, lw=0)
            ax.axhline(2, color=GRIS, lw=0.8)
            ax.axhline(-2, color=GRIS, lw=0.8)
            ax.axhline(0, color=TINTA, lw=0.8)
            for i, M in enumerate(presentes):
                zz = metricas[M].get("ppc_z") or {}
                v = [zz.get(k, np.nan) for k in ESTAD_PPC]
                ax.plot(xs + (i - (len(presentes) - 1) / 2) * 0.14, v, "o",
                        color=MODELOS[M]["color"], label=M, ms=6)
            ax.set_yscale("symlog", linthresh=2)
            ax.set_ylabel("z = (T_obs − T̄_rep) / sd_rep   [symlog]")
            ax.set_title("PPC: discrepancia estandarizada por estadístico\n"
                         "(|z| ≤ 2 replica bien; con N grande los p-values "
                         "saturan a 0/1)", fontsize=10.5)
        else:
            ax.axhspan(0.05, 0.95, color=VERDE, alpha=0.35, lw=0)
            ax.axhline(0.05, color=GRIS, lw=0.8)
            ax.axhline(0.95, color=GRIS, lw=0.8)
            for i, M in enumerate(presentes):
                pv = metricas[M].get("ppc_pvals") or {}
                v = [pv.get(k, np.nan) for k in ESTAD_PPC]
                ax.plot(xs + (i - (len(presentes) - 1) / 2) * 0.14, v, "o",
                        color=MODELOS[M]["color"], label=M, ms=6)
            ax.set_ylim(-0.03, 1.03)
            ax.set_ylabel("p-value bayesiano")
            ax.set_title("PPC: p-values bayesianos por estadístico\n"
                         "(banda [0.05, 0.95] = replica bien ese rasgo)",
                         fontsize=10.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(etiq, rotation=20)
        ax.legend(fontsize=8.5, ncols=min(4, len(presentes)))

    def p_pit_ks(ax):
        hay_d = any(np.isfinite(metricas[M].get("ks_pit_d", np.nan))
                    for M in presentes)
        if hay_d:
            dvals = {M: metricas[M].get("ks_pit_d", np.nan)
                     for M in presentes}
            dmax = max([x for x in dvals.values() if np.isfinite(x)] + [0])
            refs, refs_lab = [], []
            for lv, lb in ((0.01, "0.01"), (0.05, "0.05")):
                if dmax >= lv * 0.6:
                    refs.append(lv)
                    refs_lab.append(lb)
            _barras(ax, presentes, dvals, "PIT: distancia KS (D)",
                    "más bajo = mejor · D pequeña ≈ uniforme", fmt="{:.4f}",
                    lineas=tuple(refs), etiquetas_lineas=tuple(refs_lab))
            ax.set_title("PIT: distancia KS (D)", fontsize=11, pad=15)
            if dmax < 0.01:
                ax.text(0.5, 0.92, "todos ≈ uniformes (D < 0.01)",
                        transform=ax.transAxes, ha="center", fontsize=8,
                        color=MODELOS["M3"]["oscuro"], style="italic")
        else:
            _barras(ax, presentes,
                    {M: metricas[M].get("ks_pit", np.nan)
                     for M in presentes},
                    "PIT: KS (p-valor)", "más alto = mejor · umbral 0.05",
                    fmt="{:.3f}", lineas=(0.05,), etiquetas_lineas=("0.05",))
            ax.set_title("PIT: KS (p-valor)", fontsize=11, pad=15)

    paneles = {
        "cobertura": {"dibujar": p_cobertura, "w": 1.0},
        "ppc_z":     {"dibujar": p_ppc_z,     "w": 1.0},
        "pit_ks":    {"dibujar": p_pit_ks,    "w": 0.9},
    }
    return _emitir_grupos("comp_05_calibracion", paneles, dir_diag, pdf)
