"""
reporte.py — Consolidación de resultados: tablas CSV y reporte de texto/markdown.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime

import numpy as np


def escribir_csv(filas, ruta, columnas=None):
    if not filas:
        return
    if columnas is None:
        columnas = list(filas[0].keys())
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columnas)
        w.writeheader()
        for f in filas:
            w.writerow({k: f.get(k, "") for k in columnas})


def tabla_convergencia_csv(diag_conv, dir_out):
    """Resumen de convergencia por grupo de parámetros."""
    filas = []

    def _resumen(nombre, arr):
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"grupo": nombre, "min": "", "mediana": "", "max": "", "n": 0}
        return {"grupo": nombre, "min": round(float(a.min()), 1),
                "mediana": round(float(np.median(a)), 1),
                "max": round(float(a.max()), 1), "n": int(a.size)}

    filas.append({**_resumen("ESS mu_t", diag_conv.ess_mu), "metrica": "ESS"})
    filas.append({**_resumen("ESS lambda2", diag_conv.ess_lambda), "metrica": "ESS"})
    theta_all = np.concatenate([np.asarray(v).ravel()
                                for v in diag_conv.ess_theta.values()])
    filas.append({**_resumen("ESS theta (multi-t)", theta_all), "metrica": "ESS"})
    filas.append({**_resumen("ESS s_kt", diag_conv.ess_s), "metrica": "ESS"})

    rhat_all = np.concatenate([
        np.atleast_1d(diag_conv.rhat_mu),
        np.atleast_1d(diag_conv.rhat_lambda),
        np.concatenate([np.atleast_1d(v) for v in diag_conv.rhat_theta.values()]),
    ])
    filas.append({**_resumen("R-hat (todos)", rhat_all), "metrica": "Rhat"})

    escribir_csv(filas, os.path.join(dir_out, "convergencia_resumen.csv"),
                 columnas=["metrica", "grupo", "min", "mediana", "max", "n"])


def tabla_comparacion_modelo_csv(waic_res, loo_res, dir_out):
    """Tabla de comparación de modelo (elpd/WAIC/LOO/p_eff)."""
    filas = [{
        "criterio": "WAIC",
        "elpd": round(waic_res["elpd_waic"], 2),
        "valor_-2elpd": round(waic_res["waic"], 2),
        "p_eff": round(waic_res["p_waic"], 2),
        "se": round(waic_res["se"], 2),
        "obs_problematicas": waic_res.get("n_p_waic_alto", ""),
    }]
    if loo_res is not None:
        filas.append({
            "criterio": f"PSIS-LOO ({loo_res.get('motor','')})",
            "elpd": round(loo_res["elpd_loo"], 2),
            "valor_-2elpd": round(loo_res["loo"], 2),
            "p_eff": round(loo_res["p_loo"], 2),
            "se": round(loo_res["se"], 2),
            "obs_problematicas": loo_res.get("n_k_altos", ""),
        })
    escribir_csv(filas, os.path.join(dir_out, "comparacion_modelo.csv"),
                 columnas=["criterio", "elpd", "valor_-2elpd", "p_eff", "se",
                           "obs_problematicas"])


def generar_reporte_md(ctx, dir_out):
    """
    ctx: diccionario con todas las piezas del diagnóstico. Genera un .md
    legible en español, priorizando la sección dinámica.
    """
    L = []
    ap = L.append
    meta = ctx["meta"]

    ap("# Diagnóstico del Modelo 3 — Modelo Normal Dinámico con LASSO Bayesiano\n")
    ap(f"*Generado: {datetime.now():%Y-%m-%d %H:%M}*  ")
    ap(f"*Muestras posteriores analizadas: {ctx['n_save']}*  ")
    ap(f"*Dimensiones: p={meta.p} (1 + {meta.p_E} + {meta.p_M} + {meta.p_D}), "
       f"m={meta.m} departamentos, Tt={meta.Tt} periodos*\n")

    if ctx.get("modo_rapido"):
        ap("> ⚠ **Ejecución en modo `--rapido`**: submuestreo de cadenas y "
           "observaciones. Los valores son indicativos, no definitivos.\n")

    # ---------------------------------------------------------------
    # 1. RESULTADOS DINÁMICOS (prioridad)
    # ---------------------------------------------------------------
    ap("\n## 1. Resultados dinámicos\n")

    if "resumen_T" in ctx:
        rt = ctx["resumen_T"]
        n_sig = sum(1 for f in rt if f["ic95_no_cero"] == "*")
        n_pd = sum(1 for f in rt if f["pd_alta"] == "*")
        ap(f"En el periodo final (t=T), **{n_sig} de {len(rt)}** coeficientes "
           f"tienen intervalo de credibilidad 95% que no cruza cero, y "
           f"**{n_pd}** tienen probabilidad de dirección > 0.95.\n")

        # top efectos
        ord_rt = sorted(rt, key=lambda f: abs(f["media"]), reverse=True)[:10]
        ap("\n**Top 10 efectos por magnitud (t=T):**\n")
        ap("| Parámetro | Media | SD | IC95 | P(>0) | Signif. |")
        ap("|---|---:|---:|---|---:|:---:|")
        for f in ord_rt:
            sig = f["ic95_no_cero"] or f["pd_alta"] or ""
            ap(f"| {f['parametro']} | {f['media']:.3f} | {f['sd']:.3f} | "
               f"[{f['q025']:.3f}, {f['q975']:.3f}] | {f['p_positivo']:.2f} | {sig} |")

    if "innov" in ctx:
        ap("\n### Actividad del LASSO dinámico\n")
        ap("Innovaciones ω = θ_t − θ_{t-1} con probabilidad alta de cambio "
           "(P(|ω| > 0.1·SD) > 0.5):\n")
        for B in ("E", "M", "D"):
            d = ctx["innov"][B]
            ap(f"- **Bloque {B}**: {d['n_cambios_altos']} de {d['n_total']} "
               f"celdas (coef×periodo) con cambio significativo "
               f"({100*d['n_cambios_altos']/d['n_total']:.1f}%).")

    if "dvar" in ctx:
        ap("\n### Varianza departamental dinámica\n")
        dvar = ctx["dvar"]
        dep = dvar["depto_ids"]
        ap("Departamentos con mayor varianza promedio s̄:\n")
        for i in dvar["orden_desc"][:5]:
            lbl = dep[i] if dep and i < len(dep) else f"k={i+1}"
            ap(f"- {lbl}: s̄ = {dvar['s_prom'][i]:.3f}")

    # ---------------------------------------------------------------
    # 2. CONVERGENCIA
    # ---------------------------------------------------------------
    ap("\n## 2. Convergencia MCMC\n")
    dc = ctx["diag_conv"]
    theta_all = np.concatenate([np.asarray(v).ravel()
                                for v in dc.ess_theta.values()])
    ess_min_theta = float(np.nanmin(theta_all)) if theta_all.size else np.nan
    ess_min_s = float(np.nanmin(dc.ess_s)) if np.asarray(dc.ess_s).size else np.nan
    rhat_all = np.concatenate([
        np.atleast_1d(dc.rhat_mu), np.atleast_1d(dc.rhat_lambda),
        np.concatenate([np.atleast_1d(v) for v in dc.rhat_theta.values()])])
    rhat_max = float(np.nanmax(rhat_all)) if rhat_all.size else np.nan
    n_gew = int(np.sum(np.asarray(dc.geweke_pfdr) < 0.05))
    n_gew_tot = int(np.sum(np.isfinite(dc.geweke_pfdr)))

    ap(f"- **ESS mínimo (intercepto μ_t):** {np.nanmin(dc.ess_mu):.0f}")
    ap(f"- **ESS mínimo (θ, multi-periodo):** {ess_min_theta:.0f}")
    ap(f"- **ESS mínimo (s_kt):** {ess_min_s:.0f}")
    ap(f"- **R̂ máximo (split rank-normalizado):** {rhat_max:.3f}")
    ap(f"- **Geweke significativos tras FDR:** {n_gew}/{n_gew_tot}")
    if "raftery" in ctx and ctx["raftery"]:
        rl = ctx["raftery"]
        ap(f"- **Raftery-Lewis (λ²):** N mínimo recomendado {rl['N_min']}, "
           f"N requerido {rl['N']} (se tienen {ctx['n_save']}); "
           f"factor de dependencia I={rl['I']:.2f}")

    ap("\n**Interpretación:**")
    if ess_min_theta >= 400:
        ap("- ✓ ESS adecuado en θ — la cadena mezcla razonablemente.")
    else:
        ap("- ⚠ ESS bajo en θ — considerar más iteraciones o reparametrizar.")
    if rhat_max < 1.05:
        ap("- ✓ R̂ < 1.05 — sin evidencia de no-estacionariedad.")
    else:
        ap("- ⚠ R̂ ≥ 1.05 — posible no-convergencia.")

    # tau2
    if "tau2" in ctx:
        ap("\n### Mezclado de τ² (LASSO local)\n")
        ap("| Bloque | ESS min | ESS mediana | ESS max | ESS<200 | Bimodal |")
        ap("|---|---:|---:|---:|---:|---:|")
        for B in ("E", "M", "D"):
            d = ctx["tau2"][B]
            bi = ctx.get("bimod", {}).get(B, {})
            ap(f"| {B} | {d['ess_min']:.0f} | {d['ess_med']:.0f} | "
               f"{d['ess_max']:.0f} | {d['n_low_ess']} | "
               f"{bi.get('n_bimodal','—')} |")

    # ---------------------------------------------------------------
    # 3. AJUSTE PREDICTIVO
    # ---------------------------------------------------------------
    ap("\n## 3. Ajuste predictivo\n")
    if "ppc" in ctx:
        pv = ctx["ppc"]["pval"]
        ap("**P-values bayesianos (ideal 0.1 < p < 0.9):**\n")
        ap("| Estadístico | p_B | |")
        ap("|---|---:|:---:|")
        nombres = {"media": "Media", "sd": "Desv. estándar", "skew": "Asimetría",
                   "kurt": "Curtosis", "q05": "Cuantil 5%", "q95": "Cuantil 95%"}
        for k in ["media", "sd", "skew", "kurt", "q05", "q95"]:
            flag = "⚠" if (pv[k] < 0.05 or pv[k] > 0.95) else "✓"
            ap(f"| {nombres[k]} | {pv[k]:.3f} | {flag} |")

    if "residuos" in ctx:
        r = ctx["residuos"]
        ap(f"\n- **RMSE:** {r['rmse']:.3f}  |  **MAE:** {r['mae']:.3f}")
        ap(f"- Tendencia temporal del residuo medio: p={r['p_trend_t']:.4f} "
           f"{'⚠ sesgo temporal' if r['p_trend_t'] < 0.05 else '✓'}")
        ap(f"- Heteroscedasticidad (|resid|~ŷ): p={r['p_het']:.4f} "
           f"{'⚠ varianza no capturada' if r['p_het'] < 0.05 else '✓'}")

    if "pit" in ctx and ctx["pit"].get("ks_p") is not None:
        ksp = ctx["pit"]["ks_p"]
        ap(f"- **Calibración PIT (KS vs U(0,1)):** p={ksp:.4f} "
           f"{'✓ calibrado' if ksp > 0.05 else '⚠ calibración subóptima'}")

    # comparación de modelo
    if "waic" in ctx:
        ap("\n### Criterios de comparación de modelos\n")
        w = ctx["waic"]
        ap(f"- **WAIC:** elpd = {w['elpd_waic']:.2f} (±{w['se']:.2f}), "
           f"p_eff = {w['p_waic']:.2f}, WAIC = {w['waic']:.2f}")
        if ctx.get("loo"):
            lo = ctx["loo"]
            ap(f"- **PSIS-LOO** ({lo.get('motor','')}): elpd = "
               f"{lo['elpd_loo']:.2f} (±{lo['se']:.2f}), p_eff = {lo['p_loo']:.2f}"
               f"; observaciones con k>0.7: {lo['n_k_altos']}")
        else:
            ap("- PSIS-LOO omitido (`--sin-loo`); se usa WAIC vectorizado.")

    ap("\n---\n*Los archivos PDF/CSV asociados están en el mismo directorio "
       "de salida.*\n")

    ruta = os.path.join(dir_out, "reporte_diagnostico_M3.md")
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return ruta
