# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# @  COMPARACION DE MODELOS M1-M4 (tesis Saber 11)
# @  Convergencia | Ajuste | Calibracion | Criterios de informacion | Ranking
# _____________________________________________________________________________

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln, logsumexp, ndtr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# =============================================================================
# 1. REGISTRO DE MODELOS, ESTILO Y UMBRALES
# =============================================================================
MODELOS = {
    "M1": {"nombre": "M1 Normal estático",    "color": "#1F4E6B",
           "oscuro": "#143247"},   # azul náutico
    "M2": {"nombre": "M2 t-Student estático", "color": "#C58A2E",
           "oscuro": "#8A5E17"},   # ocre / mostaza envejecido
    "M3": {"nombre": "M3 Normal dinámico",    "color": "#2E7D74",
           "oscuro": "#1C4F49"},   # teal profundo
    "M4": {"nombre": "M4 t-Student dinámico", "color": "#8C3B4A",
           "oscuro": "#5E2530"},   # granate / burdeos
}
GRIS = "#5B6670"          # gris pizarra (más elegante que el gris neutro)
GRIS_CLARO = "#E4E7EA"    # relleno "no disponible"
TINTA = "#1E2A33"         # azul casi negro para títulos y texto principal
# Semáforo en tonos apagados que combinan con la paleta náutica.
VERDE, AMARILLO, ROJO = "#A9C7A0", "#EAD79C", "#D8A0A0"
VERDE_F, AMARILLO_F, ROJO_F = "#8FB183", "#DBC57E", "#C68686"   # variantes +sat
SECCION = "#26333D"       # banda de sección del tablero (azul tinta, no negro)

U_RHAT_OK, U_RHAT_AVISO = 1.01, 1.05
U_ESS_OK, U_ESS_MIN = 400, 100
U_COB_OK, U_COB_AVISO = 0.01, 0.03          # |cobertura - nominal|
ESTAD_PPC = ["media", "sd", "asimetria", "curtosis", "q05", "q95"]

# --- separacion de paneles de las figuras comparativas --------------------
ANCHO_TOTAL = {1: 6.6, 2: 9.4}
ALTO_FIG = 4.5

# --- agrupacion de paneles -------------------------------------------------
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




def estilo():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "text.color": TINTA, "axes.labelcolor": TINTA,
        "axes.edgecolor": "#9AA4AC", "axes.titlecolor": TINTA,
        "xtick.color": TINTA, "ytick.color": TINTA,
        "xtick.labelcolor": TINTA, "ytick.labelcolor": TINTA,
        "axes.titlesize": 11.5, "axes.titleweight": "semibold",
        "axes.labelsize": 10.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.22, "grid.linewidth": 0.6, "grid.color": "#B7BFC6",
        "legend.frameon": False,
    })


def guardar_fig(fig, dir_diag, nombre, tambien_pdf=False):
    dir_diag = Path(dir_diag); dir_diag.mkdir(parents=True, exist_ok=True)
    ruta = dir_diag / f"{nombre}.png"
    fig.savefig(ruta)
    if tambien_pdf:
        fig.savefig(dir_diag / f"{nombre}.pdf")
    plt.close(fig)
    print(f"  [fig] {ruta}")


def _variantes_nombre(ruta):
    """Genera nombres alternativos plausibles para un artefacto.

    Los pipelines de cada modelo no son homogeneos en la nomenclatura: unos
    escriben 'tabla_ppc.csv' y otros 'tabla_ppc_M4.csv'. En vez de fijar un
    solo nombre, se prueban variantes con y sin el sufijo de modelo, para que
    el lector funcione con cualquiera de las dos convenciones.
    """
    ruta = Path(ruta)
    tallo, ext = ruta.stem, ruta.suffix
    nombres = [ruta.name]
    # sufijos de modelo: agregar si no lo tiene, quitar si lo tiene
    import re as _re
    m = _re.search(r"_(M[1-4])$", tallo, flags=_re.IGNORECASE)
    if m:
        base = tallo[: m.start()]
        nombres.append(f"{base}{ext}")
        for suf in ("M1", "M2", "M3", "M4"):
            nombres.append(f"{base}_{suf}{ext}")
    else:
        for suf in ("M1", "M2", "M3", "M4"):
            nombres.append(f"{tallo}_{suf}{ext}")
    # sin duplicados, preservando orden
    vistos, out = set(), []
    for n in nombres:
        if n not in vistos:
            vistos.add(n); out.append(n)
    return out


def _resolver(ruta):
    """Devuelve la primera variante existente del archivo, o None."""
    ruta = Path(ruta)
    carpeta = ruta.parent
    for nombre in _variantes_nombre(ruta):
        cand = carpeta / nombre
        if cand.exists():
            return cand
    # ultimo recurso: busqueda insensible a mayusculas en la carpeta
    try:
        objetivo = {n.lower() for n in _variantes_nombre(ruta)}
        for f in carpeta.iterdir():
            if f.is_file() and f.name.lower() in objetivo:
                return f
    except Exception:
        pass
    return None


def _csv(ruta):
    real = _resolver(ruta)
    if real is None:
        return None
    try:
        return pd.read_csv(real)
    except Exception as e:
        print(f"  AVISO: no pude leer {real.name}: {e}")
        return None


def _num(x, default=np.nan):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return default


def _re1(patron, texto, grupo=1, default=np.nan):
    m = re.search(patron, texto)
    return _num(m.group(grupo)) if m else default


def _fmt(v, dec=2, miles=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    if miles:
        return f"{v:,.{dec}f}"
    return f"{v:.{dec}f}"


# =============================================================================
# 2. CAPA A: LECTORES DE ARTEFACTOS
# =============================================================================

def _pvals_desde_tabla(df, col_est, col_p, mapa):
    out = {}
    for _, r in df.iterrows():
        k = mapa.get(str(r[col_est]).strip())
        if k:
            out[k] = _num(r[col_p])
    return out


def _n_extremos(pvals):
    vs = [v for v in pvals.values() if np.isfinite(v)]
    if not vs:
        return np.nan
    return int(sum((v < 0.05) or (v > 0.95) for v in vs))


def _z_ppc(df, col_est, mapa, col_obs, col_med, col_sd=None,
           col_q025=None, col_q975=None):
    """Discrepancia estandarizada z = (T_obs - T_rep_medio) / sd(T_rep).

    Con N grande los p-values bayesianos saturan a 0/1 ante discrepancias
    practicamente nulas; z conserva la MAGNITUD y si es comparable entre
    modelos ajustados a los mismos datos. Si no hay sd_rep, se aproxima con
    (q975 - q025)/3.92.
    """
    out = {}
    for _, r in df.iterrows():
        k = mapa.get(str(r[col_est]).strip())
        if not k:
            continue
        obs, med = _num(r.get(col_obs)), _num(r.get(col_med))
        sd = _num(r.get(col_sd)) if col_sd else np.nan
        if not np.isfinite(sd) and col_q025 and col_q975:
            sd = (_num(r.get(col_q975)) - _num(r.get(col_q025))) / 3.92
        if np.isfinite(obs) and np.isfinite(med) and np.isfinite(sd) and sd > 0:
            out[k] = float((obs - med) / sd)
    return out


def _agregar_z(met, z):
    if z:
        met["ppc_z"] = z
        vs = [abs(v) for v in z.values() if np.isfinite(v)]
        if vs:
            met["ppc_z_max"] = float(max(vs))


def _se_desde_khat(ruta, met, fuentes, prefijo):
    """SE(elpd) = sqrt(N var(elpd_i)) desde el CSV puntual khat_MX.csv."""
    df = _csv(ruta)
    if df is None:
        return
    n = len(df)
    for col, clave in (("elpd_waic_i", "se_elpd_waic"),
                       ("elpd_loo_i", "se_elpd_loo")):
        if col in df.columns:
            met[clave] = float(np.sqrt(n * df[col].var(ddof=1)))
            fuentes[clave] = f"{prefijo} (recalculado del puntual)"
    if "khat_i" in df.columns and not np.isfinite(met.get("prop_khat_gt_07",
                                                          np.nan)):
        met["prop_khat_gt_07"] = float((df["khat_i"] > 0.7).mean())
        fuentes["prop_khat_gt_07"] = prefijo


def leer_artefactos_m1(d):
    d = Path(d); met, fu, notas = {}, {}, []
    f = d / "resumen_convergencia_M1.csv"
    df = _csv(f)
    if df is not None:
        met.update(rhat_max=float(np.nanmax(df["rhat"])),
                   ess_bulk_min=float(np.nanmin(df["ess_bulk"])),
                   ess_tail_min=float(np.nanmin(df["ess_tail"])),
                   prop_ess_lt_400=float((df["ess_bulk"] < 400).mean()),
                   geweke_frac_fdr=float((df["geweke_p_fdr"] < 0.05).mean()),
                   n_par_conv=int(len(df)))
        for k in ("rhat_max", "ess_bulk_min", "ess_tail_min",
                  "prop_ess_lt_400", "geweke_frac_fdr", "n_par_conv"):
            fu[k] = f.name
    df = _csv(d / "ajuste_M1.csv")
    if df is not None:
        r = df.iloc[0]
        met.update(elpd_waic=_num(r.get("elpd_WAIC")),
                   p_waic=_num(r.get("p_WAIC")),
                   elpd_loo=_num(r.get("elpd_LOO")),
                   p_loo=_num(r.get("p_LOO")),
                   prop_khat_gt_07=_num(r.get("prop_khat_gt_0.7")),
                   dic=_num(r.get("DIC")), p_dic=_num(r.get("p_DIC")),
                   N=_num(r.get("N")), n_save=_num(r.get("n_save")))
        for k in ("elpd_waic", "p_waic", "elpd_loo", "p_loo",
                  "prop_khat_gt_07", "dic", "p_dic", "N", "n_save"):
            fu[k] = "ajuste_M1.csv"
    _khat = _resolver(d / "khat_M1.csv")
    if _khat is not None:
        met["ruta_khat"] = str(_khat)
        _se_desde_khat(met["ruta_khat"], met, fu, _khat.name)
    df = _csv(d / "ppc_cobertura_M1.csv")
    if df is not None and "GLOBAL" in set(df["nivel"]):
        g = df[df["nivel"] == "GLOBAL"].iloc[0]
        met.update(cob_50=_num(g["cob_50"]), cob_80=_num(g["cob_80"]),
                   cob_95=_num(g["cob_95"]))
        fu.update(cob_50="ppc_cobertura_M1.csv",
                  cob_80="ppc_cobertura_M1.csv", cob_95="ppc_cobertura_M1.csv")
    df = _csv(d / "ppc_estadisticos_M1.csv")
    if df is not None:
        met["ppc_pvals"] = _pvals_desde_tabla(
            df, "estadistico", "p_bayes",
            {"media": "media", "sd": "sd", "asimetria": "asimetria",
             "curtosis": "curtosis", "q05": "q05", "q95": "q95"})
        met["n_ppc_extremos"] = _n_extremos(met["ppc_pvals"])
        _agregar_z(met, _z_ppc(df, "estadistico",
                               {"media": "media", "sd": "sd",
                                "asimetria": "asimetria", "curtosis": "curtosis",
                                "q05": "q05", "q95": "q95"},
                               "T_obs", "T_rep_medio",
                               col_q025="T_rep_q025", col_q975="T_rep_q975"))
        fu["ppc_pvals"] = "ppc_estadisticos_M1.csv"
        fu["ppc_z"] = "ppc_estadisticos_M1.csv (sd aprox. por cuantiles)"
        fu["n_ppc_extremos"] = "ppc_estadisticos_M1.csv"
    df = _csv(d / "resumen_ejecutivo_M1.csv")
    if df is not None:
        rr = dict(zip(df["indicador"], df["valor"]))
        if "r2_bayes" in rr:
            met["r2_bayes"] = _num(rr["r2_bayes"]); fu["r2_bayes"] = "resumen_ejecutivo_M1.csv"
    return met, fu, notas


def leer_artefactos_m2(d):
    d = Path(d); met, fu, notas = {}, {}, []
    fj = _resolver(d / "metricas_M2.json") or (d / "metricas_M2.json")
    J = None
    if fj.exists():
        try:
            J = json.loads(fj.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  AVISO: metricas_M2.json ilegible: {e}")
    if J:
        c = J.get("convergencia") or {}
        met.update(rhat_max=_num(c.get("rhat_max")),
                   ess_bulk_min=_num(c.get("ess_bulk_min")),
                   ess_tail_min=_num(c.get("ess_tail_min")),
                   n_par_conv=int(c.get("n_parametros", 0)) or np.nan)
        if c.get("n_parametros"):
            met["prop_ess_lt_400"] = _num(c.get("n_ess_bulk_bajos")) / c["n_parametros"]
            met["geweke_frac_fdr"] = _num(c.get("n_geweke_fdr")) / c["n_parametros"]
        a = J.get("ajuste") or {}
        met.update(elpd_waic=_num(a.get("elpd_WAIC")), p_waic=_num(a.get("p_WAIC")),
                   elpd_loo=_num(a.get("elpd_LOO")), p_loo=_num(a.get("p_LOO")),
                   prop_khat_gt_07=_num(a.get("prop_khat_gt_0.7")),
                   dic=_num(a.get("DIC")), p_dic=_num(a.get("p_DIC")),
                   N=_num(a.get("N")), n_save=_num(a.get("n_save")))
        pc = J.get("ppc") or {}
        met.update(cob_50=_num(pc.get("cob_50")), cob_80=_num(pc.get("cob_80")),
                   cob_95=_num(pc.get("cob_95")),
                   r2_bayes=_num(pc.get("r2_t_medio")),
                   rmse=_num(pc.get("rmse_medio")), mae=_num(pc.get("mae_medio")))
        if isinstance(pc.get("holdout"), dict):
            met["holdout"] = pc["holdout"]
        for k in met:
            fu.setdefault(k, "metricas_M2.json")
        if np.isfinite(_num((J.get("convergencia") or {}).get("nu_mediana"))):
            met["nu_mediana"] = _num(J["convergencia"]["nu_mediana"])
            fu["nu_mediana"] = "metricas_M2.json"
    # respaldos CSV (si falta el JSON o alguna pieza)
    if not np.isfinite(met.get("rhat_max", np.nan)):
        df = _csv(d / "resumen_convergencia_M2.csv")
        if df is not None:
            met.update(rhat_max=float(np.nanmax(df["rhat"])),
                       ess_bulk_min=float(np.nanmin(df["ess_bulk"])),
                       ess_tail_min=float(np.nanmin(df["ess_tail"])),
                       prop_ess_lt_400=float((df["ess_bulk"] < 400).mean()),
                       geweke_frac_fdr=float((df["geweke_p_fdr"] < 0.05).mean()),
                       n_par_conv=int(len(df)))
            for k in ("rhat_max", "ess_bulk_min", "ess_tail_min",
                      "prop_ess_lt_400", "geweke_frac_fdr", "n_par_conv"):
                fu[k] = "resumen_convergencia_M2.csv"
    if not np.isfinite(met.get("elpd_waic", np.nan)):
        df = _csv(d / "ajuste_M2.csv")
        if df is not None:
            r = df.iloc[0]
            met.update(elpd_waic=_num(r.get("elpd_WAIC")), p_waic=_num(r.get("p_WAIC")),
                       elpd_loo=_num(r.get("elpd_LOO")), p_loo=_num(r.get("p_LOO")),
                       prop_khat_gt_07=_num(r.get("prop_khat_gt_0.7")),
                       dic=_num(r.get("DIC")), p_dic=_num(r.get("p_DIC")),
                       N=_num(r.get("N")), n_save=_num(r.get("n_save")))
            for k in ("elpd_waic", "elpd_loo", "dic", "N", "n_save"):
                fu[k] = "ajuste_M2.csv"
    _khat = _resolver(d / "khat_M2.csv")
    if _khat is not None:
        met["ruta_khat"] = str(_khat)
        _se_desde_khat(met["ruta_khat"], met, fu, _khat.name)
    df = _csv(d / "ppc_estadisticos_M2.csv")
    if df is not None:
        met["ppc_pvals"] = _pvals_desde_tabla(
            df, "estadistico", "p_bayes",
            {"media": "media", "sd": "sd", "asimetria": "asimetria",
             "curtosis": "curtosis", "q05": "q05", "q95": "q95"})
        met["n_ppc_extremos"] = _n_extremos(met["ppc_pvals"])
        _agregar_z(met, _z_ppc(df, "estadistico",
                               {"media": "media", "sd": "sd",
                                "asimetria": "asimetria", "curtosis": "curtosis",
                                "q05": "q05", "q95": "q95"},
                               "T_obs", "T_rep_medio",
                               col_q025="T_rep_q025", col_q975="T_rep_q975"))
        fu["ppc_pvals"] = "ppc_estadisticos_M2.csv"
        fu["ppc_z"] = "ppc_estadisticos_M2.csv (sd aprox. por cuantiles)"
        fu["n_ppc_extremos"] = "ppc_estadisticos_M2.csv"
    if np.isfinite(met.get("nu_mediana", np.nan)) and met["nu_mediana"] < 30:
        notas.append(f"M2: nu mediana = {met['nu_mediana']:.1f} "
                     "(la robustez t esta activa).")
    return met, fu, notas


def leer_artefactos_m3(d):
    d = Path(d); met, fu, notas = {}, {}, []
    df = _csv(d / "convergencia_resumen.csv")
    if df is not None:
        ess = df[df["metrica"] == "ESS"]
        if len(ess):
            met["ess_bulk_min"] = float(pd.to_numeric(ess["min"]).min())
            met["n_par_conv"] = int(pd.to_numeric(ess["n"]).sum())
            fu["ess_bulk_min"] = "convergencia_resumen.csv"
            fu["n_par_conv"] = "convergencia_resumen.csv"
    fmd = _resolver(d / "reporte_diagnostico_M3.md") or (d / "reporte_diagnostico_M3.md")
    if fmd.exists():
        tx = fmd.read_text(encoding="utf-8")
        met["rhat_max"] = _re1(r"R.\s*m[aá]ximo[^:]*:\*\*\s*([0-9.]+)", tx)
        g = re.search(r"Geweke significativos tras FDR:\*\*\s*(\d+)\s*/\s*(\d+)", tx)
        if g:
            met["geweke_frac_fdr"] = int(g.group(1)) / max(int(g.group(2)), 1)
            fu["geweke_frac_fdr"] = fmd.name
        met["rmse"] = _re1(r"\*\*RMSE:\*\*\s*([0-9.]+)", tx)
        met["mae"] = _re1(r"\*\*MAE:\*\*\s*([0-9.]+)", tx)
        met["ks_pit"] = _re1(r"KS vs U\(0,1\)\):\*\*\s*p=([0-9.]+)", tx)
        pv = {}
        mapa = {"Media": "media", "Desv. estándar": "sd", "Asimetría": "asimetria",
                "Curtosis": "curtosis", "Cuantil 5%": "q05", "Cuantil 95%": "q95"}
        for lin in tx.splitlines():
            m = re.match(r"\|\s*([^|]+?)\s*\|\s*([0-9.]+)\s*\|", lin)
            if m and m.group(1).strip() in mapa:
                pv[mapa[m.group(1).strip()]] = float(m.group(2))
        if pv:
            met["ppc_pvals"] = pv
            met["n_ppc_extremos"] = _n_extremos(pv)
            fu["ppc_pvals"] = fmd.name
            fu["n_ppc_extremos"] = fmd.name
        for k in ("rhat_max", "rmse", "mae", "ks_pit"):
            if np.isfinite(met.get(k, np.nan)):
                fu[k] = fmd.name
    df = _csv(d / "comparacion_modelo.csv")
    if df is not None:
        for _, r in df.iterrows():
            cri = str(r["criterio"])
            if cri.startswith("WAIC"):
                met.update(elpd_waic=_num(r["elpd"]), p_waic=_num(r["p_eff"]),
                           se_elpd_waic=_num(r["se"]))
                fu.update(elpd_waic="comparacion_modelo.csv",
                          p_waic="comparacion_modelo.csv",
                          se_elpd_waic="comparacion_modelo.csv")
            elif cri.startswith("PSIS-LOO"):
                met.update(elpd_loo=_num(r["elpd"]), p_loo=_num(r["p_eff"]),
                           se_elpd_loo=_num(r["se"]),
                           khat_n_gt_07=_num(r.get("obs_problematicas")))
                fu.update(elpd_loo="comparacion_modelo.csv",
                          p_loo="comparacion_modelo.csv",
                          se_elpd_loo="comparacion_modelo.csv")
    for nombre in ("ajuste_dic_M3.csv", "dic_M3.csv"):
        df = _csv(d / nombre)
        if df is None:
            continue
        fila = None
        if "variante" in df.columns:
            sel = df[df["variante"].astype(str).str.strip() == "media_parametros"]
            fila = sel.iloc[0] if len(sel) else df.iloc[0]
        elif len(df):
            fila = df.iloc[0]
        if fila is not None and np.isfinite(_num(fila.get("DIC"))):
            met["dic"] = _num(fila.get("DIC"))
            fu["dic"] = f"{nombre} (variante media_parametros)"
            if np.isfinite(_num(fila.get("p_DIC"))):
                met["p_dic"] = _num(fila.get("p_DIC"))
                fu["p_dic"] = nombre
            if np.isfinite(_num(fila.get("N"))) and not np.isfinite(
                    met.get("N", np.nan)):
                met["N"] = _num(fila.get("N"))
                fu["N"] = nombre
        break
    notas.append("M3: el pipeline no persiste cobertura ni R2; "
                 "use --m3-mcmc para calcularlos de forma uniforme.")
    return met, fu, notas


def leer_artefactos_m4(d):
    d = Path(d); met, fu, notas = {}, {}, []
    df = _csv(d / "tabla_ajuste_waic.csv")
    if df is not None:
        r = df.iloc[0]
        met.update(elpd_waic=_num(r.get("elpd")), p_waic=_num(r.get("p_WAIC")),
                   se_elpd_waic=_num(r.get("se_WAIC")) / 2.0,
                   dic=_num(r.get("DIC")), p_dic=_num(r.get("p_DIC")),
                   rmse=_num(r.get("RMSE")), mae=_num(r.get("MAE")),
                   r2=_num(r.get("R2")), r2_bayes=_num(r.get("R2_bayes_mediana")))
        for k in ("elpd_waic", "p_waic", "se_elpd_waic", "dic", "p_dic",
                  "rmse", "mae", "r2", "r2_bayes"):
            fu[k] = "tabla_ajuste_waic.csv"
    # convergencia: monitoreados + malla theta, con y sin nu
    dm = _csv(d / "tabla_convergencia_monitoreados.csv")
    dt = _csv(d / "tabla_convergencia_theta.csv")
    if dm is not None or dt is not None:
        ess_all, rhat_all = [], []
        ess_sn, rhat_sn = [], []
        if dm is not None:
            ess_all += list(dm["ess"]); rhat_all += list(dm["rhat"])
            sn = dm[dm["parametro"] != "nu"]
            ess_sn += list(sn["ess"]); rhat_sn += list(sn["rhat"])
            fila_nu = dm[dm["parametro"] == "nu"]
            if len(fila_nu):
                met["nu_ess"] = _num(fila_nu["ess"].iloc[0])
                met["nu_rhat"] = _num(fila_nu["rhat"].iloc[0])
        if dt is not None:
            ess_all += list(dt["ess"]); rhat_all += list(dt["rhat"])
            ess_sn += list(dt["ess"]); rhat_sn += list(dt["rhat"])
        met.update(ess_bulk_min=float(np.nanmin(ess_all)),
                   rhat_max=float(np.nanmax(rhat_all)),
                   ess_bulk_min_sin_nu=float(np.nanmin(ess_sn)),
                   rhat_max_sin_nu=float(np.nanmax(rhat_sn)),
                   prop_ess_lt_400=float(np.nanmean(np.asarray(ess_all) < 400)),
                   n_par_conv=int(np.isfinite(np.asarray(rhat_all, float)).sum()))
        for k in ("ess_bulk_min", "rhat_max", "ess_bulk_min_sin_nu",
                  "rhat_max_sin_nu", "prop_ess_lt_400", "n_par_conv"):
            fu[k] = "tabla_convergencia_*.csv"
    df = _csv(d / "tabla_ppc.csv")
    if df is not None:
        met["ppc_pvals"] = _pvals_desde_tabla(
            df, "estadistico", "p_value_bayes",
            {"media": "media", "sd": "sd", "asimetria": "asimetria",
             "curtosis": "curtosis", "q05": "q05", "q95": "q95"})
        met["n_ppc_extremos"] = _n_extremos(met["ppc_pvals"])
        _agregar_z(met, _z_ppc(df, "estadistico",
                               {"media": "media", "sd": "sd",
                                "asimetria": "asimetria", "curtosis": "curtosis",
                                "q05": "q05", "q95": "q95"},
                               "observado", "media_rep", col_sd="sd_rep"))
        fu["ppc_pvals"] = "tabla_ppc.csv"
        fu["ppc_z"] = "tabla_ppc.csv"
        fu["n_ppc_extremos"] = "tabla_ppc.csv"
    ftx = _resolver(d / "resumen_diagnostico_M4.txt") or (d / "resumen_diagnostico_M4.txt")
    if ftx.exists():
        tx = ftx.read_text(encoding="utf-8")
        met["ks_pit"] = _re1(r"KS p=([0-9.]+)", tx)
        m = re.search(r"Cobertura 50/80/95:\s*([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)", tx)
        if m:
            met["cob_50"], met["cob_80"], met["cob_95"] = map(float, m.groups())
            fu.update(cob_50=ftx.name, cob_80=ftx.name, cob_95=ftx.name)
        g = re.search(r"Geweke:\s*(\d+)\s*/\s*(\d+)", tx)
        if g:
            met["geweke_frac_fdr"] = int(g.group(1)) / max(int(g.group(2)), 1)
            fu["geweke_frac_fdr"] = ftx.name
        met["n_save"] = _re1(r"S=(\d+)\s+guardadas", tx)
        met["nu_mediana"] = _re1(r"posterior:\s*mediana=([0-9.]+)", tx)
        for k in ("ks_pit", "n_save", "nu_mediana"):
            if np.isfinite(met.get(k, np.nan)):
                fu[k] = ftx.name
    if not np.isfinite(met.get("cob_95", np.nan)):
        df = _csv(d / "tabla_cobertura.csv")   # respaldo: promedio por periodo
        if df is not None:
            for niv, k in ((0.50, "cob_50"), (0.80, "cob_80"), (0.95, "cob_95")):
                sel = df[np.isclose(df["nivel"], niv)]
                if len(sel):
                    met[k] = float(sel["cobertura"].mean())
                    fu[k] = "tabla_cobertura.csv (promedio simple por periodo)"
    if (np.isfinite(met.get("rhat_max", np.nan))
            and np.isfinite(met.get("rhat_max_sin_nu", np.nan))
            and met["rhat_max"] - met["rhat_max_sin_nu"] > 0.02):
        notas.append(
            "M4: R-hat/ESS globales dominados por nu (verosimilitud plana en "
            "nu>100: hallazgo de especificacion, no fallo del muestreador). "
            "El tablero usa los valores SIN nu, marcados con la daga.")
    return met, fu, notas


LECTORES = {"M1": leer_artefactos_m1, "M2": leer_artefactos_m2,
            "M3": leer_artefactos_m3, "M4": leer_artefactos_m4}


# =============================================================================
# 3. CAPA B: FUENTES MCMC (draws + datos) POR MODELO
# =============================================================================
def _bin_matriz(ruta, n_save, dim):
    """(n_save x dim) float64 en orden C, como escribe writeBin en append."""
    a = np.fromfile(ruta, dtype="<f8")
    if a.size != n_save * dim:
        raise IOError(f"{Path(ruta).name}: {a.size} != {n_save}x{dim}")
    return a.reshape(n_save, dim)


def _colmajor(ruta, filas, cols):
    a = np.fromfile(ruta, dtype="<f8")
    if a.size != filas * cols:
        raise IOError(f"{Path(ruta).name}: {a.size} != {filas}x{cols}")
    return a.reshape((filas, cols), order="F")


def _meta_csv(ruta):
    df = pd.read_csv(ruta)
    return {str(r["clave"]): str(r["valor"]) for _, r in df.iterrows()}


class FuenteEstatica:
    """Base M1/M2: pool de cadenas + datos de export_python/."""

    def __init__(self, dirs):
        self.dirs = [Path(x) for x in dirs]
        exp = None
        for d in self.dirs:
            if (d / "export_python" / "y_vec.bin").exists():
                exp = d / "export_python"
                break
        if exp is None:
            raise FileNotFoundError(
                "ninguna cadena tiene export_python/y_vec.bin (re-exportar "
                "con datos=)")
        self.exp = exp
        meta = _meta_csv(exp / "meta.csv")
        self.meta = meta
        self.N = int(float(meta["N"]))
        self.n_cells = int(float(meta["n_cells"]))
        self.K = int(float(meta["K"]))
        self.y = np.fromfile(exp / "y_vec.bin", dtype="<f8")
        self.row_cell = np.fromfile(exp / "row_cell.bin", dtype="<f8").astype(int) - 1
        self.row_k = np.fromfile(exp / "row_k.bin", dtype="<f8").astype(int) - 1
        self.k_idx = self.row_k
        try:
            dd = pd.read_csv(exp / "deptos.csv").sort_values("k")
            self.depto_ids = [str(x) for x in dd["depto_id"]]
        except Exception:
            self.depto_ids = [f"k{i+1}" for i in range(self.K)]

    def _apilar(self, nombre, dim):
        partes = []
        for d in self.dirs:
            n_s = (d / "mu.bin").stat().st_size // 8 if (d / "mu.bin").exists() \
                else (d / "nu.bin").stat().st_size // 8
            partes.append(_bin_matriz(d / f"{nombre}.bin", n_s, dim))
        return np.vstack(partes)


class FuenteM1(FuenteEstatica):
    etiqueta = "M1"

    def __init__(self, dirs):
        super().__init__(dirs)
        m = self.meta
        pE, pM, pD = int(float(m["p_E"])), int(float(m["p_M"])), int(float(m["p_D"]))
        self.mu = self._apilar("mu", 1)[:, 0]
        self.bE = self._apilar("beta_E", pE)
        self.bM = self._apilar("beta_M", pM)
        self.bD = self._apilar("beta_D", pD)
        self.k2 = self._apilar("kappa2_jk", self.n_cells)
        self.S_total = self.mu.size
        self.X_E = np.memmap(self.exp / "X_E.bin", dtype="<f8", mode="r",
                                shape=(self.N, pE), order="F")
        self.Z = _colmajor(self.exp / "Z.bin", self.n_cells, pM)
        self.W = _colmajor(self.exp / "W.bin", self.K, pD)

    def df_draws(self, idx):
        return None                              # normal

    def params_para_ess(self, max_params=4000):
        # cadenas escalares: intercepto + coeficientes de los 3 bloques
        cols = [self.mu[:, None], self.bE, self.bM, self.bD]
        P = np.hstack(cols)                      # (S, 1+pE+pM+pD)
        if P.shape[1] > max_params:
            sel = np.linspace(0, P.shape[1] - 1, max_params).round().astype(int)
            P = P[:, np.unique(sel)]
        return P

    def preparar(self, idx):
        self._ZB = self.Z @ self.bM[idx].T       # (n_cells, Sd)
        self._WB = self.W @ self.bD[idx].T       # (K, Sd)
        self._mu = self.mu[idx]
        self._BE = self.bE[idx]
        self._SC = np.sqrt(self.k2[idx])         # (Sd, n_cells)

    def bloque(self, a, b, idx):
        rc, rk = self.row_cell[a:b], self.row_k[a:b]
        MU = (np.asarray(self.X_E[a:b]) @ self._BE.T
              + self._ZB[rc] + self._WB[rk] + self._mu[None, :])
        SC = self._SC[:, rc].T
        return MU, SC


class FuenteM2(FuenteEstatica):
    etiqueta = "M2"

    def __init__(self, dirs):
        super().__init__(dirs)
        pp1 = int(float(self.meta["pp1"]))
        self.beta = self._apilar("beta_star", pp1)
        self.nu = self._apilar("nu", 1)[:, 0]
        self.k2 = self._apilar("kappa2_jk", self.n_cells)
        self.S_total = self.nu.size
        self.X = np.memmap(self.exp / "X_star.bin", dtype="<f8", mode="r",
                                shape=(self.N, pp1), order="F")

    def df_draws(self, idx):
        return self.nu[idx]

    def params_para_ess(self, max_params=4000):
        P = np.asarray(self.beta)                      # (S, pp1), sin nu
        if P.shape[1] > max_params:
            sel = np.linspace(0, P.shape[1] - 1, max_params).round().astype(int)
            P = P[:, np.unique(sel)]
        return P

    def preparar(self, idx):
        self._B = self.beta[idx]
        self._SC = np.sqrt(self.k2[idx])

    def bloque(self, a, b, idx):
        rc = self.row_cell[a:b]
        MU = np.asarray(self.X[a:b]) @ self._B.T
        SC = self._SC[:, rc].T
        return MU, SC


class FuenteM3:
    etiqueta = "M3"

    def __init__(self, dirs):
        d = Path(dirs[0])
        self.dir = d
        rj = d / "metadatos.json"
        re_ = d / "datos_export" / "export_meta.json"
        src = json.loads((rj if rj.exists() else re_).read_text(encoding="utf-8"))

        def _s(v):
            return v[0] if isinstance(v, (list, tuple)) else v
        self.p = int(_s(src["p"])); self.Tt = int(_s(src["Tt"]))
        self.m = int(_s(src["m"]))
        n_k = np.asarray(src["n_k"], int).ravel()
        self.max_nk = int(_s(src.get("max_nk", n_k.max())))
        n_save = src.get("save_idx", src.get("n_save"))
        self.S_total = int(_s(n_save))
        em = json.loads((d / "datos_export" / "export_meta.json")
                        .read_text(encoding="utf-8"))
        self.N = int(_s(em["N_total"]))
        self.y = np.fromfile(d / "datos_export" / "y_obs.bin", dtype="<f8")
        self.F = np.memmap(d / "datos_export" / "F_obs.bin", dtype="<f8",
                           mode="r", shape=(self.N, self.p), order="C")
        self.depto_ids = [str(x) for x in src.get("depto_ids", [])] or \
            [f"k{i+1}" for i in range(self.m)]
        cn = np.asarray(em["cell_n"], int)
        self.t_all = np.repeat(np.asarray(em["cell_t"], int), cn)
        self.k_all = np.repeat(np.asarray(em["cell_k"], int), cn)
        self.j_all = np.repeat(np.asarray(em["cell_j"], int), cn)
        self.k_idx = self.k_all - 1
        if not np.all(np.diff(self.t_all) >= 0):
            raise ValueError("M3: t_all no viene ordenado (orden t->k->j)")

        def _leer3(nombre, d1, d2):
            raw = np.fromfile(d / f"{nombre}.bin", dtype="<f8")
            return raw.reshape((d1, d2, self.S_total), order="F")
        self.theta = _leer3("theta", self.p, self.Tt + 1)
        self.s = _leer3("s", self.m, self.Tt)
        self.v = _leer3("v", self.max_nk, self.m)

    def df_draws(self, idx):
        return None

    def params_para_ess(self, max_params=4000):
        # theta: (p, Tt+1, S) -> columnas (S, p*(Tt+1)) submuestreadas
        p, T1, S = self.theta.shape
        P = self.theta.reshape(p * T1, S, order="C").T          # (S, p*T1)
        if P.shape[1] > max_params:
            sel = np.linspace(0, P.shape[1] - 1, max_params).round().astype(int)
            P = P[:, np.unique(sel)]
        return P

    def preparar(self, idx):
        self._TH = self.theta[:, :, idx]                        # (p,Tt+1,Sd)
        self._S = self.s[:, :, idx]
        self._V = self.v[:, :, idx]

    def bloque(self, a, b, idx):
        t = self.t_all[a:b]; k = self.k_all[a:b]; j = self.j_all[a:b]
        n, Sd = b - a, self._TH.shape[2]
        MU = np.empty((n, Sd))
        F_blk = np.asarray(self.F[a:b])
        for tt in np.unique(t):
            sel = t == tt
            MU[sel] = F_blk[sel] @ self._TH[:, tt, :]
        # SIG2 = self._S[k - 1, t - 1, :] / self._V[j - 1, k - 1, :]
        s_bar = self._S.mean(axis=2); v_bar = self._V.mean(axis=2)
        sd_fix = np.sqrt(s_bar[k - 1, t - 1] / v_bar[j - 1, k - 1])
        return MU, np.broadcast_to(sd_fix[:, None], (n, Sd))


class FuenteM4:
    etiqueta = "M4"

    def __init__(self, dirs):
        d = Path(dirs[0])
        if (d / "export_python" / "meta.json").exists():
            d = d / "export_python"
        self.dir = d
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        self.p, self.Tt, self.m = meta["p"], meta["Tt"], meta["m"]
        self.S_total = meta["S"]
        n_k = np.asarray(meta["n_k"], int)
        self.N = int(meta["N"])
        self.y = np.fromfile(d / "y.bin", dtype="<f8")
        self.F = np.memmap(d / "F.bin", dtype="<f8", mode="r",
                           shape=(self.N, self.p), order="F")
        self.t_all = np.fromfile(d / "t_obs.bin", dtype="<i4")
        self.depto_ids = [str(x) for x in meta.get("depto_ids", [])]
        k_obs = np.fromfile(d / "k_obs.bin", dtype="<i4")
        j_obs = np.fromfile(d / "j_obs.bin", dtype="<i4")
        raw = np.fromfile(d / "theta.bin", dtype="<f8")
        self.theta = raw.reshape((self.p, self.Tt + 1, self.S_total), order="F")
        self.nu = np.fromfile(d / "nu.bin", dtype="<f8")
        v_mean = np.fromfile(d / "v_mean.bin", dtype="<f8").reshape(
            (int(n_k.max()), self.m), order="F")
        phi_mean = np.fromfile(d / "phi_mean.bin", dtype="<f8").reshape(
            (self.m, self.Tt), order="F")
        self.k_idx = k_obs - 1
        if not self.depto_ids:
            self.depto_ids = [f"k{i+1}" for i in range(self.m)]
        kap2 = 1.0 / (v_mean[j_obs - 1, k_obs - 1]
                      * phi_mean[k_obs - 1, self.t_all - 1])
        self.sd_obs = np.sqrt(kap2)      # escala fija (medias posteriores),
        #                                  identica a 12_diagnosticos_M4.py

    def df_draws(self, idx):
        return self.nu[idx]

    def params_para_ess(self, max_params=4000):
        # theta -> columnas (S, p*(Tt+1)). Se EXCLUYE nu a propósito: su no
        # convergencia es un hallazgo tratado aparte (daga en el tablero), y
        # contarlo aquí falsearía 'prop_ess_lt_400'. Coherente con rhat_max_vis.
        p, T1, S = self.theta.shape
        P = self.theta.reshape(p * T1, S, order="C").T
        if P.shape[1] > max_params:
            sel = np.linspace(0, P.shape[1] - 1, max_params).round().astype(int)
            P = P[:, np.unique(sel)]
        return P

    def preparar(self, idx):
        self._TH = self.theta[:, :, idx]

    def bloque(self, a, b, idx):
        t = self.t_all[a:b]
        n, Sd = b - a, self._TH.shape[2]
        MU = np.empty((n, Sd))
        F_blk = np.asarray(self.F[a:b])
        for tt in np.unique(t):
            sel = t == tt
            MU[sel] = F_blk[sel] @ self._TH[:, tt, :]
        SC = np.broadcast_to(self.sd_obs[a:b, None], (n, Sd))
        return MU, SC


FUENTES = {"M1": FuenteM1, "M2": FuenteM2, "M3": FuenteM3, "M4": FuenteM4}


# =============================================================================
# 4. PSIS-LOO (portado DIGITO A DIGITO de 19_c_ajuste_M1.py / 14_c)
# =============================================================================
def _gpd_fit_zhang(x):
    n = x.size
    if n < 5:
        return np.nan, np.nan
    x_max = x[-1]
    x_q = x[int(n / 4 + 0.5)]
    mgrid = 30 + int(np.sqrt(n))
    j = np.arange(1, mgrid + 1)
    theta = 1.0 / x_max + (1.0 - np.sqrt(mgrid / (j - 0.5))) / (3.0 * x_q)
    kth = -np.mean(np.log1p(-np.outer(theta, x)), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lth = n * (np.log(theta / kth) + kth - 1.0)
    lth[~np.isfinite(lth)] = -np.inf
    w = np.exp(lth - lth.max()); w /= w.sum()
    theta_hat = np.sum(theta * w)
    k = float(np.mean(np.log1p(-theta_hat * x)))
    sigma = float(-k / theta_hat)
    return k, sigma


def _psis_una_obs(lw, M):
    S = lw.size
    orden = np.argsort(lw)
    idx_cola = orden[S - M:]
    lw_cola = lw[idx_cola]
    umbral = lw[orden[S - M - 1]]
    exced = np.exp(lw_cola) - np.exp(umbral)
    exced = np.sort(exced[exced > 0])
    khat = np.nan
    if exced.size >= 5:
        k, sigma = _gpd_fit_zhang(exced)
        if np.isfinite(k) and np.isfinite(sigma) and sigma > 0:
            khat = k
            m = lw_cola.size
            p = (np.arange(1, m + 1) - 0.5) / m
            if abs(k) > 1e-8:
                q = sigma / k * ((1 - p) ** (-k) - 1)
            else:
                q = -sigma * np.log1p(-p)
            nuevos = np.log(np.exp(umbral) + q)
            lw_cola_ord = np.argsort(lw_cola)
            lw[idx_cola[lw_cola_ord]] = np.sort(nuevos)
    lw = lw - logsumexp(lw)
    tope = 0.75 * np.log(S) - logsumexp(lw)
    lw = np.minimum(lw, tope)
    lw = lw - logsumexp(lw)
    if np.isfinite(khat):
        khat = (M * khat + 5.0) / (M + 10.0)
    return lw, khat


# =============================================================================
# 5. CAPA B: MOTOR UNIFORME
# =============================================================================
def _ess_1d(x):
    """ESS de una cadena 1-D por autocovarianza (FFT) + regla de Geyer.

    Estimador de secuencia positiva monótona (Geyer 1992), el mismo criterio
    que usa la suite de convergencia de cada modelo. Se computa sobre TODAS
    las muestras guardadas de la cadena (no sobre el subconjunto de draws del
    resto del motor), para que 'prop_ess_lt_400' sea comparable con el que
    reporta cada pipeline por separado.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 8 or np.ptp(x) == 0 or not np.all(np.isfinite(x)):
        return np.nan
    x = x - x.mean()
    # autocovarianza vía FFT
    m = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=m)
    acov = np.fft.irfft(f * np.conjugate(f), n=m)[:n].real
    acov /= n
    if acov[0] <= 0:
        return np.nan
    rho = acov / acov[0]
    # regla de Geyer: sumar pares (rho_{2k}+rho_{2k+1}) mientras sean > 0
    tau = 1.0
    k = 1
    while k + 1 < n:
        pair = rho[k] + rho[k + 1]
        if pair <= 0:
            break
        tau += 2.0 * pair
        k += 2
    if tau <= 0:
        return np.nan
    return float(n / tau)


def prop_ess_bajo_uniforme(fuente, umbral=400, max_params=4000):
    """Proporción de parámetros escalares con ESS < umbral, y el ESS mínimo.

    Recorre las cadenas de parámetros que expone la fuente (método
    `params_para_ess`) y calcula el ESS de cada columna sobre TODAS las
    muestras. Devuelve (prop_lt_umbral, ess_min, n_evaluados). Si la fuente
    no expone parámetros, devuelve (nan, nan, 0).
    """
    if not hasattr(fuente, "params_para_ess"):
        return np.nan, np.nan, 0
    P = fuente.params_para_ess(max_params=max_params)   # (S, n_params)
    if P is None or P.size == 0:
        return np.nan, np.nan, 0
    P = np.asarray(P)
    if P.ndim == 1:
        P = P[:, None]
    ess = np.array([_ess_1d(P[:, j]) for j in range(P.shape[1])])
    ess = ess[np.isfinite(ess)]
    if ess.size == 0:
        return np.nan, np.nan, 0
    return float(np.mean(ess < umbral)), float(ess.min()), int(ess.size)


def evaluar_uniforme(fuente, draws=400, sin_loo=False, bloque_obs=50_000,
                     verbose=True):
    N, S_tot = fuente.N, fuente.S_total
    Sd = int(min(draws, S_tot))
    idx = np.unique(np.linspace(0, S_tot - 1, Sd).round().astype(int))
    Sd = idx.size
    fuente.preparar(idx)
    df = fuente.df_draws(idx)                    # None => normal
    y = fuente.y
    M_cola = max(int(min(0.2 * Sd, np.ceil(3 * np.sqrt(Sd)))), 5)

    lppd_i = np.empty(N); pw_i = np.empty(N)
    elpd_loo_i = np.full(N, np.nan); khat_i = np.full(N, np.nan)
    pit = np.empty(N); mu_media = np.empty(N)
    s_mu = np.zeros(Sd); s_mu2 = np.zeros(Sd)
    s_r = np.zeros(Sd); s_r2 = np.zeros(Sd)

    log2pi = np.log(2.0 * np.pi)
    if df is not None:
        cte_t = (gammaln((df + 1) / 2) - gammaln(df / 2)
                 - 0.5 * np.log(df * np.pi))[None, :]

    n_rep = int(min(60, Sd))
    rep_sel = np.unique(np.linspace(0, Sd - 1, n_rep).round().astype(int))
    n_rep = rep_sel.size
    rng = np.random.default_rng(20240501)
    T_rep = {e: np.full(n_rep, np.nan) for e in ESTAD_PPC}
    # acumuladores por réplica para media/sd/asim/curtosis (momentos) y
    # los cuantiles requieren el vector completo -> se arma por bloques.
    rep_sum = np.zeros(n_rep); rep_sum2 = np.zeros(n_rep)
    rep_sum3 = np.zeros(n_rep); rep_sum4 = np.zeros(n_rep)
    rep_vals = [[] for _ in range(n_rep)]        # para cuantiles

    n_blk = int(np.ceil(N / bloque_obs))
    for ib, a in enumerate(range(0, N, bloque_obs)):
        b = min(a + bloque_obs, N)
        MU, SC = fuente.bloque(a, b, idx)        # (n, Sd)
        yb = y[a:b, None]
        Z = (yb - MU) / SC
        if df is None:
            LL = -0.5 * log2pi - np.log(SC) - 0.5 * Z * Z
            pit[a:b] = ndtr(Z).mean(axis=1)
        else:
            LL = cte_t - np.log(SC) - ((df + 1) / 2)[None, :] * np.log1p(
                Z * Z / df[None, :])
            pit[a:b] = stats.t.cdf(Z, df[None, :]).mean(axis=1)
        lppd_i[a:b] = logsumexp(LL, axis=1) - np.log(Sd)
        pw_i[a:b] = LL.var(axis=1, ddof=1)
        if not sin_loo:
            for j in range(b - a):
                lw, kh = _psis_una_obs(-LL[j].copy(), M_cola)
                elpd_loo_i[a + j] = logsumexp(lw + LL[j])
                khat_i[a + j] = kh
        mu_media[a:b] = MU.mean(axis=1)
        s_mu += MU.sum(axis=0); s_mu2 += (MU * MU).sum(axis=0)
        R = yb - MU
        s_r += R.sum(axis=0); s_r2 += (R * R).sum(axis=0)
        # réplicas del PPC para las columnas seleccionadas
        MUr, SCr = MU[:, rep_sel], SC[:, rep_sel]
        if df is None:
            yrep = MUr + SCr * rng.standard_normal(MUr.shape)
        else:
            dfr = df[rep_sel][None, :]
            yrep = MUr + SCr * rng.standard_t(dfr, size=MUr.shape)
        rep_sum += yrep.sum(axis=0); rep_sum2 += (yrep ** 2).sum(axis=0)
        rep_sum3 += (yrep ** 3).sum(axis=0); rep_sum4 += (yrep ** 4).sum(axis=0)
        for r in range(n_rep):
            rep_vals[r].append(yrep[:, r])
        if verbose:
            print(f"    bloque {ib + 1}/{n_blk} ({b - a} obs x {Sd} draws)",
                  flush=True)

    # estadísticos por réplica a partir de los momentos acumulados
    m1 = rep_sum / N
    m2 = rep_sum2 / N - m1 ** 2
    m2 = np.where(m2 > 0, m2, np.nan)
    T_rep["media"] = m1
    T_rep["sd"] = np.sqrt(m2)
    mu3 = rep_sum3 / N - 3 * m1 * (rep_sum2 / N) + 2 * m1 ** 3
    mu4 = (rep_sum4 / N - 4 * m1 * (rep_sum3 / N)
           + 6 * m1 ** 2 * (rep_sum2 / N) - 3 * m1 ** 4)
    T_rep["asimetria"] = mu3 / m2 ** 1.5
    T_rep["curtosis"] = mu4 / m2 ** 2 - 3.0
    for r in range(n_rep):
        vv = np.concatenate(rep_vals[r])
        T_rep_q = np.quantile(vv, [0.05, 0.95])
        T_rep["q05"][r] = T_rep_q[0]; T_rep["q95"][r] = T_rep_q[1]
    # estadísticos observados y z
    T_obs = {"media": y.mean(), "sd": y.std(ddof=0),
             "asimetria": stats.skew(y), "curtosis": stats.kurtosis(y),
             "q05": np.quantile(y, 0.05), "q95": np.quantile(y, 0.95)}
    ppc_z = {}; ppc_p = {}
    for e in ESTAD_PPC:
        tr = T_rep[e][np.isfinite(T_rep[e])]
        if tr.size >= 3 and tr.std(ddof=1) > 0:
            ppc_z[e] = float((T_obs[e] - tr.mean()) / tr.std(ddof=1))
            ppc_p[e] = float(np.mean(tr >= T_obs[e]))

    var_mu = s_mu2 / N - (s_mu / N) ** 2
    var_r = s_r2 / N - (s_r / N) ** 2
    r2_draw = var_mu / (var_mu + var_r)
    resid = y - mu_media
    elpd_w_i = lppd_i - pw_i
    out = {
        "N": N, "draws": Sd, "y": y,
        "r2": float(1 - resid.var() / y.var()),
        "r2_bayes": float(np.median(r2_draw)),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "elpd_waic": float(elpd_w_i.sum()),
        "se_elpd_waic": float(np.sqrt(N * elpd_w_i.var(ddof=1))),
        "p_waic": float(pw_i.sum()),
        "elpd_waic_i": elpd_w_i.astype(np.float32),
        "pit": pit,
    }
    ks = stats.kstest(np.clip(pit, 0, 1), "uniform")
    out["ks_pit"] = float(ks.pvalue)
    out["ks_pit_d"] = float(ks.statistic)
    out["k_idx"] = np.asarray(getattr(fuente, "k_idx", np.zeros(N, int)),
                              dtype=np.int32)
    out["depto_ids"] = list(getattr(fuente, "depto_ids", []))
    # PPC uniforme: z por estadístico (magnitud) + p-value de referencia
    if ppc_z:
        out["ppc_z"] = ppc_z
        vs = [abs(v) for v in ppc_z.values() if np.isfinite(v)]
        if vs:
            out["ppc_z_max"] = float(max(vs))
        out["ppc_pvals"] = ppc_p
        out["n_ppc_extremos"] = int(sum((p < 0.05) or (p > 0.95)
                                         for p in ppc_p.values()))
    # ESS<400 uniforme (sobre las cadenas de parámetros de la fuente)
    prop_lt, ess_min, n_ev = prop_ess_bajo_uniforme(fuente)
    if n_ev > 0:
        out["prop_ess_lt_400"] = prop_lt
        out["ess_bulk_min_unif"] = ess_min
        out["n_par_ess_unif"] = n_ev
    for niv, k in ((0.50, "cob_50"), (0.80, "cob_80"), (0.95, "cob_95")):
        out[k] = float(np.mean((pit > (1 - niv) / 2) & (pit < (1 + niv) / 2)))
    if not sin_loo:
        out.update(
            elpd_loo=float(np.nansum(elpd_loo_i)),
            se_elpd_loo=float(np.sqrt(N * np.nanvar(elpd_loo_i, ddof=1))),
            p_loo=float(lppd_i.sum() - np.nansum(elpd_loo_i)),
            prop_khat_gt_07=float(np.nanmean(khat_i > 0.7)),
            elpd_loo_i=elpd_loo_i.astype(np.float32),
            khat_i=khat_i.astype(np.float32),
        )
    return out


# =============================================================================
# 6. DELTA elpd PAREADO
# =============================================================================
def _elpd_i_de(modelo, uni, met):
    """Devuelve dict criterio -> (vector elpd_i, origen) para el pareado."""
    out = {}
    if uni is not None:
        out["LOO"] = (uni.get("elpd_loo_i"), "capa uniforme")
        out["WAIC"] = (uni.get("elpd_waic_i"), "capa uniforme")
    elif met.get("ruta_khat"):
        df = _csv(met["ruta_khat"])
        if df is not None:
            if "elpd_loo_i" in df:
                out["LOO"] = (df["elpd_loo_i"].to_numpy(np.float64),
                              Path(met["ruta_khat"]).name)
            if "elpd_waic_i" in df:
                out["WAIC"] = (df["elpd_waic_i"].to_numpy(np.float64),
                               Path(met["ruta_khat"]).name)
    return {k: v for k, v in out.items() if v[0] is not None}


def delta_elpd_pares(presentes, uniformes, metricas):
    filas = []
    for i, A in enumerate(presentes):
        for B in presentes[i + 1:]:
            eA = _elpd_i_de(A, uniformes.get(A), metricas[A])
            eB = _elpd_i_de(B, uniformes.get(B), metricas[B])
            criterios = [c for c in ("LOO", "WAIC") if c in eA and c in eB]
            if not criterios:
                continue
            uA, uB = uniformes.get(A), uniformes.get(B)
            yA = uA["y"] if uA else None
            yB = uB["y"] if uB else None
            for cri in criterios:
                vA, oA = eA[cri]; vB, oB = eB[cri]
                if vA.size != vB.size:
                    filas.append(dict(par=f"{A} vs {B}", criterio=cri,
                                      delta_elpd=np.nan, se=np.nan, z=np.nan,
                                      preferido="NO COMPARABLE",
                                      metodo=f"N distinto ({vA.size} vs {vB.size})"))
                    continue
                if yA is not None and yB is not None:
                    if np.array_equal(yA, yB):
                        metodo = "pareado (y verificado)"
                    elif np.array_equal(np.sort(yA), np.sort(yB)):
                        metodo = "no pareado (mismo y, ORDEN distinto)"
                    else:
                        metodo = "NO COMPARABLE (y distinto)"
                else:
                    metodo = "pareado (orden asumido; verifique con --mX-mcmc)"
                d = vA.astype(np.float64) - vB.astype(np.float64)
                ok = np.isfinite(d)
                delta = float(np.nansum(d))
                if metodo.startswith("pareado"):
                    se = float(np.sqrt(ok.sum() * np.nanvar(d, ddof=1)))
                else:
                    sA = float(np.sqrt(ok.sum() * np.nanvar(vA, ddof=1)))
                    sB = float(np.sqrt(ok.sum() * np.nanvar(vB, ddof=1)))
                    se = float(np.hypot(sA, sB))
                if "NO COMPARABLE" in metodo:
                    delta, se = np.nan, np.nan
                z = delta / se if (se and np.isfinite(se) and se > 0) else np.nan
                pref = A if delta > 0 else B
                if not np.isfinite(delta):
                    pref = "--"
                elif np.isfinite(z) and abs(z) < 2:
                    pref += " (no concluyente)"
                filas.append(dict(par=f"{A} vs {B}", criterio=cri,
                                  delta_elpd=delta, se=se, z=z,
                                  preferido=pref, metodo=metodo,
                                  origen=(oA if oA == oB
                                          else f"{oA} | {oB}")))
    return pd.DataFrame(filas)


# =============================================================================
# 7. TABLERO, PERFIL Y RANKING
# =============================================================================
FILAS_TABLERO = [
    ("CONVERGENCIA", None, 0, "seccion", ""),
    ("R-hat máximo",            "rhat_max_vis",    -1, "abs_rhat", "< 1.01"),
    ("ESS bulk mínimo",         "ess_bulk_min_vis", 1, "abs_ess", "> 400"),
    ("% parámetros ESS<400",    "prop_ess_lt_400", -1, "rel", ""),
    ("% Geweke signif. (FDR)",  "geweke_frac_fdr", -1, "abs_gew", "≤ 5%"),
    ("AJUSTE (dentro de muestra)", None, 0, "seccion", ""),
    ("R² (media posterior)",    "r2",               1, "rel", ""),
    ("R² bayesiano",            "r2_bayes",         1, "rel", ""),
    ("RMSE",                    "rmse",            -1, "rel", ""),
    ("MAE",                     "mae",             -1, "rel", ""),
    ("PREDICCIÓN (criterios de información)", None, 0, "seccion", ""),
    ("elpd LOO",                "elpd_loo",         1, "rel", ""),
    ("elpd WAIC",               "elpd_waic",        1, "rel", ""),
    ("p_eff (LOO o WAIC)",      "p_eff",            0, "info", ""),
    ("% obs con k̂>0.7",         "prop_khat_gt_07", -1, "abs_khat", "= 0"),
    ("WAIC (−2·elpd)",          "waic",            -1, "rel", ""),
    ("DIC",                     "dic",             -1, "rel", ""),
    ("CALIBRACIÓN PREDICTIVA", None, 0, "seccion", ""),
    ("Cobertura IP 95%",        "cob_95",           0, "abs_cob", "≈ 0.95"),
    ("Cobertura IP 80%",        "cob_80",           0, "abs_cob80", "≈ 0.80"),
    ("Cobertura IP 50%",        "cob_50",           0, "abs_cob50", "≈ 0.50"),
    ("PIT: distancia KS (D)",   "ks_pit_d",        -1, "abs_ksd", "< 0.01"),
    ("PPC: máx |z| (6 estad.)", "ppc_z_max",       -1, "rel", ""),
    ("PPC: p-values extremos",  "n_ppc_extremos",  -1, "abs_ppc", "0 de 6"),
]


def _color_abs(regla, v):
    if not np.isfinite(v):
        return GRIS_CLARO
    if regla == "abs_rhat":
        return VERDE if v < U_RHAT_OK else AMARILLO if v < U_RHAT_AVISO else ROJO
    if regla == "abs_ess":
        return VERDE if v > U_ESS_OK else AMARILLO if v > U_ESS_MIN else ROJO
    if regla == "abs_gew":
        return VERDE if v <= 0.05 else AMARILLO if v <= 0.10 else ROJO
    if regla == "abs_khat":
        return VERDE if v == 0 else AMARILLO if v < 0.01 else ROJO
    if regla in ("abs_cob", "abs_cob80", "abs_cob50"):
        nom = {"abs_cob": 0.95, "abs_cob80": 0.80, "abs_cob50": 0.50}[regla]
        d = abs(v - nom)
        return VERDE if d <= U_COB_OK else AMARILLO if d <= U_COB_AVISO else ROJO
    if regla == "abs_ks":
        return VERDE if v > 0.05 else AMARILLO if v > 0.01 else ROJO
    if regla == "abs_ksd":
        return VERDE if v < 0.01 else AMARILLO if v < 0.05 else ROJO
    if regla == "abs_ppc":
        return VERDE if v == 0 else AMARILLO if v <= 2 else ROJO
    return GRIS_CLARO


def _colores_rel(vals, direccion):
    v = np.asarray(vals, float)
    out = [GRIS_CLARO] * len(v)
    fin = np.isfinite(v)
    if fin.sum() < 2:
        for i in range(len(v)):
            if fin[i]:
                out[i] = GRIS_CLARO
        return out
    vv = v[fin] * direccion
    mejor, peor = np.nanmax(vv), np.nanmin(vv)
    for i, (f, x) in enumerate(zip(fin, v)):
        if not f:
            continue
        x = x * direccion
        if np.isclose(x, mejor):
            out[i] = VERDE
        elif np.isclose(x, peor):
            out[i] = ROJO
        else:
            out[i] = AMARILLO
    return out


def _valores_tablero(met):
    """Deriva claves compuestas para el tablero."""
    v = dict(met)
    dag = ""
    if np.isfinite(met.get("rhat_max_sin_nu", np.nan)):
        v["rhat_max_vis"] = met["rhat_max_sin_nu"]
        v["ess_bulk_min_vis"] = met.get("ess_bulk_min_sin_nu", np.nan)
        if (np.isfinite(met.get("rhat_max", np.nan))
                and met["rhat_max"] - met["rhat_max_sin_nu"] > 0.02):
            dag = "†"
    else:
        v["rhat_max_vis"] = met.get("rhat_max", np.nan)
        v["ess_bulk_min_vis"] = met.get("ess_bulk_min", np.nan)
    v["_dag"] = dag
    v["p_eff"] = met.get("p_loo", np.nan)
    if not np.isfinite(v["p_eff"]):
        v["p_eff"] = met.get("p_waic", np.nan)
    ew = met.get("elpd_waic", np.nan)
    v["waic"] = -2.0 * ew if np.isfinite(ew) else np.nan
    return v


def _fmt_celda(clave, v, dag=""):
    if not np.isfinite(v):
        return "--"
    if clave in ("elpd_loo", "elpd_waic", "dic", "waic"):
        return f"{v:,.0f}"
    if clave in ("prop_ess_lt_400", "geweke_frac_fdr", "prop_khat_gt_07"):
        return f"{100 * v:.1f}%"
    if clave == "n_ppc_extremos":
        return f"{int(v)}/6"
    if clave in ("ess_bulk_min_vis", "p_eff"):
        return f"{v:,.0f}"
    if clave.startswith("cob"):
        return f"{v:.3f}"
    if clave == "rhat_max_vis":
        return f"{v:.3f}{dag}"
    if clave == "ks_pit":
        return f"{v:.3f}"
    if clave == "ks_pit_d":
        return f"{v:.4f}"
    if clave == "ppc_z_max":
        return f"{v:,.1f}"
    return f"{v:.3f}" if abs(v) < 100 else f"{v:,.1f}"


DIMENSIONES = {
    "Convergencia": [("rhat_max_vis", -1), ("ess_bulk_min_vis", 1),
                     ("geweke_frac_fdr", -1)],
    "Ajuste": [("r2", 1), ("r2_bayes", 1), ("rmse", -1), ("mae", -1)],
    "Predicción (elpd)": [("elpd_loo", 1), ("elpd_waic", 1)],
    "Calibración": [("err_cob", -1), ("ks_pit_d", -1), ("ppc_z_max", -1),
                    ("n_ppc_extremos", -1)],
}


def puntajes_dimensiones(presentes, metricas, mismos_datos=True):
    """Min-max por metrica (direccion-consciente) y promedio por dimension.

    Si mismos_datos es False, la dimension 'Prediccion (elpd)' se deja en NaN:
    los criterios de informacion solo son comparables sobre los mismos datos.
    """
    vals = {}
    for M in presentes:
        met = dict(metricas[M])
        # rhat/ESS "visibles" (sin nu): misma lógica que el tablero, para que
        # radar y ranking traten la no-identificación de nu igual (M2 y M4).
        if np.isfinite(met.get("rhat_max_sin_nu", np.nan)):
            met["rhat_max_vis"] = met["rhat_max_sin_nu"]
            met["ess_bulk_min_vis"] = met.get("ess_bulk_min_sin_nu", np.nan)
        else:
            met["rhat_max_vis"] = met.get("rhat_max", np.nan)
            met["ess_bulk_min_vis"] = met.get("ess_bulk_min", np.nan)
        # respaldo: si el artefacto no trae ESS sin-nu pero la capa uniforme
        # calculó uno (que ya excluye nu en M2/M4), usarlo para no penalizar
        # la convergencia por un parametro no identificado.
        if np.isfinite(met.get("ess_bulk_min_unif", np.nan)):
            ev = met["ess_bulk_min_vis"]
            if (not np.isfinite(ev)) or met.get("nu_mediana") is not None:
                met["ess_bulk_min_vis"] = met["ess_bulk_min_unif"]
        errs = [abs(met.get(k, np.nan) - nom) for k, nom in
                (("cob_50", .5), ("cob_80", .8), ("cob_95", .95))]
        errs = [e for e in errs if np.isfinite(e)]
        met["err_cob"] = float(np.mean(errs)) if errs else np.nan
        vals[M] = met

    tabla = {}
    for dim, claves in DIMENSIONES.items():
        if dim == "Predicción (elpd)" and not mismos_datos:
            tabla[dim] = {M: np.nan for M in presentes}
            continue
        claves_uso = list(claves)
        if dim == "Calibración" and not any(
                np.isfinite(vals[M].get("ks_pit_d", np.nan))
                for M in presentes):
            # sin capa uniforme no hay distancia D; usar el p-valor persistido
            claves_uso = [("ks_pit", 1) if c[0] == "ks_pit_d" else c
                          for c in claves_uso]
        normas = {M: [] for M in presentes}
        for clave, direc in claves_uso:
            x = np.array([vals[M].get(clave, np.nan) for M in presentes],
                         float) * direc
            fin = np.isfinite(x)
            if fin.sum() < 2:
                continue          # una sola cifra no permite comparar
            lo, hi = np.nanmin(x), np.nanmax(x)
            for i, M in enumerate(presentes):
                if not fin[i]:
                    continue
                normas[M].append(1.0 if hi == lo else (x[i] - lo) / (hi - lo))
        tabla[dim] = {M: (float(np.mean(normas[M])) if normas[M] else np.nan)
                      for M in presentes}
    glob = {}
    for M in presentes:
        vs = [tabla[d][M] for d in tabla if np.isfinite(tabla[d][M])]
        glob[M] = float(np.mean(vs)) if vs else np.nan
    return tabla, glob


# =============================================================================
# 8. FIGURAS
# =============================================================================
def fig_tablero(presentes, metricas, fuentes, dir_diag, pdf, nota_nu,
                mismos_datos=True):
    from matplotlib.patches import FancyBboxPatch
    vals = {M: _valores_tablero(metricas[M]) for M in presentes}
    filas = list(FILAS_TABLERO)
    if not any(np.isfinite(metricas[M].get("ks_pit_d", np.nan))
               for M in presentes):
        filas = [("PIT: KS (p-valor)", "ks_pit", 1, "abs_ks", "> 0.05")
                 if f[1] == "ks_pit_d" else f for f in filas]
    n_f, n_c = len(filas), len(presentes)

    # Geometría de columnas (en unidades de dato):
    X_LAB = 0.10          # inicio del texto de la métrica
    W_LAB = 3.05          # ancho reservado para etiqueta + flecha de dirección
    X_UMB = 3.20          # columna de umbral de referencia (borde derecho)
    X0 = 3.35             # inicio de la primera celda de modelo
    W_TOT = X0 + n_c

    fig, ax = plt.subplots(figsize=(3.5 + 1.75 * n_c, 0.44 * n_f + 2.4))
    ax.set_xlim(0, W_TOT); ax.set_ylim(0, n_f)
    ax.invert_yaxis(); ax.axis("off")

    # --- cabecera de modelos ---
    for j, M in enumerate(presentes):
        cx = X0 + j + 0.5
        resto = MODELOS[M]["nombre"].split(" ", 1)[1] \
            if " " in MODELOS[M]["nombre"] else ""
        ax.add_patch(FancyBboxPatch(
            (X0 + j + 0.06, -1.72), 0.88, 1.30,
            boxstyle="round,pad=0,rounding_size=0.10",
            facecolor=MODELOS[M]["color"], edgecolor="none", clip_on=False))
        ax.text(cx, -1.34, M, ha="center", va="center", fontweight="bold",
                fontsize=13, color="white", clip_on=False)
        ax.text(cx, -0.74, resto.replace(" ", "\n"), ha="center", va="center",
                fontsize=8.2, color="white", clip_on=False, linespacing=0.95)

    flecha_dir = {1: "▲", -1: "▼", 0: ""}
    for i, (lab, clave, direc, regla, umbral) in enumerate(filas):
        if regla == "seccion":
            ax.add_patch(FancyBboxPatch(
                (0, i + 0.10), W_TOT, 0.80,
                boxstyle="round,pad=0,rounding_size=0.08",
                facecolor=SECCION, edgecolor="none"))
            ax.text(X_LAB, i + 0.5, lab, va="center", color="white",
                    fontweight="bold", fontsize=10)
            continue
        # etiqueta de la métrica
        ax.text(X_LAB, i + 0.5, lab, va="center", fontsize=9.6, color=TINTA)
        # flecha de dirección (solo métricas relativas), en su propia posición
        if regla == "rel" and direc != 0:
            ax.text(W_LAB, i + 0.5, flecha_dir[direc], va="center",
                    ha="right", fontsize=8.5,
                    color=(MODELOS["M3"]["color"] if direc == 1
                           else MODELOS["M4"]["color"]))
        # umbral de referencia
        if umbral:
            ax.text(X_UMB, i + 0.5, umbral, va="center", ha="right",
                    fontsize=7.8, color=GRIS, style="italic")
        v = [vals[M].get(clave, np.nan) for M in presentes]
        if (not mismos_datos) and clave in ("elpd_loo", "elpd_waic", "dic",
                                            "waic"):
            cols = [GRIS_CLARO] * n_c
        elif regla == "rel":
            cols = _colores_rel(v, direc)
        elif regla == "info":
            cols = [GRIS_CLARO] * n_c
        else:
            cols = [_color_abs(regla, x) for x in v]
        for j, M in enumerate(presentes):
            ax.add_patch(FancyBboxPatch(
                (X0 + j + 0.05, i + 0.10), 0.90, 0.80,
                boxstyle="round,pad=0,rounding_size=0.09",
                facecolor=cols[j], edgecolor="white", lw=1.2))
            dag = vals[M]["_dag"] if clave == "rhat_max_vis" else ""
            ax.text(X0 + j + 0.5, i + 0.5, _fmt_celda(clave, v[j], dag),
                    ha="center", va="center", fontsize=9, color=TINTA)

    # --- leyenda inferior con parches de color ---
    y_leg = n_f + 0.55
    items = [(VERDE, "cumple / mejor"), (AMARILLO, "intermedio"),
             (ROJO, "problema / peor"), (GRIS_CLARO, "informativo o n/d")]
    x_cursor = X_LAB
    for col, txt in items:
        ax.add_patch(FancyBboxPatch(
            (x_cursor, y_leg - 0.28), 0.34, 0.42,
            boxstyle="round,pad=0,rounding_size=0.06",
            facecolor=col, edgecolor="white", lw=1, clip_on=False))
        ax.text(x_cursor + 0.46, y_leg - 0.06, txt, va="center", fontsize=8.4,
                color=TINTA, clip_on=False)
        x_cursor += 0.52 + 0.075 * len(txt) * 1.9
    extra = []
    if not mismos_datos:
        extra.append("N distinto entre modelos → elpd / WAIC / DIC en gris: "
                     "no comparables entre datos distintos.")
    if nota_nu:
        extra.append("(†) M4: R̂ / ESS reportados sin ν; con ν: " + nota_nu)
    if extra:
        ax.text(X_LAB, y_leg + 0.55, "\n".join(extra), va="top", fontsize=8,
                color=GRIS, clip_on=False, linespacing=1.3)

    # ax.set_title("Tablero comparativo   ·   convergencia · ajuste · "
    #              "predicción · calibración", pad=52, loc="left", fontsize=13.5,
    #              fontweight="bold")
    guardar_fig(fig, dir_diag, "comp_01_tablero", pdf)


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


def fig_perfil(presentes, tabla_dim, glob, dir_diag, pdf):
    dims = list(tabla_dim.keys())
    fig, axs = plt.subplots(1, 2, figsize=(13.0, 4.6),
                            gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axs[0]
    n_m = len(presentes)
    h = 0.80 / n_m
    for j, dim in enumerate(dims):
        for i, M in enumerate(presentes):
            v = tabla_dim[dim].get(M, np.nan)
            y = j + (i - (n_m - 1) / 2) * h
            if np.isfinite(v):
                ax.barh(y, v, height=h - 0.015, color=MODELOS[M]["color"],
                        alpha=0.92, edgecolor="white", linewidth=0.6,
                        zorder=3)
                ax.text(v + 0.012, y, f"{v:.2f}", va="center", fontsize=8,
                        color=TINTA)
            else:
                ax.text(0.012, y, "n/d", va="center", fontsize=7.5,
                        color=GRIS, style="italic")
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels(dims, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.16)
    ax.set_xlabel("puntaje normalizado  (1 = el mejor de los modelos "
                  "comparados)")
    ax.set_title("Perfil por dimensión", fontsize=11, pad=15)
    ax.text(0.5, 1.015, "normalización min–max entre modelos, por dimensión",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.3,
            color=GRIS, style="italic")

    ax = axs[1]
    orden = sorted([M for M in presentes if np.isfinite(glob.get(M, np.nan))],
                   key=lambda M: -glob[M])
    orden += [M for M in presentes if M not in orden]
    v = [glob.get(M, np.nan) for M in orden]
    barras_cols = [MODELOS[M]["color"] for M in orden]
    ypos = range(len(orden))
    ax.barh(ypos, [x if np.isfinite(x) else 0 for x in v], color=barras_cols,
            alpha=0.92, edgecolor="white", linewidth=0.6, height=0.62,
            zorder=3)
    for i, (M, x) in enumerate(zip(orden, v)):
        if np.isfinite(x):
            ax.text(x + 0.012, i, f"{x:.2f}", va="center", fontsize=9.5,
                    color=TINTA, fontweight="medium")
    ax.set_yticks(list(ypos))
    ax.set_yticklabels([f"{i+1}.  {M}" for i, M in enumerate(orden)],
                       fontsize=10.5)
    ax.invert_yaxis(); ax.set_xlim(0, 1.16)
    ax.set_xlabel("puntaje global (promedio de dimensiones)")
    ax.set_title("Ranking global heurístico", fontsize=11, pad=15)
    ax.text(0.5, 1.015, "orientativo · el criterio formal es Δelpd",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.3,
            color=GRIS, style="italic")

    # leyenda única arriba, fuera de las áreas de datos
    fig.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=MODELOS[M]["color"],
                                  edgecolor="white",
                                  label=f"{M} · {MODELOS[M]['nombre'].split(' ',1)[1]}")
                        for M in presentes],
               loc="lower center", bbox_to_anchor=(0.5, 0.90),
               ncols=min(4, n_m), fontsize=8.8, handlelength=1.1,
               columnspacing=1.6)
    fig.suptitle("Síntesis: fortalezas por dimensión y ranking global",
                 y=1.02, fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    guardar_fig(fig, dir_diag, "comp_06_perfil", pdf)


def _radar_ejes(ax, dims, ang):
    """Configura un eje polar con estética náutica: anillos, retícula, marco."""
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.0)
    ax.set_facecolor("#F7F9FA")
    # anillos concéntricos suaves con banda alterna
    for r0, r1, c in [(0, 0.25, "#EEF2F4"), (0.25, 0.50, "#F7F9FA"),
                      (0.50, 0.75, "#EEF2F4"), (0.75, 1.0, "#F7F9FA")]:
        ax.fill_between(np.linspace(0, 2 * np.pi, 100), r0, r1, color=c,
                        zorder=0)
    ax.set_rgrids([0.25, 0.5, 0.75, 1.0],
                  labels=["0.25", "0.50", "0.75", "1.00"], angle=90,
                  fontsize=7.5, color=GRIS)
    ax.set_xticks(ang)
    ax.set_xticklabels([])                       # etiquetas manuales (mejor pos)
    ax.grid(color="#C4CDD3", alpha=0.9, linewidth=0.8)
    ax.spines["polar"].set_color("#C4CDD3")
    ax.tick_params(colors=GRIS)
    # radios desde el centro a cada vértice
    for a in ang:
        ax.plot([a, a], [0, 1.0], color="#C4CDD3", lw=0.8, zorder=1)


def fig_radar(presentes, tabla_dim, glob, dir_diag, pdf):
    """Radar de síntesis náutico: superpuesto + facetado por modelo.

    Mismos puntajes 0–1 que comp_06, en forma polar. Útil para la defensa;
    para lectura exacta úsese comp_06 (el área de un radar no es una métrica y
    depende del orden de los ejes).
    """
    dims = [d for d in tabla_dim
            if any(np.isfinite(tabla_dim[d].get(M, np.nan))
                   for M in presentes)]
    if len(dims) < 3:
        print("  (radar omitido: se necesitan >=3 dimensiones con datos)")
        return
    ang = np.linspace(0, 2 * np.pi, len(dims), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    hay_nan = False

    def dibujar(ax, M, con_etiqueta_valor=False):
        nonlocal hay_nan
        v = np.array([tabla_dim[d].get(M, np.nan) for d in dims], float)
        if not np.all(np.isfinite(v)):
            hay_nan = True
            v = np.nan_to_num(v, nan=0.0)
        vc = np.concatenate([v, v[:1]])
        col, osc = MODELOS[M]["color"], MODELOS[M]["oscuro"]
        ax.fill(ang_c, vc, color=col, alpha=0.22, zorder=2)
        ax.plot(ang_c, vc, "-", lw=2.6, color=col, zorder=3,
                solid_capstyle="round")
        ax.scatter(ang, v, s=64, color=col, edgecolors="white",
                   linewidths=1.6, zorder=4)
        if con_etiqueta_valor:
            for a, val in zip(ang, v):
                ax.annotate(f"{val:.2f}", (a, val), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=7.5,
                            color=osc, fontweight="medium", zorder=5)

    def etiquetas_dim(ax, fs=9.5):
        for a, d in zip(ang, dims):
            ha = ("center" if abs(np.sin(a)) < 0.2
                  else ("left" if np.sin(a) > 0 else "right"))
            ax.text(a, 1.20, d, ha=ha, va="center", fontsize=fs, color=TINTA,
                    fontweight="medium")

    # ---------- figura: panel grande superpuesto + mini-múltiplos ----------
    fig = plt.figure(figsize=(13.5, 6.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.35, 1.35, 0.9, 0.9],
                          height_ratios=[1, 1], wspace=0.45, hspace=0.55,
                          top= 0.86)
    ax0 = fig.add_subplot(gs[:, :2], projection="polar")
    _radar_ejes(ax0, dims, ang)
    etiquetas_dim(ax0, fs=11)
    orden = sorted(presentes, key=lambda M: -(glob.get(M) or -np.inf))
    for M in orden:
        dibujar(ax0, M)
    ax0.set_title("Comparación superpuesta", fontsize=12, pad=26,
                  fontweight="bold", color=TINTA)
    ax0.legend(handles=[Line2D([0], [0], color=MODELOS[M]["color"], lw=3,
                               marker="o", markerfacecolor=MODELOS[M]["color"],
                               markeredgecolor="white", markersize=8,
                               label=(f"{M}  ({glob[M]:.2f})"
                                      if np.isfinite(glob.get(M, np.nan))
                                      else M))
                        for M in orden],
               loc="upper left", bbox_to_anchor=(-0.22, 1.14), fontsize=9.5,
               title="modelo (puntaje global)", title_fontsize=9,
               labelspacing=0.7)

    # mini-múltiplos: un radar por modelo, para leer cada perfil sin estorbo
    celdas = [gs[0, 2], gs[0, 3], gs[1, 2], gs[1, 3]]
    for k, M in enumerate(orden[:4]):
        axk = fig.add_subplot(celdas[k], projection="polar")
        _radar_ejes(axk, dims, ang)
        axk.set_rgrids([0.5, 1.0], labels=[], angle=90)
        for a, d in zip(ang, dims):
            corta = d.split(" ")[0][:5]
            ha = ("center" if abs(np.sin(a)) < 0.2
                  else ("left" if np.sin(a) > 0 else "right"))
            axk.text(a, 1.16, corta, ha=ha, va="center", fontsize=7,
                     color=GRIS)
        dibujar(axk, M, con_etiqueta_valor=True)
        g = glob.get(M, np.nan)
        axk.set_title(f"{M}" + (f"  ·  {g:.2f}" if np.isfinite(g) else ""),
                      fontsize=10, y=1.24, color=MODELOS[M]["oscuro"],
                      fontweight="bold")

    fig.suptitle("Síntesis en radar  ·  vértice en 1 = el mejor del grupo en "
                 "esa dimensión", y=1.02, fontsize=13.5, fontweight="bold")
    if hay_nan:
        fig.text(0.01, 0.005, "* una dimensión sin dato (p. ej. elpd con N "
                 "distinto) se dibuja en 0", fontsize=8, color=GRIS,
                 style="italic")
    guardar_fig(fig, dir_diag, "comp_07_radar", pdf)


def figs_delta_depto(df_delta, uniformes, dir_diag, pdf):
    """Δelpd puntual agregado por departamento, por cada par verificado.

    Muestra DÓNDE gana cada modelo: Δelpd_k = Σ_{i∈k} (elpd_i^A − elpd_i^B),
    con SE_k = sqrt(n_k · var_k(Δ_i)). Solo para pares con y verificado y
    el mismo índice departamental por observación.
    """
    if not len(df_delta):
        return
    filas_csv = []
    hechos = set()
    for _, r in df_delta.iterrows():
        if not str(r["metodo"]).startswith("pareado (y verificado)"):
            continue
        A, B = [s.strip() for s in r["par"].split("vs")]
        if (A, B) in hechos or A not in uniformes or B not in uniformes:
            continue
        uA, uB = uniformes[A], uniformes[B]
        clave = ("elpd_loo_i" if ("elpd_loo_i" in uA and "elpd_loo_i" in uB)
                 else "elpd_waic_i")
        cri = "LOO" if clave == "elpd_loo_i" else "WAIC"
        kA = uA.get("k_idx"); kB = uB.get("k_idx")
        if kA is None or kB is None or not np.array_equal(kA, kB):
            print(f"  (Δelpd por depto omitido para {A} vs {B}: índice "
                  "departamental no coincide)")
            continue
        hechos.add((A, B))
        d = uA[clave].astype(np.float64) - uB[clave].astype(np.float64)
        ids = uA.get("depto_ids") or []
        K = int(kA.max()) + 1
        etiq = [ids[k] if k < len(ids) else f"k{k+1}" for k in range(K)]
        delta_k = np.zeros(K); se_k = np.zeros(K); n_k = np.zeros(K, int)
        for k in range(K):
            sel = d[kA == k]
            sel = sel[np.isfinite(sel)]
            n_k[k] = sel.size
            if sel.size > 1:
                delta_k[k] = sel.sum()
                se_k[k] = np.sqrt(sel.size * sel.var(ddof=1))
        orden = np.argsort(delta_k)
        alto = max(3.4, 0.26 * K + 1.6)
        fig, ax = plt.subplots(figsize=(8.6, alto))
        ax.axvline(0, color=TINTA, lw=1.2)
        for pos, k in enumerate(orden):
            col = MODELOS[A]["color"] if delta_k[k] > 0 else MODELOS[B]["color"]
            ax.errorbar(delta_k[k], pos, xerr=2 * se_k[k], fmt="o",
                        color=col, ms=5, capsize=2.5, elinewidth=1.1)
            filas_csv.append(dict(par=f"{A} vs {B}", criterio=cri,
                                  depto=etiq[k], n=int(n_k[k]),
                                  delta_elpd=float(delta_k[k]),
                                  se=float(se_k[k])))
        ax.set_yticks(range(K))
        ax.set_yticklabels([etiq[k] for k in orden], fontsize=8)
        ax.set_xlabel(f"Δelpd ({cri}) por departamento   "
                      f"[ >0 favorece a {A}, <0 favorece a {B} ]")
        n_a = int((delta_k > 0).sum())
        ax.set_title(f"Δelpd por departamento — " # ¿Dónde gana cada modelo?
                     f"{A} vs {B}\n{A} mejor en {n_a} de {K} departamentos; "
                     f"barras = ±2 SE", fontsize=11)
        guardar_fig(fig, dir_diag, f"comp_08_delta_depto_{A}_{B}", pdf)
    if filas_csv:
        pd.DataFrame(filas_csv).to_csv(
            Path(dir_diag) / "comparacion_delta_elpd_depto.csv",
            index=False, encoding="utf-8")
        print(f"  [csv] {dir_diag}/comparacion_delta_elpd_depto.csv")


# =============================================================================
# 9. TABLAS Y REPORTE
# =============================================================================
CLAVES_TIDY = [
    ("convergencia", "rhat_max"), ("convergencia", "rhat_max_sin_nu"),
    ("convergencia", "ess_bulk_min"), ("convergencia", "ess_bulk_min_sin_nu"),
    ("convergencia", "ess_tail_min"), ("convergencia", "prop_ess_lt_400"),
    ("convergencia", "geweke_frac_fdr"), ("convergencia", "n_par_conv"),
    # ("ajuste", "r2"), ("ajuste", "r2_bayes"), 
    ("ajuste", "rmse"),
    ("ajuste", "mae"),
    ("informacion", "elpd_loo"), ("informacion", "se_elpd_loo"),
    ("informacion", "p_loo"), ("informacion", "elpd_waic"),
    ("informacion", "se_elpd_waic"), ("informacion", "p_waic"),
    ("informacion", "prop_khat_gt_07"), ("informacion", "dic"),
    ("informacion", "p_dic"), ("informacion", "N"), ("informacion", "n_save"),
    ("calibracion", "cob_50"), ("calibracion", "cob_80"),
    ("calibracion", "cob_95"), ("calibracion", "ks_pit"),
    ("calibracion", "ks_pit_d"), ("calibracion", "ppc_z_max"),
    ("calibracion", "n_ppc_extremos"),
]


def tabla_tidy(presentes, metricas, fuentes):
    filas = []
    for M in presentes:
        for dim, k in CLAVES_TIDY:
            v = metricas[M].get(k, np.nan)
            if isinstance(v, (int, float)) and np.isfinite(v):
                filas.append(dict(modelo=M, dimension=dim, metrica=k,
                                  valor=v, fuente=fuentes[M].get(k, "")))
        pv = metricas[M].get("ppc_pvals") or {}
        for est, v in pv.items():
            filas.append(dict(modelo=M, dimension="calibracion",
                              metrica=f"p_bayes_{est}", valor=v,
                              fuente=fuentes[M].get("ppc_pvals", "")))
        zz = metricas[M].get("ppc_z") or {}
        for est, v in zz.items():
            filas.append(dict(modelo=M, dimension="calibracion",
                              metrica=f"z_ppc_{est}", valor=v,
                              fuente=fuentes[M].get("ppc_z", "")))
    return pd.DataFrame(filas)


def escribir_reporte(dir_diag, presentes, metricas, fuentes, df_delta,
                     tabla_dim, glob, notas, capa_b):
    L = []
    ap = L.append
    ap("# Comparacion de modelos M1-M4\n")
    ap(f"Modelos comparados: {', '.join(MODELOS[M]['nombre'] for M in presentes)}.")
    ap(f"Capa uniforme (recomputo desde MCMC): "
       f"{', '.join(capa_b) if capa_b else 'no ejecutada (solo artefactos)'}.\n")

    Ns = {M: metricas[M].get("N", np.nan) for M in presentes}
    if len({int(v) for v in Ns.values() if np.isfinite(v)}) > 1:
        ap("> **ADVERTENCIA**: los modelos no reportan el mismo N. Los "
          "criterios de informacion (elpd/WAIC/DIC) **no son comparables** "
          "entre modelos ajustados a datos distintos: "
          + ", ".join(f"{M}: N={_fmt(Ns[M], 0, True)}" for M in presentes)
          + ".\n")

    ap("## Veredicto por dimension\n")
    for dim in tabla_dim:
        vv = {M: tabla_dim[dim][M] for M in presentes
              if np.isfinite(tabla_dim[dim][M])}
        if vv:
            mejor = max(vv, key=vv.get)
            ap(f"- **{dim}**: mejor {mejor} "
               f"({', '.join(f'{M}={vv[M]:.2f}' for M in vv)}).")
    orden = sorted((M for M in presentes if np.isfinite(glob.get(M, np.nan))),
                   key=lambda M: -glob[M])
    if orden:
        ap(f"- **Ranking global (heuristico)**: "
           + " > ".join(f"{M} ({glob[M]:.2f})" for M in orden) + ".")
    ap("\nEl puntaje global promedia dimensiones normalizadas min-max y es "
       "solo orientativo; la comparacion formal de capacidad predictiva es "
       "el Delta_elpd pareado de la tabla siguiente.\n")

    ap("## Delta_elpd por pares\n")
    if len(df_delta):
        ap("| Par | Criterio | Delta_elpd | SE | z | Preferido | Metodo | "
           "Origen |")
        ap("|---|---|---:|---:|---:|---|---|---|")
        for _, r in df_delta.iterrows():
            ap(f"| {r['par']} | {r['criterio']} | "
               f"{_fmt(r['delta_elpd'], 1, True)} "
               f"| {_fmt(r['se'], 1, True)} | {_fmt(r['z'], 2)} | "
               f"{r['preferido']} | {r['metodo']} | "
               f"{r.get('origen', '')} |")
        ap("\nRegla practica (Vehtari et al.): |Delta| < 2 SE no es "
           "concluyente; ademas, |Delta| < 4 se considera pequeno aun si "
           "es 'significativo'.\n")
    else:
        ap("No hubo pares con vectores elpd puntuales disponibles. Ejecute "
           "con --mX-mcmc para habilitar el pareado.\n")

    ho = metricas.get("M2", {}).get("holdout")
    if isinstance(ho, dict):
        ap("## Hold-out (solo disponible en M2)\n")
        ap(f"N_test={_fmt(ho.get('N_test'), 0, True)}, "
           f"RMSE_test={_fmt(ho.get('rmse'))}, MAE_test={_fmt(ho.get('mae'))}, "
           f"cobertura 95%={_fmt(ho.get('cob_95'), 3)}, "
           f"CRPS={_fmt(ho.get('crps_medio'), 3)}. Los demas modelos no "
           "persisten una capa hold-out comparable.\n")

    if notas:
        ap("## Notas y advertencias\n")
        for n in notas:
            ap(f"- {n}")
        ap("")

    if any(metricas[M].get("ppc_z") for M in presentes):
        ap("## Nota sobre el PPC\n")
        ap("Con N grande los p-values bayesianos saturan a 0 o 1 ante "
           "discrepancias practicamente nulas (p.ej. una media replicada "
           "252.1 vs 252.5 observada da p=0.000). Por eso la figura de "
           "calibracion reporta la discrepancia ESTANDARIZADA "
           "z = (T_obs - T_rep_medio)/sd(T_rep), que conserva la magnitud y "
           "es comparable entre modelos ajustados a los mismos datos. El "
           "conteo de p-values extremos se mantiene solo como referencia.\n")

    ap("## Procedencia de cada numero\n")
    ap("Ver `comparacion_metricas.csv` (columna `fuente`): distingue lo "
       "leido de los artefactos de cada suite de lo recalculado por la capa "
       "uniforme, para trazabilidad en el capitulo comparativo.\n")
    ruta = Path(dir_diag) / "reporte_comparativo.md"
    ruta.write_text("\n".join(L), encoding="utf-8")
    print(f"  [md]  {ruta}")


# =============================================================================
# 10. MAIN
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Comparacion de diagnosticos de los modelos M1-M4")
    for M in ("m1", "m2", "m3", "m4"):
        ap.add_argument(f"--{M}", default=None,
                        help=f"carpeta de diagnosticos de {M.upper()}")
        ap.add_argument(f"--{M}-mcmc", nargs="+", default=None,
                        dest=f"{M}_mcmc",
                        help=f"carpeta(s) MCMC/export de {M.upper()} "
                             "(activa la capa uniforme)")
    ap.add_argument("--diag", default="comparacion_modelos")
    ap.add_argument("--nombres", nargs=4, default=None,
                    metavar=("M1", "M2", "M3", "M4"),
                    help="nombres para las figuras (4 valores)")
    ap.add_argument("--draws", type=int, default=400,
                    help="draws de la capa uniforme (mismos en los 4 modelos)")
    ap.add_argument("--bloque-obs", type=int, default=50_000, dest="bloque",
                    help="obs por bloque de la capa uniforme (RAM ~ bloque x draws x 8B x 4)")
    ap.add_argument("--sin-loo", action="store_true", dest="sin_loo",
                    help="en la capa uniforme, omitir PSIS-LOO (usar WAIC)")
    ap.add_argument("--sin-recalculo", action="store_true", dest="sin_rec",
                    help="ignorar --mX-mcmc y usar solo artefactos")
    ap.add_argument("--pdf", action="store_true",
                    help="guardar tambien las figuras en PDF vectorial")
    ap.add_argument("--sin-radar", action="store_true", dest="sin_radar",
                    help="omitir comp_07_radar (sintesis polar)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.nombres:
        for M, n in zip(("M1", "M2", "M3", "M4"), args.nombres):
            MODELOS[M]["nombre"] = n

    estilo()
    dir_diag = Path(args.diag); dir_diag.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("COMPARACION DE MODELOS M1-M4")
    print("=" * 72)

    presentes, metricas, fuentes, notas = [], {}, {}, []
    for M in ("M1", "M2", "M3", "M4"):
        d = getattr(args, M.lower())
        if d is None:
            continue
        if not Path(d).exists():
            print(f"AVISO: no existe {d}; {M} omitido.")
            continue
        print(f"\n[{M}] leyendo artefactos de {d} ...")
        met, fu, nn = LECTORES[M](d)
        presentes.append(M)
        metricas[M], fuentes[M] = met, fu
        notas += nn
        n_ok = sum(1 for v in met.values()
                   if isinstance(v, float) and np.isfinite(v))
        print(f"  {n_ok} metricas leidas de artefactos")
        # Diagnostico explicito: avisar de las metricas clave del tablero que
        # NO se pudieron leer, para no descubrirlo despues como celda vacia.
        claves_tablero = [
            ("rhat_max", "R-hat maximo"),
            ("ess_bulk_min", "ESS bulk minimo"),
            ("elpd_waic", "elpd WAIC"),
            ("dic", "DIC"),
        ]
        faltan = [etq for k, etq in claves_tablero
                  if not np.isfinite(_num(met.get(k, np.nan)))]
        if faltan:
            print(f"  AVISO [{M}]: sin dato para -> {', '.join(faltan)}")
            try:
                pres = sorted(f.name for f in Path(d).iterdir()
                              if f.is_file() and f.suffix.lower()
                              in (".csv", ".json", ".txt", ".md"))
                if pres:
                    print(f"     archivos en la carpeta: {', '.join(pres[:12])}"
                          + (" ..." if len(pres) > 12 else ""))
            except Exception:
                pass
    if len(presentes) < 2:
        sys.exit("ERROR: se necesitan al menos 2 modelos (--m1 ... --m4).")

    # ---------------- capa uniforme ----------------
    uniformes = {}
    capa_b = []
    if not args.sin_rec:
        for M in presentes:
            dm = getattr(args, f"{M.lower()}_mcmc")
            if not dm:
                continue
            print(f"\n[{M}] capa uniforme desde {dm} ...")
            try:
                fuente = FUENTES[M](dm)
            except Exception as e:
                print(f"  AVISO: no pude cargar las muestras de {M}: {e}")
                continue
            uni = evaluar_uniforme(fuente, draws=args.draws,
                                   sin_loo=args.sin_loo,
                                   bloque_obs=args.bloque)
            uniformes[M] = uni
            capa_b.append(M)
            for k in ("r2", "r2_bayes", "rmse", "mae", "cob_50", "cob_80",
                      "cob_95", "ks_pit", "ks_pit_d", "elpd_waic",
                      "se_elpd_waic", "p_waic", "elpd_loo", "se_elpd_loo",
                      "p_loo", "prop_khat_gt_07", "N"):
                if k in uni and np.isfinite(_num(uni[k])):
                    metricas[M][k] = uni[k]
                    fuentes[M][k] = f"capa uniforme ({uni['draws']} draws)"
            for k in ("prop_ess_lt_400", "ppc_z", "ppc_z_max", "ppc_pvals",
                      "n_ppc_extremos"):
                tiene = (k in metricas[M] and metricas[M][k] is not None
                         and (not isinstance(metricas[M][k], float)
                              or np.isfinite(metricas[M][k])))
                if (not tiene) and k in uni and uni[k] is not None:
                    metricas[M][k] = uni[k]
                    fuentes[M][k] = (f"capa uniforme ({uni['draws']} draws, "
                                     "relleno)")
            metricas[M]["draws_uniforme"] = uni["draws"]
            if np.isfinite(uni.get("ess_bulk_min_unif", np.nan)):
                metricas[M]["ess_bulk_min_unif"] = uni["ess_bulk_min_unif"]
            np.savez_compressed(
                dir_diag / f"elpd_i_{M}.npz",
                elpd_waic_i=uni["elpd_waic_i"],
                **({"elpd_loo_i": uni["elpd_loo_i"], "khat_i": uni["khat_i"]}
                   if "elpd_loo_i" in uni else {}),
                N=uni["N"], draws=uni["draws"])
            print(f"  elpd_waic={uni['elpd_waic']:,.1f} "
                  f"(SE {uni['se_elpd_waic']:,.1f})"
                  + (f" | elpd_loo={uni['elpd_loo']:,.1f}"
                     f" (SE {uni['se_elpd_loo']:,.1f})"
                     if "elpd_loo" in uni else "")
                  + f" | R2={uni['r2']:.3f} RMSE={uni['rmse']:.2f}"
                  + f" | cob95={uni['cob_95']:.3f} KS={uni['ks_pit']:.3f}")
        dif = [M for M in capa_b
               if metricas[M].get("draws_uniforme") != args.draws]
        if capa_b and len({metricas[M].get("draws_uniforme")
                           for M in capa_b}) > 1:
            notas.append("Capa uniforme con numero de draws DISTINTO entre "
                         "modelos (algun modelo tiene menos guardadas): "
                         + ", ".join(f"{M}={metricas[M]['draws_uniforme']}"
                                     for M in capa_b) + ".")
        _ = dif

    # ---------------- Delta elpd pareado ----------------
    print("\nDelta_elpd por pares ...")
    df_delta = delta_elpd_pares(presentes, uniformes, metricas)
    if len(df_delta):
        df_delta.to_csv(dir_diag / "comparacion_delta_elpd.csv", index=False,
                        encoding="utf-8")
        for _, r in df_delta.iterrows():
            print(f"  {r['par']:<10} {r['criterio']:<5} "
                  f"Delta={_fmt(r['delta_elpd'], 1, True):>14}  "
                  f"SE={_fmt(r['se'], 1, True):>10}  -> {r['preferido']}  "
                  f"[{r['metodo']}]")
        print(f"  [csv] {dir_diag}/comparacion_delta_elpd.csv")
    else:
        print("  (sin pares con elpd puntual; use --mX-mcmc)")

    # ---------------- comparabilidad de datos ----------------
    Ns = {M: metricas[M].get("N", np.nan) for M in presentes}
    ns_fin = {int(v) for v in Ns.values() if np.isfinite(v)}
    mismos_datos = len(ns_fin) <= 1
    ys = [uniformes[M]["y"] for M in presentes if M in uniformes]
    if mismos_datos and len(ys) >= 2:
        if not all(np.array_equal(ys[0], yy) for yy in ys[1:]):
            mismos_datos = False
            notas.append("N coincide pero el vector y difiere (u orden "
                         "distinto) entre modelos con capa uniforme: elpd/"
                         "WAIC/DIC tratados como NO comparables.")
    if not mismos_datos and len(ns_fin) > 1:
        notas.append("N difiere entre modelos ("
                     + ", ".join(f"{M}={_fmt(Ns[M], 0, True)}"
                                 for M in presentes)
                     + "): la dimension 'Prediccion (elpd)' se excluyo del "
                       "ranking y esas filas van en gris en el tablero.")

    # ---------------- ranking ----------------
    tabla_dim, glob = puntajes_dimensiones(presentes, metricas, mismos_datos)
    df_rank = pd.DataFrame(
        [{"modelo": M, **{d: tabla_dim[d][M] for d in tabla_dim},
          "global": glob[M]} for M in presentes]).sort_values(
        "global", ascending=False)
    df_rank.to_csv(dir_diag / "comparacion_ranking.csv", index=False,
                   encoding="utf-8")
    print(f"  [csv] {dir_diag}/comparacion_ranking.csv")

    df_tidy = tabla_tidy(presentes, metricas, fuentes)
    df_tidy.to_csv(dir_diag / "comparacion_metricas.csv", index=False,
                   encoding="utf-8")
    print(f"  [csv] {dir_diag}/comparacion_metricas.csv")

    # ---------------- figuras ----------------
    print("\nFiguras ...")
    nota_nu = ""
    if "M4" in presentes and np.isfinite(
            metricas["M4"].get("rhat_max", np.nan)):
        m4 = metricas["M4"]
        if (np.isfinite(m4.get("rhat_max_sin_nu", np.nan))
                and m4["rhat_max"] - m4["rhat_max_sin_nu"] > 0.02):
            nota_nu = (f"R-hat max={m4['rhat_max']:.2f}, "
                       f"ESS min={m4.get('ess_bulk_min', np.nan):.0f}")
    fig_tablero(presentes, metricas, fuentes, dir_diag, args.pdf, nota_nu,
                mismos_datos)
    fig_convergencia(presentes, metricas, dir_diag, args.pdf)
    fig_ajuste(presentes, metricas, dir_diag, args.pdf)
    fig_informacion(presentes, metricas, df_delta, dir_diag, args.pdf,
                    mismos_datos)
    fig_calibracion(presentes, metricas, dir_diag, args.pdf)
    fig_perfil(presentes, tabla_dim, glob, dir_diag, args.pdf)
    if not args.sin_radar:
        fig_radar(presentes, tabla_dim, glob, dir_diag, args.pdf)
    figs_delta_depto(df_delta, uniformes, dir_diag, args.pdf)

    escribir_reporte(dir_diag, presentes, metricas, fuentes, df_delta,
                     tabla_dim, glob, notas, capa_b)

    # ---------------- resumen en consola ----------------
    print("\n" + "=" * 72)
    print("RESUMEN")
    print("=" * 72)
    orden = sorted((M for M in presentes if np.isfinite(glob.get(M, np.nan))),
                   key=lambda M: -glob[M])
    for dim in tabla_dim:
        vv = {M: tabla_dim[dim][M] for M in presentes
              if np.isfinite(tabla_dim[dim][M])}
        if vv:
            mejor = max(vv, key=vv.get)
            print(f"  {dim:<22}: mejor {mejor} "
                  f"({', '.join(f'{M}={vv[M]:.2f}' for M in vv)})")
    if orden:
        print("  Ranking global (heuristico): "
              + " > ".join(f"{M} ({glob[M]:.2f})" for M in orden))
    print("  Criterio formal: Delta_elpd pareado "
          "(comparacion_delta_elpd.csv / comp_04).")
    for n in notas:
        print(f"  AVISO: {n}")
    print(f"\nTodo en {dir_diag}/ (empezar por comp_01_tablero.png y "
          f"reporte_comparativo.md).")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
