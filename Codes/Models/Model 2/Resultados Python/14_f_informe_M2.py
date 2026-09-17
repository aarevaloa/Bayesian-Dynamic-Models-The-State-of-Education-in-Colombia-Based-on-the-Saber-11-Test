# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# @  MODELO 2 - t-STUDENT JERARQUICO ESTATICO CON LASSO BAYESIANO
# @  ETAPA 14f: INFORME CONSOLIDADO
# _____________________________________________________________________________
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_AQUI = Path(__file__).resolve().parent


def _cargar(nombre):
    spec = importlib.util.spec_from_file_location(nombre.replace(".py", ""),
                                                  _AQUI / nombre)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


u = _cargar("14_0_utilidades_M2.py")
plt = u.plt


# =============================================================================
# 1. EJECUCION DE LAS ETAPAS
# =============================================================================
def correr_etapas(args, dir_diag):
    """Ejecuta 14a-14d en proceso; devuelve dict etapa -> estado."""
    estados = {}

    def _etapa(nombre, script, argv):
        print("\n" + "#" * 72)
        print(f"# {nombre}")
        print("#" * 72)
        try:
            mod = _cargar(script)
            rc = mod.main(argv)
            estados[nombre] = "OK" if not rc else f"ERROR (rc={rc})"
        except Exception as e:                       # noqa: BLE001
            estados[nombre] = f"ERROR ({type(e).__name__}: {e})"
            print(f"  /!\\ {nombre} fallo: {e}")
        finally:
            plt.close("all")

    base = list(args.dirs)
    comunes = ["--diag", str(dir_diag)]

    argv_a = base + comunes
    if args.rapido or args.sin_kappa2_jk:
        argv_a += ["--sin-kappa2-jk"]
    _etapa("14a convergencia", "14_a_convergencia_M2.py", argv_a)

    argv_b = base + comunes + ["--max-draws-xi", str(min(args.draws, 400))]
    _etapa("14b posterior", "14_b_posterior_M2.py", argv_b)

    argv_c = [base[0]] + comunes
    if args.sin_loo or args.rapido:
        argv_c += ["--sin-loo"]
    if args.comparar:
        argv_c += ["--comparar", args.comparar, "--nombres", *args.nombres]
    _etapa("14c ajuste", "14_c_ajuste_M2.py", argv_c)

    argv_d = base + comunes + ["--draws", str(args.draws), "--seed",
                               str(args.seed)]
    _etapa("14d ppc", "14_d_ppc_M2.py", argv_d)
    return estados


# =============================================================================
# 2. RECOLECCION DE METRICAS
# =============================================================================
def _leer_json(dir_diag, nombre):
    f = dir_diag / nombre
    if f.exists():
        with open(f, encoding="utf-8") as h:
            return json.load(h)
    return None


def recolectar(dir_diag):
    M = {"conv": _leer_json(dir_diag, "convergencia_resumen_M2.json"),
         "post": _leer_json(dir_diag, "posterior_resumen_M2.json"),
         "ppc": _leer_json(dir_diag, "ppc_resumen_M2.json"),
         "comp": _leer_json(dir_diag, "comparacion_resumen_M2.json")}
    f_aj = dir_diag / "ajuste_M2.csv"
    M["ajuste"] = (pd.read_csv(f_aj).iloc[0].to_dict() if f_aj.exists()
                   else None)
    f_dep = dir_diag / "elpd_por_depto_M2.csv"
    if f_dep.exists():
        dep = pd.read_csv(f_dep)
        M["elpd_depto_peor"] = dep.loc[dep["elpd_medio"].idxmin()].to_dict()
        M["elpd_depto_mejor"] = dep.loc[dep["elpd_medio"].idxmax()].to_dict()
    return M


# =============================================================================
# 3. RESUMEN EN PROSA
# =============================================================================
def _parrafo(texto, ancho=88):
    return "\n".join(textwrap.wrap(" ".join(texto.split()), width=ancho))


def _lectura_metodologica(M):
    """Cruce regimen de nu x calibracion del PIT x Delta_elpd: el nucleo del
    argumento de especificacion del capitulo."""
    conv, ppc, comp = M["conv"], M["ppc"], M["comp"]
    if conv is None or ppc is None:
        return ("La lectura metodologica no puede cerrarse porque falta la "
                "salida de 14a o de 14d.")
    reg = conv["nu_regimen"]; curt = ppc["curtosis_pit"]
    p = conv["nu_probs"]
    frases = []
    if reg == "CASI NORMAL":
        frases.append(
            f"La posterior de nu (media {conv['nu_media']:.1f}, "
            f"P(nu>30) = {p['P(nu>30)']:.2f}) indica que la robustez t no "
            f"encuentra colas pesadas que corregir: el modelo converge por si "
            f"solo hacia la verosimilitud normal, lo que constituye una "
            f"prueba de especificacion favorable al supuesto gaussiano del "
            f"Modelo 1.")
    elif reg == "COLAS PESADAS":
        frases.append(
            f"La posterior de nu (media {conv['nu_media']:.1f}, "
            f"P(nu<10) = {p['P(nu<10)']:.2f}) concentra masa en la zona de "
            f"colas pesadas: la mezcla Normal-Gamma esta activa y descuenta "
            f"observaciones atipicas que la verosimilitud normal del Modelo 1 "
            f"trataria con peso pleno.")
    else:
        frases.append(
            f"La posterior de nu (media {conv['nu_media']:.1f}) queda en la "
            f"zona intermedia: hay un descuento moderado de atipicos sin un "
            f"regimen claro de colas pesadas.")
    if curt < -0.3:
        frases.append(
            f"El PIT resulta platicurtico (curtosis en exceso de "
            f"Phi^-1(PIT) = {curt:+.2f}): la predictiva t es mas ancha de lo "
            f"que piden los datos, senal de que el desajuste distribucional "
            f"dominante proviene del soporte acotado y de la agregacion de "
            f"puntajes, no de colas pesadas, de modo que la robustez t ataca "
            f"un problema que los datos no presentan.")
    elif curt > 0.3:
        frases.append(
            f"El PIT sigue leptocurtico (curtosis {curt:+.2f}): ni siquiera "
            f"la t captura el exceso de cola, lo que apunta a un desajuste "
            f"mas profundo (asimetria estructural o mezclas) que ninguna "
            f"familia eliptica simple resuelve.")
    else:
        frases.append(
            f"El PIT queda bien calibrado (curtosis {curt:+.2f}), de modo que "
            f"la predictiva t describe adecuadamente la dispersion global de "
            f"los puntajes.")
    if comp is not None:
        signo = "a favor de" if comp["delta_elpd"] > 0 else "en contra de"
        cred = ("supera el umbral de credibilidad de dos errores estandar"
                if comp["creible_2se"] else
                "no alcanza el umbral de dos errores estandar, por lo que la "
                "evidencia es no concluyente")
        frases.append(
            f"En el ajuste relativo, Delta_elpd({comp['nombre_A']} - "
            f"{comp['nombre_B']}) = {comp['delta_elpd']:+.1f} con error "
            f"estandar {comp['se']:.1f} ({comp['criterio']}), {signo} "
            f"{comp['nombre_A']}, y {cred}.")
    return " ".join(frases)


def escribir_resumen(M, estados, args, dir_diag):
    conv, post, ppc, comp, aj = (M["conv"], M["post"], M["ppc"], M["comp"],
                                 M["ajuste"])
    L = []
    L.append("=" * 88)
    L.append("MODELO 2 - t-STUDENT JERARQUICO ESTATICO CON LASSO BAYESIANO")
    L.append("RESUMEN CONSOLIDADO DE DIAGNOSTICOS (suite 14a-14d, Python)")
    L.append("=" * 88)
    L.append(f"Fecha: {datetime.now():%Y-%m-%d %H:%M} | Cadenas: "
             f"{', '.join(args.dirs)}")
    L.append("Etapas: " + " | ".join(f"{k}: {v}" for k, v in estados.items()))
    L.append("")

    L.append("CONVERGENCIA")
    if conv:
        txt = (f"Se monitorearon {conv['n_parametros']} parametros en "
               f"{conv['n_cadenas']} cadena(s) de {conv['n_save_min']} "
               f"muestras. El Rhat maximo fue {conv['rhat_max']:.4f} "
               f"({conv['param_rhat_max']}) y los minimos de tamano efectivo "
               f"fueron {conv['ess_bulk_min']:.0f} (bulk) y "
               f"{conv['ess_tail_min']:.0f} (tail); "
               f"{conv['n_rhat_altos']} parametros superan el umbral 1.01 de "
               f"Rhat, {conv['n_ess_bulk_bajos']} quedan bajo 400 de ESS bulk "
               f"y {conv['n_geweke_fdr']} presentan Geweke significativo tras "
               f"FDR. El error de Monte Carlo de la media es "
               f"{conv['mcse_mu']:.4f} para mu y {conv['mcse_nu']:.4f} para "
               f"nu. Los grados de libertad quedan en regimen "
               f"{conv['nu_regimen']} (media {conv['nu_media']:.1f}, "
               f"P(nu<10) = {conv['nu_probs']['P(nu<10)']:.3f}, "
               f"P(nu>30) = {conv['nu_probs']['P(nu>30)']:.3f}, "
               f"P(nu>100) = {conv['nu_probs']['P(nu>100)']:.3f}). "
               f"Veredicto global: "
               f"{'cadenas utilizables para inferencia' if conv['convergencia_ok'] else 'NO concluir todavia (alargar la corrida o anadir cadenas)'}.")
        L.append(_parrafo(txt))
    else:
        L.append(_parrafo("La etapa 14a no dejo resumen (ver estados arriba)."))
    L.append("")

    L.append("POSTERIOR Y ROBUSTEZ")
    if post:
        niv = ", ".join(f"{v[0]} de {v[1]} en el nivel {k}"
                        for k, v in post["senal_por_nivel"].items())
        txt = (f"De los {post['n_coeficientes']} coeficientes, "
               f"{post['n_senal_ic95']} tienen intervalo de credibilidad del "
               f"95% que excluye el cero ({niv}). La escala residual "
               f"departamental media es kappa_k ~ "
               f"{post['kappa_k_global']:.1f} puntos.")
        if "xi_spearman" in post:
            txt += (f" La capa de robustez opera como se espera: la "
                    f"correlacion de Spearman entre el residuo absoluto y "
                    f"E[xi] es {post['xi_spearman']:+.3f}"
                    + (f" (contraste con xi.bin: "
                       f"{post['xi_corr_xi_bin']:.3f})"
                       if post.get("xi_corr_xi_bin") is not None else "")
                    + f", con {post['xi_n_descuento_fuerte']} observaciones "
                      f"({post['xi_pct_descuento_fuerte']:.2f}%) bajo "
                      f"descuento fuerte E[xi] < 0.5.")
        L.append(_parrafo(txt))
    else:
        L.append(_parrafo("La etapa 14b no dejo resumen."))
    L.append("")

    L.append("AJUSTE RELATIVO (WAIC / PSIS-LOO / DIC)")
    if aj:
        txt = (f"WAIC = {aj['criterio_WAIC']:.1f} con p_WAIC = "
               f"{aj['p_WAIC']:.1f} y elpd_WAIC = {aj['elpd_WAIC']:.1f}")
        if "criterio_LOOIC" in aj and np.isfinite(aj.get("criterio_LOOIC",
                                                         np.nan)):
            txt += (f"; LOOIC = {aj['criterio_LOOIC']:.1f} "
                    f"(elpd_LOO = {aj['elpd_LOO']:.1f}, p_LOO = "
                    f"{aj['p_LOO']:.1f}, khat maximo = {aj['khat_max']:.2f})")
        if np.isfinite(aj.get("DIC", np.nan)):
            txt += f"; DIC = {aj['DIC']:.1f} (p_DIC = {aj['p_DIC']:.1f})"
        txt += (f", sobre N = {int(aj['N'])} observaciones y "
                f"{int(aj['n_save'])} muestras.")
        if "elpd_depto_peor" in M:
            txt += (f" El ajuste local es mas debil en "
                    f"{M['elpd_depto_peor']['departamento']} "
                    f"(elpd medio {M['elpd_depto_peor']['elpd_medio']:.3f}) y "
                    f"mas fuerte en {M['elpd_depto_mejor']['departamento']} "
                    f"({M['elpd_depto_mejor']['elpd_medio']:.3f}).")
        L.append(_parrafo(txt))
        if comp:
            txt = (f"Comparacion formal ({comp['criterio']}): elpd de "
                   f"{comp['nombre_A']} = {comp['elpd_A']:.1f} frente a "
                   f"{comp['nombre_B']} = {comp['elpd_B']:.1f}; Delta_elpd = "
                   f"{comp['delta_elpd']:+.1f} (SE {comp['se']:.1f}, "
                   f"{comp['ratio_se']:.1f} SE). Preferido: {comp['mejor']}"
                   f"{' (diferencia creible, >2 SE)' if comp['creible_2se'] else ' (no concluyente, <2 SE)'}.")
            L.append(_parrafo(txt))
    else:
        L.append(_parrafo("La etapa 14c no dejo resumen (loglik.bin ausente?)."))
    L.append("")

    L.append("CHEQUEO PREDICTIVO POSTERIOR (ajuste absoluto)")
    if ppc:
        band = (", ".join(f"{k} ({v})" for k, v in ppc["banderas"].items())
                if ppc["banderas"] else "ninguno")
        txt = (f"Con {ppc['S_ppc']} replicas: {ppc['n_banderas_fuertes']} "
               f"estadisticos de test con p_bayes extremo; con bandera: "
               f"{band}. La cobertura empirica de los intervalos predictivos "
               f"es {ppc['cob_50']*100:.1f}% (nominal 50), "
               f"{ppc['cob_80']*100:.1f}% (nominal 80) y "
               f"{ppc['cob_95']*100:.1f}% (nominal 95). La curtosis en exceso "
               f"de Phi^-1(PIT) es {ppc['curtosis_pit']:+.3f} "
               f"({ppc['calibracion']}). El R2 bayesiano es "
               f"{ppc['r2_t_medio']:.3f} con la varianza t "
               f"(kappa2*nu/(nu-2)) y {ppc['r2_escala_medio']:.3f} con solo "
               f"kappa2; RMSE medio {ppc['rmse_medio']:.2f} y MAE medio "
               f"{ppc['mae_medio']:.2f}. La curtosis observada de y es "
               f"{ppc['curtosis_obs']:+.3f} y la asimetria "
               f"{ppc['asimetria_obs']:+.3f}.")
        L.append(_parrafo(txt))
        if "holdout" in ppc:
            h = ppc["holdout"]
            txt = (f"Fuera de muestra (N_test = {int(h['N_test'])}): "
                   f"lppd_test = {h['lppd_test']:.1f} "
                   f"({h['lppd_test_medio']:.3f} por observacion), CRPS medio "
                   f"= {h['crps_medio']:.3f}, cobertura 50/80/95 = "
                   f"{h['cob_50']*100:.1f}/{h['cob_80']*100:.1f}/"
                   f"{h['cob_95']*100:.1f}%, RMSE = {h['rmse']:.2f} y MAE = "
                   f"{h['mae']:.2f}.")
            L.append(_parrafo(txt))
    else:
        L.append(_parrafo("La etapa 14d no dejo resumen (export de datos "
                          "ausente?)."))
    L.append("")

    L.append("LECTURA METODOLOGICA PARA EL CAPITULO")
    L.append(_parrafo(_lectura_metodologica(M)))
    L.append("")
    L.append("=" * 88)
    L.append("Archivos: figuras fig_14a..14d_*.png, tablas *.csv, "
             "metricas_M2.json e INFORME_M2.pdf en esta carpeta.")
    L.append("=" * 88)

    ruta = dir_diag / "00_RESUMEN_M2.txt"
    ruta.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n  [txt] {ruta}")
    return ruta


# =============================================================================
# 4. PDF CONSOLIDADO
# =============================================================================
def escribir_pdf(dir_diag, ruta_resumen, args):
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.image as mpimg

    pngs = sorted(dir_diag.glob("fig_14*.png"))
    ruta = dir_diag / "INFORME_M2.pdf"
    with PdfPages(ruta) as pdf:
        # portada
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.72, "Modelo 2", ha="center", fontsize=30,
                 weight="bold")
        fig.text(0.5, 0.66, "t-Student jerarquico estatico con LASSO "
                 "bayesiano", ha="center", fontsize=14)
        fig.text(0.5, 0.60, "Informe consolidado de diagnosticos "
                 "(suite 14a-14d)", ha="center", fontsize=12)
        fig.text(0.5, 0.50, f"Cadenas: {', '.join(args.dirs)}", ha="center",
                 fontsize=10)
        fig.text(0.5, 0.46, f"{datetime.now():%Y-%m-%d %H:%M}", ha="center",
                 fontsize=10, color="0.35")
        fig.text(0.5, 0.12, "Datos ICFES Saber 11 - jerarquia estudiante / "
                 "municipio / departamento", ha="center", fontsize=9,
                 color="0.35")
        pdf.savefig(fig); plt.close(fig)

        # resumen en texto (paginado)
        texto = Path(ruta_resumen).read_text(encoding="utf-8").splitlines()
        por_pag = 54
        for i0 in range(0, len(texto), por_pag):
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.06, 0.97, "\n".join(texto[i0:i0 + por_pag]),
                     ha="left", va="top", fontsize=8.2, family="monospace")
            pdf.savefig(fig); plt.close(fig)

        # figuras
        for f_png in pngs:
            img = mpimg.imread(f_png)
            alto, ancho = img.shape[:2]
            horizontal = ancho >= alto
            fig = plt.figure(figsize=(11, 8.5) if horizontal else (8.5, 11))
            ax = fig.add_axes([0.03, 0.03, 0.94, 0.90])
            ax.imshow(img); ax.axis("off")
            fig.suptitle(f_png.stem.replace("_", " "), fontsize=10, y=0.985)
            pdf.savefig(fig); plt.close(fig)
    print(f"  [pdf] {ruta}  ({1 + int(np.ceil(len(texto)/54)) + len(pngs)} "
          f"paginas)")
    return ruta


# =============================================================================
# 5. ORQUESTADOR
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="14f: informe consolidado del Modelo 2 (corre 14a-14d)")
    ap.add_argument("dirs", nargs="+", help="carpeta(s) de correr_gibbs_m2")
    ap.add_argument("--diag", default="diag_m2_py")
    ap.add_argument("--comparar", default=None,
                    help="carpeta con loglik.bin de otro modelo (M1/M3)")
    ap.add_argument("--nombres", nargs=2,
                    default=["t-Student (M2)", "Normal (M1)"])
    ap.add_argument("--rapido", action="store_true",
                    help="120 draws, sin kappa2_jk, sin PSIS-LOO")
    ap.add_argument("--sin-loo", action="store_true")
    ap.add_argument("--sin-kappa2-jk", action="store_true")
    ap.add_argument("--draws", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sin-pdf", action="store_true",
                    help="no generar INFORME_M2.pdf")
    args = ap.parse_args(argv)
    if args.draws is None:
        args.draws = 120 if args.rapido else 400

    u.estilo()
    dir_diag = Path(args.diag); dir_diag.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("INFORME CONSOLIDADO - Modelo 2 (14f)")
    print("=" * 72)
    print(f"Cadenas: {args.dirs} | diag: {dir_diag} | draws PPC: {args.draws}"
          f"{' | RAPIDO' if args.rapido else ''}"
          f"{' | sin LOO' if (args.sin_loo or args.rapido) else ''}")

    estados = correr_etapas(args, dir_diag)

    print("\n" + "#" * 72)
    print("# 14f: consolidacion")
    print("#" * 72)
    M = recolectar(dir_diag)
    ruta_resumen = escribir_resumen(M, estados, args, dir_diag)

    metricas = {"generado": f"{datetime.now():%Y-%m-%d %H:%M}",
                "cadenas": args.dirs, "etapas": estados,
                "convergencia": M["conv"], "posterior": M["post"],
                "ajuste": M["ajuste"], "comparacion": M["comp"],
                "ppc": M["ppc"]}
    with open(dir_diag / "metricas_M2.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2, default=float)
    print(f"  [json] {dir_diag}/metricas_M2.json")

    if not args.sin_pdf:
        escribir_pdf(dir_diag, ruta_resumen, args)

    print("\n" + "=" * 72)
    fallos = [k for k, v in estados.items() if v != "OK"]
    if fallos:
        print(f"14f COMPLETADO CON AVISOS: revisar {', '.join(fallos)}.")
    else:
        print("14f COMPLETADO: las cuatro etapas corrieron sin errores.")
    print(f"Todo en {dir_diag}/ (leer primero 00_RESUMEN_M2.txt).")
    print("=" * 72)
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
