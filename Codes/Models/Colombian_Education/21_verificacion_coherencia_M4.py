# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# @  MODELO 4 - VERIFICACION DE COHERENCIA ENTRE LA CORRIDA Y LAS FIGURAS
# _____________________________________________________________________________
import json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import estilo_M4 as E

# Valores de referencia tomados de las tablas del diagnostico en R.
# tabla_ajuste_waic_M4.csv y tabla_convergencia_monitoreados_M4.csv
REF = dict(RMSE=43.88908095166153, R2=0.2930606263104839,
           sd_obs=52.19685418306522, nu_mediana=273.92359492761904,
           phi2=dict(E=1.2908900972781667, M=1.0956426959628882,
                     D=1.190662461819262),
           lambda2=dict(E=0.1419582741588383, M=0.5330994796671511,
                        D=0.07275749113442606))

TOL = 0.02          # tolerancia relativa por defecto
_fallas = []

def veredicto(nombre, ok, detalle, duro=True):
    et = "OK      " if ok else ("FALLA   " if duro else "REVISAR ")
    print(f"  [{et}] {nombre}\n            {detalle}")
    if not ok and duro:
        _fallas.append(nombre)

def titulo(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

D = E.cargar_cache(con_varianzas=False)
with open(os.path.join(E.RUTA_EXPORT, "meta.json"), encoding="utf-8") as fh:
    meta = json.load(fh)
m, Tt, p, S = meta["m"], meta["Tt"], meta["p"], meta["S"]
labels = meta["theta_labels"]

# =============== 1. ORDEN DE LECTURA DE phi2 Y lambda2 =======================
titulo("1. ORDEN DE LECTURA DE LAS ESCALAS DEL LASSO (phi2, lambda2)")
Dv = E.cargar_cache(con_varianzas=False)
for nom, ref in (("phi2", (1.2909, 1.0956, 1.1907)),
                 ("lambda2", (0.1420, 0.5331, 0.0728))):
    obs = np.median(Dv[nom], axis=1)
    err = max(abs(a / b - 1) for a, b in zip(obs, ref))
    print(f"    {nom}: cache (E,M,D) = {np.round(obs,4).tolist()}")
    print(f"           resumen_M4.txt = {list(ref)}   error rel max {err:.4f}")
    veredicto(f"{nom} guardado con el orden correcto", err < 0.02,
              "si las tres medianas salen casi iguales, el modulo 16 esta "
              "leyendo (3,S) con order='F'; debe ser (S,3).T")
del Dv

# =============== 2. INDICES DE BLOQUE FRENTE A LAS LISTAS DEL SCRIPT 15 ======
titulo("2. BLOQUES E / M / D: meta.json FRENTE A LAS LISTAS LITERALES DE estilo_M4")
# respaldo literal de estilo_M4.LM / .LD
LM = ["TASA_HOMICIDIOS", "PROP_HOMICIDIOS_URBANO", "TASA_VICTIMAS",
      "TASA_MATRICULA_ES", "IES_CON_OFERTA", "SGP_EDUC_PC", "PROP_NBI",
      "COMP_INASISTENCIA_ESCOLAR", "PROP_URBANA", "LOG_POBLACION",
      "SGP_EDUC_PC_MISSING"]
LD = ["PIB_CONSTANTE", "CREC_PIB_REAL", "TASA_DESEMPLEO", "TASA_SUBOCUPACION",
      "IPM", "CREC_PIB_REAL_MISSING", "TASA_DESEMPLEO_MISSING",
      "TASA_SUBOCUPACION_MISSING"]
M_meta = {labels[i - 1] for i in meta["idx_M"]}
D_meta = {labels[i - 1] for i in meta["idx_D"]}
print(f"    p = {p}   |E| = {len(meta['idx_E'])}   |M| = {len(M_meta)}   "
      f"|D| = {len(D_meta)}")
for et, lit, real in (("M", set(LM), M_meta), ("D", set(LD), D_meta)):
    veredicto(f"bloque {et} coincide", lit == real,
              f"solo en estilo_M4: {sorted(lit - real)} | "
              f"solo en meta: {sorted(real - lit)}")
veredicto("labels[0] es el intercepto", labels[0] == "intercepto",
          f"labels[0] = {labels[0]!r}, idx_mu = {meta['idx_mu']}")

# =============== 3. UNIDADES DE LAS COVARIABLES CONTEXTUALES =================
titulo("3. ESCALA ORIGINAL DE LAS COVARIABLES M Y D (multiplicadores del 20)")
MULT = {n: (m, e) for n, m, e in list(E.MULT_MUNI) + list(E.MULT_DEPTO)}
ev, em, es = D["est_var"], D["est_media"], D["est_sd"]
print(f"    {'variable':<28} {'media':>14} {'desvest':>14} {'mult':>12} "
      f"{'mult/sd':>9}")
for nom, (mult, et) in MULT.items():
    i = np.where(ev == nom)[0]
    if not len(i):
        veredicto(f"{nom} presente en estandar_params", False,
                  "sd_de() devolveria 1.0 EN SILENCIO y la magnitud seria falsa")
        continue
    mu, sd = float(em[i[0]]), float(es[i[0]])
    print(f"    {nom:<28} {mu:>14.4f} {sd:>14.4f} {mult:>12.4g} "
          f"{mult/sd:>9.3f}   ({et})")
print("\n    Criterio: si la media de una variable 'PROP_' o de un indice esta\n"
      "    entre 0 y 1, el multiplicador para +10 pp es 0.10; si esta entre 0 y\n"
      "    100, es 10. Un multiplicador mayor que ~3 desviaciones estandar\n"
      "    (mult/sd > 3) extrapola fuera del soporte observado.")
for nom, (mult, _) in MULT.items():
    i = np.where(ev == nom)[0]
    if len(i) and mult / float(es[i[0]]) > 3:
        veredicto(f"{nom}: multiplicador dentro del soporte", False,
                  f"mult/sd = {mult/float(es[i[0]]):.2f} desviaciones estandar",
                  duro=False)

# =============== 4. LA ESCALA RESIDUAL REPRODUCE EL RMSE =====================
titulo("4. LA ESCALA RESIDUAL DEL MODELO REPRODUCE EL RMSE DEL DIAGNOSTICO")
nM = D["n_M"]
kap2_jkt = D["sd_jkt_plugin"] ** 2
w = nM / nM.sum()
kap2_bar = float(np.nansum(np.where(nM > 0, kap2_jkt * w, 0)))
nu = REF["nu_mediana"]
factor = nu / (nu - 2)
esperado = REF["RMSE"] ** 2 / factor
err = abs(kap2_bar / esperado - 1)
print(f"    media ponderada de kappa2_jkt        {kap2_bar:12.1f}")
print(f"    RMSE^2 / (nu/(nu-2))                 {esperado:12.1f}"
      f"   (RMSE = {REF['RMSE']:.2f}, nu = {nu:.1f}, factor = {factor:.4f})")
print(f"    raiz de la media ponderada           {np.sqrt(kap2_bar):12.2f} puntos")
veredicto("kappa2 reproduce el error cuadratico observado", err < 0.05,
          f"error relativo {err:.3f}; una brecha grande indicaria que el "
          f"cache no contiene la escala del modelo ajustado")
r2_implicito = 1 - REF["RMSE"] ** 2 / REF["sd_obs"] ** 2
veredicto("R2 del diagnostico es coherente con RMSE y sd",
          abs(r2_implicito - REF["R2"]) < 1e-3,
          f"1 - RMSE^2/sd^2 = {r2_implicito:.4f} contra R2 = {REF['R2']:.4f}")

# =============== 5. PLUG-IN CONTRA POSTERIOR: LA BRECHA DE JENSEN ============
titulo("5. DOS DEFINICIONES DE sd_kt CONVIVIENDO EN LAS FIGURAS")
f_var = os.path.join(E.DIR_CACHE, "varianzas_M4.npz")
if not os.path.exists(f_var):
    print("    varianzas_M4.npz no encontrado: ejecutar antes 17_varianzas_M4.py")
else:
    V = np.load(f_var)
    plug = D["sd_kt_plugin"]
    post = V["sd_kt_med"]
    rel = np.abs(post / plug - 1)
    print(f"    plug-in  (1/E[phi])^(1/2)  mediana {np.median(plug):8.3f} puntos")
    print(f"    posterior mediana de phi^(-1/2)    {np.median(post):8.3f} puntos")
    print(f"    discrepancia relativa: mediana {np.median(rel):.5f}  "
          f"max {rel.max():.5f}")
    veredicto("las dos definiciones son intercambiables", rel.max() < 0.01,
              "si supera el 1% ningun modulo debe usar sd_kt_plugin en figuras", duro=False)

    # ---- 5b. cuanto depende el cambio de la pareja de periodos ----
    pares = np.where(D["es_par"])[0]
    d_par = post[:, pares[-1]] - post[:, pares[0]]
    d_ext = plug[:, -1] - plug[:, 0]
    print(f"\n    cambio de sd entre periodos PARES  (2017-2 -> 2024-2): "
          f"media {d_par.mean():+.3f}, {int((d_par>0).sum())}/{m} suben")
    print(f"    cambio de sd de t=1 a t=16 (script 15, mezcla semestres): "
          f"media {d_ext.mean():+.3f}, {int((d_ext>0).sum())}/{m} suben")
    dif_signo = int((np.sign(d_par) != np.sign(d_ext)).sum())
    veredicto("el cambio de dispersion no depende de la pareja de periodos",
              dif_signo == 0,
              f"{dif_signo} de {m} departamentos cambian de signo y la media se "
              f"infla {100*(d_ext.mean()/d_par.mean()-1):.0f}% al comparar t=1 "
              f"con t=16. No es un defecto del pipeline: es la magnitud del "
              f"artefacto semestral, y justifica la regla par-par.", duro=False)
    print(f"    probabilidades del 17: P(Delta>0)>0.95 en "
          f"{int((V['P_dsd']>0.95).sum())} de {m} departamentos "
          f"(cota SUPERIOR: la reconstruccion estrecha los intervalos)")

# =============== 6. RECENTRADO DE v: LA RESTRICCION SE SOSTIENE ==============
titulo("6. RESTRICCION DE IDENTIFICABILIDAD SOBRE v_jk")
lv = np.log(D["v_jk"])
md = D["muni_depto"]
s_k = np.array([lv[md == k].mean() for k in range(m)])
print(f"    media de log v_jk por departamento: min {s_k.min():+.4f}  "
      f"mediana {np.median(s_k):+.4f}  max {s_k.max():+.4f}")
veredicto("la media geometrica por departamento sigue cerca de 1",
          np.abs(s_k).max() < 0.25,
          "una desviacion grande indica que v_mean no proviene de la cadena "
          "recentrada, y entonces el supuesto del 17 falla", duro=False)

# =============== 7. CONTEOS: n_D FRENTE A meta N_t ===========================
titulo("7. CONTEOS POR PERIODO")
Nt_meta = np.asarray(meta["N_t"], dtype=float)
Nt_cache = D["n_D"].sum(axis=0)
veredicto("n_D reproduce N_t de meta.json",
          np.allclose(Nt_meta, Nt_cache),
          f"max |dif| = {np.abs(Nt_meta - Nt_cache).max():.0f}")
pares = np.where(D["es_par"])[0]
imp = np.setdiff1d(np.arange(Tt), pares)
print(f"    periodos pares  : {Nt_cache[pares].sum():>10,.0f} obs "
      f"({100*Nt_cache[pares].sum()/Nt_cache.sum():.1f}%)")
print(f"    periodos impares: {Nt_cache[imp].sum():>10,.0f} obs "
      f"({100*Nt_cache[imp].sum()/Nt_cache.sum():.1f}%)")
print("    El promedio temporal de theta que usa P.resumen() pondera\n"
      "    los 16 periodos por igual, no por tamano muestral: los impares\n"
      "    pesan la mitad del promedio pese a ser una fraccion pequena de N.")

# =============== 8. EL RANKING DEL 20 Y LOS EFECTOS QUE INVIERTEN SIGNO ======
titulo("8. RANKING POR MAGNITUD PROMEDIO CONTRA MAGNITUD MAXIMA")
theta = np.fromfile(os.path.join(E.RUTA_EXPORT, "theta.bin")).reshape(
    (p, Tt + 1, S), order="F")[:, 1:, :]
prom = np.abs(np.median(theta.mean(axis=1), axis=1))     # criterio del 20
maxi = np.abs(np.median(theta, axis=2)).max(axis=1)      # magnitud maxima en t
r_prom = (-prom).argsort().argsort() + 1
r_maxi = (-maxi).argsort().argsort() + 1
print(f"    {'variable':<44} {'|prom|':>8} {'max|t|':>8} {'r_prom':>7} {'r_max':>6}")
saltos = np.argsort(-(r_prom - r_maxi))[:8]
for i in saltos:
    print(f"    {labels[i]:<44} {prom[i]:>8.2f} {maxi[i]:>8.2f} "
          f"{r_prom[i]:>7d} {r_maxi[i]:>6d}")
print("\n    Las covariables que invierten de signo (internet, por ejemplo)\n"
      "    tienen promedio temporal cercano a cero y caen al fondo del ranking\n"
      "    de la fig12 pese a ser efectos grandes en cada periodo.")

# =============== 9. PROMEDIO TEMPORAL: 16 PERIODOS CONTRA SOLO LOS PARES =====
titulo("9. EFECTO PROMEDIO: LOS 16 PERIODOS CONTRA SOLO LOS OCHO PARES")
idx_par = np.where(D["es_par"])[0]
bar_all = theta.mean(axis=1)
bar_par = theta[:, idx_par, :].mean(axis=1)
med_a, med_p = np.median(bar_all, axis=1), np.median(bar_par, axis=1)
lo_a = np.quantile(bar_all, .025, axis=1); hi_a = np.quantile(bar_all, .975, axis=1)
lo_p = np.quantile(bar_par, .025, axis=1); hi_p = np.quantile(bar_par, .975, axis=1)
cred_a = (lo_a > 0) | (hi_a < 0)
cred_p = (lo_p > 0) | (hi_p < 0)
dif = med_p - med_a
print(f"    {'variable':<44} {'16 per.':>9} {'pares':>9} {'dif':>8} {'IC95':>12}")
for i in np.argsort(-np.abs(dif))[:14]:
    et = ("credible en ambos" if cred_a[i] and cred_p[i] else
          "solo con 16" if cred_a[i] else
          "solo con pares" if cred_p[i] else "en ninguno")
    print(f"    {labels[i]:<44} {med_a[i]:>+9.2f} {med_p[i]:>+9.2f} "
          f"{dif[i]:>+8.2f} {et:>12}")
n_signo = int(((np.sign(med_a) != np.sign(med_p)) & (np.abs(dif) > 0.5)).sum())
n_cred = int((cred_a != cred_p).sum())
veredicto("el promedio de 16 periodos coincide con el de pares",
          n_signo == 0 and n_cred == 0,
          f"{n_signo} coeficientes cambian de signo y {n_cred} cambian de "
          f"estatus de credibilidad segun se promedien 16 periodos u 8 pares",
          duro=False)
if "calen_B" in labels:
    ic = labels.index("calen_B")
    sp = np.median(theta[ic], axis=1)
    print(f"\n    calen_B por periodo (mediana): impares "
          f"{np.round(sp[::2], 1).tolist()}")
    print(f"                                  pares   "
          f"{np.round(sp[1::2], 1).tolist()}")
    print("    Si las dos series estan en escalas distintas, el promedio de 16\n"
          "    periodos de calen_B no admite lectura sustantiva (figR6).")

# =============== 10. GEOJSON: ANILLOS SUELTOS Y SOLAPAMIENTOS ================
titulo("10. GEOJSON DEPARTAMENTAL: ANILLOS POR DEPARTAMENTO Y SOLAPAMIENTOS")

def _dentro(pt, poly):
    """Ray casting vectorizado; poly es un anillo cerrado (n,2)."""
    x, y = float(pt[0]), float(pt[1])
    x1, y1 = poly[:-1, 0], poly[:-1, 1]
    x2, y2 = poly[1:, 0], poly[1:, 1]
    cruza = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
    return bool(np.sum(cruza & (x < xint)) % 2)

if not os.path.exists(E.RUTA_GEOJSON):
    print(f"    {E.RUTA_GEOJSON} no encontrado: ejecutar antes 18_geografia_M4.py")
else:
    geo = E.cargar_geojson(E.RUTA_GEOJSON, "DPTO_CCDGO",
                           orden=D["depto_ids"], clave=int)
    cods = geo.POLI_COD.astype(int)
    n_an = {int(c): int((cods == int(c)).sum()) for c in D["depto_ids"]}
    multi = {k: v for k, v in n_an.items() if v > 1}
    print(f"    poligonos totales {len(cods)}; departamentos con mas de un "
          f"anillo: {sorted(multi.items())}")
    solapes = []
    for cod in D["depto_ids"]:
        pt = geo.CENTRO.get(int(cod))
        if pt is None:
            continue
        for poly, c in zip(geo.POLIS, cods):
            if int(c) != int(cod) and _dentro(pt, poly):
                solapes.append((int(cod), int(c)))
                break
    veredicto("ningun departamento queda cubierto por un anillo de otro",
              not solapes,
              f"pares (cubierto, cubridor): {solapes}. El anillo interior de un "
              f"departamento se dibuja relleno y tapa al que encierra si aparece "
              f"despues en el archivo")
    # el orden de dibujo es el del archivo, no el de depto_ids
    orden_arch = [int(c) for c in dict.fromkeys(cods.tolist())]
    print(f"    orden de dibujo (primeros 8): {orden_arch[:8]}   "
          f"un poligono posterior se pinta encima del anterior")


# =============== 11. COLISION DE CLAVES ENTRE LOS npz DEL CACHE ==============
titulo("11. COLISION DE CLAVES ENTRE prep / nivel / varianzas")
vistas, choques = {}, []
for f in ("prep_M4.npz", "nivel_M4.npz", "varianzas_M4.npz"):
    ruta = os.path.join(E.DIR_CACHE, f)
    if not os.path.exists(ruta):
        continue
    with np.load(ruta, allow_pickle=True) as z:
        for k in z.files:
            if k in vistas:
                choques.append((k, vistas[k], f))
            vistas[k] = f
veredicto("ninguna clave se repite entre los tres npz", not choques,
          f"repetidas: {choques}. Renombrar en el modulo que la produce.")

# =============== 12. NINGUN MODULO PROMEDIA LOS 16 PERIODOS POR SU CUENTA ====
titulo("12. PROMEDIO TEMPORAL DUPLICADO EN LOS MODULOS")
patron = re.compile(r"theta\[[^\]]*1:[^\]]*\]\s*\.mean\(axis=1\)")
DIR_MOD = os.path.dirname(os.path.abspath(__file__))
sospechosos = []
for f in sorted(os.listdir(DIR_MOD)):
    if not f.endswith(".py") or f.startswith("21_"):
        continue
    txt = open(os.path.join(DIR_MOD, f), encoding="utf-8").read()
    for i, ln in enumerate(txt.splitlines(), 1):
        if patron.search(ln) and "bar_16" not in ln:
            sospechosos.append(f"{f}:{i}")
veredicto("solo estilo_M4 calcula el promedio temporal", not sospechosos,
          f"promedio duplicado en {sospechosos}; deben usar P.resumen()/P.draws()")

# =============== 13. ETIQUETAS DE LOS MAPAS DENTRO DE SU POLIGONO ============
titulo("13. PUNTO REPRESENTATIVO DE CADA DEPARTAMENTO")
if os.path.exists(E.RUTA_GEOJSON):
    geo2 = E.cargar_geojson(E.RUTA_GEOJSON, "DPTO_CCDGO",
                            orden=D["depto_ids"], clave=int)
    fuera = []
    for cod in D["depto_ids"]:
        pt = geo2.CENTRO.get(int(cod))
        if pt is None:
            continue
        propios = [p for p, c in zip(geo2.POLIS, geo2.POLI_COD.astype(int))
                   if c == int(cod)]
        if not any(_dentro(pt, p) for p in propios):
            fuera.append(int(cod))
    veredicto("la etiqueta de cada departamento cae dentro de su territorio",
              not fuera, f"fuera de su poligono: {fuera}")


# =============== 14. EL MAPA DEPARTAMENTAL SUBESTIMA LA DISPERSION ===========
titulo("14. sd_kt DEPARTAMENTAL FRENTE A LA ESCALA DEL ESTUDIANTE TIPICO")
sd_kt_u = float(np.nanmean(D["sd_kt_plugin"]))
w_j = nM / nM.sum()
rms_j = float(np.sqrt(np.nansum(np.where(nM > 0, D["sd_jkt_plugin"] ** 2 * w_j, 0))))
raz = (rms_j / sd_kt_u) ** 2
print(f"    media de sd_kt sobre (k,t)                  {sd_kt_u:8.2f} puntos")
print(f"    RMS ponderado de sd_jkt sobre (j,k,t)       {rms_j:8.2f} puntos")
print(f"    razon en varianza, E_j[1/v_jk] implicito    {raz:8.3f}")
print(f"    el mapa departamental queda {100*(rms_j/sd_kt_u-1):.1f}% por debajo en "
      f"desviacion estandar")
print("    No es un error: son objetos distintos. Pero el pie del mapa debe decir\n"
      "    'municipio de dispersion tipica del departamento', no 'el departamento'.")

# =============== CIERRE ======================================================
titulo("RESUMEN")
if _fallas:
    print(f"  {len(_fallas)} verificaciones en FALLA:")
    for f in _fallas:
        print(f"    - {f}")
else:
    print("  Ninguna verificacion dura en FALLA.")
print("  Revisar ademas las marcadas REVISAR: no invalidan el calculo pero\n"
      "  obligan a precisar la definicion en el texto.")
