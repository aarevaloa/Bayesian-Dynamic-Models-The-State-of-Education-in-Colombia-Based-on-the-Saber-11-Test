# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# @  MODELO 4 - APLICADOR DE PARCHES
# _____________________________________________________________________________
import ast, os, shutil, sys

DIR = os.path.dirname(os.path.abspath(__file__))
informe = []

def leer(nombre):
    ruta = os.path.join(DIR, nombre)
    if not os.path.exists(ruta):
        informe.append(("AUSENTE", nombre, "no esta en esta carpeta"))
        return None, None
    return ruta, open(ruta, encoding="utf-8").read()

def escribir(ruta, nombre, texto, etiqueta):
    try:
        ast.parse(texto)
    except SyntaxError as e:
        informe.append(("ABORTADO", nombre, f"{etiqueta}: el resultado no compila "
                                            f"(linea {e.lineno}). No se toco nada."))
        return False
    bak = ruta + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(ruta, bak)
    open(ruta, "w", encoding="utf-8").write(texto)
    return True

def sustituir(nombre, antes, despues, etiqueta, marca_ya=None):
    """Reemplaza un bloque unico. marca_ya: texto que, si ya esta, indica que
    el parche estaba aplicado."""
    ruta, s = leer(nombre)
    if s is None:
        return
    if marca_ya and marca_ya in s:
        informe.append(("YA ESTABA", nombre, etiqueta))
        return
    n = s.count(antes)
    if n == 0:
        informe.append(("NO ENCONTRADO", nombre,
                        f"{etiqueta}: no aparece el bloque original. "
                        f"Aplicarlo a mano."))
        return
    if n > 1:
        informe.append(("AMBIGUO", nombre,
                        f"{etiqueta}: el bloque aparece {n} veces. A mano."))
        return
    if escribir(ruta, nombre, s.replace(antes, despues), etiqueta):
        informe.append(("APLICADO", nombre, etiqueta))

# ============================================================ PARCHE 7 =======
sustituir("17_varianzas_M4.py",
          "N_jk=N_jk, idx_sub=sub,",
          "N_jk=N_jk, idx_sub_phi=sub,",
          "parche 7: idx_sub -> idx_sub_phi",
          marca_ya="idx_sub_phi=sub")

# ============================================================ PARCHE 4 =======
sustituir("estilo_M4.py",
"""    def _interior(an):
        y = float(np.median(an[:, 1]))
        x0, y0 = an[:-1, 0], an[:-1, 1]""",
"""    def _interior(an):
        # Barrer ENTRE dos latitudes distintas, nunca sobre un vertice: si la
        # horizontal pasa por un vertice el cruce se cuenta una sola vez y la
        # lista de cortes queda impar.
        ys = np.unique(an[:, 1])
        if ys.size < 2:
            return an.mean(axis=0)
        k2 = ys.size // 2
        y = float((ys[k2 - 1] + ys[k2]) / 2.0)
        x0, y0 = an[:-1, 0], an[:-1, 1]""",
          "parche 4: _interior barre entre vertices",
          marca_ya="ys = np.unique(an[:, 1])")

# ============================================================ PARCHE 5 =======
sustituir("estilo_M4.py",
"""    dir_cache = dir_cache or DIR_CACHE
    D = {}
    for f in ("prep_M4.npz", "nivel_M4.npz"):
        ruta = os.path.join(dir_cache, f)
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"Falta {ruta}. Ejecuta 16_preparacion_M4.py")
        with np.load(ruta, allow_pickle=True) as z:
            for k in z.files:
                D[k] = z[k]
    if con_varianzas:
        ruta = os.path.join(dir_cache, "varianzas_M4.npz")
        if os.path.exists(ruta):
            with np.load(ruta, allow_pickle=True) as z:
                for k in z.files:
                    D[k] = z[k]
        else:""",
"""    dir_cache = dir_cache or DIR_CACHE
    D, origen = {}, {}

    def _absorber(ruta, f):
        with np.load(ruta, allow_pickle=True) as z:
            for k in z.files:
                if k in origen:
                    raise KeyError(
                        f"clave '{k}' repetida: la produce {origen[k]} y tambien "
                        f"{f}. Al fusionar, la segunda pisa a la primera y el "
                        f"consumidor recibe el objeto equivocado. Renombrala en "
                        f"el modulo que la genera.")
                D[k] = z[k]
                origen[k] = f

    for f in ("prep_M4.npz", "nivel_M4.npz"):
        ruta = os.path.join(dir_cache, f)
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"Falta {ruta}. Ejecuta 16_preparacion_M4.py")
        _absorber(ruta, f)
    if con_varianzas:
        ruta = os.path.join(dir_cache, "varianzas_M4.npz")
        if os.path.exists(ruta):
            _absorber(ruta, "varianzas_M4.npz")
        else:""",
          "parche 5: deteccion de claves repetidas",
          marca_ya="def _absorber(ruta, f):")

# ============================================================ PARCHE 9 =======
TABLAS = '''

# ======================= UNIDADES REALES DE LAS COVARIABLES ==================
# (variable, multiplicador sobre la unidad ORIGINAL, etiqueta de la unidad).
# Las usan el modulo 20 (figuras fig09 y fig10) y el 21 (verificacion). Estaban
# escritas dos veces y se desincronizaron: el 20 ya decia "+10 billones" cuando
# el 21 seguia diciendo "+10 mil millones".
#
# Comprobado contra estandar_params.csv:
#   PROP_NBI              media 13.53  -> puntos porcentuales -> multiplicador 10
#   PROP_URBANA           media  0.774 -> proporcion          -> 0.10
#   PROP_HOMICIDIOS_URB.  media  0.706 -> proporcion          -> 0.10
#   PIB_CONSTANTE         media 76 382 -> miles de millones (como publica el
#                         DANE), de modo que 10 000 unidades son DIEZ BILLONES
#   TASA_VICTIMAS         media  8 257 por cien mil: con 1 000 el efecto queda
#                         en 0.024 sd y su intervalo se ve pegado a cero; con
#                         10 000 son 10 pp de la poblacion y 0.24 sd
MULT_MUNI = [
    ("TASA_HOMICIDIOS",           10,        "+10 homicidios por 100 mil hab."),
    ("TASA_VICTIMAS",             10000,     "+10 pp de la población como víctimas"),
    ("PROP_HOMICIDIOS_URBANO",    0.10,      "+10 pp de homicidios urbanos"),
    ("PROP_NBI",                  10,        "+10 pp de necesidades básicas insatisfechas"),
    ("COMP_INASISTENCIA_ESCOLAR", 1,         "+1 punto de inasistencia escolar"),
    ("PROP_URBANA",               0.10,      "+10 pp de población urbana"),
    ("SGP_EDUC_PC",               100000,    "+100 mil pesos de SGP educativo per cápita"),
    ("TASA_MATRICULA_ES",         10,        "+10 puntos de matrícula en educación superior"),
    ("IES_CON_OFERTA",            10,        "+10 instituciones con oferta"),
    ("LOG_POBLACION",             np.log(2), "duplicar la población"),
]
MULT_DEPTO = [
    ("IPM",               10,    "+10 puntos del índice de pobreza multidimensional"),
    ("TASA_DESEMPLEO",    1,     "+1 pp de desempleo"),
    ("TASA_SUBOCUPACION", 1,     "+1 pp de subocupación"),
    ("CREC_PIB_REAL",     1,     "+1 pp de crecimiento del PIB"),
    ("PIB_CONSTANTE",     10000, "+10 billones de PIB departamental"),
]
'''
_r, _s = leer("estilo_M4.py")
if _s is not None:
    if "MULT_MUNI" in _s:
        informe.append(("YA ESTABA", "estilo_M4.py", "parche 9: tablas de unidades"))
    elif escribir(_r, "estilo_M4.py", _s.rstrip("\n") + "\n" + TABLAS,
                  "parche 9: tablas de unidades"):
        informe.append(("APLICADO", "estilo_M4.py", "parche 9: tablas de unidades"))

# =========================================================== PARCHE 10 =======
_r, _s = leer("20_covariables_M4.py")
if _s is not None:
    if "MUNI = E.MULT_MUNI" in _s:
        informe.append(("YA ESTABA", "20_covariables_M4.py",
                        "parche 10: usar las tablas de estilo_M4"))
    else:
        ini = _s.find('MUNI = [("TASA_HOMICIDIOS"')
        fin = _s.find("def contexto(")
        if ini == -1 or fin == -1 or fin < ini:
            informe.append(("NO ENCONTRADO", "20_covariables_M4.py",
                            "parche 10: no localizo el bloque MUNI/DEPTO. A mano."))
        else:
            nuevo = (_s[:ini] +
                     "# Las unidades reales viven en estilo_M4, compartidas con el "
                     "modulo 21.\nMUNI = E.MULT_MUNI\nDEPTO = E.MULT_DEPTO\n\n\n" +
                     _s[fin:])
            if escribir(_r, "20_covariables_M4.py", nuevo,
                        "parche 10: usar las tablas de estilo_M4"):
                informe.append(("APLICADO", "20_covariables_M4.py",
                                "parche 10: usar las tablas de estilo_M4"))

# ============================================================== INFORME ======
print("=" * 78)
print("APLICACION DE PARCHES")
print("=" * 78)
ancho = max((len(n) for _, n, _ in informe), default=10)
for estado, nombre, detalle in informe:
    print(f"  [{estado:<13}] {nombre:<{ancho}}  {detalle}")

malos = [i for i in informe if i[0] in ("ABORTADO", "NO ENCONTRADO", "AMBIGUO")]
print()
if malos:
    print(f"  {len(malos)} parche(s) sin aplicar: revisar arriba y hacerlo a mano.")
else:
    print("  Todos los parches quedaron aplicados o ya lo estaban.")
    print("\n  SIGUIENTE PASO: reejecutar el cuaderno completo. El 17 tiene que")
    print("  regenerar varianzas_M4.npz con la clave idx_sub_phi; si no, el 21")
    print("  seguira reportando la colision porque el npz viejo sigue en cache_M4.")
print("\n  Respaldos: cada archivo tocado tiene su .bak al lado.")
