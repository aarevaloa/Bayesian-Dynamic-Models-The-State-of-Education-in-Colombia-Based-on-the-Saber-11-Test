# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# @  MODELO 4 - GEOGRAFIA: MGN 2025 -> GEOJSON MUNICIPAL Y DEPARTAMENTAL
# _____________________________________________________________________________

import json, os, shutil, struct, sys, time, zipfile
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import estilo_M4 as E

# ============================== PARAMETROS ===================================
RUTA_ZIP   = os.path.join(E.RAIZ, "MGN2025_MPIO_GRAFICO.zip")
BASE_SHP   = "MGN_ADM_MPIO_GRAFICO"
TOL_MUNI   = 0.005      # ~500 m; 1 122 poligonos
TOL_DEPTO  = 0.002      # ~200 m; solo 158 anillos, se puede afinar mas
MIN_PTS    = 4
AREA_MIN   = 1e-5       # grados^2 (~0.12 km2): descarta astillas de la disolucion
REGENERAR_DEPTO = True  # si ya existe uno, se respalda como .bak antes

SAL_MUNI  = E.RUTA_GEOJSON_MUNI
SAL_DEPTO = E.RUTA_GEOJSON

t0 = time.time()
def paso(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

# ====================== LECTOR MINIMO DE SHAPEFILE ===========================
def leer_shp(datos):
    """Geometria de un .shp de poligonos (tipo 5) o polilineas (3).
    Devuelve [(puntos (n,2) float64, indices de parte), ...]."""
    if struct.unpack(">i", datos[:4])[0] != 9994:
        raise ValueError("no parece un .shp valido")
    formas, pos, fin = [], 100, len(datos)
    while pos < fin:
        _, largo = struct.unpack(">ii", datos[pos:pos + 8])
        ini, pos = pos + 8, pos + 8 + largo * 2
        tipo = struct.unpack("<i", datos[ini:ini + 4])[0]
        if tipo == 0:                                    # forma nula
            formas.append((np.zeros((0, 2)), [0])); continue
        if tipo not in (3, 5):
            raise ValueError(f"tipo de forma {tipo} no soportado")
        n_par, n_pts = struct.unpack("<ii", datos[ini + 36:ini + 44])
        p0 = ini + 44
        partes = np.frombuffer(datos, "<i4", n_par, p0).tolist()
        pts = np.frombuffer(datos, "<f8", n_pts * 2, p0 + n_par * 4).reshape(-1, 2)
        formas.append((pts, partes))
    return formas

def leer_dbf(datos, campos=None):
    """Atributos de un .dbf. `campos` restringe las columnas leidas."""
    n_reg, largo_cab, largo_reg = struct.unpack("<ihh", datos[4:12])
    desc, pos = [], 32
    while datos[pos] != 0x0D:
        cru = datos[pos:pos + 32]
        nom = cru[:11].split(b"\x00")[0].decode("latin-1").strip()
        desc.append((nom, cru[11:12].decode("latin-1"), cru[16]))
        pos += 32
    quiero = set([d[0] for d in desc] if campos is None else campos)
    filas = []
    for i in range(n_reg):
        fila, off = {}, largo_cab + i * largo_reg + 1     # +1: marca de borrado
        for nom, tipo, ancho in desc:
            if nom in quiero:
                cru = datos[off:off + ancho]
                try:
                    val = cru.decode("utf-8").strip()
                except UnicodeDecodeError:
                    val = cru.decode("latin-1").strip()
                fila[nom] = val
            off += ancho
        filas.append(fila)
    return [d[0] for d in desc], filas

# ============================== SIMPLIFICACION ===============================
def douglas_peucker(pts, tol):
    """Version iterativa: evita el limite de recursion en anillos de decenas
    de miles de vertices."""
    n = len(pts)
    if n <= MIN_PTS:
        return pts
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    pila = [(0, n - 1)]
    while pila:
        i, j = pila.pop()
        if j <= i + 1:
            continue
        p0, d = pts[i], pts[j] - pts[i]
        L2 = float(d @ d)
        seg = pts[i + 1:j]
        if L2 == 0.0:
            dist = np.hypot(seg[:, 0] - p0[0], seg[:, 1] - p0[1])
        else:
            dist = np.abs((seg[:, 0] - p0[0]) * d[1] -
                          (seg[:, 1] - p0[1]) * d[0]) / np.sqrt(L2)
        kmax = int(np.argmax(dist))
        if dist[kmax] > tol:
            idx = i + 1 + kmax
            keep[idx] = True
            pila.append((i, idx)); pila.append((idx, j))
    out = pts[keep]
    if len(out) < MIN_PTS:                               # no colapsar islas
        out = np.vstack([pts[::max(1, n // MIN_PTS)], pts[-1:]])
    if not np.allclose(out[0], out[-1]):
        out = np.vstack([out, out[:1]])
    return out

def a_feature(cod, anillos, campo, extra=None):
    coords = [[np.round(a, 5).tolist()] for a in anillos]
    props = {campo: cod}
    if extra:
        props.update(extra)
    return {"type": "Feature", "properties": props,
            "geometry": {"type": "MultiPolygon", "coordinates": coords}}

# ============================== DISOLUCION ===================================
def area_anillo(an):
    """Area por la formula del zapatero, en grados cuadrados."""
    x, y = an[:, 0], an[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def encadenar(aristas):
    """Encadena aristas sueltas (x1,y1,x2,y2 enteras) en anillos cerrados."""
    ady = defaultdict(list)
    lista = aristas.tolist()
    for i, (x1, y1, x2, y2) in enumerate(lista):
        ady[(x1, y1)].append(((x2, y2), i))
        ady[(x2, y2)].append(((x1, y1), i))
    usada = bytearray(len(lista))
    anillos = []
    for i, (x1, y1, x2, y2) in enumerate(lista):
        if usada[i]:
            continue
        usada[i] = 1
        ini, act, an = (x1, y1), (x2, y2), [(x1, y1), (x2, y2)]
        while act != ini:
            sig = next((p for p in ady[act] if not usada[p[1]]), None)
            if sig is None:
                break
            usada[sig[1]] = 1
            act = sig[0]
            an.append(act)
        if len(an) >= MIN_PTS:
            anillos.append(np.asarray(an, dtype=np.float64) / 1e7)
    return anillos

# ============================== LECTURA ======================================
paso(f"leyendo {BASE_SHP} desde el zip (lector propio, sin pyshp)")
with zipfile.ZipFile(RUTA_ZIP) as z:
    formas = leer_shp(z.read(BASE_SHP + ".shp"))
    _, filas = leer_dbf(z.read(BASE_SHP + ".dbf"),
                        campos=["mpio_cdpmp", "mpio_cnmbr", "dpto_ccdgo"])
paso(f"   {len(formas)} municipios, {sum(len(p) for p, _ in formas):,} vertices")

# ====================== GEOJSON MUNICIPAL ====================================
paso(f"simplificando municipios (tol = {TOL_MUNI} grados)")
feats_m, n_ini, n_fin = [], 0, 0
for (pts, partes), reg in zip(formas, filas):
    if len(pts) == 0:
        continue
    lim = list(partes) + [len(pts)]
    anillos = []
    for a, b in zip(lim[:-1], lim[1:]):
        n_ini += b - a
        s = douglas_peucker(pts[a:b], TOL_MUNI)
        n_fin += len(s)
        anillos.append(s)
    feats_m.append(a_feature(reg["mpio_cdpmp"], anillos, "mpio_cdpmp",
                             {"nombre": reg["mpio_cnmbr"]}))
paso(f"   vertices {n_ini:,} -> {n_fin:,}  ({100*n_fin/n_ini:.1f}%)")
with open(SAL_MUNI, "w", encoding="utf-8") as fh:
    json.dump({"type": "FeatureCollection", "features": feats_m}, fh)
paso(f"   {os.path.basename(SAL_MUNI)}: {os.path.getsize(SAL_MUNI)/1e6:.1f} MB")

# ====================== GEOJSON DEPARTAMENTAL ================================
if REGENERAR_DEPTO:
    paso("disolviendo municipios en departamentos (cancelacion de aristas)")
    por_dep = defaultdict(list)
    for (pts, partes), reg in zip(formas, filas):
        if len(pts) == 0:
            continue
        lim = list(partes) + [len(pts)]
        q = np.rint(pts * 1e7).astype(np.int64)          # malla entera de 1e-7 grados
        for a, b in zip(lim[:-1], lim[1:]):
            r = q[a:b]
            if len(r) < 2:
                continue
            ar = np.hstack([r[:-1], r[1:]])
            sw = (ar[:, 0] > ar[:, 2]) | ((ar[:, 0] == ar[:, 2]) & (ar[:, 1] > ar[:, 3]))
            ar[sw] = ar[sw][:, [2, 3, 0, 1]]             # arista canonica
            por_dep[reg["dpto_ccdgo"]].append(ar)

    feats_d, tot, sob, n_astillas = [], 0, 0, 0
    for d in sorted(por_dep):
        Ea = np.vstack(por_dep[d])
        u, c = np.unique(Ea, axis=0, return_counts=True)
        tot += len(Ea); sob += int((c == 1).sum())
        crudos = encadenar(u[c == 1])
        buenos = [a for a in crudos if area_anillo(a) >= AREA_MIN]
        if not buenos:
            buenos = [max(crudos, key=area_anillo)]
        n_astillas += len(crudos) - len(buenos)
        anillos = [douglas_peucker(a, TOL_DEPTO) for a in buenos]
        feats_d.append(a_feature(d, anillos, "DPTO_CCDGO"))
    paso(f"   astillas descartadas por area < {AREA_MIN} grados^2: {n_astillas}")
    paso(f"   aristas {tot:,} -> borde {sob:,} ({100*sob/tot:.1f}%); "
         f"{len(feats_d)} departamentos, "
         f"{sum(len(f['geometry']['coordinates']) for f in feats_d)} anillos")

    if os.path.exists(SAL_DEPTO):
        shutil.copy2(SAL_DEPTO, SAL_DEPTO + ".bak")
        paso(f"   respaldo del anterior en {os.path.basename(SAL_DEPTO)}.bak")
    with open(SAL_DEPTO, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": feats_d}, fh)
    paso(f"   {os.path.basename(SAL_DEPTO)}: {os.path.getsize(SAL_DEPTO)/1e6:.1f} MB")

# ============================== EMPALME ======================================
print("\n" + "=" * 74)
print("EMPALME CON EL MODELO 4")
print("=" * 74)
D = E.cargar_cache()
ids, dep = D["muni_ids"], D["depto_ids"]

if (ids < 0).any():
    print("  muni_ids.csv no cargado en el cache. Reejecutar 16_preparacion_M4.py.")
else:
    cod_mod = np.array([str(int(c)).zfill(5) for c in ids])
    cod_mgn = {f["properties"]["mpio_cdpmp"] for f in feats_m}
    falt = sorted(set(cod_mod) - cod_mgn)
    print(f"  municipios del modelo         : {len(cod_mod):,}")
    print(f"  municipios en el MGN 2025     : {len(cod_mgn):,}")
    print(f"  del modelo SIN geometria      : {len(falt)}"
          f"{'  -> ' + ', '.join(falt[:12]) if falt else ''}")
    print(f"  en el MGN sin datos del modelo: {len(cod_mgn - set(cod_mod))} "
          f"(se dibujan en gris)")
    if falt:
        print("\n  ATENCION: esos municipios quedaran fuera de los mapas. Suelen ser\n"
              "  codigos historicos o territorios en disputa; revisarlos antes de\n"
              "  publicar cualquier mapa municipal.")

    gm = E.cargar_geojson(SAL_MUNI, "mpio_cdpmp", orden=list(cod_mod), clave=str)
    print(f"\n  municipal    : {len(gm.POLIS):,} poligonos, {len(gm.CENTRO):,} centroides")

gd = E.cargar_geojson(SAL_DEPTO, "DPTO_CCDGO", orden=dep, clave=int)
falt_d = sorted(set(int(c) for c in dep) - set(gd.CENTRO))
print(f"  departamental: {len(gd.POLIS):,} poligonos, {len(gd.CENTRO):,} centroides"
      f"{'' if not falt_d else '  SIN GEOMETRIA: ' + str(falt_d)}")
print(f"  extension    : lon [{gd.XL[0]:.2f}, {gd.XL[1]:.2f}]  "
      f"lat [{gd.YL[0]:.2f}, {gd.YL[1]:.2f}]")
paso("LISTO")
