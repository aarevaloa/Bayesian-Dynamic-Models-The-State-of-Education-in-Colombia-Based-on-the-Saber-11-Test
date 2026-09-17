#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

LOG2PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# 1. Metadatos y cadenas
# ---------------------------------------------------------------------------
def cargar_metadatos(dir_mcmc: str) -> dict:
    ruta = os.path.join(dir_mcmc, "metadatos.json")
    if not os.path.isfile(ruta):
        raise FileNotFoundError(f"No se encuentra {ruta}")
    with open(ruta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    for clave in ("p", "Tt", "m", "n_k", "max_nk", "theta_labels",
                  "depto_ids", "tiempo_ids"):
        if clave not in meta:
            raise KeyError(f"metadatos.json no contiene '{clave}'")
    return meta


def abrir_cadenas(dir_mcmc: str, meta: dict):
    """Abre theta.bin, s.bin y v.bin como memmaps con forma validada."""
    p, Tt, m, mx = meta["p"], meta["Tt"], meta["m"], meta["max_nk"]
    especificacion = {
        "theta.bin": (p * (Tt + 1), (Tt + 1, p)),
        "s.bin":     (Tt * m,       (Tt, m)),
        "v.bin":     (m * mx,       (m, mx)),
    }
    salida, tamanos = {}, {}
    for nombre, (por_muestra, forma) in especificacion.items():
        ruta = os.path.join(dir_mcmc, nombre)
        if not os.path.isfile(ruta):
            raise FileNotFoundError(f"No se encuentra {ruta}")
        n_dobles = os.path.getsize(ruta) // 8
        if n_dobles % por_muestra != 0:
            raise ValueError(
                f"{nombre}: {n_dobles} dobles no es multiplo de {por_muestra}. "
                "Revise p, Tt, m o max_nk en metadatos.json.")
        n_save = n_dobles // por_muestra
        tamanos[nombre] = n_save
        salida[nombre] = np.memmap(ruta, dtype="<f8", mode="r").reshape(
            (n_save,) + forma)
    if len(set(tamanos.values())) != 1:
        raise ValueError(f"Numero de muestras inconsistente entre archivos: {tamanos}")
    n_save = next(iter(tamanos.values()))
    return salida["theta.bin"], salida["s.bin"], salida["v.bin"], n_save


# ---------------------------------------------------------------------------
# 2. Datos
# ---------------------------------------------------------------------------
COLUMNAS = {
    "y": "y",                  # puntaje observado
    "depto": "departamento",   # codigo DANE (5, 8, 11, ...) o indice 1..m
    "municipio": "municipio",  # indice del municipio DENTRO del departamento
    "periodo": "periodo",      # codigo 20171... o indice 1..Tt
}


def _normalizar_indice(valores: np.ndarray, codigos: list, n_niveles: int,
                       etiqueta: str) -> np.ndarray:
    """Devuelve indices base 0 a partir de codigos, base 1 o base 0."""
    valores = np.asarray(valores)
    unicos = np.unique(valores)
    conjunto_codigos = set(int(c) for c in codigos)
    if set(int(u) for u in unicos).issubset(conjunto_codigos):
        mapa = {int(c): i for i, c in enumerate(codigos)}
        return np.array([mapa[int(v)] for v in valores], dtype=np.int32)
    if unicos.min() == 1 and unicos.max() <= n_niveles:
        return (valores - 1).astype(np.int32)
    if unicos.min() == 0 and unicos.max() <= n_niveles - 1:
        return valores.astype(np.int32)
    raise ValueError(
        f"No se pudo interpretar el indice de {etiqueta}: rango observado "
        f"[{unicos.min()}, {unicos.max()}] con {n_niveles} niveles esperados.")


def cargar_datos_bin(dir_datos: str, meta: dict):
    """Lee la exportacion binaria producida por exportar_datos_M3.R."""
    ruta_meta = os.path.join(dir_datos, "datos_meta.json")
    with open(ruta_meta, "r", encoding="utf-8") as f:
        dm = json.load(f)
    N, p = int(dm["N"]), int(dm["p"])
    if p != meta["p"]:
        raise ValueError(f"p en los datos ({p}) != p en las cadenas ({meta['p']})")

    y = np.fromfile(os.path.join(dir_datos, "datos_y.bin"), dtype="<f8")
    X = np.fromfile(os.path.join(dir_datos, "datos_X.bin"), dtype="<f8")
    if y.size != N:
        raise ValueError(f"datos_y.bin tiene {y.size} valores, se esperaban {N}")
    if X.size != N * p:
        raise ValueError(f"datos_X.bin tiene {X.size} valores, se esperaban {N*p}")
    X = X.reshape((p, N)).T if dm.get("orden_X", "columnas") == "columnas" \
        else X.reshape((N, p))
    X = np.ascontiguousarray(X)

    k = np.fromfile(os.path.join(dir_datos, "datos_idx_depto.bin"), dtype="<i4")
    j = np.fromfile(os.path.join(dir_datos, "datos_idx_muni.bin"), dtype="<i4")
    t = np.fromfile(os.path.join(dir_datos, "datos_idx_tiempo.bin"), dtype="<i4")

    columnas = dm.get("columnas_X")
    if columnas is not None:
        X = _reordenar_columnas(X, columnas, meta["theta_labels"])
    return y, X, k, j, t


def cargar_datos_tabla(ruta: str, meta: dict):
    """Lee una tabla unica (csv/parquet/feather/npz/rds) con y, X e indices."""
    ext = os.path.splitext(ruta)[1].lower()
    if ext == ".npz":
        z = np.load(ruta, allow_pickle=False)
        y = np.asarray(z["y"], dtype=np.float64)
        X = np.asarray(z["X"], dtype=np.float64)
        k = np.asarray(z["depto"]).ravel()
        j = np.asarray(z["municipio"]).ravel()
        t = np.asarray(z["periodo"]).ravel()
        if "columnas_X" in z:
            X = _reordenar_columnas(X, [str(c) for c in z["columnas_X"]],
                                    meta["theta_labels"])
        return y, X, k, j, t

    import pandas as pd
    if ext == ".parquet":
        df = pd.read_parquet(ruta)
    elif ext in (".feather", ".fth"):
        df = pd.read_feather(ruta)
    elif ext in (".csv", ".txt"):
        df = pd.read_csv(ruta)
    elif ext == ".rds":
        import pyreadr
        resultado = pyreadr.read_r(ruta)
        df = next(iter(resultado.values()))
    else:
        raise ValueError(f"Extension no soportada: {ext}")

    faltantes = [c for c in COLUMNAS.values() if c not in df.columns]
    if faltantes:
        raise KeyError(
            f"Faltan columnas {faltantes}. Ajuste el diccionario COLUMNAS "
            f"al encabezado real: {list(df.columns)[:15]} ...")

    y = df[COLUMNAS["y"]].to_numpy(dtype=np.float64)
    k = df[COLUMNAS["depto"]].to_numpy()
    j = df[COLUMNAS["municipio"]].to_numpy()
    t = df[COLUMNAS["periodo"]].to_numpy()

    etiquetas = list(meta["theta_labels"])
    presentes = [c for c in etiquetas if c in df.columns]
    if len(presentes) < len(etiquetas) - 1:  # se tolera que falte el intercepto
        raise KeyError(
            f"La tabla solo contiene {len(presentes)} de las {len(etiquetas)} "
            "covariables de theta_labels. Verifique que sea la matriz de diseno "
            "efectivamente usada por el muestreador.")
    X = np.empty((len(y), len(etiquetas)), dtype=np.float64)
    for col, nombre in enumerate(etiquetas):
        X[:, col] = 1.0 if nombre not in df.columns else \
            df[nombre].to_numpy(dtype=np.float64)
    return y, X, k, j, t


def _reordenar_columnas(X, columnas, etiquetas):
    """Reordena/completa X para que sus columnas sigan theta_labels."""
    columnas = list(columnas)
    if columnas == list(etiquetas):
        return X
    indice = {c: i for i, c in enumerate(columnas)}
    salida = np.empty((X.shape[0], len(etiquetas)), dtype=X.dtype)
    for col, nombre in enumerate(etiquetas):
        if nombre in indice:
            salida[:, col] = X[:, indice[nombre]]
        elif nombre == "intercepto":
            salida[:, col] = 1.0
        else:
            raise KeyError(f"La matriz de diseno no contiene la columna '{nombre}'")
    return salida


# ---------------------------------------------------------------------------
# 3. Validaciones de alineacion
# ---------------------------------------------------------------------------
def validar_alineacion(k, j, t, meta, v_muestra) -> list:
    """Chequeos que detectan una correspondencia datos <-> cadenas incorrecta."""
    avisos = []
    n_k = np.asarray(meta["n_k"], dtype=np.int64)
    m, Tt, mx = meta["m"], meta["Tt"], meta["max_nk"]

    if k.min() < 0 or k.max() >= m:
        raise ValueError(f"Indice de departamento fuera de rango [0, {m-1}]")
    if t.min() < 0 or t.max() >= Tt:
        raise ValueError(f"Indice de periodo fuera de rango [0, {Tt-1}]")
    if j.min() < 0 or j.max() >= mx:
        raise ValueError(f"Indice de municipio fuera de rango [0, {mx-1}]")

    # (a) el municipio debe caber dentro de su departamento
    excede = j >= n_k[k]
    if excede.any():
        malos = np.unique(k[excede])
        raise ValueError(
            "Hay municipios con indice mayor al numero de municipios del "
            f"departamento en los departamentos (base 0) {malos.tolist()}. "
            "La codificacion del indice municipal no coincide con la del "
            "muestreador.")

    # (b) toda celda (k, j) observada debe tener v_jk definido (no NaN)
    celdas = np.unique(k.astype(np.int64) * mx + j.astype(np.int64))
    valores_v = np.asarray(v_muestra).ravel()[celdas]
    n_nan = int(np.isnan(valores_v).sum())
    if n_nan:
        raise ValueError(
            f"{n_nan} celdas (municipio, departamento) presentes en los datos "
            "tienen v = NaN en las cadenas: los indices no estan alineados.")

    # (c) cobertura de municipios
    n_celdas = celdas.size
    if n_celdas != int(n_k.sum()):
        avisos.append(
            f"Se observan {n_celdas:,} municipios y las cadenas declaran "
            f"{int(n_k.sum()):,}.")
    # (d) todos los periodos representados
    if np.unique(t).size != Tt:
        avisos.append(f"Solo {np.unique(t).size} de {Tt} periodos tienen datos.")
    return avisos


# ---------------------------------------------------------------------------
# 4. Nucleo de calculo
# ---------------------------------------------------------------------------
def preparar_celdas(y, X, k, j, t, float32=False):
    """Ordena por (t, k, j) y devuelve los limites de celda y de periodo."""
    orden = np.lexsort((j, k, t))          # clave primaria: t; luego k; luego j
    y = np.ascontiguousarray(y[orden])
    X = np.ascontiguousarray(X[orden].astype(np.float32 if float32 else np.float64))
    k, j, t = k[orden], j[orden], t[orden]

    cambio = np.empty(y.size, dtype=bool)
    cambio[0] = True
    np.logical_or((t[1:] != t[:-1]), (k[1:] != k[:-1]), out=cambio[1:])
    np.logical_or(cambio[1:], (j[1:] != j[:-1]), out=cambio[1:])
    inicio_celda = np.flatnonzero(cambio).astype(np.int64)
    g_t = t[inicio_celda].astype(np.int64)
    g_k = k[inicio_celda].astype(np.int64)
    g_j = j[inicio_celda].astype(np.int64)
    g_n = np.diff(np.append(inicio_celda, y.size)).astype(np.float64)

    # limites de los bloques de periodo (para el producto X @ theta_t)
    cambio_t = np.empty(y.size, dtype=bool)
    cambio_t[0] = True
    np.not_equal(t[1:], t[:-1], out=cambio_t[1:])
    inicio_t = np.flatnonzero(cambio_t).astype(np.int64)
    bloques = [(int(t[a]), int(a), int(b)) for a, b in
               zip(inicio_t, np.append(inicio_t[1:], y.size))]
    return y, X, inicio_celda, g_t, g_k, g_j, g_n, bloques


def logverosimilitud(y, X, bloques, inicio_celda, g_n, precision_celda,
                     theta, buffer):
    """log p(y | theta, s, v) total, con sigma2 constante dentro de cada celda."""
    for t_idx, a, b in bloques:
        # theta[t_idx + 1]: t=0 en las cadenas es el estado inicial theta_0
        np.dot(X[a:b], theta[t_idx + 1].astype(X.dtype), out=buffer[a:b])
    np.subtract(y, buffer, out=buffer)
    np.multiply(buffer, buffer, out=buffer)
    suma_r2 = np.add.reduceat(buffer, inicio_celda)
    N = float(y.size)
    return -0.5 * (N * LOG2PI
                   - float(np.dot(g_n, np.log(precision_celda)))
                   + float(np.dot(suma_r2, precision_celda)))


# ---------------------------------------------------------------------------
# 5. Programa principal
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="DIC del Modelo 3 (Normal dinamico con LASSO bayesiano)")
    ap.add_argument("--mcmc", required=True,
                    help="Carpeta con theta.bin, s.bin, v.bin y metadatos.json")
    ap.add_argument("--datos-bin", default=None,
                    help="Carpeta con la exportacion de exportar_datos_M3.R")
    ap.add_argument("--datos", default=None,
                    help="Tabla unica (.csv/.parquet/.feather/.npz/.rds)")
    ap.add_argument("--salida", default=".", help="Carpeta de salida")
    ap.add_argument("--max-draws", type=int, default=0,
                    help="Submuestrea las muestras posteriores (0 = todas)")
    ap.add_argument("--float32", action="store_true",
                    help="Guarda X en simple precision (mitad de RAM)")
    ap.add_argument("--verificar-theta", default=None,
                    help="Ruta a resumen_theta_full_M3.csv para validar la lectura")
    args = ap.parse_args(argv)

    if (args.datos_bin is None) == (args.datos is None):
        ap.error("Indique exactamente una fuente de datos: --datos-bin o --datos")
    os.makedirs(args.salida, exist_ok=True)
    t0 = time.time()

    # --- cadenas -----------------------------------------------------------
    meta = cargar_metadatos(args.mcmc)
    theta_mm, s_mm, v_mm, n_save = abrir_cadenas(args.mcmc, meta)
    p, Tt, m, mx = meta["p"], meta["Tt"], meta["m"], meta["max_nk"]
    print(f"Cadenas: n_save = {n_save:,} | p = {p} | Tt = {Tt} | m = {m}")

    if args.verificar_theta:
        import pandas as pd
        ref = pd.read_csv(args.verificar_theta)
        etiquetas = list(meta["theta_labels"])
        media = np.asarray(theta_mm).mean(axis=0)
        error = max(abs(media[int(r.t_idx), etiquetas.index(r.parametro)] - r.media)
                    for r in ref.itertuples())
        print(f"Verificacion de theta contra el CSV: error maximo = {error:.3e}")
        if error > 1e-6:
            print("  ADVERTENCIA: la lectura de theta.bin no reproduce el resumen.")

    # --- datos -------------------------------------------------------------
    if args.datos_bin:
        y, X, k_bruto, j_bruto, t_bruto = cargar_datos_bin(args.datos_bin, meta)
    else:
        y, X, k_bruto, j_bruto, t_bruto = cargar_datos_tabla(args.datos, meta)

    k = _normalizar_indice(k_bruto, meta["depto_ids"], m, "departamento")
    t = _normalizar_indice(t_bruto, meta["tiempo_ids"], Tt, "periodo")
    j_bruto = np.asarray(j_bruto)
    j = (j_bruto - 1 if j_bruto.min() == 1 else j_bruto).astype(np.int32)

    N = y.size
    if X.shape != (N, p):
        raise ValueError(f"X tiene forma {X.shape}, se esperaba ({N}, {p})")
    print(f"Datos:   N = {N:,} observaciones | matriz de diseno {X.shape}")

    for aviso in validar_alineacion(k, j, t, meta, v_mm[0]):
        print(f"  Aviso: {aviso}")

    y, X, inicio_celda, g_t, g_k, g_j, g_n, bloques = preparar_celdas(
        y, X, k, j, t, float32=args.float32)
    n_celdas = inicio_celda.size
    print(f"Celdas (municipio x periodo) con datos: {n_celdas:,}")

    # --- recorrido por muestras -------------------------------------------
    indices = np.arange(n_save)
    if args.max_draws and args.max_draws < n_save:
        indices = np.linspace(0, n_save - 1, args.max_draws).astype(int)
        print(f"Submuestreo: {indices.size:,} de {n_save:,} muestras posteriores")
    S = indices.size

    buffer = np.empty(N, dtype=X.dtype)
    ll = np.empty(S, dtype=np.float64)
    suma_theta = np.zeros((Tt + 1, p), dtype=np.float64)
    suma_s = np.zeros((Tt, m), dtype=np.float64)
    suma_v = np.zeros((m, mx), dtype=np.float64)
    suma_prec = np.zeros(n_celdas, dtype=np.float64)     # E[1/sigma2]
    suma_sigma2 = np.zeros(n_celdas, dtype=np.float64)   # E[sigma2]

    print(f"\nRecorriendo {S:,} muestras posteriores ...")
    for pos, idx in enumerate(indices):
        theta_i = np.asarray(theta_mm[idx], dtype=np.float64)
        s_i = np.asarray(s_mm[idx], dtype=np.float64)
        v_i = np.asarray(v_mm[idx], dtype=np.float64)

        precision = v_i[g_k, g_j] / s_i[g_t, g_k]        # 1 / sigma2 por celda
        ll[pos] = logverosimilitud(y, X, bloques, inicio_celda, g_n,
                                   precision, theta_i, buffer)

        suma_theta += theta_i
        suma_s += s_i
        suma_v += np.nan_to_num(v_i, nan=0.0)
        suma_prec += precision
        suma_sigma2 += 1.0 / precision

        if (pos + 1) % max(1, S // 20) == 0:
            transcurrido = time.time() - t0
            print(f"  {pos+1:>6,}/{S:,}  ({100*(pos+1)/S:5.1f}%)  "
                  f"{transcurrido:7.1f} s", flush=True)

    # --- resumenes posteriores --------------------------------------------
    theta_barra = suma_theta / S
    s_barra = suma_s / S
    v_barra = suma_v / S
    prec_media = suma_prec / S          # E[1/sigma2]
    sigma2_media = suma_sigma2 / S      # E[sigma2]

    D_barra = -2.0 * ll.mean()
    p_V = 2.0 * ll.var(ddof=1)
    variantes = {
        "media_parametros": v_barra[g_k, g_j] / s_barra[g_t, g_k],
        "media_sigma2": 1.0 / sigma2_media,
        "media_precision": prec_media,
    }
    resultados = {}
    for nombre, precision in variantes.items():
        ll_sombrero = logverosimilitud(y, X, bloques, inicio_celda, g_n,
                                       precision, theta_barra, buffer)
        D_sombrero = -2.0 * ll_sombrero
        p_dic = D_barra - D_sombrero
        resultados[nombre] = {
            "D_sombrero": D_sombrero,
            "p_DIC": p_dic,
            "DIC": D_barra + p_dic,
        }

    principal = resultados["media_parametros"]

    # --- informe -----------------------------------------------------------
    print("\n" + "=" * 74)
    print("DIC DEL MODELO 3 (Normal jerarquico dinamico con LASSO bayesiano)")
    print("=" * 74)
    print(f"N = {N:,} observaciones | S = {S:,} muestras posteriores")
    print(f"D_barra    (devianza media)          = {D_barra:>18,.2f}")
    print(f"D_sombrero (devianza en la media)    = {principal['D_sombrero']:>18,.2f}")
    print(f"p_DIC      (parametros efectivos)    = {principal['p_DIC']:>18,.2f}")
    print(f"DIC                                  = {principal['DIC']:>18,.2f}")
    print("-" * 74)
    print(f"p_V  (Gelman, 2 Var[log p(y|.)])     = {p_V:>18,.2f}")
    print(f"DIC_V = D_barra + p_V                = {D_barra + p_V:>18,.2f}")
    print("-" * 74)
    print("Sensibilidad al plug-in de la escala:")
    for nombre, r in resultados.items():
        print(f"  {nombre:<18} DIC = {r['DIC']:>16,.2f}   p_DIC = {r['p_DIC']:>10,.2f}")
    print("-" * 74)
    print(f"sigma2 posterior medio: mediana = {np.median(sigma2_media):,.1f}  "
          f"[min {sigma2_media.min():,.1f}, max {sigma2_media.max():,.1f}]")
    print("Referencia: WAIC del M3 = 5,198,902.56 (p_WAIC = 1,945.85). Se espera "
          "DIC del mismo orden\ny p_DIC algo mayor que p_WAIC, como ocurre en M1, "
          "M2 y M4.")
    print("=" * 74)

    # --- persistencia ------------------------------------------------------
    ruta_csv = os.path.join(args.salida, "ajuste_dic_M3.csv")
    with open(ruta_csv, "w", encoding="utf-8") as f:
        f.write("variante,DIC,p_DIC,D_barra,D_sombrero,p_V,DIC_V,N,n_save\n")
        for nombre, r in resultados.items():
            f.write(f"{nombre},{r['DIC']:.6f},{r['p_DIC']:.6f},{D_barra:.6f},"
                    f"{r['D_sombrero']:.6f},{p_V:.6f},{D_barra + p_V:.6f},"
                    f"{N},{S}\n")

    np.save(os.path.join(args.salida, "loglik_total_por_muestra_M3.npy"), ll)

    resumen = {
        "modelo": "M3",
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "N": int(N), "n_save_usadas": int(S), "n_save_disponibles": int(n_save),
        "D_barra": D_barra, "p_V": p_V, "DIC_V": D_barra + p_V,
        "variantes": resultados,
        "definicion": "DIC = D_sombrero + 2 p_DIC; p_DIC = D_barra - D_sombrero "
                      "(Spiegelhalter et al., 2002)",
        "verosimilitud": "y_i ~ N(x_i' theta_{t(i)}, s_{k,t} / v_{j,k})",
        "segundos": time.time() - t0,
    }
    with open(os.path.join(args.salida, "dic_M3.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    print(f"\nArchivos escritos en {os.path.abspath(args.salida)}:")
    print("  ajuste_dic_M3.csv, dic_M3.json, loglik_total_por_muestra_M3.npy")
    print(f"Tiempo total: {time.time() - t0:,.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
