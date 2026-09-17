"""
preparar_metadatos.py — Asegura que exista metadatos.json para el pipeline M3.

"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile


LINEA_R = (
    'jsonlite::write_json(readRDS("{rds}"), "{json}", '
    'auto_unbox = TRUE, digits = 12)'
)


def _campos_ok(d: dict) -> bool:
    """Verifica que el JSON tenga los campos dimensionales indispensables."""
    req = ["p", "Tt", "m", "n_k", "p_E", "p_M", "p_D", "theta_labels"]
    tiene_nsave = ("n_save" in d) or ("save_idx" in d)
    return all(k in d for k in req) and tiene_nsave


def asegurar_metadatos_json(dir_salida: str, verbose: bool = True) -> str:
    """
    Garantiza dir_salida/metadatos.json completo. Devuelve la ruta si lo logra.
    Lanza RuntimeError con instrucciones claras si no puede.

    Estrategia:
      0. Si ya existe metadatos.json válido -> listo.
      1. Convertir metadatos.rds con Rscript.
      2. Convertir metadatos.rds con pyreadr.
      3. Fallar con la línea R exacta.
    """
    ruta_json = os.path.join(dir_salida, "metadatos.json")
    ruta_rds = os.path.join(dir_salida, "metadatos.rds")

    # --- 0. ¿ya existe y es válido? ---
    if os.path.exists(ruta_json):
        try:
            with open(ruta_json, encoding="utf-8") as fh:
                d = json.load(fh)
            if _campos_ok(d):
                if verbose:
                    print(f"✓ metadatos.json ya existe y es válido "
                          f"({os.path.basename(ruta_json)})")
                return ruta_json
            elif verbose:
                print("⚠ metadatos.json existe pero le faltan campos; "
                      "se regenerará desde el .rds")
        except Exception:
            if verbose:
                print("⚠ metadatos.json ilegible; se regenerará desde el .rds")

    if not os.path.exists(ruta_rds):
        raise RuntimeError(
            f"No existe {ruta_rds}.\n"
            "El pipeline necesita metadatos.rds (lo genera 8_gibbs_M3.R) "
            "para construir metadatos.json."
        )

    # --- 1. Rscript ---
    rscript = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if rscript:
        if verbose:
            print(f"Convirtiendo metadatos.rds → metadatos.json con Rscript...")
        try:
            # jsonlite maneja bien la lista nombrada; escribe con auto_unbox
            codigo = (
                'args <- commandArgs(TRUE); '
                'if(!requireNamespace("jsonlite",quietly=TRUE)) '
                'install.packages("jsonlite",repos="https://cloud.r-project.org"); '
                'jsonlite::write_json(readRDS(args[1]), args[2], '
                'auto_unbox=TRUE, digits=12)'
            )
            res = subprocess.run(
                [rscript, "-e", codigo, ruta_rds, ruta_json],
                capture_output=True, text=True, timeout=120)
            if res.returncode == 0 and os.path.exists(ruta_json):
                with open(ruta_json, encoding="utf-8") as fh:
                    d = json.load(fh)
                if _campos_ok(d):
                    if verbose:
                        print("✓ metadatos.json generado con Rscript")
                    return ruta_json
                else:
                    if verbose:
                        print("⚠ Rscript escribió JSON pero faltan campos; "
                              "probando pyreadr")
            else:
                if verbose:
                    print(f"⚠ Rscript no pudo convertir "
                          f"({res.stderr.strip()[:120]}); probando pyreadr")
        except Exception as e:
            if verbose:
                print(f"⚠ Falló Rscript ({type(e).__name__}); probando pyreadr")

    # --- 2. pyreadr ---
    try:
        meta = _leer_rds_pyreadr(ruta_rds)
        if meta is not None and _campos_ok(meta):
            with open(ruta_json, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)
            if verbose:
                print("✓ metadatos.json generado con pyreadr")
            return ruta_json
    except Exception as e:
        if verbose:
            print(f"⚠ pyreadr no disponible o falló ({type(e).__name__})")

    # --- 3. instrucción manual ---
    raise RuntimeError(
        "No pude generar metadatos.json automáticamente.\n"
        "Solución (una sola vez, en tu sesión de R):\n\n"
        f'  {LINEA_R.format(rds=ruta_rds.replace(chr(92), "/"), json=ruta_json.replace(chr(92), "/"))}\n\n'
        "Luego vuelve a correr esta celda. Alternativamente:\n"
        "  pip install pyreadr   (para conversión automática desde Python)"
    )


def _leer_rds_pyreadr(ruta_rds: str):
    """
    Lee metadatos.rds (lista nombrada de R) con pyreadr y lo mapea a dict.
    pyreadr lee bien vectores; las listas nombradas llegan como varios
    data.frames, así que se reconstruye el dict campo por campo.
    """
    import pyreadr  # type: ignore
    res = pyreadr.read_r(ruta_rds)
    # pyreadr no siempre expone listas nombradas; si vino algo, intentar mapear
    if not res:
        return None
    # Caso frecuente: un único objeto None -> DataFrame con columnas nombradas
    # Como la robustez aquí es limitada, se intenta el mapeo directo y si no,
    # se devuelve None para caer al método R.
    posible = None
    for key, val in res.items():
        posible = val
        break
    if posible is None:
        return None
    # Si es un DataFrame de una fila con las columnas del metadato:
    try:
        cols = set(posible.columns)
        if {"p", "Tt", "m", "p_E"}.issubset(cols):
            fila = posible.iloc[0].to_dict()
            # n_k/theta_labels/etc pueden no venir bien; se avisa devolviendo
            # None si faltan listas críticas
            return fila
    except Exception:
        pass
    return None


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    try:
        ruta = asegurar_metadatos_json(d)
        print(f"\nListo: {ruta}")
    except RuntimeError as e:
        print(f"\n{e}")
        sys.exit(1)
