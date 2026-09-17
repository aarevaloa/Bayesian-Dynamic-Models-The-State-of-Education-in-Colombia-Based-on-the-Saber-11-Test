# -*- coding: utf-8 -*-
"""
@author: aarevaloa

Construcción del panel jerárquico final: individuo × municipio × departamento × año
para el modelo dinámico bayesiano con regularización LASSO.

Combina:
  - ICFES_20172_20242_COMPLETO.csv  (nivel individual)
  - panel_municipal.csv             (nivel municipal-año)
  - panel_departamental.csv         (nivel departamental-año)

Resultado: panel_jerarquico_completo.csv
"""
# =============================================================================
# 1. MÓDULOS Y RUTAS
# =============================================================================
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

RUTA_BASE = Path(
    r'C:\Users\USUARIO\Desktop\Estadística'
    r'\Maestría en Estadística\Tesis\Escritorio'
)
RUTA_DATOS = RUTA_BASE / 'Datos_procesados' / 'Python'

ARCHIVO_ICFES = RUTA_DATOS / 'ICFES_20172_20242_COMPLETO.csv'
ARCHIVO_MUN = RUTA_DATOS / 'panel_municipal.csv'
ARCHIVO_DEPTO = RUTA_DATOS / 'panel_departamental.csv'

ARCHIVO_SALIDA = RUTA_DATOS / 'panel_jerarquico_completo.csv'
ARCHIVO_SALIDA_PARQUET = RUTA_DATOS / 'panel_jerarquico_completo.parquet'

# =============================================================================
# 2. FUNCIONES AUXILIARES
# =============================================================================

def estandarizar_codigo(serie, digitos):
    """Estandariza un código DANE: quita .0, espacios, rellena ceros."""
    return (
        serie.astype(str)
        .str.replace('.0', '', regex=False)
        .str.strip()
        .str.zfill(digitos)
    )

# =============================================================================
# 3. CARGAR ARCHIVOS
# =============================================================================
print("=" * 70)
print("CARGANDO ARCHIVOS")
print("=" * 70)

# ── 3.1 ICFES (nivel individual)
print("\n── ICFES (individual) ──")
icfes = pd.read_csv(ARCHIVO_ICFES, encoding='utf-8-sig', low_memory=False)
print(f"  ✓ {icfes.shape[0]:,} estudiantes × {icfes.shape[1]} variables")

# Renombrar columnas clave para que coincidan con los paneles
icfes = icfes.rename(columns={
    'estu_cod_reside_depto': 'COD_DEPTO',
    'estu_cod_reside_mcpio': 'COD_MUNI',
})

icfes['ANIO'] = icfes['periodo'].astype(str).str[:4].astype(int)

icfes['COD_DEPTO'] = estandarizar_codigo(icfes['COD_DEPTO'], 2)
icfes['COD_MUNI'] = estandarizar_codigo(icfes['COD_MUNI'], 5)

print(f"  Años en ICFES: {sorted(icfes['ANIO'].unique())}")
print(f"  Municipios únicos: {icfes['COD_MUNI'].nunique():,}")
print(f"  Departamentos únicos: {icfes['COD_DEPTO'].nunique()}")

# ── 3.2 Panel municipal ──
print("\n── Panel municipal ──")
panel_mun = pd.read_csv(ARCHIVO_MUN, encoding='utf-8-sig')
print(f"  ✓ {panel_mun.shape[0]:,} obs × {panel_mun.shape[1]} variables")

panel_mun['COD_DEPTO'] = estandarizar_codigo(panel_mun['COD_DEPTO'], 2)
panel_mun['COD_MUNI'] = estandarizar_codigo(panel_mun['COD_MUNI'], 5)
panel_mun['ANIO'] = pd.to_numeric(panel_mun['ANIO'], errors='coerce').astype('Int64')

print(f"  Variables: {[c for c in panel_mun.columns if c not in ['COD_DEPTO', 'COD_MUNI', 'ANIO']]}")

# ── 3.3 Panel departamental ──
print("\n── Panel departamental ──")
panel_depto = pd.read_csv(ARCHIVO_DEPTO, encoding='utf-8-sig')
print(f"  ✓ {panel_depto.shape[0]:,} obs × {panel_depto.shape[1]} variables")

panel_depto['COD_DEPTO'] = estandarizar_codigo(panel_depto['COD_DEPTO'], 2)
panel_depto['ANIO'] = pd.to_numeric(panel_depto['ANIO'], errors='coerce').astype('Int64')

print(f"  Variables: {[c for c in panel_depto.columns if c not in ['COD_DEPTO', 'ANIO']]}")

# =============================================================================
# 4. EVITAR COLISIÓN DE NOMBRES
# =============================================================================

cols_mun_merge = [c for c in panel_mun.columns if c != 'COD_DEPTO']
panel_mun_merge = panel_mun[cols_mun_merge]

# =============================================================================
# 5. MERGE JERÁRQUICO
# =============================================================================
print("\n" + "=" * 70)
print("CONSTRUYENDO PANEL JERÁRQUICO")
print("=" * 70)

print("\n  Uniendo variables municipales y departamentales (por COD_MUNI + ANIO)...")
n_antes = len(icfes)

panel_jerarquico = icfes.merge(
    panel_mun_merge,
    on=['COD_MUNI', 'ANIO'],
    how='left'
)

n_despues = len(panel_jerarquico)
print(f"    Filas antes: {n_antes:,}")
print(f"    Filas después: {n_despues:,}")
assert n_antes == n_despues, "El merge alteró el número de filas"

for var in ['TASA_HOMICIDIOS', 'PIB_CONSTANTE', 'IPM', 'TASA_DESEMPLEO']:
    if var in panel_jerarquico.columns:
        n_na = panel_jerarquico[var].isna().sum()
        pct_na = n_na / n_despues * 100
        print(f"    {var:25s}  NA: {n_na:>10,} ({pct_na:5.2f}%)")
        
# =============================================================================
# 6. RESUMEN
# =============================================================================
print("\n" + "=" * 70)
print("RESUMEN DEL PANEL JERÁRQUICO")
print("=" * 70)

print(f"\n  Dimensión final: {panel_jerarquico.shape[0]:,} filas × {panel_jerarquico.shape[1]} columnas")
print(f"  Estudiantes: {len(panel_jerarquico):,}")
print(f"  Municipios: {panel_jerarquico['COD_MUNI'].nunique():,}")
print(f"  Departamentos: {panel_jerarquico['COD_DEPTO'].nunique()}")
print(f"  Años: {panel_jerarquico['ANIO'].min()} - {panel_jerarquico['ANIO'].max()}")

# Distribución de NA en variables contextuales
print(f"\n  Missings en variables contextuales:")
vars_contexto = [c for c in panel_jerarquico.columns 
                 if c in set(panel_mun.columns) | set(panel_depto.columns)
                 and c not in ['COD_DEPTO', 'COD_MUNI', 'ANIO']]
vars_contexto = list(dict.fromkeys(vars_contexto))  # quitar duplicados preservando orden
for var in vars_contexto:
    if var in panel_jerarquico.columns:
        n_na = panel_jerarquico[var].isna().sum()
        pct_na = n_na / len(panel_jerarquico) * 100
        print(f"    {var:35s}  NA: {n_na:>10,} ({pct_na:5.2f}%)")

# =============================================================================
# 7. EXPORTAR
# =============================================================================
print("\n" + "=" * 70)
print("EXPORTANDO PANEL")
print("=" * 70)

panel_jerarquico.to_csv(ARCHIVO_SALIDA, index=False, encoding='utf-8-sig')
print(f"\n  ✓ CSV: {ARCHIVO_SALIDA}")

try:
    panel_jerarquico.to_parquet(ARCHIVO_SALIDA_PARQUET, index=False)
    print(f"  ✓ Parquet: {ARCHIVO_SALIDA_PARQUET}")
except Exception as e:
    print(f"  ⚠ No se pudo guardar Parquet: {e}")

print("\n" + "=" * 70)
print("PROCESO COMPLETADO")
print("=" * 70)