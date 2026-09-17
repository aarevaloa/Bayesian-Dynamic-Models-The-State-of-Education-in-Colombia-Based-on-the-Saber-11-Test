# -*- coding: utf-8 -*-
"""
@author: aarevaloa

Validación del muestreo estratificado
======================================

Verifica que la muestra extraída del panel jerárquico sea correcta en tres
niveles:

  1. Integridad individuo-contexto: cada estudiante tiene las variables
     contextuales que corresponden a su municipio y año.
  2. Invariantes del muestreo estratificado: proporciones por periodo
     preservadas y todos los municipios representados.
  3. Consistencia interna: año-periodo coherente, variables estáticas
     constantes por municipio, sin duplicados.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
RUTA = Path(r'C:\Users\USUARIO\Desktop\Estadística'
            r'\Maestría en Estadística\Tesis\Escritorio'
            r'\Datos_procesados\Python')

ARCHIVO_PANEL = RUTA / 'panel_jerarquico_completo.csv'
ARCHIVO_MUESTRA = RUTA / 'ICFES_MUESTRA_500000_MODELO.csv'
ARCHIVO_PANEL_MUN = RUTA / 'panel_municipal.csv'

# Variables a verificar en la validación 1
VARS_VERIFICAR = ['TASA_HOMICIDIOS', 'PIB_CONSTANTE', 'IPM',
                  'TASA_DESEMPLEO', 'SGP_EDUC_PC', 'PROP_NBI']

# =============================================================================
# CARGA DE ARCHIVOS
# =============================================================================
print("=" * 80)
print("CARGANDO ARCHIVOS")
print("=" * 80)

print("\n── Panel jerárquico ──")
panel = pd.read_csv(
    ARCHIVO_PANEL,
    usecols=['periodo', 'ANIO', 'COD_DEPTO', 'COD_MUNI'] + VARS_VERIFICAR,
    low_memory=False
)
panel['COD_MUNI'] = (panel['COD_MUNI'].astype(str)
                     .str.replace('.0', '', regex=False).str.zfill(5))
panel['COD_DEPTO'] = (panel['COD_DEPTO'].astype(str)
                      .str.replace('.0', '', regex=False).str.zfill(2))
print(f"  ✓ {len(panel):,} filas")

print("\n── Muestra ──")
muestra = pd.read_csv(ARCHIVO_MUESTRA, low_memory=False)
muestra['COD_MUNI'] = (muestra['COD_MUNI'].astype(str)
                       .str.replace('.0', '', regex=False).str.zfill(5))
muestra['COD_DEPTO'] = (muestra['COD_DEPTO'].astype(str)
                        .str.replace('.0', '', regex=False).str.zfill(2))
print(f"  ✓ {len(muestra):,} filas × {muestra.shape[1]} columnas")

print("\n── Panel municipal (referencia) ──")
panel_mun = pd.read_csv(ARCHIVO_PANEL_MUN)
panel_mun['COD_MUNI'] = (panel_mun['COD_MUNI'].astype(str)
                         .str.replace('.0', '', regex=False).str.zfill(5))
panel_mun['ANIO'] = pd.to_numeric(panel_mun['ANIO'], errors='coerce').astype('Int64')
print(f"  ✓ {len(panel_mun):,} filas")

# =============================================================================
# VALIDACIÓN 1: INTEGRIDAD INDIVIDUO-CONTEXTO
# =============================================================================
print("\n" + "=" * 80)
print("VALIDACIÓN 1: INTEGRIDAD INDIVIDUO-CONTEXTO")
print("=" * 80)
print("\nTomando 1,000 estudiantes al azar y verificando que sus variables")
print("contextuales coincidan con las del panel municipal original.")

np.random.seed(123)
check = muestra.sample(1000, random_state=123).copy()
check['ANIO'] = check['ANIO'].astype(int)

resumen_val1 = {}

for var in VARS_VERIFICAR:
    if var not in muestra.columns or var not in panel_mun.columns:
        continue

    errores = 0
    ambos_na = 0
    validados = 0

    for _, row in check.iterrows():
        esperado = panel_mun[
            (panel_mun['COD_MUNI'] == row['COD_MUNI']) &
            (panel_mun['ANIO'] == row['ANIO'])
        ][var].values

        if len(esperado) == 0:
            continue

        obs = row[var]
        exp = esperado[0]

        if pd.isna(obs) and pd.isna(exp):
            ambos_na += 1
            validados += 1
            continue

        if pd.isna(obs) != pd.isna(exp):
            errores += 1
            continue

        if not np.isclose(obs, exp, rtol=1e-5, equal_nan=True):
            errores += 1
        else:
            validados += 1

    resumen_val1[var] = {'validados': validados, 'errores': errores, 'ambos_na': ambos_na}
    estado = "✓" if errores == 0 else "✗"
    print(f"  {estado} {var:25s}  validados: {validados:>4}  "
          f"errores: {errores:>3}  ambos_na: {ambos_na:>4}")

total_errores_v1 = sum(r['errores'] for r in resumen_val1.values())
print(f"\n  {'✅' if total_errores_v1 == 0 else '❌'} Total errores: {total_errores_v1}")

# =============================================================================
# VALIDACIÓN 2: INVARIANTES DEL MUESTREO ESTRATIFICADO
# =============================================================================
print("\n" + "=" * 80)
print("VALIDACIÓN 2: INVARIANTES DEL MUESTREO ESTRATIFICADO")
print("=" * 80)

# 2.1 Proporciones por periodo preservadas
print("\n── 2.1 Proporciones por periodo ──")
prop_pob = panel['periodo'].value_counts(normalize=True).sort_index()
prop_mue = muestra['periodo'].value_counts(normalize=True).sort_index()

tabla_prop = pd.DataFrame({
    'Poblacion_%': prop_pob * 100,
    'Muestra_%': prop_mue * 100,
    'Diferencia_%': (prop_mue - prop_pob) * 100
})
print(tabla_prop.round(3).to_string())

max_dif = (prop_pob - prop_mue).abs().max() * 100
print(f"\n  Máxima diferencia: {max_dif:.4f}%")
print(f"  {'✅' if max_dif < 0.1 else '⚠'} {'Proporciones preservadas' if max_dif < 0.1 else 'Diferencias notables'}")

# 2.2 Todos los municipios de la población están en la muestra
print("\n── 2.2 Cobertura de municipios ──")
munis_pob = set(panel.groupby(['COD_DEPTO', 'COD_MUNI']).groups.keys())
munis_mue = set(muestra.groupby(['COD_DEPTO', 'COD_MUNI']).groups.keys())
ausentes = munis_pob - munis_mue

print(f"  Municipios en población: {len(munis_pob):,}")
print(f"  Municipios en muestra: {len(munis_mue):,}")
print(f"  Municipios ausentes en muestra: {len(ausentes):,}")

if len(ausentes) > 0:
    print(f"  ⚠ Municipios perdidos (primeros 10): {list(ausentes)[:10]}")
else:
    print(f"  ✅ Todos los municipios representados")

# 2.3 Cobertura por año
print("\n── 2.3 Cobertura por año ──")
anios_pob = set(panel['ANIO'].dropna().astype(int).unique())
anios_mue = set(muestra['ANIO'].dropna().astype(int).unique())
print(f"  Años en población: {sorted(anios_pob)}")
print(f"  Años en muestra: {sorted(anios_mue)}")
print(f"  {'✅' if anios_pob == anios_mue else '✗'} Todos los años representados")

# =============================================================================
# VALIDACIÓN 3: CONSISTENCIA INTERNA
# =============================================================================
print("\n" + "=" * 80)
print("VALIDACIÓN 3: CONSISTENCIA INTERNA")
print("=" * 80)

# 3.1 Año coherente con periodo
print("\n── 3.1 Coherencia año-periodo ──")
muestra['anio_from_periodo'] = muestra['periodo'].astype(str).str[:4].astype(int)
inconsistentes = (muestra['anio_from_periodo'] != muestra['ANIO'].astype(int)).sum()
print(f"  Inconsistencias año-periodo: {inconsistentes:,}")
print(f"  {'✅' if inconsistentes == 0 else '❌'} "
      f"{'Año y periodo coherentes' if inconsistentes == 0 else 'Hay inconsistencias'}")

# 3.2 Variables estáticas constantes por municipio
print("\n── 3.2 Variables estáticas (NBI) constantes por municipio ──")
if 'PROP_NBI' in muestra.columns:
    nbi_var = muestra.groupby('COD_MUNI')['PROP_NBI'].nunique(dropna=True)
    munis_inconsistentes = (nbi_var > 1).sum()
    print(f"  Municipios con PROP_NBI inconsistente: {munis_inconsistentes}")
    print(f"  {'✅' if munis_inconsistentes == 0 else '❌'} "
          f"{'NBI estable por municipio' if munis_inconsistentes == 0 else 'NBI variable'}")
else:
    print("  ⚠ PROP_NBI no está en la muestra")

# 3.3 Sin duplicados exactos
print("\n── 3.3 Duplicados exactos ──")
duplicados = muestra.duplicated().sum()
print(f"  Filas duplicadas: {duplicados}")
print(f"  {'✅' if duplicados == 0 else '❌'} "
      f"{'Sin duplicados' if duplicados == 0 else 'Hay duplicados'}")

# 3.4 Los códigos de departamento y municipio son consistentes
print("\n── 3.4 Coherencia COD_DEPTO ↔ COD_MUNI ──")
# Los primeros 2 dígitos de COD_MUNI deben coincidir con COD_DEPTO
muestra['depto_from_muni'] = muestra['COD_MUNI'].str[:2]
incoherentes = (muestra['depto_from_muni'] != muestra['COD_DEPTO']).sum()
print(f"  Registros con COD_DEPTO incoherente con COD_MUNI: {incoherentes:,}")
print(f"  {'✅' if incoherentes == 0 else '❌'} "
      f"{'Códigos coherentes' if incoherentes == 0 else 'Inconsistencias detectadas'}")

# Limpiar columnas auxiliares
muestra = muestra.drop(columns=['anio_from_periodo', 'depto_from_muni'])

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 80)
print("RESUMEN DE VALIDACIÓN")
print("=" * 80)

checks = {
    "1. Integridad individuo-contexto":    total_errores_v1 == 0,
    "2.1 Proporciones por periodo":        max_dif < 0.1,
    "2.2 Todos los municipios en muestra": len(ausentes) == 0,
    "2.3 Todos los años representados":    anios_pob == anios_mue,
    "3.1 Año-periodo coherentes":          inconsistentes == 0,
    "3.2 NBI estable por municipio":       'PROP_NBI' not in muestra.columns or munis_inconsistentes == 0,
    "3.3 Sin duplicados":                  duplicados == 0,
    "3.4 COD_DEPTO-COD_MUNI coherentes":   incoherentes == 0,
}

for nombre, ok in checks.items():
    print(f"  {'✅' if ok else '❌'}  {nombre}")

if all(checks.values()):
    print("\n" + "=" * 80)
    print("✅ TODAS LAS VALIDACIONES PASARON. LA MUESTRA ES CORRECTA.")
    print("=" * 80)
else:
    print("\n" + "=" * 80)
    print("⚠ ALGUNAS VALIDACIONES FALLARON. REVISAR LOS DETALLES ARRIBA.")
    print("=" * 80)

muestra = muestra[muestra['COD_MUNI'] != '99999'].reset_index(drop=True)
muestra.to_csv(ARCHIVO_MUESTRA, index=False, encoding='utf-8-sig')
print(f"Registros finales: {len(muestra):,}")