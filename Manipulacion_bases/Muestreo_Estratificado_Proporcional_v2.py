# -*- coding: utf-8 -*-
"""
@author: aarevaloa

Muestreo Estratificado de Dos Niveles sobre el Panel Jerárquico
================================================================

Toma el panel jerárquico completo (individual + municipal + departamental)
y genera una muestra estratificada por periodo y municipio, preservando:
  1. La proporción de individuos por periodo
  2. Dentro de cada periodo, la proporción por municipio-departamento
  3. Las distribuciones de todas las variables contextuales

Genera dos archivos de salida:
  - ICFES_MUESTRA_<N>_DESCRIPTIVO.csv  → variables categóricas para análisis
  - ICFES_MUESTRA_<N>_MODELO.csv       → dummies + contexto para el modelo
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

RUTA_BASE = Path(r'C:\Users\USUARIO\Desktop\Estadística\Maestría en Estadística\Tesis\Escritorio')
RUTA_DATOS = RUTA_BASE / 'Datos_procesados' / 'Python'

ARCHIVO_PANEL = RUTA_DATOS / 'panel_jerarquico_completo.csv'

TAMAÑO_MUESTRA = 500_000
SEMILLA_ALEATORIA = 42

# Columnas de estratificación (nombres del panel jerárquico)
COL_DEPTO = 'COD_DEPTO'
COL_MUNI = 'COD_MUNI'

# =============================================================================
# VARIABLES CONTEXTUALES (IDÉNTICAS EN AMBOS ARCHIVOS)
# =============================================================================

VARIABLES_CONTEXTUALES = [
    'POBLACION', 'PROP_URBANA', 'LOG_POBLACION',
    'TASA_HOMICIDIOS', 'PROP_HOMICIDIOS_URBANO', 'TASA_VICTIMAS',
    'TASA_MATRICULA_ES', 'IES_CON_OFERTA',
    'SGP_EDUC_PC', 'SGP_TOTAL_PC',
    'PROP_NBI', 'COMP_INASISTENCIA_ESCOLAR',
    'PIB_CONSTANTE', 'CREC_PIB_REAL',
    'TASA_DESEMPLEO', 'TASA_SUBOCUPACION',
    'IPM',
]

IDS = ['periodo', 'ANIO', 'COD_DEPTO', 'estu_depto_reside',
       'COD_MUNI', 'estu_mcpio_reside']

VARS_INDIVIDUALES_DESCRIPTIVO = [
    'punt_global', 'ubicacion_cat', 'cole_bilingue_cat', 'cole_calendario_cat',
    'cole_caracter_cat', 'cole_genero_cat', 'cole_naturaleza_cat', 'inter_cat',
    'lectu_cat', 'genero_cat', 'trabaja_cat', 'come_carne_cat', 'come_cereal_cat',
    'come_leche_cat', 'edu_madre_cat', 'libros_casa_cat', 'compu_cat', 'internet_cat'
]

VARS_INDIVIDUALES_MODELO = [
    'estu_inse_individual', 'punt_global',
    'cole_area_ubicacion_RURAL', 'cole_area_ubicacion_MISSING',
    'cole_bilingue_S', 'cole_bilingue_MISSING',
    'calen_B', 'calen_OTRO', 'calen_MISSING',
    'cole_carac_TECNICO_ACADEMICO', 'cole_carac_TECNICO',
    'cole_carac_NO_APLICA', 'cole_carac_MISSING',
    'cole_gen_FEMENINO', 'cole_gen_MASCULINO', 'cole_gen_MISSING',
    'cole_natu_NO_OFICIAL', 'cole_natu_MISSING',
    'inter_ENTRE 1 Y 3 HORAS', 'inter_ENTRE 30 Y 60 MINUTOS',
    'inter_MÁS DE 3 HORAS', 'inter_30 MINUTOS O MENOS', 'inter_MISSING',
    'lectu_30 MINUTOS O MENOS', 'lectu_ENTRE 30 Y 60 MINUTOS',
    'lectu_ENTRE 1 Y 2 HORAS', 'lectu_MÁS DE 2 HORAS', 'lectu_MISSING',
    'genero_F', 'genero_MISSING',
    'trabaja_MENOS DE 10 HORAS', 'trabaja_ENTRE 11 Y 20 HORAS',
    'trabaja_ENTRE 21 Y 30 HORAS', 'trabaja_MÁS DE 30 HORAS', 'trabaja_MISSING',
    'come_carne_TODOS O CASI TODOS LOS DÍAS', 'come_carne_3 A 5 VECES POR SEMANA',
    'come_carne_1 O 2 VECES POR SEMANA', 'come_carne_MISSING',
    'come_cereal_TODOS O CASI TODOS LOS DÍAS', 'come_cereal_3 A 5 VECES POR SEMANA',
    'come_cereal_1 O 2 VECES POR SEMANA', 'come_cereal_MISSING',
    'come_leche_TODOS O CASI TODOS LOS DÍAS', 'come_leche_3 A 5 VECES POR SEMANA',
    'come_leche_1 O 2 VECES POR SEMANA', 'come_leche_MISSING',
    'edu_madre_PRIMARIA INCOMPLETA', 'edu_madre_PRIMARIA COMPLETA',
    'edu_madre_SECUNDARIA (BACHILLERATO) INCOMPLETA',
    'edu_madre_SECUNDARIA (BACHILLERATO) COMPLETA',
    'edu_madre_TÉCNICA O TECNOLÓGICA INCOMPLETA',
    'edu_madre_TÉCNICA O TECNOLÓGICA COMPLETA',
    'edu_madre_EDUCACIÓN PROFESIONAL INCOMPLETA',
    'edu_madre_EDUCACIÓN PROFESIONAL COMPLETA', 'edu_madre_POSTGRADO',
    'edu_madre_NO SABE', 'edu_madre_NO APLICA', 'edu_madre_MISSING',
    'libros_casa_11 A 25 LIBROS', 'libros_casa_26 A 100 LIBROS',
    'libros_casa_MÁS DE 100 LIBROS', 'libros_casa_MISSING',
    'compu_casa_SI', 'compu_casa_MISSING',
    'internet_SI', 'internet_MISSING'
]


# =============================================================================
# FUNCIONES DE MUESTREO
# =============================================================================

def cargar_panel(archivo: Path) -> pd.DataFrame:
    """Carga el panel jerárquico completo."""
    print("=" * 80)
    print("CARGANDO PANEL JERÁRQUICO")
    print("=" * 80)
    print(f"\nArchivo: {archivo.name}")

    df = pd.read_csv(archivo, encoding='utf-8-sig', low_memory=False)

    # Asegurar que los códigos queden como string con ceros a la izquierda
    df[COL_DEPTO] = df[COL_DEPTO].astype(str).str.replace('.0', '', regex=False).str.strip().str.zfill(2)
    df[COL_MUNI] = df[COL_MUNI].astype(str).str.replace('.0', '', regex=False).str.strip().str.zfill(5)

    print(f"✅ Panel cargado")
    print(f"   • Registros: {len(df):,}")
    print(f"   • Variables: {len(df.columns)}")
    print(f"   • Periodos únicos: {df['periodo'].nunique()}")
    print(f"   • Combinaciones depto-municipio: "
          f"{df.groupby([COL_DEPTO, COL_MUNI]).ngroups:,}")

    return df


def analizar_distribucion_periodos(df: pd.DataFrame) -> pd.DataFrame:
    """Distribución de registros por periodo."""
    periodos = df.groupby('periodo').size().reset_index(name='n_total')
    periodos['proporcion'] = periodos['n_total'] / len(df)
    periodos['porcentaje'] = periodos['proporcion'] * 100
    return periodos.sort_values('periodo')


def muestreo_estratificado_periodo(
    df_periodo: pd.DataFrame,
    tamaño_muestra_periodo: int,
    semilla: Optional[int] = None
) -> pd.DataFrame:
    """Muestreo estratificado por municipio dentro de un periodo."""
    if tamaño_muestra_periodo >= len(df_periodo):
        return df_periodo.copy()

    estratos = df_periodo.groupby(
        [COL_DEPTO, COL_MUNI]
    ).size().reset_index(name='n_total')
    estratos['proporcion'] = estratos['n_total'] / len(df_periodo)
    estratos['n_muestra'] = (estratos['proporcion'] * tamaño_muestra_periodo).round().astype(int)

    diferencia = tamaño_muestra_periodo - estratos['n_muestra'].sum()
    if diferencia != 0:
        indices_ajuste = estratos.nlargest(abs(diferencia), 'n_total').index
        estratos.loc[indices_ajuste, 'n_muestra'] += np.sign(diferencia)

    muestras = []
    for _, row in estratos.iterrows():
        depto = row[COL_DEPTO]
        mcpio = row[COL_MUNI]
        n_muestra_estrato = int(row['n_muestra'])

        if n_muestra_estrato == 0:
            continue

        mask = (df_periodo[COL_DEPTO] == depto) & (df_periodo[COL_MUNI] == mcpio)
        datos_estrato = df_periodo[mask]

        n_tomar = int(min(n_muestra_estrato, len(datos_estrato)))
        muestra_estrato = datos_estrato.sample(n=n_tomar, random_state=semilla)
        muestras.append(muestra_estrato)

    return pd.concat(muestras, ignore_index=True) if muestras else pd.DataFrame()


def muestreo_estratificado_dos_niveles(
    df: pd.DataFrame,
    tamaño_muestra: int,
    semilla: Optional[int] = None
) -> pd.DataFrame:
    """Muestreo estratificado de dos niveles: periodo + municipio."""
    print("\n" + "=" * 80)
    print("MUESTREO ESTRATIFICADO DE DOS NIVELES")
    print("=" * 80)

    if semilla is not None:
        np.random.seed(semilla)

    if tamaño_muestra >= len(df):
        print(f"\n⚠️  Tamaño de muestra ≥ población. Se retorna población completa.")
        return df.copy()

    print(f"\nPoblación: {len(df):,} | Muestra solicitada: {tamaño_muestra:,} "
          f"| Fracción: {tamaño_muestra/len(df)*100:.2f}%")

    # Nivel 1: por periodo
    print("\n" + "-" * 80)
    print("NIVEL 1: ESTRATIFICACIÓN POR PERIODO")
    print("-" * 80)

    periodos = analizar_distribucion_periodos(df)
    periodos['n_muestra'] = (periodos['proporcion'] * tamaño_muestra).round().astype(int)

    diferencia = tamaño_muestra - periodos['n_muestra'].sum()
    if diferencia != 0:
        indices_ajuste = periodos.nlargest(abs(diferencia), 'n_total').index
        periodos.loc[indices_ajuste, 'n_muestra'] += np.sign(diferencia)

    print(f"\n📋 Distribución por periodo:")
    print(periodos[['periodo', 'n_total', 'porcentaje', 'n_muestra']]
          .round(2).to_string(index=False))

    # Nivel 2: por municipio dentro del periodo
    print("\n" + "-" * 80)
    print("NIVEL 2: ESTRATIFICACIÓN POR MUNICIPIO (DENTRO DE CADA PERIODO)")
    print("-" * 80)

    muestras_periodos = []
    for _, row in periodos.iterrows():
        periodo = row['periodo']
        n_muestra_periodo = int(row['n_muestra'])
        if n_muestra_periodo == 0:
            continue

        print(f"\n📅 Periodo {periodo}: población {row['n_total']:,} → "
              f"muestra {n_muestra_periodo:,}")

        df_periodo = df[df['periodo'] == periodo]
        n_estratos = df_periodo.groupby([COL_DEPTO, COL_MUNI]).ngroups
        print(f"   Municipios en el periodo: {n_estratos:,}")

        muestra_periodo = muestreo_estratificado_periodo(
            df_periodo, n_muestra_periodo, semilla
        )
        print(f"   ✅ Muestra obtenida: {len(muestra_periodo):,}")
        muestras_periodos.append(muestra_periodo)

    muestra_final = pd.concat(muestras_periodos, ignore_index=True)

    print("\n" + "=" * 80)
    print("✅ MUESTREO COMPLETADO")
    print("=" * 80)
    print(f"\n   • Registros en muestra: {len(muestra_final):,}")
    print(f"   • Periodos representados: {muestra_final['periodo'].nunique()}")
    print(f"   • Estratos representados: "
          f"{muestra_final.groupby([COL_DEPTO, COL_MUNI]).ngroups:,}")

    return muestra_final


# =============================================================================
# COMPARACIÓN DE DISTRIBUCIONES
# =============================================================================

def comparar_periodos(df_original: pd.DataFrame, df_muestra: pd.DataFrame) -> pd.DataFrame:
    """Compara distribución por periodo."""
    print("\n" + "=" * 80)
    print("COMPARACIÓN DE DISTRIBUCIÓN POR PERIODO")
    print("=" * 80)

    dist_pob = df_original['periodo'].value_counts(normalize=True).sort_index() * 100
    dist_muestra = df_muestra['periodo'].value_counts(normalize=True).sort_index() * 100

    comparacion = pd.DataFrame({
        'Poblacion_%': dist_pob,
        'Muestra_%': dist_muestra,
        'Diferencia_%': dist_muestra - dist_pob
    })
    print("\n" + comparacion.round(3).to_string())
    print(f"\nDiferencia absoluta promedio: {comparacion['Diferencia_%'].abs().mean():.3f}%")
    print(f"Máxima diferencia: {comparacion['Diferencia_%'].abs().max():.3f}%")

    return comparacion


def comparar_variables_contextuales(
    df_original: pd.DataFrame,
    df_muestra: pd.DataFrame,
    variables: list
) -> pd.DataFrame:
    """Compara medias de variables contextuales entre población y muestra."""
    print("\n" + "=" * 80)
    print("COMPARACIÓN DE VARIABLES CONTEXTUALES (medias)")
    print("=" * 80)

    resultados = []
    for var in variables:
        if var not in df_original.columns:
            continue
        media_pob = df_original[var].mean()
        media_muestra = df_muestra[var].mean()
        diferencia = media_muestra - media_pob
        dif_rel = (diferencia / media_pob * 100) if media_pob != 0 else 0

        resultados.append({
            'Variable': var,
            'Media_Poblacion': media_pob,
            'Media_Muestra': media_muestra,
            'Diferencia': diferencia,
            'Diferencia_Rel_%': dif_rel
        })

    df_res = pd.DataFrame(resultados)
    print("\n" + df_res.round(4).to_string(index=False))
    return df_res


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def ejecutar_muestreo():
    """Ejecuta el muestreo y genera los dos archivos de salida."""

    print("=" * 80)
    print("MUESTREO SOBRE PANEL JERÁRQUICO - ICFES SABER 11")
    print("=" * 80)
    print(f"\nConfiguración:")
    print(f"  • Tamaño de muestra: {TAMAÑO_MUESTRA:,}")
    print(f"  • Semilla: {SEMILLA_ALEATORIA}")

    if not ARCHIVO_PANEL.exists():
        raise FileNotFoundError(f"No se encontró: {ARCHIVO_PANEL}")

    # 1. Cargar panel
    df_panel = cargar_panel(ARCHIVO_PANEL)

    # 2. Muestreo
    df_muestra = muestreo_estratificado_dos_niveles(
        df_panel, TAMAÑO_MUESTRA, SEMILLA_ALEATORIA
    )

    # 3. Diagnóstico
    df_comp_periodos = comparar_periodos(df_panel, df_muestra)
    df_comp_contexto = comparar_variables_contextuales(
        df_panel, df_muestra, VARIABLES_CONTEXTUALES
    )

    # 4. Construir los dos archivos de salida
    print("\n" + "=" * 80)
    print("GENERANDO ARCHIVOS DE SALIDA")
    print("=" * 80)

    # 4.1 Versión DESCRIPTIVA
    cols_descriptivo = [c for c in (IDS + VARS_INDIVIDUALES_DESCRIPTIVO + VARIABLES_CONTEXTUALES)
                        if c in df_muestra.columns]
    df_descriptivo = df_muestra[cols_descriptivo].copy()

    archivo_desc = RUTA_DATOS / f'ICFES_MUESTRA_{TAMAÑO_MUESTRA}_DESCRIPTIVO.csv'
    df_descriptivo.to_csv(archivo_desc, index=False, encoding='utf-8-sig')
    print(f"\n✅ Análisis descriptivo: {archivo_desc.name}")
    print(f"   • Registros: {len(df_descriptivo):,}")
    print(f"   • Variables: {len(df_descriptivo.columns)}")

    # 4.2 Versión MODELO
    cols_modelo = [c for c in (IDS + VARS_INDIVIDUALES_MODELO + VARIABLES_CONTEXTUALES)
                   if c in df_muestra.columns]
    df_modelo = df_muestra[cols_modelo].copy()

    archivo_mod = RUTA_DATOS / f'ICFES_MUESTRA_{TAMAÑO_MUESTRA}_MODELO.csv'
    df_modelo.to_csv(archivo_mod, index=False, encoding='utf-8-sig')
    print(f"\n✅ Modelo: {archivo_mod.name}")
    print(f"   • Registros: {len(df_modelo):,}")
    print(f"   • Variables: {len(df_modelo.columns)}")

    # 4.3 Guardar comparaciones
    (RUTA_BASE / 'Resultados' / 'Datos').mkdir(parents=True, exist_ok=True)
    df_comp_periodos.to_csv(
        RUTA_BASE / 'Resultados' / 'Datos' / f'COMPARACION_PERIODOS_{TAMAÑO_MUESTRA}.csv',
        encoding='utf-8-sig'
    )
    df_comp_contexto.to_csv(
        RUTA_BASE / 'Resultados' / 'Datos' / f'COMPARACION_CONTEXTUALES_{TAMAÑO_MUESTRA}.csv',
        index=False, encoding='utf-8-sig'
    )
    print(f"\n✅ Comparaciones guardadas en Resultados/Datos/")

    # Resumen final
    print("\n" + "=" * 80)
    print("✅ PROCESO COMPLETADO")
    print("=" * 80)
    print(f"\n📈 RESUMEN:")
    print(f"   • Población: {len(df_panel):,}")
    print(f"   • Muestra: {len(df_muestra):,}")
    print(f"   • Fracción: {len(df_muestra)/len(df_panel)*100:.2f}%")
    print(f"   • Estratos población: "
          f"{df_panel.groupby([COL_DEPTO, COL_MUNI]).ngroups:,}")
    print(f"   • Estratos muestra: "
          f"{df_muestra.groupby([COL_DEPTO, COL_MUNI]).ngroups:,}")
    print(f"   • Diferencia promedio periodos: "
          f"{df_comp_periodos['Diferencia_%'].abs().mean():.3f}%")

    return df_muestra, df_comp_periodos, df_comp_contexto


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    df_muestra, df_comp_periodos, df_comp_contexto = ejecutar_muestreo()