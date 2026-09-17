# -*- coding: utf-8 -*-
'''
Análisis de combinaciones Municipio-Departamento en datos ICFES SABER 11
Agrupa datos por año (uniendo ambos semestres) y analiza presencia/ausencia

@author: USUARIO
Created on: Fri Feb 6 10:52:26 2026
'''

import pandas as pd
from pathlib import Path
from typing import Dict, Set, Tuple, List
import warnings
warnings.filterwarnings('ignore')

#_________________________________________________________________________________________________________________________
# CONFIGURACIÓN
#_________________________________________________________________________________________________________________________

RUTA_BASE = Path(r'C:\Users\USUARIO\Desktop\Estadística\Maestría en Estadística\Tesis\Escritorio')
RUTA_DATOS = RUTA_BASE / 'Datos_originales' / 'Datos_nivel_individual'

AÑOS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
SEMESTRES = [1, 2]

#_________________________________________________________________________________________________________________________
# FUNCIONES DE CARGA Y PROCESAMIENTO
#_________________________________________________________________________________________________________________________

def cargar_combinaciones_archivo(archivo: Path) -> Set[Tuple[str, str]]:
    '''
    Carga un archivo individual y extrae combinaciones únicas (depto, mcpio).
    
    Parameters
    ----------
    archivo : Path
        Ruta al archivo de datos
        
    Returns
    -------
    Set[Tuple[str, str]]
        Conjunto de tuplas (departamento, municipio)
    '''
    try:
        df = pd.read_csv(
            archivo,
            sep=';',
            usecols=['estu_cod_reside_depto', 'estu_cod_reside_mcpio'],
            dtype=str,
            encoding='latin-1'  # Codificación común para archivos ICFES
        )
        
        # Limpiar datos
        df = df.dropna()
        df['estu_cod_reside_depto'] = df['estu_cod_reside_depto'].str.strip()
        df['estu_cod_reside_mcpio'] = df['estu_cod_reside_mcpio'].str.strip()
        
        # Crear tuplas únicas
        combinaciones = set(zip(
            df['estu_cod_reside_depto'], 
            df['estu_cod_reside_mcpio']
        ))
        
        return combinaciones
    
    except FileNotFoundError:
        print(f'  ⚠️  Archivo no encontrado: {archivo.name}')
        return set()
    except Exception as e:
        print(f'  ❌ Error al cargar {archivo.name}: {e}')
        return set()


def cargar_combinaciones_año(año: int, ruta_datos: Path) -> Set[Tuple[str, str]]:
    '''
    Carga y combina las combinaciones de ambos semestres de un año.
    
    Parameters
    ----------
    año : int
        Año a procesar (ej: 2017, 2018, ...)
    ruta_datos : Path
        Ruta a la carpeta de datos
        
    Returns
    -------
    Set[Tuple[str, str]]
        Unión de todas las combinaciones encontradas en ambos semestres
    '''
    print(f'\n  📅 Procesando año {año}...')
    
    combinaciones_año = set()
    archivos_cargados = 0
    
    for semestre in SEMESTRES:
        # Formato: Examen_Saber_11_20242.txt (año completo + semestre)
        periodo = f'{año}{semestre}'
        archivo = ruta_datos / f'Examen_Saber_11_{periodo}.txt'
        
        if archivo.exists():
            combinaciones_semestre = cargar_combinaciones_archivo(archivo)
            combinaciones_año.update(combinaciones_semestre)
            archivos_cargados += 1
            print(f'    ✅ Semestre {semestre}: {len(combinaciones_semestre):,} combinaciones')
        else:
            print(f'    ⚠️  Semestre {semestre}: archivo no encontrado')
    
    if archivos_cargados == 0:
        print(f'    ❌ No se cargaron archivos para {año}')
    else:
        print(f'    📊 Total año {año}: {len(combinaciones_año):,} combinaciones únicas')
    
    return combinaciones_año


#_________________________________________________________________________________________________________________________
# FUNCIONES DE ANÁLISIS
#_________________________________________________________________________________________________________________________

def encontrar_combinaciones_presentes_todos_años(
    datos_por_año: Dict[int, Set[Tuple[str, str]]]
) -> Set[Tuple[str, str]]:
    '''
    Encuentra combinaciones presentes en TODOS los años.
    
    Parameters
    ----------
    datos_por_año : Dict[int, Set[Tuple[str, str]]]
        Diccionario con año como clave y set de combinaciones como valor
        
    Returns
    -------
    Set[Tuple[str, str]]
        Combinaciones presentes en todos los años
    '''
    if not datos_por_año:
        return set()
    
    # Intersección de todos los conjuntos
    combinaciones_comunes = set.intersection(*datos_por_año.values())
    return combinaciones_comunes


def encontrar_combinaciones_ausentes_por_año(
    datos_por_año: Dict[int, Set[Tuple[str, str]]],
    combinaciones_totales: Set[Tuple[str, str]]
) -> Dict[int, Set[Tuple[str, str]]]:
    '''
    Para cada año, encuentra las combinaciones que NO están presentes
    pero sí están en al menos otro año.
    
    Parameters
    ----------
    datos_por_año : Dict[int, Set[Tuple[str, str]]]
        Diccionario con año como clave y set de combinaciones como valor
    combinaciones_totales : Set[Tuple[str, str]]
        Unión de todas las combinaciones de todos los años
        
    Returns
    -------
    Dict[int, Set[Tuple[str, str]]]
        Diccionario con año y sus combinaciones ausentes
    '''
    ausentes_por_año = {}
    
    for año, combinaciones_año in datos_por_año.items():
        # Combinaciones que están en otros años pero no en este
        ausentes = combinaciones_totales - combinaciones_año
        ausentes_por_año[año] = ausentes
    
    return ausentes_por_año


#_________________________________________________________________________________________________________________________
# FUNCIONES DE FORMATO Y EXPORTACIÓN
#_________________________________________________________________________________________________________________________

def formatear_dataframe(
    combinaciones: Set[Tuple[str, str]]
) -> pd.DataFrame:
    '''
    Convierte un conjunto de combinaciones en un DataFrame ordenado.
    
    Parameters
    ----------
    combinaciones : Set[Tuple[str, str]]
        Conjunto de tuplas (departamento, municipio)
        
    Returns
    -------
    pd.DataFrame
        DataFrame con columnas 'Cod_Departamento' y 'Cod_Municipio'
    '''
    if not combinaciones:
        return pd.DataFrame(columns=['Cod_Departamento', 'Cod_Municipio'])
    
    df = pd.DataFrame(
        sorted(combinaciones),
        columns=['Cod_Departamento', 'Cod_Municipio']
    )
    
    return df


def guardar_resultados(
    df_comunes: pd.DataFrame,
    ausentes_por_año: Dict[int, Set[Tuple[str, str]]],
    df_resumen: pd.DataFrame,
    ruta_base: Path
):
    '''
    Guarda todos los resultados en archivos CSV.
    
    Parameters
    ----------
    df_comunes : pd.DataFrame
        DataFrame con combinaciones presentes en todos los años
    ausentes_por_año : Dict[int, Set[Tuple[str, str]]]
        Diccionario con combinaciones ausentes por año
    df_resumen : pd.DataFrame
        DataFrame con resumen estadístico
    ruta_base : Path
        Ruta base para guardar archivos
    '''
    print('\n' + '='*80)
    print('💾 GUARDANDO RESULTADOS')
    print('='*80)
    
    # 1. Combinaciones presentes en todos los años
    archivo_comunes = ruta_base / 'Resultados' / 'Datos' / 'combinaciones_presentes_todos_años.csv'
    df_comunes.to_csv(archivo_comunes, index=False, encoding='utf-8-sig')
    print(f'\n✅ Combinaciones comunes: {archivo_comunes}')
    
    # 2. Combinaciones ausentes por año (archivos individuales)
    carpeta_ausentes = ruta_base / 'Resultados' / 'Datos' / 'combinaciones_ausentes_por_año'
    carpeta_ausentes.mkdir(exist_ok=True)
    
    print(f'\n📁 Creando carpeta: {carpeta_ausentes}')
    
    for año, combinaciones in sorted(ausentes_por_año.items()):
        df_ausente = formatear_dataframe(combinaciones)
        archivo_ausente = carpeta_ausentes / f'ausentes_{año}.csv'
        df_ausente.to_csv(archivo_ausente, index=False, encoding='utf-8-sig')
        print(f'  ✅ {año}: {len(df_ausente):,} combinaciones ausentes')
    
    # 3. Resumen consolidado
    archivo_resumen = ruta_base / 'Resultados' / 'Datos' / 'resumen_analisis_por_año.csv'
    df_resumen.to_csv(archivo_resumen, index=False, encoding='utf-8-sig')
    print(f'\n✅ Resumen consolidado: {archivo_resumen}')


#_________________________________________________________________________________________________________________________
# FUNCIÓN PRINCIPAL
#_________________________________________________________________________________________________________________________

def analizar_municipios_departamentos_por_año():
    '''
    Función principal que ejecuta todo el análisis.
    
    Returns
    -------
    tuple
        (df_comunes, ausentes_por_año, df_resumen)
    '''
    
    print('='*80)
    print('ANÁLISIS DE COMBINACIONES MUNICIPIO-DEPARTAMENTO - ICFES SABER 11')
    print('Agrupado por AÑO (uniendo ambos semestres)')
    print('='*80)
    
    # -------------------------------------------------------------------------
    # PASO 1: CARGAR DATOS POR AÑO
    # -------------------------------------------------------------------------
    
    print('\n' + '='*80)
    print('📂 PASO 1: CARGANDO DATOS POR AÑO')
    print('='*80)
    
    datos_por_año: Dict[int, Set[Tuple[str, str]]] = {}
    
    for año in AÑOS:
        combinaciones = cargar_combinaciones_año(año, RUTA_DATOS)
        if combinaciones:  # Solo añadir si se cargaron datos
            datos_por_año[año] = combinaciones
    
    if not datos_por_año:
        print('\n❌ ERROR: No se cargaron datos de ningún año')
        return None, None, None
    
    print(f'\n✅ Total de años procesados: {len(datos_por_año)}')
    
    # -------------------------------------------------------------------------
    # PASO 2: COMBINACIONES PRESENTES EN TODOS LOS AÑOS
    # -------------------------------------------------------------------------
    
    print('\n' + '='*80)
    print('🔍 PASO 2: COMBINACIONES PRESENTES EN TODOS LOS AÑOS')
    print('='*80)
    
    combinaciones_comunes = encontrar_combinaciones_presentes_todos_años(datos_por_año)
    
    print(f'\n📊 Total de combinaciones presentes en TODOS los años: {len(combinaciones_comunes):,}')
    
    df_comunes = formatear_dataframe(combinaciones_comunes)
    
    if len(df_comunes) > 0:
        print('\n📋 Primeras 15 combinaciones comunes:')
        print(df_comunes.head(15).to_string(index=False))
    
    # -------------------------------------------------------------------------
    # PASO 3: COMBINACIONES AUSENTES POR AÑO
    # -------------------------------------------------------------------------
    
    print('\n' + '='*80)
    print('🔍 PASO 3: COMBINACIONES AUSENTES POR AÑO')
    print('='*80)
    print('\n(Combinaciones que existen en otros años pero NO en el año indicado)')
    
    # Unión de todas las combinaciones
    combinaciones_totales = set.union(*datos_por_año.values())
    print(f'\n📊 Total de combinaciones únicas (todos los años): {len(combinaciones_totales):,}')
    
    ausentes_por_año = encontrar_combinaciones_ausentes_por_año(
        datos_por_año, 
        combinaciones_totales
    )
    
    print('\n📋 Resumen de ausencias por año:')
    print('-' * 80)
    for año in sorted(ausentes_por_año.keys()):
        num_ausentes = len(ausentes_por_año[año])
        print(f'  {año}: {num_ausentes:,} combinaciones ausentes')
    
    # -------------------------------------------------------------------------
    # PASO 4: RESUMEN CONSOLIDADO
    # -------------------------------------------------------------------------
    
    print('\n' + '='*80)
    print('📊 PASO 4: RESUMEN CONSOLIDADO')
    print('='*80)
    
    resumen_data = []
    for año in sorted(datos_por_año.keys()):
        num_total = len(datos_por_año[año])
        num_ausentes = len(ausentes_por_año[año])
        num_presentes = num_total
        num_comunes = len(combinaciones_comunes)
        
        resumen_data.append({
            'Año': año,
            'Total_Combinaciones': num_total,
            'Presentes_Todos_Años': num_comunes,
            'Solo_Este_Año': num_total - num_comunes,
            'Ausentes_Este_Año': num_ausentes,
            'Cobertura_%': f'{(num_total / len(combinaciones_totales) * 100):.2f}%'
        })
    
    df_resumen = pd.DataFrame(resumen_data)
    print('\n' + df_resumen.to_string(index=False))
    
    # -------------------------------------------------------------------------
    # PASO 5: GUARDAR RESULTADOS
    # -------------------------------------------------------------------------
    
    guardar_resultados(df_comunes, ausentes_por_año, df_resumen, RUTA_BASE)
    
    # -------------------------------------------------------------------------
    # RESUMEN FINAL
    # -------------------------------------------------------------------------
    
    print('\n' + '='*80)
    print('📈 ESTADÍSTICAS FINALES')
    print('='*80)
    print(f'\n  • Años analizados: {len(datos_por_año)}')
    print(f'  • Combinaciones únicas (unión): {len(combinaciones_totales):,}')
    print(f'  • Combinaciones presentes en TODOS los años: {len(combinaciones_comunes):,}')
    print(f'  • Porcentaje de cobertura común: {(len(combinaciones_comunes)/len(combinaciones_totales)*100):.2f}%')
    
    print('\n' + '='*80)
    print('✅ ANÁLISIS COMPLETADO EXITOSAMENTE')
    print('='*80)
    
    return df_comunes, ausentes_por_año, df_resumen


#_________________________________________________________________________________________________________________________
# EJECUCIÓN
#_________________________________________________________________________________________________________________________

if __name__ == '__main__':
    df_comunes, ausentes_por_año, df_resumen = analizar_municipios_departamentos_por_año()