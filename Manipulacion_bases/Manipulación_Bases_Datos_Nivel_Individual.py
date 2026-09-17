# -*- coding: utf-8 -*-
'''
Created on Sun Dec 14 08:32:33 2025

@author: aarevaloa

'''
#_________________________________________________________________________________________________________________________
# Modulos
#_________________________________________________________________________________________________________________________
import pandas as pd
import numpy as np
from pathlib import Path
import re
import unicodedata

#_________________________________________________________________________________________________________________________
# Ruta
#_________________________________________________________________________________________________________________________
ruta = Path(r'C:\Users\USUARIO\Desktop\Estadística\Maestría en Estadística\Tesis\Escritorio')

#_________________________________________________________________________________________________________________________
# NUEVO: Cargar combinaciones válidas (presentes en todos los años)
#_________________________________________________________________________________________________________________________
print('='*80)
print('CARGANDO COMBINACIONES VÁLIDAS')
print('='*80)

archivo_combinaciones = ruta / 'Resultados/Datos/combinaciones_presentes_todos_años.csv'

if not archivo_combinaciones.exists():
    print(f'\n⚠️  ADVERTENCIA: No se encontró el archivo de combinaciones válidas.')
    print(f'   Esperado en: {archivo_combinaciones}')
    print(f'\n   Por favor, ejecuta primero el script "analizar_municipios_departamentos_por_año.py"')
    print(f"   para generar el archivo de combinaciones válidas.\n")
    raise FileNotFoundError(f"No se encontró: {archivo_combinaciones}")

# Cargar combinaciones válidas
df_combinaciones_validas = pd.read_csv(
    archivo_combinaciones,
    dtype=str,
    encoding='utf-8-sig'
)

print(f'✅ Combinaciones válidas cargadas del archivo: {len(df_combinaciones_validas):,}')
print(f'\n📋 Primeras 5 combinaciones del archivo:')
print(df_combinaciones_validas.head())
print(f'\nEjemplos de valores:')
print(f"  Departamento: '{df_combinaciones_validas['Cod_Departamento'].iloc[0]}'")
print(f"  Municipio: '{df_combinaciones_validas['Cod_Municipio'].iloc[0]}'")

# Crear set de tuplas para filtrado eficiente
combinaciones_validas = set(
    zip(
        df_combinaciones_validas['Cod_Departamento'].astype(str).str.strip(),
        df_combinaciones_validas['Cod_Municipio'].astype(str).str.strip()
    )
)

print(f'\n✅ Set de combinaciones creado: {len(combinaciones_validas):,} combinaciones')
print()

#_________________________________________________________________________________________________________________________
# Importar datos
#_________________________________________________________________________________________________________________________
print('='*80)
print('IMPORTANDO DATOS ICFES')
print('='*80)
print()

base = ruta / 'Datos_originales' / 'Datos_nivel_individual'
años = [17, 18, 19, 20, 21, 22, 23, 24]
semestres = [1, 2]

dfs = []

for añio in años:
    for semestre in semestres:
        archivo = base / f'Examen_Saber_11_20{añio}{semestre}.txt'
        if not archivo.exists():
            print(f'⚠️  No existe: {archivo}')
            continue

        df = pd.read_csv(archivo, sep=';')
        df['anio'] = 2000 + añio
        df['semestre'] = semestre
        dfs.append(df)
        print(f'✅ Cargado: 20{añio}{semestre} ({len(df):,} registros)')

icfes_original = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
print(f"\n📊 Total registros cargados: {len(icfes_original):,}")

#_________________________________________________________________________________________________________________________
# Filtrar datos
#_________________________________________________________________________________________________________________________
print('\n' + '='*80)
print('FILTRANDO DATOS')
print('='*80)
print()

registros_inicial = len(icfes_original)
print(f"Registros iniciales: {registros_inicial:,}")

# Filtro 1: Nacionalidad/País de residencia
icfes_original = icfes_original[
    (icfes_original['estu_nacionalidad'] == 'COLOMBIA') | 
    (icfes_original['estu_pais_reside'] == 'COLOMBIA')
]
print(f"Después de filtro nacionalidad/país: {len(icfes_original):,} ({len(icfes_original)/registros_inicial*100:.2f}%)")

# Filtro 2: Valores no nulos
icfes_original = icfes_original.dropna(
    subset=['estu_cod_reside_depto', 'estu_cod_reside_mcpio', 'punt_global']
).reset_index(drop=True)
print(f"Después de eliminar nulos (depto/mcpio/puntaje): {len(icfes_original):,}")

# NUEVO FILTRO 3: Mantener solo combinaciones presentes en todos los años
print("\n🔍 Aplicando filtro de combinaciones válidas...")
print("-" * 80)

# Mostrar valores ANTES de la conversión
print(f"\n📊 Análisis de datos ANTES del filtro:")
print(f"Tipos de datos originales:")
print(f"  estu_cod_reside_depto: {icfes_original['estu_cod_reside_depto'].dtype}")
print(f"  estu_cod_reside_mcpio: {icfes_original['estu_cod_reside_mcpio'].dtype}")

# Convertir a string y limpiar espacios
icfes_original['estu_cod_reside_depto'] = (
    icfes_original['estu_cod_reside_depto']
    .astype(str)
    .str.strip()
    .str.replace('.0', '', regex=False)  # Por si vienen como float
)

icfes_original['estu_cod_reside_mcpio'] = (
    icfes_original['estu_cod_reside_mcpio']
    .astype(str)
    .str.strip()
    .str.replace('.0', '', regex=False)  # Por si vienen como float
)

print(f"\nEjemplos de códigos en los datos:")
muestra = icfes_original[['estu_cod_reside_depto', 'estu_cod_reside_mcpio']].head(10)
print(muestra.to_string())

# Contar combinaciones únicas ANTES del filtro
combinaciones_antes = icfes_original.groupby(
    ['estu_cod_reside_depto', 'estu_cod_reside_mcpio']
).size().reset_index(name='count')
print(f"\nCombinaciones únicas en los datos: {len(combinaciones_antes):,}")

# Ver cuántas de estas combinaciones están en el set de válidas
combinaciones_en_datos = set(
    zip(
        icfes_original['estu_cod_reside_depto'],
        icfes_original['estu_cod_reside_mcpio']
    )
)

overlap = combinaciones_en_datos & combinaciones_validas
print(f"Combinaciones que coinciden con las válidas: {len(overlap):,}")

if len(overlap) == 0:
    print("\n⚠️  ADVERTENCIA: No hay coincidencias!")
    print("\nEjemplos de combinaciones en archivo de válidas:")
    for i, (d, m) in enumerate(list(combinaciones_validas)[:5]):
        print(f"  {i+1}. Depto: '{d}', Municipio: '{m}'")
    
    print("\nEjemplos de combinaciones en los datos ICFES:")
    for i, (d, m) in enumerate(list(combinaciones_en_datos)[:5]):
        print(f"  {i+1}. Depto: '{d}', Municipio: '{m}'")
    
    print("\n¿Los formatos coinciden? Verificando...")

# Crear columna temporal para filtrado
icfes_original['_combinacion_temp'] = list(
    zip(
        icfes_original['estu_cod_reside_depto'],
        icfes_original['estu_cod_reside_mcpio']
    )
)

# Filtrar solo las combinaciones válidas
registros_antes_filtro = len(icfes_original)

icfes_original = icfes_original[
    icfes_original['_combinacion_temp'].isin(combinaciones_validas)
].copy()

# Eliminar columna temporal
icfes_original = icfes_original.drop(columns=['_combinacion_temp'])

registros_despues_filtro = len(icfes_original)
registros_eliminados = registros_antes_filtro - registros_despues_filtro

print(f"\n✅ Filtrado completado:")
print(f"   • Registros antes: {registros_antes_filtro:,}")
print(f"   • Registros después: {registros_despues_filtro:,}")
print(f"   • Registros eliminados: {registros_eliminados:,} ({registros_eliminados/registros_antes_filtro*100:.2f}%)")

if registros_despues_filtro > 0:
    combinaciones_finales = icfes_original.groupby(
        ['estu_cod_reside_depto', 'estu_cod_reside_mcpio']
    ).ngroups
    print(f"   • Combinaciones únicas restantes: {combinaciones_finales:,}")
else:
    print("\n❌ ERROR: No quedaron registros después del filtro!")
    print("   Revisa que los códigos en el archivo de combinaciones válidas")
    print("   tengan el mismo formato que los códigos en los datos ICFES.")
    raise ValueError("No quedaron registros después del filtro de combinaciones válidas")

#_________________________________________________________________________________________________________________________
# Manipular datos
#_________________________________________________________________________________________________________________________
print("\n" + "="*80)
print("PROCESANDO VARIABLES")
print("="*80)
print()

Variables = ['periodo', 'cole_area_ubicacion', 'cole_bilingue', 'cole_calendario', 'cole_caracter', 'cole_genero', 'cole_naturaleza',
      'estu_cod_reside_depto', 'estu_depto_reside', 'estu_cod_reside_mcpio', 'estu_mcpio_reside', 'estu_dedicacioninternet',
      'estu_dedicacionlecturadiaria', 'estu_genero', 'estu_horassemanatrabaja', 'estu_inse_individual', 'fami_comecarnepescadohuevo',
      'fami_comecerealfrutoslegumbre', 'fami_comelechederivados', 'fami_educacionmadre', 'fami_numlibros', 'fami_tienecomputador', 
      'fami_tieneinternet', 'punt_global']

icfes = icfes_original[Variables].copy()

def slugify(x: str) -> str:
    # quita acentos, reemplaza separadores raros por '_' y deja [A-Z0-9_]
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('ascii')
    x = re.sub(r'[^A-Z0-9]+', '_', x.upper()).strip('_')
    return x

# Cole area ubicacion
s = (icfes['cole_area_ubicacion'].astype('string').str.strip().str.upper())
ubic = s.where(s.isin(['RURAL', 'URBANO']), other=pd.NA).fillna('MISSING')
icfes['ubicacion_cat'] = pd.Categorical(ubic, categories=['URBANO', 'RURAL', 'MISSING'])
X_ubic = pd.get_dummies(icfes['ubicacion_cat'], drop_first=True)
icfes = icfes.join(X_ubic)
icfes = icfes.rename(columns={'RURAL': 'cole_area_ubicacion_RURAL', 'MISSING': 'cole_area_ubicacion_MISSING'})

# Cole bilingue
s = icfes['cole_bilingue'].astype('string').str.strip().str.upper()
cole = s.where(s.isin(['S', 'N']), other=pd.NA).fillna('MISSING')
icfes['cole_bilingue_cat'] = pd.Categorical(cole, categories=['N', 'S', 'MISSING'])
X_cole = pd.get_dummies(
    icfes['cole_bilingue_cat'],
    prefix='cole_bilingue',
    prefix_sep='_',
    drop_first=True)
icfes = icfes.join(X_cole)

# Cole calendario
s = icfes['cole_calendario'].astype('string').str.strip().str.upper()
calen = s.where(s.isin(['A', 'B', 'OTRO']), other=pd.NA).fillna('MISSING')
icfes['cole_calendario_cat'] = pd.Categorical(calen, categories=['A', 'B', 'OTRO', 'MISSING'])
X_calen = pd.get_dummies(
    icfes['cole_calendario_cat'],
    prefix='calen',
    prefix_sep='_',
    drop_first=True)
icfes = icfes.join(X_calen)

# Cole caracter
s = icfes['cole_caracter'].astype('string').str.strip().str.upper()
s_norm = s.map(lambda v: slugify(v) if pd.notna(v) else v)
valid = {'ACADEMICO', 'TECNICO_ACADEMICO', 'TECNICO', 'NO_APLICA'}
carac = s_norm.where(s_norm.isin(valid), other=pd.NA).fillna('MISSING')
icfes['cole_caracter_cat'] = pd.Categorical(carac, categories=['ACADEMICO', 'TECNICO_ACADEMICO', 'TECNICO', 'NO_APLICA', 'MISSING'])
X_carac = pd.get_dummies(
    icfes['cole_caracter_cat'],
    prefix='cole_carac',
    prefix_sep='_',
    drop_first=True)
icfes = icfes.join(X_carac)

# Cole genero
s = icfes['cole_genero'].astype('string').str.strip().str.upper()
gene = s.where(s.isin(['MIXTO', 'FEMENINO', 'MASCULINO']), other=pd.NA).fillna('MISSING')
icfes['cole_genero_cat'] = pd.Categorical(gene, categories=['MIXTO', 'FEMENINO', 'MASCULINO', 'MISSING'])
X_gene = pd.get_dummies(
    icfes['cole_genero_cat'],
    prefix='cole_gen',
    prefix_sep='_',
    drop_first=True)
icfes = icfes.join(X_gene)

# Cole naturaleza
s = icfes['cole_naturaleza'].astype('string').str.strip().str.upper()
s = s.replace({'NO OFICIAL': 'NO_OFICIAL'})
natu = s.where(s.isin(['OFICIAL', 'NO_OFICIAL']), other=pd.NA).fillna('MISSING')
icfes['cole_naturaleza_cat'] = pd.Categorical(natu, categories=['OFICIAL', 'NO_OFICIAL', 'MISSING'])
X_natu = pd.get_dummies(
    icfes['cole_naturaleza_cat'],
    prefix='cole_natu',
    prefix_sep='_',
    drop_first=True)
icfes = icfes.join(X_natu)

# Dedicacion internet
s = (icfes['estu_dedicacioninternet'].astype('string').str.strip().str.upper())
valid = ['NO NAVEGA INTERNET', 'ENTRE 1 Y 3 HORAS', 'ENTRE 30 Y 60 MINUTOS', 'MÁS DE 3 HORAS', '30 MINUTOS O MENOS']
inter = s.where(s.isin(valid), other=pd.NA).fillna('MISSING')
icfes['inter_cat'] = pd.Categorical(inter, categories=['NO NAVEGA INTERNET',
        'ENTRE 1 Y 3 HORAS', 'ENTRE 30 Y 60 MINUTOS', 'MÁS DE 3 HORAS', '30 MINUTOS O MENOS', 'MISSING'], ordered=True)
X_inter = pd.get_dummies(
    icfes['inter_cat'],
    prefix='inter',
    prefix_sep='_',
    drop_first=True)
icfes = icfes.join(X_inter)

# Dedicacion lectura
s = (icfes['estu_dedicacionlecturadiaria'].astype('string').str.strip().str.upper())
valid = ['30 MINUTOS O MENOS', 'ENTRE 30 Y 60 MINUTOS', 'NO LEO POR ENTRETENIMIENTO', 'ENTRE 1 Y 2 HORAS', 'MÁS DE 2 HORAS']
lectu = s.where(s.isin(valid), other=pd.NA).fillna('MISSING')
icfes['lectu_cat'] = pd.Categorical(lectu, categories=['NO LEO POR ENTRETENIMIENTO',
        '30 MINUTOS O MENOS', 'ENTRE 30 Y 60 MINUTOS', 'ENTRE 1 Y 2 HORAS', 'MÁS DE 2 HORAS', 'MISSING'], ordered=True)
X_lectu = pd.get_dummies(
    icfes['lectu_cat'],
    prefix='lectu',
    prefix_sep='_',
    drop_first=True)
icfes = icfes.join(X_lectu)

# Genero estudiante
s = icfes['estu_genero'].astype('string').str.strip().str.upper()
genero = s.where(s.isin(['F', 'M']), other=pd.NA).fillna('MISSING')
icfes['genero_cat'] = pd.Categorical(genero, categories=['M', 'F', 'MISSING'])
X_genero = pd.get_dummies(
    icfes['genero_cat'],
    prefix='genero',
    prefix_sep='_',
    drop_first=True)
icfes = icfes.join(X_genero)

# Horas trabajo
s = icfes['estu_horassemanatrabaja'].astype('string').str.strip().str.upper()
valid = ['0', 'MENOS DE 10 HORAS', 'ENTRE 11 Y 20 HORAS', 'ENTRE 21 Y 30 HORAS', 'MÁS DE 30 HORAS']
trabaja = s.where(s.isin(valid), other=pd.NA).fillna('MISSING')
icfes['trabaja_cat'] = pd.Categorical(
    trabaja,
    categories=['0', 'MENOS DE 10 HORAS', 'ENTRE 11 Y 20 HORAS', 'ENTRE 21 Y 30 HORAS', 'MÁS DE 30 HORAS', 'MISSING'],
    ordered=True)
X_trabaja = pd.get_dummies(icfes['trabaja_cat'], prefix='trabaja', prefix_sep='_', drop_first=True)
icfes = icfes.join(X_trabaja)

# Come carne
s = icfes['fami_comecarnepescadohuevo'].astype('string').str.strip().str.upper()
valid = ['TODOS O CASI TODOS LOS DÍAS', '3 A 5 VECES POR SEMANA', '1 O 2 VECES POR SEMANA', 'NUNCA O RARA VEZ COMEMOS ESO']
come_carne = s.where(s.isin(valid), other=pd.NA).fillna('MISSING')
icfes['come_carne_cat'] = pd.Categorical(
    come_carne,
    categories=['NUNCA O RARA VEZ COMEMOS ESO', 'TODOS O CASI TODOS LOS DÍAS', '3 A 5 VECES POR SEMANA', '1 O 2 VECES POR SEMANA', 'MISSING'],
    ordered=True)
X_carne = pd.get_dummies(icfes['come_carne_cat'], prefix='come_carne', prefix_sep='_', drop_first=True)
icfes = icfes.join(X_carne)

# Come cereal
s = icfes['fami_comecerealfrutoslegumbre'].astype('string').str.strip().str.upper()
valid = ['TODOS O CASI TODOS LOS DÍAS', '3 A 5 VECES POR SEMANA', '1 O 2 VECES POR SEMANA', 'NUNCA O RARA VEZ COMEMOS ESO']
come_cereal = s.where(s.isin(valid), other=pd.NA).fillna('MISSING')
icfes['come_cereal_cat'] = pd.Categorical(
    come_cereal,
    categories=['NUNCA O RARA VEZ COMEMOS ESO', 'TODOS O CASI TODOS LOS DÍAS', '3 A 5 VECES POR SEMANA', '1 O 2 VECES POR SEMANA', 'MISSING'],
    ordered=True)
X_cereal = pd.get_dummies(icfes['come_cereal_cat'], prefix='come_cereal', prefix_sep='_', drop_first=True)
icfes = icfes.join(X_cereal)

# Come leche
s = icfes['fami_comelechederivados'].astype('string').str.strip().str.upper()
valid = ['TODOS O CASI TODOS LOS DÍAS', '3 A 5 VECES POR SEMANA', '1 O 2 VECES POR SEMANA', 'NUNCA O RARA VEZ COMEMOS ESO']
come_leche = s.where(s.isin(valid), other=pd.NA).fillna('MISSING')
icfes['come_leche_cat'] = pd.Categorical(
    come_leche,
    categories=['NUNCA O RARA VEZ COMEMOS ESO', 'TODOS O CASI TODOS LOS DÍAS', '3 A 5 VECES POR SEMANA', '1 O 2 VECES POR SEMANA', 'MISSING'],
    ordered=True)
X_leche = pd.get_dummies(icfes['come_leche_cat'], prefix='come_leche', prefix_sep='_', drop_first=True)
icfes = icfes.join(X_leche)

# Educacion madre
s = (icfes['fami_educacionmadre'].astype('string').str.strip().str.replace(r'\s+', ' ', regex=True).str.upper())
valid = ['SECUNDARIA (BACHILLERATO) COMPLETA', 'SECUNDARIA (BACHILLERATO) INCOMPLETA', 'PRIMARIA INCOMPLETA', 'PRIMARIA COMPLETA', 'TÉCNICA O TECNOLÓGICA INCOMPLETA',
    'TÉCNICA O TECNOLÓGICA COMPLETA', 'EDUCACIÓN PROFESIONAL INCOMPLETA', 'EDUCACIÓN PROFESIONAL COMPLETA', 'POSTGRADO', 'NO SABE', 'NINGUNO', 'NO APLICA']
edu_madre = s.where(s.isin(valid), other=pd.NA).fillna('MISSING')
categories = ['NINGUNO', 'PRIMARIA INCOMPLETA', 'PRIMARIA COMPLETA', 'SECUNDARIA (BACHILLERATO) INCOMPLETA', 'SECUNDARIA (BACHILLERATO) COMPLETA', 'TÉCNICA O TECNOLÓGICA INCOMPLETA',
    'TÉCNICA O TECNOLÓGICA COMPLETA', 'EDUCACIÓN PROFESIONAL INCOMPLETA', 'EDUCACIÓN PROFESIONAL COMPLETA', 'POSTGRADO', 'NO SABE', 'NO APLICA', 'MISSING',]
icfes['edu_madre_cat'] = pd.Categorical(edu_madre, categories=categories, ordered=False)
X_edu_madre = pd.get_dummies(
    icfes['edu_madre_cat'],
    prefix='edu_madre',
    prefix_sep='_',
    drop_first=True)
icfes = icfes.join(X_edu_madre)

# Numero libros
s = (icfes['fami_numlibros'].astype('string').str.strip().str.replace(r'\s+', ' ', regex=True).str.upper())
valid = ['0 A 10 LIBROS', '11 A 25 LIBROS', '26 A 100 LIBROS', 'MÁS DE 100 LIBROS']
libros = s.where(s.isin(valid), other=pd.NA).fillna('MISSING')
categories = ['0 A 10 LIBROS', '11 A 25 LIBROS', '26 A 100 LIBROS', 'MÁS DE 100 LIBROS', 'MISSING']
icfes['libros_casa_cat'] = pd.Categorical(libros, categories=categories, ordered=False)
X_libros = pd.get_dummies(icfes['libros_casa_cat'], prefix='libros_casa', prefix_sep='_', drop_first=True)
icfes = icfes.join(X_libros)

# Tiene computador
s = (icfes['fami_tienecomputador'].astype('string').str.strip().str.upper())
valid = ['NO', 'SI']
compu_cat = s.where(s.isin(valid), other=pd.NA).fillna('MISSING')
categories = ['NO', 'SI', 'MISSING']
icfes['compu_cat'] = pd.Categorical(compu_cat, categories=categories, ordered=False)
X_compu = pd.get_dummies(icfes['compu_cat'], prefix='compu_casa', prefix_sep='_', drop_first=True)
icfes = icfes.join(X_compu)

# Tiene internet
s = (icfes['fami_tieneinternet'].astype('string').str.strip().str.upper())
valid = ['NO', 'SI']
internet_cat = s.where(s.isin(valid), other=pd.NA).fillna('MISSING')
categories = ['NO', 'SI', 'MISSING']
icfes['internet_cat'] = pd.Categorical(internet_cat, categories=categories, ordered=False)
X_inter = pd.get_dummies(icfes['internet_cat'], prefix='internet', prefix_sep='_', drop_first=True)
icfes = icfes.join(X_inter)

print("✅ Variables categóricas procesadas")

#_________________________________________________________________________________________________________________________
# Guardar resultados
#_________________________________________________________________________________________________________________________
print("\n" + "="*80)
print("GUARDANDO RESULTADOS")
print("="*80)
print()

# Variables para análisis descriptivo
variables_para_analisis_descriptivo = [
    'periodo', 'estu_cod_reside_depto', 'estu_depto_reside', 'estu_cod_reside_mcpio','estu_mcpio_reside','punt_global', 'ubicacion_cat', 
    'cole_bilingue_cat', 'cole_calendario_cat', 'cole_caracter_cat', 'cole_genero_cat', 'cole_naturaleza_cat', 'inter_cat', 'lectu_cat', 
    'genero_cat', 'trabaja_cat', 'come_carne_cat', 'come_cereal_cat', 'come_leche_cat', 'edu_madre_cat', 'libros_casa_cat', 'compu_cat', 
    'internet_cat']

salida = Path(ruta) / 'Datos_procesados' / 'Python' / 'ICFES_20172_20242_PARA_ANALISIS_DESCRIPTIVO.csv'
salida.parent.mkdir(parents=True, exist_ok=True)

icfes[variables_para_analisis_descriptivo].to_csv(salida, index=False, encoding='utf-8-sig')
print(f"✅ Archivo guardado: {salida}")
print(f"   Registros: {len(icfes):,}")
print(f"   Combinaciones únicas depto-municipio: {icfes.groupby(['estu_cod_reside_depto', 'estu_cod_reside_mcpio']).ngroups:,}")

# Guardar también versión con TODAS las variables (incluyendo dummies)
salida_completa = Path(ruta) / 'Datos_procesados' / 'Python' / 'ICFES_20172_20242_COMPLETO.csv'
icfes.to_csv(salida_completa, index=False, encoding='utf-8-sig')
print(f"✅ Archivo completo guardado: {salida_completa}")
print(f"   Variables: {len(icfes.columns)}")

variables_modelo = ['periodo', 'estu_cod_reside_depto', 'estu_depto_reside', 'estu_cod_reside_mcpio', 'estu_mcpio_reside',
                    'estu_inse_individual', 'punt_global', 'cole_area_ubicacion_RURAL', 'cole_area_ubicacion_MISSING',
                    'cole_bilingue_S', 'cole_bilingue_MISSING', 'calen_B', 'calen_OTRO', 'calen_MISSING', 'cole_carac_TECNICO_ACADEMICO',
                    'cole_carac_TECNICO', 'cole_carac_NO_APLICA', 'cole_carac_MISSING', 'cole_gen_FEMENINO', 'cole_gen_MASCULINO',
                    'cole_gen_MISSING', 'cole_natu_NO_OFICIAL', 'cole_natu_MISSING', 'inter_ENTRE 1 Y 3 HORAS',
                    'inter_ENTRE 30 Y 60 MINUTOS', 'inter_MÁS DE 3 HORAS', 'inter_30 MINUTOS O MENOS', 'inter_MISSING',  
                    'lectu_30 MINUTOS O MENOS', 'lectu_ENTRE 30 Y 60 MINUTOS', 'lectu_ENTRE 1 Y 2 HORAS', 'lectu_MÁS DE 2 HORAS',
                    'lectu_MISSING', 'genero_F', 'genero_MISSING', 'trabaja_MENOS DE 10 HORAS', 'trabaja_ENTRE 11 Y 20 HORAS',
                    'trabaja_ENTRE 21 Y 30 HORAS', 'trabaja_MÁS DE 30 HORAS', 'trabaja_MISSING', 'come_carne_TODOS O CASI TODOS LOS DÍAS',
                    'come_carne_3 A 5 VECES POR SEMANA', 'come_carne_1 O 2 VECES POR SEMANA', 'come_carne_MISSING', 'come_cereal_TODOS O CASI TODOS LOS DÍAS',
                    'come_cereal_3 A 5 VECES POR SEMANA', 'come_cereal_1 O 2 VECES POR SEMANA', 'come_cereal_MISSING', 'come_leche_TODOS O CASI TODOS LOS DÍAS',
                    'come_leche_3 A 5 VECES POR SEMANA', 'come_leche_1 O 2 VECES POR SEMANA', 'come_leche_MISSING', 'edu_madre_PRIMARIA INCOMPLETA',
                    'edu_madre_PRIMARIA COMPLETA', 'edu_madre_SECUNDARIA (BACHILLERATO) INCOMPLETA', 'edu_madre_SECUNDARIA (BACHILLERATO) COMPLETA',
                    'edu_madre_TÉCNICA O TECNOLÓGICA INCOMPLETA', 'edu_madre_TÉCNICA O TECNOLÓGICA COMPLETA',
                    'edu_madre_EDUCACIÓN PROFESIONAL INCOMPLETA', 'edu_madre_EDUCACIÓN PROFESIONAL COMPLETA', 'edu_madre_POSTGRADO',
                    'edu_madre_NO SABE', 'edu_madre_NO APLICA', 'edu_madre_MISSING', 'libros_casa_11 A 25 LIBROS',
                    'libros_casa_26 A 100 LIBROS', 'libros_casa_MÁS DE 100 LIBROS', 'libros_casa_MISSING', 'compu_casa_SI', 'compu_casa_MISSING',
                    'internet_SI', 'internet_MISSING']

salida_MODELO = Path(ruta) / 'Datos_procesados' / 'Python' / 'ICFES_20172_20242_PARA_MODELO.csv'
salida_MODELO.parent.mkdir(parents=True, exist_ok=True)
icfes[variables_modelo].to_csv(salida_MODELO, index=False, encoding='utf-8-sig')
print(f"✅ Archivo modelo guardado: {salida_MODELO}")
print(f"   Variables: {len(icfes.columns)}")

print("\n" + "="*80)
print("✅ PROCESAMIENTO COMPLETADO")
print("="*80)
print(f"\nRESUMEN:")
print(f"  • Registros finales: {len(icfes):,}")
print(f"  • Combinaciones depto-municipio: {icfes.groupby(['estu_cod_reside_depto', 'estu_cod_reside_mcpio']).ngroups:,} de 1,108 esperadas")
print(f"  • Años incluidos: {sorted(icfes['periodo'].unique())}")
print()