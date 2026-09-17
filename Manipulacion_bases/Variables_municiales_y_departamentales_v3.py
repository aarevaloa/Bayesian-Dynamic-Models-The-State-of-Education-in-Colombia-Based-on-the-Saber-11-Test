# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 10:08:31 2025
@author: aarevaloa

Construcción de variables municipales y departamentales para el modelo
dinámico bayesiano con regularización LASSO - Tesis Maestría en Estadística.

Variable dependiente: Puntaje Saber 11
Niveles jerárquicos: Individuo → Municipio → Departamento
Dimensión temporal: Panel municipio-año (2017-2024)
"""
# =============================================================================
# 1. MÓDULOS
# =============================================================================
import pandas as pd
import numpy as np
import unicodedata
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 2. RUTAS
# =============================================================================
RUTA_BASE = Path(
    r'C:\Users\USUARIO\Desktop\Estadística'
    r'\Maestría en Estadística\Tesis\Escritorio'
)
MUNICIPIOS_BASES = RUTA_BASE / 'Datos_originales' / 'Datos_nivel_municipal'
DEPARTAMENTOS_BASES = RUTA_BASE / 'Datos_originales' / 'Datos_nivel_departamental'

SALIDA = RUTA_BASE / 'Datos_procesados' / 'Python'
SALIDA.mkdir(parents=True, exist_ok=True)

ANIO_INICIO = 2017
ANIO_FIN = 2024

# =============================================================================
# 3. FUNCIONES AUXILIARES
# =============================================================================

def leer_csv_seguro(ruta, **kwargs):
    """Intenta leer un CSV con distintas codificaciones y separadores."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    separators = kwargs.pop('sep', None)
    seps_to_try = [separators] if separators else [',', ';', '\t']
    for enc in encodings:
        for sep in seps_to_try:
            try:
                df = pd.read_csv(ruta, encoding=enc, sep=sep, **kwargs)
                if df.shape[1] > 1:
                    print(f"  ✓ {ruta.name}: {df.shape[0]:,} filas × {df.shape[1]} cols "
                          f"(enc={enc}, sep='{sep}')")
                    return df
            except Exception:
                continue
    raise ValueError(f"No se pudo leer: {ruta.name}")


def limpiar_numerico(serie):
    """Convierte una serie con puntos de miles y comas decimales a numérico."""
    if serie.dtype == 'object':
        return pd.to_numeric(
            serie.str.replace('.', '', regex=False)
                 .str.replace(',', '.', regex=False),
            errors='coerce'
        )
    return pd.to_numeric(serie, errors='coerce')


def estandarizar_codigo(serie, digitos):
    """Estandariza un código DANE: quita .0, espacios, rellena ceros."""
    return (
        serie.astype(str)
        .str.replace('.0', '', regex=False)
        .str.strip()
        .str.zfill(digitos)
    )


def normalizar(texto):
    """Normaliza texto: mayúsculas, sin tildes, sin espacios extra."""
    texto = str(texto).upper().strip()
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    return texto


# =============================================================================
# 4. IMPORTACIÓN Y LIMPIEZA DE DATOS
# =============================================================================
print("=" * 70)
print("IMPORTANDO BASES DE DATOS")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 4.1 HOMICIDIOS
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4.1 Homicidios ──")
homicidios = leer_csv_seguro(MUNICIPIOS_BASES / 'HOMICIDIO_20251222.csv', sep=',')

homicidios['ANIO'] = pd.to_datetime(
    homicidios['FECHA HECHO'], dayfirst=True, errors='coerce'
).dt.year
homicidios['CANTIDAD'] = limpiar_numerico(homicidios['CANTIDAD'])
homicidios['COD_DEPTO'] = estandarizar_codigo(homicidios['COD_DEPTO'], 2)
homicidios['COD_MUNI'] = estandarizar_codigo(homicidios['COD_MUNI'], 5)
homicidios = homicidios[homicidios['ANIO'].between(ANIO_INICIO, ANIO_FIN)].reset_index(drop=True)
print(f"  {homicidios.shape[0]:,} registros ({ANIO_INICIO}-{ANIO_FIN})")

# ─────────────────────────────────────────────────────────────────────────────
# 4.2 PROYECCIONES DE POBLACIÓN
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4.2 Proyecciones de Población ──")
archivo_pob = MUNICIPIOS_BASES / 'PPED-AreaMun-2018-2042_VP(PobMunicipalxÁrea).csv'

poblacion = None
# Intentar como Excel
for skip in [7, 6, 8, 5, 0]:
    try:
        poblacion = pd.read_excel(archivo_pob, skiprows=skip)
        if poblacion.shape[1] > 3:
            print(f"  ✓ Leído como Excel con skiprows={skip}: {poblacion.shape}")
            break
        else:
            poblacion = None
    except Exception:
        continue
# Si Excel falla, intentar como CSV
if poblacion is None:
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        for sep in [';', ',', '\t']:
            for skip in [7, 6, 8, 5, 0]:
                try:
                    poblacion = pd.read_csv(
                        archivo_pob, encoding=enc, sep=sep,
                        skiprows=skip, on_bad_lines='skip'
                    )
                    if poblacion.shape[1] > 3:
                        print(f"  ✓ Leído como CSV (enc={enc}, sep='{sep}', "
                              f"skiprows={skip}): {poblacion.shape}")
                        break
                    else:
                        poblacion = None
                except Exception:
                    continue
                if poblacion is not None:
                    break
            if poblacion is not None:
                break

# Normalizar nombres de columnas con problemas de encoding
poblacion.columns = [
    c.encode('latin-1').decode('utf-8') if isinstance(c, str) else c
    for c in poblacion.columns
]
# Renombrar
poblacion = poblacion.rename(columns={
    'DP': 'COD_DEPTO', 'DPNOM': 'DEPARTAMENTO', 'MPIO': 'COD_MUNI',
    'DPMP': 'MUNICIPIO', 'AÑO': 'ANIO', 'TOTAL': 'POBLACION'
})
# Estandarizar códigos
poblacion['COD_MUNI'] = estandarizar_codigo(poblacion['COD_MUNI'], 5)
poblacion['COD_DEPTO'] = estandarizar_codigo(poblacion['COD_DEPTO'], 2)
# Limpiar numéricos y filtrar
poblacion['POBLACION'] = limpiar_numerico(poblacion['POBLACION'])
poblacion = poblacion.dropna(subset=['ANIO', 'POBLACION', 'COD_MUNI'])
poblacion['ANIO'] = poblacion['ANIO'].astype(int)
poblacion = poblacion[poblacion['ANIO'].between(ANIO_INICIO, ANIO_FIN)]

# Separar por área geográfica
col_area = [c for c in poblacion.columns if 'REA' in c.upper()]
if col_area:
    col_area = col_area[0]
    pob_total = poblacion[
        poblacion[col_area].str.contains('Total', case=False, na=False)
    ].reset_index(drop=True)
    pob_urbana = poblacion[
        poblacion[col_area].str.contains('Cabecera', case=False, na=False)
    ].reset_index(drop=True)
else:
    pob_total = poblacion.copy().reset_index(drop=True)
    pob_urbana = None

# Retroproyectar 2017 (la base empieza en 2018)
def retroproyectar_2017(df):
    """Estima población 2017 usando tasa de cambio 2018→2019."""
    d2018 = df[df['ANIO'] == 2018][['COD_DEPTO', 'COD_MUNI', 'POBLACION']].copy()
    d2019 = df[df['ANIO'] == 2019][['COD_MUNI', 'POBLACION']].copy()
    retro = d2018.merge(d2019, on='COD_MUNI', suffixes=('_2018', '_2019'))
    retro['TASA_CREC'] = retro['POBLACION_2019'] / retro['POBLACION_2018']
    retro['POBLACION'] = (retro['POBLACION_2018'] / retro['TASA_CREC']).round(0)
    retro['ANIO'] = 2017
    retro = retro[['COD_DEPTO', 'COD_MUNI', 'ANIO', 'POBLACION']]
    result = pd.concat([retro, df], ignore_index=True)
    return result.sort_values(['COD_MUNI', 'ANIO']).reset_index(drop=True), retro.shape[0]

pob_total, n_retro = retroproyectar_2017(pob_total)
print(f"  ✓ 2017 retroproyectado: {n_retro} municipios")

if pob_urbana is not None:
    pob_urbana, _ = retroproyectar_2017(pob_urbana)

print(f"  ✓ pob_total: {pob_total.shape[0]:,} filas")
if pob_urbana is not None:
    print(f"  ✓ pob_urbana: {pob_urbana.shape[0]:,} filas")

# ─────────────────────────────────────────────────────────────────────────────
# 4.3 VÍCTIMAS DEL CONFLICTO ARMADO
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4.3 Víctimas del conflicto ──")
cols_victimas = ['COD_ESTADO_DEPTO', 'COD_CIUDAD_MUNI', 'VIGENCIA', 'PER_OCU']
victimas = leer_csv_seguro(
    MUNICIPIOS_BASES / 'VICTIMAS_POR_AÑOS_MUNICIPAL_20251222.csv',
    usecols=cols_victimas,
)
victimas['VIGENCIA'] = pd.to_numeric(victimas['VIGENCIA'], errors='coerce')
victimas['PER_OCU'] = pd.to_numeric(victimas['PER_OCU'], errors='coerce')
victimas = victimas[victimas['VIGENCIA'].between(ANIO_INICIO, ANIO_FIN)].copy()
victimas = victimas.rename(columns={
    'COD_ESTADO_DEPTO': 'COD_DEPTO',
    'COD_CIUDAD_MUNI': 'COD_MUNI',
    'VIGENCIA': 'ANIO'
})
victimas['COD_MUNI'] = estandarizar_codigo(victimas['COD_MUNI'], 5)
victimas['COD_DEPTO'] = estandarizar_codigo(victimas['COD_DEPTO'], 2)

victimas_mun = (
    victimas
    .groupby(['COD_DEPTO', 'COD_MUNI', 'ANIO'], as_index=False)['PER_OCU']
    .sum()
    .rename(columns={'PER_OCU': 'VICTIMAS_TOTAL'})
)
print(f"  ✓ {victimas_mun.shape[0]:,} municipios-año")

# ─────────────────────────────────────────────────────────────────────────────
# 4.4 MATRÍCULAS EDUCACIÓN SUPERIOR
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4.4 Matrículas educativas ──")
matriculas = leer_csv_seguro(
    MUNICIPIOS_BASES / 'MEN_ESTADISTICAS_MATRICULA_POR_MUNICIPIOS_ES_20251222.csv'
)
matriculas = matriculas.rename(columns={
    'AÑO': 'ANIO', 'Código delDepartamento': 'COD_DEPTO',
    'Código delMunicipio': 'COD_MUNI', 'IES CON OFERTA': 'IES_CON_OFERTA'
})
matriculas['COD_MUNI'] = estandarizar_codigo(matriculas['COD_MUNI'], 5)
matriculas['COD_DEPTO'] = estandarizar_codigo(matriculas['COD_DEPTO'], 2)

cols_mat = ['TECNICA PROFESIONAL', 'TECNOLOGICA', 'UNIVERSITARIA',
            'ESPECIALIZACION', 'MAESTRIA', 'DOCTORADO']
for col in cols_mat:
    matriculas[col] = limpiar_numerico(matriculas[col])
matriculas['MATRICULA_ES_TOTAL'] = matriculas[cols_mat].sum(axis=1)
matriculas = matriculas[matriculas['ANIO'].between(ANIO_INICIO, ANIO_FIN)].reset_index(drop=True)

matriculas_mun = (
    matriculas
    .groupby(['COD_DEPTO', 'COD_MUNI', 'ANIO'], as_index=False)
    .agg({'MATRICULA_ES_TOTAL': 'sum', 'IES_CON_OFERTA': 'max'})
)
print(f"  ✓ {matriculas_mun.shape[0]:,} municipios-año")

# ─────────────────────────────────────────────────────────────────────────────
# 4.5 NBI (Necesidades Básicas Insatisfechas) - CENSO 2018
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4.5 NBI ──")
nbi_raw = leer_csv_seguro(
    MUNICIPIOS_BASES / 'CNPV-2018-NBI(Municipios).csv',
    header=None, skiprows=9
)
nbi_cols = [
    'COD_DEPTO', 'DEPARTAMENTO', 'COD_MUN_CORTO', 'MUNICIPIO',
    'PROP_NBI', 'PROP_MISERIA', 'COMP_VIVIENDA_INADECUADA',
    'COMP_SERVICIOS_INADECUADOS', 'COMP_HACINAMIENTO',
    'COMP_INASISTENCIA_ESCOLAR', 'COMP_DEPENDENCIA_ECONOMICA'
]
nbi_raw.columns = nbi_cols[:nbi_raw.shape[1]] if nbi_raw.shape[1] < len(nbi_cols) \
    else nbi_cols + [f'COL_EXTRA_{i}' for i in range(nbi_raw.shape[1] - len(nbi_cols))]

nbi = nbi_raw.dropna(subset=['COD_DEPTO', 'COD_MUN_CORTO']).copy()
nbi['COD_DEPTO'] = estandarizar_codigo(nbi['COD_DEPTO'], 2)
nbi['COD_MUN_CORTO'] = estandarizar_codigo(nbi['COD_MUN_CORTO'], 3)
nbi['COD_MUNI'] = nbi['COD_DEPTO'] + nbi['COD_MUN_CORTO']

for col in ['PROP_NBI', 'COMP_INASISTENCIA_ESCOLAR']:
    nbi[col] = limpiar_numerico(nbi[col])

nbi = nbi[nbi['COD_DEPTO'] != '00'].reset_index(drop=True)
nbi_mun = nbi[['COD_DEPTO', 'COD_MUNI', 'PROP_NBI', 'COMP_INASISTENCIA_ESCOLAR']].copy()
print(f"  ✓ {nbi_mun.shape[0]} municipios")

# ─────────────────────────────────────────────────────────────────────────────
# 4.6 TRANSFERENCIAS SGP
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4.6 SGP ──")
lista_sgp = []
for anio in range(ANIO_INICIO, ANIO_FIN + 1):
    archivo_sgp = MUNICIPIOS_BASES / f'ResumenDsitribucionSGP_{anio}.xlsx'
    if archivo_sgp.exists():
        try:
            df_temp = pd.read_excel(
                archivo_sgp, sheet_name='Datos Reporte SGP Detalle', skiprows=13
            )
            df_temp = df_temp.dropna(how='all')
            if 'Tipo Entidad' in df_temp.columns:
                df_temp = df_temp[df_temp['Tipo Entidad'] == 'Municipio'].copy()
            df_temp['ANIO'] = anio
            lista_sgp.append(df_temp)
            print(f"  ✓ {anio}: {df_temp.shape[0]:,} filas")
        except Exception as e:
            try:
                df_temp = pd.read_excel(archivo_sgp, sheet_name=1, skiprows=13)
                df_temp = df_temp.dropna(how='all')
                if 'Tipo Entidad' in df_temp.columns:
                    df_temp = df_temp[df_temp['Tipo Entidad'] == 'Municipio'].copy()
                df_temp['ANIO'] = anio
                lista_sgp.append(df_temp)
                print(f"  ✓ {anio} (hoja por índice)")
            except Exception as e2:
                print(f"  ✗ {anio}: {e2}")
    else:
        print(f"  ⚠ No encontrado: {archivo_sgp.name}")

if lista_sgp:
    cols_comunes = set(lista_sgp[0].columns)
    for df in lista_sgp[1:]:
        cols_comunes &= set(df.columns)
    cols_ordenadas = [c for c in lista_sgp[0].columns if c in cols_comunes]
    sgp = pd.concat([df[cols_ordenadas] for df in lista_sgp], ignore_index=True)
    print(f"  ✓ SGP consolidado: {sgp.shape[0]:,} filas ({len(lista_sgp)} años)")
else:
    sgp = pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# 4.7 DIVIPOLA DEPARTAMENTOS (para merges departamentales)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4.7 DIVIPOLA Departamentos ──")
divipola = pd.read_excel(DEPARTAMENTOS_BASES / 'DIVIPOLA_Departamentos.xlsx', skiprows=9)
divipola = divipola.rename(columns={
    divipola.columns[0]: 'COD_DEPTO',
    divipola.columns[1]: 'DEPARTAMENTO_DIVIPOLA'
})
divipola['COD_DEPTO'] = estandarizar_codigo(divipola['COD_DEPTO'], 2)
divipola['DEPTO_NORM'] = divipola['DEPARTAMENTO_DIVIPOLA'].apply(normalizar)
print(f"  ✓ {divipola.shape[0]} departamentos")

# Correcciones manuales para nombres que no hacen match
CORRECCIONES_DEPTO = {
    'Bogotá D.C.': '11',
    'San Andrés': '88',
}

# ─────────────────────────────────────────────────────────────────────────────
# 4.8 PIB DEPARTAMENTAL (Precios constantes, base 2015)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4.8 PIB Departamental ──")
pib_raw = pd.read_excel(
    DEPARTAMENTOS_BASES / 'anex-PIBDep-TotalDep-2024pr.xlsx',
    sheet_name='Cuadro 2', header=None
)
# Buscar encabezados
fila_header = None
col_dept = None
for i, row in pib_raw.iterrows():
    for j, val in enumerate(row):
        if str(val).strip().upper() == 'DEPARTAMENTOS':
            fila_header = i
            col_dept = j
            break
    if fila_header is not None:
        break

# Detectar fin del primer cuadro
fila_vals = pib_raw.iloc[fila_header].astype(str).str.upper()
apariciones_codigo = [j for j, v in enumerate(fila_vals)
                      if 'DIGO' in v and 'DEPARTAMENTO' in v]
if len(apariciones_codigo) < 2 and fila_header > 0:
    fila_anterior = pib_raw.iloc[fila_header - 1].astype(str).str.upper()
    apariciones_codigo = [j for j, v in enumerate(fila_anterior)
                          if 'DIGO' in v and 'DEPARTAMENTO' in v]
if len(apariciones_codigo) >= 2:
    col_fin = apariciones_codigo[1]
else:
    col_fin = pib_raw.shape[1]
    for j in range(col_dept + 5, pib_raw.shape[1]):
        val = str(pib_raw.iloc[fila_header, j]).strip().upper()
        if val == 'DEPARTAMENTOS' or 'DIGO' in val:
            col_fin = j
            break

pib_raw = pib_raw.iloc[:, :col_fin]
pib_raw.columns = pib_raw.iloc[fila_header]
pib_raw = pib_raw.iloc[fila_header + 1:].reset_index(drop=True)
pib_raw.columns = ['COD_DEPTO', 'DEPARTAMENTO'] + list(pib_raw.columns[2:])

pib_raw = pib_raw.dropna(subset=['DEPARTAMENTO'])
pib_raw = pib_raw[~pib_raw['DEPARTAMENTO'].astype(str).str.contains(
    'COLOMBIA|Fuente|preliminar|Actualizado', case=False, na=False)]
pib_raw = pib_raw[pib_raw['COD_DEPTO'].notna()]

cols_anios = []
for c in pib_raw.columns[2:]:
    limpio = str(c).strip().replace('.0', '').replace('*', '').replace('p', '').replace('r', '')
    if limpio.isdigit() and len(limpio) == 4:
        cols_anios.append(c)

pib = pib_raw.melt(
    id_vars=['COD_DEPTO', 'DEPARTAMENTO'], value_vars=cols_anios,
    var_name='ANIO', value_name='PIB_CONSTANTE'
)
pib['COD_DEPTO'] = estandarizar_codigo(pib['COD_DEPTO'], 2)
pib['ANIO'] = pd.to_numeric(
    pib['ANIO'].astype(str)
    .str.replace('.0', '', regex=False)
    .str.replace('*', '', regex=False)
    .str.replace('p', '', regex=False)
    .str.replace('r', '', regex=False),
    errors='coerce'
).astype(int)
pib['PIB_CONSTANTE'] = pd.to_numeric(pib['PIB_CONSTANTE'], errors='coerce')
pib = pib[pib['ANIO'].between(ANIO_INICIO, ANIO_FIN)].reset_index(drop=True)

pib = pib.sort_values(['COD_DEPTO', 'ANIO'])
pib['CREC_PIB_REAL'] = pib.groupby('COD_DEPTO')['PIB_CONSTANTE'].pct_change() * 100

pib = pib[['COD_DEPTO', 'ANIO', 'PIB_CONSTANTE', 'CREC_PIB_REAL']]
print(f"  ✓ PIB: {pib.shape[0]} obs ({pib['COD_DEPTO'].nunique()} deptos)")

# ─────────────────────────────────────────────────────────────────────────────
# 4.9 MERCADO LABORAL DEPARTAMENTAL (GEIH)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4.9 Mercado Laboral ──")
ml_raw = pd.read_csv(
    DEPARTAMENTOS_BASES / 'anex-GEIHDepartamentos-2024(Departamentos anual).csv',
    header=None, sep=';', encoding='utf-8-sig'
)
INDICADORES_INTERES = {
    'Tasa de Desocupación (TD)': 'TASA_DESEMPLEO',
    'Tasa de Subocupación (TS)': 'TASA_SUBOCUPACION',
}
registros = []
depto_actual = None
anios = None
for i, row in ml_raw.iterrows():
    valor_col0 = str(row.iloc[0]).strip()
    if valor_col0 == 'Concepto':
        anios_raw = row.iloc[1:].tolist()
        anios = [int(float(a)) for a in anios_raw if str(a).replace('.0', '').strip().isdigit()]
        for k in range(i - 1, max(i - 5, 0), -1):
            candidato = str(ml_raw.iloc[k, 0]).strip()
            if candidato not in ['nan', '', 'None']:
                depto_actual = candidato
                break
        continue
    if depto_actual and anios:
        for indicador, nombre_var in INDICADORES_INTERES.items():
            if indicador.lower() in valor_col0.lower():
                valores = row.iloc[1:len(anios) + 1].tolist()
                for anio, valor in zip(anios, valores):
                    registros.append({
                        'DEPARTAMENTO': depto_actual, 'ANIO': anio,
                        'VARIABLE': nombre_var, 'VALOR': valor
                    })
                break

mercado_laboral = pd.DataFrame(registros)
mercado_laboral['VALOR'] = limpiar_numerico(mercado_laboral['VALOR'])
mercado_laboral = mercado_laboral.pivot_table(
    index=['DEPARTAMENTO', 'ANIO'], columns='VARIABLE', values='VALOR'
).reset_index()
mercado_laboral.columns.name = None
mercado_laboral = mercado_laboral[
    mercado_laboral['ANIO'].between(ANIO_INICIO, ANIO_FIN)
].reset_index(drop=True)

# Agregar código departamental
mercado_laboral['DEPTO_NORM'] = mercado_laboral['DEPARTAMENTO'].apply(normalizar)
mercado_laboral = mercado_laboral.merge(
    divipola[['COD_DEPTO', 'DEPTO_NORM']], on='DEPTO_NORM', how='left'
)
for depto, codigo in CORRECCIONES_DEPTO.items():
    mercado_laboral.loc[mercado_laboral['DEPARTAMENTO'] == depto, 'COD_DEPTO'] = codigo
mercado_laboral = mercado_laboral.drop(columns=['DEPTO_NORM'])

sin_match = mercado_laboral[mercado_laboral['COD_DEPTO'].isna()]['DEPARTAMENTO'].unique()
if len(sin_match) > 0:
    print(f"  ⚠ Sin código: {sin_match}")

mercado_laboral = mercado_laboral[['COD_DEPTO', 'ANIO', 'TASA_DESEMPLEO', 'TASA_SUBOCUPACION']]
print(f"  ✓ Mercado laboral: {mercado_laboral.shape[0]} obs "
      f"({mercado_laboral['COD_DEPTO'].nunique()} deptos)")

# ─────────────────────────────────────────────────────────────────────────────
# 4.10 POBREZA MULTIDIMENSIONAL DEPARTAMENTAL (IPM)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4.10 IPM ──")
pm_raw = pd.read_csv(
    DEPARTAMENTOS_BASES / 'anex-PMultidimensional-Departamental-2024(IPM_Departamentos).csv',
    header=None, sep=';', encoding='utf-8-sig'
)
fila_anios = pm_raw.iloc[11, 1:].tolist()
fila_categorias = pm_raw.iloc[12, 1:].tolist()

anio_actual = None
cols_total = {}
for j, (anio_val, cat_val) in enumerate(zip(fila_anios, fila_categorias), start=1):
    if pd.notna(anio_val):
        anio_limpio = str(anio_val).replace('*', '').strip()
        if anio_limpio.isdigit():
            anio_actual = int(anio_limpio)
    if anio_actual and pd.notna(cat_val) and str(cat_val).strip() == 'Total':
        cols_total[anio_actual] = j

datos_pm = pm_raw.iloc[13:].copy()
datos_pm = datos_pm.dropna(subset=[0])
datos_pm = datos_pm[~datos_pm[0].astype(str).str.contains(
    'Fuente|Nota|Incidencia|Cifras|Departamento|Datos expandidos|Actualizado',
    case=False, na=False
)]

registros_pm = []
for _, fila in datos_pm.iterrows():
    depto = str(fila.iloc[0]).strip()
    for anio, col_idx in cols_total.items():
        registros_pm.append({'DEPARTAMENTO': depto, 'ANIO': anio, 'IPM': fila.iloc[col_idx]})

pobreza_multi = pd.DataFrame(registros_pm)
pobreza_multi['IPM'] = limpiar_numerico(pobreza_multi['IPM'])
pobreza_multi = pobreza_multi[
    pobreza_multi['ANIO'].between(ANIO_INICIO, ANIO_FIN)
].reset_index(drop=True)

# Agregar código departamental
pobreza_multi['DEPTO_NORM'] = pobreza_multi['DEPARTAMENTO'].apply(normalizar)
pobreza_multi = pobreza_multi.merge(
    divipola[['COD_DEPTO', 'DEPTO_NORM']], on='DEPTO_NORM', how='left'
)
for depto, codigo in CORRECCIONES_DEPTO.items():
    pobreza_multi.loc[pobreza_multi['DEPARTAMENTO'] == depto, 'COD_DEPTO'] = codigo
pobreza_multi = pobreza_multi.drop(columns=['DEPTO_NORM'])

sin_match = pobreza_multi[pobreza_multi['COD_DEPTO'].isna()]['DEPARTAMENTO'].unique()
if len(sin_match) > 0:
    print(f"  ⚠ Sin código: {sin_match}")

pobreza_multi = pobreza_multi[['COD_DEPTO', 'ANIO', 'IPM']]
print(f"  ✓ IPM: {pobreza_multi.shape[0]} obs ({pobreza_multi['COD_DEPTO'].nunique()} deptos)")


# =============================================================================
# 5. CONSTRUCCIÓN DE VARIABLES
# =============================================================================
print("\n" + "=" * 70)
print("CONSTRUYENDO VARIABLES")
print("=" * 70)

pob_merge = pob_total[['COD_MUNI', 'ANIO', 'POBLACION']].copy()

# ─────────────────────────────────────────────────────────────────────────────
# 5.1 TASA DE HOMICIDIOS POR 100,000 HAB (municipio-año)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 5.1 Tasa de homicidios ──")

hom_total = (
    homicidios
    .groupby(['COD_DEPTO', 'COD_MUNI', 'ANIO'], as_index=False)['CANTIDAD']
    .sum()
    .rename(columns={'CANTIDAD': 'HOMICIDIOS_TOTAL'})
)

hom_zona = (
    homicidios
    .groupby(['COD_DEPTO', 'COD_MUNI', 'ANIO', 'ZONA'], as_index=False)['CANTIDAD']
    .sum()
    .pivot_table(
        index=['COD_DEPTO', 'COD_MUNI', 'ANIO'],
        columns='ZONA', values='CANTIDAD', fill_value=0
    )
    .reset_index()
)
hom_zona.columns.name = None
col_rename_zona = {}
for col in hom_zona.columns:
    if 'URBAN' in str(col).upper():
        col_rename_zona[col] = 'HOMICIDIOS_URBANO'
    elif 'RURAL' in str(col).upper() or 'CENTRO' in str(col).upper():
        col_rename_zona[col] = 'HOMICIDIOS_RURAL'
hom_zona = hom_zona.rename(columns=col_rename_zona)

hom_mun = hom_total.merge(hom_zona, on=['COD_DEPTO', 'COD_MUNI', 'ANIO'], how='left')
hom_mun = hom_mun.merge(pob_merge, on=['COD_MUNI', 'ANIO'], how='left')

hom_mun['TASA_HOMICIDIOS'] = hom_mun['HOMICIDIOS_TOTAL'] / hom_mun['POBLACION'] * 100_000
if 'HOMICIDIOS_URBANO' in hom_mun.columns:
    hom_mun['PROP_HOMICIDIOS_URBANO'] = (
        hom_mun['HOMICIDIOS_URBANO'] / hom_mun['HOMICIDIOS_TOTAL']
    ).fillna(0)

hom_mun = hom_mun[['COD_MUNI', 'ANIO', 'TASA_HOMICIDIOS', 'PROP_HOMICIDIOS_URBANO']]
print(f"  ✓ {hom_mun.shape[0]:,} municipios-año")

# ─────────────────────────────────────────────────────────────────────────────
# 5.2 TASA DE VÍCTIMAS POR 100,000 HAB (municipio-año)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 5.2 Tasa de víctimas ──")

victimas_mun = victimas_mun.merge(pob_merge, on=['COD_MUNI', 'ANIO'], how='left')
victimas_mun['TASA_VICTIMAS'] = victimas_mun['VICTIMAS_TOTAL'] / victimas_mun['POBLACION'] * 100_000
victimas_mun = victimas_mun[['COD_MUNI', 'ANIO', 'TASA_VICTIMAS']]
print(f"  ✓ {victimas_mun.shape[0]:,} municipios-año")

# ─────────────────────────────────────────────────────────────────────────────
# 5.3 OFERTA DE EDUCACIÓN SUPERIOR (municipio-año)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 5.3 Oferta educación superior ──")

matriculas_mun = matriculas_mun.merge(pob_merge, on=['COD_MUNI', 'ANIO'], how='left')
matriculas_mun['TASA_MATRICULA_ES'] = (
    matriculas_mun['MATRICULA_ES_TOTAL'] / matriculas_mun['POBLACION'] * 1_000
)
matriculas_mun = matriculas_mun[['COD_MUNI', 'ANIO', 'TASA_MATRICULA_ES', 'IES_CON_OFERTA']]
print(f"  ✓ {matriculas_mun.shape[0]:,} municipios-año")

# ─────────────────────────────────────────────────────────────────────────────
# 5.4 SGP PER CÁPITA (municipio-año)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 5.4 SGP per cápita ──")

if not sgp.empty:
    sgp = sgp.rename(columns={
        'Codigo DANE Entidad': 'COD_MUNI',
        'Codigo DANE Departamento': 'COD_DEPTO',
        'Educación': 'SGP_EDUCACION',
        'Total': 'SGP_TOTAL'
    })
    sgp['COD_MUNI'] = estandarizar_codigo(sgp['COD_MUNI'], 5)
    for col in ['SGP_EDUCACION', 'SGP_TOTAL']:
        sgp[col] = limpiar_numerico(sgp[col])

    sgp_mun = sgp[['COD_MUNI', 'ANIO', 'SGP_EDUCACION', 'SGP_TOTAL']].copy()
    sgp_mun = sgp_mun.merge(pob_merge, on=['COD_MUNI', 'ANIO'], how='left')
    sgp_mun['SGP_EDUC_PC'] = sgp_mun['SGP_EDUCACION'] / sgp_mun['POBLACION']
    sgp_mun['SGP_TOTAL_PC'] = sgp_mun['SGP_TOTAL'] / sgp_mun['POBLACION']
    sgp_mun = sgp_mun[['COD_MUNI', 'ANIO', 'SGP_EDUC_PC', 'SGP_TOTAL_PC']]
    print(f"  ✓ {sgp_mun.shape[0]:,} municipios-año")
else:
    sgp_mun = pd.DataFrame(columns=['COD_MUNI', 'ANIO', 'SGP_EDUC_PC', 'SGP_TOTAL_PC'])
    print("  ⚠ SGP vacío")

# ─────────────────────────────────────────────────────────────────────────────
# 5.5 PROPORCIÓN URBANA Y LOG POBLACIÓN (municipio-año)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 5.5 Estructura poblacional ──")

if pob_urbana is not None and 'COD_MUNI' in pob_total.columns:
    pob_urb_merge = pob_urbana[['COD_MUNI', 'ANIO', 'POBLACION']].rename(
        columns={'POBLACION': 'POB_URBANA'}
    )
    pob_estructura = pob_total[['COD_DEPTO', 'COD_MUNI', 'ANIO', 'POBLACION']].merge(
        pob_urb_merge, on=['COD_MUNI', 'ANIO'], how='left'
    )
    pob_estructura['PROP_URBANA'] = pob_estructura['POB_URBANA'] / pob_estructura['POBLACION']
    pob_estructura['LOG_POBLACION'] = np.log(pob_estructura['POBLACION'].clip(lower=1))
    pob_estructura = pob_estructura[['COD_DEPTO', 'COD_MUNI', 'ANIO',
                                      'POBLACION', 'PROP_URBANA', 'LOG_POBLACION']]
    print(f"  ✓ {pob_estructura.shape[0]:,} municipios-año")
else:
    pob_estructura = pob_total[['COD_DEPTO', 'COD_MUNI', 'ANIO', 'POBLACION']].copy()
    pob_estructura['LOG_POBLACION'] = np.log(pob_estructura['POBLACION'].clip(lower=1))
    print("  ⚠ Sin proporción urbana, solo LOG_POBLACION")


# =============================================================================
# 6. CONSOLIDACIÓN DEL PANEL
# =============================================================================
print("\n" + "=" * 70)
print("CONSOLIDANDO PANEL MUNICIPAL")
print("=" * 70)

# ── 6.1 Esqueleto: estructura poblacional como base ──
panel = pob_estructura.copy()
print(f"\n  Base del panel: {panel.shape[0]:,} obs "
      f"({panel['COD_MUNI'].nunique()} municipios × {panel['ANIO'].nunique()} años)")

# ── 6.2 Variables municipales dinámicas (municipio-año) ──
print("\n  Agregando variables municipales...")

# Homicidios
panel = panel.merge(hom_mun, on=['COD_MUNI', 'ANIO'], how='left')
print(f"    + TASA_HOMICIDIOS, PROP_HOMICIDIOS_URBANO")

# Víctimas
panel = panel.merge(victimas_mun, on=['COD_MUNI', 'ANIO'], how='left')
print(f"    + TASA_VICTIMAS")

# Matrícula educación superior
panel = panel.merge(matriculas_mun, on=['COD_MUNI', 'ANIO'], how='left')
print(f"    + TASA_MATRICULA_ES, IES_CON_OFERTA")

# SGP per cápita
if not sgp_mun.empty:
    panel = panel.merge(sgp_mun, on=['COD_MUNI', 'ANIO'], how='left')
    print(f"    + SGP_EDUC_PC, SGP_TOTAL_PC")

# ── 6.3 Variables municipales estáticas (NBI, Censo 2018) ──
print("\n  Agregando NBI (estático, Censo 2018)...")
panel = panel.merge(nbi_mun[['COD_MUNI', 'PROP_NBI', 'COMP_INASISTENCIA_ESCOLAR']],
                     on='COD_MUNI', how='left')
print(f"    + PROP_NBI, COMP_INASISTENCIA_ESCOLAR")

# ── 6.4 Variables departamentales (departamento-año) ──
print("\n  Agregando variables departamentales...")

# PIB
panel = panel.merge(pib, on=['COD_DEPTO', 'ANIO'], how='left')
print(f"    + PIB_CONSTANTE, CREC_PIB_REAL")

# Mercado laboral
panel = panel.merge(mercado_laboral, on=['COD_DEPTO', 'ANIO'], how='left')
print(f"    + TASA_DESEMPLEO, TASA_SUBOCUPACION")

# IPM
panel = panel.merge(pobreza_multi, on=['COD_DEPTO', 'ANIO'], how='left')
print(f"    + IPM")

# ── 6.5 Municipios sin homicidios/víctimas registrados = 0, no NA ──
print("\n  Imputando ceros en municipios sin registros de violencia...")
panel['TASA_HOMICIDIOS'] = panel['TASA_HOMICIDIOS'].fillna(0)
panel['PROP_HOMICIDIOS_URBANO'] = panel['PROP_HOMICIDIOS_URBANO'].fillna(0)
panel['TASA_VICTIMAS'] = panel['TASA_VICTIMAS'].fillna(0)
panel['TASA_MATRICULA_ES'] = panel['TASA_MATRICULA_ES'].fillna(0)
panel['IES_CON_OFERTA'] = panel['IES_CON_OFERTA'].fillna(0)


# =============================================================================
# 7. RESUMEN Y EXPORTACIÓN
# =============================================================================
print("\n" + "=" * 70)
print("RESUMEN DEL PANEL")
print("=" * 70)

print(f"\n  Dimensión: {panel.shape[0]:,} filas × {panel.shape[1]} columnas")
print(f"  Municipios: {panel['COD_MUNI'].nunique()}")
print(f"  Años: {panel['ANIO'].min()} - {panel['ANIO'].max()}")
print(f"  Departamentos: {panel['COD_DEPTO'].nunique()}")

print(f"\n  Variables ({panel.shape[1]}):")
for col in panel.columns:
    n_miss = panel[col].isna().sum()
    pct_miss = n_miss / len(panel) * 100
    tipo = 'estática' if col in ['PROP_NBI', 'COMP_INASISTENCIA_ESCOLAR'] else 'dinámica'
    nivel = 'depto' if col in ['PIB_CONSTANTE', 'CREC_PIB_REAL', 'TASA_DESEMPLEO',
                                'TASA_SUBOCUPACION', 'IPM'] else 'muni'
    if col in ['COD_DEPTO', 'COD_MUNI', 'ANIO', 'POBLACION']:
        nivel = 'id'
    print(f"    {col:30s}  NA: {n_miss:>6,} ({pct_miss:5.1f}%)  [{nivel}]")

# Guardar
panel.to_csv(SALIDA / 'panel_municipal.csv', index=False)
panel.to_parquet(SALIDA / 'panel_municipal.parquet', index=False)
print(f"\n  ✓ Guardado en: {SALIDA / 'panel_municipal.csv'}")
print(f"  ✓ Guardado en: {SALIDA / 'panel_municipal.parquet'}")

# Guardar también las variables departamentales por separado
panel_depto = panel.groupby(['COD_DEPTO', 'ANIO']).agg({
    'PIB_CONSTANTE': 'first',
    'CREC_PIB_REAL': 'first',
    'TASA_DESEMPLEO': 'first',
    'TASA_SUBOCUPACION': 'first',
    'IPM': 'first'
}).reset_index()
panel_depto.to_csv(SALIDA / 'panel_departamental.csv', index=False)
print(f"  ✓ Guardado en: {SALIDA / 'panel_departamental.csv'}")

print("\n" + "=" * 70)
print("PROCESO COMPLETADO")
print("=" * 70)