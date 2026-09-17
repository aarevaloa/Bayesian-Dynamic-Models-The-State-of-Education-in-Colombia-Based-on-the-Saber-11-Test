# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# @  MODELO 4 - PARCHES v3 
# _____________________________________________________________________________

MULT_MUNI = [
    ("TASA_HOMICIDIOS",           10,      "+10 homicidios por 100 mil hab."),
    ("TASA_VICTIMAS",             10000,   "+10 pp de la población como víctimas"),
    ("PROP_HOMICIDIOS_URBANO",    0.10,    "+10 pp de homicidios urbanos"),
    ("PROP_NBI",                  10,      "+10 pp de necesidades básicas insatisfechas"),
    ("COMP_INASISTENCIA_ESCOLAR", 1,       "+1 punto de inasistencia escolar"),
    ("PROP_URBANA",               0.10,    "+10 pp de población urbana"),
    ("SGP_EDUC_PC",               100000,  "+100 mil pesos de SGP educativo per cápita"),
    ("TASA_MATRICULA_ES",         10,      "+10 puntos de matrícula en educación superior"),
    ("IES_CON_OFERTA",            10,      "+10 instituciones con oferta"),
    ("LOG_POBLACION",             np.log(2), "duplicar la población"),
]
MULT_DEPTO = [
    ("IPM",               10,    "+10 puntos del índice de pobreza multidimensional"),
    ("TASA_DESEMPLEO",    1,     "+1 pp de desempleo"),
    ("TASA_SUBOCUPACION", 1,     "+1 pp de subocupación"),
    ("CREC_PIB_REAL",     1,     "+1 pp de crecimiento del PIB"),
    ("PIB_CONSTANTE",     10000, "+10 billones de PIB departamental"),
]

MUNI = E.MULT_MUNI
DEPTO = E.MULT_DEPTO
