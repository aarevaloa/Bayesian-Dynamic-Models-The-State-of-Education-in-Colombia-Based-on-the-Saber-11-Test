# Comparacion de modelos M1-M4

Modelos comparados: M1 Normal estático, M2 t-Student estático, M3 Normal dinámico, M4 t-Student dinámico.
Capa uniforme (recomputo desde MCMC): M1, M2, M3, M4.

## Veredicto por dimension

- **Convergencia**: mejor M3 (M1=0.53, M2=0.01, M3=0.67, M4=0.36).
- **Ajuste**: mejor M4 (M1=0.00, M2=0.00, M3=0.99, M4=1.00).
- **Predicción (elpd)**: mejor M4 (M1=0.01, M2=0.00, M3=0.91, M4=1.00).
- **Calibración**: mejor M3 (M1=0.66, M2=0.50, M3=0.71, M4=0.69).
- **Ranking global (heuristico)**: M3 (0.82) > M4 (0.76) > M1 (0.30) > M2 (0.13).

El puntaje global promedia dimensiones normalizadas min-max y es solo orientativo; la comparacion formal de capacidad predictiva es el Delta_elpd pareado de la tabla siguiente.

## Delta_elpd por pares

| Par | Criterio | Delta_elpd | SE | z | Preferido | Metodo | Origen |
|---|---|---:|---:|---:|---|---|---|
| M1 vs M2 | LOO | 86.0 | 6.6 | 13.10 | M1 | pareado (y verificado) | capa uniforme |
| M1 vs M2 | WAIC | 87.7 | 6.6 | 13.32 | M1 | pareado (y verificado) | capa uniforme |
| M1 vs M3 | LOO | -5,373.8 | 113.5 | -47.34 | M3 | pareado (y verificado) | capa uniforme |
| M1 vs M3 | WAIC | -5,367.3 | 113.4 | -47.31 | M3 | pareado (y verificado) | capa uniforme |
| M1 vs M4 | LOO | -5,882.2 | 110.4 | -53.30 | M4 | pareado (y verificado) | capa uniforme |
| M1 vs M4 | WAIC | -5,875.7 | 110.3 | -53.28 | M4 | pareado (y verificado) | capa uniforme |
| M2 vs M3 | LOO | -5,459.8 | 114.2 | -47.80 | M3 | pareado (y verificado) | capa uniforme |
| M2 vs M3 | WAIC | -5,455.0 | 114.2 | -47.78 | M3 | pareado (y verificado) | capa uniforme |
| M2 vs M4 | LOO | -5,968.2 | 111.0 | -53.77 | M4 | pareado (y verificado) | capa uniforme |
| M2 vs M4 | WAIC | -5,963.4 | 110.9 | -53.75 | M4 | pareado (y verificado) | capa uniforme |
| M3 vs M4 | LOO | -508.4 | 31.4 | -16.17 | M4 | pareado (y verificado) | capa uniforme |
| M3 vs M4 | WAIC | -508.4 | 31.4 | -16.17 | M4 | pareado (y verificado) | capa uniforme |

Regla practica (Vehtari et al.): |Delta| < 2 SE no es concluyente; ademas, |Delta| < 4 se considera pequeno aun si es 'significativo'.

## Notas y advertencias

- M3: el pipeline no persiste cobertura ni R2; use --m3-mcmc para calcularlos de forma uniforme.
- M4: R-hat/ESS globales dominados por nu (verosimilitud plana en nu>100: hallazgo de especificacion, no fallo del muestreador). El tablero usa los valores SIN nu, marcados con la daga.

## Nota sobre el PPC

Con N grande los p-values bayesianos saturan a 0 o 1 ante discrepancias practicamente nulas (p.ej. una media replicada 252.1 vs 252.5 observada da p=0.000). Por eso la figura de calibracion reporta la discrepancia ESTANDARIZADA z = (T_obs - T_rep_medio)/sd(T_rep), que conserva la magnitud y es comparable entre modelos ajustados a los mismos datos. El conteo de p-values extremos se mantiene solo como referencia.

## Procedencia de cada numero

Ver `comparacion_metricas.csv` (columna `fuente`): distingue lo leido de los artefactos de cada suite de lo recalculado por la capa uniforme, para trazabilidad en el capitulo comparativo.
