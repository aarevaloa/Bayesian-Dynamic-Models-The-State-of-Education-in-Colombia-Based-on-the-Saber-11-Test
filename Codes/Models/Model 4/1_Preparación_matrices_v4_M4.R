#_______________________________________________________________________________
#@  MODELO 4 - MODELO DINÁMICO BAYESIANO t-STUDENT ROBUSTO
#@  PREPARACIÓN Y ESTRUCTURA DE DATOS
#_______________________________________________________________________________

# 1. Módulos -----
# ______________________________________________________________________________
suppressMessages(suppressWarnings(library(dplyr)))


# ______________________________________________________________________________
# 2. Cargar datos -----
# ______________________________________________________________________________

ruta    <- 'C:/Users/USUARIO/OneDrive - Universidad Nacional de Colombia/Tesis/Escritorio'
archivo <- file.path(ruta, 'Datos_procesados/Python/ICFES_MUESTRA_500000_MODELO.csv')
datos   <- read.csv(archivo, fileEncoding = "UTF-8")

cat(sprintf("Filas cargadas : %d\n", nrow(datos)))
cat(sprintf("Años presentes : %s\n", paste(sort(unique(datos$ANIO)), collapse = ", ")))
cat(sprintf("Periodos       : %s\n", paste(sort(unique(datos$periodo)), collapse = ", ")))


# ______________________________________________________________________________
# 3. Limpieza básica -----
# ______________________________________________________________________________

datos <- datos[datos$COD_DEPTO != "99999", ]
datos <- datos[datos$COD_MUNI  != "99999", ]
datos <- datos[!is.na(datos$punt_global), ]
cat(sprintf("Filas tras limpieza: %d\n", nrow(datos)))


# ______________________________________________________________________________
# 4. Índices secuenciales k, j, t -----
# ______________________________________________________________________________
depto_ids  <- sort(unique(datos$COD_DEPTO))
tiempo_ids <- sort(unique(datos$periodo))

m  <- length(depto_ids)       # departamentos
Tt <- length(tiempo_ids)      # periodos (Tt, no T, para no pisar TRUE)

datos$k_idx <- match(datos$COD_DEPTO, depto_ids)
datos$t_idx <- match(datos$periodo,   tiempo_ids)

# j reindexado dentro de cada departamento.
datos$j_idx <- NA_integer_
for (kk in 1:m) {
  idx_k    <- which(datos$k_idx == kk)
  mcpios_k <- sort(unique(datos$COD_MUNI[idx_k]))
  datos$j_idx[idx_k] <- match(datos$COD_MUNI[idx_k], mcpios_k)
}

n_k <- integer(m)
for (kk in 1:m) n_k[kk] <- max(datos$j_idx[datos$k_idx == kk])

# Diccionarios para reconstrucción
muni_ids <- vector("list", m)
for (kk in 1:m) {
  muni_ids[[kk]] <- sort(unique(datos$COD_MUNI[datos$k_idx == kk]))
}

cat(sprintf("m=%d deptos, Tt=%d periodos, %d municipios totales\n", m, Tt, sum(n_k)))


# ______________________________________________________________________________
# 5. Definición de covariables del modelo -----
# ______________________________________________________________________________-

x_ijk_names <- c(
  "estu_inse_individual",
  "cole_area_ubicacion_RURAL", "cole_area_ubicacion_MISSING",
  "cole_bilingue_S", "cole_bilingue_MISSING",
  "calen_B", "calen_OTRO", "calen_MISSING",
  "cole_carac_TECNICO_ACADEMICO", "cole_carac_TECNICO",
  "cole_carac_NO_APLICA", "cole_carac_MISSING",
  "cole_gen_FEMENINO", "cole_gen_MASCULINO", "cole_gen_MISSING",
  "cole_natu_NO_OFICIAL", "cole_natu_MISSING",
  "inter_ENTRE.1.Y.3.HORAS", "inter_ENTRE.30.Y.60.MINUTOS",
  "inter_MÁS.DE.3.HORAS", "inter_30.MINUTOS.O.MENOS", "inter_MISSING",
  "lectu_30.MINUTOS.O.MENOS", "lectu_ENTRE.30.Y.60.MINUTOS",
  "lectu_ENTRE.1.Y.2.HORAS", "lectu_MÁS.DE.2.HORAS", "lectu_MISSING",
  "genero_F", "genero_MISSING",
  "trabaja_MENOS.DE.10.HORAS", "trabaja_ENTRE.11.Y.20.HORAS",
  "trabaja_ENTRE.21.Y.30.HORAS", "trabaja_MÁS.DE.30.HORAS", "trabaja_MISSING",
  "come_carne_TODOS.O.CASI.TODOS.LOS.DÍAS",
  "come_carne_3.A.5.VECES.POR.SEMANA",
  "come_carne_1.O.2.VECES.POR.SEMANA", "come_carne_MISSING",
  "come_cereal_TODOS.O.CASI.TODOS.LOS.DÍAS",
  "come_cereal_3.A.5.VECES.POR.SEMANA",
  "come_cereal_1.O.2.VECES.POR.SEMANA", "come_cereal_MISSING",
  "come_leche_TODOS.O.CASI.TODOS.LOS.DÍAS",
  "come_leche_3.A.5.VECES.POR.SEMANA",
  "come_leche_1.O.2.VECES.POR.SEMANA", "come_leche_MISSING",
  "edu_madre_PRIMARIA.INCOMPLETA", "edu_madre_PRIMARIA.COMPLETA",
  "edu_madre_SECUNDARIA..BACHILLERATO..INCOMPLETA",
  "edu_madre_SECUNDARIA..BACHILLERATO..COMPLETA",
  "edu_madre_TÉCNICA.O.TECNOLÓGICA.INCOMPLETA",
  "edu_madre_TÉCNICA.O.TECNOLÓGICA.COMPLETA",
  "edu_madre_EDUCACIÓN.PROFESIONAL.INCOMPLETA",
  "edu_madre_EDUCACIÓN.PROFESIONAL.COMPLETA",
  "edu_madre_POSTGRADO", "edu_madre_NO.SABE",
  "edu_madre_NO.APLICA", "edu_madre_MISSING",
  "libros_casa_11.A.25.LIBROS", "libros_casa_26.A.100.LIBROS",
  "libros_casa_MÁS.DE.100.LIBROS", "libros_casa_MISSING",
  "compu_casa_SI", "compu_casa_MISSING",
  "internet_SI", "internet_MISSING"
)

# -- z_{j,k,t}: nivel municipio --
z_jk_names <- c(
  "TASA_HOMICIDIOS", "PROP_HOMICIDIOS_URBANO", "TASA_VICTIMAS",
  "TASA_MATRICULA_ES", "IES_CON_OFERTA", "SGP_EDUC_PC",
  "PROP_NBI", "COMP_INASISTENCIA_ESCOLAR", "PROP_URBANA",
  "LOG_POBLACION"
)

# -- w_{k,t}: nivel departamento --
w_k_names <- c(
  "PIB_CONSTANTE", "CREC_PIB_REAL", "TASA_DESEMPLEO",
  "TASA_SUBOCUPACION", "IPM"
)

todas_vars <- c(x_ijk_names, z_jk_names, w_k_names)
faltantes  <- setdiff(todas_vars, names(datos))
if (length(faltantes) > 0) {
  cat("Variables faltantes:\n"); print(faltantes)
  stop("Corregir nombres antes de continuar")
}
cat("OK Todas las variables del modelo presentes\n")


# ______________________________________________________________________________
# 6. Imputación de contextuales + flags -----
# ______________________________________________________________________________

# Las variables contextuales (M y D) comparten valor dentro de cada celda (j,k,t).
# Se imputan sobre un panel colapsado y luego se devuelven a las filas individuales.
#   B1) Interpolación lineal + LOCF/NOCB dentro del municipio
#   B2) Mediana por (departamento, periodo)
#   B3) Mediana por periodo
#   B4) Mediana global

contextuales <- c(z_jk_names, w_k_names)

na_antes <- sapply(datos[, contextuales], function(v) mean(is.na(v)))
cat("Tasa de NA en contextuales (antes):\n")
print(round(na_antes[na_antes > 0], 4))

# (A) Flags de missingness
flags_creados <- character(0)
for (v in contextuales) {
  if (na_antes[v] > 0.01) {
    flag_nm <- paste0(v, "_MISSING")
    datos[[flag_nm]] <- as.integer(is.na(datos[[v]]))
    flags_creados <- c(flags_creados, flag_nm)
  }
}
cat(sprintf("Flags creados (tasa NA > 1%%): %d\n", length(flags_creados)))

# (B) Panel colapsado a celdas únicas (j,k,t)
panel_mt <- datos %>%
  group_by(k_idx, j_idx, t_idx) %>%
  summarise(across(all_of(contextuales), ~ median(.x, na.rm = TRUE)),
            .groups = "drop") %>%
  mutate(across(all_of(contextuales), ~ ifelse(is.nan(.x), NA_real_, .x))) %>%
  arrange(k_idx, j_idx, t_idx) %>%
  as.data.frame()

# -- Función de imputación intramunicipal --
fill_interp <- function(x) {
  n <- length(x)
  if (n <= 1 || all(is.na(x))) return(x)
  
  obs <- which(!is.na(x))
  if (length(obs) == 0) return(x)
  
  # Huecos interiores: interpolación lineal
  if (length(obs) >= 2) {
    for (pp in 1:(length(obs) - 1)) {
      i_left  <- obs[pp]
      i_right <- obs[pp + 1]
      if (i_right - i_left > 1) {
        gap_idx <- (i_left + 1):(i_right - 1)
        fracs <- (gap_idx - i_left) / (i_right - i_left)
        x[gap_idx] <- x[i_left] + fracs * (x[i_right] - x[i_left])
      }
    }
  }
  
  # Huecos de frontera: LOCF/NOCB
  first_obs <- obs[1]
  last_obs  <- obs[length(obs)]
  if (first_obs > 1) x[1:(first_obs - 1)] <- x[first_obs]   # NOCB
  if (last_obs < n)  x[(last_obs + 1):n]  <- x[last_obs]    # LOCF
  x
}

# (B1) Interpolación lineal + LOCF/NOCB dentro de cada municipio
mk_key <- paste(panel_mt$k_idx, panel_mt$j_idx, sep = "_")
for (v in contextuales) {
  if (!any(is.na(panel_mt[[v]]))) next
  panel_mt[[v]] <- ave(panel_mt[[v]], mk_key, FUN = fill_interp)
}

# (B2) Mediana por (depto, periodo)
kt_key <- paste(panel_mt$k_idx, panel_mt$t_idx, sep = "_")
for (v in contextuales) {
  if (!any(is.na(panel_mt[[v]]))) next
  panel_mt[[v]] <- ave(panel_mt[[v]], kt_key, FUN = function(x) {
    x[is.na(x)] <- median(x, na.rm = TRUE); x
  })
}

# (B3) Mediana por periodo
for (v in contextuales) {
  if (!any(is.na(panel_mt[[v]]))) next
  panel_mt[[v]] <- ave(panel_mt[[v]], panel_mt$t_idx, FUN = function(x) {
    x[is.na(x)] <- median(x, na.rm = TRUE); x
  })
}

# (B4) Mediana global (último recurso)
for (v in contextuales) {
  if (any(is.na(panel_mt[[v]])))
    panel_mt[[v]][is.na(panel_mt[[v]])] <- median(panel_mt[[v]], na.rm = TRUE)
}

# (C) Devolver valores imputados a 'datos' vía match indexado
clave_datos <- paste(datos$k_idx, datos$j_idx, datos$t_idx, sep = "_")
clave_panel <- paste(panel_mt$k_idx, panel_mt$j_idx, panel_mt$t_idx, sep = "_")
idx_match   <- match(clave_datos, clave_panel)
stopifnot(!any(is.na(idx_match)))

for (v in contextuales) datos[[v]] <- panel_mt[[v]][idx_match]

na_despues <- sapply(datos[, contextuales], function(v) mean(is.na(v)))
if (any(na_despues > 0)) stop("Imputación incompleta")
cat("OK Imputación completa\n")

# (D) Anexar flags a los bloques correspondientes
for (f in flags_creados) {
  base <- sub("_MISSING$", "", f)
  if (base %in% z_jk_names) z_jk_names <- c(z_jk_names, f)
  if (base %in% w_k_names)  w_k_names  <- c(w_k_names,  f)
}

rm(panel_mt, mk_key, kt_key, clave_datos, clave_panel,
   idx_match, na_antes, na_despues, flags_creados, contextuales)


#_______________________________________________________________________________
# 7. Estandarización de covariables continuas -----
#_______________________________________________________________________________

todas_modelo <- c(x_ijk_names, z_jk_names, w_k_names)

n_convertidas <- 0L
for (v in todas_modelo) {
  if (!is.numeric(datos[[v]])) {
    col_char <- as.character(datos[[v]])
    if (any(col_char %in% c("True", "False"), na.rm = TRUE)) {
      datos[[v]] <- as.numeric(as.logical(col_char))
    } else {
      datos[[v]] <- as.numeric(col_char)
    }
    n_convertidas <- n_convertidas + 1L
  }
}
if (n_convertidas > 0) cat(sprintf("Columnas convertidas a numérico: %d\n", n_convertidas))

for (v in todas_modelo) {
  if (!is.numeric(datos[[v]]))
    stop(sprintf("Columna '%s' no es numérica después de coerción", v))
}

# -- Identificar dummies vs continuas --
es_dummy <- function(x) all(x %in% c(0, 1, NA))

vars_continuas_E <- x_ijk_names[!sapply(datos[, x_ijk_names], es_dummy)]
vars_continuas_M <- z_jk_names[!sapply(datos[, z_jk_names], es_dummy)]
vars_continuas_D <- w_k_names[!sapply(datos[, w_k_names],  es_dummy)]
vars_continuas   <- c(vars_continuas_E, vars_continuas_M, vars_continuas_D)

cat(sprintf("Continuas: %d (%d E, %d M, %d D)\n",
            length(vars_continuas), length(vars_continuas_E),
            length(vars_continuas_M), length(vars_continuas_D)))
cat(sprintf("Dummies:   %d\n", length(todas_modelo) - length(vars_continuas)))

estandar_params <- data.frame(
  variable = vars_continuas, media = NA_real_, desvest = NA_real_,
  n_na = NA_integer_, stringsAsFactors = FALSE
)

for (ii in seq_along(vars_continuas)) {
  v    <- vars_continuas[ii]
  mu_v <- mean(datos[[v]], na.rm = TRUE)
  sd_v <- sd(datos[[v]],   na.rm = TRUE)
  n_na <- sum(is.na(datos[[v]]))
  
  if (is.na(sd_v) || sd_v < 1e-12) {
    warning(sprintf("'%s' varianza ~0 o todo NA. No se estandariza.", v))
    sd_v <- 1
    if (is.na(mu_v)) mu_v <- 0
  }
  
  estandar_params$media[ii]   <- mu_v
  estandar_params$desvest[ii] <- sd_v
  estandar_params$n_na[ii]    <- n_na
  
  datos[[v]] <- (datos[[v]] - mu_v) / sd_v
  if (n_na > 0) datos[[v]][is.na(datos[[v]])] <- 0
}

cat(sprintf("Variables estandarizadas: %d\n", length(vars_continuas)))

na_report <- estandar_params[estandar_params$n_na > 0, ]
if (nrow(na_report) > 0) {
  cat("    NAs reemplazados con 0 (media estandarizada):\n")
  for (rr in 1:nrow(na_report)) {
    cat(sprintf("      %s: %d NAs (%.1f%%)\n",
                na_report$variable[rr], na_report$n_na[rr],
                100 * na_report$n_na[rr] / nrow(datos)))
  }
}

# -- NAs en dummies -> 0 (categoría de referencia) --
todas_dummies <- setdiff(todas_modelo, vars_continuas)
n_na_dummies <- 0L
for (v in todas_dummies) {
  na_d <- sum(is.na(datos[[v]]))
  if (na_d > 0) {
    datos[[v]][is.na(datos[[v]])] <- 0L
    n_na_dummies <- n_na_dummies + na_d
  }
}
if (n_na_dummies > 0) cat(sprintf("NAs en dummies reemplazados con 0: %d\n", n_na_dummies))


#_______________________________________________________________________________
# 8. Diagnóstico y limpieza de colinealidad -----
#_______________________________________________________________________________

# Columnas con varianza cero
todas_modelo <- c(x_ijk_names, z_jk_names, w_k_names)
eliminadas_varianza <- character(0)

for (v in todas_modelo) {
  col_var <- var(datos[[v]], na.rm = TRUE)
  if (is.na(col_var) || col_var < 1e-15) eliminadas_varianza <- c(eliminadas_varianza, v)
}

if (length(eliminadas_varianza) > 0) {
  cat(sprintf("    Varianza ~0 eliminadas: %d\n", length(eliminadas_varianza)))
  for (v in eliminadas_varianza) {
    val_unico <- unique(datos[[v]][!is.na(datos[[v]])])
    cat(sprintf("      - %s (valor: %s)\n", v, paste(round(val_unico, 4), collapse = "/")))
  }
  x_ijk_names <- setdiff(x_ijk_names, eliminadas_varianza)
  z_jk_names  <- setdiff(z_jk_names,  eliminadas_varianza)
  w_k_names   <- setdiff(w_k_names,   eliminadas_varianza)
}

# Detección de grupos de dummies completos (trampa de dummies)
dummies_E <- x_ijk_names[sapply(datos[, x_ijk_names], es_dummy)]

asignar_grupos <- function(nombres_dummy) {
  grupos <- list()
  for (nm in nombres_dummy) {
    partes <- strsplit(nm, "_")[[1]]
    asignado <- FALSE
    for (len in (length(partes) - 1):1) {
      prefijo <- paste(partes[1:len], collapse = "_")
      if (prefijo %in% names(grupos)) {
        grupos[[prefijo]] <- c(grupos[[prefijo]], nm); asignado <- TRUE; break
      }
      otras <- nombres_dummy[nombres_dummy != nm]
      coinciden <- otras[startsWith(otras, paste0(prefijo, "_"))]
      if (length(coinciden) >= 1) {
        grupos[[prefijo]] <- c(nm); asignado <- TRUE; break
      }
    }
    if (!asignado) grupos[[nm]] <- nm
  }
  grupos[sapply(grupos, length) >= 2]
}

grupos_dummy <- asignar_grupos(dummies_E)
cat(sprintf("\n    Grupos de dummies detectados: %d\n", length(grupos_dummy)))
eliminadas_trampa <- character(0)

for (pref in names(grupos_dummy)) {
  grupo <- grupos_dummy[[pref]]
  suma_grupo <- rowSums(datos[, grupo, drop = FALSE])
  rango_suma <- range(suma_grupo, na.rm = TRUE)
  if (abs(rango_suma[2] - rango_suma[1]) < 1e-10) {
    ref_a_eliminar <- grupo[length(grupo)]
    cat(sprintf("      %s_ (%d vars, suman=%.0f) -> eliminar '%s'\n",
                pref, length(grupo), rango_suma[1], ref_a_eliminar))
    eliminadas_trampa <- c(eliminadas_trampa, ref_a_eliminar)
  } else {
    cat(sprintf("      %s_ (%d vars, suman=[%.1f,%.1f]) -> OK\n",
                pref, length(grupo), rango_suma[1], rango_suma[2]))
  }
}

if (length(eliminadas_trampa) > 0) {
  cat(sprintf("    Trampa de dummies eliminadas: %d\n", length(eliminadas_trampa)))
  x_ijk_names <- setdiff(x_ijk_names, eliminadas_trampa)
}

# Verificación con QR pivotado
todas_modelo_limpio <- c(x_ijk_names, z_jk_names, w_k_names)
F_test <- cbind(1, as.matrix(datos[, todas_modelo_limpio]))

qr_F <- qr(F_test)
rango_actual <- qr_F$rank
p_actual <- ncol(F_test)
eliminadas_qr <- character(0)

if (rango_actual < p_actual) {
  cat(sprintf("\n  Rango %d < %d tras limpiezas. Eliminando residuales...\n",
              rango_actual, p_actual))
  cols_dep <- sort(qr_F$pivot[(rango_actual + 1):p_actual])
  nombres_dep <- c("intercepto", todas_modelo_limpio)[cols_dep]
  cat("    Columnas dependientes eliminadas por QR:\n")
  for (nd in nombres_dep) if (nd != "intercepto") cat(sprintf("      - %s\n", nd))
  eliminadas_qr <- nombres_dep[nombres_dep != "intercepto"]
  x_ijk_names <- setdiff(x_ijk_names, eliminadas_qr)
  z_jk_names  <- setdiff(z_jk_names,  eliminadas_qr)
  w_k_names   <- setdiff(w_k_names,   eliminadas_qr)
} else {
  cat(sprintf("    OK Rango completo (%d) tras limpiezas\n", rango_actual))
}

rm(F_test, qr_F)

# Recalcular dimensiones
p_E <- length(x_ijk_names)
p_M <- length(z_jk_names)
p_D <- length(w_k_names)
p   <- 1L + p_E + p_M + p_D

idx_mu <- 1L
# seq.int(..., length.out=) en lugar de a:b: evita la reversión del operador ':'
# si algún bloque quedara vacío (p_B = 0 daría 2:1 = c(2,1), corrompiendo idx).
idx_E  <- if (p_E > 0L) seq.int(2L,                   length.out = p_E) else integer(0)
idx_M  <- if (p_M > 0L) seq.int(2L + p_E,             length.out = p_M) else integer(0)
idx_D  <- if (p_D > 0L) seq.int(2L + p_E + p_M,       length.out = p_D) else integer(0)

theta_labels <- c("intercepto", x_ijk_names, z_jk_names, w_k_names)

estandar_params <- estandar_params[estandar_params$variable %in%
                                     c(x_ijk_names, z_jk_names, w_k_names), ]

total_elim <- length(eliminadas_varianza) + length(eliminadas_trampa) +
  length(eliminadas_qr)
cat(sprintf("Eliminadas: %d (var=0: %d, trampa: %d, QR: %d)\n",
            total_elim, length(eliminadas_varianza),
            length(eliminadas_trampa), length(eliminadas_qr)))
cat(sprintf("p final: %d (1 + %d + %d + %d)\n", p, p_E, p_M, p_D))

rm(eliminadas_varianza, eliminadas_trampa, eliminadas_qr,
   todas_modelo, grupos_dummy, dummies_E,
   todas_modelo_limpio, rango_actual, p_actual, total_elim,
   asignar_grupos, es_dummy, fill_interp,
   n_convertidas, na_report, todas_dummies, n_na_dummies,
   vars_continuas, vars_continuas_E, vars_continuas_M, vars_continuas_D)


#_______________________________________________________________________________
# 9. Matriz de diseño F_mat y estadísticos globales ------
#_______________________________________________________________________________

# theta_t: p = 1 + p_E + p_M + p_D
#   pos 1       : intercepto mu_t
#   pos idx_E   : coeficientes individuales beta_{E,t}
#   pos idx_M   : coeficientes municipales beta_{M,t}
#   pos idx_D   : coeficientes departamentales beta_{D,t}

F_mat <- cbind(intercepto = 1,
               as.matrix(datos[, x_ijk_names]),
               as.matrix(datos[, z_jk_names]),
               as.matrix(datos[, w_k_names]))

y_vec <- datos$punt_global

# Chequeos
stopifnot(is.numeric(F_mat))
stopifnot(!any(is.na(F_mat)))
stopifnot(!any(is.na(y_vec)))
stopifnot(ncol(F_mat) == p)

rk <- qr(F_mat)$rank
if (rk < p) stop(sprintf("F_mat rango %d < p=%d tras limpieza. Revisar.", rk, p))
cat(sprintf("OK F_mat %d x %d, rango completo (%d)\n", nrow(F_mat), p, rk))

# Estadísticos globales
y_media_global <- mean(y_vec)
y_var_global   <- var(y_vec)

ols_global <- tryCatch(
  qr.solve(crossprod(F_mat), crossprod(F_mat, y_vec)),
  error = function(e) { cat("OLS falló, usando semilla cero\n"); rep(0, p) }
)
names(ols_global) <- theta_labels

s2_depto <- numeric(m)
for (kk in 1:m) s2_depto[kk] <- var(y_vec[datos$k_idx == kk])

# var() devuelve NA si un departamento tiene <= 1 observación. Eso propagaría
# NA a d_k0 (bloque 12g) y haría fallar stopifnot(all(d_k0 > 0)) con un mensaje
# poco informativo. Se detiene aquí con diagnóstico explícito.
if (any(is.na(s2_depto) | s2_depto <= 0)) {
  malos <- which(is.na(s2_depto) | s2_depto <= 0)
  stop(sprintf("Departamento(s) con <2 obs o varianza nula: %s. Revisar n_k / limpieza.",
               paste(malos, collapse = ", ")))
}

cat(sprintf("y_barra = %.2f, Var(y) = %.1f\n", y_media_global, y_var_global))
cat(sprintf("OLS intercepto = %.2f\n", ols_global[1]))
cat(sprintf("s2_depto rango: [%.1f, %.1f]\n", min(s2_depto), max(s2_depto)))


#_______________________________________________________________________________
# 10. Estructuras anidadas Y, F_lst, N_jkt -----
#_______________________________________________________________________________

# El muestreador recorre los datos por (t, k, j, i). La estructura anidada
# permite acceso directo Y[[t]][[k]][[j]] y, crucialmente para el Modelo 4,
# permitirá alojar las escalas latentes xi con la MISMA topología (ver bloque 14).

clave <- paste(datos$t_idx, datos$k_idx, datos$j_idx, sep = "_")
idx_por_celda <- split(seq_len(nrow(datos)), clave)

Y       <- vector("list", Tt)
F_lst   <- vector("list", Tt)
idx_lst <- vector("list", Tt)
N_jkt   <- array(0L, dim = c(max(n_k), m, Tt))

for (tt in 1:Tt) {
  Y[[tt]]       <- vector("list", m)
  F_lst[[tt]]   <- vector("list", m)
  idx_lst[[tt]] <- vector("list", m)
  for (kk in 1:m) {
    Y[[tt]][[kk]]       <- vector("list", n_k[kk])
    F_lst[[tt]][[kk]]   <- vector("list", n_k[kk])
    idx_lst[[tt]][[kk]] <- vector("list", n_k[kk])
  }
}

for (nm in names(idx_por_celda)) {
  partes <- as.integer(strsplit(nm, "_", fixed = TRUE)[[1]])
  tt <- partes[1]; kk <- partes[2]; jj <- partes[3]
  sel <- idx_por_celda[[nm]]
  Y[[tt]][[kk]][[jj]]       <- y_vec[sel]
  F_lst[[tt]][[kk]][[jj]]   <- F_mat[sel, , drop = FALSE]
  idx_lst[[tt]][[kk]][[jj]] <- sel
  N_jkt[jj, kk, tt]         <- length(sel)
}

# Conteos precomputados
# n_{j,k}: total por municipio en TODOS los periodos -> shape de la cond. de v_{j,k}
N_jk <- matrix(0L, nrow = max(n_k), ncol = m)
for (kk in 1:m) for (jj in 1:n_k[kk]) N_jk[jj, kk] <- sum(N_jkt[jj, kk, ])

# N_{k,t}: total por departamento-periodo -> recursión beta-gamma de phi_{k,t}
N_kt <- matrix(0L, nrow = m, ncol = Tt)
for (kk in 1:m) for (tt in 1:Tt) N_kt[kk, tt] <- sum(N_jkt[1:n_k[kk], kk, tt])

# N total de observaciones = número total de escalas latentes xi
N_total <- sum(N_jkt)

celdas_ocupadas <- sum(N_jkt > 0)
cat(sprintf("Celdas (j,k,t) ocupadas: %d de %d (%.1f%%)\n",
            celdas_ocupadas, sum(n_k) * Tt,
            100 * celdas_ocupadas / (sum(n_k) * Tt)))
cat(sprintf("N total de observaciones (= N escalas latentes xi): %d\n", N_total))

# Liberar objetos que ya cumplieron su propósito.
# Se conservan: Y, F_lst, idx_lst, N_jkt/N_jk/N_kt, y_vec, ols_global, s2_depto.
rm(datos, F_mat, idx_por_celda, clave, todas_vars, faltantes)
gc(verbose = FALSE)


# ==============================================================================
#  BLOQUE II - CONFIGURACIÓN MCMC
# ==============================================================================

#_______________________________________________________________________________
# 11. Configuración MCMC -----
#_______________________________________________________________________________

n_iter  <- 100000L     # iteraciones totales
burn_in <- 10000L     # calentamiento a descartar
thin    <- 5L        # guardar cada 'thin' iteraciones post burn-in
n_save  <- as.integer((n_iter - burn_in) / thin)

set.seed(2025L)

stopifnot(burn_in < n_iter,
          (n_iter - burn_in) %% thin == 0,
          n_save > 0)

cat(sprintf("MCMC: %d iter, %d burn-in, thin=%d -> %d muestras\n",
            n_iter, burn_in, thin, n_save))


#_______________________________________________________________________________
# 12. Hiperparámetros del modelo ----
#_______________________________________________________________________________

# Todos los hiperparámetros se tratan como FIJOS y CONOCIDOS. La parametrización
# debe ser EXACTAMENTE la de la derivación de la posterior:
#   G(a,b)   = shape-rate,  densidad propto x^{a-1} e^{-bx},      media a/b
#   IG(a,b)  = shape-rate,  densidad propto x^{-a-1} e^{-b/x},    media b/(a-1)
#   Exp(l)   = tasa,        densidad propto e^{-l x},             media 1/l

# --- 12a. Varianza de innovación del intercepto (q_mu) ---
# omega_{mu,t} ~ N(0, q_mu). El intercepto absorbe la media global (~250).
# q_mu = 10 -> SD(Delta mu) ~ 3.2 puntos/periodo. Es FIJO (el intercepto no se
# regulariza con LASSO; ver W_t = diag(q_mu, phi2_B*tau2_{l,B,t})).
q_mu <- 10.0

# --- 12b. Escalas globales por bloque: phi2_B ~ IG(a_phiB, b_phiB) ---
# CAMBIO CLAVE RESPECTO AL MODELO 3: en el Modelo 3 la escala de bloque
# (varsigma2) era FIJA = 1. En el Modelo 4 la posterior especifica
#       omega_{l,B,t} | phi2_B, tau2_{l,B,t} ~ N(0, phi2_B * tau2_{l,B,t}),
# con phi2_B un PARÁMETRO con prior Inversa-Gamma. phi2_B ocupa el lugar de
# varsigma2 pero ahora se MUESTREA (condicional IG, Etapa 5).
# Calibración: a_phi=3, b_phi=2 -> E[phi2_B] = b/(a-1) = 1 ; Var = 1 (a>2 finita).
# Centrar en 1 preserva la calibración previa: con E[tau2]=0.2 se mantiene
# SD(Delta beta) ~ sqrt(E[phi2]*E[tau2]) = sqrt(0.2) ~ 0.45/periodo.
a_phi <- list(E = 3.0, M = 3.0, D = 3.0)
b_phi <- list(E = 2.0, M = 2.0, D = 2.0)

# --- 12c. Previas LASSO: lambda2_B ~ G(a_lamB, b_lamB) ---
# Cadena jerárquica: lambda2 -> tau2 ~ Exp(lambda2/2) -> omega.
# E[lambda2] = a/b = 10 -> E[tau2] = 2/E[lambda2] = 0.2.
a_lambda <- list(E = 1.0, M = 1.0, D = 1.0)
b_lambda <- list(E = 0.1, M = 0.1, D = 0.1)

# --- 12d. Precisión municipal estática: v_{j,k} ~ G(a_v, b_v) ---
# a_v = b_v = 1 -> E[v] = 1 (prior difuso, Exp(1)). La elección a_v=b_v centra
# la media a priori en 1, coherente con la restricción de identificabilidad
# sum_j log v_{j,k} = 0 (recentrado posterior, ver Etapa 4).
a_v <- 1.0
b_v <- 1.0

# --- 12e. Factor de descuento beta-gamma (beta_k) ---
# Ventana efectiva ~ 1/(1 - beta_k). beta = 0.95 -> ~20 periodos.
# beta_k en (0,1]; beta_k = 1 corresponde a varianza departamental estática.
beta_discount <- rep(0.95, m)

# --- 12f. Prior del estado inicial: theta_0 ~ N_p(m_0, C_0) ---
# m_0 = OLS global. C_0 diagonal: intercepto SD ~ sqrt(4*Var(y)), coefs SD = 10.
m_0 <- ols_global

C_0 <- diag(p)
C_0[1, 1] <- 4 * y_var_global
for (idx_block in list(idx_E, idx_M, idx_D)) diag(C_0)[idx_block] <- 100.0

# --- 12g. Prior inicial del descuento beta-gamma: phi_{k,0} ~ G(n_{k,0}/2, d_{k,0}/2) ---
# OJO: la parametrización lleva /2 en shape Y rate. Para phi ~ G(n0/2, d0/2):
#   1/phi ~ IG(n0/2, d0/2)  =>  E[s_{k,0}] = E[1/phi] = d0/(n0-2).
# Se calibra d_{k,0} = (n_{k,0}-2) * s2_depto[k]  =>  E[s_{k,0}] = Var(y en depto k).
n_k0 <- rep(4.0, m)                       # g.l. del prior chi2; shape = n0/2 = 2
d_k0 <- (n_k0 - 2) * s2_depto

stopifnot(all(d_k0 > 0), all(n_k0 > 2))

# --- 12h. Umbral numérico para la Gaussiana Inversa (cond. de tau2) ---
# Cuando |omega_{l,B,t}| < eps, el parámetro mu = lambda_B*phi_B/|omega| diverge;
# se acota |omega| inferiormente para evitar inestabilidad (Etapa 5).
eps_omega <- 1e-8

# --- 12i. Grados de libertad: nu - 2 ~ Exp(lambda_nu) ---
# Prior exponencial trasladada sobre (2, Inf). lambda_nu = 0.1 -> E[nu] = 2 + 1/0.1 = 12.
lambda_nu <- 0.1

# --- 12j. Tuning del paso Metropolis-Hastings de nu ---
# La condicional de nu NO tiene forma cerrada -> MH sobre rho = log(nu-2).
# Propuesta de caminata aleatoria gaussiana rho* = rho + N(0, sigma_MH^2).
# sigma_MH es el valor INICIAL; se adapta en burn-in para tasa de aceptación
# objetivo 0.25-0.45 (lógica adaptativa en la Etapa 6/7).
sigma_MH_init <- 0.20


#_______________________________________________________________________________
# 13. Estructura de bloques y resumen ----
#_______________________________________________________________________________

p_vec <- c(E = p_E, M = p_M, D = p_D)

# Cada bloque B in {E,M,D} agrupa: dimensión, índices en theta_t, e
# hiperparámetros propios. NOTA: 'varsigma2' del Modelo 3 se elimina; en su
# lugar el bloque guarda (a_phi, b_phi), la prior IG de la escala global phi2_B.
bloques <- list(
  E = list(nombre = "Individual",    dim = p_E, idx = idx_E,
           a_phi = a_phi$E, b_phi = b_phi$E,
           a_lambda = a_lambda$E, b_lambda = b_lambda$E),
  M = list(nombre = "Municipal",     dim = p_M, idx = idx_M,
           a_phi = a_phi$M, b_phi = b_phi$M,
           a_lambda = a_lambda$M, b_lambda = b_lambda$M),
  D = list(nombre = "Departamental", dim = p_D, idx = idx_D,
           a_phi = a_phi$D, b_phi = b_phi$D,
           a_lambda = a_lambda$D, b_lambda = b_lambda$D)
)

# Verificaciones de hiperparámetros
stopifnot(length(m_0) == p, nrow(C_0) == p, ncol(C_0) == p, all(diag(C_0) > 0))
stopifnot(length(beta_discount) == m,
          all(beta_discount > 0 & beta_discount <= 1))
stopifnot(q_mu > 0, a_v > 0, b_v > 0, eps_omega > 0, lambda_nu > 0,
          sigma_MH_init > 0)
for (B in names(bloques)) {
  blk <- bloques[[B]]
  stopifnot(blk$a_phi > 1, blk$b_phi > 0,        # a_phi>1 -> E[phi2] finita
            blk$a_lambda > 0, blk$b_lambda > 0)
}

cat(sprintf("Intercepto: q_mu=%.0f -> SD(Delta mu)=%.2f\n", q_mu, sqrt(q_mu)))
for (B in names(bloques)) {
  blk    <- bloques[[B]]
  E_lam  <- blk$a_lambda / blk$b_lambda          # E[lambda2_B]
  E_tau  <- 2 / E_lam                            # E[tau2]  (Exp(lambda2/2))
  E_phi2 <- blk$b_phi / (blk$a_phi - 1)          # E[phi2_B] (IG)
  sd_db  <- sqrt(E_phi2 * E_tau)                 # SD(Delta beta) a priori
  cat(sprintf("    Bloque %s (p=%d): E[lambda2]=%.1f, E[phi2]=%.2f, E[tau2]=%.2f -> SD(Delta beta)~%.2f\n",
              B, blk$dim, E_lam, E_phi2, E_tau, sd_db))
}
cat(sprintf("    v_{j,k}: E=%.0f | beta_k=%.2f -> ventana~%.0f periodos\n",
            a_v / b_v, beta_discount[1], 1 / (1 - beta_discount[1])))
cat(sprintf("    nu: prior nu-2 ~ Exp(%.2f) -> E[nu]=%.1f\n",
            lambda_nu, 2 + 1 / lambda_nu))


# ==============================================================================
#  BLOQUE III - INICIALIZACIÓN
# ==============================================================================
#
# Tres tipos de objetos:
#   *_curr  : valores corrientes (se actualizan cada iteración del Gibbs)
#   *_work  : arrays de trabajo (se sobreescriben, no se guardan)
#   *_save  : almacenamiento MCMC (se llenan post burn-in)
#
# Convención de indexación temporal (offset +1 por R base-1):
#   theta_curr[, t+1]        = theta_t            (t = 0, ..., T)
#   m_filt[, t+1]            = m_t                (t = 0, ..., T)
#   C_filt[,, t+1]           = C_t                (t = 0, ..., T)
#   xi_curr[[t]][[k]][[j]][i]= xi_{i,j,k,t}       (t = 1, ..., T)
#   tau2_curr[[B]][l, t]     = tau2_{l,B,t}       (t = 1, ..., T)
#   W_diag[, t]              = diag(W_t)          (t = 1, ..., T)
#   phi_curr[k, t]           = phi_{k,t}          (k = 1..m, t = 1..T)  PRIMITIVO
#   s_curr[k, t]             = s_{k,t} = 1/phi_{k,t}                    DERIVADO
#   v_curr[j, k]             = v_{j,k}            (j = 1..n_k[k], k = 1..m)
#   phi2_curr[B]             = phi2_B             (escalar por bloque)
#   nu_curr                  = nu                 (escalar)
#
# Nota: phi_{k,0} NO se almacena como valor; su prior G(n_{k,0}/2,d_{k,0}/2)
# entra al filtro forward sólo a través de (n_k0, d_k0). El backward beta-gamma
# muestrea phi_{k,1:T}.

#_______________________________________________________________________________
# 14. Parámetros corrientes (*_curr) ----
#_______________________________________________________________________________

# --- 14a. Trayectoria de estados theta_{0:T} ---
# p x (Tt+1). Todos los periodos arrancan en el OLS global.
theta_curr <- matrix(m_0, nrow = p, ncol = Tt + 1)
rownames(theta_curr) <- theta_labels

# --- 14b. Escalas latentes xi_{i,j,k,t}  (NUEVO en el Modelo 4) ---
# Lista anidada con la MISMA topología que Y: xi_curr[[t]][[k]][[j]] es un vector
# de longitud N_jkt[j,k,t]. Las celdas vacías quedan NULL (centinela).
# Inicialización en 1: es la media a priori, E[xi] = (nu/2)/(nu/2) = 1
# (prior xi ~ G(nu/2, nu/2)). xi=1 reproduce el modelo normal de partida.
xi_curr <- vector("list", Tt)
for (tt in 1:Tt) {
  xi_curr[[tt]] <- vector("list", m)
  for (kk in 1:m) {
    xi_curr[[tt]][[kk]] <- vector("list", n_k[kk])
    for (jj in 1:n_k[kk]) {
      n_ijkt <- N_jkt[jj, kk, tt]
      if (n_ijkt > 0L) xi_curr[[tt]][[kk]][[jj]] <- rep(1.0, n_ijkt)
      # celdas sin observaciones permanecen NULL
    }
  }
}

# --- 14c. Escalas locales LASSO tau2_{l,B,t} ---
# Lista de 3 matrices p_B x Tt. Init en E[tau2 | E[lambda2]] = 2 / E[lambda2].
tau2_curr <- list()
for (B in names(bloques)) {
  blk <- bloques[[B]]
  E_lam <- blk$a_lambda / blk$b_lambda
  tau2_curr[[B]] <- matrix(2.0 / E_lam, nrow = blk$dim, ncol = Tt)
}

# --- 14d. Escalas globales por bloque phi2_B  (NUEVO en el Modelo 4) ---
# Vector nombrado (E,M,D). Init en la media a priori IG: E[phi2_B] = b/(a-1).
phi2_curr <- sapply(bloques, function(blk) blk$b_phi / (blk$a_phi - 1))
names(phi2_curr) <- names(bloques)

# --- 14e. Escalas globales lambda2_B ---
# Init en su media a priori E[lambda2] = a/b.
lambda2_curr <- sapply(bloques, function(blk) blk$a_lambda / blk$b_lambda)
names(lambda2_curr) <- names(bloques)

# --- 14f. Precisiones municipales estáticas v_{j,k} ---
# max(n_k) x m. Entradas j > n_k[k] quedan NA (centinela). Init en a_v/b_v = 1;
# con v == 1 para todo (j,k) se cumple sum_j log v_{j,k} = 0 (restricción de
# identificabilidad satisfecha trivialmente en el arranque).
v_curr <- matrix(NA_real_, nrow = max(n_k), ncol = m)
for (kk in 1:m) v_curr[1:n_k[kk], kk] <- a_v / b_v

# --- 14g. Precisiones departamentales dinámicas phi_{k,t}  (PRIMITIVO) ---
# phi es el objeto primitivo (la dinámica se especifica nativamente en precisión);
# s_{k,t} = 1/phi_{k,t} es derivado. Init: phi = 1 / Var(y depto).
phi_curr <- matrix(rep(1.0 / s2_depto, Tt), nrow = m, ncol = Tt)
s_curr   <- 1.0 / phi_curr

# --- 14h. Grados de libertad nu  (NUEVO en el Modelo 4) ---
# Init en la media a priori E[nu] = 2 + 1/lambda_nu.
nu_curr <- 2.0 + 1.0 / lambda_nu

# --- 14i. Diagonal de W_t ---
# W_t = diag(q_mu, phi2_B * tau2_{l,B,t}). Se reconstruye al inicio de cada
# iteración con los tau2 y phi2 actualizados. Función auxiliar + 1a construcción.
# NOTA: el argumento phi2_vec REEMPLAZA al antiguo varsigma2 fijo del Modelo 3.
construir_W_diag <- function(tau2_list, phi2_vec, bloques_info,
                             q_mu_val, p_total, Tt_val) {
  W <- matrix(NA_real_, nrow = p_total, ncol = Tt_val)
  W[1, ] <- q_mu_val                                  # intercepto: q_mu fijo
  for (B in names(bloques_info)) {
    blk <- bloques_info[[B]]
    W[blk$idx, ] <- phi2_vec[[B]] * tau2_list[[B]]     # bloque: phi2_B * tau2
  }
  W
}

W_diag <- construir_W_diag(tau2_curr, phi2_curr, bloques, q_mu, p, Tt)

cat(sprintf("theta_curr  : %d x %d\n", p, Tt + 1))
cat(sprintf("xi_curr     : lista anidada, %d escalas latentes\n",
            length(unlist(xi_curr))))
cat(sprintf("tau2_curr   : E=%dx%d, M=%dx%d, D=%dx%d\n",
            p_E, Tt, p_M, Tt, p_D, Tt))
cat(sprintf("phi2_curr   : E=%.2f, M=%.2f, D=%.2f\n",
            phi2_curr["E"], phi2_curr["M"], phi2_curr["D"]))
cat(sprintf("lambda2_curr: E=%.1f, M=%.1f, D=%.1f\n",
            lambda2_curr["E"], lambda2_curr["M"], lambda2_curr["D"]))
cat(sprintf("v_curr      : %d x %d (init=%.1f)\n", max(n_k), m, a_v / b_v))
cat(sprintf("phi_curr    : %d x %d (rango [%.4f, %.4f])\n",
            m, Tt, min(phi_curr), max(phi_curr)))
cat(sprintf("nu_curr     : %.1f\n", nu_curr))


#_______________________________________________________________________________
# 15. Arrays de trabajo (*_work) -----
#_______________________________________________________________________________

# Se sobreescriben en cada iteración del Gibbs; no se almacenan.
#
# --- 15a. Momentos del filtro de Kalman (FFBS gaussiano de theta_{0:T}) ---
# m_filt y C_filt almacenan los momentos filtrados (m_t, C_t). En la fase
# backward, a_{t+1}=m_t y R_{t+1}=C_t+W_{t+1} se reconstruyen sobre la marcha
# (válido porque la matriz de transición es G = I_p, caminata aleatoria).
m_filt <- matrix(NA_real_, nrow = p, ncol = Tt + 1)
C_filt <- array(NA_real_,  dim = c(p, p, Tt + 1))
m_filt[, 1]  <- m_0
C_filt[,, 1] <- C_0

# --- 15b. Suma de cuadrados ponderada por xi, por celda (j,k,t) ---
# SSxi_jkt[j,k,t] = sum_{i in celda} xi_{i,j,k,t} * (y - F'theta_t)^2.
# Es el bloque comun a las condicionales de v y de phi:
#   SSW_{j,k}   = sum_t phi_{k,t} * SSxi_jkt[j,k,t]     (cond. de v_{j,k})
#   SSD_{k,t}   = sum_j v_{j,k}   * SSxi_jkt[j,k,t]     (cond. de phi_{k,t})
# Se recalcula cada iteración porque depende de theta_t y de xi.
SSxi_jkt <- array(0.0, dim = c(max(n_k), m, Tt))

# --- 15c. Recursión beta-gamma para phi_{k,t} ---
# n_{k,t} y d_{k,t} se recalculan secuencialmente en t dentro de cada iteración.
# Convención: columnas 1..Tt corresponden a t=1..T; el par inicial (n_k0,d_k0)
# del prior se mantiene aparte.
n_kt_work <- matrix(NA_real_, nrow = m, ncol = Tt)
d_kt_work <- matrix(NA_real_, nrow = m, ncol = Tt)

# --- 15d. Estadísticos suficientes para la condicional de nu ---
# La condicional de nu depende de Xi sólo a través de:
#   P = sum log(xi_{i,j,k,t})  ,  Q = sum xi_{i,j,k,t}
# Escalares; se recalculan una vez por iteración (O(N)).
P_xi_work <- NA_real_
Q_xi_work <- NA_real_


#_______________________________________________________________________________
# 16. Almacenamiento MCMC (*_save) ----
#_______________________________________________________________________________

# --- 16a. theta_{0:T} ---
theta_save <- array(NA_real_, dim = c(p, Tt + 1, n_save))
cat(sprintf("    theta_save  : %d x %d x %d (%.1f MB)\n",
            p, Tt + 1, n_save, prod(dim(theta_save)) * 8 / 1e6))

# --- 16b. phi2_B  (escalas globales por bloque) ---
phi2_save <- matrix(NA_real_, nrow = n_save, ncol = 3)
colnames(phi2_save) <- c("E", "M", "D")

# --- 16c. lambda2_B ---
lambda2_save <- matrix(NA_real_, nrow = n_save, ncol = 3)
colnames(lambda2_save) <- c("E", "M", "D")

# --- 16d. nu ---
nu_save <- numeric(n_save)

# --- 16e. v_{j,k} ---
v_save <- array(NA_real_, dim = c(max(n_k), m, n_save))
cat(sprintf("    v_save      : %d x %d x %d (%.1f MB)\n",
            max(n_k), m, n_save, prod(dim(v_save)) * 8 / 1e6))

# --- 16f. s_{k,t}  (varianzas departamentales, derivadas de phi) ---
s_save <- array(NA_real_, dim = c(m, Tt, n_save))

# --- 16g. tau2 (opcional: voluminoso) ---
almacenar_tau2 <- TRUE
tau2_save <- list()
if (almacenar_tau2) {
  for (B in names(bloques)) {
    blk <- bloques[[B]]
    tau2_save[[B]] <- array(NA_real_, dim = c(blk$dim, Tt, n_save))
  }
}

# --- 16h. xi (escalas latentes) ---
# Xi es un latente de nuisance de dimensión N (decenas a cientos de miles).
# Guardar TODAS las trazas costaria O(N * n_save) doubles -> prohibitivo y
# raramente necesario (PPC y diagnósticos se reconstruyen desde theta,phi,v,nu).
# Por defecto NO se almacena; activar el flag sólo si se requiere explícitamente.
almacenar_xi <- FALSE
xi_save <- NULL
if (almacenar_xi) {
  warning("almacenar_xi=TRUE: uso de memoria O(N_total * n_save).")
  xi_save <- array(NA_real_, dim = c(N_total, n_save))
}

# --- 16i. Tasa de aceptación del MH de nu ---
nu_accept_count <- 0L            # nº de propuestas aceptadas (diagnóstico)
sigma_MH        <- sigma_MH_init  # se adapta en burn-in (Etapa 6/7)

# Contador de muestras guardadas
save_idx <- 0L

# -- (Opcional) persistir datos del modelo para diagnósticos --
# El muestreador LEE estos objetos; nunca los modifica.
# saveRDS(list(
#   Y = Y, F_lst = F_lst, idx_lst = idx_lst, N_jkt = N_jkt,
#   N_jk = N_jk, N_kt = N_kt, N_total = N_total,
#   n_k = n_k, m = m, Tt = Tt, p = p, p_E = p_E, p_M = p_M, p_D = p_D,
#   idx_mu = idx_mu, idx_E = idx_E, idx_M = idx_M, idx_D = idx_D,
#   bloques = bloques, theta_labels = theta_labels,
#   depto_ids = depto_ids, tiempo_ids = tiempo_ids, muni_ids = muni_ids,
#   y_vec = y_vec, estandar_params = estandar_params
# ), file.path(getwd(), "datos_modelo4.rds"))
# cat("OK datos_modelo4.rds guardado\n")


#_______________________________________________________________________________
# 17. Validaciones finales ----
#_______________________________________________________________________________

# --- (A) Positividad y finitud ---
stopifnot(all(is.finite(theta_curr)))
stopifnot(all(sapply(tau2_curr, function(x) all(x > 0 & is.finite(x)))))
stopifnot(all(phi2_curr > 0), all(is.finite(phi2_curr)))
stopifnot(all(lambda2_curr > 0), all(is.finite(lambda2_curr)))
for (kk in 1:m) stopifnot(all(v_curr[1:n_k[kk], kk] > 0))
stopifnot(all(phi_curr > 0), all(s_curr > 0))
stopifnot(max(abs(s_curr * phi_curr - 1)) < 1e-10)   # s == 1/phi exacto
stopifnot(all(W_diag > 0), !any(is.na(W_diag)))
stopifnot(nu_curr > 2)                               # restricción nu > 2
stopifnot(sigma_MH > 0)

# Escalas latentes: una por observación, todas > 0
xi_all <- unlist(xi_curr)
stopifnot(length(xi_all) == N_total)                 # topología xi == topología Y
stopifnot(all(is.finite(xi_all)), all(xi_all > 0))

# Restricción de identificabilidad en el arranque: sum_j log v_{j,k} = 0
for (kk in 1:m) {
  s_logv <- sum(log(v_curr[1:n_k[kk], kk]))
  stopifnot(abs(s_logv) < 1e-10)
}

# --- (B) Dimensiones cruzadas ---
stopifnot(nrow(theta_curr) == p, ncol(theta_curr) == Tt + 1)
stopifnot(nrow(phi_curr) == m,   ncol(phi_curr) == Tt)
stopifnot(nrow(s_curr)   == m,   ncol(s_curr)   == Tt)
stopifnot(nrow(v_curr)   == max(n_k), ncol(v_curr) == m)
stopifnot(nrow(W_diag)   == p,   ncol(W_diag)   == Tt)
stopifnot(length(phi2_curr) == 3, length(lambda2_curr) == 3)
for (B in names(bloques)) {
  blk <- bloques[[B]]
  stopifnot(nrow(tau2_curr[[B]]) == blk$dim, ncol(tau2_curr[[B]]) == Tt)
}
# El chequeo de tau2_save debe ir DENTRO del if: con almacenar_tau2 = FALSE,
# tau2_save[[B]] es NULL y stopifnot(dim(NULL)[1] == ...) evalúa logical(0),
# que stopifnot acepta como TRUE -> validación silenciosamente desactivada.
if (almacenar_tau2) {
  for (B in names(bloques))
    stopifnot(all(dim(tau2_save[[B]]) == c(bloques[[B]]$dim, Tt, n_save)))
}
stopifnot(all(m_filt[, 1] == m_0), all(C_filt[,, 1] == C_0))
stopifnot(dim(SSxi_jkt)[1] == max(n_k), dim(SSxi_jkt)[2] == m,
          dim(SSxi_jkt)[3] == Tt)

# Coherencia p = 1 + p_E + p_M + p_D y particion de indices
stopifnot(p == 1L + p_E + p_M + p_D)
stopifnot(length(c(idx_mu, idx_E, idx_M, idx_D)) == p)
stopifnot(!any(duplicated(c(idx_mu, idx_E, idx_M, idx_D))))

# --- (C) Memoria de arrays ---
# Huella de parámetros corrientes, arrays de trabajo y cadenas MCMC.
bytes_param <- (
  prod(dim(theta_curr)) + length(xi_all) + prod(dim(W_diag)) +
    sum(sapply(tau2_curr, function(x) prod(dim(x)))) +
    length(phi2_curr) + length(lambda2_curr) + 1 +              # phi2, lambda2, nu
    prod(dim(v_curr)) + prod(dim(phi_curr)) + prod(dim(s_curr)) +
    prod(dim(m_filt)) + prod(dim(C_filt)) + prod(dim(SSxi_jkt)) +
    prod(dim(n_kt_work)) + prod(dim(d_kt_work)) +
    prod(dim(theta_save)) + prod(dim(phi2_save)) +
    prod(dim(lambda2_save)) + length(nu_save) +
    prod(dim(v_save)) + prod(dim(s_save)) +
    { if (almacenar_tau2) sum(sapply(tau2_save, function(x) prod(dim(x)))) else 0 } +
    { if (almacenar_xi)   prod(dim(xi_save)) else 0 }
) * 8

# Huella de los datos del modelo (Y, F_lst, idx_lst): F_lst replica toda la
# matriz de diseño (~N*p doubles) y suele dominar la memoria. Se mide con
# object.size() porque son listas anidadas, no arrays regulares.
bytes_datos <- as.numeric(object.size(Y) + object.size(F_lst) +
                            object.size(idx_lst))

cat(sprintf("\n=== OK Validaciones de la Etapa 1 completadas ===\n"))
cat(sprintf("    p = %d  (1 + %d E + %d M + %d D)\n", p, p_E, p_M, p_D))
cat(sprintf("    m = %d deptos, Tt = %d periodos, N = %d observaciones\n",
            m, Tt, N_total))
cat(sprintf("    Memoria parámetros + cadenas: %.1f MB\n", bytes_param / 1e6))
cat(sprintf("    Memoria datos (Y, F_lst, idx_lst): %.1f MB\n", bytes_datos / 1e6))
cat(sprintf("    Memoria total aproximada: %.1f MB\n",
            (bytes_param + bytes_datos) / 1e6))

rm(xi_all)

# # ______________________________________________________________________________
# # 18. Exports para el analisis en Python -----
# # ______________________________________________________________________________
# RUTA <- 'C:/Users/USUARIO/OneDrive - Universidad Nacional de Colombia/Tesis/Escritorio/Códigos/Modelos/Modelo 4/export_python'
# 
# # --- 1. Equivalencia (k, j) -> codigo DANE ---
# muni_tab <- do.call(rbind, lapply(seq_len(m), function(kk)
#   data.frame(k         = kk,
#              j         = seq_along(muni_ids[[kk]]),
#              cod_depto = depto_ids[kk],
#              cod_dane  = muni_ids[[kk]])))
# stopifnot(nrow(muni_tab) == sum(n_k),
#           all(tapply(muni_tab$j, muni_tab$k, max) == n_k))
# write.csv(muni_tab, file.path(RUTA, "muni_ids.csv"), row.names = FALSE)
# 
# # --- 2. Etiquetas reales de los periodos ---
# write.csv(data.frame(t = seq_along(tiempo_ids), periodo = tiempo_ids),
#           file.path(RUTA, "periodo_ids.csv"), row.names = FALSE)
# 
# # --- 3. Parametros de estandarizacion (media y sd en unidades originales) ---
# write.csv(estandar_params, file.path(RUTA, "estandar_params.csv"), row.names = FALSE)

# # --- 3. Parametros de estandarizacion (media y sd en unidades originales) ---
# write.csv(estandar_params, file.path(RUTA, "estandar_params.csv"), row.names = FALSE)
