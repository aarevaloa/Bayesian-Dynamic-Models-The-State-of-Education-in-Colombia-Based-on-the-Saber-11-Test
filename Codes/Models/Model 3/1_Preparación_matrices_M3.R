#_______________________________________________________________________________
#@  MODELO NORMAL DINÁMICO CON COVARIABLES — GIBBS SAMPLER
#_______________________________________________________________________________
# ______________________________________________________________________________
# 1. Modulos -----
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
# ______________________________________________________________________________

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

# ── z_{j,k,t}: nivel municipio ──
# Variables socioeconómicas y de violencia que varían entre municipios.

z_jk_names <- c(
  "TASA_HOMICIDIOS", "PROP_HOMICIDIOS_URBANO", "TASA_VICTIMAS",
  "TASA_MATRICULA_ES", "IES_CON_OFERTA", "SGP_EDUC_PC",
  "PROP_NBI", "COMP_INASISTENCIA_ESCOLAR", "PROP_URBANA",
  "LOG_POBLACION"
)

# ── w_{k,t}: nivel departamento ──
# Indicadores macroeconómicos departamentales.

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
cat("✓ Todas las variables del modelo presentes\n")

# ______________________________________________________________________________
# 6. Imputación de contextuales + flags -----
# ______________________________________________________________________________
# Estrategia jerárquica:
#   B1) Interpolación lineal + LOCF/NOCB dentro del municipio (Huecos interiores y de frontera)
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

# ── Función de imputación intramunicipal ──

fill_interp <- function(x) {
  n <- length(x)
  if (n <= 1 || all(is.na(x))) return(x)
  
  obs <- which(!is.na(x))
  if (length(obs) == 0) return(x)

  # ── Huecos interiores: interpolación lineal ──

  if (length(obs) >= 2) {
    for (pp in 1:(length(obs) - 1)) {
      i_left  <- obs[pp]
      i_right <- obs[pp + 1]
      if (i_right - i_left > 1) {
        # Hay un hueco entre i_left e i_right
        gap_idx <- (i_left + 1):(i_right - 1)
        # Interpolación lineal: x[i_left] + frac * (x[i_right] - x[i_left])
        fracs <- (gap_idx - i_left) / (i_right - i_left)
        x[gap_idx] <- x[i_left] + fracs * (x[i_right] - x[i_left])
      }
    }
  }
  
  # ── Huecos de frontera: LOCF/NOCB ──
  first_obs <- obs[1]
  last_obs  <- obs[length(obs)]
  
  if (first_obs > 1) {
    x[1:(first_obs - 1)] <- x[first_obs]          # NOCB
  }
  if (last_obs < n) {
    x[(last_obs + 1):n] <- x[last_obs]             # LOCF
  }
  
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
cat("✓ Imputación completa\n")

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
if (n_convertidas > 0) {
  cat(sprintf("Columnas convertidas a numérico: %d\n", n_convertidas))
}

for (v in todas_modelo) {
  if (!is.numeric(datos[[v]])) {
    stop(sprintf("Columna '%s' no es numérica después de coerción", v))
  }
}

# ── Identificar dummies vs continuas ──

es_dummy <- function(x) all(x %in% c(0, 1, NA))

vars_continuas_E <- x_ijk_names[!sapply(datos[, x_ijk_names], es_dummy)]
vars_continuas_M <- z_jk_names[!sapply(datos[, z_jk_names], es_dummy)]
vars_continuas_D <- w_k_names[!sapply(datos[, w_k_names],  es_dummy)]
vars_continuas   <- c(vars_continuas_E, vars_continuas_M, vars_continuas_D)

cat(sprintf("Continuas: %d (%d E, %d M, %d D)\n",
            length(vars_continuas), length(vars_continuas_E),
            length(vars_continuas_M), length(vars_continuas_D)))
cat(sprintf("Dummies:   %d\n",
            length(todas_modelo) - length(vars_continuas)))

estandar_params <- data.frame(
  variable = vars_continuas,
  media    = NA_real_,
  desvest  = NA_real_,
  n_na     = NA_integer_,
  stringsAsFactors = FALSE
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
  
  if (n_na > 0) {
    datos[[v]][is.na(datos[[v]])] <- 0
  }
}

cat(sprintf("Variables estandarizadas: %d (%d E, %d M, %d D)\n",
            length(vars_continuas), length(vars_continuas_E),
            length(vars_continuas_M), length(vars_continuas_D)))

# Reportar NAs imputados a 0
na_report <- estandar_params[estandar_params$n_na > 0, ]
if (nrow(na_report) > 0) {
  cat("    NAs reemplazados con 0 (media estandarizada):\n")
  for (rr in 1:nrow(na_report)) {
    cat(sprintf("      %s: %d NAs (%.1f%%)\n",
                na_report$variable[rr], na_report$n_na[rr],
                100 * na_report$n_na[rr] / nrow(datos)))
  }
}

# ── NAs en dummies ──
# Si las dummies tienen NAs romperían F_mat. Se reemplazan con 0 (categoría de referencia).
todas_dummies <- setdiff(todas_modelo, vars_continuas)
n_na_dummies <- 0L
for (v in todas_dummies) {
  na_d <- sum(is.na(datos[[v]]))
  if (na_d > 0) {
    datos[[v]][is.na(datos[[v]])] <- 0L
    n_na_dummies <- n_na_dummies + na_d
  }
}
if (n_na_dummies > 0) {
  cat(sprintf("NAs en dummies reemplazados con 0: %d\n", n_na_dummies))
}

#_______________________________________________________________________________
# 8. Diagnóstico y limpieza de colinealidad -----
#_______________________________________________________________________________

# Columnas con vbarianza cero

todas_modelo <- c(x_ijk_names, z_jk_names, w_k_names)
eliminadas_varianza <- character(0)

for (v in todas_modelo) {
  col_var <- var(datos[[v]], na.rm = TRUE)
  if (is.na(col_var) || col_var < 1e-15) {
    eliminadas_varianza <- c(eliminadas_varianza, v)
  }
}

if (length(eliminadas_varianza) > 0) {
  cat(sprintf("    Varianza ~0 eliminadas: %d\n", length(eliminadas_varianza)))
  for (v in eliminadas_varianza) {
    val_unico <- unique(datos[[v]][!is.na(datos[[v]])])
    cat(sprintf("      - %s (valor: %s)\n", v,
                paste(round(val_unico, 4), collapse = "/")))
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
        grupos[[prefijo]] <- c(grupos[[prefijo]], nm)
        asignado <- TRUE
        break
      }
      
      otras <- nombres_dummy[nombres_dummy != nm]
      coinciden <- otras[startsWith(otras, paste0(prefijo, "_"))]
      if (length(coinciden) >= 1) {
        grupos[[prefijo]] <- c(nm)
        asignado <- TRUE
        break
      }
    }
    
    if (!asignado) {
      grupos[[nm]] <- nm
    }
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
    cat(sprintf("      %s_ (%d vars, suman=%.0f) → eliminar '%s'\n",
                pref, length(grupo), rango_suma[1], ref_a_eliminar))
    eliminadas_trampa <- c(eliminadas_trampa, ref_a_eliminar)
  } else {
    cat(sprintf("      %s_ (%d vars, suman=[%.1f,%.1f]) → OK\n",
                pref, length(grupo), rango_suma[1], rango_suma[2]))
  }
}

if (length(eliminadas_trampa) > 0) {
  cat(sprintf("    Trampa de dummies eliminadas: %d\n", length(eliminadas_trampa)))
  x_ijk_names <- setdiff(x_ijk_names, eliminadas_trampa)
}

# Verificwación con QR pivotado

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
  for (nd in nombres_dep) {
    if (nd != "intercepto") cat(sprintf("      - %s\n", nd))
  }
  
  eliminadas_qr <- nombres_dep[nombres_dep != "intercepto"]
  x_ijk_names <- setdiff(x_ijk_names, eliminadas_qr)
  z_jk_names  <- setdiff(z_jk_names,  eliminadas_qr)
  w_k_names   <- setdiff(w_k_names,   eliminadas_qr)
} else {
  cat(sprintf("    ✓ Rango completo (%d) tras limpiezas\n", rango_actual))
}

rm(F_test, qr_F)

# Recalcular dimensiones

p_E <- length(x_ijk_names)
p_M <- length(z_jk_names)
p_D <- length(w_k_names)
p   <- 1L + p_E + p_M + p_D

idx_mu <- 1L
idx_E  <- 2L:(1L + p_E)
idx_M  <- (1L + p_E + 1L):(1L + p_E + p_M)
idx_D  <- (1L + p_E + p_M + 1L):p

theta_labels <- c("intercepto", x_ijk_names, z_jk_names, w_k_names)

# Actualizar estandar_params
estandar_params <- estandar_params[estandar_params$variable %in%
                                     c(x_ijk_names, z_jk_names, w_k_names), ]

total_elim <- length(eliminadas_varianza) + length(eliminadas_trampa) +
  length(eliminadas_qr)

cat(sprintf("Eliminadas: %d (var=0: %d, trampa: %d, QR: %d)\n",
            total_elim, length(eliminadas_varianza),
            length(eliminadas_trampa), length(eliminadas_qr)))
cat(sprintf("p final: %d (1 + %d + %d + %d)\n", p, p_E, p_M, p_D))

rm(eliminadas_varianza, eliminadas_trampa, eliminadas_qr,
   todas_modelo, grupos_dummy, dummies_E)

#_______________________________________________________________________________
# 9. Matriz de diseño F_mat y estadísticos globales ------
#_______________________________________________________________________________

# Dimensiones del vector de estado theta_t:
#   p = 1 + p_E + p_M + p_D
#   posición 1       : intercepto mu_t
#   posiciones idx_E : coeficientes individuales beta_{E,t}
#   posiciones idx_M : coeficientes municipales beta_{M,t}
#   posiciones idx_D : coeficientes departamentales beta_{D,t}

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

# Rango debe ser completo
rk <- qr(F_mat)$rank
if (rk < p) {
  stop(sprintf("F_mat rango %d < p=%d tras limpieza. Revisar.", rk, p))
}
cat(sprintf("✓ F_mat %d × %d, rango completo (%d)\n",
            nrow(F_mat), p, rk))

# Estadísticos globales
y_media_global <- mean(y_vec)
y_var_global   <- var(y_vec)

ols_global <- tryCatch(
  qr.solve(crossprod(F_mat), crossprod(F_mat, y_vec)),
  error = function(e) {
    cat("OLS falló, usando semilla cero\n")
    rep(0, p)
  }
)
names(ols_global) <- theta_labels

s2_depto <- numeric(m)
for (kk in 1:m) s2_depto[kk] <- var(y_vec[datos$k_idx == kk])

cat(sprintf("ȳ = %.2f, Var(y) = %.1f\n", y_media_global, y_var_global))
cat(sprintf("OLS intercepto = %.2f\n", ols_global[1]))
cat(sprintf("s2_depto rango: [%.1f, %.1f]\n", min(s2_depto), max(s2_depto)))

#_______________________________________________________________________________
# 10. Estructuras anidadas Y, F_lst, N_jkt -----
#_______________________________________________________________________________
# El filtro de Kalman recorre los datos por (t, k, j, i).
# La estructura anidada permite acceso directo Y[[t]][[k]][[j]]

clave <- paste(datos$t_idx, datos$k_idx, datos$j_idx, sep = "_")
idx_por_celda <- split(seq_len(nrow(datos)), clave)

Y <- vector("list", Tt)
F_lst <- vector("list", Tt)
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
# N_{j,k}: total por municipio en todos los periodos (para v_{j,k})
N_jk <- matrix(0L, nrow = max(n_k), ncol = m)
for (kk in 1:m) for (jj in 1:n_k[kk]) N_jk[jj, kk] <- sum(N_jkt[jj, kk, ])

# N_{k,t}: total por departamento-periodo (para phi_{k,t})
N_kt <- matrix(0L, nrow = m, ncol = Tt)
for (kk in 1:m) for (tt in 1:Tt) N_kt[kk, tt] <- sum(N_jkt[1:n_k[kk], kk, tt])

celdas_ocupadas <- sum(N_jkt > 0)
cat(sprintf("Celdas (j,k,t) ocupadas: %d de %d (%.1f%%)\n",
            celdas_ocupadas, sum(n_k) * Tt,
            100 * celdas_ocupadas / (sum(n_k) * Tt)))

# Liberar objetos que ya cumplieron su propósito

rm(datos, F_mat, idx_por_celda, clave, todas_vars, faltantes)
gc(verbose = FALSE)

#_______________________________________________________________________________
# 11. Configuración MCMC -----
#_______________________________________________________________________________

n_iter  <- 35000L     # iteraciones totales
burn_in <- 5000L     # calentamiento a descartar
thin    <- 5L        # guardar cada thin iteraciones post burn-in
n_save  <- as.integer((n_iter - burn_in) / thin)

set.seed(2025L)

stopifnot(burn_in < n_iter,
          (n_iter - burn_in) %% thin == 0,
          n_save > 0)

cat(sprintf("MCMC: %d iter, %d burn-in, thin=%d → %d muestras\n",
            n_iter, burn_in, thin, n_save))

#_______________________________________________________________________________
# 12. Hiperparámetros del modelo ----
#_______________________________________________________________________________
q_mu <- 10.0
varsigma2 <- list(E = 1.0, M = 1.0, D = 1.0)
a_lambda <- list(E = 1.0, M = 1.0, D = 1.0)
b_lambda <- list(E = 0.1, M = 0.1, D = 0.1)
a_v <- 1.0
b_v <- 1.0
beta_discount <- rep(0.95, m)
m_0 <- ols_global

C_0 <- diag(p)
C_0[1, 1] <- 4 * y_var_global
for (idx_block in list(idx_E, idx_M, idx_D)) {
  diag(C_0)[idx_block] <- 100.0
}

n_k0 <- rep(4.0, m)
d_k0 <- (n_k0 - 2) * s2_depto

stopifnot(all(d_k0 > 0), all(n_k0 > 2))

eps_omega <- 1e-8

#_______________________________________________________________________________
# 13. Estructura de bloques y resumen ----
#_______________________________________________________________________________

p_vec <- c(E = p_E, M = p_M, D = p_D)

bloques <- list(
  E = list(nombre = "Individual",     dim = p_E, idx = idx_E,
           varsigma2 = varsigma2$E, a_lambda = a_lambda$E, b_lambda = b_lambda$E),
  M = list(nombre = "Municipal",      dim = p_M, idx = idx_M,
           varsigma2 = varsigma2$M, a_lambda = a_lambda$M, b_lambda = b_lambda$M),
  D = list(nombre = "Departamental",  dim = p_D, idx = idx_D,
           varsigma2 = varsigma2$D, a_lambda = a_lambda$D, b_lambda = b_lambda$D)
)

# Verificaciones
stopifnot(length(m_0) == p, nrow(C_0) == p, ncol(C_0) == p,
          all(diag(C_0) > 0),
          length(beta_discount) == m,
          all(beta_discount > 0 & beta_discount <= 1),
          q_mu > 0, a_v > 0, b_v > 0, eps_omega > 0)

cat(sprintf("Intercepto: q_mu=%.0f → SD(Δμ)=%.2f\n", q_mu, sqrt(q_mu)))
for (B in names(bloques)) {
  blk <- bloques[[B]]
  E_lam <- blk$a_lambda / blk$b_lambda
  cat(sprintf("    Bloque %s (p=%d): E[λ²]=%.0f → SD(Δβ)≈%.2f\n",
              B, blk$dim, E_lam, sqrt(blk$varsigma2 * 2 / E_lam)))
}
cat(sprintf("    v_{j,k}: E=%.0f | β_k=%.2f → ventana≈%.0f periodos\n",
            a_v/b_v, beta_discount[1], 1/(1 - beta_discount[1])))

#_______________________________________________________________________________
# 14. Parámetros corrientes
#_______________________________________________________________________________

theta_curr <- matrix(m_0, nrow = p, ncol = Tt + 1)
rownames(theta_curr) <- theta_labels

tau2_curr <- list()
for (B in names(bloques)) {
  blk <- bloques[[B]]
  tau2_curr[[B]] <- matrix(2.0 / (blk$a_lambda / blk$b_lambda),
                           nrow = blk$dim, ncol = Tt)
}

lambda2_curr <- sapply(bloques, function(blk) blk$a_lambda / blk$b_lambda)

v_curr <- matrix(NA_real_, nrow = max(n_k), ncol = m)
for (kk in 1:m) v_curr[1:n_k[kk], kk] <- a_v / b_v

s_curr   <- matrix(rep(s2_depto, Tt), nrow = m, ncol = Tt)
phi_curr <- 1.0 / s_curr

construir_W_diag <- function(tau2_list, bloques_info, q_mu_val, p_total, Tt_val) {
  W <- matrix(NA_real_, nrow = p_total, ncol = Tt_val)
  W[1, ] <- q_mu_val
  for (B in names(bloques_info)) {
    blk <- bloques_info[[B]]
    W[blk$idx, ] <- blk$varsigma2 * tau2_list[[B]]
  }
  W
}

W_diag <- construir_W_diag(tau2_curr, bloques, q_mu, p, Tt)

cat(sprintf("theta_curr  : %d × %d\n", p, Tt + 1))
cat(sprintf("tau2_curr   : E=%d×%d, M=%d×%d, D=%d×%d\n",
            p_E, Tt, p_M, Tt, p_D, Tt))
cat(sprintf("lambda2_curr: E=%.1f, M=%.1f, D=%.1f\n",
            lambda2_curr["E"], lambda2_curr["M"], lambda2_curr["D"]))
cat(sprintf("v_curr      : %d × %d (init=%.1f)\n", max(n_k), m, a_v/b_v))
cat(sprintf("s_curr      : %d × %d (rango [%.1f, %.1f])\n",
            m, Tt, min(s_curr), max(s_curr)))

#_______________________________________________________________________________
# 15. Arrays de trabajo 
#_______________________________________________________________________________

m_filt <- matrix(NA_real_, nrow = p, ncol = Tt + 1)
C_filt <- array(NA_real_,  dim = c(p, p, Tt + 1))

m_filt[, 1]  <- m_0
C_filt[,, 1] <- C_0

SSE_jkt <- array(0, dim = c(max(n_k), m, Tt))

n_kt_work <- matrix(NA_real_, nrow = m, ncol = Tt)
d_kt_work <- matrix(NA_real_, nrow = m, ncol = Tt)

#_______________________________________________________________________________
# 16. Almacenamiento MCMC ----
#_______________________________________________________________________________

# 16a. theta_{0:T}
theta_save <- array(NA_real_, dim = c(p, Tt + 1, n_save))
cat(sprintf("    theta_save  : %d × %d × %d (%.1f MB)\n",
            p, Tt + 1, n_save, prod(dim(theta_save)) * 8 / 1e6))

# 16b. lambda2_B
lambda2_save <- matrix(NA_real_, nrow = n_save, ncol = 3)
colnames(lambda2_save) <- c("E", "M", "D")

# 16c. v_{j,k}
v_save <- array(NA_real_, dim = c(max(n_k), m, n_save))
cat(sprintf("    v_save      : %d × %d × %d (%.1f MB)\n",
            max(n_k), m, n_save, prod(dim(v_save)) * 8 / 1e6))

# 16d. s_{k,t}
s_save <- array(NA_real_, dim = c(m, Tt, n_save))

# 16e. tau2
almacenar_tau2 <- TRUE
tau2_save <- list()
if (almacenar_tau2) {
  for (B in names(bloques)) {
    blk <- bloques[[B]]
    tau2_save[[B]] <- array(NA_real_, dim = c(blk$dim, Tt, n_save))
  }
}

# Muestras guardadas
save_idx <- 0L

# F_mat ya está distribuido en F_lst; datos ya no se usa.
# ── Guardar datos del modelo para diagnósticos ──
# El muestreador lee estos objetos; nunca los modifica.
# saveRDS(list(
#   Y = Y, F_lst = F_lst, N_jkt = N_jkt,
#   n_k = n_k, m = m, Tt = Tt, p = p,
#   p_E = p_E, p_M = p_M, p_D = p_D,
#   N_jk = N_jk, N_kt = N_kt,
#   bloques = bloques, theta_labels = theta_labels,
#   depto_ids = depto_ids, tiempo_ids = tiempo_ids,
#   y_vec = y_vec
# ), file.path(getwd(), "datos_modelo.rds"))
# cat("✓ datos_modelo.rds guardado\n")

#_______________________________________________________________________________
# 17. Validaciones finales ----
#_______________________________________________________________________________

# Positividad y finitud
stopifnot(all(is.finite(theta_curr)))
stopifnot(all(sapply(tau2_curr, function(x) all(x > 0))))
stopifnot(all(lambda2_curr > 0))
for (kk in 1:m) stopifnot(all(v_curr[1:n_k[kk], kk] > 0))
stopifnot(all(s_curr > 0), all(phi_curr > 0))
stopifnot(max(abs(s_curr * phi_curr - 1)) < 1e-10)
stopifnot(all(W_diag > 0), !any(is.na(W_diag)))

# Dimensiones cruzadas
stopifnot(nrow(theta_curr) == p, ncol(theta_curr) == Tt + 1)
stopifnot(nrow(s_curr) == m,     ncol(s_curr) == Tt)
stopifnot(nrow(v_curr) == max(n_k), ncol(v_curr) == m)
stopifnot(nrow(W_diag) == p,     ncol(W_diag) == Tt)
for (B in names(bloques)) {
  blk <- bloques[[B]]
  stopifnot(nrow(tau2_curr[[B]]) == blk$dim, ncol(tau2_curr[[B]]) == Tt)
}
stopifnot(all(m_filt[, 1] == m_0), all(C_filt[,, 1] == C_0))

# Memoria
bytes_total <- (
  prod(dim(theta_curr)) + prod(dim(W_diag)) +
    sum(sapply(tau2_curr, function(x) prod(dim(x)))) +
    length(lambda2_curr) +
    prod(dim(v_curr)) + prod(dim(s_curr)) + prod(dim(phi_curr)) +
    prod(dim(m_filt)) + prod(dim(C_filt)) + prod(dim(SSE_jkt)) +
    prod(dim(n_kt_work)) + prod(dim(d_kt_work)) +
    prod(dim(theta_save)) + prod(dim(lambda2_save)) +
    prod(dim(v_save)) + prod(dim(s_save)) +
    { if (almacenar_tau2) sum(sapply(tau2_save, function(x) prod(dim(x)))) else 0 }
) * 8

cat(sprintf("\n§✓ Validaciones completadas\n"))
cat(sprintf("    Memoria arrays: %.1f MB\n", bytes_total / 1e6))

