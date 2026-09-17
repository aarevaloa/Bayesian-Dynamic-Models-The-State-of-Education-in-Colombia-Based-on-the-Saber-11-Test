#_______________________________________________________________________________
#@  MODELO 2 — t-STUDENT JERÁRQUICO ESTÁTICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________
# ==============================================================================
# 1. PARÁMETROS DE CONFIGURACIÓN (rutas, nombres de covariables, opciones)
# ==============================================================================

cfg <- list(
  # --- Ruta al CSV (idéntica al Modelo 3) ---
  ruta    = 'C:/Users/USUARIO/OneDrive - Universidad Nacional de Colombia/Tesis/Escritorio',
  archivo = 'Datos_procesados/Python/ICFES_MUESTRA_500000_MODELO.csv',
  
  # --- Nombre de la columna respuesta ---
  resp_col = "punt_global",
  
  # --- Códigos de identificación jerárquica ---
  depto_col = "COD_DEPTO",
  muni_col  = "COD_MUNI",
  
  # --- Columna de periodo temporal (para colapsar contextuales, Opción 1) ---
  periodo_col = "periodo",
  
  # --- Método de agregación temporal de las contextuales ---
  # "media"   -> z_{j,k} = mean_t(z_{j,k,t})  (nivel promedio histórico; estándar)
  # "mediana" -> z_{j,k} = median_t(z_{j,k,t}) (robusto a periodos atípicos, p.ej. 2020)
  metodo_agg_temporal = "media",
  
  # --- Centinela de códigos inválidos ---
  cod_invalido = "99999",
  
  # --- Opción de centrado (A = estandarizar continuas, dummies intactas) ---
  modo_centrado = "A",
  
  # --- Umbrales numéricos ---
  eps_var     = 1e-15,   # varianza por debajo de la cual una columna es "constante"
  eps_sd      = 1e-12,   # SD por debajo de la cual no se estandariza
  tasa_na_flag = 0.01    # tasa de NA por encima de la cual se crea flag _MISSING
)


# --- Definición de covariables por nivel (idéntica al Modelo 3) ---------------
# Nivel individual x_{i,j,k}
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

# Nivel municipal z_{j,k}
z_jk_names <- c(
  "TASA_HOMICIDIOS", "PROP_HOMICIDIOS_URBANO", "TASA_VICTIMAS",
  "TASA_MATRICULA_ES", "IES_CON_OFERTA", "SGP_EDUC_PC",
  "PROP_NBI", "COMP_INASISTENCIA_ESCOLAR", "PROP_URBANA",
  "LOG_POBLACION"
)

# Nivel departamental w_k
w_k_names <- c(
  "PIB_CONSTANTE", "CREC_PIB_REAL", "TASA_DESEMPLEO",
  "TASA_SUBOCUPACION", "IPM"
)


# ==============================================================================
# 2. FUNCIONES DE LECTURA E INTEGRIDAD
# ==============================================================================
importar_datos <- function(cfg) {
  archivo_full <- file.path(cfg$ruta, cfg$archivo)
  if (!file.exists(archivo_full)) {
    stop(sprintf("No se encuentra el archivo: %s", archivo_full))
  }
  datos <- read.csv(archivo_full, fileEncoding = "UTF-8")
  
  # Columnas estructurales obligatorias
  obligatorias <- c(cfg$resp_col, cfg$depto_col, cfg$muni_col)
  faltan <- setdiff(obligatorias, names(datos))
  if (length(faltan) > 0) {
    stop(sprintf("Faltan columnas estructurales: %s",
                 paste(faltan, collapse = ", ")))
  }
  if (nrow(datos) == 0) stop("El archivo no contiene filas.")
  
  cat(sprintf("Filas cargadas : %d\n", nrow(datos)))
  cat(sprintf("Columnas       : %d\n", ncol(datos)))
  datos
}
verificar_covariables <- function(datos, nombres_E, nombres_M, nombres_D) {
  todas <- c(nombres_E, nombres_M, nombres_D)
  faltan <- setdiff(todas, names(datos))
  if (length(faltan) > 0) {
    cat("Variables del modelo faltantes:\n"); print(faltan)
    stop("Corregir nombres de covariables antes de continuar.")
  }
  cat(sprintf("\u2713 Todas las covariables del modelo presentes (%d)\n",
              length(todas)))
  invisible(TRUE)
}


# ==============================================================================
# 3. LIMPIEZA BÁSICA
# ==============================================================================
limpieza_basica <- function(datos, cfg) {
  n0 <- nrow(datos)
  datos <- datos[as.character(datos[[cfg$depto_col]]) != cfg$cod_invalido, ]
  datos <- datos[as.character(datos[[cfg$muni_col]])  != cfg$cod_invalido, ]
  datos <- datos[!is.na(datos[[cfg$resp_col]]), ]
  cat(sprintf("Filas tras limpieza basica: %d (eliminadas %d)\n",
              nrow(datos), n0 - nrow(datos)))
  if (nrow(datos) == 0) stop("No quedan filas tras la limpieza basica.")
  datos
}


# ==============================================================================
# 4. ÍNDICES JERÁRQUICOS SECUENCIALES (k, j) — anidamiento estricto j en k
# ==============================================================================

construir_indices <- function(datos, cfg) {
  depto_chr <- as.character(datos[[cfg$depto_col]])
  muni_chr  <- as.character(datos[[cfg$muni_col]])
  
  # --- Índice de departamento ---
  depto_ids <- sort(unique(depto_chr))
  K <- length(depto_ids)
  datos$k_idx <- match(depto_chr, depto_ids)
  
  # --- Índice de municipio anidado dentro de cada departamento ---
  datos$j_idx <- NA_integer_
  muni_ids <- vector("list", K)
  for (kk in 1:K) {
    idx_k        <- which(datos$k_idx == kk)
    mcpios_k     <- sort(unique(muni_chr[idx_k]))
    muni_ids[[kk]] <- mcpios_k
    datos$j_idx[idx_k] <- match(muni_chr[idx_k], mcpios_k)
  }
  
  # --- Número de municipios por departamento ---
  J_k <- integer(K)
  for (kk in 1:K) J_k[kk] <- length(muni_ids[[kk]])
  
  # --- Validaciones ---
  stopifnot(!any(is.na(datos$k_idx)), !any(is.na(datos$j_idx)))
  for (kk in 1:K) {
    stopifnot(max(datos$j_idx[datos$k_idx == kk]) == J_k[kk])
  }
  
  cat(sprintf("K=%d departamentos, %d municipios totales\n", K, sum(J_k)))
  list(datos = datos, K = K, J_k = J_k,
       depto_ids = depto_ids, muni_ids = muni_ids)
}


# ==============================================================================
# 5. IMPUTACIÓN DE CONTEXTUALES (municipal M y departamental D) + flags
# ==============================================================================
imputar_contextuales <- function(datos, z_names, w_names, cfg) {
  contextuales <- c(z_names, w_names)
  
  # --- Validar columna de periodo y método de agregación ---
  if (!cfg$periodo_col %in% names(datos)) {
    stop(sprintf("No se encuentra la columna de periodo '%s' (necesaria para colapsar contextuales).",
                 cfg$periodo_col))
  }
  metodo <- cfg$metodo_agg_temporal
  if (!metodo %in% c("media", "mediana")) {
    stop(sprintf("metodo_agg_temporal debe ser 'media' o 'mediana', no '%s'.", metodo))
  }
  agg_fun <- if (metodo == "media") {
    function(x) mean(x, na.rm = TRUE)
  } else {
    function(x) median(x, na.rm = TRUE)
  }
  cat(sprintf("Colapso temporal de contextuales: metodo = '%s'\n", metodo))
  
  na_antes <- sapply(datos[, contextuales, drop = FALSE],
                     function(v) mean(is.na(v)))
  cat("Tasa de NA en contextuales (antes del colapso):\n")
  pr <- round(na_antes[na_antes > 0], 4)
  if (length(pr) > 0) print(pr) else cat("  (ninguna con NA)\n")
  
  clave_muni  <- paste(datos$k_idx, datos$j_idx, sep = "_")
  clave_depto <- as.character(datos$k_idx)
  
  # Un valor agregado por celda (NaN si la celda no tiene ningún dato observado)
  for (v in z_names) {
    datos[[v]] <- ave(datos[[v]], clave_muni, FUN = agg_fun)
  }
  for (v in w_names) {
    datos[[v]] <- ave(datos[[v]], clave_depto, FUN = agg_fun)
  }
  # mean/median de un grupo todo-NA con na.rm=TRUE devuelve NaN -> normalizar a NA
  for (v in contextuales) {
    datos[[v]][is.nan(datos[[v]])] <- NA_real_
  }
  
  # --- Diagnóstico: ¿cuántas celdas quedaron sin dato tras el colapso? ---
  na_tras_colapso <- sapply(datos[, contextuales, drop = FALSE],
                            function(v) mean(is.na(v)))
  pr2 <- round(na_tras_colapso[na_tras_colapso > 0], 4)
  if (length(pr2) > 0) {
    cat("Tasa de NA tras el colapso (celdas sin ningun periodo observado):\n")
    print(pr2)
  } else {
    cat("\u2713 Todas las celdas tienen al menos un periodo observado tras el colapso\n")
  }
  
  flags_creados <- character(0)
  for (v in contextuales) {
    if (na_tras_colapso[v] > cfg$tasa_na_flag) {
      flag_nm <- paste0(v, "_MISSING")
      datos[[flag_nm]] <- as.integer(is.na(datos[[v]]))
      flags_creados <- c(flags_creados, flag_nm)
    }
  }
  cat(sprintf("Flags _MISSING creados (celdas sin dato, tasa > %.0f%%): %d\n",
              100 * cfg$tasa_na_flag, length(flags_creados)))
  
  # (3a) mediana por departamento
  for (v in contextuales) {
    if (!any(is.na(datos[[v]]))) next
    datos[[v]] <- ave(datos[[v]], clave_depto, FUN = function(x) {
      x[is.na(x)] <- median(x, na.rm = TRUE); x
    })
  }
  # (3b) mediana global (ultimo recurso)
  for (v in contextuales) {
    if (any(is.na(datos[[v]]))) {
      datos[[v]][is.na(datos[[v]])] <- median(datos[[v]], na.rm = TRUE)
    }
  }
  
  na_despues <- sapply(datos[, contextuales, drop = FALSE],
                       function(v) mean(is.na(v)))
  if (any(na_despues > 0)) stop("Imputacion de contextuales incompleta.")
  cat("\u2713 Imputacion de contextuales completa\n")
  
  max_rango_z <- 0
  for (v in z_names) {
    rangos <- tapply(datos[[v]], clave_muni, function(x) diff(range(x)))
    max_rango_z <- max(max_rango_z, max(rangos, na.rm = TRUE))
  }
  max_rango_w <- 0
  for (v in w_names) {
    rangos <- tapply(datos[[v]], clave_depto, function(x) diff(range(x)))
    max_rango_w <- max(max_rango_w, max(rangos, na.rm = TRUE))
  }
  if (max_rango_z > 1e-8 || max_rango_w > 1e-8) {
    stop(sprintf(paste0("El colapso no produjo constancia por celda ",
                        "(rango_max z=%.2e, w=%.2e). Revisar."),
                 max_rango_z, max_rango_w))
  }
  cat(sprintf("\u2713 Contextuales constantes por celda tras colapso (rango z=%.1e, w=%.1e)\n",
              max_rango_z, max_rango_w))
  
  # ---------------------------------------------------------------------------
  # (4) Anexar flags a sus bloques correspondientes
  # ---------------------------------------------------------------------------
  for (f in flags_creados) {
    base <- sub("_MISSING$", "", f)
    if (base %in% z_names) z_names <- c(z_names, f)
    if (base %in% w_names) w_names <- c(w_names, f)
  }
  
  list(datos = datos, z_names = z_names, w_names = w_names)
}


# ==============================================================================
# 6. ESTANDARIZACIÓN DE CONTINUAS (opción A) + manejo de NA en dummies
# ==============================================================================
estandarizar_covariables <- function(datos, nombres_E, nombres_M, nombres_D, cfg) {
  todas <- c(nombres_E, nombres_M, nombres_D)
  
  # --- (1) Coerción a numérico ---
  n_conv <- 0L
  for (v in todas) {
    if (!is.numeric(datos[[v]])) {
      col_char <- as.character(datos[[v]])
      if (any(col_char %in% c("True", "False"), na.rm = TRUE)) {
        datos[[v]] <- as.numeric(as.logical(col_char))
      } else {
        datos[[v]] <- as.numeric(col_char)
      }
      n_conv <- n_conv + 1L
    }
  }
  if (n_conv > 0) cat(sprintf("Columnas convertidas a numerico: %d\n", n_conv))
  for (v in todas) {
    if (!is.numeric(datos[[v]]))
      stop(sprintf("Columna '%s' no es numerica tras coercion.", v))
  }
  
  # --- (2) Clasificación dummy vs continua ---
  es_dummy <- function(x) all(x %in% c(0, 1, NA))
  vars_continuas <- todas[!sapply(datos[, todas, drop = FALSE], es_dummy)]
  cat(sprintf("Continuas: %d | Dummies: %d\n",
              length(vars_continuas), length(todas) - length(vars_continuas)))
  
  # --- (3) Estandarización de continuas (opción A) ---
  estandar_params <- data.frame(
    variable = vars_continuas, media = NA_real_, desvest = NA_real_,
    n_na = NA_integer_, stringsAsFactors = FALSE
  )
  for (ii in seq_along(vars_continuas)) {
    v    <- vars_continuas[ii]
    mu_v <- mean(datos[[v]], na.rm = TRUE)
    sd_v <- sd(datos[[v]], na.rm = TRUE)
    n_na <- sum(is.na(datos[[v]]))
    
    if (is.na(sd_v) || sd_v < cfg$eps_sd) {
      warning(sprintf("'%s' con SD ~0 o todo NA. No se estandariza.", v))
      sd_v <- 1; if (is.na(mu_v)) mu_v <- 0
    }
    estandar_params$media[ii]   <- mu_v
    estandar_params$desvest[ii] <- sd_v
    estandar_params$n_na[ii]    <- n_na
    
    datos[[v]] <- (datos[[v]] - mu_v) / sd_v
    if (n_na > 0) datos[[v]][is.na(datos[[v]])] <- 0  # media estandarizada
  }
  cat(sprintf("Variables estandarizadas: %d\n", length(vars_continuas)))
  
  # --- (4) NA en dummies -> 0 (categoría de referencia) ---
  todas_dummies <- setdiff(todas, vars_continuas)
  n_na_dum <- 0L
  for (v in todas_dummies) {
    nd <- sum(is.na(datos[[v]]))
    if (nd > 0) { datos[[v]][is.na(datos[[v]])] <- 0L; n_na_dum <- n_na_dum + nd }
  }
  if (n_na_dum > 0)
    cat(sprintf("NAs en dummies reemplazados con 0: %d\n", n_na_dum))
  
  # --- Validación final de ausencia de NA ---
  for (v in todas) {
    if (any(is.na(datos[[v]])))
      stop(sprintf("Quedan NA en la covariable '%s' tras estandarizacion.", v))
  }
  
  list(datos = datos, estandar_params = estandar_params,
       vars_continuas = vars_continuas)
}


# ==============================================================================
# 7. DIAGNÓSTICO Y LIMPIEZA DE COLINEALIDAD
# ==============================================================================
limpiar_colinealidad <- function(datos, nombres_E, nombres_M, nombres_D, cfg) {
  es_dummy <- function(x) all(x %in% c(0, 1, NA))
  
  # ---------- (1) Varianza ~0 ----------
  todas <- c(nombres_E, nombres_M, nombres_D)
  elim_var <- character(0)
  for (v in todas) {
    cv <- var(datos[[v]], na.rm = TRUE)
    if (is.na(cv) || cv < cfg$eps_var) elim_var <- c(elim_var, v)
  }
  if (length(elim_var) > 0) {
    cat(sprintf("Varianza ~0 eliminadas: %d\n", length(elim_var)))
    for (v in elim_var) cat(sprintf("  - %s\n", v))
    nombres_E <- setdiff(nombres_E, elim_var)
    nombres_M <- setdiff(nombres_M, elim_var)
    nombres_D <- setdiff(nombres_D, elim_var)
  }
  
  # ---------- (2) Trampa de dummies (solo nivel E, como en Modelo 3) ----------
  dummies_E <- nombres_E[sapply(datos[, nombres_E, drop = FALSE], es_dummy)]
  
  asignar_grupos <- function(nombres_dummy) {
    grupos <- list()
    for (nm in nombres_dummy) {
      partes <- strsplit(nm, "_")[[1]]
      asignado <- FALSE
      if (length(partes) >= 2) {
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
      }
      if (!asignado) grupos[[nm]] <- nm
    }
    grupos[sapply(grupos, length) >= 2]
  }
  
  grupos_dummy <- asignar_grupos(dummies_E)
  cat(sprintf("Grupos de dummies detectados: %d\n", length(grupos_dummy)))
  elim_trampa <- character(0)
  for (pref in names(grupos_dummy)) {
    grupo <- grupos_dummy[[pref]]
    suma_grupo <- rowSums(datos[, grupo, drop = FALSE])
    rango_suma <- range(suma_grupo, na.rm = TRUE)
    if (abs(rango_suma[2] - rango_suma[1]) < 1e-10) {
      ref <- grupo[length(grupo)]
      cat(sprintf("  %s_ (%d vars, suman=%.0f) -> eliminar '%s'\n",
                  pref, length(grupo), rango_suma[1], ref))
      elim_trampa <- c(elim_trampa, ref)
    } else {
      cat(sprintf("  %s_ (%d vars, suman=[%.1f,%.1f]) -> OK\n",
                  pref, length(grupo), rango_suma[1], rango_suma[2]))
    }
  }
  if (length(elim_trampa) > 0) {
    cat(sprintf("Trampa de dummies eliminadas: %d\n", length(elim_trampa)))
    nombres_E <- setdiff(nombres_E, elim_trampa)
  }
  
  # ---------- (3) QR pivotado sobre [1, todas] ----------
  todas_limpio <- c(nombres_E, nombres_M, nombres_D)
  F_test <- cbind(1, as.matrix(datos[, todas_limpio, drop = FALSE]))
  qr_F <- qr(F_test)
  rk <- qr_F$rank; pc <- ncol(F_test)
  elim_qr <- character(0)
  if (rk < pc) {
    cat(sprintf("Rango %d < %d. Eliminando dependencias por QR...\n", rk, pc))
    cols_dep <- sort(qr_F$pivot[(rk + 1):pc])
    nombres_dep <- c("intercepto", todas_limpio)[cols_dep]
    for (nd in nombres_dep) if (nd != "intercepto") cat(sprintf("  - %s\n", nd))
    elim_qr <- nombres_dep[nombres_dep != "intercepto"]
    nombres_E <- setdiff(nombres_E, elim_qr)
    nombres_M <- setdiff(nombres_M, elim_qr)
    nombres_D <- setdiff(nombres_D, elim_qr)
  } else {
    cat(sprintf("\u2713 Rango completo (%d) tras limpiezas\n", rk))
  }
  
  total_elim <- length(elim_var) + length(elim_trampa) + length(elim_qr)
  cat(sprintf("Eliminadas en total: %d (var0=%d, trampa=%d, QR=%d)\n",
              total_elim, length(elim_var), length(elim_trampa), length(elim_qr)))
  
  list(nombres_E = nombres_E, nombres_M = nombres_M, nombres_D = nombres_D,
       eliminadas = list(varianza = elim_var, trampa = elim_trampa, qr = elim_qr))
}


# ==============================================================================
# 8. MATRICES DE DISEÑO POR NIVEL Y MATRIZ AUMENTADA GLOBAL X*
# ==============================================================================
construir_matrices <- function(datos, nombres_E, nombres_M, nombres_D, resp_col) {
  p_E <- length(nombres_E); p_M <- length(nombres_M); p_D <- length(nombres_D)
  p   <- p_E + p_M + p_D
  
  X_E <- as.matrix(datos[, nombres_E, drop = FALSE])
  X_M <- as.matrix(datos[, nombres_M, drop = FALSE])
  X_D <- as.matrix(datos[, nombres_D, drop = FALSE])
  
  X_star <- cbind(Intercept = 1, X_E, X_M, X_D)
  y_vec  <- datos[[resp_col]]
  
  # --- Índices de bloque dentro de beta* (1 = intercepto) ---
  idx_mu <- 1L
  idx_E  <- if (p_E > 0) (2L):(1L + p_E) else integer(0)
  idx_M  <- if (p_M > 0) (1L + p_E + 1L):(1L + p_E + p_M) else integer(0)
  idx_D  <- if (p_D > 0) (1L + p_E + p_M + 1L):(1L + p) else integer(0)
  
  theta_labels <- c("intercepto", nombres_E, nombres_M, nombres_D)
  
  # --- Validaciones ---
  stopifnot(ncol(X_E) == p_E, ncol(X_M) == p_M, ncol(X_D) == p_D)
  stopifnot(ncol(X_star) == 1L + p)
  stopifnot(all(X_star[, 1] == 1))
  stopifnot(is.numeric(X_star), !any(is.na(X_star)), !any(is.na(y_vec)))
  rk <- qr(X_star)$rank
  if (rk < 1L + p)
    stop(sprintf("X_star rango %d < %d tras limpieza.", rk, 1L + p))
  cat(sprintf("\u2713 X_star %d x %d, rango completo (%d)\n",
              nrow(X_star), 1L + p, rk))
  cat(sprintf("p final: %d (1 + %d E + %d M + %d D)\n", p, p_E, p_M, p_D))
  
  list(y_vec = y_vec, X_E = X_E, X_M = X_M, X_D = X_D, X_star = X_star,
       p_E = p_E, p_M = p_M, p_D = p_D, p = p,
       idx_mu = idx_mu, idx_E = idx_E, idx_M = idx_M, idx_D = idx_D,
       theta_labels = theta_labels)
}


# ==============================================================================
# 9. CONTEOS, MAPEO DE FILAS Y ESTRUCTURAS ANIDADAS
# ==============================================================================
construir_estructuras <- function(datos, mats, hier) {
  K <- hier$K; J_k <- hier$J_k
  N <- nrow(datos)
  maxJ <- max(J_k)
  
  k_idx <- datos$k_idx
  j_idx <- datos$j_idx
  
  # --- Mapeo fila -> (k,j) (orden idéntico al de X_star/y_vec) ---
  row_k <- k_idx
  row_j <- j_idx
  
  # --- Tamaños muestrales por municipio ---
  n_jk <- matrix(0L, nrow = maxJ, ncol = K)
  for (i in 1:N) n_jk[j_idx[i], k_idx[i]] <- n_jk[j_idx[i], k_idx[i]] + 1L
  stopifnot(sum(n_jk) == N)
  
  # --- Particionar filas por celda (k,j) de una sola pasada ---
  clave <- paste(k_idx, j_idx, sep = "_")
  idx_por_celda <- split(seq_len(N), clave)
  
  # --- Esqueleto de estructuras anidadas ---
  Y          <- vector("list", K)
  X_E_lst    <- vector("list", K)
  X_M_lst    <- vector("list", K)
  X_D_lst    <- vector("list", K)
  X_star_lst <- vector("list", K)
  idx_obs    <- vector("list", K)
  for (kk in 1:K) {
    Y[[kk]]          <- vector("list", J_k[kk])
    X_E_lst[[kk]]    <- vector("list", J_k[kk])
    X_M_lst[[kk]]    <- vector("list", J_k[kk])
    X_D_lst[[kk]]    <- vector("list", J_k[kk])
    X_star_lst[[kk]] <- vector("list", J_k[kk])
    idx_obs[[kk]]    <- vector("list", J_k[kk])
  }
  
  # --- Contextuales compactas (un valor por celda) ---
  z_jk <- vector("list", K)  # z_jk[[k]] : matriz J_k[k] x p_M
  w_k  <- matrix(NA_real_, nrow = K, ncol = mats$p_D)  # w_k[k, ] : 1 x p_D
  for (kk in 1:K) z_jk[[kk]] <- matrix(NA_real_, nrow = J_k[kk], ncol = mats$p_M)
  
  # --- Llenado ---
  for (nm in names(idx_por_celda)) {
    partes <- as.integer(strsplit(nm, "_", fixed = TRUE)[[1]])
    kk <- partes[1]; jj <- partes[2]
    sel <- idx_por_celda[[nm]]
    
    Y[[kk]][[jj]]          <- mats$y_vec[sel]
    X_E_lst[[kk]][[jj]]    <- mats$X_E[sel, , drop = FALSE]
    X_M_lst[[kk]][[jj]]    <- mats$X_M[sel, , drop = FALSE]
    X_D_lst[[kk]][[jj]]    <- mats$X_D[sel, , drop = FALSE]
    X_star_lst[[kk]][[jj]] <- mats$X_star[sel, , drop = FALSE]
    idx_obs[[kk]][[jj]]    <- sel
    
    # Contextual: constante dentro de la celda -> tomar la primera fila
    if (mats$p_M > 0) z_jk[[kk]][jj, ] <- mats$X_M[sel[1], ]
    if (mats$p_D > 0) w_k[kk, ]        <- mats$X_D[sel[1], ]
  }
  
  # --- Validaciones de coherencia ---
  for (kk in 1:K) for (jj in 1:J_k[kk]) {
    njk <- n_jk[jj, kk]
    stopifnot(length(Y[[kk]][[jj]]) == njk)
    stopifnot(nrow(X_star_lst[[kk]][[jj]]) == njk)
    stopifnot(length(idx_obs[[kk]][[jj]]) == njk)
  }
  
  cat(sprintf("\u2713 Estructuras anidadas: %d celdas (k,j), N=%d obs\n",
              sum(J_k), N))
  
  list(N = N, n_jk = n_jk, row_k = row_k, row_j = row_j,
       Y = Y, X_E_lst = X_E_lst, X_M_lst = X_M_lst, X_D_lst = X_D_lst,
       X_star_lst = X_star_lst, idx_obs = idx_obs,
       z_jk = z_jk, w_k = w_k)
}


# ==============================================================================
# 10. VERIFICACIÓN DE CONSTANCIA DE CONTEXTUALES DENTRO DE GRUPO
# ==============================================================================
verificar_constancia_contextual <- function(datos, nombres_M, nombres_D,
                                            tol = 1e-8) {
  problemas <- character(0)
  
  # --- Municipales: constantes dentro de (k,j) ---
  if (length(nombres_M) > 0) {
    celda_kj <- paste(datos$k_idx, datos$j_idx, sep = "_")
    for (v in nombres_M) {
      rangos <- tapply(datos[[v]], celda_kj,
                       function(x) diff(range(x, na.rm = TRUE)))
      if (any(rangos > tol, na.rm = TRUE)) problemas <- c(problemas, paste0("M:", v))
    }
  }
  # --- Departamentales: constantes dentro de k ---
  if (length(nombres_D) > 0) {
    for (v in nombres_D) {
      rangos <- tapply(datos[[v]], datos$k_idx,
                       function(x) diff(range(x, na.rm = TRUE)))
      if (any(rangos > tol, na.rm = TRUE)) problemas <- c(problemas, paste0("D:", v))
    }
  }
  
  if (length(problemas) > 0) {
    warning(sprintf("Covariables NO constantes dentro de su celda: %s",
                    paste(problemas, collapse = ", ")))
    cat("  ADVERTENCIA: revisar la asignacion de nivel de estas covariables.\n")
  } else {
    cat("\u2713 Contextuales constantes dentro de su celda (M en muni, D en depto)\n")
  }
  invisible(length(problemas) == 0)
}


# ==============================================================================
# 11. ENSAMBLAJE DEL OBJETO MAESTRO model_data
# ==============================================================================
ensamblar_model_data <- function(hier, mats, estr, est_p, cfg) {
  model_data <- list(
    
    # ---- Dimensiones ----
    N   = estr$N,
    K   = hier$K,
    J_k = hier$J_k,
    n_jk = estr$n_jk,
    p   = mats$p,
    p_E = mats$p_E,
    p_M = mats$p_M,
    p_D = mats$p_D,
    
    # ---- Índices de bloque dentro de beta* ----
    idx_mu = mats$idx_mu,
    idx_E  = mats$idx_E,
    idx_M  = mats$idx_M,
    idx_D  = mats$idx_D,
    
    # ---- Mapeo fila -> (k,j) (alineado con y_vec y X_star) ----
    row_k = estr$row_k,
    row_j = estr$row_j,
    
    # ---- Datos observados (forma global apilada) ----
    y_vec  = mats$y_vec,
    X_E    = mats$X_E,
    X_M    = mats$X_M,
    X_D    = mats$X_D,
    X_star = mats$X_star,
    
    # ---- Estructuras anidadas [[k]][[j]] ----
    Y          = estr$Y,
    X_E_lst    = estr$X_E_lst,
    X_M_lst    = estr$X_M_lst,
    X_D_lst    = estr$X_D_lst,
    X_star_lst = estr$X_star_lst,
    idx_obs    = estr$idx_obs,
    
    # ---- Contextuales compactas ----
    z_jk = estr$z_jk,
    w_k  = estr$w_k,
    
    # ---- Etiquetas y diccionarios de reconstrucción ----
    theta_labels = mats$theta_labels,
    depto_ids    = hier$depto_ids,
    muni_ids     = hier$muni_ids,
    
    # ---- Metadatos ----
    estandar_params = est_p,
    cfg             = cfg
  )
  class(model_data) <- "model_data_m2"
  model_data
}


# ==============================================================================
# 12. VALIDACIÓN INTEGRAL DE model_data
# ==============================================================================
validar_model_data <- function(md) {
  # ---- (a) Dimensiones básicas ----
  stopifnot(md$p == md$p_E + md$p_M + md$p_D)
  stopifnot(length(md$y_vec) == md$N)
  stopifnot(nrow(md$X_star) == md$N, ncol(md$X_star) == 1L + md$p)
  stopifnot(nrow(md$X_E) == md$N, ncol(md$X_E) == md$p_E)
  stopifnot(nrow(md$X_M) == md$N, ncol(md$X_M) == md$p_M)
  stopifnot(nrow(md$X_D) == md$N, ncol(md$X_D) == md$p_D)
  stopifnot(length(md$row_k) == md$N, length(md$row_j) == md$N)
  
  # ---- (b) Intercepto y bloques ----
  stopifnot(all(md$X_star[, 1] == 1))
  stopifnot(md$idx_mu == 1L)
  stopifnot(length(md$idx_E) == md$p_E,
            length(md$idx_M) == md$p_M,
            length(md$idx_D) == md$p_D)
  # Partición exacta de columnas (sin solapes ni huecos)
  if (md$p > 0) {
    todos_idx <- c(md$idx_mu, md$idx_E, md$idx_M, md$idx_D)
    stopifnot(identical(sort(todos_idx), 1:(1L + md$p)))
  }
  
  # ---- (c) Consistencia jerárquica de conteos ----
  stopifnot(length(md$J_k) == md$K)
  stopifnot(sum(md$n_jk) == md$N)
  for (kk in 1:md$K) {
    # municipios con observaciones coinciden con J_k (sin municipios vacíos)
    stopifnot(all(md$n_jk[1:md$J_k[kk], kk] > 0))
    if (md$J_k[kk] < nrow(md$n_jk))
      stopifnot(all(md$n_jk[(md$J_k[kk] + 1):nrow(md$n_jk), kk] == 0))
  }
  
  # ---- (d) Correspondencia estructuras anidadas <-> global ----
  total_obs_anid <- 0L
  for (kk in 1:md$K) for (jj in 1:md$J_k[kk]) {
    njk <- md$n_jk[jj, kk]
    sel <- md$idx_obs[[kk]][[jj]]
    stopifnot(length(sel) == njk)
    stopifnot(length(md$Y[[kk]][[jj]]) == njk)
    stopifnot(nrow(md$X_star_lst[[kk]][[jj]]) == njk)
    # los índices apuntan a las filas correctas de la matriz global
    stopifnot(all(md$row_k[sel] == kk), all(md$row_j[sel] == jj))
    stopifnot(identical(md$Y[[kk]][[jj]], md$y_vec[sel]))
    total_obs_anid <- total_obs_anid + njk
  }
  stopifnot(total_obs_anid == md$N)   # sin observaciones huérfanas
  
  # ---- (e) Ausencia de NA / Inf ----
  stopifnot(!any(is.na(md$y_vec)), all(is.finite(md$y_vec)))
  stopifnot(!any(is.na(md$X_star)), all(is.finite(md$X_star)))
  
  # ---- (f) Diccionarios ----
  stopifnot(length(md$depto_ids) == md$K)
  stopifnot(length(md$muni_ids) == md$K)
  for (kk in 1:md$K) stopifnot(length(md$muni_ids[[kk]]) == md$J_k[kk])
  
  cat("\u2713\u2713 model_data validado: todas las invariantes se cumplen\n")
  invisible(TRUE)
}


# ==============================================================================
# 13. ORQUESTADOR: construir model_data
# ==============================================================================
construir_model_data <- function(cfg, nombres_E, nombres_M, nombres_D) {
  cat("================ ETAPA 1: ESTRUCTURACION DE DATOS ================\n\n")
  
  cat("--- [1] Lectura e integridad ---\n")
  datos <- importar_datos(cfg)
  verificar_covariables(datos, nombres_E, nombres_M, nombres_D)
  
  cat("\n--- [2] Limpieza basica ---\n")
  datos <- limpieza_basica(datos, cfg)
  
  cat("\n--- [3] Indices jerarquicos ---\n")
  hier <- construir_indices(datos, cfg)
  datos <- hier$datos
  
  cat("\n--- [4] Imputacion de contextuales ---\n")
  imp <- imputar_contextuales(datos, nombres_M, nombres_D, cfg)
  datos     <- imp$datos
  nombres_M <- imp$z_names
  nombres_D <- imp$w_names
  
  cat("\n--- [5] Estandarizacion (opcion A) ---\n")
  est <- estandarizar_covariables(datos, nombres_E, nombres_M, nombres_D, cfg)
  datos <- est$datos
  
  cat("\n--- [6] Verificacion de constancia contextual ---\n")
  verificar_constancia_contextual(datos, nombres_M, nombres_D)
  
  cat("\n--- [7] Limpieza de colinealidad ---\n")
  col <- limpiar_colinealidad(datos, nombres_E, nombres_M, nombres_D, cfg)
  nombres_E <- col$nombres_E
  nombres_M <- col$nombres_M
  nombres_D <- col$nombres_D
  
  cat("\n--- [8] Matrices de diseno ---\n")
  mats <- construir_matrices(datos, nombres_E, nombres_M, nombres_D, cfg$resp_col)
  
  cat("\n--- [9] Estructuras anidadas ---\n")
  estr <- construir_estructuras(datos, mats, hier)
  
  cat("\n--- [10] Ensamblaje y validacion ---\n")
  est_p <- est$estandar_params[
    est$estandar_params$variable %in% c(nombres_E, nombres_M, nombres_D), ]
  model_data <- ensamblar_model_data(hier, mats, estr, est_p, cfg)
  validar_model_data(model_data)
  
  cat("\n================ ETAPA 1 COMPLETADA ================\n")
  model_data
}


# ==============================================================================
# 14. EJECUCIÓN
# ==============================================================================

# model_data <- construir_model_data(cfg, x_ijk_names, z_jk_names, w_k_names)

# Resumen rápido:
# str(model_data, max.level = 1)

