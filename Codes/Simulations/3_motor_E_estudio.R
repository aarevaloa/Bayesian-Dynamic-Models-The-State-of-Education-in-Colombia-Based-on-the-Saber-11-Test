#_______________________________________________________________________________
#@  ORQUESTADOR DEL ESTUDIO COMPLETO
#@  Corre un conjunto de escenarios del catalogo, guardando cada uno al terminar
#_______________________________________________________________________________

.dir_estudio <- function() file.path(RUTA_SIMULACION, "resultados_estudio")


# =============================================================================
# CORRER EL ESTUDIO
# =============================================================================
correr_estudio <- function(ids = NULL, R = 20L,
                           control = list(n_iter = 2000L, burn_in = 1000L, thin = 2L),
                           usar_paralelo = FALSE, forzar = FALSE,
                           semilla_base = 1000L, verbose = TRUE) {
  dir_est <- .dir_estudio()
  dir.create(dir_est, recursive = TRUE, showWarnings = FALSE)

  if (is.null(ids)) ids <- names(CATALOGO_ESCENARIOS)
  ids <- intersect(ids, names(CATALOGO_ESCENARIOS))
  if (length(ids) == 0) stop("Ningun id valido. Usa listar_escenarios().")

  # detectar ya hechos
  hechos <- character(0)
  for (id in ids) {
    f <- file.path(dir_est, sprintf("escenario_%s.rds", id))
    if (file.exists(f) && !forzar) hechos <- c(hechos, id)
  }
  pendientes <- setdiff(ids, hechos)

  if (verbose) {
    cat(sprintf("\n===== ESTUDIO: %d escenarios solicitados =====\n", length(ids)))
    if (length(hechos) > 0)
      cat(sprintf("  Ya hechos (se omiten): %s\n", paste(hechos, collapse=", ")))
    cat(sprintf("  Por correr: %s\n", paste(pendientes, collapse=", ")))
    if (usar_paralelo) cat("  Modo: PARALELO\n") else cat("  Modo: SERIE\n")
  }

  # correr los pendientes, uno por uno, guardando al terminar
  for (i in seq_along(pendientes)) {
    id <- pendientes[i]
    e <- CATALOGO_ESCENARIOS[[id]]
    cfg <- e$fn()
    modelos <- e$modelos
    # offset de semilla por escenario (para que no compartan paneles)
    sem_esc <- semilla_base + which(names(CATALOGO_ESCENARIOS) == id) * 1000L

    if (verbose)
      cat(sprintf("\n[%d/%d] Escenario %s — %s\n", i, length(pendientes), id, e$desc))

    t0 <- Sys.time()
    res <- tryCatch({
      if (usar_paralelo) {
        correr_escenario_par(cfg, modelos = modelos, R = R, control = control,
                             id_escenario = id, semilla_base = sem_esc,
                             ref_estatico = "theta_bar", verbose = verbose)
      } else {
        correr_escenario(cfg, modelos = modelos, R = R, control = control,
                        id_escenario = id, semilla_base = sem_esc,
                        ref_estatico = "theta_bar",
                        dir_trabajo = file.path(RUTA_TRABAJO, id), verbose = verbose)
      }
    }, error = function(err) {
      cat(sprintf("  ERROR en escenario %s: %s\n", id, conditionMessage(err)))
      NULL
    })

    if (!is.null(res)) {
      res$tiempo_total_min <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
      saveRDS(res, file.path(dir_est, sprintf("escenario_%s.rds", id)))
      if (verbose)
        cat(sprintf("  -> guardado (%.1f min)\n", res$tiempo_total_min))
    }
  }

  if (verbose) cat("\n===== ESTUDIO: tanda completada =====\n")
  invisible(.dir_estudio())
}


# =============================================================================
# GENERAR TABLA-RESUMEN  (consolida todos los .rds en una tabla comparativa)
# =============================================================================
generar_tabla_resumen <- function(archivo_csv = NULL, verbose = TRUE) {
  dir_est <- .dir_estudio()
  archivos <- list.files(dir_est, pattern = "^escenario_.*\\.rds$", full.names = TRUE)
  if (length(archivos) == 0) {
    cat("No hay resultados en", dir_est, "\n"); return(invisible(NULL))
  }

  filas <- list()
  for (f in archivos) {
    res <- readRDS(f)
    id <- res$id
    for (mod in res$modelos) {
      a <- res$agregado[[mod]]
      if (is.null(a)) next
      cr <- a$coef; getm <- function(nm) {
        v <- cr$media[cr$metrica == nm]; if (length(v)==0) NA else v
      }
      esp <- a$especificacion
      filas[[length(filas)+1L]] <- data.frame(
        escenario = id,
        modelo = mod,
        especificacion = if (is.null(esp)) NA else esp$estado,
        n_rep_ok = a$n_replicas_ok,
        rmse = getm("rmse"),
        mae = getm("mae"),
        sesgo = getm("sesgo_medio"),
        cobertura95 = getm("cobertura95"),
        amplitud = getm("amplitud_media"),
        ess_min = getm("ess_min"),
        cor_verdad_post = getm("cor_verdad_post"),
        lasso_sens = if (!is.null(a$seleccion)) a$seleccion$sensibilidad else NA,
        lasso_espec = if (!is.null(a$seleccion)) a$seleccion$especificidad else NA,
        var_ratio = if (!is.null(a$varianza)) a$varianza$ratio_post_true else NA,
        var_err_mediano = if (!is.null(a$varianza)) a$varianza$error_rel_mediano else NA,
        nu_post = if (!is.null(a$nu)) a$nu$nu_post_media else NA,
        nu_cobertura = if (!is.null(a$nu)) a$nu$nu_cobertura else NA,
        waic = if (!is.null(a$waic)) a$waic$waic else NA,
        waic_lppd = if (!is.null(a$waic)) a$waic$lppd else NA,
        waic_pwaic = if (!is.null(a$waic)) a$waic$p_waic else NA,
        tiempo_medio_seg = a$tiempo_medio_seg,
        stringsAsFactors = FALSE)
    }
  }
  tabla <- do.call(rbind, filas)
  tabla <- tabla[order(tabla$escenario, tabla$modelo), ]

  if (is.null(archivo_csv))
    archivo_csv <- file.path(dir_est, "tabla_resumen.csv")
  write.csv(tabla, archivo_csv, row.names = FALSE)
  if (verbose) {
    cat(sprintf("Tabla-resumen: %d filas (%d escenarios x modelos)\n",
                nrow(tabla), length(archivos)))
    cat(sprintf("Guardada en: %s\n", archivo_csv))
  }
  tabla
}


# =============================================================================
# IMPRIMIR PROGRESO DEL ESTUDIO  (que escenarios estan hechos)
# =============================================================================
estado_estudio <- function() {
  dir_est <- .dir_estudio()
  todos <- names(CATALOGO_ESCENARIOS)
  cat(sprintf("\nEstado del estudio (%s):\n", dir_est))
  for (id in todos) {
    f <- file.path(dir_est, sprintf("escenario_%s.rds", id))
    if (file.exists(f)) {
      res <- readRDS(f)
      cat(sprintf("  [HECHO] %s  (R=%d, %.1f min)\n", id, res$R,
                  if (!is.null(res$tiempo_total_min)) res$tiempo_total_min else NA))
    } else {
      cat(sprintf("  [     ] %s  (pendiente)\n", id))
    }
  }
  invisible(NULL)
}


# =============================================================================
# VALIDACION DEL WAIC COMO SELECTOR DE MODELOS
# =============================================================================
validar_waic_seleccion <- function(verbose = TRUE) {
  dir_est <- .dir_estudio()
  archivos <- list.files(dir_est, pattern = "^escenario_.*\\.rds$", full.names = TRUE)
  if (length(archivos) == 0) { cat("No hay resultados.\n"); return(invisible(NULL)) }

  filas <- list()
  for (f in archivos) {
    res <- readRDS(f)
    id <- res$id
    # recolectar WAIC y especificacion de cada modelo
    mods <- res$modelos
    waics <- sapply(mods, function(m) {
      a <- res$agregado[[m]]
      if (!is.null(a) && !is.null(a$waic)) a$waic$waic else NA_real_
    })
    especs <- sapply(mods, function(m) {
      a <- res$agregado[[m]]
      if (!is.null(a) && !is.null(a$especificacion)) a$especificacion$estado else NA_character_
    })
    # si no hay WAIC en este escenario, omitir
    if (all(is.na(waics))) next

    # modelo de menor WAIC (el que el WAIC "elige")
    elegido <- names(waics)[which.min(waics)]
    # modelo(s) correctamente especificado(s)
    correcto <- names(especs)[which(especs == "correcta")]
    correcto <- if (length(correcto) == 0) NA_character_ else correcto[1]
    # margen: diferencia entre el mejor y el segundo mejor WAIC
    waics_ord <- sort(waics)
    margen <- if (sum(is.finite(waics)) >= 2) waics_ord[2] - waics_ord[1] else NA_real_

    filas[[length(filas)+1L]] <- data.frame(
      escenario = id,
      waic_elige = elegido,
      especificado = correcto,
      acierto = !is.na(correcto) && elegido == correcto,
      waic_min = min(waics, na.rm = TRUE),
      margen_2do = margen,
      stringsAsFactors = FALSE)
  }
  if (length(filas) == 0) {
    cat("Ningun escenario tiene WAIC. Corre con control$calcular_waic = TRUE.\n")
    return(invisible(NULL))
  }
  tabla <- do.call(rbind, filas)
  tabla <- tabla[order(tabla$escenario), ]

  if (verbose) {
    n_con_correcto <- sum(!is.na(tabla$especificado))
    n_acierto <- sum(tabla$acierto)
    cat("=== Validacion del WAIC como selector de modelo ===\n")
    cat(sprintf("Escenarios con WAIC: %d\n", nrow(tabla)))
    cat(sprintf("Escenarios con un modelo correctamente especificado: %d\n", n_con_correcto))
    if (n_con_correcto > 0)
      cat(sprintf("Aciertos del WAIC (elige el correcto): %d de %d (%.0f%%)\n",
                  n_acierto, n_con_correcto, 100 * n_acierto / n_con_correcto))
    cat("\nDetalle por escenario:\n")
    print(tabla, row.names = FALSE)
  }
  invisible(tabla)
}
