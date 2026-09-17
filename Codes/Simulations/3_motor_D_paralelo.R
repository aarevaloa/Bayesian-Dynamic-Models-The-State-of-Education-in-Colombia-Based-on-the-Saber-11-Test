#_______________________________________________________________________________
#@  MOTOR DE SIMULACION 
#@  Corre las R replicas en paralelo (cada worker = una replica completa)
#_______________________________________________________________________________

configurar_paralelo <- function(n_workers = NULL, verbose = TRUE) {
  if (!requireNamespace("future", quietly = TRUE) ||
      !requireNamespace("furrr", quietly = TRUE))
    stop("Faltan paquetes. Instala: install.packages(c('future','furrr'))")

  # nucleos fisicos (sin hyperthreading); fallback a logicos si no se detectan
  n_fis <- tryCatch(parallel::detectCores(logical = FALSE), error = function(e) NA)
  if (is.na(n_fis) || n_fis < 1) n_fis <- parallel::detectCores(logical = TRUE)
  if (is.null(n_workers)) n_workers <- max(1L, n_fis - 1L)
  n_workers <- min(n_workers, parallel::detectCores(logical = TRUE))

  future::plan(future::multisession, workers = n_workers)
  if (verbose)
    cat(sprintf("  Paralelizacion ACTIVA: %d workers (de %d nucleos fisicos)\n",
                n_workers, n_fis))
  invisible(n_workers)
}

# Vuelve a modo secuencial (un solo proceso).
configurar_serie <- function(verbose = TRUE) {
  if (requireNamespace("future", quietly = TRUE))
    future::plan(future::sequential)
  if (verbose) cat("  Modo SERIE activo (sin paralelizacion).\n")
  invisible(TRUE)
}


# =============================================================================
# CUERPO DE UNA REPLICA (funcion pura, ejecutable en un worker)
# =============================================================================
.correr_una_replica <- function(r, cfg, modelos, control, semilla_base,
                                ref_estatico, dir_trabajo) {
  cfg_r <- cfg
  cfg_r$seed <- semilla_base + r
  control_r <- control; control_r$semilla <- semilla_base + r
  sim <- simulate_panel(cfg_r)
  sufijo_unico <- sprintf("rep%03d_p%d_%s", r, Sys.getpid(),
                          format(Sys.time(), "%H%M%OS3"))
  sufijo_unico <- gsub("[^A-Za-z0-9_]", "", sufijo_unico)
  dir_r <- file.path(dir_trabajo, sufijo_unico)
  dir.create(dir_r, recursive = TRUE, showWarnings = FALSE)

  met_r <- list()
  for (mod in modelos) {
    m <- .correr_modelo_seguro(mod, sim$panel, sim$truth, dir_r, control_r, ref_estatico)
    if (!is.null(m)) met_r[[mod]] <- m
  }
  unlink(dir_r, recursive = TRUE)
  met_r
}


# =============================================================================
# CORRER ESCENARIO EN PARALELO
# =============================================================================
correr_escenario_par <- function(cfg, modelos = c("M1","M2","M3","M4"), R = 20L,
                                 control = list(n_iter = 2000L, burn_in = 1000L, thin = 2L),
                                 id_escenario = "S00", semilla_base = 1000L,
                                 ref_estatico = "theta_bar",
                                 dir_trabajo = file.path(RUTA_TRABAJO, "par"),
                                 archivo_motor = NULL,
                                 archivo_paralelo = NULL,
                                 verbose = TRUE) {
  dir.create(dir_trabajo, recursive = TRUE, showWarnings = FALSE)
  if (verbose) cat(sprintf("\n=== [PARALELO] Escenario %s | modelos: %s | R=%d ===\n",
                           id_escenario, paste(modelos, collapse=","), R))
  if (is.null(archivo_motor))
    archivo_motor <- if (exists("ARCHIVO_MOTOR")) ARCHIVO_MOTOR else "02_motor.R"
  if (is.null(archivo_paralelo))
    archivo_paralelo <- if (exists("ARCHIVO_PARALELO")) ARCHIVO_PARALELO else "02_motor_D_paralelo.R"

  # rutas absolutas para que el worker las cargue sin depender del setwd
  .ruta_sim_abs   <- RUTA_SIMULACION
  .ruta_motor_abs <- file.path(RUTA_SIMULACION, archivo_motor)
  .ruta_par_abs   <- file.path(RUTA_SIMULACION, archivo_paralelo)
  # verificar que existen ANTES de lanzar los workers (mensaje claro si no)
  if (!file.exists(.ruta_motor_abs))
    stop(sprintf("No encuentro el motor para los workers: %s\n  (declara ARCHIVO_MOTOR en config_rutas.R con el nombre real)", .ruta_motor_abs))
  if (!file.exists(.ruta_par_abs))
    stop(sprintf("No encuentro el modulo paralelo para los workers: %s\n  (declara ARCHIVO_PARALELO en config_rutas.R con el nombre real)", .ruta_par_abs))

  t0 <- Sys.time()
  met_por_replica <- furrr::future_map(
    1:R,
    function(r) {
      # arranque del worker: cargar el motor completo por RUTA ABSOLUTA
      setwd(.ruta_sim_abs)
      source(.ruta_motor_abs)             # config_rutas + generador + adaptadores + motor
      source(.ruta_par_abs)               # .correr_una_replica
      .correr_una_replica(r, cfg, modelos, control, semilla_base,
                          ref_estatico, dir_trabajo)
    },
    .options = furrr::furrr_options(
      seed = TRUE,
      globals = c("cfg","modelos","control","semilla_base","ref_estatico",
                  "dir_trabajo",".ruta_sim_abs",".ruta_motor_abs",".ruta_par_abs"),
      packages = c("statmod")
    ),
    .progress = verbose
  )
  dur <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  if (verbose) cat(sprintf("  [PARALELO] %d replicas en %.1f min\n", R, dur))

  agg <- agregar_metricas(met_por_replica, modelos)
  list(id = id_escenario, config = cfg, modelos = modelos, R = R,
       control = control, por_replica = met_por_replica, agregado = agg,
       tiempo_min = dur, paralelo = TRUE)
}
