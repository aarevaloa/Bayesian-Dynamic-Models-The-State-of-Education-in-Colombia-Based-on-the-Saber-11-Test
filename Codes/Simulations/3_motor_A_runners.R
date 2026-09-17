#_______________________________________________________________________________
#@  MOTOR DE SIMULACION — RUNNERS 
#@  Lee rutas y nombres de archivo desde config_rutas.R 
#_______________________________________________________________________________
# =============================================================================
# LOCALIZADOR DE Rscript
# =============================================================================

.ruta_Rscript <- function() {
  # R.home("bin") apunta a la instalacion actual de R; Rscript esta ahi.
  cand <- file.path(R.home("bin"), if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript")
  if (file.exists(cand)) return(cand)
  # fallback: confiar en el PATH
  "Rscript"
}

# =============================================================================
# CARGA DE FUNCIONES DE CADA MODELO
# =============================================================================
.cargar_en_aislado <- function(ruta, archivos) {
  env <- new.env(parent = globalenv())
  for (f in archivos) {
    ruta_f <- file.path(ruta, f)
    if (!file.exists(ruta_f)) {
      warning(sprintf("Archivo no encontrado: %s", ruta_f)); next
    }
    tryCatch(suppressWarnings(suppressMessages(sys.source(ruta_f, envir = env))),
             error = function(e) invisible(NULL))
  }
  for (nm in ls(env))
    if (is.function(get(nm, envir = env)))
      assign(nm, get(nm, envir = env), envir = globalenv())
  invisible()
}

cargar_funciones_M1 <- function() {
  if (exists("correr_gibbs_disco") && exists("leer_muestras_m1") &&
      exists("crear_hiperparametros")) return(invisible())
  .cargar_en_aislado(RUTA_M1, ARCHIVOS_M1)
}

cargar_funciones_M2 <- function() {
  if (exists("gibbs_sampler_m2") && exists("leer_muestras_m2") &&
      exists("create_hyperparameters")) return(invisible())
  .cargar_en_aislado(RUTA_M2, ARCHIVOS_M2)
}

cargar_funciones_M3 <- function() {
  if (exists("ffbs") && exists("construir_W_diag")) return(invisible())
  .cargar_en_aislado(RUTA_M3, ARCHIVOS_M3_FUNCIONES)
}

# Extrae SOLO la funcion leer_muestras del Gibbs original de M3 (sin correr el cuerpo)
cargar_lector_M3 <- function() {
  if (exists("leer_muestras_m3_fn")) return(invisible())
  src <- readLines(file.path(RUTA_M3, ARCHIVO_M3_GIBBS_ORIGINAL), warn = FALSE)
  i0 <- grep("^leer_muestras <- function", src)[1]
  if (is.na(i0)) stop("No se encontro 'leer_muestras <- function' en ", ARCHIVO_M3_GIBBS_ORIGINAL)
  cierres <- grep("^}", src)
  i1 <- cierres[cierres > i0][1]
  codigo <- paste(src[i0:i1], collapse = "\n")
  codigo <- sub("^leer_muestras <- function", "leer_muestras_m3_fn <- function", codigo)
  eval(parse(text = codigo), envir = globalenv())
  invisible()
}


# =============================================================================
# RUNNER M1 (normal estatico)
# =============================================================================

runner_M1 <- function(panel, truth, dir_base, control) {
  cargar_funciones_M1()
  datos <- adaptar_a_M1(panel, truth, estandarizar = FALSE)
  hiper <- crear_hiperparametros(
    mu_mu = mean(datos$y_vec), nu_mu = 3, gamma_mu = sd(datos$y_vec),
    varsigma2_E = 1, varsigma2_M = 1, varsigma2_D = 1,
    a_lambda_E = 1, b_lambda_E = 1, a_lambda_M = 1, b_lambda_M = 1,
    a_lambda_D = 1, b_lambda_D = 1, nu_kappa = 3, alpha_kappa = 2, beta_kappa = 2)

  dir_out <- file.path(dir_base, "M1")
  if (dir.exists(dir_out)) unlink(dir_out, recursive = TRUE)
  dir.create(dir_out, recursive = TRUE, showWarnings = FALSE)

  t0 <- Sys.time()
  invisible(capture.output(suppressMessages(
    correr_gibbs_disco(datos, hiper, n_iter = control$n_iter, burnin = control$burn_in,
                       thin = control$thin, dir_salida = dir_out, semilla = control$semilla,
                       print_cada = 0, guardar_kappa2_jk = TRUE))))
  seg <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  mu  <- leer_muestras_m1(dir_out, "mu")
  bE  <- leer_muestras_m1(dir_out, "beta_E"); if (is.null(dim(bE))) bE <- matrix(bE, ncol = datos$p_E)
  bM  <- leer_muestras_m1(dir_out, "beta_M"); if (is.null(dim(bM))) bM <- matrix(bM, ncol = datos$p_M)
  bD  <- leer_muestras_m1(dir_out, "beta_D"); if (is.null(dim(bD))) bD <- matrix(bD, ncol = datos$p_D)
  n_save <- length(mu)
  p <- 1L + datos$p_E + datos$p_M + datos$p_D
  theta_draws <- array(NA_real_, dim = c(p, 1L, n_save))
  theta_draws[1, 1, ] <- mu
  if (datos$p_E > 0) theta_draws[2:(1+datos$p_E), 1, ] <- t(bE)
  if (datos$p_M > 0) theta_draws[(2+datos$p_E):(1+datos$p_E+datos$p_M), 1, ] <- t(bM)
  if (datos$p_D > 0) theta_draws[(2+datos$p_E+datos$p_M):p, 1, ] <- t(bD)

  k2jk <- tryCatch(leer_muestras_m1(dir_out, "kappa2_jk"), error = function(e) NULL)

  list(theta_draws = theta_draws,
       theta_labels = c("intercepto", truth$labels$nombres_E,
                        truth$labels$nombres_M, truth$labels$nombres_D),
       kappa2_jk_draws = k2jk, datos = datos,
       extra = list(modelo = "M1", estatico = TRUE), tiempo_seg = seg)
}


# =============================================================================
# RUNNER M2 (t estatico)
# =============================================================================

runner_M2 <- function(panel, truth, dir_base, control) {
  cargar_funciones_M2()
  md <- adaptar_a_M2(panel, truth, estandarizar = FALSE)
  hyper <- create_hyperparameters(
    mu_mu = mean(md$y_vec), nu_mu = 5, gamma_mu2 = var(md$y_vec) * 4,
    nu_kappa = 5, alpha_kappa = 2, beta_kappa = 2, lambda_nu = 0.1)

  dir_out <- file.path(dir_base, "M2")
  if (dir.exists(dir_out)) unlink(dir_out, recursive = TRUE)
  dir.create(dir_out, recursive = TRUE, showWarnings = FALSE)

  t0 <- Sys.time()
  invisible(capture.output(suppressMessages(
    gibbs_sampler_m2(md, hyper, n_iter = control$n_iter, burn_in = control$burn_in,
                     thin = control$thin, dir_salida = dir_out, seed = control$semilla,
                     verbose = 0, guardar_xi = FALSE, guardar_loglik = FALSE))))
  seg <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  beta_mat <- leer_muestras_m2(dir_out, "beta_star")
  nu_ch    <- tryCatch(leer_muestras_m2(dir_out, "nu"), error = function(e) NULL)
  n_save <- nrow(beta_mat); p <- ncol(beta_mat)
  theta_draws <- array(NA_real_, dim = c(p, 1L, n_save))
  for (l in 1:p) theta_draws[l, 1, ] <- beta_mat[, l]

  k2jk <- tryCatch(leer_muestras_m2(dir_out, "kappa2_jk"), error = function(e) NULL)

  list(theta_draws = theta_draws, theta_labels = md$theta_labels,
       kappa2_jk_draws = k2jk, model_data = md,
       extra = list(modelo = "M2", estatico = TRUE, nu = nu_ch), tiempo_seg = seg)
}


# =============================================================================
# RUNNER M3 (normal dinamico)
# =============================================================================

runner_M3 <- function(panel, truth, dir_base, control) {
  dir_out <- file.path(dir_base, "M3")
  if (dir.exists(dir_out)) unlink(dir_out, recursive = TRUE)
  dir.create(dir_out, recursive = TRUE, showWarnings = FALSE)

  saveRDS(list(panel = panel, truth = truth, control = control),
          file.path(dir_out, "_input.rds"))

  # preludio: carga generador + adaptador + funciones M3, define globales
  preludio <- c(
    'args <- commandArgs(trailingOnly=TRUE); dir_out <- args[1]',
    # heredar las librerias del proceso padre (por si R del subproceso no las ve)
    sprintf('.libPaths(c(%s))',
            paste(sprintf('"%s"', .libPaths()), collapse = ", ")),
    'sink(file.path(dir_out, "_log.txt"))',
    sprintf('source("%s")', file.path(RUTA_SIMULACION, ARCHIVO_GENERADOR)),
    sprintf('RUTA_M3 <- "%s"', RUTA_M3),
    sprintf('for (f in c(%s))',
            paste(sprintf('"%s"', ARCHIVOS_M3_FUNCIONES), collapse=", ")),
    '  suppressWarnings(suppressMessages(source(file.path(RUTA_M3, f))))',
    sprintf('source("%s")', file.path(RUTA_SIMULACION, ARCHIVO_ADAPTADOR_M3)),
    'inp <- readRDS(file.path(dir_out, "_input.rds"))',
    'CHECKPOINT_CADA_M3 <- 99999L',
    'preparar_globales_M3(inp$panel, inp$truth, dir_out,',
    '  control = list(n_iter=inp$control$n_iter, burn_in=inp$control$burn_in, thin=inp$control$thin))'
  )
  # cuerpo del Gibbs parcheado, inline (NO via source, para preservar top-level)
  cuerpo <- readLines(file.path(RUTA_M3, ARCHIVO_M3_GIBBS_PATCHED), warn = FALSE)
  runner_script <- file.path(dir_out, "_run.R")
  writeLines(c(preludio, cuerpo, 'sink()'), runner_script)

  t0 <- Sys.time()
  # Ejecutar el subproceso CAPTURANDO stdout/stderr a un archivo, para poder
  # diagnosticar si algo falla (rutas con espacios, Rscript no hallado, etc.).
  log_err <- file.path(dir_out, "_error.txt")
  rc <- tryCatch(
    system2(.ruta_Rscript(), c(shQuote(runner_script), shQuote(dir_out)),
            stdout = log_err, stderr = log_err),
    error = function(e) { writeLines(conditionMessage(e), log_err); 1L })
  seg <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  # Si no se genero metadatos.rds, el subproceso fallo: mostrar el error real.
  if (!file.exists(file.path(dir_out, "metadatos.rds"))) {
    err_txt <- if (file.exists(log_err)) paste(tail(readLines(log_err, warn=FALSE), 15),
                                               collapse = "\n") else "(sin log)"
    stop(sprintf("El subproceso de M3 no genero salida. Codigo=%s. Ultimas lineas del log:\n%s",
                 rc, err_txt), call. = FALSE)
  }

  cargar_lector_M3()
  theta_arr <- leer_muestras_m3_fn(dir_out, "theta")
  p <- dim(theta_arr)[1]; Tt1 <- dim(theta_arr)[2]
  theta_draws <- theta_arr[, 2:Tt1, , drop = FALSE]
  v_arr <- tryCatch(leer_muestras_m3_fn(dir_out, "v"), error = function(e) NULL)
  s_arr <- tryCatch(leer_muestras_m3_fn(dir_out, "s"), error = function(e) NULL)

  list(theta_draws = theta_draws, theta_labels = truth$labels$theta_labels,
       v_draws = v_arr, s_draws = s_arr,
       extra = list(modelo = "M3", estatico = FALSE), tiempo_seg = seg)
}


# =============================================================================
# RUNNER M4 (t dinamico)
# =============================================================================

cargar_funciones_M4 <- function() {
  if (exists("gibbs_modelo4") && exists("ffbs_gaussian") &&
      exists("omega_inv_apilado")) return(invisible())
  if (!requireNamespace("statmod", quietly = TRUE))
    stop("Falta el paquete 'statmod'. Instalalo con: install.packages('statmod')")
  # M4 debe cargarse en ORDEN (dependencias) y en el global directamente, porque
  # las funciones del ensamblador se llaman entre si por nombre global.
  for (f in ARCHIVOS_M4_FUN) {
    ruta_f <- file.path(RUTA_M4, f)
    if (!file.exists(ruta_f)) { warning(sprintf("No encontrado: %s", ruta_f)); next }
    suppressWarnings(suppressMessages(source(ruta_f)))
  }
  invisible()
}

runner_M4 <- function(panel, truth, dir_base, control) {
  cargar_funciones_M4()
  ad <- adaptar_a_M4(panel, truth, estandarizar = FALSE)

  t0 <- Sys.time()
  salida <- gibbs_modelo4(ad$datos, ad$dims, ad$hip,
                          n_iter = control$n_iter, burn_in = control$burn_in,
                          thin = control$thin, semilla = control$semilla, verbose = FALSE)
  seg <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  p <- dim(salida$theta)[1]; Tt1 <- dim(salida$theta)[2]
  theta_draws <- salida$theta[, 2:Tt1, , drop = FALSE]

  list(theta_draws = theta_draws, theta_labels = ad$theta_labels,
       v_mean = salida$v_mean, phi_mean = salida$phi_mean,
       v_draws = salida$v_save, phi_draws = salida$phi_save,
       extra = list(modelo = "M4", estatico = FALSE, nu = salida$nu,
                    tasa_MH = salida$tasa_MH), tiempo_seg = seg)
}


# Registro de runners
RUNNERS <- list(M1 = runner_M1, M2 = runner_M2, M3 = runner_M3, M4 = runner_M4)
