#_______________________________________________________________________________
#@  MODELO 2 — t-STUDENT JERÁRQUICO ESTÁTICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________

loglik_t_pointwise <- function(model_data, state) {
  md <- model_data; st <- state
  zeta <- drop(md$X_star %*% st$beta_star)
  kap  <- expand_kappa2_to_obs(md, st$kappa2_jk)   # kappa2 por observacion
  nu   <- st$nu
  eps2 <- (md$y_vec - zeta)^2
  lgamma((nu + 1) / 2) - lgamma(nu / 2) -
    0.5 * log(nu * pi * kap) -
    ((nu + 1) / 2) * log1p(eps2 / (nu * kap))
}


# ==============================================================================
# Utilidades de extracción de celdas kappa2_jk
# ==============================================================================

flatten_kappa2_jk <- function(model_data, kappa2_jk) {
  md <- model_data
  out <- numeric(sum(md$J_k))
  pos <- 1L
  for (kk in 1:md$K) {
    Jk <- md$J_k[kk]
    out[pos:(pos + Jk - 1L)] <- kappa2_jk[1:Jk, kk]
    pos <- pos + Jk
  }
  out
}


# ==============================================================================
# Reporte de progreso detallado
# ==============================================================================
reporte_progreso <- function(iter, n_iter, state, model_data, t_ini,
                             guardadas, fase) {
  md <- model_data; st <- state
  
  # --- Tiempo y ETA ---
  el  <- as.numeric(difftime(Sys.time(), t_ini, units = "secs"))
  s_it <- el / iter
  eta <- s_it * (n_iter - iter)
  fmt_t <- function(s) {
    if (s < 60) sprintf("%.0fs", s)
    else if (s < 3600) sprintf("%.1fmin", s / 60)
    else sprintf("%.1fh", s / 3600)
  }
  
  # --- Resumen de bloques (estadisticos compactos) ---
  mu_val   <- st$mu
  # magnitud de coeficientes por nivel (norma media |beta|) y nº "activos"
  norma_E <- if (md$p_E > 0) mean(abs(st$beta_E)) else NA
  norma_M <- if (md$p_M > 0) mean(abs(st$beta_M)) else NA
  norma_D <- if (md$p_D > 0) mean(abs(st$beta_D)) else NA
  # coeficientes "activos" (|beta| > 0.1) como proxy de la accion del LASSO
  act_E <- if (md$p_E > 0) sum(abs(st$beta_E) > 0.1) else 0
  act_M <- if (md$p_M > 0) sum(abs(st$beta_M) > 0.1) else 0
  act_D <- if (md$p_D > 0) sum(abs(st$beta_D) > 0.1) else 0
  
  # escalas residuales municipales (rango y mediana)
  kap_vals <- flatten_kappa2_jk(md, st$kappa2_jk)
  kap_med  <- median(kap_vals); kap_min <- min(kap_vals); kap_max <- max(kap_vals)
  
  # --- Señales de salud ---
  alertas <- character(0)
  if (st$nu < 2.5) alertas <- c(alertas, "nu->2 (colas extremas)")
  if (kap_max > 1e7) alertas <- c(alertas, "kappa2 muy grande")
  if (max(abs(st$beta_star)) > 1e5) alertas <- c(alertas, "beta* muy grande")
  if (!all(is.finite(st$beta_star)) || !is.finite(st$nu) ||
      !all(is.finite(kap_vals))) alertas <- c(alertas, "VALORES NO FINITOS!")
  if (st$sigma2_mu > 1e8) alertas <- c(alertas, "sigma2_mu muy grande")
  
  # --- Impresión ---
  cat(sprintf(
    "[%s] iter %d/%d (%.1f%%) | %s transcurrido, ETA %s (%.3fs/it) | guardadas=%d\n",
    fase, iter, n_iter, 100 * iter / n_iter,
    fmt_t(el), fmt_t(eta), s_it, guardadas))
  cat(sprintf(
    "        mu=%.2f | nu=%.2f | sigma2_mu=%.1f\n",
    mu_val, st$nu, st$sigma2_mu))
  cat(sprintf(
    "        |beta| medio  E=%.3f M=%.3f D=%s | activos(>0.1)  E=%d/%d M=%d/%d D=%d/%d\n",
    norma_E, norma_M, if (is.na(norma_D)) "-" else sprintf("%.3f", norma_D),
    act_E, md$p_E, act_M, md$p_M, act_D, md$p_D))
  cat(sprintf(
    "        lambda2  E=%.3f M=%.3f D=%.3f | kappa2_jk: med=%.1f [%.1f, %.1f]\n",
    st$lambda2_E, st$lambda2_M, st$lambda2_D, kap_med, kap_min, kap_max))
  cat(sprintf(
    "        kappa2_k: med=%.1f [%.1f, %.1f]\n",
    median(st$kappa2_k), min(st$kappa2_k), max(st$kappa2_k)))
  if (length(alertas) > 0) {
    cat(sprintf("        /!\\ ALERTA: %s\n", paste(alertas, collapse = "; ")))
  }
  invisible(NULL)
}


# ==============================================================================
# GIBBS SAMPLER COMPLETO
# ==============================================================================
gibbs_sampler_m2 <- function(model_data, hyper,
                             n_iter = 5000L, burn_in = 2000L, thin = 5L,
                             dir_salida = file.path(getwd(), "mcmc_salida_m2"),
                             state_init = NULL,
                             guardar_loglik = TRUE,
                             guardar_xi = FALSE,
                             seed = NULL,
                             slice_w = 1, slice_m = 50L,
                             verbose = 500L) {
  if (!is.null(seed)) set.seed(seed)
  
  # --- Validación de configuración MCMC ---
  stopifnot(burn_in < n_iter, thin >= 1)
  n_save <- (n_iter - burn_in) %/% thin
  stopifnot(n_save > 0)
  
  md <- model_data
  pp1 <- 1L + md$p
  n_cells <- sum(md$J_k)           # celdas kappa2_jk con observaciones
  
  # --- Directorio de salida ---
  if (!dir.exists(dir_salida)) dir.create(dir_salida, recursive = TRUE)
  
  # --- Estado inicial ---
  if (is.null(state_init)) {
    state <- initialize_state(md, hyper, seed = NULL)
  } else {
    state <- state_init
    validate_state(md, state)
  }
  

  rutas <- list(
    beta_star = file.path(dir_salida, "beta_star.bin"),
    sigma2_mu = file.path(dir_salida, "sigma2_mu.bin"),
    nu        = file.path(dir_salida, "nu.bin"),
    lambda2   = file.path(dir_salida, "lambda2.bin"),
    kappa2_k  = file.path(dir_salida, "kappa2_k.bin"),
    kappa2_jk = file.path(dir_salida, "kappa2_jk.bin"),
    tau2_E    = file.path(dir_salida, "tau2_E.bin"),
    tau2_M    = file.path(dir_salida, "tau2_M.bin"),
    tau2_D    = file.path(dir_salida, "tau2_D.bin")
  )
  if (guardar_loglik) rutas$loglik <- file.path(dir_salida, "loglik.bin")
  if (guardar_xi)     rutas$xi     <- file.path(dir_salida, "xi.bin")
  
  # Borrar archivos previos (para no mezclar con corridas anteriores)
  for (r in rutas) if (file.exists(r)) file.remove(r)
  
  con <- lapply(rutas, function(r) file(r, open = "ab"))
  on.exit({
    for (cc in con) try(close(cc), silent = TRUE)
  }, add = TRUE)
  
  # Helper de escritura (doubles, tamaño fijo por iteracion)
  escribir <- function(nombre, x) {
    writeBin(as.double(x), con[[nombre]], size = 8)
  }
  
  # ---------------------------------------------------------------------------
  # Bucle MCMC
  # ---------------------------------------------------------------------------
  if (verbose > 0) {
    cat(strrep("=", 70), "\n", sep = "")
    cat("GIBBS SAMPLER — Modelo 2 (t-Student jerarquico + LASSO Bayesiano)\n")
    cat(strrep("=", 70), "\n", sep = "")
    cat(sprintf("  Datos     : N=%d obs, K=%d deptos, %d municipios\n",
                md$N, md$K, sum(md$J_k)))
    cat(sprintf("  Parametros: p=%d (1 + %d E + %d M + %d D), bloque beta*=%d\n",
                md$p, md$p_E, md$p_M, md$p_D, pp1))
    cat(sprintf("  MCMC      : %d iter, %d burn-in, thin=%d -> %d muestras\n",
                n_iter, burn_in, thin, n_save))
    cat(sprintf("  Disco     : %s\n", dir_salida))
    cat(sprintf("  Guardar   : loglik=%s, xi=%s\n",
                guardar_loglik, guardar_xi))
    cat(strrep("-", 70), "\n", sep = "")
  }
  t_ini <- Sys.time()
  guardadas <- 0L
  
  for (iter in 1:n_iter) {
    # --- Orden del documento ---
    state <- update_xi(md, state, hyper)
    state <- update_beta_star(md, state, hyper)
    state <- update_sigma2_mu(md, state, hyper)
    state <- update_tau2(md, state, hyper)
    state <- update_kappa2_jk(md, state, hyper)
    state <- update_kappa2_k(md, state, hyper)
    state <- update_lambda2(md, state, hyper)
    state <- update_nu(md, state, hyper, w = slice_w, m = slice_m)
    
    # --- Guardar en disco si corresponde (post burn-in, cada thin) ---
    if (iter > burn_in && ((iter - burn_in) %% thin == 0L)) {
      escribir("beta_star", state$beta_star)
      escribir("sigma2_mu", state$sigma2_mu)
      escribir("nu",        state$nu)
      escribir("lambda2",   c(state$lambda2_E, state$lambda2_M, state$lambda2_D))
      escribir("kappa2_k",  state$kappa2_k)
      escribir("kappa2_jk", flatten_kappa2_jk(md, state$kappa2_jk))
      escribir("tau2_E",    state$tau2_E)
      escribir("tau2_M",    state$tau2_M)
      escribir("tau2_D",    state$tau2_D)
      if (guardar_loglik) escribir("loglik", loglik_t_pointwise(md, state))
      if (guardar_xi)     escribir("xi", state$xi)
      guardadas <- guardadas + 1L
    }
    
    # --- Progreso detallado ---
    if (verbose > 0 ) { # && iter %% verbose == 0L
      fase <- if (iter <= burn_in) "BURN-IN " else "MUESTREO"
      reporte_progreso(iter, n_iter, state, md, t_ini, guardadas, fase)
    }
  }
  
  # ---------------------------------------------------------------------------
  # Metadatos para lectura posterior (estilo Modelo 3)
  # ---------------------------------------------------------------------------
  meta <- list(
    n_save       = guardadas,
    n_iter       = n_iter, burn_in = burn_in, thin = thin,
    p = md$p, p_E = md$p_E, p_M = md$p_M, p_D = md$p_D,
    pp1 = pp1, K = md$K, J_k = md$J_k, n_cells = n_cells,
    N = md$N,
    theta_labels = md$theta_labels,
    depto_ids = md$depto_ids, muni_ids = md$muni_ids,
    idx_mu = md$idx_mu, idx_E = md$idx_E, idx_M = md$idx_M, idx_D = md$idx_D,
    guardar_loglik = guardar_loglik, guardar_xi = guardar_xi,
    # tamaños por iteracion (en doubles) para readBin/seek
    sizes = list(
      beta_star = pp1, sigma2_mu = 1L, nu = 1L, lambda2 = 3L,
      kappa2_k = md$K, kappa2_jk = n_cells,
      tau2_E = md$p_E, tau2_M = md$p_M, tau2_D = md$p_D,
      loglik = if (guardar_loglik) md$N else 0L,
      xi = if (guardar_xi) md$N else 0L
    ),
    hyper = hyper,
    timestamp = Sys.time()
  )
  saveRDS(meta, file.path(dir_salida, "metadatos.rds"))
  
  el_total <- as.numeric(difftime(Sys.time(), t_ini, units = "secs"))
  if (verbose > 0) {
    fmt_t <- function(s) if (s < 60) sprintf("%.0fs", s) else if (s < 3600) sprintf("%.1fmin", s/60) else sprintf("%.1fh", s/3600)
    cat(strrep("-", 70), "\n", sep = "")
    cat(sprintf("COMPLETADO: %d muestras guardadas en %s (%.3fs/iter)\n",
                guardadas, fmt_t(el_total), el_total / n_iter))
    cat(sprintf("  nu final=%.2f | mu final=%.2f | sigma2_mu final=%.1f\n",
                state$nu, state$mu, state$sigma2_mu))
    cat(sprintf("  Muestras en disco: %s\n", dir_salida))
    cat(sprintf("  Leer con: leer_muestras_m2('%s', 'beta_star')\n", dir_salida))
    cat(strrep("=", 70), "\n", sep = "")
  }
  
  invisible(list(dir_salida = dir_salida, n_save = guardadas,
                 rutas = rutas, meta = meta,
                 segundos = el_total, state_final = state))
}


# ==============================================================================
# LECTOR DE MUESTRAS DESDE DISCO
# ==============================================================================

leer_muestras_m2 <- function(dir_salida, que, desde = 1L, hasta = NULL) {
  meta <- readRDS(file.path(dir_salida, "metadatos.rds"))
  dim_bloque <- meta$sizes[[que]]
  if (is.null(dim_bloque) || dim_bloque == 0L) {
    stop(sprintf("Bloque '%s' no disponible o de dimension 0.", que))
  }
  if (is.null(hasta)) hasta <- meta$n_save
  n_leer <- hasta - desde + 1L
  
  ruta <- file.path(dir_salida, paste0(que, ".bin"))
  con <- file(ruta, open = "rb")
  on.exit(close(con), add = TRUE)
  if (desde > 1L) seek(con, (desde - 1L) * dim_bloque * 8)
  raw <- readBin(con, "double", n = n_leer * dim_bloque)
  
  if (dim_bloque == 1L) {
    return(raw)   # escalar -> vector de n_leer
  }
  mat <- matrix(raw, nrow = n_leer, ncol = dim_bloque, byrow = TRUE)
  if (que == "beta_star") colnames(mat) <- meta$theta_labels
  if (que == "lambda2")   colnames(mat) <- c("E", "M", "D")
  mat
}


# ==============================================================================
# EJECUCIÓN
# ==============================================================================

source("1_manupulacion_matrices_M2.R")
model_data <- construir_model_data(cfg, x_ijk_names, z_jk_names, w_k_names)
source("2_xi_M2.R")
source("3_beta_M2.R")
source("4_sigma_mu_M2.R")
source("5_tau2_M2.R")
source("6_kappa2_jk_M2.R")
source("7_kappa2_k_M2.R")
source("8_lambda2_M2.R")
source("9_nu_M2.R")
hyper <- create_hyperparameters()
res <- gibbs_sampler_m2(model_data, hyper,
                        n_iter = 35000, burn_in = 5000, thin = 5,
                        dir_salida = "mcmc_salida_m2", seed = 2026)
# # Leer muestras de beta* para diagnostico:
# beta_chain <- leer_muestras_m2("mcmc_salida_m2", "beta_star")
# nu_chain   <- leer_muestras_m2("mcmc_salida_m2", "nu")