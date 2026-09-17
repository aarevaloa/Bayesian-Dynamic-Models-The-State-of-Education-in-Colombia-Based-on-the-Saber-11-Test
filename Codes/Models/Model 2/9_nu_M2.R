#_______________________________________________________________________________
#@  MODELO 2 — t-STUDENT JERÁRQUICO ESTÁTICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________
log_cond_nu <- function(nu, N, T_log, T_sum, lambda_nu) {
  if (nu <= 2) return(-Inf)
  (N * nu / 2) * log(nu / 2) -
    N * lgamma(nu / 2) +
    (nu / 2 - 1) * T_log -
    (nu / 2) * T_sum -
    lambda_nu * (nu - 2)
}

log_cond_eta <- function(eta, N, T_log, T_sum, lambda_nu) {
  nu <- exp(eta) + 2
  log_cond_nu(nu, N, T_log, T_sum, lambda_nu) + eta
}


# ==============================================================================
# Slice sampler univariado de Neal (2003): stepping-out + shrinkage
# ==============================================================================
slice_sample_step <- function(x0, g, w = 1, m = 50,
                              lower = -Inf, upper = Inf,
                              max_shrink = 200L) {
  gx0 <- g(x0)
  # 1. Nivel del slice en log-escala
  log_y <- gx0 - rexp(1)
  
  # 2. Stepping-out: posicionar [L,R] aleatoriamente alrededor de x0
  u <- runif(1)
  L <- x0 - w * u
  R <- L + w
  # Expandir a la izquierda
  J <- floor(m * runif(1))
  K <- (m - 1) - J
  while (J > 0 && L > lower && g(L) > log_y) { L <- L - w; J <- J - 1 }
  # Expandir a la derecha
  while (K > 0 && R < upper && g(R) > log_y) { R <- R + w; K <- K - 1 }
  # Recortar a cotas del soporte
  if (L < lower) L <- lower
  if (R > upper) R <- upper
  
  # 3. Shrinkage: proponer y contraer hasta aceptar
  for (s in 1:max_shrink) {
    x1 <- L + runif(1) * (R - L)
    if (g(x1) >= log_y) return(x1)
    # contraer hacia x0
    if (x1 < x0) L <- x1 else R <- x1
  }
  # Salvaguarda: si no aceptó (muy raro), devolver x0 (paso nulo, deja invariante)
  x0
}


# ==============================================================================
# CONDICIONAL COMPLETA: update_nu()
# ==============================================================================

update_nu <- function(model_data, state, hyper, w = 1, m = 50L) {
  md <- model_data; st <- state
  
  # --- Estadisticos suficientes (reutiliza suff_stats_nu de la Etapa 2) ---
  ss <- suff_stats_nu(st$xi)            # lista con T_log, T_sum
  N <- md$N
  
  # --- Funcion objetivo en eta (cierre sobre los estadisticos) ---
  g_eta <- function(eta) {
    log_cond_eta(eta, N = N, T_log = ss$T_log, T_sum = ss$T_sum,
                 lambda_nu = hyper$lambda_nu)
  }
  
  # --- Punto de partida: eta del nu actual ---
  eta0 <- log(st$nu - 2)
  # (si st$nu fuese <= 2 por algun error previo, reiniciar en un valor seguro)
  if (!is.finite(eta0)) eta0 <- log(5 - 2)
  
  # --- Un paso de slice sampling sobre eta in R ---
  eta1 <- slice_sample_step(eta0, g_eta, w = w, m = m,
                            lower = -Inf, upper = Inf)
  
  nu_new <- exp(eta1) + 2
  
  # --- Salvaguarda de soporte ---
  if (!is.finite(nu_new) || nu_new <= 2) nu_new <- st$nu   # paso nulo si algo falla
  
  st$nu <- nu_new
  st
}


# ==============================================================================
# VERIFICACIONES COMPUTACIONALES
# ==============================================================================
check_nu <- function(state) {
  nu <- state$nu
  stopifnot(length(nu) == 1)
  stopifnot(is.finite(nu))
  stopifnot(nu > 2)
  invisible(TRUE)
}


# ==============================================================================
# EJECUCIÓN
# ==============================================================================
#
# source("1_manupulacion_matrices.R")
# model_data <- construir_model_data(cfg, x_ijk_names, z_jk_names, w_k_names)
# source("2_xi.R")
# source("3_beta.R")
# source("4_sigma_mu.R")
# source("5_tau2.R")
# source("6_kappa2_jk.R")
# source("7_kappa2_k.R")
# source("8_lambda2.R")
# source("9_nu.R")
# hyper <- create_hyperparameters()
# state <- initialize_state(model_data, hyper, seed = 2025)
# state <- update_xi(model_data, state, hyper)
state <- update_nu(model_data, state, hyper)
check_nu(state)