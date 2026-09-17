#_______________________________________________________________________________
#@  MODELO 2 — t-STUDENT JERÁRQUICO ESTÁTICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________
rinvgamma_rate <- function(n, shape, scale) {
  if (any(shape <= 0) || any(scale <= 0)) {
    stop("rinvgamma_rate: shape y scale deben ser > 0.")
  }
  # G ~ Gamma(shape=a, rate=b)  =>  1/G ~ IG(shape=a, scale=b)
  g <- rgamma(n, shape = shape, rate = scale)
  out <- 1 / g
  # Salvaguarda: si g desborda a Inf (colas), 1/g -> 0; reemplazar por valor min.
  if (any(out <= 0)) out[out <= 0] <- .Machine$double.xmin
  out
}


# ==============================================================================
# CONDICIONAL COMPLETA: update_sigma2_mu()
# ==============================================================================
update_sigma2_mu <- function(model_data, state, hyper) {
  st <- state
  
  # --- Parámetros de la IG ---
  shape <- (hyper$nu_mu + 1) / 2
  scale <- (hyper$nu_mu * hyper$gamma_mu2 + (st$mu - hyper$mu_mu)^2) / 2
  
  # --- Muestreo (recíproco de Gamma; `rate` = scale de la IG) ---
  st$sigma2_mu <- rinvgamma_rate(1, shape = shape, scale = scale)
  
  st
}


# ==============================================================================
# VERIFICACIONES COMPUTACIONALES
# ==============================================================================
check_sigma2_mu <- function(state) {
  s2 <- state$sigma2_mu
  stopifnot(length(s2) == 1)
  stopifnot(is.finite(s2))
  stopifnot(s2 > 0)
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
# hyper <- create_hyperparameters()
# state <- initialize_state(model_data, hyper, seed = 2025)
# state <- update_xi(model_data, state, hyper)
# state <- update_beta_star(model_data, state, hyper)
state <- update_sigma2_mu(model_data, state, hyper)
check_sigma2_mu(state)