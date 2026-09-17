#_______________________________________________________________________________
#@  MODELO 2 — t-STUDENT JERÁRQUICO ESTÁTICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________
update_kappa2_k <- function(model_data, state, hyper) {
  md <- model_data; st <- state
  K <- md$K

  sum_prec <- numeric(K)
  for (kk in 1:K) {
    Jk <- md$J_k[kk]
    sum_prec[kk] <- sum(1 / st$kappa2_jk[1:Jk, kk])
  }
  
  # --- Parámetros de la Gamma (shape-rate), vectorizados sobre departamentos ---
  shape <- (hyper$alpha_kappa + hyper$nu_kappa * md$J_k) / 2          # vector K
  rate  <- (hyper$beta_kappa  + hyper$nu_kappa * sum_prec) / 2        # vector K
  
  # --- Muestreo Gamma vectorizado. `rate` NOMBRADO (no scale). ---
  kappa2_k_new <- rgamma(K, shape = shape, rate = rate)
  
  # --- Salvaguarda numérica de positividad ---
  if (any(kappa2_k_new <= 0)) {
    kappa2_k_new[kappa2_k_new <= 0] <- .Machine$double.xmin
  }
  
  st$kappa2_k <- kappa2_k_new
  st
}


# ==============================================================================
# VERIFICACIONES COMPUTACIONALES
# ==============================================================================

check_kappa2_k <- function(model_data, state) {
  st <- state
  stopifnot(length(st$kappa2_k) == model_data$K)
  stopifnot(all(st$kappa2_k > 0))
  stopifnot(all(is.finite(st$kappa2_k)))
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
# hyper <- create_hyperparameters()
# state <- initialize_state(model_data, hyper, seed = 2025)
# state <- update_xi(model_data, state, hyper)
# state <- update_beta_star(model_data, state, hyper)
# state <- update_sigma2_mu(model_data, state, hyper)
# state <- update_tau2(model_data, state, hyper)
# state <- update_kappa2_jk(model_data, state, hyper)
state <- update_kappa2_k(model_data, state, hyper)
check_kappa2_k(model_data, state)
