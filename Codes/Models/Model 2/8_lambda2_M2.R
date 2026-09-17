#_______________________________________________________________________________
#@  MODELO 2 — t-STUDENT JERÁRQUICO ESTÁTICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________
update_lambda2_block <- function(tau2, a_lambda_L, b_lambda_L) {
  p_L <- length(tau2)
  shape <- a_lambda_L + p_L
  rate  <- b_lambda_L + 0.5 * sum(tau2)   # sum(numeric(0)) = 0 -> maneja p_L=0
  
  lambda2 <- rgamma(1, shape = shape, rate = rate)   # `rate` NOMBRADO (no scale)
  
  if (!is.finite(lambda2) || lambda2 <= 0) lambda2 <- .Machine$double.xmin
  lambda2
}


# ==============================================================================
# CONDICIONAL COMPLETA: update_lambda2()
# ==============================================================================
update_lambda2 <- function(model_data, state, hyper) {
  st <- state
  
  st$lambda2_E <- update_lambda2_block(st$tau2_E, hyper$a_lambda_E, hyper$b_lambda_E)
  st$lambda2_M <- update_lambda2_block(st$tau2_M, hyper$a_lambda_M, hyper$b_lambda_M)
  st$lambda2_D <- update_lambda2_block(st$tau2_D, hyper$a_lambda_D, hyper$b_lambda_D)
  
  st
}


# ==============================================================================
# VERIFICACIONES COMPUTACIONALES
# ==============================================================================
check_lambda2 <- function(state) {
  st <- state
  for (nm in c("lambda2_E", "lambda2_M", "lambda2_D")) {
    v <- st[[nm]]
    stopifnot(length(v) == 1, is.finite(v), v > 0)
  }
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
# hyper <- create_hyperparameters()
# state <- initialize_state(model_data, hyper, seed = 2025)
# state <- update_xi(model_data, state, hyper)
# state <- update_beta_star(model_data, state, hyper)
# state <- update_sigma2_mu(model_data, state, hyper)
# state <- update_tau2(model_data, state, hyper)
# state <- update_kappa2_jk(model_data, state, hyper)
# state <- update_kappa2_k(model_data, state, hyper)
state <- update_lambda2(model_data, state, hyper)
check_lambda2(state)
