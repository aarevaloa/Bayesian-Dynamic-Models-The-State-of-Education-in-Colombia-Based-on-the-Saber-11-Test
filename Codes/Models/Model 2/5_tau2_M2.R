#_______________________________________________________________________________
#@  MODELO 2 — t-STUDENT JERÁRQUICO ESTÁTICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________
# ==============================================================================
# Muestreador de la Inversa Gaussiana: algoritmo Michael, Schucany & Haas (1976)
# ==============================================================================
rinvgauss_msh <- function(n, mu_prime, lambda_shape) {
  if (any(mu_prime <= 0) || any(lambda_shape <= 0)) {
    stop("rinvgauss_msh: mu' y lambda' deben ser > 0.")
  }
  mu_prime     <- rep(mu_prime, length.out = n)
  lambda_shape <- rep(lambda_shape, length.out = n)
  
  # Paso 1: y = nu^2 con nu ~ N(0,1)  (chi-cuadrado de 1 g.l.)
  nu_n <- rnorm(n)
  y <- nu_n^2
  
  # Paso 2: raíz menor de la cuadrática (forma factorizada por mu' para estabilidad)
  # x = mu' * [ 1 + (mu' y)/(2L) - (1/(2L)) sqrt(4 mu' L y + mu'^2 y^2) ]
  term_sqrt <- sqrt(4 * mu_prime * lambda_shape * y + mu_prime^2 * y^2)
  x <- mu_prime * (1 + (mu_prime * y) / (2 * lambda_shape)
                   - term_sqrt / (2 * lambda_shape))
  
  # Salvaguarda: errores de redondeo pueden dar x ligeramente <= 0; corregir.
  bad <- (!is.finite(x)) | (x <= 0)
  if (any(bad)) x[bad] <- mu_prime[bad]   # fallback al modo aproximado
  
  # Paso 3: aceptación/reflexión
  z <- runif(n)
  acept <- z <= mu_prime / (mu_prime + x)
  out <- ifelse(acept, x, mu_prime^2 / x)
  
  # Salvaguarda final de positividad/finitud
  bad2 <- (!is.finite(out)) | (out <= 0)
  if (any(bad2)) out[bad2] <- .Machine$double.xmin
  out
}


# ==============================================================================
# Actualización de tau2 para UN bloque (E, M o D)
# ==============================================================================

update_tau2_block <- function(beta, varsigma2_L, lambda2_L, beta_floor = 1e-8) {
  p_L <- length(beta)
  if (p_L == 0) return(numeric(0))
  
  # |beta| acotado por debajo para evitar mu' -> Inf
  abs_beta <- pmax(abs(beta), beta_floor)
  
  # Parámetros de la Inversa Gaussiana para omega = 1/tau2
  #   mu'     = sqrt( lambda2_L * varsigma2_L / beta^2 )
  #   lambda' = lambda2_L
  mu_prime     <- sqrt(lambda2_L * varsigma2_L / abs_beta^2)
  lambda_shape <- lambda2_L
  
  # Muestrear omega ~ InvGauss(mu', lambda') y devolver tau2 = 1/omega
  omega <- rinvgauss_msh(p_L, mu_prime = mu_prime, lambda_shape = lambda_shape)
  tau2  <- 1 / omega
  
  # Salvaguarda de positividad/finitud
  bad <- (!is.finite(tau2)) | (tau2 <= 0)
  if (any(bad)) tau2[bad] <- .Machine$double.xmin
  tau2
}


# ==============================================================================
# CONDICIONAL COMPLETA: update_tau2()
# ==============================================================================
update_tau2 <- function(model_data, state, hyper) {
  st <- state
  
  st$tau2_E <- update_tau2_block(st$beta_E, hyper$varsigma2_E, st$lambda2_E)
  st$tau2_M <- update_tau2_block(st$beta_M, hyper$varsigma2_M, st$lambda2_M)
  st$tau2_D <- update_tau2_block(st$beta_D, hyper$varsigma2_D, st$lambda2_D)
  
  st
}


# ==============================================================================
# VERIFICACIONES COMPUTACIONALES
# ==============================================================================
check_tau2 <- function(model_data, state) {
  md <- model_data; st <- state
  stopifnot(length(st$tau2_E) == md$p_E,
            length(st$tau2_M) == md$p_M,
            length(st$tau2_D) == md$p_D)
  if (md$p_E > 0) stopifnot(all(st$tau2_E > 0), all(is.finite(st$tau2_E)))
  if (md$p_M > 0) stopifnot(all(st$tau2_M > 0), all(is.finite(st$tau2_M)))
  if (md$p_D > 0) stopifnot(all(st$tau2_D > 0), all(is.finite(st$tau2_D)))
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
# hyper <- create_hyperparameters()
# state <- initialize_state(model_data, hyper, seed = 2025)
# state <- update_xi(model_data, state, hyper)
# state <- update_beta_star(model_data, state, hyper)
# state <- update_sigma2_mu(model_data, state, hyper)
state <- update_tau2(model_data, state, hyper)
check_tau2(model_data, state)
