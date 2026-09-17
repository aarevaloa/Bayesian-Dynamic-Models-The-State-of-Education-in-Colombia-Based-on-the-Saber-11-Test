#_______________________________________________________________________________
#@  MODELO 2 — t-STUDENT JERÁRQUICO ESTÁTICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________
# ==============================================================================
# FASE 3. MATRIZ DE DISEÑO GLOBAL X*
# ==============================================================================

build_X_star <- function(model_data) {
  md <- model_data
  Xs <- md$X_star
  stopifnot(ncol(Xs) == 1L + md$p)
  stopifnot(all(Xs[, md$idx_mu] == 1))            # intercepto
  # Partición exacta de columnas
  todos_idx <- c(md$idx_mu, md$idx_E, md$idx_M, md$idx_D)
  stopifnot(identical(sort(todos_idx), 1:(1L + md$p)))
  stopifnot(length(md$idx_E) == md$p_E,
            length(md$idx_M) == md$p_M,
            length(md$idx_D) == md$p_D)
  Xs
}


# ==============================================================================
# Construcción del vector diagonal de la precisión prior: (D*)^{-1}
# ==============================================================================
build_Dstar_inv_diag <- function(model_data, state, hyper) {
  md <- model_data; st <- state
  d <- numeric(1L + md$p)
  
  # Intercepto
  d[md$idx_mu] <- 1 / st$sigma2_mu
  # Bloques LASSO: varianza prior = varsigma2_L * tau2_{l,L}
  if (md$p_E > 0) d[md$idx_E] <- 1 / (hyper$varsigma2_E * st$tau2_E)
  if (md$p_M > 0) d[md$idx_M] <- 1 / (hyper$varsigma2_M * st$tau2_M)
  if (md$p_D > 0) d[md$idx_D] <- 1 / (hyper$varsigma2_D * st$tau2_D)
  
  if (any(!is.finite(d)) || any(d <= 0)) {
    stop("build_Dstar_inv_diag: diagonal de (D*)^{-1} no finita o no positiva.")
  }
  d
}


# ==============================================================================
# FASE 4. PRECISIÓN POSTERIOR
# ==============================================================================
build_Lambda_star <- function(model_data, state, hyper) {
  md <- model_data; st <- state
  Xs <- md$X_star
  
  # --- Pesos por observación: w_i = xi_i / kappa2_{j(i),k(i)} ---
  kappa2_obs <- expand_kappa2_to_obs(md, st$kappa2_jk)   # longitud N
  w <- st$xi / kappa2_obs
  if (any(!is.finite(w)) || any(w <= 0)) {
    stop("build_Lambda_star: pesos w no finitos o no positivos.")
  }
  
  # --- Contribución de la verosimilitud: X*' W X*  (vectorizado) ---
  # crossprod(Xs, w * Xs) = t(Xs) %*% diag(w) %*% Xs, sin formar diag(w).
  Lambda_lik <- crossprod(Xs, w * Xs)                    # (1+p) x (1+p)
  
  # --- Sumar (D*)^{-1} a la diagonal ---
  d_inv <- build_Dstar_inv_diag(md, st, hyper)
  Lambda <- Lambda_lik
  diag(Lambda) <- diag(Lambda) + d_inv
  
  # --- Simetrizar (corrige asimetría por redondeo de orden 1e-16) ---
  Lambda <- (Lambda + t(Lambda)) / 2
  
  # --- Validaciones ---
  if (any(!is.finite(Lambda))) stop("build_Lambda_star: Lambda* con valores no finitos.")
  if (any(diag(Lambda) <= 0)) stop("build_Lambda_star: diagonal de Lambda* no positiva.")
  Lambda
}


# ==============================================================================
# FASE 5. VECTOR LINEAL
# ==============================================================================

build_eta_star <- function(model_data, state, hyper) {
  md <- model_data; st <- state
  Xs <- md$X_star
  
  # --- Pesos por observación (idénticos a los de Lambda*) ---
  kappa2_obs <- expand_kappa2_to_obs(md, st$kappa2_jk)
  w <- st$xi / kappa2_obs
  
  # --- Contribución de la verosimilitud: X*' W y ---
  eta_lik <- drop(crossprod(Xs, w * md$y_vec))           # longitud (1+p)
  
  # --- Término prior (D*)^{-1} m0*: solo la entrada del intercepto ---
  # m0* = (mu_mu, 0, ..., 0)'  =>  (D*)^{-1} m0* = (mu_mu/sigma2_mu, 0, ..., 0)'
  eta <- eta_lik
  eta[md$idx_mu] <- eta[md$idx_mu] + hyper$mu_mu / st$sigma2_mu
  
  if (any(!is.finite(eta))) stop("build_eta_star: eta* con valores no finitos.")
  stopifnot(length(eta) == 1L + md$p)
  eta
}


# ==============================================================================
# FASE 6. MUESTREO NORMAL MULTIVARIADO POR CHOLESKY
# ==============================================================================

sample_beta_star <- function(Lambda_star, eta_star, jitter_rel = 1e-8) {
  pp1 <- nrow(Lambda_star)
  
  # --- Cholesky superior: R'R = Lambda*. Respaldo con jitter si falla. ---
  R <- tryCatch(chol(Lambda_star), error = function(e) NULL)
  if (is.null(R)) {
    jit <- jitter_rel * mean(diag(Lambda_star))
    R <- tryCatch(chol(Lambda_star + diag(jit, pp1)),
                  error = function(e) stop("sample_beta_star: Cholesky falló incluso con jitter."))
    warning(sprintf("sample_beta_star: jitter numerico %.2e aplicado para estabilizar Cholesky.", jit))
  }
  
  # --- Media posterior m* = (Lambda*)^{-1} eta* via dos sustituciones ---
  # L w = eta*  con L = t(R) triangular inferior -> forwardsolve(t(R), eta*)
  w_vec  <- forwardsolve(t(R), eta_star)
  # L' m* = w  con L' = R triangular superior -> backsolve(R, w)
  m_star <- backsolve(R, w_vec)
  
  # --- Muestra: z ~ N(0,I); L' u = z -> backsolve(R, z); beta* = m* + u ---
  z <- rnorm(pp1)
  u <- backsolve(R, z)
  beta_star <- m_star + u
  
  list(beta_star = beta_star, m_star = m_star, R = R)
}


# ==============================================================================
# FASE 7. ACTUALIZACIÓN  (update_beta_star)
# ==============================================================================
update_beta_star <- function(model_data, state, hyper) {
  md <- model_data; st <- state
  
  # --- Construir precisión y vector lineal posteriores ---
  Lambda_star <- build_Lambda_star(md, st, hyper)
  eta_star    <- build_eta_star(md, st, hyper)
  
  # --- Muestrear beta* por Cholesky ---
  draw <- sample_beta_star(Lambda_star, eta_star)
  beta_star <- draw$beta_star
  names(beta_star) <- md$theta_labels
  
  # --- Volcar en el estado: bloques y vector aumentado ---
  st$beta_star <- beta_star
  st$mu        <- beta_star[md$idx_mu]
  st$beta_E    <- if (md$p_E > 0) beta_star[md$idx_E] else numeric(0)
  st$beta_M    <- if (md$p_M > 0) beta_star[md$idx_M] else numeric(0)
  st$beta_D    <- if (md$p_D > 0) beta_star[md$idx_D] else numeric(0)
  
  # --- Refrescar cachés (beta* cambió -> zeta y epsilon cambian) ---
  st$zeta    <- drop(md$X_star %*% beta_star)
  st$epsilon <- md$y_vec - st$zeta
  
  st
}


# ==============================================================================
# FASE 8. VALIDACIONES NUMÉRICAS
# ==============================================================================

check_beta_star <- function(model_data, Lambda_star, state) {
  md <- model_data
  pp1 <- 1L + md$p
  
  # --- Dimensiones ---
  stopifnot(nrow(Lambda_star) == pp1, ncol(Lambda_star) == pp1)
  stopifnot(length(state$beta_star) == pp1)
  
  # --- Simetría ---
  asim <- max(abs(Lambda_star - t(Lambda_star)))
  stopifnot(asim < 1e-8)
  
  # --- Definida positiva: autovalores estrictamente positivos ---
  ev <- eigen(Lambda_star, symmetric = TRUE, only.values = TRUE)$values
  stopifnot(all(ev > 0))
  
  # --- Finitud ---
  stopifnot(all(is.finite(Lambda_star)))
  stopifnot(all(is.finite(state$beta_star)))
  
  # --- Coherencia del bloque aumentado ---
  recomp <- c(state$mu, state$beta_E, state$beta_M, state$beta_D)
  stopifnot(max(abs(recomp - state$beta_star)) < 1e-10)
  
  invisible(TRUE)
}


# ==============================================================================
# EJECUCIÓN
# ==============================================================================
#
# source("1_manupulacion_matrices.R")
# model_data <- construir_model_data(cfg, x_ijk_names, z_jk_names, w_k_names)
# source("2_xi.R")
# hyper <- create_hyperparameters()
# state <- initialize_state(model_data, hyper, seed = 2025)
# state <- update_xi(model_data, state, hyper)
state <- update_beta_star(model_data, state, hyper)
Lambda_star <- build_Lambda_star(model_data, state, hyper)
check_beta_star(model_data, Lambda_star, state)
