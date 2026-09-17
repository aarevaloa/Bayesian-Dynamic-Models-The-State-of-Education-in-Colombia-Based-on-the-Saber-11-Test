#_______________________________________________________________________________
#@  MODELO 2 — t-STUDENT JERÁRQUICO ESTÁTICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________
# ==============================================================================
# 1. FÁBRICA DE HIPERPARÁMETROS: create_hyperparameters()
# ==============================================================================
create_hyperparameters <- function(
    # --- prior de mu ---
  mu_mu       = 0,
  nu_mu       = 5,
  gamma_mu2   = 100,
  # --- escalas base LASSO (fijas) ---
  varsigma2_E = 1,
  varsigma2_M = 1,
  varsigma2_D = 1,
  # --- jerarquía kappa ---
  nu_kappa    = 5,
  alpha_kappa = 2,
  beta_kappa  = 2,
  # --- lambda^2 por nivel ---
  a_lambda_E  = 1,
  b_lambda_E  = 1,
  a_lambda_M  = 1,
  b_lambda_M  = 1,
  a_lambda_D  = 1,
  b_lambda_D  = 1,
  # --- prior de nu ---
  lambda_nu   = 0.1
) {
  hyper <- list(
    mu_mu = mu_mu, nu_mu = nu_mu, gamma_mu2 = gamma_mu2,
    varsigma2_E = varsigma2_E, varsigma2_M = varsigma2_M, varsigma2_D = varsigma2_D,
    nu_kappa = nu_kappa, alpha_kappa = alpha_kappa, beta_kappa = beta_kappa,
    a_lambda_E = a_lambda_E, b_lambda_E = b_lambda_E,
    a_lambda_M = a_lambda_M, b_lambda_M = b_lambda_M,
    a_lambda_D = a_lambda_D, b_lambda_D = b_lambda_D,
    lambda_nu = lambda_nu
  )
  
  # --- Validación de soporte estrictamente positivo ---
  pos <- c("nu_mu", "gamma_mu2", "varsigma2_E", "varsigma2_M", "varsigma2_D",
           "nu_kappa", "alpha_kappa", "beta_kappa",
           "a_lambda_E", "b_lambda_E", "a_lambda_M", "b_lambda_M",
           "a_lambda_D", "b_lambda_D", "lambda_nu")
  for (nm in pos) {
    val <- hyper[[nm]]
    if (!is.numeric(val) || length(val) != 1 || !is.finite(val) || val <= 0) {
      stop(sprintf("Hiperparametro '%s' debe ser un escalar finito > 0 (recibido: %s).",
                   nm, as.character(val)))
    }
  }
  if (!is.numeric(mu_mu) || length(mu_mu) != 1 || !is.finite(mu_mu)) {
    stop("Hiperparametro 'mu_mu' debe ser un escalar finito.")
  }
  
  # --- Advertencias sobre existencia de momentos (modelo válido igualmente) ---
  if (nu_mu <= 2) {
    warning(sprintf("nu_mu = %.2f <= 2: la varianza marginal del intercepto no es finita.", nu_mu))
  }
  if (nu_kappa <= 2) {
    warning(sprintf("nu_kappa = %.2f <= 2: E[kappa2_jk | kappa2_k] no es finita.", nu_kappa))
  }
  
  class(hyper) <- "hyper_m2"
  hyper
}


# ==============================================================================
# 2. FUNCIONES AUXILIARES REUTILIZABLES
# ==============================================================================
compute_zeta <- function(model_data, beta_star) {
  stopifnot(length(beta_star) == ncol(model_data$X_star))
  drop(model_data$X_star %*% beta_star)
}

compute_residuals <- function(model_data, beta_star) {
  model_data$y_vec - compute_zeta(model_data, beta_star)
}


expand_kappa2_to_obs <- function(model_data, kappa2_jk) {
  # Indexación matricial: kappa2_jk[cbind(fila_j, fila_k)] toma, para cada
  # observación, el elemento (j_i, k_i). Es O(N) y totalmente vectorizado.
  idx <- cbind(model_data$row_j, model_data$row_k)
  kap <- kappa2_jk[idx]
  if (any(is.na(kap))) {
    stop("expand_kappa2_to_obs: kappa2 NA en alguna celda con observaciones.")
  }
  kap
}


suff_stats_nu <- function(xi) {
  stopifnot(all(xi > 0))
  list(T_log = sum(log(xi)), T_sum = sum(xi))
}


weighted_sq_by_muni <- function(model_data, xi, epsilon) {
  contrib <- xi * epsilon^2
  maxJ <- nrow(model_data$n_jk); K <- model_data$K
  out <- matrix(0, nrow = maxJ, ncol = K)
  # Sumar contribuciones por celda. split por (k,j) y sumar.
  clave <- paste(model_data$row_k, model_data$row_j, sep = "_")
  sumas <- tapply(contrib, clave, sum)
  partes <- do.call(rbind, strsplit(names(sumas), "_", fixed = TRUE))
  kk <- as.integer(partes[, 1]); jj <- as.integer(partes[, 2])
  out[cbind(jj, kk)] <- as.numeric(sumas)
  out
}


# ==============================================================================
# 3. INICIALIZACIÓN DEL ESTADO: initialize_state()
# ==============================================================================

initialize_state <- function(model_data, hyper, nu_init = 5,
                             ridge_eps = 1e-6, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  stopifnot(nu_init > 2)
  
  md <- model_data
  pp1 <- ncol(md$X_star)         # 1 + p
  
  # ---------------------------------------------------------------------------
  # beta_star inicial por OLS
  # ---------------------------------------------------------------------------
  XtX <- crossprod(md$X_star)              # (1+p) x (1+p)
  Xty <- crossprod(md$X_star, md$y_vec)    # (1+p) x 1
  # Ridge relativo a la traza para garantizar invertibilidad numérica.
  ridge <- ridge_eps * mean(diag(XtX))
  beta_star <- tryCatch(
    drop(solve(XtX + diag(ridge, pp1), Xty)),
    error = function(e) {
      warning("OLS inicial falló; usando beta_star = 0 salvo intercepto = mean(y).")
      b <- rep(0, pp1); b[md$idx_mu] <- mean(md$y_vec); b
    }
  )
  names(beta_star) <- md$theta_labels
  
  # Separar en bloques (vista coherente con la notación del modelo)
  mu     <- beta_star[md$idx_mu]
  beta_E <- if (md$p_E > 0) beta_star[md$idx_E] else numeric(0)
  beta_M <- if (md$p_M > 0) beta_star[md$idx_M] else numeric(0)
  beta_D <- if (md$p_D > 0) beta_star[md$idx_D] else numeric(0)
  
  # ---------------------------------------------------------------------------
  # Residuos iniciales (con beta_star OLS) para calibrar kappa2
  # ---------------------------------------------------------------------------
  zeta    <- drop(md$X_star %*% beta_star)
  epsilon <- md$y_vec - zeta
  
  # ---------------------------------------------------------------------------
  # kappa2_k (departamental) y kappa2_jk (municipal) desde varianzas residuales
  # ---------------------------------------------------------------------------
  var_global <- var(epsilon)
  piso_kappa <- max(1e-6, 1e-3 * var_global)   # cota inferior para evitar ceros
  
  # Varianza residual por departamento
  kappa2_k <- numeric(md$K)
  for (kk in 1:md$K) {
    e_k <- epsilon[md$row_k == kk]
    v_k <- if (length(e_k) > 1) var(e_k) else var_global
    kappa2_k[kk] <- max(piso_kappa, v_k)
  }
  
  # Varianza residual por municipio (fallback al departamento si n pequeño)
  maxJ <- nrow(md$n_jk)
  kappa2_jk <- matrix(NA_real_, nrow = maxJ, ncol = md$K)
  clave <- paste(md$row_k, md$row_j, sep = "_")
  e_split <- split(epsilon, clave)
  for (kk in 1:md$K) for (jj in 1:md$J_k[kk]) {
    key <- paste(kk, jj, sep = "_")
    e_jk <- e_split[[key]]
    njk <- md$n_jk[jj, kk]
    v_jk <- if (!is.null(e_jk) && njk >= 5) var(e_jk) else kappa2_k[kk]
    if (is.na(v_jk) || v_jk <= 0) v_jk <- kappa2_k[kk]
    kappa2_jk[jj, kk] <- max(piso_kappa, v_jk)
  }
  
  # ---------------------------------------------------------------------------
  # Escalas LASSO y varianza del intercepto
  # ---------------------------------------------------------------------------
  tau2_E <- if (md$p_E > 0) rep(1, md$p_E) else numeric(0)
  tau2_M <- if (md$p_M > 0) rep(1, md$p_M) else numeric(0)
  tau2_D <- if (md$p_D > 0) rep(1, md$p_D) else numeric(0)
  
  lambda2_E <- hyper$a_lambda_E / hyper$b_lambda_E
  lambda2_M <- hyper$a_lambda_M / hyper$b_lambda_M
  lambda2_D <- hyper$a_lambda_D / hyper$b_lambda_D
  
  sigma2_mu <- hyper$gamma_mu2   # escala de la hiperprior; > 0, moderado
  
  # ---------------------------------------------------------------------------
  # Latentes de la mezcla escala e índice de g.l.
  # ---------------------------------------------------------------------------
  xi <- rep(1, md$N)             # media de la prior G(nu/2, nu/2)
  nu <- nu_init
  
  state <- list(
    # bloque beta*
    mu = mu, beta_E = beta_E, beta_M = beta_M, beta_D = beta_D,
    beta_star = beta_star,
    # mezcla escala t
    xi = xi,
    # varianza del intercepto
    sigma2_mu = sigma2_mu,
    # lasso local
    tau2_E = tau2_E, tau2_M = tau2_M, tau2_D = tau2_D,
    # varianzas municipales / departamentales
    kappa2_jk = kappa2_jk, kappa2_k = kappa2_k,
    # lasso global
    lambda2_E = lambda2_E, lambda2_M = lambda2_M, lambda2_D = lambda2_D,
    # grados de libertad
    nu = nu,
    # cantidades cacheadas (coherentes con beta_star inicial)
    zeta = zeta, epsilon = epsilon
  )
  class(state) <- "state_m2"
  
  validate_state(model_data, state)
  state
}


# ==============================================================================
# 4. VALIDACIÓN DEL ESTADO: validate_state()
# ==============================================================================
validate_state <- function(model_data, state) {
  md <- model_data; st <- state
  
  # --- Dimensiones del bloque beta* ---
  stopifnot(length(st$beta_star) == ncol(md$X_star))
  stopifnot(length(st$mu) == 1)
  stopifnot(length(st$beta_E) == md$p_E,
            length(st$beta_M) == md$p_M,
            length(st$beta_D) == md$p_D)
  # Coherencia beta_star == c(mu, beta_E, beta_M, beta_D)
  recompuesto <- c(st$mu, st$beta_E, st$beta_M, st$beta_D)
  if (max(abs(recompuesto - st$beta_star)) > 1e-10) {
    stop("Incoherencia: beta_star != c(mu, beta_E, beta_M, beta_D).")
  }
  
  # --- Soportes positivos ---
  stopifnot(all(st$xi > 0), length(st$xi) == md$N)
  stopifnot(st$sigma2_mu > 0)
  if (md$p_E > 0) stopifnot(all(st$tau2_E > 0))
  if (md$p_M > 0) stopifnot(all(st$tau2_M > 0))
  if (md$p_D > 0) stopifnot(all(st$tau2_D > 0))
  stopifnot(st$lambda2_E > 0, st$lambda2_M > 0, st$lambda2_D > 0)
  stopifnot(st$nu > 2)
  
  # --- Dimensiones de tau2 ---
  stopifnot(length(st$tau2_E) == md$p_E,
            length(st$tau2_M) == md$p_M,
            length(st$tau2_D) == md$p_D)
  
  # --- kappa2: positividad en celdas con observaciones, dimensiones ---
  stopifnot(nrow(st$kappa2_jk) == nrow(md$n_jk), ncol(st$kappa2_jk) == md$K)
  stopifnot(length(st$kappa2_k) == md$K, all(st$kappa2_k > 0))
  for (kk in 1:md$K) {
    stopifnot(all(st$kappa2_jk[1:md$J_k[kk], kk] > 0))
  }
  
  # --- Finitud general ---
  stopifnot(all(is.finite(st$beta_star)), all(is.finite(st$xi)))
  stopifnot(all(is.finite(st$kappa2_k)))
  for (kk in 1:md$K) stopifnot(all(is.finite(st$kappa2_jk[1:md$J_k[kk], kk])))
  
  invisible(TRUE)
}


# ==============================================================================
# 5. PRIMERA CONDICIONAL COMPLETA: update_xi()
# ==============================================================================

update_xi <- function(model_data, state, hyper) {
  md <- model_data; st <- state
  
  # --- Residuos coherentes con el beta_star actual ---
  zeta    <- drop(md$X_star %*% st$beta_star)
  epsilon <- md$y_vec - zeta
  
  # --- kappa2 por observación (expandido desde la matriz por municipio) ---
  kappa2_obs <- expand_kappa2_to_obs(md, st$kappa2_jk)
  
  # --- Parámetros de la Gamma (shape escalar, rate vectorial) ---
  shape <- (st$nu + 1) / 2                                  # escalar
  rate  <- 0.5 * (st$nu + epsilon^2 / kappa2_obs)           # vector de longitud N
  
  # --- Muestreo vectorizado. `rate` NOMBRADO (no posicional como scale). ---
  xi_new <- rgamma(md$N, shape = shape, rate = rate)
  
  # --- Salvaguarda numérica: underflow puede dar 0 exacto en colas extremas ---
  if (any(xi_new <= 0)) {
    xi_new[xi_new <= 0] <- .Machine$double.xmin
  }
  
  # --- Actualizar estado y cachés ---
  st$xi      <- xi_new
  st$zeta    <- zeta
  st$epsilon <- epsilon
  st
}


# ==============================================================================
# 6. VERIFICACIONES COMPUTACIONALES DE update_xi()
# ==============================================================================

check_xi <- function(model_data, state) {
  xi <- state$xi
  stopifnot(length(xi) == model_data$N)   # dimensión
  stopifnot(all(xi > 0))                  # positividad (soporte de la Gamma)
  stopifnot(all(is.finite(xi)))           # sin NaN ni Inf
  invisible(TRUE)
}


# ==============================================================================
# 7. EJECUCIÓN
# ==============================================================================

# source("1_manupulacion_matrices.R")
# model_data <- construir_model_data(cfg, x_ijk_names, z_jk_names, w_k_names)
hyper      <- create_hyperparameters()   
state      <- initialize_state(model_data, hyper, seed = 2025)
state      <- update_xi(model_data, state, hyper) 
check_xi(model_data, state)
