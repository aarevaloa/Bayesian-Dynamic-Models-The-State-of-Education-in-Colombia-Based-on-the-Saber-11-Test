#_______________________________________________________________________________
#@  MODELO 1 — NORMAL JERARQUICO ESTATICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________
# ==============================================================================
# 1. FABRICA DE HIPERPARAMETROS
# ==============================================================================

crear_hiperparametros <- function(
    mu_mu = 0, nu_mu = 3, gamma_mu = 1,
    varsigma2_E = 1, varsigma2_M = 1, varsigma2_D = 1,
    a_lambda_E = 1, b_lambda_E = 1,
    a_lambda_M = 1, b_lambda_M = 1,
    a_lambda_D = 1, b_lambda_D = 1,
    nu_kappa = 3, alpha_kappa = 1, beta_kappa = 1) {
  
  hiper <- list(
    mu_mu = mu_mu, nu_mu = nu_mu, gamma_mu = gamma_mu,
    varsigma2_E = varsigma2_E, varsigma2_M = varsigma2_M, varsigma2_D = varsigma2_D,
    a_lambda_E = a_lambda_E, b_lambda_E = b_lambda_E,
    a_lambda_M = a_lambda_M, b_lambda_M = b_lambda_M,
    a_lambda_D = a_lambda_D, b_lambda_D = b_lambda_D,
    nu_kappa = nu_kappa, alpha_kappa = alpha_kappa, beta_kappa = beta_kappa
  )
  
  positivos <- c("nu_mu", "gamma_mu",
                 "varsigma2_E", "varsigma2_M", "varsigma2_D",
                 "a_lambda_E", "b_lambda_E", "a_lambda_M", "b_lambda_M",
                 "a_lambda_D", "b_lambda_D",
                 "nu_kappa", "alpha_kappa", "beta_kappa")
  for (nm in positivos) {
    v <- hiper[[nm]]
    if (!is.numeric(v) || length(v) != 1 || !is.finite(v) || v <= 0)
      stop(sprintf("Hiperparametro '%s' debe ser un escalar finito > 0 (es %s).",
                   nm, as.character(v)))
  }
  if (!is.numeric(mu_mu) || length(mu_mu) != 1 || !is.finite(mu_mu))
    stop("Hiperparametro 'mu_mu' debe ser un escalar finito.")

  class(hiper) <- "model1_hiper"
  hiper
}


# ==============================================================================
# 2. AUXILIAR: expandir un valor por celda a longitud N
# ==============================================================================
expandir_celda_a_obs <- function(vec_cell, datos) {
  stopifnot(length(vec_cell) == datos$n_cells)
  vec_cell[datos$row_cell]
}


# ==============================================================================
# 3. INICIALIZACION DEL ESTADO
# ==============================================================================
inicializar_estado <- function(datos, hiper, semilla = NULL) {
  if (!is.null(semilla)) set.seed(semilla)
  vy <- var(datos$y_vec)
  if (!is.finite(vy) || vy <= 0) vy <- 1
  
  estado <- list(
    mu        = mean(datos$y_vec),
    sigma2_mu = hiper$gamma_mu^2,
    
    beta_E = numeric(datos$p_E),
    beta_M = numeric(datos$p_M),
    beta_D = numeric(datos$p_D),
    
    tau2_E = rep(1, datos$p_E),
    tau2_M = rep(1, datos$p_M),
    tau2_D = rep(1, datos$p_D),
    
    lambda2_E = 1,
    lambda2_M = 1,
    lambda2_D = 1,
    
    kappa2_jk = rep(vy, datos$n_cells),   # orden cell_index (k->j)
    kappa2_k  = rep(vy, datos$K)
  )
  class(estado) <- "model1_estado"
  estado
}


# ==============================================================================
# 4. VALIDACION DEL ESTADO
# ==============================================================================
validar_estado <- function(datos, estado) {
  # ---- Escalares ----
  stopifnot(is.numeric(estado$mu), length(estado$mu) == 1, is.finite(estado$mu))
  stopifnot(is.numeric(estado$sigma2_mu), length(estado$sigma2_mu) == 1,
            is.finite(estado$sigma2_mu), estado$sigma2_mu > 0)
  for (nm in c("lambda2_E", "lambda2_M", "lambda2_D")) {
    v <- estado[[nm]]
    stopifnot(length(v) == 1, is.finite(v), v > 0)
  }
  
  # ---- Coeficientes: dimensiones correctas, finitos ----
  stopifnot(length(estado$beta_E) == datos$p_E, all(is.finite(estado$beta_E)))
  stopifnot(length(estado$beta_M) == datos$p_M, all(is.finite(estado$beta_M)))
  stopifnot(length(estado$beta_D) == datos$p_D, all(is.finite(estado$beta_D)))
  
  # ---- Escalas locales: dimensiones, positividad, finitud ----
  stopifnot(length(estado$tau2_E) == datos$p_E, all(is.finite(estado$tau2_E)),
            all(estado$tau2_E > 0))
  stopifnot(length(estado$tau2_M) == datos$p_M, all(is.finite(estado$tau2_M)),
            all(estado$tau2_M > 0))
  stopifnot(length(estado$tau2_D) == datos$p_D, all(is.finite(estado$tau2_D)),
            all(estado$tau2_D > 0))
  
  # ---- Varianzas: dimensiones, positividad, finitud ----
  stopifnot(length(estado$kappa2_jk) == datos$n_cells,
            all(is.finite(estado$kappa2_jk)), all(estado$kappa2_jk > 0))
  stopifnot(length(estado$kappa2_k) == datos$K,
            all(is.finite(estado$kappa2_k)), all(estado$kappa2_k > 0))
  
  invisible(TRUE)
}


# ==============================================================================
# 5. AUXILIAR: media condicional zeta y residuos parciales
# ==============================================================================
zeta_obs <- function(datos, estado) {
  z <- rep(estado$mu, datos$N)
  if (datos$p_E > 0) z <- z + as.vector(datos$X_E %*% estado$beta_E)
  if (datos$p_M > 0) z <- z + as.vector(datos$X_M %*% estado$beta_M)
  if (datos$p_D > 0) z <- z + as.vector(datos$X_D %*% estado$beta_D)
  z
}

residuo_parcial <- function(datos, estado, excluir = c("mu","E","M","D","none")) {
  excluir <- match.arg(excluir)
  r <- datos$y_vec
  # restar mu salvo si se excluye mu
  if (excluir != "mu") r <- r - estado$mu
  # restar nivel E salvo si se excluye E
  if (excluir != "E" && datos$p_E > 0) r <- r - as.vector(datos$X_E %*% estado$beta_E)
  # restar nivel M salvo si se excluye M
  if (excluir != "M" && datos$p_M > 0) r <- r - as.vector(datos$X_M %*% estado$beta_M)
  # restar nivel D salvo si se excluye D
  if (excluir != "D" && datos$p_D > 0) r <- r - as.vector(datos$X_D %*% estado$beta_D)
  r
}


# ==============================================================================
# 6. AUXILIAR: SSR por celda
# ==============================================================================
SSR_por_celda <- function(datos, estado) {
  resid <- residuo_parcial(datos, estado, "none")   # y - zeta
  sq <- resid^2
  # agregacion por celda en orden 1..n_cells
  out <- numeric(datos$n_cells)
  agg <- tapply(sq, datos$row_cell, sum)
  # tapply indexa por los valores de row_cell presentes; alinear a 1..n_cells
  idx <- as.integer(names(agg))
  out[idx] <- as.numeric(agg)
  out
}

# ==============================================================================
# 7. AUXILIAR: A y B de la condicional de mu
# ==============================================================================
precision_total_y_residuos <- function(datos, estado) {
  r <- residuo_parcial(datos, estado, "mu")          # y - x'bE - z'bM - w'bD
  kappa2_obs <- expandir_celda_a_obs(estado$kappa2_jk, datos)
  prec_obs <- 1 / kappa2_obs                          # 1/kappa2_{cell(i)}
  A <- sum(prec_obs)                                  # = sum_{k,j} n_jk/kappa2_jk
  B <- sum(prec_obs * r)                              # = sum_{k,j} (1/k2) sum_i r
  list(A = A, B = B, r = r)
}

