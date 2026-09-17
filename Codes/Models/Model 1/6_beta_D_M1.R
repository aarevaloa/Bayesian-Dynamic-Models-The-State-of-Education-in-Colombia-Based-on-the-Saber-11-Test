#_______________________________________________________________________________
#@  MODELO 1 — NORMAL JERARQUICO ESTATICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________
precision_agregada_depto <- function(datos, estado) {
  w_cell <- datos$n_jk_flat / estado$kappa2_jk       # n_jk/kappa2_jk por celda
  P <- numeric(datos$K)
  agg <- tapply(w_cell, datos$cell_k, sum)           # agrega celdas -> depto
  idx <- as.integer(names(agg)); P[idx] <- as.numeric(agg)
  P
}

suma_residuos_por_depto <- function(datos, estado) {
  u <- residuo_parcial(datos, estado, "D")           # y - mu - x'bE - z'bM
  w_obs <- 1 / expandir_celda_a_obs(estado$kappa2_jk, datos)  # 1/kappa2 por obs
  contrib <- w_obs * u
  U <- numeric(datos$K)
  agg <- tapply(contrib, datos$row_k, sum)           # agrega obs -> depto
  idx <- as.integer(names(agg)); U[idx] <- as.numeric(agg)
  U
}

ensamblar_SD_mD <- function(datos, estado) {
  P <- precision_agregada_depto(datos, estado)       # P_k por depto
  U <- suma_residuos_por_depto(datos, estado)        # U_k por depto
  Wp <- datos$W * P                                  # fila k escalada por P_k
  S_D <- crossprod(datos$W, Wp)                      # W' diag(P) W  (p_D x p_D)
  m_D <- as.vector(crossprod(datos$W, U))            # W' U  (p_D)
  list(S_D = S_D, m_D = m_D, P = P, U = U)
}


actualizar_beta_D <- function(datos, estado, hiper) {
  if (datos$p_D == 0) return(estado)
  comp <- ensamblar_SD_mD(datos, estado)
  Omega_D <- diag(1 / (hiper$varsigma2_D * estado$tau2_D), nrow = datos$p_D)
  Lambda_D <- comp$S_D + Omega_D
  draw <- muestrear_mvn_precision(Lambda_D, comp$m_D)
  estado$beta_D <- draw$beta
  estado
}


parametros_beta_D <- function(datos, estado, hiper) {
  comp <- ensamblar_SD_mD(datos, estado)
  Omega_D <- diag(1 / (hiper$varsigma2_D * estado$tau2_D), nrow = datos$p_D)
  Lambda_D <- comp$S_D + Omega_D
  R <- chol(Lambda_D)
  mu_D_star <- backsolve(R, forwardsolve(t(R), comp$m_D))
  list(Lambda_D = Lambda_D, mu_D_star = as.vector(mu_D_star),
       Sigma_D = chol2inv(R), S_D = comp$S_D, m_D = comp$m_D, Omega_D = Omega_D,
       P = comp$P, U = comp$U)
}


check_beta_D <- function(datos, estado) {
  stopifnot(length(estado$beta_D) == datos$p_D, all(is.finite(estado$beta_D)))
  invisible(TRUE)
}