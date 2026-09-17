#_______________________________________________________________________________
#@  MODELO 1 — NORMAL JERARQUICO ESTATICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________
suma_residuos_por_celda <- function(datos, estado) {
  s <- residuo_parcial(datos, estado, "M")           # y - mu - x'bE - w'bD
  out <- numeric(datos$n_cells)
  agg <- tapply(s, datos$row_cell, sum)
  idx <- as.integer(names(agg)); out[idx] <- as.numeric(agg)
  out
}

ensamblar_SM_mM <- function(datos, estado) {
  S_cell <- suma_residuos_por_celda(datos, estado)   # S_jk por celda
  inv_k2 <- 1 / estado$kappa2_jk                     # 1/kappa2_c por celda
  w_quad <- datos$n_jk_flat * inv_k2                 # n_jk/kappa2_jk
  w_lin  <- S_cell * inv_k2                          # S_jk/kappa2_jk
  Zw <- datos$Z * w_quad                             # fila c escalada por w_quad[c]
  S_M <- crossprod(datos$Z, Zw)                      # Z' W_M Z  (p_M x p_M)
  m_M <- as.vector(crossprod(datos$Z, w_lin))        # Z' (v * S_cell)  (p_M)
  list(S_M = S_M, m_M = m_M, S_cell = S_cell, w_quad = w_quad, w_lin = w_lin)
}


actualizar_beta_M <- function(datos, estado, hiper) {
  if (datos$p_M == 0) return(estado)
  comp <- ensamblar_SM_mM(datos, estado)
  Omega_M <- diag(1 / (hiper$varsigma2_M * estado$tau2_M), nrow = datos$p_M)
  Lambda_M <- comp$S_M + Omega_M
  draw <- muestrear_mvn_precision(Lambda_M, comp$m_M)
  estado$beta_M <- draw$beta
  estado
}


parametros_beta_M <- function(datos, estado, hiper) {
  comp <- ensamblar_SM_mM(datos, estado)
  Omega_M <- diag(1 / (hiper$varsigma2_M * estado$tau2_M), nrow = datos$p_M)
  Lambda_M <- comp$S_M + Omega_M
  R <- chol(Lambda_M)
  mu_M_star <- backsolve(R, forwardsolve(t(R), comp$m_M))
  list(Lambda_M = Lambda_M, mu_M_star = as.vector(mu_M_star),
       Sigma_M = chol2inv(R), S_M = comp$S_M, m_M = comp$m_M, Omega_M = Omega_M)
}


check_beta_M <- function(datos, estado) {
  stopifnot(length(estado$beta_M) == datos$p_M, all(is.finite(estado$beta_M)))
  invisible(TRUE)
}

