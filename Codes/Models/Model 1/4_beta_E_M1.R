#_______________________________________________________________________________
#@  MODELO 1 — NORMAL JERARQUICO ESTATICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________
ensamblar_SE_mE <- function(datos, estado) {
  r <- residuo_parcial(datos, estado, "E")              # y - mu - z'bM - w'bD
  w <- 1 / expandir_celda_a_obs(estado$kappa2_jk, datos) # 1/kappa2 por obs
  Xw <- datos$X_E * w                                    # fila i escalada por w_i
  S_E <- crossprod(datos$X_E, Xw)                        # X_E' W X_E  (p_E x p_E)
  m_E <- as.vector(crossprod(datos$X_E, w * r))          # X_E' (w*r)  (p_E)
  list(S_E = S_E, m_E = m_E, r = r, w = w)
}

muestrear_mvn_precision <- function(Lambda, m) {
  p <- length(m)
  R <- chol(Lambda)                                 
  mu_star <- backsolve(R, forwardsolve(t(R), m))
  z <- rnorm(p)
  v <- backsolve(R, z)
  list(beta = as.vector(mu_star + v), mu_star = as.vector(mu_star), R = R)
}


actualizar_beta_E <- function(datos, estado, hiper) {
  if (datos$p_E == 0) return(estado)
  comp <- ensamblar_SE_mE(datos, estado)
  Omega_E <- diag(1 / (hiper$varsigma2_E * estado$tau2_E), nrow = datos$p_E)
  Lambda_E <- comp$S_E + Omega_E
  draw <- muestrear_mvn_precision(Lambda_E, comp$m_E)
  estado$beta_E <- draw$beta
  estado
}


parametros_beta_E <- function(datos, estado, hiper) {
  comp <- ensamblar_SE_mE(datos, estado)
  Omega_E <- diag(1 / (hiper$varsigma2_E * estado$tau2_E), nrow = datos$p_E)
  Lambda_E <- comp$S_E + Omega_E
  R <- chol(Lambda_E)
  mu_E_star <- backsolve(R, forwardsolve(t(R), comp$m_E))
  list(Lambda_E = Lambda_E, mu_E_star = as.vector(mu_E_star),
       Sigma_E = chol2inv(R), S_E = comp$S_E, m_E = comp$m_E, Omega_E = Omega_E)
}


check_beta_E <- function(datos, estado) {
  stopifnot(length(estado$beta_E) == datos$p_E, all(is.finite(estado$beta_E)))
  invisible(TRUE)
}