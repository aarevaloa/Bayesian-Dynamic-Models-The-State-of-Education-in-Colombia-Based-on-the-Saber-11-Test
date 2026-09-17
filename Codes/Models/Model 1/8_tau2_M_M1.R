#_______________________________________________________________________________
#@  MODELO 1 — NORMAL JERARQUICO ESTATICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________
actualizar_tau2_M <- function(datos, estado, hiper) {
  if (datos$p_M == 0) return(estado)
  lambda_M   <- sqrt(estado$lambda2_M)
  varsigma_M <- sqrt(hiper$varsigma2_M)
  abs_beta   <- pmax(abs(estado$beta_M), EPS_BETA)
  mu_gin     <- lambda_M * varsigma_M / abs_beta        # eq:m8-despeje-mu
  shape      <- estado$lambda2_M                        # eq:m8-igualdad-reciproco
  eta <- rinvgauss_msh(datos$p_M, mean = mu_gin, shape = shape)
  estado$tau2_M <- 1 / eta
  estado
}


parametros_tau2_M <- function(datos, estado, hiper) {
  lambda_M <- sqrt(estado$lambda2_M); varsigma_M <- sqrt(hiper$varsigma2_M)
  abs_beta <- pmax(abs(estado$beta_M), EPS_BETA)
  list(mu_gin = lambda_M * varsigma_M / abs_beta,
       shape  = rep(estado$lambda2_M, datos$p_M))
}


check_tau2_M <- function(datos, estado) {
  stopifnot(length(estado$tau2_M) == datos$p_M,
            all(is.finite(estado$tau2_M)), all(estado$tau2_M > 0))
  invisible(TRUE)
}