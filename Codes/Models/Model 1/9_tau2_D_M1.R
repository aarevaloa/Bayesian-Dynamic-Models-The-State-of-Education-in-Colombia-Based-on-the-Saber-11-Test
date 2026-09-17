#_______________________________________________________________________________
#@  MODELO 1 — NORMAL JERARQUICO ESTATICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________
actualizar_tau2_D <- function(datos, estado, hiper) {
  if (datos$p_D == 0) return(estado)
  lambda_D   <- sqrt(estado$lambda2_D)
  varsigma_D <- sqrt(hiper$varsigma2_D)
  abs_beta   <- pmax(abs(estado$beta_D), EPS_BETA)
  mu_gin     <- lambda_D * varsigma_D / abs_beta        # eq:m9-despeje-mu
  shape      <- estado$lambda2_D                        # eq:m9-igualdad-reciproco
  eta <- rinvgauss_msh(datos$p_D, mean = mu_gin, shape = shape)
  estado$tau2_D <- 1 / eta
  estado
}

parametros_tau2_D <- function(datos, estado, hiper) {
  if (datos$p_D == 0) return(list(mu_gin = numeric(0), shape = numeric(0)))
  lambda_D <- sqrt(estado$lambda2_D); varsigma_D <- sqrt(hiper$varsigma2_D)
  abs_beta <- pmax(abs(estado$beta_D), EPS_BETA)
  list(mu_gin = lambda_D * varsigma_D / abs_beta,
       shape  = rep(estado$lambda2_D, datos$p_D))
}

check_tau2_D <- function(datos, estado) {
  stopifnot(length(estado$tau2_D) == datos$p_D)
  if (datos$p_D > 0)
    stopifnot(all(is.finite(estado$tau2_D)), all(estado$tau2_D > 0))
  invisible(TRUE)
}