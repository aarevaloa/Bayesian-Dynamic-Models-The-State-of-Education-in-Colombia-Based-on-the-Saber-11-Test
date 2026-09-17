#_______________________________________________________________________________
#@  MODELO 1 — NORMAL JERARQUICO ESTATICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________
EPS_BETA <- 1e-8

rinvgauss_msh <- function(n, mean, shape) {
  mu  <- rep_len(mean,  n)
  lam <- rep_len(shape, n)
  stopifnot(all(mu > 0), all(lam > 0))
  nu <- rnorm(n)
  y  <- nu^2                                          
  x1 <- mu + (mu^2 * y) / (2 * lam) -
    (mu / (2 * lam)) * sqrt(4 * mu * lam * y + mu^2 * y^2)
  u  <- runif(n)
  x  <- ifelse(u <= mu / (mu + x1), x1, mu^2 / x1)
  x[!is.finite(x) | x <= 0] <- mu[!is.finite(x) | x <= 0]
  x
}


actualizar_tau2_E <- function(datos, estado, hiper) {
  if (datos$p_E == 0) return(estado)
  lambda_E   <- sqrt(estado$lambda2_E)
  varsigma_E <- sqrt(hiper$varsigma2_E)
  abs_beta   <- pmax(abs(estado$beta_E), EPS_BETA)      # cota inferior
  mu_gin     <- lambda_E * varsigma_E / abs_beta        # eq:m7-despeje-mu
  shape      <- estado$lambda2_E                        # eq:m7-igualdad-reciproco
  eta <- rinvgauss_msh(datos$p_E, mean = mu_gin, shape = shape)
  estado$tau2_E <- 1 / eta
  estado
}


parametros_tau2_E <- function(datos, estado, hiper) {
  lambda_E <- sqrt(estado$lambda2_E); varsigma_E <- sqrt(hiper$varsigma2_E)
  abs_beta <- pmax(abs(estado$beta_E), EPS_BETA)
  list(mu_gin = lambda_E * varsigma_E / abs_beta,
       shape  = rep(estado$lambda2_E, datos$p_E))
}


check_tau2_E <- function(datos, estado) {
  stopifnot(length(estado$tau2_E) == datos$p_E,
            all(is.finite(estado$tau2_E)), all(estado$tau2_E > 0))
  invisible(TRUE)
}
