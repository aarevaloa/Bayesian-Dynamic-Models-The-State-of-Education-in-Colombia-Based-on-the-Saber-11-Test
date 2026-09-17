#_______________________________________________________________________________
#@  MODELO 1 — NORMAL JERARQUICO ESTATICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________
actualizar_mu <- function(datos, estado, hiper) {
  AB <- precision_total_y_residuos(datos, estado)
  A <- AB$A
  B <- AB$B
  
  prec_previa <- 1 / estado$sigma2_mu          # 1/sigma2_mu > 0
  prec_post   <- A + prec_previa               # precision posterior
  var_star    <- 1 / prec_post                 # sigma2_star_mu
  mu_star     <- var_star * (B + hiper$mu_mu * prec_previa)
  
  estado$mu <- rnorm(1, mean = mu_star, sd = sqrt(var_star))
  estado
}


parametros_mu <- function(datos, estado, hiper) {
  AB <- precision_total_y_residuos(datos, estado)
  prec_previa <- 1 / estado$sigma2_mu
  var_star <- 1 / (AB$A + prec_previa)
  mu_star  <- var_star * (AB$B + hiper$mu_mu * prec_previa)
  list(mu_star = mu_star, var_star = var_star, A = AB$A, B = AB$B)
}

check_mu <- function(estado) {
  stopifnot(is.numeric(estado$mu), length(estado$mu) == 1, is.finite(estado$mu))
  invisible(TRUE)
}
