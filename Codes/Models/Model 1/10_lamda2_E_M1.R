#_______________________________________________________________________________
#@  MODELO 1 — NORMAL JERARQUICO ESTATICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________
actualizar_lambda2_E <- function(datos, estado, hiper) {
  a_post <- hiper$a_lambda_E + datos$p_E
  b_post <- hiper$b_lambda_E + 0.5 * sum(estado$tau2_E)   # sum() de vacio = 0
  estado$lambda2_E <- rgamma(1, shape = a_post, rate = b_post)  # rate EXPLICITO
  estado
}

parametros_lambda2_E <- function(datos, estado, hiper) {
  a_post <- hiper$a_lambda_E + datos$p_E
  b_post <- hiper$b_lambda_E + 0.5 * sum(estado$tau2_E)
  list(a_post = a_post, b_post = b_post,
       media = a_post / b_post, var = a_post / b_post^2)
}

check_lambda2_E <- function(estado) {
  stopifnot(is.numeric(estado$lambda2_E), length(estado$lambda2_E) == 1,
            is.finite(estado$lambda2_E), estado$lambda2_E > 0)
  invisible(TRUE)
}