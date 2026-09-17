#_______________________________________________________________________________
#@  MODELO 1 — NORMAL JERARQUICO ESTATICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________

actualizar_lambda2_M <- function(datos, estado, hiper) {
  a_post <- hiper$a_lambda_M + datos$p_M
  b_post <- hiper$b_lambda_M + 0.5 * sum(estado$tau2_M)
  estado$lambda2_M <- rgamma(1, shape = a_post, rate = b_post)
  estado
}


parametros_lambda2_M <- function(datos, estado, hiper) {
  a_post <- hiper$a_lambda_M + datos$p_M
  b_post <- hiper$b_lambda_M + 0.5 * sum(estado$tau2_M)
  list(a_post = a_post, b_post = b_post,
       media = a_post / b_post, var = a_post / b_post^2)
}


check_lambda2_M <- function(estado) {
  stopifnot(is.numeric(estado$lambda2_M), length(estado$lambda2_M) == 1,
            is.finite(estado$lambda2_M), estado$lambda2_M > 0)
  invisible(TRUE)
}
