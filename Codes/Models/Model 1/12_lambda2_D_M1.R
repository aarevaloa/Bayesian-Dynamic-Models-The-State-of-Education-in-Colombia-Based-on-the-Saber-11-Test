#_______________________________________________________________________________
#@  MODELO 1 — NORMAL JERARQUICO ESTATICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________
actualizar_lambda2_D <- function(datos, estado, hiper) {
  a_post <- hiper$a_lambda_D + datos$p_D
  b_post <- hiper$b_lambda_D + 0.5 * sum(estado$tau2_D)   
  estado$lambda2_D <- rgamma(1, shape = a_post, rate = b_post)
  estado
}


parametros_lambda2_D <- function(datos, estado, hiper) {
  a_post <- hiper$a_lambda_D + datos$p_D
  b_post <- hiper$b_lambda_D + 0.5 * sum(estado$tau2_D)
  list(a_post = a_post, b_post = b_post,
       media = a_post / b_post, var = a_post / b_post^2)
}


check_lambda2_D <- function(estado) {
  stopifnot(is.numeric(estado$lambda2_D), length(estado$lambda2_D) == 1,
            is.finite(estado$lambda2_D), estado$lambda2_D > 0)
  invisible(TRUE)
}