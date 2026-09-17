#_______________________________________________________________________________
#@  MODELO 1 — NORMAL JERARQUICO ESTATICO CON COVARIABLES Y LASSO BAYESIANO

#_______________________________________________________________________________
actualizar_sigma2_mu <- function(datos, estado, hiper) {
  a_post <- (hiper$nu_mu + 1) / 2
  b_post <- ((estado$mu - hiper$mu_mu)^2 + hiper$nu_mu * hiper$gamma_mu^2) / 2
  g <- rgamma(1, shape = a_post, rate = b_post)        # rate = ESCALA de la GI
  estado$sigma2_mu <- 1 / g                            # reciproco -> GI
  estado
}

parametros_sigma2_mu <- function(datos, estado, hiper) {
  a_post <- (hiper$nu_mu + 1) / 2
  b_post <- ((estado$mu - hiper$mu_mu)^2 + hiper$nu_mu * hiper$gamma_mu^2) / 2
  list(a_post = a_post, b_post = b_post,
       media = if (a_post > 1) b_post / (a_post - 1) else NA_real_)
}


check_sigma2_mu <- function(estado) {
  stopifnot(is.numeric(estado$sigma2_mu), length(estado$sigma2_mu) == 1,
            is.finite(estado$sigma2_mu), estado$sigma2_mu > 0)
  invisible(TRUE)
}