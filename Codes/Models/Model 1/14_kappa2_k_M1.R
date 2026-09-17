#_______________________________________________________________________________
#@  MODELO 1 — NORMAL JERARQUICO ESTATICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________
suma_precisiones_por_depto <- function(datos, estado) {
  prec_cell <- 1 / estado$kappa2_jk                    # 1/kappa2 por celda
  R <- numeric(datos$K)
  agg <- tapply(prec_cell, datos$cell_k, sum)
  idx <- as.integer(names(agg)); R[idx] <- as.numeric(agg)
  R
}

actualizar_kappa2_k <- function(datos, estado, hiper) {
  nu <- hiper$nu_kappa
  R  <- suma_precisiones_por_depto(datos, estado)      # R_k (eq:m14-Rk)
  a_k <- (hiper$alpha_kappa + nu * datos$J_k) / 2      # forma (con /2)
  b_k <- (hiper$beta_kappa + nu * R) / 2               # tasa
  estado$kappa2_k <- rgamma(datos$K, shape = a_k, rate = b_k)  # Gamma DIRECTA
  estado
}

parametros_kappa2_k <- function(datos, estado, hiper) {
  nu <- hiper$nu_kappa
  R  <- suma_precisiones_por_depto(datos, estado)
  a_k <- (hiper$alpha_kappa + nu * datos$J_k) / 2
  b_k <- (hiper$beta_kappa + nu * R) / 2
  list(a_k = a_k, b_k = b_k, media = a_k / b_k, var = a_k / b_k^2, R = R)
}


check_kappa2_k <- function(datos, estado) {
  stopifnot(length(estado$kappa2_k) == datos$K,
            all(is.finite(estado$kappa2_k)), all(estado$kappa2_k > 0))
  invisible(TRUE)
}