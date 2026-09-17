#_______________________________________________________________________________
#@  MODELO 1 — NORMAL JERARQUICO ESTATICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________

actualizar_kappa2_jk <- function(datos, estado, hiper) {
  nu <- hiper$nu_kappa
  SSR <- SSR_por_celda(datos, estado)                  # longitud n_cells (eq:m13-SSR)
  kappa2_k_cell <- estado$kappa2_k[datos$cell_k]       # kappa2_k del depto de cada celda
  a_c <- (datos$n_jk_flat + nu) / 2                    # forma por celda
  b_c <- (SSR + nu * kappa2_k_cell) / 2                # ESCALA por celda
  g <- rgamma(datos$n_cells, shape = a_c, rate = b_c)  # rate = ESCALA de la GI
  estado$kappa2_jk <- 1 / g                            # reciproco -> GI
  estado
}

parametros_kappa2_jk <- function(datos, estado, hiper) {
  nu <- hiper$nu_kappa
  SSR <- SSR_por_celda(datos, estado)
  kappa2_k_cell <- estado$kappa2_k[datos$cell_k]
  a_c <- (datos$n_jk_flat + nu) / 2
  b_c <- (SSR + nu * kappa2_k_cell) / 2
  media <- ifelse(a_c > 1, b_c / (a_c - 1), NA_real_)
  list(a_c = a_c, b_c = b_c, media = media, SSR = SSR)
}


check_kappa2_jk <- function(datos, estado) {
  stopifnot(length(estado$kappa2_jk) == datos$n_cells,
            all(is.finite(estado$kappa2_jk)), all(estado$kappa2_jk > 0))
  invisible(TRUE)
}