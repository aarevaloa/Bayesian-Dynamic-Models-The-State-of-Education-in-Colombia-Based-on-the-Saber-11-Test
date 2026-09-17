#_______________________________________________________________________________
#@  MODELO 2 — t-STUDENT JERÁRQUICO ESTÁTICO CON COVARIABLES Y LASSO BAYESIANO
#_______________________________________________________________________________
update_kappa2_jk <- function(model_data, state, hyper) {
  md <- model_data; st <- state
  
  epsilon <- md$y_vec - drop(md$X_star %*% st$beta_star)
  SS <- weighted_sq_by_muni(md, st$xi, epsilon)
  
  # --- Construir shape y scale por celda, y muestrear ---
  kappa2_jk <- st$kappa2_jk          # parte de la matriz actual (preserva vacías)
  

  for (kk in 1:md$K) {
    Jk <- md$J_k[kk]
    idx_j <- 1:Jk
    
    n_jk_col <- md$n_jk[idx_j, kk]           # tamaños muestrales (vector Jk)
    SS_col   <- SS[idx_j, kk]                # sumas ponderadas (vector Jk)
    kappa2_k_kk <- st$kappa2_k[kk]           # ancla departamental (escalar)
    
    # shape = (n_jk + nu_kappa)/2 ; scale = (nu_kappa*kappa2_k + SS)/2
    shape_col <- (n_jk_col + hyper$nu_kappa) / 2
    scale_col <- (hyper$nu_kappa * kappa2_k_kk + SS_col) / 2
    
    # Muestreo Inversa Gamma vectorizado sobre los Jk municipios del depto
    kappa2_jk[idx_j, kk] <- rinvgamma_rate(Jk, shape = shape_col, scale = scale_col)
  }
  
  st$kappa2_jk <- kappa2_jk
  st
}


# ==============================================================================
# VERIFICACIONES COMPUTACIONALES
# ==============================================================================

check_kappa2_jk <- function(model_data, state) {
  md <- model_data; st <- state
  stopifnot(nrow(st$kappa2_jk) == nrow(md$n_jk), ncol(st$kappa2_jk) == md$K)
  for (kk in 1:md$K) {
    vals <- st$kappa2_jk[1:md$J_k[kk], kk]
    stopifnot(all(vals > 0), all(is.finite(vals)))
  }
  invisible(TRUE)
}


# ==============================================================================
# EJECUCIÓN
# ==============================================================================
#
# source("1_manupulacion_matrices.R")
# model_data <- construir_model_data(cfg, x_ijk_names, z_jk_names, w_k_names)
# source("2_xi.R")
# source("3_beta.R")
# source("4_sigma_mu.R")
# source("5_tau2.R")
# source("6_kappa2_jk.R")
# hyper <- create_hyperparameters()
# state <- initialize_state(model_data, hyper, seed = 2025)
# state <- update_xi(model_data, state, hyper)
# state <- update_beta_star(model_data, state, hyper)
# state <- update_sigma2_mu(model_data, state, hyper)
# state <- update_tau2(model_data, state, hyper)
state <- update_kappa2_jk(model_data, state, hyper)
check_kappa2_jk(model_data, state)
