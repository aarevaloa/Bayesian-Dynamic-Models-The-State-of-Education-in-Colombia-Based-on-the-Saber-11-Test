#_______________________________________________________________________________
#@  MODELO 1 — CORRIDA DEL MUESTREADOR SOBRE DATOS REALES
#_______________________________________________________________________________

# ---- (1) Carpeta del codigo (AJUSTAR) ----
RUTA_CODIGO <- "C:/Users/USUARIO/OneDrive - Universidad Nacional de Colombia/Tesis/Escritorio/Códigos/Modelos/Modelo 1" 

cargar <- function(f) source(file.path(RUTA_CODIGO, f))
cargar("1_manupulacion_matrices_M1.R")     
cargar("2_Hiperparametros_M1.R")
cargar("3_mu_M1.R")    
cargar("4_beta_E_M1.R")    
cargar("5_beta_M_M1.R") 
cargar("6_beta_D_M1.R")
cargar("7_tau2_E_M1.R")
cargar("8_tau2_M_M1.R")
cargar("9_tau2_D_M1.R")
cargar("10_lamda2_E_M1.R")
cargar("11_lambda2_M_M1.R")
cargar("12_lambda2_D_M1.R")
cargar("13_kappa2_jk_M1.R")
cargar("14_kappa2_k_M1.R")
cargar("15_sigma2_mu_M1.R")
cargar("16_Muestreador_GIBBS_Modelo_1_a_M1.R")
cargar("16_Muestreador_GIBBS_Modelo_1_b_M1.R")

# ---- (2) Construir el objeto de datos a partir del CSV real ----
datos <- construir_datos(cfg, x_ijk_names, z_jk_names, w_k_names)
cat(sprintf("\nDimensiones: N=%d, K=%d, n_cells=%d, p=%d (E=%d, M=%d, D=%d)\n\n",
            datos$N, datos$K, datos$n_cells, datos$p, datos$p_E, datos$p_M, datos$p_D))

# ---- (3) Hiperparametros (defaults razonables; ajustar si se desea) ----
hiper <- crear_hiperparametros(
  mu_mu = mean(datos$y_vec),   
  nu_mu = 3, gamma_mu = sd(datos$y_vec),
  varsigma2_E = 1, varsigma2_M = 1, varsigma2_D = 1,
  a_lambda_E = 1, b_lambda_E = 1,
  a_lambda_M = 1, b_lambda_M = 1,
  a_lambda_D = 1, b_lambda_D = 1,
  nu_kappa = 3, alpha_kappa = 2, beta_kappa = 2
)

# Carpeta donde se escribiran los .bin
DIR_SALIDA <- file.path(RUTA_CODIGO, "mcmc_salida_m1")

#_______________________________________________________________________________
# PRUEBA RAPIDA: 
#_______________________________________________________________________________
# prueba <- correr_gibbs_disco(datos, hiper,
#                              n_iter = 25, burnin = 25, thin = 1,
#                              dir_salida = file.path(DIR_SALIDA, "prueba"),
#                              semilla = 2026, print_cada = 1,
#                              guardar_kappa2_jk = TRUE)

#_______________________________________________________________________________
# PRODUCCION 
#_______________________________________________________________________________
cat("\n################## CORRIDA DE PRODUCCION ##################\n")
salida <- correr_gibbs_disco(datos, hiper,
                             n_iter = 6000,    # muestras guardadas
                             burnin = 5000,    # descartadas
                             thin   = 5,       # 1 de cada 2 tras burn-in
                             dir_salida = DIR_SALIDA,
                             semilla = 2026,
                             print_cada = 1, # progreso cada 200 iter
                             guardar_kappa2_jk = TRUE,  # se guarda en disco
                             guardar_loglik = TRUE)    # TRUE si quieres WAIC/LOO

# mu_chain     <- leer_muestras_m1(DIR_SALIDA, "mu")
# beta_E_chain <- leer_muestras_m1(DIR_SALIDA, "beta_E")
# kappa2_k_ch  <- leer_muestras_m1(DIR_SALIDA, "kappa2_k")
# cat(sprintf("mu posterior: media=%.2f IC95%%=[%.2f, %.2f]\n",
#             mean(mu_chain), quantile(mu_chain,0.025), quantile(mu_chain,0.975)))