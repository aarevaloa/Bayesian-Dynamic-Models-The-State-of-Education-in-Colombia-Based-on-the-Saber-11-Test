#_______________________________________________________________________________
#@  CONFIGURACION DE RUTAS — MOTOR DE SIMULACION 
#_______________________________________________________________________________

# =============================================================================
# (1) CARPETA RAIZ DEL PROYECTO DE SIMULACION
# =============================================================================
RUTA_SIMULACION <- "C:/Users/USUARIO/OneDrive - Universidad Nacional de Colombia/Tesis/Escritorio/Códigos/Simulaciones"

# Carpeta temporal donde el motor escribira los .bin de M1/M2/M3 durante las
RUTA_TRABAJO <- "C:/Users/USUARIO/OneDrive - Universidad Nacional de Colombia/Tesis/Escritorio/Códigos/Simulaciones/_prueba"

# =============================================================================
# (2) CARPETAS DE LOS 4 MODELOS
# =============================================================================
RUTA_M1 <- "C:/Users/USUARIO/OneDrive - Universidad Nacional de Colombia/Tesis/Escritorio/Códigos/Modelos/Modelo 1"
RUTA_M2 <- "C:/Users/USUARIO/OneDrive - Universidad Nacional de Colombia/Tesis/Escritorio/Códigos/Modelos/Modelo 2"
RUTA_M3 <- "C:/Users/USUARIO/OneDrive - Universidad Nacional de Colombia/Tesis/Escritorio/Códigos/Modelos/Modelo 3"
RUTA_M4 <- "C:/Users/USUARIO/OneDrive - Universidad Nacional de Colombia/Tesis/Escritorio/Códigos/Modelos/Modelo 4"

# =============================================================================
# (3) NOMBRES DE ARCHIVO POR MODELO
# =============================================================================

ARCHIVOS_M1 <- c(
  "1_manupulacion_matrices_M1.R", "2_Hiperparametros_M1.R", "3_mu_M1.R",
  "4_beta_E_M1.R", "5_beta_M_M1.R", "6_beta_D_M1.R", "7_tau2_E_M1.R",
  "8_tau2_M_M1.R", "9_tau2_D_M1.R", "10_lamda2_E_M1.R", "11_lambda2_M_M1.R",
  "12_lambda2_D_M1.R", "13_kappa2_jk_M1.R", "14_kappa2_k_M1.R", "15_sigma2_mu_M1.R",
  "16_Muestreador_GIBBS_Modelo_1_a_M1.R", "16_Muestreador_GIBBS_Modelo_1_b_M1.R"
)

# --- MODELO 2 (t estatico):
ARCHIVOS_M2 <- c(
  "1_manupulacion_matrices_M2.R", "2_xi_M2.R", "3_beta_M2.R", "4_sigma_mu_M2.R",
  "5_tau2_M2.R", "6_kappa2_jk_M2.R", "7_kappa2_k_M2.R", "8_lambda2_M2.R", "9_nu_M2.R",
  "10_gibbs_samper_M2.R"
)

# --- MODELO 3 (normal dinamico):
ARCHIVOS_M3_FUNCIONES <- c(
  "2_forward_filter_M3.R", "3_backward_sample_M3.R", "4_tau2_M3.R",
  "5_lambda2_M3.R", "6_v_M3.R", "7_phi_M3.R"
)

ARCHIVO_M3_GIBBS_PATCHED <- "8_gibbs_M3_PATCHED.R"
ARCHIVO_M3_GIBBS_ORIGINAL <- "8_gibbs_M3.R"

# --- MODELO 4 (t dinamico):
ARCHIVOS_M4_FUN <- c(
  "2_FFBS_theta_v5_FUN_SIM.R", "3_Escalas_latentes_xi_v3_FUN_SIM.R",
  "4_precision_municipal_v_v2_FUN_SIM.R", "5_Muestreo_retrospectivo_beta_gamma_phi_v6_FUN_SIM.R",
  "6_LASSO_dinamico_tau_lambda_phi_v3_FUN_SIM.R", "7_MH_nu_v2_FUN_SIM.R",
  "8_Muestreador_GIBBS_v2_FUN_SIM.R"
)


# =============================================================================
# (3-bis) NOMBRES DE LOS ARCHIVOS DEL MOTOR EN RUTA_SIMULACION
# =============================================================================
ARCHIVO_GENERADOR    <- "2_generador_datos.R"   
ARCHIVO_ADAPTADOR_M3 <- "adaptador_M3.R"       
ARCHIVO_MOTOR    <- "3_motor.R"              
ARCHIVO_PARALELO <- "3_motor_D_paralelo.R"   

# =============================================================================
# (4) VALIDACION DE LA CONFIGURACION (no editar)
# =============================================================================
.validar_config_rutas <- function() {
  err <- character(0)
  for (nm in c("RUTA_SIMULACION","RUTA_M1","RUTA_M2","RUTA_M3","RUTA_M4")) {
    ruta <- get(nm)
    if (!dir.exists(ruta)) err <- c(err, sprintf("  %s no existe: %s", nm, ruta))
  }
  if (!dir.exists(RUTA_TRABAJO)) {
    dir.create(RUTA_TRABAJO, recursive = TRUE, showWarnings = FALSE)
    cat(sprintf("  [creada] carpeta de trabajo: %s\n", RUTA_TRABAJO))
  }
  if (length(err) > 0) {
    cat("PROBLEMAS DE CONFIGURACION:\n"); cat(paste(err, collapse="\n"), "\n")
    stop("Corrige las rutas en config_rutas.R antes de continuar.", call. = FALSE)
  }
  cat("  Rutas validadas correctamente.\n")
  invisible(TRUE)
}
