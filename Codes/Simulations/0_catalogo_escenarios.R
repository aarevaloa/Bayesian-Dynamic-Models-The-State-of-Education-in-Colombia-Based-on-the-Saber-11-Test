#_______________________________________________________________________________
# @author: aarevaloa
# Define los 28 escenarios (S01-S19 estaticos + D01-D09 dinamicos) como configs
#_______________________________________________________________________________

.cfg_base <- function() {
  cfg <- default_config()
  # tamano de referencia (mediano)
  cfg$D <- 15L; cfg$M_k <- 10L; cfg$n_jk <- 50L
  # dimension de covariables (del documento)
  cfg$p_E <- 5L; cfg$p_M <- 3L; cfg$p_D <- 2L
  # varianza de referencia
  cfg$alpha_kappa <- 8; cfg$beta_kappa <- 8   # E[kappa2_k]=1, nu_kappa=8
  cfg$nu_kappa <- 5
  # colinealidad de referencia
  cfg$rho_X <- 0.3
  # efectos: mezcla heterogenea con algo de esparsidad
  cfg$effect_type_E <- "mixed"; cfg$effect_type_M <- "mixed"; cfg$effect_type_D <- "strong"
  cfg$sparsity_E <- 0.4; cfg$sparsity_M <- 0.0; cfg$sparsity_D <- 0.0
  # por defecto: estatico, errores normales (cada familia cambia lo suyo)
  cfg$dynamics <- "static"
  cfg$error_family <- "normal"
  cfg
}


# =============================================================================
# FAMILIA 1 — TAMANO MUESTRAL  (S01, S02, S03)
# =============================================================================

S01_pequeno <- function() {
  cfg <- .cfg_base()
  cfg$D <- 5L; cfg$M_k <- 5L; cfg$n_jk <- 15L
  cfg
}

S02_mediano <- function() {
  cfg <- .cfg_base()   # ya es el regimen mediano (referencia)
  cfg
}

S03_grande <- function() {
  cfg <- .cfg_base()
  cfg$D <- 30L; cfg$M_k <- 15L; cfg$n_jk <- 150L
  cfg
}


# =============================================================================
# FAMILIA 2 — DESBALANCE POBLACIONAL  (S04, S05)
# =============================================================================

S04_desbalance_concentrado <- function() {
  cfg <- .cfg_base()
  cfg$imbalance_dept <- "concentrated"
  cfg$imbalance_muni <- "concentrated"
  cfg
}

S05_desbalance_extremo <- function() {
  cfg <- .cfg_base()
  cfg$imbalance_dept <- "extreme"
  cfg$imbalance_muni <- "extreme"
  cfg
}


# =============================================================================
# FAMILIA 3 — NIVELES DE VARIANZA  (S06, S07, S08)
# =============================================================================

S06_varianza_baja <- function() {
  cfg <- .cfg_base()
  cfg$alpha_kappa <- 4; cfg$beta_kappa <- 16
  cfg$nu_kappa <- 8          
  cfg
}

S07_varianza_alta_municipal <- function() {
  cfg <- .cfg_base()
  cfg$alpha_kappa <- 8; cfg$beta_kappa <- 8   
  cfg$nu_kappa <- 4                 
  cfg
}

S08_varianza_alta_departamental <- function() {
  cfg <- .cfg_base()
  cfg$alpha_kappa <- 2; cfg$beta_kappa <- 2  
  cfg$nu_kappa <- 8                 
  cfg
}

# =============================================================================
# FAMILIA 4 — COLINEALIDAD  (S09, S10)
# =============================================================================

S09_colinealidad_moderada <- function() {
  cfg <- .cfg_base()
  cfg$rho_X <- 0.7
  cfg
}

S10_colinealidad_severa <- function() {
  cfg <- .cfg_base()
  cfg$rho_X <- 0.95
  cfg
}


# =============================================================================
# FAMILIA 5 — CONTAMINACION POR ATIPICOS  (S11-S16)
# =============================================================================

.cfg_contam <- function(tipo, frac) {
  cfg <- .cfg_base()
  cfg$contam_type <- tipo
  cfg$contam_frac <- frac
  cfg$contam_magnitude <- 8.0
  cfg
}

S11_contam_ysym_05  <- function() .cfg_contam("y_sym",   0.05)
S12_contam_ysym_10  <- function() .cfg_contam("y_sym",   0.10)
S13_contam_ysym_20  <- function() .cfg_contam("y_sym",   0.20)
S14_contam_yasym_10 <- function() .cfg_contam("y_asym",  0.10)
S15_contam_context_10 <- function() .cfg_contam("context", 0.10)
S16_contam_covariable_10 <- function() {
  cfg <- .cfg_contam("X", 0.10)
  cfg$contam_col_X <- 1L   # perturbar la primera covariable continua
  cfg
}


# =============================================================================
# FAMILIA 6 — COLAS PESADAS  (S17, S18, S19)
# =============================================================================

S17_colas_t_nu3 <- function() {
  cfg <- .cfg_base()
  cfg$error_family <- "t"; cfg$nu_error <- 3
  cfg
}

S18_colas_t_nu5 <- function() {
  cfg <- .cfg_base()
  cfg$error_family <- "t"; cfg$nu_error <- 5
  cfg
}

S19_colas_mezcla <- function() {
  cfg <- .cfg_base()
  cfg$error_family <- "mixture"
  cfg$mixture_w <- 0.10       # 10% en la componente minoritaria
  cfg$mixture_scale <- 5.0    # varianza 5x mayor
  cfg
}


# =============================================================================
# FAMILIA 7 — MALA ESPECIFICACION ESTRUCTURAL  (S20)
# =============================================================================

S20_mala_espec_estructural <- function() {
  cfg <- .cfg_base()
  cfg$random_intercepts_in_truth <- TRUE
  cfg$sigma_u <- 1.0    
  cfg$sigma_v <- 1.0    
  cfg
}


# =============================================================================
# FAMILIA 8 — DINAMICOS: REGIMEN TEMPORAL  (D01-D04)
# =============================================================================
.cfg_base_dyn <- function() {
  cfg <- .cfg_base()
  cfg$n_periodos <- 5L      
  cfg
}


D01_dyn_estatico <- function() {
  cfg <- .cfg_base_dyn()
  cfg$dynamics <- "static"   
  cfg
}

D02_dyn_rw_lento <- function() {
  cfg <- .cfg_base_dyn()
  cfg$dynamics <- "rw"; cfg$sigma_omega <- 0.1   
  cfg
}

D03_dyn_rw_rapido <- function() {
  cfg <- .cfg_base_dyn()
  cfg$dynamics <- "rw"; cfg$sigma_omega <- 0.5   
  cfg
}

D04_dyn_ruptura <- function() {
  cfg <- .cfg_base_dyn()
  cfg$dynamics <- "jump"
  cfg$jump_time <- NULL        
  cfg$jump_magnitude <- 2.0    
  cfg
}

# =============================================================================
# FAMILIA 9 — DINAMICOS + ATIPICOS / COLAS PESADAS  (D05-D06)
# =============================================================================

D05_dyn_contaminacion <- function() {
  cfg <- .cfg_base_dyn()
  cfg$dynamics <- "rw"; cfg$sigma_omega <- 0.3
  cfg$error_family <- "normal"
  cfg$contam_type <- "y_sym"; cfg$contam_frac <- 0.10; cfg$contam_magnitude <- 8.0
  cfg
}

D06_dyn_colas_t <- function() {
  cfg <- .cfg_base_dyn()
  cfg$dynamics <- "rw"; cfg$sigma_omega <- 0.3
  cfg$error_family <- "t"; cfg$nu_error <- 4
  cfg
}


# =============================================================================
# FAMILIA 10 — DINAMICOS: VARIANZAS DINAMICAS  (D07-D08)
# =============================================================================

D07_var_dinamica_betagamma <- function() {
  cfg <- .cfg_base_dyn()
  cfg$dynamics <- "rw"; cfg$sigma_omega <- 0.1     # coeficientes RW lenta
  cfg$sigma2_dynamics <- "beta_gamma"; cfg$beta_k_discount <- 0.95
  cfg$error_family <- "normal"
  cfg
}

D08_var_dinamica_rwlog <- function() {
  cfg <- .cfg_base_dyn()
  cfg$dynamics <- "rw"; cfg$sigma_omega <- 0.1     # coeficientes RW lenta
  cfg$sigma2_dynamics <- "rw_log"; cfg$sigma_logs <- 0.3
  cfg$error_family <- "normal"
  cfg
}


# =============================================================================
# FAMILIA 11 — ESCENARIOS CULMINANTES  (S21 shrinkage extremo, D09 ruptura compuesta)
# =============================================================================

S21_shrinkage_extremo <- function() {
  cfg <- .cfg_base()
  cfg$D <- 5L          # pocos departamentos: el pooling departamental es el foco
  cfg$M_k <- 40L       # muchos municipios por departamento
  cfg$n_jk <- 3L       # apenas 3 individuos por municipio (informacion minima)
  cfg
}

D09_ruptura_compuesta <- function() {
  cfg <- .cfg_base_dyn()
  cfg$dynamics <- "jump"                              # ruptura estructural
  cfg$jump_time <- NULL                               # mitad del horizonte
  cfg$jump_magnitude <- 2.0
  cfg$error_family <- "t"; cfg$nu_error <- 4          # colas pesadas
  cfg$contam_type <- "y_sym"                          # + contaminacion
  cfg$contam_frac <- 0.10; cfg$contam_magnitude <- 8.0
  cfg
}


# =============================================================================
# REGISTRO DE ESCENARIOS DISPONIBLES
# =============================================================================
CATALOGO_ESCENARIOS <- list(
  # --- Familia 1: tamano muestral ---
  S01 = list(fn = S01_pequeno, modelos = c("M1","M2"),
             desc = "Tamano pequeno (D=5,M_k=5,n=15), estatico normal"),
  S02 = list(fn = S02_mediano, modelos = c("M1","M2"),
             desc = "Tamano mediano/referencia (D=15,M_k=10,n=50), estatico normal"),
  S03 = list(fn = S03_grande,  modelos = c("M1","M2"),
             desc = "Tamano grande (D=30,M_k=15,n=150), estatico normal"),
  # --- Familia 2: desbalance poblacional (S02 = balanceado de referencia) ---
  S04 = list(fn = S04_desbalance_concentrado, modelos = c("M1","M2"),
             desc = "Desbalance concentrado (pesos Gamma), estatico normal"),
  S05 = list(fn = S05_desbalance_extremo, modelos = c("M1","M2"),
             desc = "Desbalance extremo (~10% acapara ~70%), estatico normal"),
  # --- Familia 3: niveles de varianza ---
  S06 = list(fn = S06_varianza_baja, modelos = c("M1","M2"),
             desc = "Varianza residual baja (E[kappa2_k]=0.25), estatico normal"),
  S07 = list(fn = S07_varianza_alta_municipal, modelos = c("M1","M2"),
             desc = "Alta varianza entre municipios (nu_kappa=4), estatico normal"),
  S08 = list(fn = S08_varianza_alta_departamental, modelos = c("M1","M2"),
             desc = "Alta varianza entre departamentos (alpha=beta=2), estatico normal"),
  # --- Familia 4: colinealidad (S02 = bajo rho_X=0.3 de referencia) ---
  S09 = list(fn = S09_colinealidad_moderada, modelos = c("M1","M2"),
             desc = "Colinealidad moderada (rho_X=0.7), estatico normal"),
  S10 = list(fn = S10_colinealidad_severa, modelos = c("M1","M2"),
             desc = "Colinealidad severa (rho_X=0.95), estatico normal"),
  # --- Familia 5: contaminacion por atipicos (M1 vs M2) ---
  S11 = list(fn = S11_contam_ysym_05, modelos = c("M1","M2"),
             desc = "Contam. respuesta simetrica 5%, estatico"),
  S12 = list(fn = S12_contam_ysym_10, modelos = c("M1","M2"),
             desc = "Contam. respuesta simetrica 10%, estatico"),
  S13 = list(fn = S13_contam_ysym_20, modelos = c("M1","M2"),
             desc = "Contam. respuesta simetrica 20%, estatico"),
  S14 = list(fn = S14_contam_yasym_10, modelos = c("M1","M2"),
             desc = "Contam. respuesta asimetrica 10%, estatico"),
  S15 = list(fn = S15_contam_context_10, modelos = c("M1","M2"),
             desc = "Contam. contextual (municipios) 10%, estatico"),
  S16 = list(fn = S16_contam_covariable_10, modelos = c("M1","M2"),
             desc = "Contam. covariable 10%, estatico"),
  # --- Familia 6: colas pesadas (nu tiene valor verdadero en t) ---
  S17 = list(fn = S17_colas_t_nu3, modelos = c("M1","M2"),
             desc = "Colas t nu=3 (kurtosis alta), estatico"),
  S18 = list(fn = S18_colas_t_nu5, modelos = c("M1","M2"),
             desc = "Colas t nu=5 (colas moderadas), estatico"),
  S19 = list(fn = S19_colas_mezcla, modelos = c("M1","M2"),
             desc = "Mezcla normales 90/10 var 5x, estatico"),
  # --- Familia 7: mala especificacion estructural ---
  S20 = list(fn = S20_mala_espec_estructural, modelos = c("M1","M2"),
             desc = "Interceptos aleatorios en la media (sigma=1), estatico"),
  # --- Familia 8: dinamicos regimen temporal (LOS 4 MODELOS) ---
  D01 = list(fn = D01_dyn_estatico, modelos = c("M1","M2","M3","M4"),
             desc = "Dinamico: coeficientes estaticos (costo sobreparametrizacion)"),
  D02 = list(fn = D02_dyn_rw_lento, modelos = c("M1","M2","M3","M4"),
             desc = "Dinamico: RW lenta (sigma_omega=0.1)"),
  D03 = list(fn = D03_dyn_rw_rapido, modelos = c("M1","M2","M3","M4"),
             desc = "Dinamico: RW rapida (sigma_omega=0.5)"),
  D04 = list(fn = D04_dyn_ruptura, modelos = c("M1","M2","M3","M4"),
             desc = "Dinamico: ruptura estructural (jump a mitad de horizonte)"),
  # --- Familia 9: dinamicos + atipicos / colas (LOS 4 MODELOS) ---
  D05 = list(fn = D05_dyn_contaminacion, modelos = c("M1","M2","M3","M4"),
             desc = "Dinamico RW + contaminacion y_sym 10%"),
  D06 = list(fn = D06_dyn_colas_t, modelos = c("M1","M2","M3","M4"),
             desc = "Dinamico RW + colas t (nu=4)"),
  # --- Familia 10: varianzas dinamicas (LOS 4 MODELOS) ---
  D07 = list(fn = D07_var_dinamica_betagamma, modelos = c("M1","M2","M3","M4"),
             desc = "Varianza dinamica beta_gamma + coef RW lenta"),
  D08 = list(fn = D08_var_dinamica_rwlog, modelos = c("M1","M2","M3","M4"),
             desc = "Varianza dinamica rw_log + coef RW lenta"),
  # --- Familia 11: escenarios culminantes ---
  S21 = list(fn = S21_shrinkage_extremo, modelos = c("M1","M2"),
             desc = "Shrinkage extremo (40 muni x 3 indiv), estatico"),
  D09 = list(fn = D09_ruptura_compuesta, modelos = c("M1","M2","M3","M4"),
             desc = "Ruptura compuesta: jump + contaminacion + colas t (prueba de estres)")
)


listar_escenarios <- function() {
  cat(sprintf("Catalogo: %d escenarios disponibles\n", length(CATALOGO_ESCENARIOS)))
  for (id in names(CATALOGO_ESCENARIOS)) {
    e <- CATALOGO_ESCENARIOS[[id]]
    cat(sprintf("  %s: %s [modelos: %s]\n", id, e$desc, paste(e$modelos, collapse=",")))
  }
  invisible(names(CATALOGO_ESCENARIOS))
}
