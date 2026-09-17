#_______________________________________________________________________________
#@  CORRER VALIDACION DEL MOTOR
#@  Ejecuta un escenario completo sobre los 4 modelos y reporta metricas
#_______________________________________________________________________________


# =============================================================================
#  CONFIGURACION — EDITA AQUI
# =============================================================================
ESCENARIO <- "PEQUENO"

# --- Que modelos correr ---
MODELOS <- c("M1", "M2", "M3", "M4")

# --- Numero de replicas Monte Carlo ---
R_REPLICAS <- 3L          

# --- Parametros MCMC ---
N_ITER  <- 2000L            
BURN_IN <- 1000L
THIN    <- 2L

# --- Semilla base (cada replica usa SEMILLA_BASE + r) ---
SEMILLA_BASE <- 2000L

# --- Paralelizacion: correr las R replicas en paralelo? ---
# TRUE acelera mucho con R grande (usa nucleos fisicos - 1). Requiere future/furrr.
# Si algo falla en paralelo, ponlo en FALSE para ver el error real en serie.
USAR_PARALELO <- FALSE        # <- cambia a TRUE para activar
N_WORKERS     <- NULL         # NULL = automatico (nucleos fisicos - 1)


# =============================================================================
#  DEFINICION DE LOS ESCENARIOS  (no editar salvo que quieras otra config)
# =============================================================================

construir_cfg <- function(escenario) {
  cfg <- default_config()
  if (escenario == "PEQUENO") {
    # Escenario chico dinamico con errores t — para verificar que todo corre.
    cfg$D <- 6L; cfg$M_k <- 5L; cfg$n_jk <- 50L
    cfg$n_periodos <- 5L; cfg$dynamics <- "rw"; cfg$sigma_omega <- 0.3
    cfg$p_E <- 4L; cfg$p_M <- 2L; cfg$p_D <- 1L; cfg$sparsity_E <- 0.5
    cfg$effect_type_E <- "mixed"; cfg$effect_type_M <- "mixed"; cfg$effect_type_D <- "strong"
    cfg$error_family <- "t"; cfg$nu_error <- 5
    cfg$alpha_kappa <- 8; cfg$beta_kappa <- 8
  } else if (escenario == "REFERENCIA") {
    # Escenario de referencia completo (grande) — dinamico con errores t.
    cfg$D <- 15L; cfg$M_k <- 10L; cfg$n_jk <- 50L
    cfg$n_periodos <- 5L; cfg$dynamics <- "rw"; cfg$sigma_omega <- 0.3
    cfg$p_E <- 5L; cfg$p_M <- 3L; cfg$p_D <- 2L; cfg$sparsity_E <- 0.4
    cfg$effect_type_E <- "mixed"; cfg$effect_type_M <- "mixed"; cfg$effect_type_D <- "strong"
    cfg$error_family <- "t"; cfg$nu_error <- 5
    cfg$alpha_kappa <- 8; cfg$beta_kappa <- 8
  } else stop("ESCENARIO debe ser 'PEQUENO' o 'REFERENCIA'")
  cfg
}


# =============================================================================
#  EJECUCION
# =============================================================================

source("3_motor.R")   # carga todo el motor

cfg <- construir_cfg(ESCENARIO)
N_total_aprox <- cfg$D * cfg$M_k * cfg$n_jk * cfg$n_periodos

cat(sprintf("\n========================================================\n"))
cat(sprintf(" VALIDACION DEL MOTOR — escenario %s\n", ESCENARIO))
cat(sprintf("========================================================\n"))
cat(sprintf("  Dimensiones: D=%d, M_k=%d, n_jk=%d, T=%d, p=%d\n",
            cfg$D, cfg$M_k, cfg$n_jk, cfg$n_periodos,
            1 + cfg$p_E + cfg$p_M + cfg$p_D))
cat(sprintf("  N total aprox: %d observaciones\n", N_total_aprox))
cat(sprintf("  Modelos: %s | R=%d | MCMC: %d iter, %d burn-in, thin=%d\n",
            paste(MODELOS, collapse=","), R_REPLICAS, N_ITER, BURN_IN, THIN))
cat(sprintf("  Errores: %s (nu=%s) | dinamica: %s\n\n",
            cfg$error_family, cfg$nu_error, cfg$dynamics))

t_inicio <- Sys.time()
if (USAR_PARALELO) {
  source("3_motor_D_paralelo.R")
  configurar_paralelo(n_workers = N_WORKERS)
  resultado <- correr_escenario_par(
    cfg, modelos = MODELOS, R = R_REPLICAS,
    control = list(n_iter = N_ITER, burn_in = BURN_IN, thin = THIN),
    id_escenario = ESCENARIO, semilla_base = SEMILLA_BASE,
    ref_estatico = "theta_bar",
    dir_trabajo = file.path(RUTA_TRABAJO, "par"),
    verbose = TRUE)
  configurar_serie()   # liberar workers al terminar
} else {
  resultado <- correr_escenario(
    cfg, modelos = MODELOS, R = R_REPLICAS,
    control = list(n_iter = N_ITER, burn_in = BURN_IN, thin = THIN),
    id_escenario = ESCENARIO, semilla_base = SEMILLA_BASE,
    ref_estatico = "theta_bar",
    dir_trabajo = RUTA_TRABAJO, verbose = TRUE)
}
dur <- as.numeric(difftime(Sys.time(), t_inicio, units = "mins"))

# Guardar resultado
archivo_salida <- file.path(RUTA_SIMULACION,
                            sprintf("resultado_%s.rds", tolower(ESCENARIO)))
saveRDS(resultado, archivo_salida)

cat(sprintf("\n>>> TIEMPO TOTAL: %.1f min\n", dur))
cat(sprintf(">>> Resultado guardado en: %s\n", archivo_salida))

# Reporte
imprimir_agregado(resultado)
