# =============================================================================
#  Script maestro de la corrida definitiva
# =============================================================================
#  @author: aarevaloa
#  Ejecuta el estudio completo (30 escenarios) y WAIC.
# =============================================================================

setwd("C:/Users/USUARIO/OneDrive - Universidad Nacional de Colombia/Tesis/Escritorio/Códigos/Simulaciones")
source("3_motor.R")
source("0_catalogo_escenarios.R")
source("3_motor_E_estudio.R")
source("3_motor_D_paralelo.R")
source("waic.R")
configurar_paralelo()

verificacion_de_humo <- function() {
  cat("\n========== VERIFICACION DE HUMO ==========\n")
  ok <- TRUE

  # (a) funciones de cada modelo
  cat("1) Cargando funciones de los 4 modelos...\n")
  tryCatch(cargar_funciones_M1(), error = function(e) NULL)
  tryCatch(cargar_funciones_M2(), error = function(e) NULL)
  tryCatch(cargar_funciones_M3(), error = function(e) NULL)
  tryCatch(cargar_funciones_M4(), error = function(e) NULL)
  for (fn in c("gibbs_modelo4")) { 
    existe <- exists(fn, mode = "function")
    cat(sprintf("   %s: %s\n", fn, if (existe) "OK" else "FALTA <<<"))
    if (!existe) ok <- FALSE
  }

  cat("2) Verificando que el muestreador de M4 es la version con WAIC...\n")
  src_m4 <- tryCatch(deparse(gibbs_modelo4), error = function(e) "")
  tiene_vsave <- any(grepl("v_save", src_m4))
  cat(sprintf("   M4 guarda cadenas (v_save): %s\n", if (tiene_vsave) "OK" else "FALTA <<<"))
  if (!tiene_vsave) ok <- FALSE

  cat("3) Verificando modulo WAIC...\n")
  tiene_waic <- exists("waic_modelo", mode = "function")
  cat(sprintf("   waic_modelo: %s\n", if (tiene_waic) "OK" else "FALTA <<<"))
  if (!tiene_waic) ok <- FALSE

  cat("4) Corrida mini de un escenario dinamico (4 modelos, R=2)...\n")
  res_mini <- tryCatch({
    correr_estudio(ids = "D02", R = 2L,
                   control = list(n_iter = 60L, burn_in = 15L, thin = 2L,
                                  calcular_waic = TRUE),
                   usar_paralelo = FALSE, forzar = TRUE)
    rds <- readRDS(file.path(.dir_estudio(), "escenario_D02.rds"))
    rds
  }, error = function(e) { cat("   ERROR:", conditionMessage(e), "\n"); NULL })

  if (!is.null(res_mini)) {
    modelos_ok <- names(res_mini$agregado)
    cat(sprintf("   Modelos que completaron: %s\n", paste(modelos_ok, collapse = ", ")))
    m4_ok <- "M4" %in% modelos_ok && !is.null(res_mini$agregado$M4$waic)
    cat(sprintf("   M4 corrio y tiene WAIC: %s\n", if (m4_ok) "OK" else "FALTA <<<"))
    if (!m4_ok) ok <- FALSE
    
    unlink(file.path(.dir_estudio(), "escenario_D02.rds"))
  } else ok <- FALSE

  cat("==========================================\n")
  if (ok) {
    cat(">>> HUMO OK: todo en su lugar. Puedes lanzar la corrida completa.\n\n")
  } else {
    cat(">>> HUMO FALLO: revisa los <<< arriba ANTES de lanzar la corrida.\n")
    cat(">>> (causa tipica: el muestreador de M4 no esta con el nombre\n")
    cat(">>>  '8_Muestreador_GIBBS_v2_FUN_SIM.R' o no es la version con WAIC.)\n\n")
  }
  invisible(ok)
}

humo_ok <- verificacion_de_humo()
stopifnot("La verificacion de humo fallo; revisa los mensajes." = humo_ok)



CTRL_TESIS <- list(n_iter = 10000L, burn_in = 2000L, thin = 5L, calcular_waic = TRUE)
R_TESIS <- 30L


# =============================================================================
# [1] ESCENARIOS ESTATICOS (S01-S21) —  M1 vs M2
# =============================================================================

correr_estudio(ids = c("S01","S02","S03"),            R = R_TESIS, control = CTRL_TESIS, usar_paralelo = TRUE, forzar = FALSE)
correr_estudio(ids = c("S04","S05"),                  R = R_TESIS, control = CTRL_TESIS, usar_paralelo = TRUE, forzar = TRUE)
correr_estudio(ids = c("S06","S07","S08"),            R = R_TESIS, control = CTRL_TESIS, usar_paralelo = TRUE, forzar = TRUE)
correr_estudio(ids = c("S09","S10"),                  R = R_TESIS, control = CTRL_TESIS, usar_paralelo = TRUE, forzar = TRUE)
correr_estudio(ids = c("S11","S12","S13","S14","S15","S16"), R = R_TESIS, control = CTRL_TESIS, usar_paralelo = TRUE, forzar = TRUE)
correr_estudio(ids = c("S17","S18","S19"),            R = R_TESIS, control = CTRL_TESIS, usar_paralelo = TRUE, forzar = TRUE)
correr_estudio(ids = c("S20"),                        R = R_TESIS, control = CTRL_TESIS, usar_paralelo = TRUE, forzar = TRUE)
correr_estudio(ids = c("S21"),                        R = R_TESIS, control = CTRL_TESIS, usar_paralelo = TRUE, forzar = TRUE)


# =============================================================================
# [2] ESCENARIOS DINAMICOS (D01-D09) — los 4 modelos
# =============================================================================

correr_estudio(ids = c("D01","D02","D03","D04"), R = R_TESIS, control = CTRL_TESIS, usar_paralelo = TRUE, forzar = TRUE)
correr_estudio(ids = c("D05","D06"),             R = R_TESIS, control = CTRL_TESIS, usar_paralelo = TRUE, forzar = TRUE)
correr_estudio(ids = c("D07","D08"),             R = R_TESIS, control = CTRL_TESIS, usar_paralelo = TRUE, forzar = TRUE)
correr_estudio(ids = c("D09"),                   R = R_TESIS, control = CTRL_TESIS, usar_paralelo = TRUE, forzar = TRUE)


# =============================================================================
# [3] RESULTADOS: tabla, WAIC, validacion
# =============================================================================
estado_estudio()
tabla <- generar_tabla_resumen()

validar_waic_seleccion()
resumen_seleccion_waic(tabla)
sel <- tabla_seleccion_waic(tabla)

print(tabla[tabla$escenario %in% c("S21","D09"),
            c("escenario","modelo","n_rep_ok","rmse","cobertura95",
              "lasso_sens","lasso_espec","waic","waic_lppd","waic_pwaic")])


# =============================================================================
# [4] FIGURAS — todo el set a disco
# =============================================================================
source("graficos.R")
g <- cargar_resultados()

fig_var_ratio(g, "contaminacion")                                
fig_cobertura(g, "colas")                                        
fig_rmse(g, "tamano")                                            
fig_brecha_robustez(g, "var_ratio_post_true", par = "dinamico") 
fig_cuatro_modelos_dinamicos(g, "rmse")                         
fig_brecha_dinamica(g, "rmse", verosimilitud = "t")            
fig_trayectoria_coef(g, "D04")                                   
fig_recuperacion_nu(g)                                           


generar_todas_las_figuras(g)
