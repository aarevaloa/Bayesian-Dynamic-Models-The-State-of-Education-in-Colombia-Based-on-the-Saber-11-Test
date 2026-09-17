#_______________________________________________________________________________
#@  VERIFICADOR DE ENTORNO
#_______________________________________________________________________________
#
#_______________________________________________________________________________

cat("\n===== VERIFICACION DE ENTORNO =====\n\n")
ok_total <- TRUE
marca <- function(ok, msg) {
  cat(sprintf("  [%s] %s\n", if (ok) "OK " else "FALTA", msg))
  if (!ok) ok_total <<- FALSE
}

# (1) Version de R
v <- getRversion()
marca(v >= "4.0.0", sprintf("R version %s (se requiere >= 4.0)", v))

# (2) Paquete statmod
tiene_statmod <- requireNamespace("statmod", quietly = TRUE)
marca(tiene_statmod, "paquete 'statmod' instalado")
if (!tiene_statmod)
  cat("       -> instalalo con:  install.packages('statmod')\n")

# (3) config_rutas.R existe y se puede cargar
if (!file.exists("0_config_rutas.R")) {
  marca(FALSE, "0_config_rutas.R en el directorio actual")
  cat("       -> fija el directorio:  setwd('C:/.../Simulaciones')\n")
  cat("\n  No puedo continuar sin 0_config_rutas.R. Deten aqui.\n")
} else {
  source("0_config_rutas.R")
  marca(TRUE, "0_config_rutas.R cargado")

  # (4) Rutas de carpetas
  for (nm in c("RUTA_SIMULACION","RUTA_M1","RUTA_M2","RUTA_M3","RUTA_M4")) {
    ruta <- get(nm)
    marca(dir.exists(ruta), sprintf("%s -> %s", nm, ruta))
  }

  # (5) Archivos del motor en RUTA_SIMULACION
  cat("\n  -- Archivos del motor (en RUTA_SIMULACION) --\n")
  for (f in c("2_generador_datos.R","adaptador_M1.R","adaptador_M2.R",
              "adaptador_M3.R","adaptador_M4.R","3_motor_A_runners.R",
              "3_motor_B_metricas.R","3_motor_C_orquestador.R","3_motor.R")) {
    marca(file.exists(file.path(RUTA_SIMULACION, f)), f)
  }

  # (6) Archivos de cada modelo
  cat("\n  -- Archivos del Modelo 1 --\n")
  for (f in ARCHIVOS_M1) marca(file.exists(file.path(RUTA_M1, f)), f)

  cat("\n  -- Archivos del Modelo 2 --\n")
  for (f in ARCHIVOS_M2) marca(file.exists(file.path(RUTA_M2, f)), f)

  cat("\n  -- Archivos del Modelo 3 --\n")
  for (f in ARCHIVOS_M3_FUNCIONES) marca(file.exists(file.path(RUTA_M3, f)), f)
  marca(file.exists(file.path(RUTA_M3, ARCHIVO_M3_GIBBS_ORIGINAL)),
        sprintf("%s (Gibbs original)", ARCHIVO_M3_GIBBS_ORIGINAL))
  ok_patch <- file.exists(file.path(RUTA_M3, ARCHIVO_M3_GIBBS_PATCHED))
  marca(ok_patch, sprintf("%s (Gibbs PARCHEADO)", ARCHIVO_M3_GIBBS_PATCHED))
  if (!ok_patch)
    cat("       -> crealo siguiendo el PASO 3 del instructivo\n")

  cat("\n  -- Archivos del Modelo 4 (versiones _FUN) --\n")
  for (f in ARCHIVOS_M4_FUN) {
    ok_f <- file.exists(file.path(RUTA_M4, f))
    marca(ok_f, f)
    if (!ok_f) cat("       -> crealo siguiendo el PASO 4 del instructivo\n")
  }
}

cat("\n===== RESULTADO =====\n")
if (ok_total) {
  cat("  TODO LISTO. Puedes correr:  source('correr_validacion.R')\n\n")
} else {
  cat("  Faltan elementos (ver arriba). Corrigelos antes de continuar.\n\n")
}
