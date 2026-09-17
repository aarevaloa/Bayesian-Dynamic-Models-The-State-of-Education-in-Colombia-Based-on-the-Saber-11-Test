#_______________________________________________________________________________
#@  MOTOR DE SIMULACION — CARGADOR MAESTRO
#@  Carga config + generador + adaptadores + las 3 partes del motor
#_______________________________________________________________________________
#_______________________________________________________________________________

# (1) Configuracion de rutas — DEBE existir y estar editado
if (!file.exists("0_config_rutas.R"))
  stop("No encuentro 0_config_rutas.R. Ejecuta R desde la carpeta RUTA_SIMULACION.")
source("0_config_rutas.R")
.validar_config_rutas()

# (2) statmod (necesario para M4)
if (!requireNamespace("statmod", quietly = TRUE))
  stop("Falta 'statmod'. Instalalo con: install.packages('statmod')")

# (3) Generador de datos + adaptadores (deben estar en RUTA_SIMULACION)
for (f in c("2_generador_datos.R", "adaptador_M1.R", "adaptador_M2.R",
            "adaptador_M3.R", "adaptador_M4.R")) {
  if (!file.exists(f)) stop(sprintf("Falta %s en %s", f, RUTA_SIMULACION))
  source(f)
}

# (4) Motor (3 partes)
source("3_motor_A_runners.R")
source("3_motor_B_metricas.R")
if (file.exists("waic.R")) source("waic.R")   # criterios de informacion (WAIC)
source("3_motor_C_orquestador.R")

cat("\n=== Motor cargado correctamente ===\n")
cat("Modelos disponibles:", paste(names(RUNNERS), collapse=", "), "\n")
