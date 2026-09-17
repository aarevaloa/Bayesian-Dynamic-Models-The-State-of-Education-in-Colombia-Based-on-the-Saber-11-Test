#_______________________________________________________________________________
#   EXPORTADOR DE DATOS OBSERVADOS PARA EL PIPELINE PYTHON — MODELO 3
#_______________________________________________________________________________

suppressMessages({
  if (!requireNamespace("jsonlite", quietly = TRUE))
    install.packages("jsonlite")
  library(jsonlite)
})

# Directorio de salida (junto a las muestras MCMC)
setwd('C:/Users/USUARIO/OneDrive - Universidad Nacional de Colombia/Tesis/Escritorio/Códigos/Modelos/Modelo 3')
dir_export <- file.path(getwd(), "1_Resultados_mcmc_output", "datos_export")
if (!dir.exists(dir_export)) dir.create(dir_export, recursive = TRUE)

# Si las estructuras no están en memoria, intentar cargarlas
if (!exists("Y") || !exists("F_lst") || !exists("N_jkt")) {
  cand <- file.path(getwd(), "datos_modelo.rds")
  if (file.exists(cand)) {
    cat("Cargando estructuras desde datos_modelo.rds...\n")
    dm <- readRDS(cand)
    list2env(dm, envir = .GlobalEnv)
    rm(dm)
  } else {
    stop("No hay Y/F_lst/N_jkt en memoria ni datos_modelo.rds. ",
         "Ejecute primero 1_Preparación_matrices_M3.R.")
  }
}

N_total <- sum(N_jkt)
cat(sprintf("Exportando %d observaciones (p = %d)...\n", N_total, p))

# ── Abrir conexiones binarias ──
con_y <- file(file.path(dir_export, "y_obs.bin"), "wb")
con_F <- file(file.path(dir_export, "F_obs.bin"), "wb")

# ── Índice de celdas: para cada (t,k,j) ocupada, guardar offset y n ──
# Esto permite a Python reconstruir k_all/t_all sin recorrer las listas.
celdas <- list()
pos <- 0L   # offset acumulado en número de observaciones

for (tt in 1:Tt) {
  for (kk in 1:m) {
    for (jj in 1:n_k[kk]) {
      nn <- N_jkt[jj, kk, tt]
      if (nn == 0L) next
      y_obs <- Y[[tt]][[kk]][[jj]]
      F_jkt <- F_lst[[tt]][[kk]][[jj]]

      # y: escribir directamente
      writeBin(as.double(y_obs), con_y)

      # F: escribir fila por fila (row-major) para que Python lo lea como
      # matriz (nn x p) con np.frombuffer(...).reshape(nn, p)
      # t(F_jkt) apila por columnas = filas originales consecutivas.
      writeBin(as.double(as.vector(t(F_jkt))), con_F)

      celdas[[length(celdas) + 1L]] <- list(
        t = tt, k = kk, j = jj, n = nn, offset = pos
      )
      pos <- pos + nn
    }
  }
}

close(con_y); close(con_F)

# ── Metadatos JSON ──
meta_export <- list(
  N_total      = N_total,
  p            = p,
  m            = m,
  Tt           = Tt,
  n_k          = as.integer(n_k),
  theta_labels = theta_labels,
  depto_ids    = as.character(depto_ids),
  tiempo_ids   = as.character(tiempo_ids),
  # celdas como columnas paralelas (más compacto que lista de objetos)
  cell_t       = vapply(celdas, function(c) c$t, integer(1)),
  cell_k       = vapply(celdas, function(c) c$k, integer(1)),
  cell_j       = vapply(celdas, function(c) c$j, integer(1)),
  cell_n       = vapply(celdas, function(c) c$n, integer(1)),
  cell_offset  = vapply(celdas, function(c) c$offset, integer(1)),
  F_layout     = "row_major_per_cell",
  descripcion  = "y_obs.bin y F_obs.bin en orden t->k->j; F por filas (nn x p)"
)

write_json(meta_export,
           file.path(dir_export, "export_meta.json"),
           auto_unbox = TRUE, digits = 12)

# Verificación de tamaños
bytes_y <- file.size(file.path(dir_export, "y_obs.bin"))
bytes_F <- file.size(file.path(dir_export, "F_obs.bin"))
cat(sprintf("  y_obs.bin: %.1f MB (esperado %.1f)\n",
            bytes_y / 1e6, N_total * 8 / 1e6))
cat(sprintf("  F_obs.bin: %.1f MB (esperado %.1f)\n",
            bytes_F / 1e6, N_total * p * 8 / 1e6))
cat(sprintf("  Celdas ocupadas exportadas: %d\n", length(celdas)))
cat(sprintf("✓ Datos exportados en %s/\n", basename(dir_export)))

jsonlite::write_json(
  readRDS("C:/Users/USUARIO/OneDrive - Universidad Nacional de Colombia/Tesis/Escritorio/Códigos/Modelos/Modelo 3/1_Resultados_mcmc_output/metadatos.rds"),
  "C:/Users/USUARIO/OneDrive - Universidad Nacional de Colombia/Tesis/Escritorio/Códigos/Modelos/Modelo 3/1_Resultados_mcmc_output/metadatos.json",
  auto_unbox = TRUE, digits = 12
)