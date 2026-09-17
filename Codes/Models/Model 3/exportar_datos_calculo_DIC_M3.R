# -----------------------------------------------------------------------------
# 1. CONFIGURACION
# -----------------------------------------------------------------------------
DIR_MCMC <- file.path(
  "C:/Users/USUARIO/OneDrive - Universidad Nacional de Colombia/Tesis",
  "Escritorio/Códigos/Modelos/Modelo 3/mcmc_salida_m3"
)
DIR_SALIDA <- file.path(DIR_MCMC, "datos_DIC")

# -----------------------------------------------------------------------------
# 2. Localizacion de los objetos del muestreador
# -----------------------------------------------------------------------------
if (!exists("Y") || !exists("F_lst") || !exists("N_jkt")) {
  if (nzchar(RUTA_DATOS_MODELO) && file.exists(RUTA_DATOS_MODELO)) {
    cat("Cargando", RUTA_DATOS_MODELO, "\n")
    .dm <- readRDS(RUTA_DATOS_MODELO)
    .necesarios <- c("Y", "F_lst", "N_jkt", "n_k", "m", "Tt", "p",
                     "theta_labels", "depto_ids", "tiempo_ids", "y_vec")
    list2env(.dm[intersect(names(.dm), .necesarios)], envir = globalenv())
    rm(.dm, .necesarios)
  } else {
    stop("No estan en memoria Y / F_lst / N_jkt y RUTA_DATOS_MODELO no apunta ",
         "a un archivo valido. Vuelva a correr la preparacion del M3 hasta la ",
         "seccion 10, o descomente el saveRDS de datos_modelo.rds.")
  }
}

stopifnot(exists("Y"), exists("F_lst"), exists("N_jkt"), exists("theta_labels"))

if (!exists("Tt")) Tt <- dim(N_jkt)[3]
if (!exists("m"))  m  <- dim(N_jkt)[2]
if (!exists("n_k"))
  n_k <- apply(N_jkt, 2, function(A) max(which(rowSums(A) > 0)))
n_k <- as.integer(n_k)
etiquetas <- as.character(theta_labels)
p <- length(etiquetas)

cat(sprintf("Estructuras del muestreador: m = %d, Tt = %d, p = %d, municipios = %d\n",
            m, Tt, p, sum(n_k)))

# -----------------------------------------------------------------------------
# 3. Cotejo con los metadatos de las cadenas (control de consistencia)
# -----------------------------------------------------------------------------
ruta_meta <- file.path(DIR_MCMC, "metadatos.rds")
if (file.exists(ruta_meta)) {
  meta_mcmc <- readRDS(ruta_meta)
  if (length(meta_mcmc$theta_labels) != p)
    stop("p de las cadenas (", length(meta_mcmc$theta_labels),
         ") difiere de p de la sesion (", p, ").")
  if (!identical(as.integer(meta_mcmc$n_k), n_k))
    warning("n_k de las cadenas y de la sesion no coinciden exactamente.")
  if (!identical(as.character(meta_mcmc$theta_labels), etiquetas))
    warning("Los nombres de theta difieren entre cadenas y sesion; se usa el ",
            "orden de la sesion, que es el de F_lst.")
  cat("Cotejo con metadatos.rds: correcto\n")
} else {
  cat("Aviso: no se encontro metadatos.rds en DIR_MCMC; se omite el cotejo.\n")
}

# -----------------------------------------------------------------------------
# 4. Escritura incremental recorriendo (t, k, j)
# -----------------------------------------------------------------------------
dir.create(DIR_SALIDA, recursive = TRUE, showWarnings = FALSE)

N_esperado <- sum(N_jkt)
cat(sprintf("Observaciones esperadas: %s | tamano de datos_X.bin: %.0f MB\n",
            format(N_esperado, big.mark = ","), N_esperado * p * 8 / 1024^2))

con_y <- file(file.path(DIR_SALIDA, "datos_y.bin"),          open = "wb")
con_X <- file(file.path(DIR_SALIDA, "datos_X.bin"),          open = "wb")
con_k <- file(file.path(DIR_SALIDA, "datos_idx_depto.bin"),  open = "wb")
con_j <- file(file.path(DIR_SALIDA, "datos_idx_muni.bin"),   open = "wb")
con_t <- file(file.path(DIR_SALIDA, "datos_idx_tiempo.bin"), open = "wb")

orden_cols <- NULL
N_escritas <- 0L
suma_y     <- 0
celdas     <- 0L

for (tt in 1:Tt) {
  for (kk in 1:m) {
    for (jj in 1:n_k[kk]) {

      y_blk <- Y[[tt]][[kk]][[jj]]
      if (is.null(y_blk) || length(y_blk) == 0L) next
      F_blk <- F_lst[[tt]][[kk]][[jj]]

      if (is.null(orden_cols)) {
        if (is.null(colnames(F_blk))) {
          if (ncol(F_blk) != p)
            stop("F_lst no tiene nombres de columna y ncol difiere de p.")
          orden_cols <- seq_len(p)
          cat("Aviso: F_lst sin nombres; se asume el orden de theta_labels.\n")
        } else {
          orden_cols <- match(etiquetas, colnames(F_blk))
          if (anyNA(orden_cols))
            stop("Hay etiquetas de theta ausentes en F_lst: ",
                 paste(head(setdiff(etiquetas, colnames(F_blk)), 5),
                       collapse = ", "))
        }
      }

      if (nrow(F_blk) != length(y_blk))
        stop(sprintf("Celda (t=%d, k=%d, j=%d): nrow(F) = %d y length(y) = %d",
                     tt, kk, jj, nrow(F_blk), length(y_blk)))
      if (N_jkt[jj, kk, tt] != length(y_blk))
        stop(sprintf("Celda (t=%d, k=%d, j=%d): N_jkt = %d y length(y) = %d",
                     tt, kk, jj, N_jkt[jj, kk, tt], length(y_blk)))

      n_b <- length(y_blk)
      # t(F) recorrido por columnas equivale a F por filas: orden "filas".
      writeBin(as.double(y_blk), con_y, size = 8)
      writeBin(as.double(t(F_blk[, orden_cols, drop = FALSE])), con_X, size = 8)
      writeBin(rep.int(as.integer(kk), n_b), con_k, size = 4)
      writeBin(rep.int(as.integer(jj), n_b), con_j, size = 4)
      writeBin(rep.int(as.integer(tt), n_b), con_t, size = 4)

      N_escritas <- N_escritas + n_b
      suma_y     <- suma_y + sum(y_blk)
      celdas     <- celdas + 1L
    }
  }
  cat(sprintf("  periodo %2d/%d — acumuladas %s observaciones\n",
              tt, Tt, format(N_escritas, big.mark = ",")))
  flush.console()
}

close(con_y); close(con_X); close(con_k); close(con_j); close(con_t)

# -----------------------------------------------------------------------------
# 5. Metadatos y verificaciones finales
# -----------------------------------------------------------------------------
json <- sprintf('{"N":%d,"p":%d,"orden_X":"filas","columnas_X":null}',
                N_escritas, p)
con_json <- file(file.path(DIR_SALIDA, "datos_meta.json"), open = "wb")
writeBin(charToRaw(json), con_json)
close(con_json)

if (N_escritas != N_esperado)
  stop("Se escribieron ", N_escritas, " observaciones y N_jkt suma ", N_esperado)

tam <- function(f) file.size(file.path(DIR_SALIDA, f))
stopifnot(tam("datos_y.bin")          == N_escritas * 8,
          tam("datos_X.bin")          == N_escritas * p * 8,
          tam("datos_idx_depto.bin")  == N_escritas * 4,
          tam("datos_idx_muni.bin")   == N_escritas * 4,
          tam("datos_idx_tiempo.bin") == N_escritas * 4)

if (exists("y_vec")) {
  if (length(y_vec) != N_escritas)
    warning("length(y_vec) = ", length(y_vec), " y N escritas = ", N_escritas)
  if (abs(sum(y_vec) - suma_y) > 1e-6 * abs(suma_y))
    warning("La suma de y exportada no coincide con sum(y_vec).")
  else
    cat("Cotejo con y_vec: suma de puntajes identica\n")
}

cat(sprintf("\nExportacion completa en:\n  %s\n", DIR_SALIDA))
cat(sprintf("N = %s | p = %d | celdas (municipio x periodo) = %s | media de y = %.3f\n",
            format(N_escritas, big.mark = ","), p,
            format(celdas, big.mark = ","), suma_y / N_escritas))

cat("\nPaso siguiente, en la consola del sistema:\n")
cat(sprintf('  python dic_M3.py --mcmc "%s" --datos-bin "%s" --salida "%s"\n',
            DIR_MCMC, DIR_SALIDA, file.path(DIR_MCMC, "dic")))
cat("  (agregue --max-draws 200 para un ensayo rapido antes de la corrida completa)\n")

# =============================================================================
