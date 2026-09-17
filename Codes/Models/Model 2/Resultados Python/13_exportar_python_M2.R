# ==============================================================================
# @  MODELO 2 - t-STUDENT JERARQUICO ESTATICO CON LASSO BAYESIANO
# EXPORTADOR A PYTHON (export_python/)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. AUXILIARES
# ------------------------------------------------------------------------------
.escribir_bin <- function(x, ruta) {
  con <- file(ruta, open = "wb")
  on.exit(close(con), add = TRUE)
  writeBin(as.double(x), con, size = 8, endian = "little")
  invisible(ruta)
}


#' Escribir lineas de texto en UTF-8 (una por elemento).
.escribir_txt <- function(x, ruta) {
  con <- file(ruta, open = "w", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  writeLines(as.character(x), con)
  invisible(ruta)
}


#' Escribir un data.frame como CSV UTF-8 sin nombres de fila.
.escribir_csv <- function(df, ruta) {
  write.csv(df, ruta, row.names = FALSE, fileEncoding = "UTF-8")
  invisible(ruta)
}


#' Identificador municipal de la celda (k, j) tolerante al formato de muni_ids.
.muni_id_kj <- function(muni_ids, k, j) {
  if (is.list(muni_ids))        as.character(muni_ids[[k]][j])
  else if (is.matrix(muni_ids)) as.character(muni_ids[j, k])
  else                          as.character(muni_ids[j])
}

calcular_row_cell_m2 <- function(J_k, row_k, row_j) {
  stopifnot(length(row_k) == length(row_j),
            all(row_k >= 1L), all(row_k <= length(J_k)),
            all(row_j >= 1L), all(row_j <= J_k[row_k]))
  offsets <- cumsum(c(0L, J_k))
  as.integer(offsets[row_k] + row_j)
}


# ------------------------------------------------------------------------------
# 2. EXPORTADOR PRINCIPAL
# ------------------------------------------------------------------------------
exportar_para_python_m2 <- function(dir_salida, model_data = NULL,
                                    holdout = NULL, verbose = TRUE) {
  f_meta <- file.path(dir_salida, "metadatos.rds")
  if (!file.exists(f_meta)) {
    stop("No existe ", f_meta, ": correr primero la Etapa 10 (Gibbs).")
  }
  meta <- readRDS(f_meta)
  dir_exp <- file.path(dir_salida, "export_python")
  dir.create(dir_exp, recursive = TRUE, showWarnings = FALSE)
  archivos <- character(0)
  msg <- function(...) if (verbose) cat(sprintf(...), "\n")

  n_save <- as.integer(meta$n_save)
  pp1    <- as.integer(meta$pp1)
  K      <- as.integer(meta$K)
  J_k    <- as.integer(meta$J_k)
  n_cells <- as.integer(meta$n_cells)

  # ---- meta.csv --------------------------------------------------------------
  claves <- list(
    n_save = n_save, n_iter = meta$n_iter, burn_in = meta$burn_in,
    thin = meta$thin, N = meta$N, K = K, n_cells = n_cells,
    p = meta$p, p_E = meta$p_E, p_M = meta$p_M, p_D = meta$p_D, pp1 = pp1,
    guardar_loglik = as.integer(isTRUE(meta$guardar_loglik) ||
                                  identical(meta$guardar_loglik, 1L)),
    guardar_xi = as.integer(isTRUE(meta$guardar_xi) ||
                              identical(meta$guardar_xi, 1L)),
    timestamp = as.character(meta$timestamp)
  )
  claves <- claves[!vapply(claves, is.null, logical(1))]
  archivos <- c(archivos, .escribir_csv(
    data.frame(clave = names(claves),
               valor = vapply(claves, function(v) as.character(v[1]),
                              character(1)),
               stringsAsFactors = FALSE),
    file.path(dir_exp, "meta.csv")))

  # ---- sizes.csv (solo bloques con dim > 0) ----------------------------------
  sz <- meta$sizes
  sz <- sz[vapply(sz, function(d) as.integer(d) > 0L, logical(1))]
  archivos <- c(archivos, .escribir_csv(
    data.frame(bloque = names(sz),
               dim = vapply(sz, as.integer, integer(1)),
               stringsAsFactors = FALSE),
    file.path(dir_exp, "sizes.csv")))

  # ---- verificacion de tamanos de los .bin existentes ------------------------
  for (b in names(sz)) {
    f_bin <- file.path(dir_salida, paste0(b, ".bin"))
    if (!file.exists(f_bin)) {
      warning("Falta ", f_bin, " aunque sizes lo declara (dim=", sz[[b]], ").")
      next
    }
    esperado <- 8 * as.integer(sz[[b]]) * n_save
    real <- file.size(f_bin)
    if (real != esperado) {
      warning(sprintf(
        "%s: %d bytes pero se esperaban %d (8 x %d x n_save=%d). %s",
        f_bin, real, esperado, as.integer(sz[[b]]), n_save,
        "Corrida interrumpida o n_save desactualizado?"))
    }
  }

  # ---- etiquetas -------------------------------------------------------------
  tl <- as.character(meta$theta_labels)
  stopifnot(length(tl) == pp1)
  archivos <- c(archivos,
                .escribir_txt(tl, file.path(dir_exp, "theta_labels.txt")),
                .escribir_txt(tl[meta$idx_E], file.path(dir_exp, "nombres_E.txt")),
                .escribir_txt(tl[meta$idx_M], file.path(dir_exp, "nombres_M.txt")),
                .escribir_txt(tl[meta$idx_D], file.path(dir_exp, "nombres_D.txt")))

  # ---- deptos.csv ------------------------------------------------------------
  archivos <- c(archivos, .escribir_csv(
    data.frame(k = seq_len(K),
               depto_id = as.character(meta$depto_ids),
               stringsAsFactors = FALSE),
    file.path(dir_exp, "deptos.csv")))

  # ---- celdas.csv (orden aplanado k, luego j) --------------------------------
  n_jk_flat <- rep(NA_integer_, n_cells)
  if (!is.null(model_data)) {
    rc_full <- calcular_row_cell_m2(J_k, model_data$row_k, model_data$row_j)
    n_jk_flat <- tabulate(rc_full, nbins = n_cells)
  }
  celda <- 0L
  filas <- vector("list", n_cells)
  for (kk in seq_len(K)) {
    for (jj in seq_len(J_k[kk])) {
      celda <- celda + 1L
      filas[[celda]] <- data.frame(
        celda = celda, k = kk, j = jj,
        depto_id = as.character(meta$depto_ids[kk]),
        muni_id = .muni_id_kj(meta$muni_ids, kk, jj),
        n_jk = n_jk_flat[celda],
        stringsAsFactors = FALSE)
    }
  }
  stopifnot(celda == n_cells)
  archivos <- c(archivos, .escribir_csv(do.call(rbind, filas),
                                        file.path(dir_exp, "celdas.csv")))

  # ---- control.bin -----------------------------------------------------------
  archivos <- c(archivos, .escribir_bin(c(pi, as.double(n_save)),
                                        file.path(dir_exp, "control.bin")))

  # ---- datos a nivel observacion (opcional) ----------------------------------
  if (!is.null(model_data)) {
    md <- model_data
    stopifnot(length(md$y_vec) == meta$N,
              nrow(md$X_star) == meta$N,
              ncol(md$X_star) == pp1)
    rc <- calcular_row_cell_m2(J_k, md$row_k, md$row_j)
    archivos <- c(
      archivos,
      .escribir_bin(md$y_vec, file.path(dir_exp, "y_vec.bin")),
      .escribir_bin(md$row_k, file.path(dir_exp, "row_k.bin")),
      .escribir_bin(rc,       file.path(dir_exp, "row_cell.bin")),
      .escribir_bin(md$X_star, file.path(dir_exp, "X_star.bin")),
      .escribir_csv(data.frame(objeto = "X_star",
                               filas = nrow(md$X_star),
                               columnas = ncol(md$X_star),
                               orden = "col-major",
                               stringsAsFactors = FALSE),
                    file.path(dir_exp, "dims.csv")))
    msg("  datos: y_vec/row_k/row_cell/X_star exportados (N=%d, pp1=%d)",
        meta$N, pp1)
  } else {
    msg("  datos: model_data no suministrado -> solo metadatos (14_d y el")
    msg("         DIC de 14_c quedan deshabilitados; ver encabezado).")
  }

  # ---- hold-out (opcional) ---------------------------------------------------
  if (!is.null(holdout)) {
    h <- holdout
    if (is.null(h$y) && !is.null(h$y_test)) h$y <- h$y_test
    stopifnot(!is.null(h$y), !is.null(h$X_star),
              !is.null(h$row_k), !is.null(h$row_j),
              length(h$y) == nrow(h$X_star),
              ncol(h$X_star) == pp1,
              length(h$row_k) == length(h$y),
              length(h$row_j) == length(h$y))
    rc_t <- calcular_row_cell_m2(J_k, h$row_k, h$row_j)
    dir_h <- file.path(dir_exp, "holdout")
    dir.create(dir_h, showWarnings = FALSE)
    archivos <- c(
      archivos,
      .escribir_bin(h$y,      file.path(dir_h, "y_test.bin")),
      .escribir_bin(h$row_k,  file.path(dir_h, "row_k_test.bin")),
      .escribir_bin(rc_t,     file.path(dir_h, "row_cell_test.bin")),
      .escribir_bin(h$X_star, file.path(dir_h, "X_star_test.bin")),
      .escribir_csv(data.frame(objeto = "X_star_test",
                               filas = nrow(h$X_star),
                               columnas = ncol(h$X_star),
                               orden = "col-major",
                               stringsAsFactors = FALSE),
                    file.path(dir_h, "dims.csv")))
    msg("  holdout: %d observaciones de prueba exportadas", length(h$y))
  }

  msg("[OK] Etapa 13: export_python/ escrito en %s (%d archivos, n_save=%d)",
      dir_exp, length(archivos), n_save)
  invisible(list(dir_export = dir_exp, n_save = n_save, archivos = archivos))
}


# ------------------------------------------------------------------------------
# 3. AUTOVERIFICACION
# ------------------------------------------------------------------------------
check_exportador_m2 <- function() {
  dir_tmp <- file.path(tempdir(), sprintf("chk_etapa13_%d", Sys.getpid()))
  unlink(dir_tmp, recursive = TRUE)
  dir.create(dir_tmp, recursive = TRUE)
  on.exit(unlink(dir_tmp, recursive = TRUE), add = TRUE)

  K <- 2L; J_k <- c(2L, 3L); n_cells <- sum(J_k)
  p_E <- 1L; p_M <- 1L; p_D <- 1L; pp1 <- 4L; n_save <- 7L; N <- 9L
  sizes <- list(beta_star = pp1, sigma2_mu = 1L, nu = 1L, lambda2 = 3L,
                kappa2_k = K, kappa2_jk = n_cells,
                tau2_E = p_E, tau2_M = p_M, tau2_D = p_D)
  meta <- list(n_save = n_save, n_iter = 70L, burn_in = 0L, thin = 10L,
               p = p_E + p_M + p_D, p_E = p_E, p_M = p_M, p_D = p_D,
               pp1 = pp1, K = K, J_k = J_k, n_cells = n_cells, N = N,
               theta_labels = c("intercepto", "E1", "M1", "D1"),
               depto_ids = c("D01", "D02"),
               muni_ids = list(c("M11", "M12"), c("M21", "M22", "M23")),
               idx_mu = 1L, idx_E = 2L, idx_M = 3L, idx_D = 4L,
               guardar_loglik = FALSE, guardar_xi = FALSE,
               sizes = sizes, hyper = list(), timestamp = Sys.time())
  saveRDS(meta, file.path(dir_tmp, "metadatos.rds"))
  set.seed(13)
  for (b in names(sizes)) {
    .escribir_bin(rnorm(as.integer(sizes[[b]]) * n_save),
                  file.path(dir_tmp, paste0(b, ".bin")))
  }

  row_k <- c(1L, 1L, 1L, 2L, 2L, 2L, 2L, 2L, 1L)
  row_j <- c(1L, 2L, 1L, 3L, 1L, 2L, 3L, 1L, 2L)
  md <- list(y_vec = rnorm(N, 250, 10),
             X_star = matrix(rnorm(N * pp1), N, pp1),
             row_k = row_k, row_j = row_j)
  ho <- list(y = rnorm(3, 250, 10), X_star = matrix(rnorm(3 * pp1), 3, pp1),
             row_k = c(1L, 2L, 2L), row_j = c(2L, 3L, 1L))

  res <- exportar_para_python_m2(dir_tmp, model_data = md, holdout = ho,
                                 verbose = FALSE)
  de <- res$dir_export

  # control.bin
  ctl <- readBin(file.path(de, "control.bin"), "double", 2, size = 8,
                 endian = "little")
  stopifnot(abs(ctl[1] - pi) < 1e-12, ctl[2] == n_save)

  # y_vec y X_star (column-major) identicos
  y_rt <- readBin(file.path(de, "y_vec.bin"), "double", N, size = 8,
                  endian = "little")
  stopifnot(identical(y_rt, as.double(md$y_vec)))
  x_rt <- matrix(readBin(file.path(de, "X_star.bin"), "double", N * pp1,
                         size = 8, endian = "little"), N, pp1)
  stopifnot(identical(x_rt, md$X_star))

  # row_cell = offsets[row_k] + row_j
  rc <- readBin(file.path(de, "row_cell.bin"), "double", N, size = 8,
                endian = "little")
  offsets <- cumsum(c(0L, J_k))
  stopifnot(identical(as.integer(rc), as.integer(offsets[row_k] + row_j)))

  # celdas.csv: orden aplanado y n_jk = tabulate
  celdas <- read.csv(file.path(de, "celdas.csv"), stringsAsFactors = FALSE)
  stopifnot(nrow(celdas) == n_cells,
            identical(celdas$k, c(1L, 1L, 2L, 2L, 2L)),
            identical(celdas$j, c(1L, 2L, 1L, 2L, 3L)),
            identical(celdas$n_jk, tabulate(as.integer(rc), nbins = n_cells)))

  # hold-out
  rc_t <- readBin(file.path(de, "holdout", "row_cell_test.bin"), "double", 3,
                  size = 8, endian = "little")
  stopifnot(identical(as.integer(rc_t),
                      as.integer(offsets[ho$row_k] + ho$row_j)))

  # deteccion de .bin truncado
  f_nu <- file.path(dir_tmp, "nu.bin")
  .escribir_bin(rnorm(n_save - 1L), f_nu)   # truncar a proposito
  ws <- character(0)
  withCallingHandlers(
    exportar_para_python_m2(dir_tmp, verbose = FALSE),
    warning = function(cond) {
      ws <<- c(ws, conditionMessage(cond))
      invokeRestart("muffleWarning")
    })
  stopifnot(any(grepl("nu\\.bin", ws)))

  cat("[OK] Etapa 13 (exportador a Python): autoverificacion superada.\n")
  invisible(TRUE)
}


# ==============================================================================
# EJECUCION
# ==============================================================================

if (isTRUE(getOption("VERIFICAR_MODULOS", TRUE))) {
  check_exportador_m2()
}
exportar_para_python_m2("C:/Users/USUARIO/OneDrive - Universidad Nacional de Colombia/Tesis/Escritorio/Códigos/Modelos/Modelo 2/mcmc_salida_m2", model_data = model_data)
