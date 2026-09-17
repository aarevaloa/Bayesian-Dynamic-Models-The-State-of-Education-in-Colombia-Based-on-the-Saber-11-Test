# ______________________________________________________________________________
#            BUCLE PRINCIPAL DEL MUESTREADOR DE GIBBS             

# ______________________________________________________________________________
# @A. CONFIGURACIÓN DE ALMACENAMIENTO ----
# ______________________________________________________________________________
setwd('C:/Users/USUARIO/OneDrive - Universidad Nacional de Colombia/Tesis/Escritorio/Códigos/Modelos/Modelo 3')
# Directorio donde se guardan las muestras y checkpoints
dir_salida <- file.path(getwd(), "1_Resultados_mcmc_output")
if (!dir.exists(dir_salida)) dir.create(dir_salida, recursive = TRUE)

# Frecuencia de checkpoints
checkpoint_cada <- 500L

cat(sprintf("║  Iteraciones     : %-27d ║\n", n_iter))
cat(sprintf("║  Burn-in         : %-27d ║\n", burn_in))
cat(sprintf("║  Thinning        : %-27d ║\n", thin))
cat(sprintf("║  Muestras        : %-27d ║\n", n_save))
cat(sprintf("║  Checkpoint cada : %-27d ║\n", checkpoint_cada))
cat(sprintf("║  Directorio      : %-27s ║\n",
            basename(dir_salida)))


# Estimar espacio en disco
bytes_theta   <- p * (Tt + 1) * 8
bytes_lambda2 <- 3 * 8
bytes_v       <- max(n_k) * m * 8
bytes_s       <- m * Tt * 8
bytes_tau2    <- sum(sapply(bloques, function(b) b$dim)) * Tt * 8
bytes_por_muestra <- bytes_theta + bytes_lambda2 + bytes_v +
  bytes_s + bytes_tau2
bytes_total <- bytes_por_muestra * n_save

cat(sprintf("Espacio estimado en disco: %.0f MB (%.2f GB)\n",
            bytes_total / 1e6, bytes_total / 1e9))
cat(sprintf("  theta:   %.0f MB\n", bytes_theta * n_save / 1e6))
cat(sprintf("  v:       %.0f MB\n", bytes_v * n_save / 1e6))
cat(sprintf("  s:       %.0f MB\n", bytes_s * n_save / 1e6))
cat(sprintf("  tau2:    %.0f MB\n", bytes_tau2 * n_save / 1e6))


# ______________________________________________________________________________
# B. GUARDAR METADATOS ----
# ______________________________________________________________________________
metadatos <- list(
  p         = p,
  Tt        = Tt,
  m         = m,
  n_k       = n_k,
  max_nk    = max(n_k),
  n_iter    = n_iter,
  burn_in   = burn_in,
  thin      = thin,
  n_save    = n_save,
  p_E       = p_E,
  p_M       = p_M,
  p_D       = p_D,
  bloques_dim = sapply(bloques, function(b) b$dim),
  theta_labels = theta_labels,
  depto_ids    = depto_ids,
  tiempo_ids   = tiempo_ids,
  fecha_inicio = Sys.time()
)
saveRDS(metadatos, file.path(dir_salida, "metadatos.rds"))
n_phi_fix <- 0L

# ______________________________________________________________________________
# C. SISTEMA DE CHECKPOINTS: VERIFICAR REINICIO
# ______________________________________________________________________________

archivo_checkpoint <- file.path(dir_salida, "checkpoint.rds")
iter_inicio <- 1L
save_idx    <- 0L

if (file.exists(archivo_checkpoint)) {
  cat("═══ CHECKPOINT DETECTADO ═══\n")
  ckpt <- readRDS(archivo_checkpoint)
  
  iter_inicio  <- ckpt$iter + 1L
  save_idx     <- ckpt$save_idx
  theta_curr   <- ckpt$theta_curr
  tau2_curr    <- ckpt$tau2_curr
  lambda2_curr <- ckpt$lambda2_curr
  v_curr       <- ckpt$v_curr
  s_curr       <- ckpt$s_curr
  phi_curr     <- ckpt$phi_curr
  W_diag       <- ckpt$W_diag
  n_kt_work    <- ckpt$n_kt_work
  d_kt_work    <- ckpt$d_kt_work
  
  # Restaurar estado del generador aleatorio
  .Random.seed <<- ckpt$random_seed
  
  cat(sprintf("  Reanudando desde iteración %d (muestras guardadas: %d)\n",
              iter_inicio, save_idx))
  rm(ckpt)
  gc(verbose = FALSE)
  
} else {
  cat("Sin checkpoint previo. Iniciando desde cero.\n")
}


# ______________________________________________________________________________
# D. ABRIR ARCHIVOS BINARIOS PARA ESCRITURA ----
# ______________________________________________________________________________

modo <- if (iter_inicio == 1L) "wb" else "ab"

con_theta   <- file(file.path(dir_salida, "theta.bin"),   modo)
con_lambda2 <- file(file.path(dir_salida, "lambda2.bin"), modo)
con_v       <- file(file.path(dir_salida, "v.bin"),       modo)
con_s       <- file(file.path(dir_salida, "s.bin"),       modo)

con_tau2 <- list()
for (B in names(bloques)) {
  con_tau2[[B]] <- file(file.path(dir_salida,
                                  sprintf("tau2_%s.bin", B)), modo)
}

# Asegurar que los archivos se cierren al salir
on.exit({
  try(close(con_theta),   silent = TRUE)
  try(close(con_lambda2), silent = TRUE)
  try(close(con_v),       silent = TRUE)
  try(close(con_s),       silent = TRUE)
  for (B in names(con_tau2)) try(close(con_tau2[[B]]), silent = TRUE)
  cat("Conexiones a archivos binarios cerradas.\n")
}, add = TRUE)


# ______________________________________________________________________________
# E. ALOCAR ARRAYS DE TRABAJO ----
# ______________________________________________________________________________

# a_pred y R_pred para el forward filter
if (!exists("a_pred") || is.null(a_pred)) {
  a_pred <- matrix(NA_real_, nrow = p, ncol = Tt)
  R_pred <- array(NA_real_, dim = c(p, p, Tt))
}

t_inicio   <- proc.time()["elapsed"]
iter_print <- max(1L, as.integer(n_iter / 50))


# ______________________________________________________________________________
# F. BUCLE PRINCIPAL ----
# ______________________________________________________________________________

for (iter in iter_inicio:n_iter) {
  
  # ── 1. tau2 ──
  tau2_curr <- muestrear_tau2(theta_curr, tau2_curr, lambda2_curr,
                              bloques, Tt)
  
  # ── 2. lambda2 ──
  lambda2_curr <- muestrear_lambda2(lambda2_curr, tau2_curr,
                                    bloques, Tt)
  
  # ── 3. W_diag ──
  W_diag <- construir_W_diag(tau2_curr, bloques, q_mu, p, Tt)
  
  # ── 4. FFBS ──
  res_ffbs <- ffbs(theta_curr, m_filt, C_filt, a_pred, R_pred,
                   Y, F_lst, N_jkt, n_k, m, Tt, p,
                   W_diag, s_curr, v_curr, N_kt)
  
  theta_curr <- res_ffbs$theta_curr
  m_filt     <- res_ffbs$m_filt
  C_filt     <- res_ffbs$C_filt
  a_pred     <- res_ffbs$a_pred
  R_pred     <- res_ffbs$R_pred
  
  # ── 5. SSE ──
  SSE_jkt <- calcular_SSE(SSE_jkt, Y, F_lst, theta_curr,
                          N_jkt, n_k, m, Tt)
  
  # ── 6. v ──
  v_curr <- muestrear_v(v_curr, SSE_jkt, s_curr, N_jk,
                        n_k, m, Tt, a_v, b_v)
  
  # ── 7. phi y s ──
  res_phi <- muestrear_phi(phi_curr, s_curr, SSE_jkt, v_curr,
                           N_kt, n_k, m, Tt,
                           beta_discount, n_k0, d_k0,
                           n_kt_work, d_kt_work)
  phi_curr  <- res_phi$phi_curr
  s_curr    <- res_phi$s_curr
  n_kt_work <- res_phi$n_kt_work
  d_kt_work <- res_phi$d_kt_work
  
  # ── 8. ESCRIBIR MUESTRA A DISCO ──
  if (iter > burn_in && (iter - burn_in) %% thin == 0L) {
    save_idx <- save_idx + 1L
    
    # theta: p × (Tt+1) escritos como vector columna-major
    writeBin(as.double(theta_curr), con_theta)
    
    # lambda2: 3 doubles
    writeBin(as.double(lambda2_curr), con_lambda2)
    
    # v: max(n_k) × m
    writeBin(as.double(v_curr), con_v)
    
    # s: m × Tt
    writeBin(as.double(s_curr), con_s)
    
    # tau2: por bloque
    for (B in names(bloques)) {
      writeBin(as.double(tau2_curr[[B]]), con_tau2[[B]])
    }
  }
  
  # ── 9. CHECKPOINT ──
  if (iter %% checkpoint_cada == 0L) {
    # Flush buffers a disco antes de guardar checkpoint
    flush(con_theta); flush(con_lambda2)
    flush(con_v); flush(con_s)
    for (B in names(con_tau2)) flush(con_tau2[[B]])
    
    saveRDS(list(
      iter         = iter,
      save_idx     = save_idx,
      theta_curr   = theta_curr,
      tau2_curr    = tau2_curr,
      lambda2_curr = lambda2_curr,
      v_curr       = v_curr,
      s_curr       = s_curr,
      phi_curr     = phi_curr,
      W_diag       = W_diag,
      n_kt_work    = n_kt_work,
      d_kt_work    = d_kt_work,
      random_seed  = .Random.seed
    ), archivo_checkpoint)
  }
  
  # ── 10. PROGRESO ──
  if (iter %% 1L == 0L || iter == iter_inicio || iter == n_iter) {
    t_actual <- proc.time()["elapsed"]
    t_trans  <- t_actual - t_inicio
    # Tiempo por iteración basado en las iteraciones ejecutadas en esta sesión
    iters_esta_sesion <- iter - iter_inicio + 1L
    t_por_it <- t_trans / iters_esta_sesion
    t_rest   <- t_por_it * (n_iter - iter)
    
    fase <- if (iter <= burn_in) "burn-in" else "muestreo"
    cat(sprintf("[%6d/%d] %s | %.0fs (rest: %.0fs ≈ %.1fh) | μ_T=%.1f ṽ=%.2f | disco: %d\n",
                iter, n_iter, fase,
                t_trans, t_rest, t_rest / 3600,
                theta_curr[1, Tt + 1],
                median(v_curr, na.rm = TRUE),
                save_idx))
  }
}


# ______________________________________________________________________________
# G. CERRAR Y VERIFICAR ----
# ______________________________________________________________________________

close(con_theta);   close(con_lambda2)
close(con_v);       close(con_s)
for (B in names(con_tau2)) close(con_tau2[[B]])
on.exit(NULL)

# Verificar tamaños de archivo
bytes_esperado_theta <- n_save * p * (Tt + 1) * 8
bytes_real_theta     <- file.size(file.path(dir_salida, "theta.bin"))

cat(sprintf("\n  theta.bin: %.1f MB (esperado: %.1f MB) %s\n",
            bytes_real_theta / 1e6,
            bytes_esperado_theta / 1e6,
            ifelse(abs(bytes_real_theta - bytes_esperado_theta) < 8,
                   "✓", "⚠ TAMAÑO INCORRECTO")))

# Actualizar metadatos con resultados finales y timing
t_total <- proc.time()["elapsed"] - t_inicio

metadatos$fecha_fin       <- Sys.time()
metadatos$save_idx        <- save_idx
metadatos$tiempo_total_s  <- t_total
metadatos$tiempo_total_h  <- t_total / 3600
metadatos$tiempo_por_iter <- t_total / (n_iter - iter_inicio + 1L)
metadatos$iters_ejecutadas <- n_iter - iter_inicio + 1L

saveRDS(metadatos, file.path(dir_salida, "metadatos.rds"))

# Eliminar checkpoint (corrida completa)
if (save_idx == n_save && file.exists(archivo_checkpoint)) {
  file.remove(archivo_checkpoint)
  cat("  Checkpoint eliminado (corrida completa)\n")
}

cat(sprintf("\n✓ %d muestras escritas en %s/\n", save_idx, dir_salida))
cat(sprintf("  Iteraciones : %d (burn-in: %d, thin: %d)\n",
            n_iter, burn_in, thin))
cat(sprintf("  Tiempo total: %.1f min (%.2f h)\n",
            t_total / 60, t_total / 3600))
cat(sprintf("  Tiempo/iter : %.3f s\n", t_total / (n_iter - iter_inicio + 1L)))

# ==================================================================
# §H. FUNCIÓN DE LECTURA POST-HOC
# ==================================================================

leer_muestras <- function(dir_salida, que = "theta",
                          desde = 1, hasta = NULL) {
  
  meta <- readRDS(file.path(dir_salida, "metadatos.rds"))
  if (is.null(hasta)) hasta <- meta$save_idx
  n_leer <- hasta - desde + 1L
  
  if (que == "theta") {
    n_por_muestra <- meta$p * (meta$Tt + 1)
    con <- file(file.path(dir_salida, "theta.bin"), "rb")
    # Saltar las primeras (desde-1) muestras
    if (desde > 1) seek(con, (desde - 1) * n_por_muestra * 8)
    raw <- readBin(con, "double", n = n_leer * n_por_muestra)
    close(con)
    arr <- array(raw, dim = c(meta$p, meta$Tt + 1, n_leer))
    rownames(arr) <- meta$theta_labels
    return(arr)
    
  } else if (que == "lambda2") {
    n_por_muestra <- 3L
    con <- file(file.path(dir_salida, "lambda2.bin"), "rb")
    if (desde > 1) seek(con, (desde - 1) * n_por_muestra * 8)
    raw <- readBin(con, "double", n = n_leer * n_por_muestra)
    close(con)
    mat <- matrix(raw, nrow = n_leer, ncol = 3, byrow = TRUE)
    colnames(mat) <- c("E", "M", "D")
    return(mat)
    
  } else if (que == "v") {
    n_por_muestra <- meta$max_nk * meta$m
    con <- file(file.path(dir_salida, "v.bin"), "rb")
    if (desde > 1) seek(con, (desde - 1) * n_por_muestra * 8)
    raw <- readBin(con, "double", n = n_leer * n_por_muestra)
    close(con)
    return(array(raw, dim = c(meta$max_nk, meta$m, n_leer)))
    
  } else if (que == "s") {
    n_por_muestra <- meta$m * meta$Tt
    con <- file(file.path(dir_salida, "s.bin"), "rb")
    if (desde > 1) seek(con, (desde - 1) * n_por_muestra * 8)
    raw <- readBin(con, "double", n = n_leer * n_por_muestra)
    close(con)
    return(array(raw, dim = c(meta$m, meta$Tt, n_leer)))
    
  } else if (startsWith(que, "tau2_")) {
    B <- sub("tau2_", "", que)
    p_B <- meta$bloques_dim[B]
    n_por_muestra <- p_B * meta$Tt
    con <- file(file.path(dir_salida, sprintf("tau2_%s.bin", B)), "rb")
    if (desde > 1) seek(con, (desde - 1) * n_por_muestra * 8)
    raw <- readBin(con, "double", n = n_leer * n_por_muestra)
    close(con)
    return(array(raw, dim = c(p_B, meta$Tt, n_leer)))
    
  } else {
    stop(sprintf("Tipo '%s' no reconocido. Use: theta, lambda2, v, s, tau2_E/M/D", que))
  }
}


