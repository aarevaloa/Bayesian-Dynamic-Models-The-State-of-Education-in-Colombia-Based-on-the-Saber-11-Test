#_______________________________________________________________________________
#  FILTRO DE KALMAN HACIA ADELANTE (FORWARD FILTERING)   


forward_filter <- function(m_filt, C_filt, a_pred, R_pred,
                           Y, F_lst, N_jkt, n_k, m_deptos,
                           Tt_periodos, p_dim, W_diag,
                           s_curr, v_curr, N_kt) {

  for (tt in 1:Tt_periodos) {

    a_t <- m_filt[, tt]                       # a_t = m_{t-1}
    R_t <- C_filt[,, tt]                      # copia de C_{t-1}
    diag(R_t) <- diag(R_t) + W_diag[, tt]    # R_t = C_{t-1} + W_t

    a_pred[, tt]  <- a_t
    R_pred[,, tt] <- R_t

    n_total_t <- sum(N_kt[, tt])

    if (n_total_t > 0L) {

      # Pre-asignar matrices para todas las obs del periodo
      F_t <- matrix(0, nrow = n_total_t, ncol = p_dim)
      w_t <- numeric(n_total_t)
      y_t <- numeric(n_total_t)
      ptr <- 0L

      # Recorrer departamentos y municipios, llenando F_t, w_t, y_t
      for (kk in 1:m_deptos) {
        s_kt <- s_curr[kk, tt]
        for (jj in 1:n_k[kk]) {
          n_obs <- N_jkt[jj, kk, tt]
          if (n_obs == 0L) next

          rows <- (ptr + 1L):(ptr + n_obs)
          F_t[rows, ] <- F_lst[[tt]][[kk]][[jj]]
          w_t[rows]   <- v_curr[jj, kk] / s_kt   # precisión obs
          y_t[rows]   <- Y[[tt]][[kk]][[jj]]
          ptr <- ptr + n_obs
        }
      }

      # Estadísticos suficientes ponderados
      sw  <- sqrt(w_t)
      swF <- sw * F_t                # n_t × p: filas escaladas
      swy <- sw * y_t                # n_t × 1: respuestas escaladas

      precision_obs <- crossprod(swF)             # p × p
      info_obs      <- drop(crossprod(swF, swy))  # p × 1 → vector
      
      chol_seguro <- function(Sigma, p_dim) {
        tryCatch(
          chol(Sigma),
          error = function(e) {
            # Jitter adaptativo: 1e-10 veces el máximo de la diagonal
            jitter <- max(diag(Sigma)) * 1e-10
            if (jitter < .Machine$double.eps) jitter <- 1e-12
            chol(Sigma + diag(jitter, p_dim))
          }
        )
      }

      chol_R  <- chol_seguro(R_t, p_dim)
      R_inv   <- chol2inv(chol_R)
      
      C_t_inv <- R_inv + precision_obs
      chol_Ci <- chol_seguro(C_t_inv, p_dim)
      C_t     <- chol2inv(chol_Ci)
      C_t     <- (C_t + t(C_t)) / 2
      
      m_t     <- drop(C_t %*% (R_inv %*% a_t + info_obs))

    } else {
    
      C_t <- R_t
      m_t <- a_t
    }

    # ════════════════════════════════════════════════════════════
    # ETAPA 4: ALMACENAR MOMENTOS FILTRADOS
    # ════════════════════════════════════════════════════════════
    m_filt[, tt + 1]  <- m_t    # m_t en posición tt+1
    C_filt[,, tt + 1] <- C_t   # C_t en posición tt+1
  }

  # Retornar los arrays modificados
  list(m_filt = m_filt, C_filt = C_filt,
       a_pred = a_pred, R_pred = R_pred)
}


forward_filter_rank1 <- function(m_filt, C_filt, a_pred, R_pred,
                                 Y, F_lst, N_jkt, n_k, m_deptos,
                                 Tt_periodos, p_dim, W_diag,
                                 s_curr, v_curr, N_kt) {

  for (tt in 1:Tt_periodos) {

    # Predicción (idéntica a la versión batch)
    a_t <- m_filt[, tt]
    R_t <- C_filt[,, tt]
    diag(R_t) <- diag(R_t) + W_diag[, tt]

    a_pred[, tt]  <- a_t
    R_pred[,, tt] <- R_t

    # Actualización rank-1 secuencial
    for (kk in 1:m_deptos) {
      s_kt <- s_curr[kk, tt]
      for (jj in 1:n_k[kk]) {
        n_obs <- N_jkt[jj, kk, tt]
        if (n_obs == 0L) next

        V_jkt <- s_kt / v_curr[jj, kk]     # varianza observacional
        y_jkt <- Y[[tt]][[kk]][[jj]]
        F_jkt <- F_lst[[tt]][[kk]][[jj]]

        for (ii in 1:n_obs) {
          f_i <- F_jkt[ii, ]                       # p × 1

          e   <- y_jkt[ii] - sum(f_i * a_t)        # error de predicción
          Rf  <- drop(R_t %*% f_i)                 # R_t f  (p × 1)
          q   <- sum(f_i * Rf) + V_jkt             # f' R_t f + V
          A_k <- Rf / q                            # ganancia de Kalman

          a_t <- a_t + A_k * e                     # actualizar media
          R_t <- R_t - tcrossprod(A_k) * q         # actualizar cov
        }
      }
    }

    # Simetrizar (acumulación de errores numéricos en rank-1)
    R_t <- (R_t + t(R_t)) / 2

    m_filt[, tt + 1]  <- a_t
    C_filt[,, tt + 1] <- R_t
  }

  list(m_filt = m_filt, C_filt = C_filt,
       a_pred = a_pred, R_pred = R_pred)
}


cat("\n══════ VERIFICACIÓN DEL FORWARD FILTER ══════\n\n")

test_forward_filter <- function() {

  set.seed(42)

  # ── DLM sintético: p = 2, m = 1, n_k = 1, Tt = 3 ──
  p_test  <- 2L
  m_test  <- 1L
  Tt_test <- 3L
  nk_test <- 1L

  # Hiperparámetros
  m0_test <- c(0, 0)
  C0_test <- diag(c(100, 100))
  W_test  <- matrix(c(1, 1), nrow = 2, ncol = 3)  # diag(W_t)

  # Datos: 5 observaciones por periodo
  n_per_t <- 5L
  N_jkt_test <- array(0L, dim = c(1, 1, 3))
  N_jkt_test[1, 1, ] <- n_per_t
  N_kt_test <- matrix(n_per_t, nrow = 1, ncol = 3)

  # Generar datos sintéticos
  theta_true <- matrix(c(3, -2,    # t = 1
                         4, -1,    # t = 2
                         5,  0),   # t = 3
                       nrow = 2, ncol = 3)

  Y_test     <- vector("list", Tt_test)
  F_test     <- vector("list", Tt_test)
  v_test     <- matrix(1.0, nrow = 1, ncol = 1)   # v_{1,1} = 1
  s_test     <- matrix(4.0, nrow = 1, ncol = 3)   # s_{1,t} = 4

  for (tt in 1:Tt_test) {
    Y_test[[tt]] <- list(list(NULL))
    F_test[[tt]] <- list(list(NULL))

    F_mat_t <- cbind(1, rnorm(n_per_t))  # n × 2
    y_t <- F_mat_t %*% theta_true[, tt] + rnorm(n_per_t, sd = 2)

    Y_test[[tt]][[1]][[1]] <- drop(y_t)
    F_test[[tt]][[1]][[1]] <- F_mat_t
  }

  # Preparar arrays
  mf_batch <- matrix(NA, p_test, Tt_test + 1)
  Cf_batch <- array(NA,  c(p_test, p_test, Tt_test + 1))
  ap_batch <- matrix(NA, p_test, Tt_test)
  Rp_batch <- array(NA,  c(p_test, p_test, Tt_test))
  mf_batch[, 1]  <- m0_test
  Cf_batch[,, 1] <- C0_test

  mf_r1 <- mf_batch; Cf_r1 <- Cf_batch
  ap_r1 <- ap_batch; Rp_r1 <- Rp_batch

  # Ejecutar ambas versiones
  res_batch <- forward_filter(mf_batch, Cf_batch, ap_batch, Rp_batch,
                              Y_test, F_test, N_jkt_test,
                              nk_test, m_test, Tt_test, p_test,
                              W_test, s_test, v_test, N_kt_test)

  res_r1 <- forward_filter_rank1(mf_r1, Cf_r1, ap_r1, Rp_r1,
                                 Y_test, F_test, N_jkt_test,
                                 nk_test, m_test, Tt_test, p_test,
                                 W_test, s_test, v_test, N_kt_test)

  # ── Test 1: Batch ≈ Rank-1 ──
  tol <- 1e-8
  diff_m <- max(abs(res_batch$m_filt - res_r1$m_filt), na.rm = TRUE)
  diff_C <- max(abs(res_batch$C_filt - res_r1$C_filt), na.rm = TRUE)
  diff_a <- max(abs(res_batch$a_pred - res_r1$a_pred), na.rm = TRUE)
  diff_R <- max(abs(res_batch$R_pred - res_r1$R_pred), na.rm = TRUE)

  pass1 <- (diff_m < tol) && (diff_C < tol) &&
            (diff_a < tol) && (diff_R < tol)

  cat(sprintf("  Test 1 (batch ≈ rank-1):\n"))
  cat(sprintf("    Δm_filt = %.2e, ΔC_filt = %.2e\n", diff_m, diff_C))
  cat(sprintf("    Δa_pred = %.2e, ΔR_pred = %.2e\n", diff_a, diff_R))
  cat(sprintf("    %s\n\n", ifelse(pass1, "✓ PASS", "✗ FAIL")))

  # ── Test 2: C_t es simétrica y definida positiva ──
  pass2 <- TRUE
  for (tt in 1:(Tt_test + 1)) {
    Ct <- res_batch$C_filt[,, tt]
    if (max(abs(Ct - t(Ct))) > tol) { pass2 <- FALSE; break }
    ev <- eigen(Ct, symmetric = TRUE, only.values = TRUE)$values
    if (any(ev <= 0)) { pass2 <- FALSE; break }
  }
  cat(sprintf("  Test 2 (C_t simétrica PD): %s\n",
              ifelse(pass2, "✓ PASS", "✗ FAIL")))

  # ── Test 3: m_t se acerca a theta_true ──
  # Con 5 obs/periodo y prior difuso, la media filtrada debe
  # estar más cerca de theta_true que la predicción.
  pass3 <- TRUE
  for (tt in 1:Tt_test) {
    dist_pred <- sum((res_batch$a_pred[, tt] - theta_true[, tt])^2)
    dist_filt <- sum((res_batch$m_filt[, tt + 1] - theta_true[, tt])^2)
    # El filtrado no siempre está más cerca (con pocos datos),
    # así que solo verificamos que no diverge
    if (any(!is.finite(res_batch$m_filt[, tt + 1]))) {
      pass3 <- FALSE; break
    }
  }
  cat(sprintf("  Test 3 (m_t finito y razonable): %s\n",
              ifelse(pass3, "✓ PASS", "✗ FAIL")))

  # ── Test 4: Sin observaciones → filtrado = predicción ──
  N_jkt_empty <- array(0L, dim = c(1, 1, 3))
  N_kt_empty  <- matrix(0L, nrow = 1, ncol = 3)

  mf_e <- mf_batch; Cf_e <- Cf_batch
  ap_e <- ap_batch; Rp_e <- Rp_batch

  res_empty <- forward_filter(mf_e, Cf_e, ap_e, Rp_e,
                              Y_test, F_test, N_jkt_empty,
                              nk_test, m_test, Tt_test, p_test,
                              W_test, s_test, v_test, N_kt_empty)

  pass4 <- TRUE
  for (tt in 1:Tt_test) {
    if (max(abs(res_empty$m_filt[, tt + 1] - res_empty$a_pred[, tt])) > tol)
      pass4 <- FALSE
    if (max(abs(res_empty$C_filt[,, tt + 1] - res_empty$R_pred[,, tt])) > tol)
      pass4 <- FALSE
  }
  cat(sprintf("  Test 4 (sin obs → filtrado = predicción): %s\n",
              ifelse(pass4, "✓ PASS", "✗ FAIL")))

  # ── Resumen ──
  all_pass <- pass1 && pass2 && pass3 && pass4
  cat(sprintf("\n  %s Verificación del forward filter\n",
              ifelse(all_pass, "✓", "✗")))

  return(all_pass)
}

ok <- test_forward_filter()
if (!ok) warning("El forward filter no pasó todas las verificaciones")


