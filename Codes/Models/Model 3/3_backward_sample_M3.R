#_______________________________________________________________________________
#                                                                 
#   MUESTREO HACIA ATRÁS (BACKWARD SAMPLING)              
#                                                             

backward_sample <- function(theta_curr, m_filt, C_filt,
                            a_pred, R_pred,
                            Tt_periodos, p_dim) {

  m_T <- m_filt[, Tt_periodos + 1]
  C_T <- C_filt[,, Tt_periodos + 1]
  C_T <- (C_T + t(C_T)) / 2                  # simetría exacta

  U_T <- chol_seguro(C_T, p_dim)
  theta_curr[, Tt_periodos + 1] <- m_T +
    drop(crossprod(U_T, rnorm(p_dim)))


  for (tt in (Tt_periodos - 1):0) {

    # ── Recuperar momentos filtrados en t ──
    m_t <- m_filt[, tt + 1]          # m_t
    C_t <- C_filt[,, tt + 1]        # C_t

    # ── Recuperar momentos de predicción en t+1 ──
    # a_{t+1} = m_t y R_{t+1} = C_t + W_{t+1}, almacenados
    # durante el forward filter.
    a_tp1 <- a_pred[, tt + 1]        # a_{t+1}
    R_tp1 <- R_pred[,, tt + 1]      # R_{t+1}

    # ── Matriz de suavizado ──
    # B_t = C_t G_{t+1}' R_{t+1}^{-1} = C_t R_{t+1}^{-1}  (G = I)
    #
    # Implementación: solve(R_{t+1}, C_t) da R_{t+1}^{-1} C_t;
    # como ambas son simétricas, su transpuesta es
    # C_t R_{t+1}^{-1} = B_t.
    B_t <- t(solve(R_tp1, C_t))

    # ── Media suavizada ──
    # m_t* = m_t + B_t (θ_{t+1} - a_{t+1})
    #
    # Interpretación:
    #   m_t resume lo que se sabe de θ_t usando datos hasta t.
    #   (θ_{t+1} - a_{t+1}) cuantifica cuánto se desvió el
    #   estado futuro (ya muestreado) de su predicción.
    #   B_t controla en qué medida esa desviación corrige m_t.
    theta_tp1 <- theta_curr[, tt + 2]    # θ_{t+1} ya muestreado
    m_star <- drop(m_t + B_t %*% (theta_tp1 - a_tp1))

    # ── Covarianza suavizada ──
    # C_t* = C_t - B_t R_{t+1} B_t'
    #
    # Simplificación: B_t R_{t+1} = C_t R_{t+1}^{-1} R_{t+1} = C_t
    # ⇒  C_t* = C_t - B_t R_{t+1} B_t' = C_t - C_t B_t'
    #         = C_t - B_t C_t            (por simetría de C_t)
    #
    # Esto ahorra una multiplicación matricial respecto a la
    # fórmula original con tres factores.
    C_star <- C_t - B_t %*% C_t
    C_star <- (C_star + t(C_star)) / 2    # simetría exacta

    # ── Muestrear θ_t ~ N(m_t*, C_t*) ──
    U <- chol_seguro(C_star, p_dim)
    theta_curr[, tt + 1] <- m_star +
      drop(crossprod(U, rnorm(p_dim)))
  }

  theta_curr
}



chol_seguro <- function(Sigma, p_dim) {
  tryCatch(
    chol(Sigma),
    error = function(e) {
      jitter <- max(abs(diag(Sigma))) * 1e-10
      if (jitter < .Machine$double.eps) jitter <- 1e-12
      chol(Sigma + diag(jitter, p_dim))
    }
  )
}


# ==================================================================
# FUNCIÓN COMPLETA: FFBS (FORWARD FILTER + BACKWARD SAMPLE)
# ==================================================================
ffbs <- function(theta_curr, m_filt, C_filt, a_pred, R_pred,
                 Y, F_lst, N_jkt, n_k, m_deptos,
                 Tt_periodos, p_dim, W_diag,
                 s_curr, v_curr, N_kt) {

  # Paso 3: Forward Filter
  ff <- forward_filter(m_filt, C_filt, a_pred, R_pred,
                       Y, F_lst, N_jkt, n_k, m_deptos,
                       Tt_periodos, p_dim, W_diag,
                       s_curr, v_curr, N_kt)

  # Paso 4: Backward Sample
  theta_curr <- backward_sample(theta_curr,
                                ff$m_filt, ff$C_filt,
                                ff$a_pred, ff$R_pred,
                                Tt_periodos, p_dim)

  # Retornar todo lo que el Gibbs necesita
  list(theta_curr = theta_curr,
       m_filt     = ff$m_filt,
       C_filt     = ff$C_filt,
       a_pred     = ff$a_pred,
       R_pred     = ff$R_pred)
}


# ==================================================================
# VERIFICACIÓN NUMÉRICA
# ==================================================================

cat("\n══════ VERIFICACIÓN DEL BACKWARD SAMPLE ══════\n\n")

test_backward_sample <- function() {

  set.seed(123)

  # ── DLM sintético: p = 2, m = 1, n_k = 1, Tt = 3 ──
  p_test  <- 2L
  m_test  <- 1L
  Tt_test <- 3L
  nk_test <- 1L

  # Hiperparámetros
  m0_test <- c(0, 0)
  C0_test <- diag(c(100, 100))
  W_test  <- matrix(c(1, 1), nrow = 2, ncol = 3)

  # Datos sintéticos
  theta_true <- matrix(c(3, -2, 4, -1, 5, 0),
                        nrow = 2, ncol = 3)
  n_per_t <- 20L
  N_jkt_test <- array(0L, dim = c(1, 1, 3))
  N_jkt_test[1, 1, ] <- n_per_t
  N_kt_test <- matrix(n_per_t, nrow = 1, ncol = 3)

  Y_test <- vector("list", Tt_test)
  F_test <- vector("list", Tt_test)
  v_test <- matrix(1.0, nrow = 1, ncol = 1)
  s_test <- matrix(4.0, nrow = 1, ncol = 3)

  set.seed(42)
  for (tt in 1:Tt_test) {
    Y_test[[tt]] <- list(list(NULL))
    F_test[[tt]] <- list(list(NULL))
    F_mat_t <- cbind(1, rnorm(n_per_t))
    y_t <- F_mat_t %*% theta_true[, tt] + rnorm(n_per_t, sd = 2)
    Y_test[[tt]][[1]][[1]] <- drop(y_t)
    F_test[[tt]][[1]][[1]] <- F_mat_t
  }

  # ── Forward filter (una vez) ──
  mf <- matrix(NA, p_test, Tt_test + 1)
  Cf <- array(NA,  c(p_test, p_test, Tt_test + 1))
  ap <- matrix(NA, p_test, Tt_test)
  Rp <- array(NA,  c(p_test, p_test, Tt_test))
  mf[, 1]  <- m0_test
  Cf[,, 1] <- C0_test

  ff <- forward_filter(mf, Cf, ap, Rp,
                       Y_test, F_test, N_jkt_test,
                       nk_test, m_test, Tt_test, p_test,
                       W_test, s_test, v_test, N_kt_test)

  # ── Computar momentos suavizados analíticos ──
  # Estos son E[θ_t | D_T] y Var(θ_t | D_T), contra los cuales
  # se comparan las medias y varianzas muestrales del FFBS.
  m_smooth <- matrix(NA, p_test, Tt_test + 1)
  C_smooth <- array(NA, c(p_test, p_test, Tt_test + 1))

  # Terminal: suavizado = filtrado
  m_smooth[, Tt_test + 1] <- ff$m_filt[, Tt_test + 1]
  C_smooth[,, Tt_test + 1] <- ff$C_filt[,, Tt_test + 1]

  # Recursión hacia atrás
  for (tt in (Tt_test - 1):0) {
    C_t   <- ff$C_filt[,, tt + 1]
    m_t   <- ff$m_filt[, tt + 1]
    R_tp1 <- ff$R_pred[,, tt + 1]
    a_tp1 <- ff$a_pred[, tt + 1]

    B_t <- t(solve(R_tp1, C_t))

    m_smooth[, tt + 1] <- m_t +
      drop(B_t %*% (m_smooth[, tt + 2] - a_tp1))
    C_smooth[,, tt + 1] <- C_t +
      B_t %*% (C_smooth[,, tt + 2] - R_tp1) %*% t(B_t)
  }

  # ── Muestrear muchas trayectorias FFBS ──
  n_rep <- 5000L
  theta_samples <- array(NA, dim = c(p_test, Tt_test + 1, n_rep))
  th_curr <- matrix(0, p_test, Tt_test + 1)

  set.seed(999)
  for (rr in 1:n_rep) {
    th_curr <- backward_sample(th_curr,
                               ff$m_filt, ff$C_filt,
                               ff$a_pred, ff$R_pred,
                               Tt_test, p_test)
    theta_samples[,, rr] <- th_curr
  }

  # ── Test 1: Dimensiones correctas ──
  pass1 <- all(dim(th_curr) == c(p_test, Tt_test + 1))
  cat(sprintf("  Test 1 (dimensiones %dx%d): %s\n",
              p_test, Tt_test + 1,
              ifelse(pass1, "✓ PASS", "✗ FAIL")))

  # ── Test 2: Todas las muestras finitas ──
  pass2 <- all(is.finite(theta_samples))
  cat(sprintf("  Test 2 (muestras finitas): %s\n",
              ifelse(pass2, "✓ PASS", "✗ FAIL")))

  # ── Test 3: Media muestral ≈ media suavizada ──
  # Con 5000 muestras y p = 2, la tolerancia es
  # ~3 * SD / sqrt(5000) ≈ 3 * 10 / 70 ≈ 0.4
  tol_media <- 0.5
  pass3 <- TRUE
  for (tt in 0:Tt_test) {
    media_muestral  <- rowMeans(theta_samples[, tt + 1, ])
    media_analitica <- m_smooth[, tt + 1]
    diff <- max(abs(media_muestral - media_analitica))
    if (diff > tol_media) {
      cat(sprintf("    t=%d: diff = %.3f > %.3f\n",
                  tt, diff, tol_media))
      pass3 <- FALSE
    }
  }
  cat(sprintf("  Test 3 (media muestral ≈ suavizada, tol=%.1f): %s\n",
              tol_media, ifelse(pass3, "✓ PASS", "✗ FAIL")))

  # ── Test 4: Varianza muestral ≈ varianza suavizada ──
  # Comparamos las diagonales de la covarianza.
  tol_var <- 3.0   # tolerancia más amplia para varianzas
  pass4 <- TRUE
  for (tt in 0:Tt_test) {
    for (dd in 1:p_test) {
      var_muestral  <- var(theta_samples[dd, tt + 1, ])
      var_analitica <- C_smooth[dd, dd, tt + 1]
      diff_rel <- abs(var_muestral - var_analitica) / var_analitica
      if (diff_rel > 0.15) {   # 15% de diferencia relativa
        cat(sprintf("    t=%d, d=%d: var_muest=%.3f, var_anal=%.3f (%.1f%%)\n",
                    tt, dd, var_muestral, var_analitica, diff_rel * 100))
        pass4 <- FALSE
      }
    }
  }
  cat(sprintf("  Test 4 (varianza muestral ≈ suavizada, <15%%): %s\n",
              ifelse(pass4, "✓ PASS", "✗ FAIL")))

  # ── Test 5: Función ffbs() integrada funciona ──
  th_curr2 <- matrix(0, p_test, Tt_test + 1)
  mf2 <- mf; Cf2 <- Cf; ap2 <- ap; Rp2 <- Rp
  mf2[, 1] <- m0_test; Cf2[,, 1] <- C0_test

  res_ffbs <- ffbs(th_curr2, mf2, Cf2, ap2, Rp2,
                   Y_test, F_test, N_jkt_test,
                   nk_test, m_test, Tt_test, p_test,
                   W_test, s_test, v_test, N_kt_test)

  pass5 <- all(is.finite(res_ffbs$theta_curr)) &&
            all(dim(res_ffbs$theta_curr) == c(p_test, Tt_test + 1))
  cat(sprintf("  Test 5 (ffbs() integrada): %s\n",
              ifelse(pass5, "✓ PASS", "✗ FAIL")))

  # ── Resumen ──
  all_pass <- pass1 && pass2 && pass3 && pass4 && pass5
  cat(sprintf("\n  %s Verificación del backward sample\n",
              ifelse(all_pass, "✓", "✗")))

  return(all_pass)
}

ok <- test_backward_sample()
if (!ok) warning("El backward sample no pasó todas las verificaciones")



