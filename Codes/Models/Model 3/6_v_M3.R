#_______________________________________________________________________________
#                                                                 
#   CÁLCULO DE SSE Y MUESTREO DE v_{j,k}                  
#                                                               

# ==================================================================
# PARTE (A): CÁLCULO DE SSE_{j,k,t}
# ==================================================================

calcular_SSE <- function(SSE_jkt, Y, F_lst, theta_curr,
                         N_jkt, n_k, m_deptos, Tt_periodos) {

  for (tt in 1:Tt_periodos) {
    theta_t <- theta_curr[, tt + 1]     # θ_t (offset +1)

    for (kk in 1:m_deptos) {
      for (jj in 1:n_k[kk]) {

        n_obs <- N_jkt[jj, kk, tt]
        if (n_obs == 0L) {
          SSE_jkt[jj, kk, tt] <- 0
          next
        }

        # Residuos: y_{i,j,k,t} − F'_{i,j,k,t} θ_t
        # F_jkt es n_obs × p; theta_t es p × 1
        # El producto F_jkt %*% theta_t da n_obs × 1
        y_jkt <- Y[[tt]][[kk]][[jj]]
        F_jkt <- F_lst[[tt]][[kk]][[jj]]
        resid <- y_jkt - drop(F_jkt %*% theta_t)

        SSE_jkt[jj, kk, tt] <- sum(resid * resid)
      }
    }
  }

  SSE_jkt
}


# ==================================================================
# PARTE (B): MUESTREO DE v_{j,k}
# ==================================================================

muestrear_v <- function(v_curr, SSE_jkt, s_curr, N_jk,
                        n_k, m_deptos, Tt_periodos, a_v, b_v) {

  for (kk in 1:m_deptos) {

    inv_s_k <- 1.0 / pmax(s_curr[kk, ], .Machine$double.eps)    # longitud Tt

    for (jj in 1:n_k[kk]) {

      shape <- a_v + N_jk[jj, kk] / 2.0

      sum_SSE_over_s <- sum(SSE_jkt[jj, kk, ] * inv_s_k)
      rate <- b_v + 0.5 * sum_SSE_over_s
      
      if (!is.finite(rate) || rate <= 0) {
        warning(sprintf("rate inválido en v[%d,%d]: %.2e. Usando prior.", jj, kk, rate))
        rate <- b_v
        shape <- a_v
      }

      # ── Muestrear ──
      v_curr[jj, kk] <- rgamma(1L, shape = shape, rate = rate)
    }
  }

  v_curr
}


# ==================================================================
# VERIFICACIÓN NUMÉRICA
# ==================================================================

cat("\n══════ VERIFICACIÓN DE SSE Y MUESTREO DE v ══════\n\n")

test_SSE_y_v <- function() {

  set.seed(456)

  # ── DLM sintético: p = 2, m = 2 deptos, n_k = [2, 1], Tt = 3 ──
  p_test  <- 2L
  m_test  <- 2L
  Tt_test <- 3L
  nk_test <- c(2L, 1L)
  max_nk  <- max(nk_test)

  # Trayectoria de estados conocida
  theta_test <- matrix(0, nrow = p_test, ncol = Tt_test + 1)
  theta_test[1, ] <- c(0, 10, 10, 10)   # intercepto
  theta_test[2, ] <- c(0, 2, 2, 2)      # pendiente

  # Datos sintéticos
  N_jkt_test <- array(0L, dim = c(max_nk, m_test, Tt_test))
  N_jkt_test[1, 1, ] <- c(5L, 8L, 6L)   # muni (1,1): 19 obs total
  N_jkt_test[2, 1, ] <- c(3L, 0L, 4L)   # muni (2,1): 7 obs total
  N_jkt_test[1, 2, ] <- c(10L, 10L, 10L) # muni (1,2): 30 obs total

  N_jk_test <- matrix(0L, nrow = max_nk, ncol = m_test)
  N_jk_test[1, 1] <- 19L
  N_jk_test[2, 1] <- 7L
  N_jk_test[1, 2] <- 30L

  Y_test     <- vector("list", Tt_test)
  F_test     <- vector("list", Tt_test)

  # Varianza real por municipio: v_true = precisión
  sigma2_true <- c(4, 9, 1)  # var para (1,1), (2,1), (1,2)

  for (tt in 1:Tt_test) {
    Y_test[[tt]] <- vector("list", m_test)
    F_test[[tt]] <- vector("list", m_test)
    for (kk in 1:m_test) {
      Y_test[[tt]][[kk]] <- vector("list", nk_test[kk])
      F_test[[tt]][[kk]] <- vector("list", nk_test[kk])
    }
  }

  # Generar datos
  set.seed(100)
  muni_idx <- 0L
  for (kk in 1:m_test) {
    for (jj in 1:nk_test[kk]) {
      muni_idx <- muni_idx + 1L
      for (tt in 1:Tt_test) {
        nn <- N_jkt_test[jj, kk, tt]
        if (nn == 0L) next
        F_mat <- cbind(1, rnorm(nn))
        theta_t <- theta_test[, tt + 1]
        y <- drop(F_mat %*% theta_t) +
             rnorm(nn, sd = sqrt(sigma2_true[muni_idx]))
        Y_test[[tt]][[kk]][[jj]] <- y
        F_test[[tt]][[kk]][[jj]] <- F_mat
      }
    }
  }

  # ═══════════════════════════════════════════
  # Test 1: SSE se calcula correctamente
  # ═══════════════════════════════════════════
  SSE_test <- array(0, dim = c(max_nk, m_test, Tt_test))
  SSE_test <- calcular_SSE(SSE_test, Y_test, F_test, theta_test,
                           N_jkt_test, nk_test, m_test, Tt_test)

  # Verificar manualmente para celda (1,1,1): 5 observaciones
  y_111 <- Y_test[[1]][[1]][[1]]
  F_111 <- F_test[[1]][[1]][[1]]
  theta_1 <- theta_test[, 2]
  resid_111 <- y_111 - drop(F_111 %*% theta_1)
  SSE_manual <- sum(resid_111^2)

  pass1 <- abs(SSE_test[1, 1, 1] - SSE_manual) < 1e-10
  cat(sprintf("  Test 1 (SSE manual): %.4f vs %.4f %s\n",
              SSE_test[1, 1, 1], SSE_manual,
              ifelse(pass1, "✓ PASS", "✗ FAIL")))

  # ═══════════════════════════════════════════
  # Test 2: SSE = 0 para celdas vacías
  # ═══════════════════════════════════════════
  pass2 <- SSE_test[2, 1, 2] == 0     # muni (2,1) en t=2: 0 obs
  cat(sprintf("  Test 2 (SSE=0 celda vacía): %s\n",
              ifelse(pass2, "✓ PASS", "✗ FAIL")))

  # ═══════════════════════════════════════════
  # Test 3: v muestreado tiene momentos correctos
  # ═══════════════════════════════════════════
  # Fijar s_{k,t} = 1 para simplificar: kappa² = s/v = 1/v
  s_test <- matrix(1.0, nrow = m_test, ncol = Tt_test)
  a_v_test <- 1.0
  b_v_test <- 1.0

  # Parámetros teóricos de la Gamma para muni (1,1):
  #   shape = a_v + N_{1,1}/2 = 1 + 19/2 = 10.5
  #   rate = b_v + 0.5 * Σ_t SSE_{1,1,t}/s_{1,t}
  #        = 1 + 0.5 * (SSE[1,1,1] + SSE[1,1,2] + SSE[1,1,3])
  shape_11 <- a_v_test + N_jk_test[1, 1] / 2.0
  sum_SSE_11 <- sum(SSE_test[1, 1, ])
  rate_11 <- b_v_test + 0.5 * sum_SSE_11
  mean_theo_11 <- shape_11 / rate_11

  # Muestrear muchas veces
  n_rep <- 50000L
  v_samples_11 <- numeric(n_rep)
  v_test <- matrix(1.0, nrow = max_nk, ncol = m_test)

  for (rr in 1:n_rep) {
    v_test <- muestrear_v(v_test, SSE_test, s_test, N_jk_test,
                          nk_test, m_test, Tt_test,
                          a_v_test, b_v_test)
    v_samples_11[rr] <- v_test[1, 1]
  }

  err_mean <- abs(mean(v_samples_11) - mean_theo_11) / mean_theo_11
  pass3 <- err_mean < 0.02
  cat(sprintf("  Test 3 (E[v_{1,1}]): %.4f vs %.4f (err=%.2f%%) %s\n",
              mean(v_samples_11), mean_theo_11, err_mean * 100,
              ifelse(pass3, "✓ PASS", "✗ FAIL")))

  # ═══════════════════════════════════════════
  # Test 4: v positivo y finito en todas las entradas válidas
  # ═══════════════════════════════════════════
  pass4 <- TRUE
  for (kk in 1:m_test) {
    vals <- v_test[1:nk_test[kk], kk]
    if (!all(vals > 0) || !all(is.finite(vals))) pass4 <- FALSE
  }
  # Entradas inválidas deben seguir como NA (no tocadas)
  if (max_nk > max(nk_test)) {
    # Hay filas que no corresponden a ningún municipio en algún depto
    for (kk in 1:m_test) {
      if (nk_test[kk] < max_nk) {
        if (!all(is.na(v_test[(nk_test[kk]+1):max_nk, kk]))) pass4 <- FALSE
      }
    }
  }
  cat(sprintf("  Test 4 (v>0 válidos, NA inválidos): %s\n",
              ifelse(pass4, "✓ PASS", "✗ FAIL")))

  # ═══════════════════════════════════════════
  # Test 5: Municipio con σ² baja → v alta (más preciso)
  # ═══════════════════════════════════════════
  # muni (1,2) tiene σ² = 1 (más preciso)
  # muni (1,1) tiene σ² = 4 (menos preciso)
  # Con datos suficientes, E[v_{1,2}] > E[v_{1,1}]
  v_samples_12 <- numeric(n_rep)
  v_test2 <- matrix(1.0, nrow = max_nk, ncol = m_test)
  for (rr in 1:n_rep) {
    v_test2 <- muestrear_v(v_test2, SSE_test, s_test, N_jk_test,
                           nk_test, m_test, Tt_test,
                           a_v_test, b_v_test)
    v_samples_12[rr] <- v_test2[1, 2]
  }
  pass5 <- mean(v_samples_12) > mean(v_samples_11)
  cat(sprintf("  Test 5 (v preciso > v disperso): E[v_{1,2}]=%.3f > E[v_{1,1}]=%.3f %s\n",
              mean(v_samples_12), mean(v_samples_11),
              ifelse(pass5, "✓ PASS", "✗ FAIL")))

  # ═══════════════════════════════════════════
  # Test 6: Dimensiones de SSE_jkt
  # ═══════════════════════════════════════════
  pass6 <- all(dim(SSE_test) == c(max_nk, m_test, Tt_test))
  cat(sprintf("  Test 6 (dim SSE): %s\n",
              ifelse(pass6, "✓ PASS", "✗ FAIL")))

  # ── Resumen ──
  all_pass <- pass1 && pass2 && pass3 && pass4 && pass5 && pass6
  cat(sprintf("\n  %s Verificación de SSE y muestreo v\n",
              ifelse(all_pass, "✓", "✗")))
  return(all_pass)
}

ok <- test_SSE_y_v()
if (!ok) warning("El cálculo de SSE / muestreo de v no pasó las verificaciones")
