#_______________________________________________________________________________
#                                                                
#   MUESTREO DE φ_{k,t} = 1/s_{k,t}                       


muestrear_phi <- function(phi_curr, s_curr, SSE_jkt, v_curr,
                          N_kt, n_k, m_deptos, Tt_periodos,
                          beta_discount, n_k0, d_k0,
                          n_kt_work, d_kt_work) {

  n_phi_fix <- 0L
  for (kk in 1:m_deptos) {

    beta_k <- beta_discount[kk]

    # ── Precomputar Σ_j v_{j,k} SSE_{j,k,t} para cada t ──
    # wSSE[t] = Σ_{j=1}^{n_k[k]} v_{j,k} * SSE_{j,k,t}
    #
    # Vectorización: v_{j,k} es un vector de longitud n_k[k],
    # SSE_jkt[1:n_k[k], kk, ] es una matriz n_k[k] × Tt.
    # El producto matricial v' SSE da un vector de longitud Tt.
    nk_kk <- n_k[kk]
    v_k   <- v_curr[1:nk_kk, kk]                       # n_k × 1
    SSE_k <- SSE_jkt[1:nk_kk, kk, , drop = FALSE]      # n_k × Tt

    # Convertir a matriz 2D si n_k = 1 (drop puede colapsar)
    dim(SSE_k) <- c(nk_kk, Tt_periodos)

    wSSE <- drop(crossprod(v_k, SSE_k))                 # 1 × Tt → vector

    # ── Recursión secuencial t = 1, ..., T ──
    n_prev <- n_k0[kk]       # n_{k,0}
    d_prev <- d_k0[kk]       # d_{k,0}

    for (tt in 1:Tt_periodos) {

      # Hiperparámetros descontados
      bn <- beta_k * n_prev    # β_k n_{k,t-1}
      bd <- beta_k * d_prev    # β_k d_{k,t-1}

      # Actualización
      n_kt <- bn + N_kt[kk, tt]      # n_{k,t} = β_k n_{k,t-1} + N_{k,t}
      d_kt <- bd + wSSE[tt]          # d_{k,t} = β_k d_{k,t-1} + Σ_j v SSE

      # Muestrear φ_{k,t} ~ G(n_{k,t}/2, d_{k,t}/2)
      shape <- n_kt / 2.0
      rate  <- d_kt / 2.0

      if (shape <= 0 || rate <= 0) {

        warning(sprintf("phi[%d,%d]: shape=%.4f, rate=%.4f. Usando prior.",
                        kk, tt, shape, rate))
        shape <- n_k0[kk] / 2.0
        rate  <- d_k0[kk] / 2.0
      }

      phi_kk_tt <- rgamma(1L, shape = shape, rate = rate)

      # Protección contra phi = 0 (underflow) o Inf (overflow)
      if (!is.finite(phi_kk_tt) || phi_kk_tt <= 0) {
        n_phi_fix <- n_phi_fix + 1L
        phi_kk_tt <- n_kt / d_kt
      }

      phi_curr[kk, tt] <- phi_kk_tt
      s_curr[kk, tt]   <- 1.0 / phi_kk_tt

      # Almacenar hiperparámetros (para diagnósticos)
      n_kt_work[kk, tt] <- n_kt
      d_kt_work[kk, tt] <- d_kt

      # Avanzar: los hiperparámetros de este periodo son
      # los "previos" del siguiente
      n_prev <- n_kt
      d_prev <- d_kt
    }
  }
  
  if (n_phi_fix > 0L) {
    cat(sprintf("phi fixes esta iteración: %d\n", n_phi_fix))
  }

  list(phi_curr  = phi_curr,
       s_curr    = s_curr,
       n_kt_work = n_kt_work,
       d_kt_work = d_kt_work)
}


# ==================================================================
# VERIFICACIÓN NUMÉRICA
# ==================================================================

test_phi <- function() {

  set.seed(555)

  # ── Configuración sintética ──
  m_test  <- 2L
  Tt_test <- 4L
  nk_test <- c(2L, 1L)
  max_nk  <- max(nk_test)

  beta_test <- c(1.0, 0.95)
  n_k0_test <- c(4.0, 6.0)
  d_k0_test <- c(8.0, 12.0)

  N_kt_test <- matrix(0L, nrow = m_test, ncol = Tt_test)
  N_kt_test[1, ] <- c(10L, 20L, 15L, 5L)
  N_kt_test[2, ] <- c(8L, 8L, 8L, 8L)

  # SSE y v fijos para control
  SSE_test <- array(0, dim = c(max_nk, m_test, Tt_test))
  SSE_test[1, 1, ] <- c(20, 40, 25, 10)
  SSE_test[2, 1, ] <- c(30, 60, 35, 20)
  SSE_test[1, 2, ] <- c(40, 40, 40, 40)

  v_test <- matrix(NA_real_, nrow = max_nk, ncol = m_test)
  v_test[1, 1] <- 1.0
  v_test[2, 1] <- 1.0
  v_test[1, 2] <- 1.0

  wSSE_expected <- c(50, 100, 60, 30)

  # Recursión teórica para k=1 (beta=1):
  n_theo <- numeric(Tt_test)
  d_theo <- numeric(Tt_test)
  n_p <- n_k0_test[1]; d_p <- d_k0_test[1]
  for (tt in 1:Tt_test) {
    n_theo[tt] <- 1.0 * n_p + N_kt_test[1, tt]
    d_theo[tt] <- 1.0 * d_p + wSSE_expected[tt]
    n_p <- n_theo[tt]; d_p <- d_theo[tt]
  }

  # Muestrear y verificar hiperparámetros
  phi_test <- matrix(1.0, nrow = m_test, ncol = Tt_test)
  s_test   <- matrix(1.0, nrow = m_test, ncol = Tt_test)
  nw_test  <- matrix(NA, nrow = m_test, ncol = Tt_test)
  dw_test  <- matrix(NA, nrow = m_test, ncol = Tt_test)

  res <- muestrear_phi(phi_test, s_test, SSE_test, v_test,
                       N_kt_test, nk_test, m_test, Tt_test,
                       beta_test, n_k0_test, d_k0_test,
                       nw_test, dw_test)

  # Verificar n_{k,t} y d_{k,t} para k=1
  pass1 <- max(abs(res$n_kt_work[1, ] - n_theo)) < 1e-10 &&
           max(abs(res$d_kt_work[1, ] - d_theo)) < 1e-10
  cat(sprintf("  Test 1 (recursión n,d para k=1):\n"))
  cat(sprintf("    n: [%s] vs [%s]\n",
              paste(round(res$n_kt_work[1,], 1), collapse=", "),
              paste(round(n_theo, 1), collapse=", ")))
  cat(sprintf("    d: [%s] vs [%s]\n",
              paste(round(res$d_kt_work[1,], 1), collapse=", "),
              paste(round(d_theo, 1), collapse=", ")))
  cat(sprintf("    %s\n", ifelse(pass1, "✓ PASS", "✗ FAIL")))

  # ═══════════════════════════════════════════
  # Test 2: Descuento beta < 1 reduce n y d
  # ═══════════════════════════════════════════
  # Para k=2 con beta=0.95:
  # t=1: n = 0.95*6 + 8 = 13.7, d = 0.95*12 + 40 = 51.4
  # t=2: n = 0.95*13.7 + 8 = 21.015, d = 0.95*51.4 + 40 = 88.83
  n_theo2 <- numeric(Tt_test)
  d_theo2 <- numeric(Tt_test)
  n_p2 <- n_k0_test[2]; d_p2 <- d_k0_test[2]
  for (tt in 1:Tt_test) {
    n_theo2[tt] <- 0.95 * n_p2 + N_kt_test[2, tt]
    d_theo2[tt] <- 0.95 * d_p2 + 1.0 * SSE_test[1, 2, tt]  # v=1, 1 muni
    n_p2 <- n_theo2[tt]; d_p2 <- d_theo2[tt]
  }

  pass2 <- max(abs(res$n_kt_work[2, ] - n_theo2)) < 1e-10 &&
           max(abs(res$d_kt_work[2, ] - d_theo2)) < 1e-10
  cat(sprintf("  Test 2 (recursión con β=0.95 para k=2): %s\n",
              ifelse(pass2, "✓ PASS", "✗ FAIL")))

  # ═══════════════════════════════════════════
  # Test 3: Momentos muestrales ≈ teóricos
  # ═══════════════════════════════════════════
  # E[φ_{k,t}] = (n_{k,t}/2) / (d_{k,t}/2) = n/d
  # Para k=1, t=1: E[φ] = 14/58 ≈ 0.2414
  mean_theo_11 <- n_theo[1] / d_theo[1]

  n_rep <- 50000L
  phi_samples <- numeric(n_rep)
  for (rr in 1:n_rep) {
    phi_t <- matrix(1.0, m_test, Tt_test)
    s_t   <- matrix(1.0, m_test, Tt_test)
    nw_t  <- matrix(NA, m_test, Tt_test)
    dw_t  <- matrix(NA, m_test, Tt_test)
    r <- muestrear_phi(phi_t, s_t, SSE_test, v_test,
                       N_kt_test, nk_test, m_test, Tt_test,
                       beta_test, n_k0_test, d_k0_test,
                       nw_t, dw_t)
    phi_samples[rr] <- r$phi_curr[1, 1]
  }

  err_mean <- abs(mean(phi_samples) - mean_theo_11) / mean_theo_11
  pass3 <- err_mean < 0.02
  cat(sprintf("  Test 3 (E[φ_{1,1}]): %.5f vs %.5f (err=%.2f%%) %s\n",
              mean(phi_samples), mean_theo_11, err_mean * 100,
              ifelse(pass3, "✓ PASS", "✗ FAIL")))

  # ═══════════════════════════════════════════
  # Test 4: φ·s = 1 (reciprocidad)
  # ═══════════════════════════════════════════
  pass4 <- max(abs(res$phi_curr * res$s_curr - 1)) < 1e-10
  cat(sprintf("  Test 4 (φ·s = 1): %s\n",
              ifelse(pass4, "✓ PASS", "✗ FAIL")))

  # ═══════════════════════════════════════════
  # Test 5: Todas φ > 0 y finitas
  # ═══════════════════════════════════════════
  pass5 <- all(res$phi_curr > 0) && all(is.finite(res$phi_curr)) &&
           all(res$s_curr > 0)   && all(is.finite(res$s_curr))
  cat(sprintf("  Test 5 (φ,s > 0 y finitos): %s\n",
              ifelse(pass5, "✓ PASS", "✗ FAIL")))

  # ═══════════════════════════════════════════
  # Test 6: Dimensiones
  # ═══════════════════════════════════════════
  pass6 <- all(dim(res$phi_curr) == c(m_test, Tt_test)) &&
           all(dim(res$s_curr) == c(m_test, Tt_test)) &&
           all(dim(res$n_kt_work) == c(m_test, Tt_test)) &&
           all(dim(res$d_kt_work) == c(m_test, Tt_test))
  cat(sprintf("  Test 6 (dimensiones m×Tt): %s\n",
              ifelse(pass6, "✓ PASS", "✗ FAIL")))

  # ═══════════════════════════════════════════
  # Test 7: n_{k,t} es creciente (con beta=1)
  # ═══════════════════════════════════════════
  # Con beta=1, n_{k,t} solo crece (acumula grados de libertad)
  pass7 <- all(diff(res$n_kt_work[1, ]) > 0)
  cat(sprintf("  Test 7 (n_{1,t} creciente con β=1): %s\n",
              ifelse(pass7, "✓ PASS", "✗ FAIL")))

  # ── Resumen ──
  all_pass <- pass1 && pass2 && pass3 && pass4 && pass5 && pass6 && pass7
  cat(sprintf("\n  %s Verificación de muestreo φ\n",
              ifelse(all_pass, "✓", "✗")))
  return(all_pass)
}

ok <- test_phi()
if (!ok) warning("El muestreo de phi no pasó todas las verificaciones")


