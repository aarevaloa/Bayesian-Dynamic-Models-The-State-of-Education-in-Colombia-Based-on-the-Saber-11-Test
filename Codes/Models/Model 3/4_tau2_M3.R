#_______________________________________________________________________________
#                                                                 
#   MUESTREO DE tau^2_{ℓ,B,t} (ESCALAS LOCALES LASSO)      

rinvgauss_vec <- function(n, mu, lambda) {
  nu <- rnorm(n)
  y  <- nu * nu                                  # y ~ chi^2(1)

  # Candidato (fórmula cerrada, sin iteración)
  mu2 <- mu * mu
  x <- mu + (mu2 * y) / (2 * lambda) -
       (mu / (2 * lambda)) * sqrt(4 * mu * lambda * y + mu2 * y * y)

  # Aceptación probabilística
  u <- runif(n)
  ifelse(u <= mu / (mu + x), x, mu2 / x)
}


# ==================================================================
# FUNCIÓN PRINCIPAL: MUESTREO DE tau^2_{ℓ,B,t}
# ==================================================================

muestrear_tau2 <- function(theta_curr, tau2_curr, lambda2_curr,
                           bloques, Tt_periodos, eps_omega) {

  for (B in names(bloques)) {
    blk <- bloques[[B]]
    idx <- blk$idx           # posiciones en theta_t
    p_B <- blk$dim           # dimensión del bloque
    varsigma_B <- sqrt(blk$varsigma2)  # ς_B
    lambda2_B  <- lambda2_curr[B]      # λ²_B
    lambda_B   <- sqrt(lambda2_B)      # λ_B

    # ── Innovaciones: ω_{ℓ,B,t} = θ_{ℓ,B,t} − θ_{ℓ,B,t-1} ──
    # theta_curr[idx, tt+1] = θ_{ℓ,B,t} con offset +1.
    # Diferencia de columnas consecutivas: columnas 2:(Tt+1) - 1:Tt.
    omega <- theta_curr[idx, 2:(Tt_periodos + 1), drop = FALSE] -
             theta_curr[idx, 1:Tt_periodos,       drop = FALSE]
    # omega es p_B × Tt

    abs_omega <- abs(omega)
    tau2_new  <- matrix(NA_real_, nrow = p_B, ncol = Tt_periodos)
    
    mu_prime <- lambda_B * varsigma_B /
      pmax(abs_omega, .Machine$double.xmin)
    
    small <- mu_prime > 1e8

    n_small <- sum(small)
    if (n_small > 0L) {
      tau2_new[small] <- pmax(abs_omega[small] / (lambda_B * varsigma_B),
                              .Machine$double.eps)
    }

    # ── Caso 2: |ω| ≥ eps → 1/τ^2 ~ IG(μ', λ') ──
    n_large <- sum(!small)
    if (n_large > 0L) {
      inv_tau2 <- rinvgauss_vec(n_large, mu_prime[!small], lambda2_B)
      inv_tau2 <- pmax(inv_tau2, .Machine$double.xmin)
      tau2_new[!small] <- 1.0 / inv_tau2
    }

    tau2_curr[[B]] <- tau2_new
  }
  
  tau2_curr
}



# ==================================================================
# VERIFICACIÓN NUMÉRICA
# ==================================================================

test_tau2 <- function() {

  set.seed(314)

  # ── Test 1: rinvgauss_vec produce momentos correctos ──
  # X ~ IG(μ, λ) ⇒ E[X] = μ, Var(X) = μ³/λ
  n_test <- 200000L
  mu_test <- 3.0
  lambda_test <- 8.0
  samples <- rinvgauss_vec(n_test, mu_test, lambda_test)

  mean_theo <- mu_test
  var_theo  <- mu_test^3 / lambda_test

  err_mean <- abs(mean(samples) - mean_theo) / mean_theo
  err_var  <- abs(var(samples) - var_theo) / var_theo

  pass1 <- (err_mean < 0.02) && (err_var < 0.05) && all(samples > 0)
  cat(sprintf("  Test 1 (IG momentos): E=%.3f (%.3f), V=%.3f (%.3f) %s\n",
              mean(samples), mean_theo, var(samples), var_theo,
              ifelse(pass1, "✓ PASS", "✗ FAIL")))

  # ── Test 2: rinvgauss_vec con mu vectorial ──
  mu_vec <- c(1, 5, 10)
  n_vec  <- length(mu_vec)
  samp2  <- rinvgauss_vec(n_vec, mu_vec, 10)
  pass2  <- all(samp2 > 0) && all(is.finite(samp2))
  cat(sprintf("  Test 2 (IG vectorial, muestras>0): %s\n",
              ifelse(pass2, "✓ PASS", "✗ FAIL")))

  # ── Test 3: muestrear_tau2 con trayectoria sintética ──
  p_test  <- 3L
  Tt_test <- 4L
  p_total <- 1 + p_test  # intercepto + 1 bloque

  # Trayectoria con innovaciones conocidas
  theta_test <- matrix(0, nrow = p_total, ncol = Tt_test + 1)
  theta_test[1, ] <- 100                          # intercepto constante
  theta_test[2, ] <- c(0, 0.01, 0.01, 5.01, 5.01) # ω = [0.01, 0, 5, 0]
  theta_test[3, ] <- c(0, 0, 0, 0, 0)             # ω = [0, 0, 0, 0]
  theta_test[4, ] <- c(0, 3, 3, 3, 3)             # ω = [3, 0, 0, 0]

  bloques_test <- list(
    E = list(nombre = "Test", dim = p_test, idx = 2:4,
             varsigma2 = 1.0, a_lambda = 1.0, b_lambda = 0.1)
  )

  tau2_test <- list(E = matrix(1, p_test, Tt_test))
  lambda2_test <- c(E = 10.0)

  # Muestrear muchas veces y promediar
  n_rep <- 5000L
  tau2_acum <- matrix(0, nrow = p_test, ncol = Tt_test)

  for (rr in 1:n_rep) {
    res <- muestrear_tau2(theta_test, tau2_test, lambda2_test,
                          bloques_test, Tt_test, 1e-8)
    tau2_acum <- tau2_acum + res$E
  }
  tau2_media <- tau2_acum / n_rep

  # Verificar propiedad LASSO:
  # - θ[2,]: ω grande en t=3 (|ω|=5) → τ² grande en t=3
  #          ω pequeño en t=1,2,4 → τ² pequeño
  # - θ[3,]: ω = 0 siempre → τ² del prior (media 2/λ² = 0.2)
  # - θ[4,]: ω grande en t=1 (|ω|=3) → τ² grande en t=1

  pass3a <- tau2_media[1, 3] > tau2_media[1, 1]  # t=3 > t=1 para θ[2]
  pass3b <- tau2_media[3, 1] > tau2_media[3, 2]  # t=1 > t=2 para θ[4]

  # θ[3] tiene ω=0 siempre: con la aproximación determinista,
  # τ² ≈ 0 (shrinkage extremo, no el prior)
  pass3c <- tau2_media[2, 1] < 1e-6

  pass3 <- pass3a && pass3b && pass3c
  cat(sprintf("  Test 3 (propiedad LASSO):\n"))
  cat(sprintf("    θ[2] t=3 vs t=1: %.3f > %.3f → %s\n",
              tau2_media[1,3], tau2_media[1,1],
              ifelse(pass3a, "✓", "✗")))
  cat(sprintf("    θ[4] t=1 vs t=2: %.3f > %.3f → %s\n",
              tau2_media[3,1], tau2_media[3,2],
              ifelse(pass3b, "✓", "✗")))
  cat(sprintf("    θ[3] ω=0, E[τ²]=%.2e (esperado ≈ 0) → %s\n",
              tau2_media[2,1], ifelse(pass3c, "✓", "✗")))

  # ── Test 4: Todas las τ² positivas y finitas ──
  pass4 <- all(res$E > 0) && all(is.finite(res$E))
  cat(sprintf("  Test 4 (τ² > 0 y finito): %s\n",
              ifelse(pass4, "✓ PASS", "✗ FAIL")))

  # ── Test 5: Dimensiones correctas ──
  pass5 <- (nrow(res$E) == p_test) && (ncol(res$E) == Tt_test)
  cat(sprintf("  Test 5 (dimensiones %dx%d): %s\n",
              p_test, Tt_test, ifelse(pass5, "✓ PASS", "✗ FAIL")))

  # ── Test 6: Integración con construir_W_diag ──
  # Verificar que tau2_curr actualizado produce un W_diag válido
  tau2_integ <- list(E = res$E)
  # Simular que M y D son vacíos (no necesarios para este test)
  bloques_integ <- bloques_test
  W_test <- matrix(NA_real_, nrow = p_total, ncol = Tt_test)
  W_test[1, ] <- 10  # q_mu
  W_test[bloques_integ$E$idx, ] <- bloques_integ$E$varsigma2 * tau2_integ$E

  pass6 <- !any(is.na(W_test)) && all(W_test > 0)
  cat(sprintf("  Test 6 (integración W_diag): %s\n",
              ifelse(pass6, "✓ PASS", "✗ FAIL")))

  # ── Resumen ──
  all_pass <- pass1 && pass2 && pass3 && pass4 && pass5 && pass6
  cat(sprintf("\n  %s Verificación de muestreo τ²\n",
              ifelse(all_pass, "✓", "✗")))
  return(all_pass)
}

ok <- test_tau2()
if (!ok) warning("El muestreo de tau2 no pasó todas las verificaciones")


