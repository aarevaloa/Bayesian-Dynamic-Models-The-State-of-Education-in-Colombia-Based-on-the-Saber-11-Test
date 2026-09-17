#_______________________________________________________________________________
#                                                                 
#   MUESTREO DE λ^2_B (ESCALAS GLOBALES LASSO)             

muestrear_lambda2 <- function(lambda2_curr, tau2_curr, bloques,
                              Tt_periodos) {
  
  for (B in names(bloques)) {
    blk <- bloques[[B]]

    # ── Parámetro de forma ──
    # T·p_B contribuciones de las Exp(λ^2/2) + a priori
    shape <- Tt_periodos * blk$dim + blk$a_lambda

    # ── Parámetro de tasa ──
    # Suma de TODAS las escalas locales del bloque B
    sum_tau2 <- sum(tau2_curr[[B]])
    rate <- blk$b_lambda + 0.5 * sum_tau2

    # ── Muestrear ──
    lambda2_curr[B] <- rgamma(1L, shape = shape, rate = rate)
  }

  lambda2_curr
}


# ==================================================================
# VERIFICACIÓN NUMÉRICA
# ==================================================================

cat("\n══════ VERIFICACIÓN DEL MUESTREO DE λ² ══════\n\n")

test_lambda2 <- function() {

  set.seed(777)

  # ── Configuración sintética ──
  Tt_test <- 5L
  p_B_test <- 3L

  bloques_test <- list(
    E = list(nombre = "Test", dim = p_B_test,
             a_lambda = 2.0, b_lambda = 0.5)
  )

  # tau2 conocido: todos = 0.4
  tau2_test <- list(E = matrix(0.4, nrow = p_B_test, ncol = Tt_test))

  # Parámetros teóricos de la condicional:
  #   shape = T*p_B + a = 5*3 + 2 = 17
  #   rate  = b + 0.5 * sum(tau2) = 0.5 + 0.5 * (15 * 0.4) = 0.5 + 3 = 3.5
  #   E[λ²] = shape/rate = 17/3.5 ≈ 4.857
  #   Var[λ²] = shape/rate² = 17/12.25 ≈ 1.388
  shape_theo <- Tt_test * p_B_test + 2.0
  rate_theo  <- 0.5 + 0.5 * sum(tau2_test$E)
  mean_theo  <- shape_theo / rate_theo
  var_theo   <- shape_theo / rate_theo^2

  # Muestrear muchas veces
  n_rep <- 100000L
  samples <- numeric(n_rep)
  lambda2_test <- c(E = 1.0)

  for (rr in 1:n_rep) {
    lambda2_test <- muestrear_lambda2(lambda2_test, tau2_test,
                                      bloques_test, Tt_test)
    samples[rr] <- lambda2_test["E"]
  }

  # ── Test 1: Media muestral ≈ teórica ──
  err_mean <- abs(mean(samples) - mean_theo) / mean_theo
  pass1 <- err_mean < 0.02
  cat(sprintf("  Test 1 (media): %.4f vs %.4f (err=%.3f%%) %s\n",
              mean(samples), mean_theo, err_mean * 100,
              ifelse(pass1, "✓ PASS", "✗ FAIL")))

  # ── Test 2: Varianza muestral ≈ teórica ──
  err_var <- abs(var(samples) - var_theo) / var_theo
  pass2 <- err_var < 0.05
  cat(sprintf("  Test 2 (varianza): %.4f vs %.4f (err=%.1f%%) %s\n",
              var(samples), var_theo, err_var * 100,
              ifelse(pass2, "✓ PASS", "✗ FAIL")))

  # ── Test 3: Todas las muestras positivas ──
  pass3 <- all(samples > 0)
  cat(sprintf("  Test 3 (λ² > 0): %s\n",
              ifelse(pass3, "✓ PASS", "✗ FAIL")))

  # ── Test 4: Autorregulación LASSO ──
  # tau2 grandes → rate grande → λ² más pequeño
  tau2_small <- list(E = matrix(0.01, p_B_test, Tt_test))
  tau2_large <- list(E = matrix(10.0, p_B_test, Tt_test))

  n_rep2 <- 10000L
  samp_small <- samp_large <- numeric(n_rep2)
  l2s <- l2l <- c(E = 1.0)
  for (rr in 1:n_rep2) {
    l2s <- muestrear_lambda2(l2s, tau2_small, bloques_test, Tt_test)
    l2l <- muestrear_lambda2(l2l, tau2_large, bloques_test, Tt_test)
    samp_small[rr] <- l2s["E"]
    samp_large[rr] <- l2l["E"]
  }
  pass4 <- mean(samp_small) > mean(samp_large)
  cat(sprintf("  Test 4 (autorregulación): E[λ²|τ² chico]=%.2f > E[λ²|τ² grande]=%.2f %s\n",
              mean(samp_small), mean(samp_large),
              ifelse(pass4, "✓ PASS", "✗ FAIL")))

  # ── Test 5: Tres bloques simultáneamente ──
  bloques_3 <- list(
    E = list(dim = 5L, a_lambda = 1.0, b_lambda = 0.1),
    M = list(dim = 3L, a_lambda = 1.0, b_lambda = 0.1),
    D = list(dim = 2L, a_lambda = 1.0, b_lambda = 0.1)
  )
  tau2_3 <- list(
    E = matrix(0.2, 5L, Tt_test),
    M = matrix(0.5, 3L, Tt_test),
    D = matrix(1.0, 2L, Tt_test)
  )
  l2_3 <- c(E = 1.0, M = 1.0, D = 1.0)
  l2_3 <- muestrear_lambda2(l2_3, tau2_3, bloques_3, Tt_test)
  pass5 <- length(l2_3) == 3 && all(l2_3 > 0) &&
            all(names(l2_3) == c("E", "M", "D"))
  cat(sprintf("  Test 5 (3 bloques): E=%.2f M=%.2f D=%.2f %s\n",
              l2_3["E"], l2_3["M"], l2_3["D"],
              ifelse(pass5, "✓ PASS", "✗ FAIL")))

  # ── Resumen ──
  all_pass <- pass1 && pass2 && pass3 && pass4 && pass5
  cat(sprintf("\n  %s Verificación de muestreo λ²\n",
              ifelse(all_pass, "✓", "✗")))
  return(all_pass)
}

ok <- test_lambda2()
if (!ok) warning("El muestreo de lambda2 no pasó todas las verificaciones")
