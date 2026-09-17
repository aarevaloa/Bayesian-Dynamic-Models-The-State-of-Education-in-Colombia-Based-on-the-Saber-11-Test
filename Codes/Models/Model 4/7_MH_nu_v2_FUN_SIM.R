#_______________________________________________________________________________
#@  MODELO 4 - MODELO DINÁMICO BAYESIANO t-STUDENT ROBUSTO
#@  CONDICIONAL COMPLETA DE LOS GRADOS DE LIBERTAD nu
#@           (paso de Metropolis-Hastings dentro del Gibbs)
#_______________________________________________________________________________
# ==============================================================================
#  1. ESTADÍSTICOS SUFICIENTES DE LAS ESCALAS LATENTES
# ==============================================================================
estadisticos_nu <- function(xi_list) {
  Tt <- length(xi_list)
  stopifnot(Tt >= 1L)
  P <- 0.0; Q <- 0.0; N <- 0L
  for (tt in 1:Tt) {
    xi_t <- xi_list[[tt]]
    if (length(xi_t) == 0L) next
    stopifnot(all(xi_t > 0))         # log xi exige xi > 0 (piso de la Etapa 3)
    P <- P + sum(log(xi_t))
    Q <- Q + sum(xi_t)
    N <- N + length(xi_t)
  }
  stopifnot(N >= 1L)
  list(P = P, Q = Q, N = N)
}

# ==============================================================================
#  2. LOG-DENSIDAD DE LA CONDICIONAL DE nu
# ==============================================================================
log_post_nu <- function(nu, P, Q, N, lambda_nu) {
  if (nu <= 2) return(-Inf)
  a <- nu / 2
  N * (a * log(a) - lgamma(a)) + a * (P - Q) - lambda_nu * nu
}


# ==============================================================================
#  3. GRADIENTE DE LA LOG-DENSIDAD
# ==============================================================================
grad_log_post_nu <- function(nu, P, Q, N, lambda_nu) {
  if (nu <= 2) return(NA_real_)
  a <- nu / 2
  (N / 2) * (log(a) + 1 - digamma(a)) + (P - Q) / 2 - lambda_nu
}

# ==============================================================================
#  4. LOG-DENSIDAD EN LA REPARAMETRIZACIÓN  rho = log(nu - 2)
# ==============================================================================

log_post_rho <- function(rho, P, Q, N, lambda_nu) {
  nu <- 2 + exp(rho)
  # log(nu - 2) = log(exp(rho)) = rho  (estable, evita log de un número pequeño)
  log_post_nu(nu, P, Q, N, lambda_nu) + rho
}


# ==============================================================================
#  5. PASO DE METROPOLIS-HASTINGS PARA nu
# ==============================================================================
muestrear_nu <- function(nu_actual, xi_list = NULL, lambda_nu = 0.1,
                         sigma_MH = 0.2, suf = NULL) {
  stopifnot(nu_actual > 2, sigma_MH > 0, lambda_nu > 0)
  if (is.null(suf)) {
    stopifnot(!is.null(xi_list))
    suf <- estadisticos_nu(xi_list)
  }
  P <- suf$P; Q <- suf$Q; N <- suf$N
  
  rho_act  <- log(nu_actual - 2)
  rho_prop <- rho_act + rnorm(1, 0, sigma_MH)         # propuesta RW simétrica
  nu_prop  <- 2 + exp(rho_prop)
  
  l_act  <- log_post_rho(rho_act,  P, Q, N, lambda_nu)
  l_prop <- log_post_rho(rho_prop, P, Q, N, lambda_nu)
  
  log_alpha <- l_prop - l_act
  alpha <- min(1, exp(min(log_alpha, 0)))             # estable: exp de <= 0
  
  if (is.finite(log_alpha) && log(runif(1)) < log_alpha) {
    list(nu = nu_prop, aceptado = TRUE,  alpha = alpha)
  } else {
    list(nu = nu_actual, aceptado = FALSE, alpha = alpha)
  }
}


# ==============================================================================
#  6. ADAPTACIÓN DE sigma_MH DURANTE EL BURN-IN
# ==============================================================================
adaptar_sigma_MH <- function(sigma_MH, tasa_obs, objetivo = 0.35,
                             factor = 0.5) {
  # ajuste multiplicativo suave: sube si se acepta de más, baja si de menos
  log_sig <- log(sigma_MH) + factor * (tasa_obs - objetivo)
  exp(log_sig)
}


# ==============================================================================
#  7. VERIFICACIÓN NUMÉRICA
# ==============================================================================
verificar_etapa6 <- function(semilla = 20250517L, verbose = TRUE) {
  set.seed(semilla)
  ok_global <- TRUE
  reg <- function(nombre, ok, detalle = "") {
    ok_global <<- ok_global && ok
    if (verbose)
      cat(sprintf("  [%s] %-52s %s\n",
                  if (ok) "PASS" else "FALLA", nombre, detalle))
    invisible(ok)
  }
  if (verbose)
    cat("\n===== VERIFICACIÓN ETAPA 7 - GRADOS DE LIBERTAD nu =====\n\n")
  
  ## --- V1: estadisticos_nu -----------------------------------------------
  {
    xi_list <- list(c(1.0, 2.0), c(0.5), numeric(0), c(4.0, 0.25))
    suf <- estadisticos_nu(xi_list)
    P_teo <- sum(log(c(1,2,0.5,4,0.25)))
    Q_teo <- sum(c(1,2,0.5,4,0.25))
    ok <- abs(suf$P - P_teo) < 1e-12 && abs(suf$Q - Q_teo) < 1e-12 &&
      suf$N == 5L
    reg("V1 estadisticos_nu (P, Q, N)", ok,
        sprintf("(P=%.4f, Q=%.4f, N=%d)", suf$P, suf$Q, suf$N))
  }
  
  ## --- V2: gradiente analítico vs numérico -------------------------------
  {
    P <- -30; Q <- 110; N <- 100L; lam <- 0.1
    nus <- c(3, 5, 8, 15, 30)
    err <- 0
    for (nu in nus) {
      h <- 1e-5
      g_num <- (log_post_nu(nu+h,P,Q,N,lam) -
                  log_post_nu(nu-h,P,Q,N,lam)) / (2*h)
      g_ana <- grad_log_post_nu(nu, P, Q, N, lam)
      err <- max(err, abs(g_num - g_ana))
    }
    reg("V2 gradiente l'(nu) analítico vs numérico", err < 1e-4,
        sprintf("(max|dif| = %.2e)", err))
  }
  
  ## --- V3: log-concavidad  l''(nu) < 0 -----------------------------------
  {
    P <- 10; Q <- 95; N <- 120L; lam <- 0.1
    nus <- seq(2.1, 50, by = 0.5)
    h <- 1e-3
    segundas <- sapply(nus, function(nu) {
      (log_post_nu(nu+h,P,Q,N,lam) - 2*log_post_nu(nu,P,Q,N,lam) +
         log_post_nu(nu-h,P,Q,N,lam)) / h^2
    })
    # comparación con la fórmula l''(nu) = (N/4)(2/nu - trigamma(nu/2))
    teo <- sapply(nus, function(nu) (N/4)*(2/nu - trigamma(nu/2)))
    ok <- all(segundas < 0) && max(abs(segundas - teo)) < 1e-2
    reg("V3 log-concavidad l''(nu) < 0", ok,
        sprintf("(max l'' = %.4f < 0)", max(segundas)))
  }
  
  ## --- V4: jacobiano correcto --------------------------------------------
  {
    P <- -5; Q <- 60; N <- 80L; lam <- 0.1
    rhos <- c(-1, 0, 1, 2.5)
    err <- 0
    for (rho in rhos) {
      nu <- 2 + exp(rho)
      lhs <- log_post_rho(rho, P, Q, N, lam)
      rhs <- log_post_nu(nu, P, Q, N, lam) + log(nu - 2)
      err <- max(err, abs(lhs - rhs))
    }
    reg("V4 jacobiano log_post_rho = log_post_nu + log(nu-2)",
        err < 1e-12, sprintf("(max|dif| = %.1e)", err))
  }
  
  ## --- V5: invariancia MH -- la cadena reproduce la condicional ----------
  {
    P <- -42; Q <- 138; N <- 130L; lam <- 0.1
    suf <- list(P = P, Q = Q, N = N)
    # malla de referencia para p(nu|Xi)
    gr <- seq(2.001, 60, length.out = 6000)
    log_d <- sapply(gr, function(nu) log_post_nu(nu, P, Q, N, lam))
    log_d <- log_d - max(log_d)
    dens  <- exp(log_d); dens <- dens / (sum(dens) * (gr[2]-gr[1]))
    media_grid <- sum(gr * dens) * (gr[2]-gr[1])
    sd_grid    <- sqrt(sum((gr-media_grid)^2 * dens) * (gr[2]-gr[1]))
    # cadena MH
    n_iter <- 60000L; burn <- 10000L
    nu_cad <- numeric(n_iter)
    nu_cur <- 10
    n_acc  <- 0L
    for (s in 1:n_iter) {
      paso <- muestrear_nu(nu_cur, lambda_nu = lam,
                           sigma_MH = 0.45, suf = suf)
      nu_cur <- paso$nu
      if (paso$aceptado) n_acc <- n_acc + 1L
      nu_cad[s] <- nu_cur
    }
    post <- nu_cad[(burn+1):n_iter]
    err_m <- abs(mean(post) - media_grid) / media_grid
    err_s <- abs(sd(post)   - sd_grid)    / sd_grid
    tasa  <- n_acc / n_iter
    ok <- err_m < 0.02 && err_s < 0.05
    reg("V5 invariancia MH (cadena vs malla)", ok,
        sprintf("(media=%.3f vs %.3f; sd=%.3f vs %.3f; acept=%.2f)",
                mean(post), media_grid, sd(post), sd_grid, tasa))
  }
  
  ## --- V6: imprescindibilidad del jacobiano ------------------------------
  {
    P <- -42; Q <- 138; N <- 130L; lam <- 0.1
    # malla correcta (con la densidad de nu)
    gr <- seq(2.001, 60, length.out = 6000)
    log_d <- sapply(gr, function(nu) log_post_nu(nu,P,Q,N,lam))
    log_d <- log_d - max(log_d)
    dens <- exp(log_d); dens <- dens/(sum(dens)*(gr[2]-gr[1]))
    media_correcta <- sum(gr*dens)*(gr[2]-gr[1])
    # cadena MH SIN jacobiano (paso de aceptación usa log_post_nu(nu), no rho)
    n_iter <- 60000L; burn <- 10000L
    nu_cur <- 10; nu_cad <- numeric(n_iter)
    for (s in 1:n_iter) {
      rho_act  <- log(nu_cur - 2)
      rho_prop <- rho_act + rnorm(1, 0, 0.45)
      nu_prop  <- 2 + exp(rho_prop)
      # SIN jacobiano: se usa l(nu) directamente, omitiendo +log(nu-2)
      la <- log_post_nu(nu_cur,  P,Q,N,lam)
      lp <- log_post_nu(nu_prop, P,Q,N,lam)
      if (log(runif(1)) < lp - la) nu_cur <- nu_prop
      nu_cad[s] <- nu_cur
    }
    media_sesgada <- mean(nu_cad[(burn+1):n_iter])
    # la media sin jacobiano debe DIFERIR apreciablemente de la correcta
    sesgo_detectado <- abs(media_sesgada - media_correcta) > 0.05
    reg("V6 jacobiano imprescindible (sesgo si se omite)",
        sesgo_detectado,
        sprintf("(sin jac.=%.3f vs correcta=%.3f)",
                media_sesgada, media_correcta))
  }
  
  ## --- V7: recuperación de nu --------------------------------------------
  # Datos xi ~ G(nu0/2, nu0/2); la posterior de nu debe concentrarse cerca de nu0.
  {
    nu0 <- 6; lam <- 0.1
    N_obs <- 4000L
    xi <- rgamma(N_obs, shape = nu0/2, rate = nu0/2)
    xi_list <- list(xi)
    suf <- estadisticos_nu(xi_list)
    n_iter <- 40000L; burn <- 8000L
    nu_cur <- 12; nu_cad <- numeric(n_iter)
    for (s in 1:n_iter) {
      nu_cur <- muestrear_nu(nu_cur, lambda_nu = lam,
                             sigma_MH = 0.25, suf = suf)$nu
      nu_cad[s] <- nu_cur
    }
    post <- nu_cad[(burn+1):n_iter]
    err_rel <- abs(mean(post) - nu0) / nu0
    cubre <- (quantile(post, 0.025) < nu0) && (quantile(post, 0.975) > nu0)
    reg("V7 recuperación de nu (datos simulados)", err_rel < 0.15 && cubre,
        sprintf("(E[nu|.]=%.3f vs nu0=%d; IC95=[%.2f, %.2f])",
                mean(post), nu0, quantile(post,0.025), quantile(post,0.975)))
  }
  
  ## --- V8: adaptador de sigma_MH -----------------------------------------
  {
    # tasa observada alta -> sigma sube ; tasa baja -> sigma baja
    s_sube  <- adaptar_sigma_MH(0.2, tasa_obs = 0.80, objetivo = 0.35)
    s_baja  <- adaptar_sigma_MH(0.2, tasa_obs = 0.05, objetivo = 0.35)
    s_igual <- adaptar_sigma_MH(0.2, tasa_obs = 0.35, objetivo = 0.35)
    ok <- s_sube > 0.2 && s_baja < 0.2 && abs(s_igual - 0.2) < 1e-12
    reg("V8 adaptador de sigma_MH", ok,
        sprintf("(sube->%.3f, baja->%.3f, neutro->%.3f)",
                s_sube, s_baja, s_igual))
  }
  
  if (verbose) {
    cat(sprintf("\n===== RESULTADO: %s =====\n\n",
                if (ok_global) "TODOS LOS TESTS PASAN"
                else "HAY TESTS QUE FALLAN"))
  }
  invisible(ok_global)
}


# ==============================================================================
#  EJECUCIÓN DE LA VERIFICACIÓN
# ==============================================================================
# ok_etapa6 <- verificar_etapa6()