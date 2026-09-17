#_______________________________________________________________________________
#@  MODELO 4 - MODELO DINÁMICO BAYESIANO t-STUDENT ROBUSTO
#@  MUESTREO RETROSPECTIVO BETA-GAMMA DE Phi_{1:T}
#@            + RECENTRADO DE IDENTIFICABILIDAD
#_______________________________________________________________________________
acumular_SSD <- function(e_list, xi_list, orden_celdas, v_curr, n_k, m, Tt) {
  stopifnot(length(e_list) == Tt, length(xi_list) == Tt,
            length(orden_celdas) == Tt)
  SSD <- matrix(0.0, nrow = m, ncol = Tt)
  for (tt in 1:Tt) {
    e_t  <- e_list[[tt]]
    xi_t <- xi_list[[tt]]
    kj   <- orden_celdas[[tt]]
    if (nrow(kj) == 0L) next
    stopifnot(length(e_t) == length(xi_t), length(e_t) == sum(kj[, 3L]))
    pos <- 0L
    for (r in 1:nrow(kj)) {
      kk <- kj[r, 1L]; jj <- kj[r, 2L]; n_ijkt <- kj[r, 3L]
      rng <- (pos + 1L):(pos + n_ijkt)
      # sum_i xi_i e_i^2 dentro de la celda, ponderada por v_{j,k}
      SSD[kk, tt] <- SSD[kk, tt] + v_curr[jj, kk] * sum(xi_t[rng] * e_t[rng]^2)
      pos <- pos + n_ijkt
    }
  }
  SSD
}


# ==============================================================================
#  2. FILTRO FORWARD
# ==============================================================================
filtro_forward_phi <- function(SSD, N_kt, beta_disc, n_k0, d_k0, m, Tt) {
  stopifnot(all(dim(SSD) == c(m, Tt)), all(dim(N_kt) == c(m, Tt)),
            length(beta_disc) == m, length(n_k0) == m, length(d_k0) == m,
            all(beta_disc > 0 & beta_disc <= 1), all(n_k0 > 0), all(d_k0 > 0))
  n_kt <- matrix(NA_real_, nrow = m, ncol = Tt)
  d_kt <- matrix(NA_real_, nrow = m, ncol = Tt)
  for (kk in 1:m) {
    n_prev <- n_k0[kk]; d_prev <- d_k0[kk]
    for (tt in 1:Tt) {
      n_kt[kk, tt] <- beta_disc[kk] * n_prev + N_kt[kk, tt]
      d_kt[kk, tt] <- beta_disc[kk] * d_prev + SSD[kk, tt]
      n_prev <- n_kt[kk, tt]; d_prev <- d_kt[kk, tt]
    }
  }
  list(n_kt = n_kt, d_kt = d_kt)
}


# ==============================================================================
#  3. BACKWARD SAMPLING
# ==============================================================================
backward_sample_phi <- function(n_kt, d_kt, beta_disc, m, Tt) {
  stopifnot(all(dim(n_kt) == c(m, Tt)), all(dim(d_kt) == c(m, Tt)),
            length(beta_disc) == m)
  phi <- matrix(NA_real_, nrow = m, ncol = Tt)
  for (kk in 1:m) {
    # phi_{k,T} ~ G(n_{k,T}/2, d_{k,T}/2)
    phi[kk, Tt] <- rgamma(1, shape = n_kt[kk, Tt] / 2,
                          rate  = d_kt[kk, Tt] / 2)
    if (Tt >= 2L) for (tt in (Tt - 1L):1L) {
      shape_g <- (1 - beta_disc[kk]) * n_kt[kk, tt] / 2
      if (shape_g > 0) {
        gamma_kt <- rgamma(1, shape = shape_g, rate = d_kt[kk, tt] / 2)
      } else {
        gamma_kt <- 0.0                       # beta_k = 1: precisión estática
      }
      phi[kk, tt] <- beta_disc[kk] * phi[kk, tt + 1L] + gamma_kt
    }
  }
  phi
}


# ==============================================================================
#  4. MUESTREO DE Phi_{1:T} (forward + backward)
# ==============================================================================
muestrear_phi <- function(e_list, xi_list, orden_celdas, v_curr, N_kt,
                          beta_disc, n_k0, d_k0, n_k, m, Tt) {
  SSD  <- acumular_SSD(e_list, xi_list, orden_celdas, v_curr, n_k, m, Tt)
  filt <- filtro_forward_phi(SSD, N_kt, beta_disc, n_k0, d_k0, m, Tt)
  phi  <- backward_sample_phi(filt$n_kt, filt$d_kt, beta_disc, m, Tt)
  list(phi = phi, s = 1.0 / phi, n_kt = filt$n_kt, d_kt = filt$d_kt)
}


# ==============================================================================
#  5. RECENTRADO DE IDENTIFICABILIDAD
# ==============================================================================
recentrar_identificabilidad <- function(v_curr, phi_curr, n_k, m) {
  stopifnot(ncol(v_curr) == m, nrow(phi_curr) == m)
  v_new   <- v_curr
  phi_new <- phi_curr
  for (kk in 1:m) {
    jj_idx <- 1:n_k[kk]
    v_k <- v_curr[jj_idx, kk]
    stopifnot(all(v_k > 0))
    # media geométrica = exp(media de log) -> estable frente a over/underflow
    log_vbar <- mean(log(v_k))
    v_bar    <- exp(log_vbar)
    v_new[jj_idx, kk] <- v_k / v_bar          # sum_j log v_jk pasa a 0
    phi_new[kk, ]     <- phi_curr[kk, ] * v_bar
  }
  list(v = v_new, phi = phi_new, s = 1.0 / phi_new)
}


# ==============================================================================
#  6. VERIFICACIÓN NUMÉRICA
# ==============================================================================
verificar_etapa4b <- function(semilla = 20250517L, verbose = TRUE) {
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
    cat("\n===== VERIFICACIÓN ETAPA 4b - Phi_{1:T} BETA-GAMMA =====\n\n")
  
  ## --- V1: acumular_SSD sobre ejemplo conocido ---------------------------
  {
    m <- 2L; n_k <- c(2L, 1L); Tt <- 2L
    # t=1: celda (k1,j1) 2 obs, (k1,j2) 1 obs, (k2,j1) 1 obs
    # t=2: celda (k2,j1) 2 obs
    orden <- list(rbind(c(1L,1L,2L), c(1L,2L,1L), c(2L,1L,1L)),
                  rbind(c(2L,1L,2L)))
    e_list  <- list(c(1.0, 2.0, 3.0, 4.0), c(0.5, 1.5))
    xi_list <- list(c(1.0, 0.5, 2.0, 1.0), c(2.0, 1.0))
    v_curr  <- matrix(c(10, 20, 30, NA), nrow = 2, ncol = 2)  # v[j,k]
    SSD <- acumular_SSD(e_list, xi_list, orden, v_curr, n_k, m, Tt)
    # SSD[k1,t1]: v[1,1]*(1*1+0.5*4) + v[2,1]*(2*9) = 10*3 + 20*18 = 30+360=390
    # SSD[k2,t1]: v[1,2]*(1*16) = 30*16 = 480
    # SSD[k2,t2]: v[1,2]*(2*0.25 + 1*2.25) = 30*2.75 = 82.5
    ok <- abs(SSD[1,1] - 390) < 1e-9 && abs(SSD[2,1] - 480) < 1e-9 &&
      abs(SSD[2,2] - 82.5) < 1e-9 && SSD[1,2] == 0
    reg("V1 acumular_SSD (suma ponderada v*xi*e^2)", ok,
        sprintf("(SSD=[%.0f, %.0f, %.1f])", SSD[1,1], SSD[2,1], SSD[2,2]))
  }
  
  ## --- V2: recursión del filtro forward ----------------------------------
  {
    m <- 1L; Tt <- 3L
    beta_disc <- 0.9; n_k0 <- 4.0; d_k0 <- 8.0
    SSD  <- matrix(c(10, 20, 5),  nrow = 1)
    N_kt <- matrix(c(6L, 8L, 0L), nrow = 1)
    fl <- filtro_forward_phi(SSD, N_kt, beta_disc, n_k0, d_k0, m, Tt)
    # t=1: n=0.9*4+6=9.6 ; d=0.9*8+10=17.2
    # t=2: n=0.9*9.6+8=16.64 ; d=0.9*17.2+20=35.48
    # t=3: n=0.9*16.64+0=14.976 ; d=0.9*35.48+5=36.932
    ok <- max(abs(fl$n_kt[1,] - c(9.6, 16.64, 14.976))) < 1e-9 &&
      max(abs(fl$d_kt[1,] - c(17.2, 35.48, 36.932))) < 1e-9
    reg("V2 recursión filtro forward (n,d)", ok,
        sprintf("(n_T=%.3f, d_T=%.3f)", fl$n_kt[1,3], fl$d_kt[1,3]))
  }
  
  ## --- V3: identidad Gamma-Beta de la predicción -------------------------
  {
    n_prev <- 12; d_prev <- 9; beta <- 0.85
    R <- 300000L
    phi_prev <- rgamma(R, shape = n_prev/2, rate = d_prev/2)   # X ~ G(n/2,d/2)
    eta <- rbeta(R, beta*n_prev/2, (1-beta)*n_prev/2)          # Y ~ Beta(...)
    # (a) identidad base:  X*Y ~ G(beta n/2, d/2)
    XY <- phi_prev * eta
    ks_a <- suppressWarnings(ks.test(XY, "pgamma",
                                     shape = beta*n_prev/2, rate = d_prev/2))
    # (b) predicción del modelo:  phi_pred = X*Y/beta ~ G(beta n/2, beta d/2)
    phi_pred <- XY / beta
    ks_b <- suppressWarnings(ks.test(phi_pred, "pgamma",
                                     shape = beta*n_prev/2, rate = beta*d_prev/2))
    teo_m <- n_prev / d_prev                       # media predictiva preservada
    ok <- ks_a$p.value > 0.01 && ks_b$p.value > 0.01 &&
      abs(mean(phi_pred) - teo_m) < 0.01
    reg("V3 identidad Gamma-Beta y eq. phi_predict", ok,
        sprintf("(KS XY=%.3f, KS phi_pred=%.3f; media=%.4f vs %.4f)",
                ks_a$p.value, ks_b$p.value, mean(phi_pred), teo_m))
  }
  
  ## --- V4: backward == posterior SUAVIZADA (referencia por fuerza bruta) --
  {
    m <- 1L; Tt <- 2L; beta <- 0.8
    n0 <- 6; d0 <- 6; N1 <- 5L; SSD1 <- 4; N2 <- 7L; SSD2 <- 6
    # forward consistente
    n1 <- beta*n0 + N1; d1 <- beta*d0 + SSD1
    n2 <- beta*n1 + N2; d2 <- beta*d1 + SSD2
    n_kt <- matrix(c(n1, n2), nrow = 1)
    d_kt <- matrix(c(d1, d2), nrow = 1)
    # backward
    R <- 300000L
    phi1 <- numeric(R)
    for (r in 1:R) phi1[r] <- backward_sample_phi(n_kt, d_kt, beta, m, Tt)[1,1]
    # referencia: malla para p(phi_1|D_2)
    gr <- seq(0.01, 6, length.out = 2500)
    dens <- numeric(length(gr))
    a2 <- beta*n1/2; b2 <- (1-beta)*n1/2
    eta2 <- rbeta(30000, a2, b2)                  # muestra fija para E[.|phi_1]
    for (ii in seq_along(gr)) {
      p1   <- gr[ii]
      lpd1 <- dgamma(p1, shape = n1/2, rate = d1/2)
      phi2s <- p1 * eta2 / beta
      lik2 <- mean(phi2s^(N2/2) * exp(-SSD2 * phi2s / 2))
      dens[ii] <- lpd1 * lik2
    }
    dens <- dens / (sum(dens) * (gr[2]-gr[1]))
    media_grid <- sum(gr * dens) * (gr[2]-gr[1])
    err <- abs(mean(phi1) - media_grid) / media_grid
    # comprobación adicional: la suavizada NO coincide con el filtrado
    distinta_del_filtrado <- abs(media_grid - n1/d1) > 1e-3
    ok <- err < 0.01 && distinta_del_filtrado
    reg("V4 backward == posterior suavizada (grid)", ok,
        sprintf("(E[phi1] backward=%.4f vs grid=%.4f; filtrado=%.4f)",
                mean(phi1), media_grid, n1/d1))
  }
  
  ## --- V5: caso límite beta_k = 1 -> precisión estática ------------------
  {
    m <- 1L; Tt <- 5L; beta <- 1.0
    n_kt <- matrix(c(10,12,14,16,18), nrow = 1)
    d_kt <- matrix(c(8, 9, 10,11,12), nrow = 1)
    ph <- backward_sample_phi(n_kt, d_kt, beta, m, Tt)
    # con beta=1, gamma=0 -> phi constante en t
    ok <- max(abs(ph[1,] - ph[1,Tt])) == 0
    reg("V5 caso beta_k=1 (precisión estática)", ok,
        sprintf("(rango phi = [%.4f, %.4f])", min(ph), max(ph)))
  }
  
  ## --- V6: recentrado preserva kappa2 y fija sum log v = 0 ---------------
  {
    m <- 2L; n_k <- c(3L, 2L); Tt <- 4L
    v_curr <- matrix(NA_real_, nrow = max(n_k), ncol = m)
    v_curr[1:3,1] <- c(0.5, 2.0, 4.0)
    v_curr[1:2,2] <- c(3.0, 0.7)
    phi_curr <- matrix(runif(m*Tt, 0.5, 2.0), nrow = m, ncol = Tt)
    # kappa2 antes
    kap_antes <- matrix(NA_real_, nrow = 5, ncol = 0)
    rec <- recentrar_identificabilidad(v_curr, phi_curr, n_k, m)
    # sum_j log v = 0 por departamento
    s1 <- sum(log(rec$v[1:n_k[1],1])); s2 <- sum(log(rec$v[1:n_k[2],2]))
    sum_ok <- abs(s1) < 1e-10 && abs(s2) < 1e-10
    # kappa2 = 1/(v phi) preservado para toda celda
    err_kap <- 0
    for (kk in 1:m) for (jj in 1:n_k[kk]) for (tt in 1:Tt) {
      kap_old <- 1/(v_curr[jj,kk]   * phi_curr[kk,tt])
      kap_new <- 1/(rec$v[jj,kk]    * rec$phi[kk,tt])
      err_kap <- max(err_kap, abs(kap_old - kap_new))
    }
    reg("V6 recentrado: kappa2 invariante y sum log v=0",
        sum_ok && err_kap < 1e-12,
        sprintf("(sum log v=%.1e/%.1e; max|d kappa2|=%.1e)", s1, s2, err_kap))
  }
  
  ## --- V7: recuperación de la varianza departamental ---------------------
  {
    m <- 1L; Tt <- 10L; n_k <- 1L
    beta <- 0.95; n_k0 <- 4.0
    s_true <- 3.0
    d_k0 <- (n_k0 - 2) * s_true
    N_por_t <- 400L
    # genera observaciones: e ~ N(0, s_true) (peso v*xi = 1), residuos directos
    e_list <- xi_list <- vector("list", Tt)
    orden  <- vector("list", Tt)
    N_kt   <- matrix(N_por_t, nrow = 1, ncol = Tt)
    v_curr <- matrix(1.0, nrow = 1, ncol = 1)
    for (tt in 1:Tt) {
      e_list[[tt]]  <- rnorm(N_por_t, 0, sd = sqrt(s_true))
      xi_list[[tt]] <- rep(1.0, N_por_t)
      orden[[tt]]   <- rbind(c(1L, 1L, N_por_t))
    }
    R <- 3000L
    s_acum <- matrix(0, nrow = 1, ncol = Tt)
    for (r in 1:R) {
      out <- muestrear_phi(e_list, xi_list, orden, v_curr, N_kt,
                           beta, n_k0, d_k0, n_k, m, Tt)
      s_acum <- s_acum + out$s
    }
    s_post <- s_acum / R
    err_rel <- max(abs(s_post - s_true) / s_true)
    reg("V7 recuperación de la varianza departamental", err_rel < 0.06,
        sprintf("(s_post en [%.3f, %.3f] vs s_true=%.1f; err.rel max=%.4f)",
                min(s_post), max(s_post), s_true, err_rel))
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
# ok_etapa4b <- verificar_etapa4b()