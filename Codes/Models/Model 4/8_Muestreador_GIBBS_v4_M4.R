#_______________________________________________________________________________
#@  MODELO 4 - MODELO DINÁMICO BAYESIANO t-STUDENT ROBUSTO
#@  ENSAMBLAJE DEL MUESTREADOR DE GIBBS HÍBRIDO COMPLETO
#_______________________________________________________________________________
# Carga de los módulos de las etapas previas.
.dir_etapas <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) ".")
if (!exists("ffbs_gaussian"))
  source(file.path(.dir_etapas, "2_FFBS_theta_v6_M4.R"))
if (!exists("muestrear_xi"))
  source(file.path(.dir_etapas, "3_Escalas_latentes_xi_v4_M4.R"))
if (!exists("muestrear_v"))
  source(file.path(.dir_etapas, "4_precision_municipal_v_v3_M4.R"))
if (!exists("muestrear_phi"))
  source(file.path(.dir_etapas, "5_Muestreo_retrospectivo_beta_gamma_phi_v7_M4.R"))
if (!exists("muestrear_lasso"))
  source(file.path(.dir_etapas, "6_LASSO_dinamico_tau_lambda_phi_v4_M4.R"))
if (!exists("muestrear_nu"))
  source(file.path(.dir_etapas, "7_MH_nu_v3_M4.R"))


# ==============================================================================
#  1. MATRIZ DE COVARIANZA OBSERVACIONAL CON xi APILADO
# ==============================================================================
omega_inv_apilado <- function(vphi_list, xi_list) {
  Tt <- length(vphi_list)
  stopifnot(length(xi_list) == Tt)
  oi <- vector("list", Tt)
  for (tt in 1:Tt) {
    stopifnot(length(vphi_list[[tt]]) == length(xi_list[[tt]]))
    oi[[tt]] <- vphi_list[[tt]] * xi_list[[tt]]
  }
  oi
}


# ==============================================================================
#  2. DIAGONAL DE LA MATRIZ DE INNOVACIÓN W_t
# ==============================================================================
construir_Wt <- function(q_mu, phi2, tau2, idx_bloques, idx_mu, p, Tt) {
  W <- matrix(NA_real_, nrow = p, ncol = Tt)
  W[idx_mu, ] <- q_mu                                  # intercepto: q_mu fijo
  for (B in names(idx_bloques)) {
    idx <- idx_bloques[[B]]
    tt2 <- tau2[[B]]                                   # p_B x Tt
    stopifnot(length(idx) == nrow(tt2), ncol(tt2) == Tt)
    W[idx, ] <- phi2[B] * tt2                          # phi2_B * tau2_{.,B,t}
  }
  stopifnot(all(is.finite(W)), all(W > 0))
  W
}


# ==============================================================================
#  3. INICIALIZACIÓN DEL ESTADO DEL MUESTREADOR
# ==============================================================================
inicializar_estado <- function(dims, hip) {
  with(c(dims, hip), {
    # theta_0:T : matriz p x (Tt+1), arranca en m_0 y se mantiene constante
    theta <- matrix(rep(m_0, Tt + 1L), nrow = p, ncol = Tt + 1L)
    # xi apilado: 1 por observación
    xi_list <- lapply(1:Tt, function(tt) rep(1.0, N_t[tt]))
    # v_{j,k}: media de la prior G(a_v,b_v)
    v_curr <- matrix(NA_real_, nrow = max(n_k), ncol = m)
    for (kk in 1:m) v_curr[1:n_k[kk], kk] <- a_v / b_v
    # phi_{k,t}: media de la prior inicial G(n_k0/2, d_k0/2) = n_k0/d_k0
    phi_curr <- matrix(rep(n_k0 / d_k0, Tt), nrow = m, ncol = Tt)
    # tau2: 1 por (l,B,t)
    tau2 <- lapply(idx_bloques, function(idx)
      matrix(1.0, nrow = length(idx), ncol = Tt))
    names(tau2) <- names(idx_bloques)
    # phi2_B: media de la prior IG(a_phi,b_phi) = b/(a-1)
    phi2 <- b_phi / (a_phi - 1)
    # lambda2_B: media de la prior G(a_lam,b_lam)
    lambda2 <- a_lam / b_lam
    # nu: media de la prior trasladada = 2 + 1/lambda_nu
    nu <- 2 + 1 / lambda_nu
    list(theta = theta, xi_list = xi_list, v_curr = v_curr,
         phi_curr = phi_curr, tau2 = tau2, phi2 = phi2,
         lambda2 = lambda2, nu = nu)
  })
}


# ==============================================================================
#  4. UNA ITERACIÓN COMPLETA DEL MUESTREADOR
# ==============================================================================
gibbs_sweep <- function(est, datos, dims, hip) {
  p  <- dims$p; Tt <- dims$Tt; m <- dims$m; n_k <- dims$n_k
  idx_mu <- dims$idx_mu; idx_bloques <- dims$idx_bloques
  
  ## --- (1) Escalas latentes Xi ------------------------------------------
  # residuos con el theta actual, y v*phi por observación
  e_list   <- residuos_por_tiempo(datos$y_list, datos$F_list, est$theta)
  vphi_lst <- construir_vphi_list(datos$orden_celdas, est$v_curr,
                                  est$phi_curr, Tt)
  est$xi_list <- muestrear_xi(e_list, vphi_lst, est$nu)
  
  ## --- (2) Estados theta_0:T  (FFBS gaussiano) --------------------------
  omega_inv <- omega_inv_apilado(vphi_lst, est$xi_list)
  W_diag    <- construir_Wt(hip$q_mu, est$phi2, est$tau2,
                            idx_bloques, idx_mu, p, Tt)
  est$theta <- ffbs_gaussian(datos$y_list, datos$F_list, omega_inv,
                             W_diag, hip$m_0, hip$C_0)
  
  ## --- (3) Precisiones municipales V ------------------------------------
  # residuos recalculados con el theta recién muestreado
  e_list <- residuos_por_tiempo(datos$y_list, datos$F_list, est$theta)
  est$v_curr <- muestrear_v(e_list, est$xi_list, datos$orden_celdas,
                            est$phi_curr, datos$N_jk, n_k, m, Tt,
                            hip$a_v, hip$b_v)
  
  ## --- (4) Precisiones departamentales Phi_1:T --------------------------
  phi_out <- muestrear_phi(e_list, est$xi_list, datos$orden_celdas,
                           est$v_curr, datos$N_kt, hip$beta_disc,
                           hip$n_k0, hip$d_k0, n_k, m, Tt)
  
  ## --- (5) Recentrado de identificabilidad ------------------------------
  rec <- recentrar_identificabilidad(est$v_curr, phi_out$phi, n_k, m)
  est$v_curr   <- rec$v
  est$phi_curr <- rec$phi
  
  ## --- (6-8) LASSO dinámico: tau2 -> phi2 -> lambda2 --------------------
  hiper_lasso <- list(a_phi = hip$a_phi, b_phi = hip$b_phi,
                      a_lam = hip$a_lam, b_lam = hip$b_lam)
  las <- muestrear_lasso(est$theta, idx_bloques, est$phi2, est$lambda2,
                         hiper_lasso)
  est$tau2    <- las$tau2
  est$phi2    <- las$phi2
  est$lambda2 <- las$lambda2
  
  ## --- (9) Grados de libertad nu  (Metropolis-Hastings) -----------------
  paso_nu <- muestrear_nu(est$nu, xi_list = est$xi_list,
                          lambda_nu = hip$lambda_nu,
                          sigma_MH = hip$sigma_MH)
  est$nu <- paso_nu$nu
  
  list(est = est, nu_aceptado = paso_nu$aceptado)
}


# ==============================================================================
#  4b. REPORTE DE PROGRESO
# ==============================================================================
.fmt_tiempo <- function(seg) {
  if (!is.finite(seg)) return("--")
  if (seg < 90)        sprintf("%.0fs", seg)
  else if (seg < 5400) sprintf("%.1fm", seg / 60)
  else                 sprintf("%.1fh", seg / 3600)
}

imprimir_progreso_M4 <- function(it, total, burn_in, it_guardadas, t0,
                                 est, dims, sigma_MH, acc_tot,
                                 umbral_activo = 0.1) {
  transc <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  s_it   <- transc / it
  eta    <- (total - it) * s_it
  fase   <- if (it <= burn_in) "BURN-IN " else "MUESTREO"

  ## linea 1: avance y tiempos
  cat(sprintf("[%s] iter %d/%d (%.1f%%) | %s, ETA %s (%.3fs/it) | guardadas=%d\n",
              fase, it, total, 100 * it / total,
              .fmt_tiempo(transc), .fmt_tiempo(eta), s_it, it_guardadas))

  ## linea 2: intercepto dinamico + robustez t
  mu_t <- est$theta[dims$idx_mu, -1, drop = TRUE]
  cat(sprintf("        mu_t med=%.2f [%.2f, %.2f] | nu=%.2f | acept MH=%.2f sigma_MH=%.3f\n",
              median(mu_t), min(mu_t), max(mu_t),
              est$nu, acc_tot / it, sigma_MH))

  ## linea 3: magnitud y actividad de los bloques de coeficientes
  nb    <- names(dims$idx_bloques)
  bm    <- setNames(numeric(length(nb)), nb)
  act   <- setNames(integer(length(nb)), nb)
  tot_b <- setNames(integer(length(nb)), nb)
  for (B in nb) {
    th_B     <- abs(est$theta[dims$idx_bloques[[B]], -1, drop = FALSE])  # p_B x Tt
    med_l    <- rowMeans(th_B)
    bm[B]    <- mean(med_l)
    act[B]   <- sum(med_l > umbral_activo)
    tot_b[B] <- nrow(th_B)
  }
  cat(sprintf("        |beta| medio  E=%.3f M=%.3f D=%.3f | activos(>%.1f)  E=%d/%d M=%d/%d D=%d/%d\n",
              bm["E"], bm["M"], bm["D"], umbral_activo,
              act["E"], tot_b["E"], act["M"], tot_b["M"], act["D"], tot_b["D"]))

  ## linea 4: regularizacion LASSO
  cat(sprintf("        lambda2  E=%.3f M=%.3f D=%.3f | phi2  E=%.3f M=%.3f D=%.3f\n",
              est$lambda2["E"], est$lambda2["M"], est$lambda2["D"],
              est$phi2["E"], est$phi2["M"], est$phi2["D"]))

  ## linea 5: escalas observacionales corrientes
  kap <- numeric(0)
  for (kk in 1:dims$m) {
    v_k <- est$v_curr[1:dims$n_k[kk], kk]
    kap <- c(kap, as.numeric(1 / outer(v_k, est$phi_curr[kk, ])))
  }
  s_kt <- as.numeric(1 / est$phi_curr)
  cat(sprintf("        kappa2_jk: med=%.1f [%.1f, %.1f] | s_kt=1/phi: med=%.1f [%.1f, %.1f]\n",
              median(kap), min(kap), max(kap),
              median(s_kt), min(s_kt), max(s_kt)))
  invisible(NULL)
}


# ==============================================================================
#  5. MUESTREADOR DE GIBBS COMPLETO
# ==============================================================================
gibbs_modelo4 <- function(datos, dims, hip,
                          n_iter = 2000L, burn_in = 1000L, thin = 1L,
                          lote_adap = 100L, semilla = NULL, verbose = TRUE,
                          print_cada = NULL) {
  if (!is.null(semilla)) set.seed(semilla)
  p <- dims$p; Tt <- dims$Tt; m <- dims$m; n_k <- dims$n_k
  
  est <- inicializar_estado(dims, hip)
  hip_run <- hip                                        # copia mutable (sigma_MH)
  
  # almacenamiento
  nu_save      <- numeric(n_iter)
  phi2_save    <- matrix(NA_real_, n_iter, 3, dimnames = list(NULL, c("E","M","D")))
  lambda2_save <- matrix(NA_real_, n_iter, 3, dimnames = list(NULL, c("E","M","D")))
  theta_save   <- array(NA_real_, dim = c(p, Tt + 1L, n_iter))
  v_acum   <- matrix(0.0, nrow = max(n_k), ncol = m)
  phi_acum <- matrix(0.0, nrow = m, ncol = Tt)
  
  total <- burn_in + n_iter * thin
  acc_lote <- 0L; n_lote <- 0L
  acc_post <- 0L; n_post <- 0L
  it_guardadas <- 0L
  acc_tot <- 0L                                         # solo para el reporte
  t0 <- Sys.time()
  if (is.null(print_cada)) print_cada <- max(1L, total %/% 100L)
  
  for (it in 1:total) {
    paso <- gibbs_sweep(est, datos, dims, hip_run)
    est  <- paso$est
    acc_tot <- acc_tot + paso$nu_aceptado             # solo para el reporte
    
    if (it <= burn_in) {
      ## fase de burn-in: adaptar sigma_MH por lotes
      acc_lote <- acc_lote + paso$nu_aceptado
      n_lote   <- n_lote + 1L
      if (n_lote == lote_adap) {
        tasa <- acc_lote / n_lote
        hip_run$sigma_MH <- adaptar_sigma_MH(hip_run$sigma_MH, tasa)
        acc_lote <- 0L; n_lote <- 0L
      }
    } else {
      ## fase de muestreo: thinning + almacenamiento
      it_post <- it - burn_in
      acc_post <- acc_post + paso$nu_aceptado
      n_post   <- n_post + 1L
      if (it_post %% thin == 0L) {
        it_guardadas <- it_guardadas + 1L
        nu_save[it_guardadas]       <- est$nu
        phi2_save[it_guardadas, ]   <- est$phi2[c("E","M","D")]
        lambda2_save[it_guardadas, ]<- est$lambda2[c("E","M","D")]
        theta_save[, , it_guardadas]<- est$theta
        v_acum   <- v_acum   + ifelse(is.na(est$v_curr), 0, est$v_curr)
        phi_acum <- phi_acum + est$phi_curr
      }
    }
    
    if (verbose && (it == 1L || it %% print_cada == 0L || it == total))
      imprimir_progreso_M4(it, total, burn_in, it_guardadas, t0,
                           est, dims, hip_run$sigma_MH, acc_tot)
  }
  
  list(nu = nu_save, phi2 = phi2_save, lambda2 = lambda2_save,
       theta = theta_save,
       v_mean = v_acum / it_guardadas, phi_mean = phi_acum / it_guardadas,
       tasa_MH = acc_post / n_post, sigma_MH_final = hip_run$sigma_MH,
       n_iter = n_iter, burn_in = burn_in, thin = thin)
}


# ==============================================================================
#  6. VERIFICACIÓN NUMÉRICA
# ==============================================================================
verificar_etapa7 <- function(semilla = 20250517L, verbose = TRUE) {
  set.seed(semilla)
  ok_global <- TRUE
  reg <- function(nombre, ok, detalle = "") {
    ok_global <<- ok_global && ok
    if (verbose)
      cat(sprintf("  [%s] %-50s %s\n",
                  if (ok) "PASS" else "FALLA", nombre, detalle))
    invisible(ok)
  }
  if (verbose)
    cat("\n===== VERIFICACIÓN ETAPA 7 - MUESTREADOR COMPLETO =====\n\n")
  
  # ---- generador de un escenario de prueba pequeño -----------------------
  # p = 4 : mu, E, M, D  (p_E=p_M=p_D=1).  m departamentos, n_k municipios.
  construir_escenario <- function(Tt, m, n_k_vec, N_celda, semilla_esc) {
    set.seed(semilla_esc)
    p <- 4L
    idx_mu <- 1L
    idx_bloques <- list(E = 2L, M = 3L, D = 4L)
    # estado verdadero: caminata aleatoria suave
    theta_true <- matrix(0.0, p, Tt + 1L)
    for (tt in 2:(Tt+1L))
      theta_true[, tt] <- theta_true[, tt-1L] + rnorm(p, 0, 0.15)
    nu_true <- 6
    # estructura de celdas: cada (k,j) con N_celda obs en cada t
    y_list <- F_list <- vector("list", Tt)
    orden_celdas <- vector("list", Tt)
    N_jkt <- array(0L, dim = c(max(n_k_vec), m, Tt))
    for (tt in 1:Tt) {
      filas_kj <- list(); yt <- numeric(0); Ft <- matrix(0, 0, p)
      for (kk in 1:m) for (jj in 1:n_k_vec[kk]) {
        nobs <- N_celda
        # covariables: intercepto + 3 regresores
        Xc <- cbind(1, matrix(rnorm(nobs*3), nobs, 3))
        mu_obs <- as.numeric(Xc %*% theta_true[, tt+1L])
        # error t-Student: N(0, kappa2/xi), kappa2=1 aprox
        xi <- rgamma(nobs, nu_true/2, nu_true/2)
        err <- rnorm(nobs, 0, sd = sqrt(1 / xi))
        yt <- c(yt, mu_obs + err)
        Ft <- rbind(Ft, Xc)
        filas_kj[[length(filas_kj)+1L]] <- c(kk, jj, nobs)
        N_jkt[jj, kk, tt] <- nobs
      }
      y_list[[tt]] <- yt
      F_list[[tt]] <- Ft
      orden_celdas[[tt]] <- do.call(rbind, filas_kj)
    }
    N_jk <- apply(N_jkt, c(1,2), sum)         # max(n_k) x m
    N_kt <- apply(N_jkt, c(2,3), sum)         # m x Tt (apply ya da esta forma)
    list(p = p, idx_mu = idx_mu, idx_bloques = idx_bloques,
         y_list = y_list, F_list = F_list, orden_celdas = orden_celdas,
         N_jk = N_jk, N_kt = N_kt, theta_true = theta_true, nu_true = nu_true)
  }
  
  Tt <- 8L; m <- 3L; n_k_vec <- c(3L, 2L, 4L)
  esc <- construir_escenario(Tt, m, n_k_vec, N_celda = 60L, semilla_esc = 11L)
  
  dims <- list(p = esc$p, Tt = Tt, m = m, n_k = n_k_vec,
               idx_mu = esc$idx_mu, idx_bloques = esc$idx_bloques,
               N_t = sapply(esc$y_list, length))
  hip <- list(
    m_0 = rep(0, esc$p), C_0 = diag(esc$p) * 10,
    q_mu = 0.25, a_v = 2, b_v = 2,
    beta_disc = rep(0.95, m), n_k0 = rep(6, m), d_k0 = rep(6, m),
    a_phi = c(E=3,M=3,D=3), b_phi = c(E=2,M=2,D=2),
    a_lam = c(E=1,M=1,D=1), b_lam = c(E=1,M=1,D=1),
    lambda_nu = 0.1, sigma_MH = 0.3)
  datos <- list(y_list = esc$y_list, F_list = esc$F_list,
                orden_celdas = esc$orden_celdas,
                N_jk = esc$N_jk, N_kt = esc$N_kt)
  
  ## --- V1: omega_inv_apilado ---------------------------------------------
  {
    vphi <- list(c(2.0, 3.0), c(1.5))
    xi   <- list(c(0.5, 2.0), c(4.0))
    oi <- omega_inv_apilado(vphi, xi)
    ok <- all(oi[[1]] == c(1.0, 6.0)) && oi[[2]] == 6.0
    reg("V1 omega_inv_apilado (v*phi*xi por obs)", ok)
  }
  
  ## --- V2: construir_Wt --------------------------------------------------
  {
    p <- 4L; Tt2 <- 3L
    idxb <- list(E=2L, M=3L, D=4L)
    phi2 <- c(E=2, M=3, D=4)
    tau2 <- list(E = matrix(1:3, 1, 3), M = matrix(c(2,2,2),1,3),
                 D = matrix(c(0.5,1,1.5),1,3))
    W <- construir_Wt(q_mu = 0.7, phi2, tau2, idxb, idx_mu = 1L, p, Tt2)
    ok <- all(W[1,] == 0.7) &&                       # intercepto
      all(W[2,] == 2 * (1:3)) &&                 # E: phi2_E * tau2_E
      all(W[3,] == 3 * 2) &&                     # M
      all(W[4,] == 4 * c(0.5,1,1.5))             # D
    reg("V2 construir_Wt (q_mu y phi2_B*tau2)", ok)
  }
  
  ## --- V3: inicializar_estado --------------------------------------------
  {
    est0 <- inicializar_estado(dims, hip)
    ok <- all(dim(est0$theta) == c(esc$p, Tt+1L)) &&
      length(est0$xi_list) == Tt &&
      all(dim(est0$v_curr) == c(max(n_k_vec), m)) &&
      all(dim(est0$phi_curr) == c(m, Tt)) &&
      length(est0$tau2) == 3 && length(est0$phi2) == 3 &&
      est0$nu > 2
    reg("V3 inicializar_estado (coherente y dimensionado)", ok,
        sprintf("(nu0=%.1f, phi2_E0=%.2f)", est0$nu, est0$phi2["E"]))
  }
  
  ## --- V4: gibbs_sweep ejecuta los 9 pasos -------------------------------
  {
    est0 <- inicializar_estado(dims, hip)
    paso <- tryCatch(gibbs_sweep(est0, datos, dims, hip),
                     error = function(e) e)
    if (inherits(paso, "error")) {
      reg("V4 gibbs_sweep (9 pasos sin error)", FALSE,
          paste("ERROR:", conditionMessage(paso)))
    } else {
      est1 <- paso$est
      ok <- all(dim(est1$theta) == c(esc$p, Tt+1L)) &&
        all(is.finite(est1$theta)) &&
        est1$nu > 2 && all(est1$phi2 > 0) && all(est1$lambda2 > 0) &&
        all(sapply(est1$xi_list, function(x) all(x > 0)))
      reg("V4 gibbs_sweep (9 pasos sin error)", ok)
    }
  }
  
  ## --- V5: estabilidad -- 200 sweeps sin NA/Inf --------------------------
  {
    est <- inicializar_estado(dims, hip)
    estable <- TRUE
    for (s in 1:200) {
      est <- gibbs_sweep(est, datos, dims, hip)$est
      if (!all(is.finite(est$theta)) || est$nu <= 2 ||
          any(est$phi2 <= 0) || any(est$lambda2 <= 0) ||
          any(!is.finite(unlist(est$tau2)))) { estable <- FALSE; break }
    }
    reg("V5 estabilidad (200 sweeps sin NA/Inf)", estable,
        sprintf("(nu=%.2f tras 200 sweeps)", est$nu))
  }
  
  ## --- V6: recuperación de parámetros (cadena corta) ---------------------
  {
    out <- gibbs_modelo4(datos, dims, hip,
                         n_iter = 1200L, burn_in = 800L, thin = 1L,
                         lote_adap = 100L, semilla = 7L, verbose = FALSE)
    # nu verdadero = 6
    nu_post <- mean(out$nu)
    nu_ic <- quantile(out$nu, c(0.025, 0.975))
    nu_ok <- (nu_ic[1] < esc$nu_true) && (nu_ic[2] > esc$nu_true)
    # estados: correlación entre la media posterior y el theta verdadero
    theta_post <- apply(out$theta, c(1,2), mean)
    cor_theta <- cor(as.numeric(theta_post[, -1]),
                     as.numeric(esc$theta_true[, -1]))
    # tasa de aceptación MH en rango razonable
    mh_ok <- out$tasa_MH > 0.10 && out$tasa_MH < 0.70
    ok <- nu_ok && cor_theta > 0.9 && mh_ok
    reg("V6 recuperación de parámetros (cadena corta)", ok,
        sprintf("(nu post=%.2f IC[%.1f,%.1f] true=6; cor theta=%.3f; MH=%.2f)",
                nu_post, nu_ic[1], nu_ic[2], cor_theta, out$tasa_MH))
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
if (!exists("VERIFICAR_MODULOS") || isTRUE(VERIFICAR_MODULOS))
  ok_etapa7 <- verificar_etapa7()
