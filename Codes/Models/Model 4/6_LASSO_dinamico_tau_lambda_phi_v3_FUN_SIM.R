#_______________________________________________________________________________
#@  MODELO 4 - MODELO DINÁMICO BAYESIANO t-STUDENT ROBUSTO
#@  CONDICIONALES COMPLETAS DEL LASSO DINÁMICO
#@           escalas locales tau2_{l,B,t}, escalas globales phi2_B,
#@           hiperparámetros de regularización lambda2_B
#_______________________________________________________________________________

if (!requireNamespace("statmod", quietly = TRUE))
  stop("La Etapa 6 requiere el paquete 'statmod' (statmod::rinvgauss).")


# ==============================================================================
#  1. EXTRACCIÓN DE LAS INNOVACIONES POR BLOQUE
# ==============================================================================
extraer_innovaciones <- function(theta, idx_bloques) {
  Tt1 <- ncol(theta)
  Tt  <- Tt1 - 1L
  stopifnot(Tt >= 1L, length(idx_bloques) >= 1L)
  # diferencias por columna: theta[,t+1]-theta[,t], t=1..T  -> p x Tt
  dif <- theta[, 2:Tt1, drop = FALSE] - theta[, 1:Tt, drop = FALSE]
  omega <- vector("list", length(idx_bloques))
  names(omega) <- names(idx_bloques)
  for (B in seq_along(idx_bloques)) {
    idx <- idx_bloques[[B]]
    stopifnot(length(idx) >= 1L, all(idx >= 1L), all(idx <= nrow(theta)))
    omega[[B]] <- dif[idx, , drop = FALSE]      # p_B x Tt
  }
  omega
}


# ==============================================================================
#  2. ESCALAS LOCALES tau2_{l,B,t}
# ==============================================================================
muestrear_tau2 <- function(omega, phi2, lambda2,
                           omega_min = 1e-8) {
  nb <- length(omega)
  stopifnot(length(phi2) == nb, length(lambda2) == nb,
            all(phi2 > 0), all(lambda2 > 0))
  tau2 <- vector("list", nb)
  names(tau2) <- names(omega)
  for (B in seq_len(nb)) {
    om   <- omega[[B]]
    pB   <- nrow(om); Tt <- ncol(om)
    lamB <- sqrt(lambda2[B])           # lambda_B
    phiB <- sqrt(phi2[B])              # phi_B
    # |omega| con cota inferior anti-degeneración
    abs_om <- pmax(abs(as.numeric(om)), omega_min)
    mu_vec <- lamB * phiB / abs_om     # mu = lambda_B phi_B / |omega|
    # u ~ IGauss(mu, Lambda=lambda2_B); shape de la IGauss es Lambda
    u <- statmod::rinvgauss(length(mu_vec), mean = mu_vec,
                            shape = lambda2[B])
    tau2[[B]] <- matrix(1.0 / u, nrow = pB, ncol = Tt)   # tau2 = 1/u
  }
  tau2
}


# ==============================================================================
#  3. ESCALAS GLOBALES phi2_B
# ==============================================================================
muestrear_phi2 <- function(omega, tau2, a_phi, b_phi) {
  nb <- length(omega)
  stopifnot(length(tau2) == nb, length(a_phi) == nb, length(b_phi) == nb,
            all(a_phi > 0), all(b_phi > 0))
  phi2 <- numeric(nb)
  names(phi2) <- names(omega)
  for (B in seq_len(nb)) {
    om <- omega[[B]]; tt2 <- tau2[[B]]
    stopifnot(all(dim(om) == dim(tt2)), all(tt2 > 0))
    pB <- nrow(om); Tt <- ncol(om)
    S_B   <- sum(om^2 / tt2)                       # innovaciones estandarizadas
    shape <- a_phi[B] + pB * Tt / 2
    rate  <- b_phi[B] + S_B / 2
    g <- rgamma(1, shape = shape, rate = rate)     # IG via 1/Gamma
    phi2[B] <- 1.0 / g
  }
  phi2
}


# ==============================================================================
#  4. HIPERPARÁMETROS DE REGULARIZACIÓN lambda2_B  (Gamma)
# ==============================================================================
muestrear_lambda2 <- function(tau2, a_lam, b_lam) {
  nb <- length(tau2)
  stopifnot(length(a_lam) == nb, length(b_lam) == nb,
            all(a_lam > 0), all(b_lam > 0))
  lambda2 <- numeric(nb)
  names(lambda2) <- names(tau2)
  for (B in seq_len(nb)) {
    tt2 <- tau2[[B]]
    stopifnot(all(tt2 > 0))
    pB <- nrow(tt2); Tt <- ncol(tt2)
    T_B   <- sum(tt2)
    shape <- a_lam[B] + pB * Tt                    # p_B*T COMPLETO
    rate  <- b_lam[B] + T_B / 2
    lambda2[B] <- rgamma(1, shape = shape, rate = rate)
  }
  lambda2
}


# ==============================================================================
#  5. MUESTREO DEL BLOQUE LASSO ( tau2 -> phi2 -> lambda2)
# ==============================================================================
muestrear_lasso <- function(theta, idx_bloques, phi2, lambda2, hiper,
                            omega_min = 1e-8) {
  omega <- extraer_innovaciones(theta, idx_bloques)
  # (1) escalas locales con phi2, lambda2 actuales
  tau2_new    <- muestrear_tau2(omega, phi2, lambda2, omega_min)
  # (2) escala global con los tau2 recién muestreados
  phi2_new    <- muestrear_phi2(omega, tau2_new, hiper$a_phi, hiper$b_phi)
  # (3) hiperparámetro con los tau2 recién muestreados
  lambda2_new <- muestrear_lambda2(tau2_new, hiper$a_lam, hiper$b_lam)
  list(omega = omega, tau2 = tau2_new,
       phi2 = phi2_new, lambda2 = lambda2_new)
}


# ==============================================================================
#  6. VERIFICACIÓN NUMÉRICA
# ==============================================================================
verificar_etapa5 <- function(semilla = 20250517L, verbose = TRUE) {
  set.seed(semilla)
  ok_global <- TRUE
  reg <- function(nombre, ok, detalle = "") {
    ok_global <<- ok_global && ok
    if (verbose)
      cat(sprintf("  [%s] %-52s %s\n",
                  if (ok) "PASS" else "FALLA", nombre, detalle))
    invisible(ok)
  }
  if (verbose) cat("\n===== VERIFICACIÓN ETAPA 6 - LASSO DINÁMICO =====\n\n")
  
  ## --- V1: extraer_innovaciones ------------------------------------------
  {
    # theta 4 x 4  (p=4, Tt=3): filas 1=mu, 2=E, 3=M, 4=D
    theta <- rbind(c(0, 1, 3, 6),      # mu
                   c(0, 2, 2, 5),      # E  -> omega_E = (2,0,3)
                   c(0,-1, 1, 0),      # M  -> omega_M = (-1,2,-1)
                   c(0, 4, 4, 9))      # D  -> omega_D = (4,0,5)
    idx <- list(E = 2L, M = 3L, D = 4L)
    om <- extraer_innovaciones(theta, idx)
    ok <- all(om$E == c(2,0,3)) && all(om$M == c(-1,2,-1)) &&
      all(om$D == c(4,0,5)) &&
      all(dim(om$E) == c(1,3))
    reg("V1 extraer_innovaciones (theta_t - theta_{t-1})", ok)
  }
  
  ## --- V2: parametrización IGauss de 1/tau2 ------------------------------
  # 1/tau2 ~ IGauss(mu, Lambda). E[1/tau2]=mu, Var[1/tau2]=mu^3/Lambda.
  {
    omega_val <- 0.5; phi2_val <- 4.0; lambda2_val <- 9.0
    lamB <- sqrt(lambda2_val); phiB <- sqrt(phi2_val)
    mu_teo <- lamB * phiB / abs(omega_val)               # 3*2/0.5 = 12
    Lam    <- lambda2_val
    R <- 300000L
    omega <- list(B = matrix(omega_val, 1, R))
    tau2  <- muestrear_tau2(omega, c(B = phi2_val), c(B = lambda2_val))
    u <- 1.0 / tau2$B[1, ]                                # u = 1/tau2
    var_teo <- mu_teo^3 / Lam
    err_m <- abs(mean(u) - mu_teo) / mu_teo
    err_v <- abs(var(u) / var_teo - 1)
    ok <- err_m < 0.01 && err_v < 0.05
    reg("V2 parametrización IGauss de 1/tau2", ok,
        sprintf("(E[u]=%.3f vs %.3f; var rel.err=%.3f)",
                mean(u), mu_teo, err_v))
  }
  
  ## --- V3: test conjunto de Geweke para tau2 -----------------------------
  {
    phi2_val <- 2.0; lambda2_val <- 5.0
    R <- 200000L
    # tau2* ~ prior Exp(lambda2/2)  (rate = lambda2/2)
    tau2_star <- rexp(R, rate = lambda2_val / 2)
    # omega | tau2* ~ N(0, phi2*tau2*)
    omega_v <- rnorm(R, 0, sd = sqrt(phi2_val * tau2_star))
    # condicional: 1/tau2' ~ IGauss(lambda*phi/|omega|, lambda2)
    om <- list(B = matrix(omega_v, 1, R))
    tau2_new <- muestrear_tau2(om, c(B = phi2_val), c(B = lambda2_val),
                               omega_min = 1e-12)$B[1, ]
    # tau2' debe ser marginalmente Exp(lambda2/2): media 2/lambda2
    ks <- suppressWarnings(ks.test(tau2_new, "pexp", rate = lambda2_val/2))
    err_m <- abs(mean(tau2_new) - 2/lambda2_val) / (2/lambda2_val)
    ok <- ks$p.value > 0.01 && err_m < 0.02
    reg("V3 Geweke tau2 (condicional IGauss correcta)", ok,
        sprintf("(KS p=%.3f; media=%.4f vs %.4f)",
                ks$p.value, mean(tau2_new), 2/lambda2_val))
  }
  
  ## --- V4: parametrización IG de phi2_B ----------------------------------
  {
    pB <- 1L; Tt <- 40L
    a_phi <- 3.0; b_phi <- 2.0
    om_mat   <- matrix(rnorm(pB*Tt), pB, Tt)
    tau2_mat <- matrix(runif(pB*Tt, 0.5, 2.0), pB, Tt)
    S_B <- sum(om_mat^2 / tau2_mat)
    a_post <- a_phi + pB*Tt/2
    b_post <- b_phi + S_B/2
    R <- 200000L
    draws <- numeric(R)
    for (r in 1:R)
      draws[r] <- muestrear_phi2(list(B = om_mat), list(B = tau2_mat),
                                 c(B = a_phi), c(B = b_phi))[1]
    m_teo <- b_post / (a_post - 1)
    err_m <- abs(mean(draws) - m_teo) / m_teo
    ok <- err_m < 0.02
    reg("V4 parametrización IG de phi2_B", ok,
        sprintf("(E[phi2]=%.4f vs %.4f; a*=%.1f b*=%.3f)",
                mean(draws), m_teo, a_post, b_post))
  }
  
  ## --- V5: test conjunto de Geweke para phi2_B ---------------------------
  {
    pB <- 1L; Tt <- 8L
    a_phi <- 4.0; b_phi <- 3.0
    tau2_mat <- matrix(runif(pB*Tt, 0.3, 1.5), pB, Tt)
    R <- 150000L
    phi2_new <- numeric(R)
    for (r in 1:R) {
      g <- rgamma(1, shape = a_phi, rate = b_phi)
      phi2_star <- 1.0 / g                                  # IG(a,b)
      om <- matrix(rnorm(pB*Tt, 0, sd = sqrt(phi2_star * tau2_mat)),
                   pB, Tt)
      phi2_new[r] <- muestrear_phi2(list(B = om), list(B = tau2_mat),
                                    c(B = a_phi), c(B = b_phi))[1]
    }
    # phi2' marginalmente IG(a,b): 1/phi2' ~ G(a,b)
    inv <- 1.0 / phi2_new
    ks <- suppressWarnings(ks.test(inv, "pgamma", shape = a_phi, rate = b_phi))
    err_m <- abs(mean(inv) - a_phi/b_phi) / (a_phi/b_phi)
    ok <- ks$p.value > 0.01 && err_m < 0.02
    reg("V5 Geweke phi2_B (condicional IG correcta)", ok,
        sprintf("(KS p=%.3f; E[1/phi2]=%.4f vs %.4f)",
                ks$p.value, mean(inv), a_phi/b_phi))
  }
  
  ## --- V6: parametrización Gamma de lambda2_B (shape p_B*T COMPLETO) -----
  {
    pB <- 2L; Tt <- 10L
    a_lam <- 1.0; b_lam <- 1.0
    tau2_mat <- matrix(runif(pB*Tt, 0.2, 3.0), pB, Tt)
    T_B <- sum(tau2_mat)
    a_post <- a_lam + pB*Tt           # COMPLETO, no /2
    b_post <- b_lam + T_B/2
    R <- 200000L
    draws <- numeric(R)
    for (r in 1:R)
      draws[r] <- muestrear_lambda2(list(B = tau2_mat),
                                    c(B = a_lam), c(B = b_lam))[1]
    m_teo <- a_post / b_post
    v_teo <- a_post / b_post^2
    err_m <- abs(mean(draws) - m_teo)
    err_v <- abs(var(draws)/v_teo - 1)
    # control: si se usara p_B*T/2 (mitad) el shape seria 11, no 21 -> media distinta
    shape_mitad <- a_lam + pB*Tt/2
    distingue <- abs(m_teo - shape_mitad/b_post) > 0.05
    ok <- err_m < 5*sqrt(v_teo/R) && err_v < 0.03 && distingue
    reg("V6 Gamma lambda2_B (shape p_B*T completo)", ok,
        sprintf("(media=%.4f vs %.4f; a*=%.0f)", mean(draws), m_teo, a_post))
  }
  
  ## --- V7: test conjunto de Geweke para lambda2_B ------------------------
  # lambda2* ~ prior G(a,b); tau2 | lambda2* ~ Exp(lambda2*/2) (p_B*T iid);
  # lambda2' ~ condicional debe dar lambda2' ~ prior G(a,b).
  {
    pB <- 2L; Tt <- 6L
    a_lam <- 3.0; b_lam <- 2.0
    R <- 150000L
    lam_new <- numeric(R)
    for (r in 1:R) {
      lambda2_star <- rgamma(1, shape = a_lam, rate = b_lam)        # prior
      tau2_mat <- matrix(rexp(pB*Tt, rate = lambda2_star/2), pB, Tt)
      lam_new[r] <- muestrear_lambda2(list(B = tau2_mat),
                                      c(B = a_lam), c(B = b_lam))[1]
    }
    ks <- suppressWarnings(ks.test(lam_new, "pgamma",
                                   shape = a_lam, rate = b_lam))
    err_m <- abs(mean(lam_new) - a_lam/b_lam) / (a_lam/b_lam)
    ok <- ks$p.value > 0.01 && err_m < 0.02
    reg("V7 Geweke lambda2_B (condicional Gamma correcta)", ok,
        sprintf("(KS p=%.3f; media=%.4f vs %.4f)",
                ks$p.value, mean(lam_new), a_lam/b_lam))
  }
  
  ## --- V8: muestrear_lasso encadena y respeta dimensiones por bloque -----
  {
    # p=6: mu(1) + E(2) + M(2) + D(1)  -> p_E=2, p_M=2, p_D=1
    p <- 6L; Tt <- 5L
    theta <- matrix(cumsum(rnorm(p*(Tt+1), 0, 0.5)), nrow = p, ncol = Tt+1)
    idx <- list(E = 2:3, M = 4:5, D = 6L)
    hiper <- list(a_phi = c(E=3,M=3,D=3), b_phi = c(E=2,M=2,D=2),
                  a_lam = c(E=1,M=1,D=1), b_lam = c(E=1,M=1,D=1))
    phi2_0    <- c(E=1, M=1, D=1)
    lambda2_0 <- c(E=1, M=1, D=1)
    out <- muestrear_lasso(theta, idx, phi2_0, lambda2_0, hiper)
    dim_ok <- all(dim(out$tau2$E) == c(2,Tt)) &&
      all(dim(out$tau2$M) == c(2,Tt)) &&
      all(dim(out$tau2$D) == c(1,Tt)) &&
      length(out$phi2) == 3 && length(out$lambda2) == 3
    pos_ok <- all(unlist(out$tau2) > 0) && all(out$phi2 > 0) &&
      all(out$lambda2 > 0)
    nom_ok <- all(names(out$phi2) == c("E","M","D")) &&
      all(names(out$lambda2) == c("E","M","D"))
    reg("V8 muestrear_lasso (encadenado y dimensiones)",
        dim_ok && pos_ok && nom_ok)
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
# ok_etapa5 <- verificar_etapa5()