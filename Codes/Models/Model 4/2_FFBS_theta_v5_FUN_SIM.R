#_______________________________________________________________________________
#@  MODELO 4 - MODELO DINÁMICO BAYESIANO t-STUDENT ROBUSTO
#@  COMPONENTES BÁSICOS DEL MODELO DINÁMICO + FFBS GAUSSIANO DE theta_{0:T}
#_______________________________________________________________________________
diagv <- function(v) {
  v <- as.numeric(v)
  diag(v, nrow = length(v))
}

# Simetrización explícita: corrige asimetrías por redondeo.
simetrizar <- function(M) (M + t(M)) / 2

chol_robusto <- function(M, jitter_base = 1e-10, max_intentos = 8L) {
  M <- simetrizar(M)
  d <- nrow(M)
  U <- tryCatch(chol(M), error = function(e) NULL)
  if (!is.null(U)) return(U)
  esc <- mean(diag(M))
  if (!is.finite(esc) || esc <= 0) esc <- 1
  for (k in 0:max_intentos) {
    jit <- jitter_base * esc * (10^k)
    U <- tryCatch(chol(M + diag(jit, d)), error = function(e) NULL)
    if (!is.null(U)) return(U)
  }
  stop("chol_robusto: la matriz no es SPD ni tras el jitter máximo.")
}

inv_spd <- function(M, jitter_base = 1e-10) {
  U <- chol_robusto(M, jitter_base)
  chol2inv(U)
}

rmvnorm_chol <- function(mu, Sigma, jitter_base = 1e-10) {
  d <- length(mu)
  U <- chol_robusto(Sigma, jitter_base)        # Sigma = U' U
  z <- rnorm(d)
  as.numeric(mu + crossprod(U, z))             # crossprod(U,z) = U' z
}


simular_dlm_t <- function(Tt, p, N_por_t, m_0, C_0, W_diag, nu,
                          prec_base = 1.0, semilla = NULL) {
  if (!is.null(semilla)) set.seed(semilla)
  stopifnot(Tt >= 1L, p >= 1L, nu > 2,
            length(m_0) == p, all(dim(C_0) == c(p, p)),
            all(dim(W_diag) == c(p, Tt)))
  if (length(N_por_t)   == 1L) N_por_t   <- rep(N_por_t,   Tt)
  if (length(prec_base) == 1L) prec_base <- rep(prec_base, Tt)
  stopifnot(length(N_por_t) == Tt, length(prec_base) == Tt,
            all(N_por_t >= 0L), all(prec_base > 0))
  
  # Trayectoria verdadera de estados: caminata aleatoria (sub-bloque 3).
  theta_true <- simular_trayectoria_estado(m_0, C_0, W_diag)
  
  y_list <- vector("list", Tt); F_list  <- vector("list", Tt)
  o_list <- vector("list", Tt); xi_list <- vector("list", Tt)
  
  for (tt in 1:Tt) {
    N_t <- N_por_t[tt]
    # Matriz de diseño: intercepto + (p-1) regresores N(0,1).
    Ft <- cbind(1, matrix(rnorm(N_t * (p - 1L)), nrow = N_t, ncol = p - 1L))
    # Escalas latentes de la mezcla Normal-Gamma: xi ~ G(nu/2, nu/2).
    xi <- rgamma(N_t, shape = nu / 2, rate = nu / 2)
    # Precisión observacional por observación = (v*phi) * xi.
    oinv <- prec_base[tt] * xi
    # Media y muestreo de la observación: y | theta ~ N(F theta, 1/omega_inv).
    mu_t <- as.numeric(Ft %*% theta_true[, tt + 1L])
    yt   <- rnorm(N_t, mean = mu_t, sd = 1 / sqrt(oinv))
    
    y_list[[tt]] <- yt;  F_list[[tt]] <- Ft
    o_list[[tt]] <- oinv; xi_list[[tt]] <- xi
  }
  
  list(y = y_list, F = F_list, omega_inv = o_list, xi = xi_list,
       theta_true = theta_true, W_diag = W_diag, m_0 = m_0, C_0 = C_0,
       nu = nu, Tt = Tt, p = p)
}


# ==============================================================================
#  2. ECUACIÓN DE OBSERVACIÓN
# ==============================================================================
ensamblar_yF <- function(Y, F_lst, N_jkt, n_k, m, Tt, p) {
  y_list <- vector("list", Tt)
  F_list <- vector("list", Tt)
  orden  <- vector("list", Tt)
  for (tt in 1:Tt) {
    N_t <- sum(N_jkt[, , tt])
    yt  <- numeric(N_t)
    Ft  <- matrix(NA_real_, nrow = N_t, ncol = p)
    kj  <- matrix(NA_integer_, nrow = 0L, ncol = 3L)  # (k, j, n_ijkt)
    pos <- 0L
    for (kk in 1:m) for (jj in 1:n_k[kk]) {
      n_ijkt <- N_jkt[jj, kk, tt]
      if (n_ijkt == 0L) next
      rng <- (pos + 1L):(pos + n_ijkt)
      yt[rng]   <- Y[[tt]][[kk]][[jj]]
      Ft[rng, ] <- F_lst[[tt]][[kk]][[jj]]
      kj  <- rbind(kj, c(kk, jj, n_ijkt))
      pos <- pos + n_ijkt
    }
    stopifnot(pos == N_t)
    y_list[[tt]] <- yt
    F_list[[tt]] <- Ft
    orden[[tt]]  <- kj
  }
  list(y = y_list, F = F_list, orden_celdas = orden)
}

ensamblar_omega_inv <- function(orden_celdas, v_curr, phi_curr, xi_curr, Tt) {
  o_list <- vector("list", Tt)
  for (tt in 1:Tt) {
    kj  <- orden_celdas[[tt]]
    N_t <- if (nrow(kj) > 0L) sum(kj[, 3L]) else 0L
    oinv <- numeric(N_t)
    pos  <- 0L
    if (nrow(kj) > 0L) for (r in 1:nrow(kj)) {
      kk <- kj[r, 1L]; jj <- kj[r, 2L]; n_ijkt <- kj[r, 3L]
      rng <- (pos + 1L):(pos + n_ijkt)
      oinv[rng] <- v_curr[jj, kk] * phi_curr[kk, tt] * xi_curr[[tt]][[kk]][[jj]]
      pos <- pos + n_ijkt
    }
    o_list[[tt]] <- oinv
  }
  o_list
}

ensamblar_observaciones <- function(Y, F_lst, v_curr, phi_curr, xi_curr,
                                    N_jkt, n_k, m, Tt, p) {
  yF <- ensamblar_yF(Y, F_lst, N_jkt, n_k, m, Tt, p)
  oi <- ensamblar_omega_inv(yF$orden_celdas, v_curr, phi_curr, xi_curr, Tt)
  list(y = yF$y, F = yF$F, omega_inv = oi, orden_celdas = yF$orden_celdas)
}


# ==============================================================================
#  3. ECUACIÓN DE ESTADO
# ==============================================================================
simular_trayectoria_estado <- function(m_0, C_0, W_diag) {
  p  <- length(m_0)
  Tt <- ncol(W_diag)
  stopifnot(all(dim(C_0) == c(p, p)), nrow(W_diag) == p, Tt >= 1L,
            all(W_diag > 0))
  theta <- matrix(NA_real_, nrow = p, ncol = Tt + 1L)
  theta[, 1L] <- rmvnorm_chol(m_0, C_0)                       # theta_0
  for (tt in 1:Tt) {
    omega <- rnorm(p, mean = 0, sd = sqrt(W_diag[, tt]))      # omega_t ~ N(0,W_t)
    theta[, tt + 1L] <- theta[, tt] + omega                   # theta_t
  }
  theta
}


# ==============================================================================
#  4. FORWARD FILTER (FILTRO DE KALMAN)
# ==============================================================================
forward_filter <- function(y_list, F_list, omega_inv_list, W_diag, m_0, C_0,
                           jitter_base = 1e-10) {
  Tt <- length(y_list)
  p  <- length(m_0)
  stopifnot(length(F_list) == Tt, length(omega_inv_list) == Tt,
            nrow(W_diag) == p, ncol(W_diag) == Tt,
            all(dim(C_0) == c(p, p)), Tt >= 1L)
  
  m_filt <- matrix(NA_real_, nrow = p, ncol = Tt + 1L)
  C_filt <- array(NA_real_,  dim = c(p, p, Tt + 1L))
  m_filt[, 1L]  <- m_0
  C_filt[, , 1L] <- simetrizar(C_0)
  
  for (tt in 1:Tt) {
    ## Paso 1: predicción del estado.
    a_t <- m_filt[, tt]
    R_t <- simetrizar(C_filt[, , tt] + diagv(W_diag[, tt]))
    R_t_inv <- inv_spd(R_t, jitter_base)
    
    Ft   <- F_list[[tt]]
    oinv <- omega_inv_list[[tt]]
    yt   <- y_list[[tt]]
    N_t  <- length(yt)
    
    ## Pasos 2-3: actualización por la observación (forma de información).
    if (N_t > 0L) {
      stopifnot(nrow(Ft) == N_t, ncol(Ft) == p, length(oinv) == N_t,
                all(oinv > 0))
      FtW       <- Ft * oinv                       # fila i escalada por oinv[i]
      FtT_Oi_Ft <- crossprod(Ft, FtW)              # F_t' Omega^{-1} F_t  (p x p)
      FtT_Oi_yt <- crossprod(Ft, oinv * yt)        # F_t' Omega^{-1} y_t  (p x 1)
    } else {
      FtT_Oi_Ft <- matrix(0, p, p)
      FtT_Oi_yt <- numeric(p)
    }
    C_t_inv <- simetrizar(R_t_inv + FtT_Oi_Ft)     # C_t^{-1}
    C_t     <- inv_spd(C_t_inv, jitter_base)       # C_t
    m_t     <- as.numeric(C_t %*% (R_t_inv %*% a_t + FtT_Oi_yt))
    
    ## Paso 4: almacenamiento.
    m_filt[, tt + 1L]  <- m_t
    C_filt[, , tt + 1L] <- simetrizar(C_t)
  }
  list(m = m_filt, C = C_filt)
}

forward_filter_covform <- function(y_list, F_list, omega_inv_list, W_diag,
                                   m_0, C_0, jitter_base = 1e-10) {
  Tt <- length(y_list); p <- length(m_0)
  m_filt <- matrix(NA_real_, p, Tt + 1L)
  C_filt <- array(NA_real_,  c(p, p, Tt + 1L))
  m_filt[, 1L] <- m_0; C_filt[, , 1L] <- simetrizar(C_0)
  for (tt in 1:Tt) {
    a_t <- m_filt[, tt]
    R_t <- simetrizar(C_filt[, , tt] + diagv(W_diag[, tt]))
    Ft <- F_list[[tt]]; oinv <- omega_inv_list[[tt]]; yt <- y_list[[tt]]
    N_t <- length(yt)
    if (N_t > 0L) {
      f_t   <- as.numeric(Ft %*% a_t)
      Omega <- diagv(1 / oinv)                          # Omega_t (varianzas)
      Q_t   <- simetrizar(Ft %*% R_t %*% t(Ft) + Omega) # N_t x N_t
      A_t   <- R_t %*% t(Ft) %*% inv_spd(Q_t, jitter_base)
      m_t   <- a_t + A_t %*% (yt - f_t)
      C_t   <- R_t - A_t %*% Q_t %*% t(A_t)
    } else { m_t <- a_t; C_t <- R_t }
    m_filt[, tt + 1L]  <- as.numeric(m_t)
    C_filt[, , tt + 1L] <- simetrizar(C_t)
  }
  list(m = m_filt, C = C_filt)
}


# ==============================================================================
#  5. BACKWARD SAMPLING (FFBS)
# ==============================================================================
backward_sample <- function(filt, W_diag, jitter_base = 1e-10) {
  m_filt <- filt$m
  C_filt <- filt$C
  p  <- nrow(m_filt)
  Tt <- ncol(m_filt) - 1L
  stopifnot(nrow(W_diag) == p, ncol(W_diag) == Tt, Tt >= 1L,
            dim(C_filt)[3] == Tt + 1L)
  
  theta <- matrix(NA_real_, nrow = p, ncol = Tt + 1L)
  # theta_T ~ N(m_T, C_T).
  theta[, Tt + 1L] <- rmvnorm_chol(m_filt[, Tt + 1L],
                                   C_filt[, , Tt + 1L], jitter_base)
  # Recursión hacia atrás t = T-1, ..., 0.
  for (tt in (Tt - 1L):0L) {
    C_t   <- C_filt[, , tt + 1L]                       # C_t
    a_nxt <- m_filt[, tt + 1L]                         # a_{t+1} = m_t
    R_nxt <- simetrizar(C_t + diagv(W_diag[, tt + 1L])) # R_{t+1} = C_t + W_{t+1}
    Bt_T  <- solve(R_nxt, C_t)                         # B_t' = R_{t+1}^{-1} C_t
    B_t   <- t(Bt_T)                                   # B_t  = C_t R_{t+1}^{-1}
    m_cond <- m_filt[, tt + 1L] + B_t %*% (theta[, tt + 2L] - a_nxt)
    C_cond <- simetrizar(C_t - B_t %*% R_nxt %*% Bt_T)
    theta[, tt + 1L] <- rmvnorm_chol(as.numeric(m_cond), C_cond, jitter_base)
  }
  theta
}

ffbs_gaussian <- function(y_list, F_list, omega_inv_list, W_diag, m_0, C_0,
                          jitter_base = 1e-10) {
  filt <- forward_filter(y_list, F_list, omega_inv_list, W_diag, m_0, C_0,
                         jitter_base)
  backward_sample(filt, W_diag, jitter_base)
}


kalman_smoother <- function(filt, W_diag) {
  m_filt <- filt$m; C_filt <- filt$C
  p <- nrow(m_filt); Tt <- ncol(m_filt) - 1L
  m_sm <- matrix(NA_real_, p, Tt + 1L)
  C_sm <- array(NA_real_,  c(p, p, Tt + 1L))
  m_sm[, Tt + 1L]  <- m_filt[, Tt + 1L]
  C_sm[, , Tt + 1L] <- C_filt[, , Tt + 1L]
  for (tt in (Tt - 1L):0L) {
    C_t   <- C_filt[, , tt + 1L]
    a_nxt <- m_filt[, tt + 1L]
    R_nxt <- simetrizar(C_t + diagv(W_diag[, tt + 1L]))
    Bt_T  <- solve(R_nxt, C_t); B_t <- t(Bt_T)
    m_sm[, tt + 1L]  <- m_filt[, tt + 1L] +
      B_t %*% (m_sm[, tt + 2L] - a_nxt)
    C_sm[, , tt + 1L] <- simetrizar(
      C_t + B_t %*% (C_sm[, , tt + 2L] - R_nxt) %*% Bt_T)
  }
  list(m = m_sm, C = C_sm)
}


# ==============================================================================
#  6. VERIFICACIÓN NUMÉRICA DEL FFBS
# ==============================================================================
#
#   V0 : ensamblaje de observaciones (sub-bloque 2) reconstruye y/F/omega_inv.
#   V1 : consistencia dimensional de forward + backward.
#   V2 : SPD de C_t y R_t en todo el filtro (Cholesky sin jitter).
#   V3 : forma de información == forma de covarianza (max|.| < 1e-8).
#   V4 : media y varianza Monte Carlo de R draws FFBS == suavizador RTS.
#   V5 : cobertura frecuentista de las bandas 95% del suavizador ~ 0.95.
#   V6 : covarianza cruzada lag-1 de los draws FFBS == B_t C_{t+1|T} (conjunta).
verificar_etapa2 <- function(semilla = 20250517L, verbose = TRUE) {
  set.seed(semilla)
  ok_global <- TRUE
  reg <- function(nombre, ok, detalle = "") {
    ok_global <<- ok_global && ok
    if (verbose)
      cat(sprintf("  [%s] %-46s %s\n",
                  if (ok) "PASS" else "FALLA", nombre, detalle))
    invisible(ok)
  }
  if (verbose) cat("\n===== VERIFICACIÓN ETAPA 2 - FFBS GAUSSIANO =====\n\n")
  
  ## --- V0: ensamblaje de observaciones -----------------------------------
  {
    Tt <- 2L; m <- 2L; n_k <- c(1L, 2L); p <- 3L
    N_jkt <- array(0L, dim = c(max(n_k), m, Tt))
    N_jkt[1, 1, 1] <- 2L; N_jkt[1, 2, 1] <- 1L; N_jkt[2, 2, 2] <- 3L
    Y <- F_lst <- xi_curr <- vector("list", Tt)
    for (tt in 1:Tt) {
      Y[[tt]] <- F_lst[[tt]] <- xi_curr[[tt]] <- vector("list", m)
      for (kk in 1:m) {
        Y[[tt]][[kk]] <- F_lst[[tt]][[kk]] <- xi_curr[[tt]][[kk]] <-
          vector("list", n_k[kk])
      }
    }
    Y[[1]][[1]][[1]] <- c(10, 11);          F_lst[[1]][[1]][[1]] <- matrix(1:6, 2, 3)
    Y[[1]][[2]][[1]] <- c(20);              F_lst[[1]][[2]][[1]] <- matrix(7:9, 1, 3)
    Y[[2]][[2]][[2]] <- c(30, 31, 32);      F_lst[[2]][[2]][[2]] <- matrix(1:9, 3, 3)
    xi_curr[[1]][[1]][[1]] <- c(1.0, 2.0)
    xi_curr[[1]][[2]][[1]] <- c(0.5)
    xi_curr[[2]][[2]][[2]] <- c(1.5, 1.0, 2.0)
    v_curr   <- matrix(c(2, NA, 3, 4), nrow = 2, ncol = 2)   # v[j,k]
    phi_curr <- matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)    # phi[k,t]
    ens <- ensamblar_observaciones(Y, F_lst, v_curr, phi_curr, xi_curr,
                                   N_jkt, n_k, m, Tt, p)
    # Esperado en t=1: celda (k=1,j=1) [2 obs] luego (k=2,j=1) [1 obs].
    y1_ok <- identical(as.numeric(ens$y[[1]]), c(10, 11, 20))
    # omega_inv = v_{j,k} * phi_{k,t} * xi.  Para la 3a obs (k=2,j=1,t=1):
    #   v_curr[j=1,k=2]=3 ; phi_curr[k=2,t=1]=2 ; xi=0.5  ->  3*2*0.5 = 3.
    o1_esp <- c(2 * 1 * 1.0, 2 * 1 * 2.0, 3 * 2 * 0.5)
    o2_esp <- c(4 * 4 * 1.5, 4 * 4 * 1.0, 4 * 4 * 2.0)
    o_ok <- max(abs(ens$omega_inv[[1]] - o1_esp)) < 1e-12 &&
      max(abs(ens$omega_inv[[2]] - o2_esp)) < 1e-12
    F_ok <- identical(ens$F[[1]][1:2, ], matrix(1:6, 2, 3) * 1.0) &&
      identical(ens$F[[2]][, ],    matrix(1:9, 3, 3) * 1.0)
    reg("V0 ensamblaje y/F/omega_inv", y1_ok && o_ok && F_ok)
  }
  
  ## --- Datos sintéticos para V1-V4, V6 -----------------------------------
  p <- 4L; Tt <- 8L
  m_0 <- rep(0, p)
  C_0 <- diagv(rep(4, p))
  W_diag <- matrix(runif(p * Tt, 0.05, 0.30), nrow = p, ncol = Tt)
  sim <- simular_dlm_t(Tt = Tt, p = p, N_por_t = 150L, m_0 = m_0, C_0 = C_0,
                       W_diag = W_diag, nu = 6, prec_base = 1.2,
                       semilla = semilla + 1L)
  filt <- forward_filter(sim$y, sim$F, sim$omega_inv, W_diag, m_0, C_0)
  th1  <- backward_sample(filt, W_diag)
  
  ## --- V1: dimensiones ----------------------------------------------------
  dim_ok <- all(dim(filt$m) == c(p, Tt + 1L)) &&
    all(dim(filt$C) == c(p, p, Tt + 1L)) &&
    all(dim(th1)    == c(p, Tt + 1L))
  reg("V1 consistencia dimensional", dim_ok)
  
  ## --- V2: SPD de C_t y R_t ----------------------------------------------
  spd_ok <- TRUE; min_eig <- Inf
  for (tt in 0:Tt) {
    ev <- min(eigen(filt$C[, , tt + 1L], symmetric = TRUE,
                    only.values = TRUE)$values)
    min_eig <- min(min_eig, ev); if (ev <= 0) spd_ok <- FALSE
  }
  for (tt in 1:Tt) {
    R_t <- filt$C[, , tt] + diagv(W_diag[, tt])
    ev  <- min(eigen(R_t, symmetric = TRUE, only.values = TRUE)$values)
    min_eig <- min(min_eig, ev); if (ev <= 0) spd_ok <- FALSE
  }
  reg("V2 SPD de C_t y R_t", spd_ok,
      sprintf("(autovalor mínimo = %.3e)", min_eig))
  
  ## --- V3: forma información == forma covarianza -------------------------
  filt_cov <- forward_filter_covform(sim$y, sim$F, sim$omega_inv,
                                     W_diag, m_0, C_0)
  d_m <- max(abs(filt$m - filt_cov$m))
  d_C <- max(abs(filt$C - filt_cov$C))
  reg("V3 info-form == cov-form", max(d_m, d_C) < 1e-7,
      sprintf("(max|dif| m=%.2e, C=%.2e)", d_m, d_C))
  
  ## --- V4: FFBS Monte Carlo == suavizador RTS ----------------------------
  sm <- kalman_smoother(filt, W_diag)
  R_draws <- 6000L
  acum_m  <- matrix(0, p, Tt + 1L)
  acum_m2 <- matrix(0, p, Tt + 1L)
  draws_arr <- array(NA_real_, c(p, Tt + 1L, R_draws))   # para V6
  for (r in 1:R_draws) {
    th <- backward_sample(filt, W_diag)
    draws_arr[, , r] <- th
    acum_m  <- acum_m  + th
    acum_m2 <- acum_m2 + th^2
  }
  emp_m <- acum_m / R_draws
  emp_v <- acum_m2 / R_draws - emp_m^2
  sm_v  <- apply(sm$C, 3L, diag)                          # p x (Tt+1)
  # Error estandarizado de la media: ~ N(0, 1/R).
  err_m_std <- max(abs(emp_m - sm$m) / sqrt(sm_v))
  tol_m <- 5 / sqrt(R_draws)
  # Error relativo de la varianza: ~ N(0, 2/R).
  err_v_rel <- max(abs(emp_v / sm_v - 1))
  tol_v <- 6 * sqrt(2 / R_draws)
  reg("V4 media FFBS-MC == suavizador", err_m_std < tol_m,
      sprintf("(err.std=%.4f, tol=%.4f)", err_m_std, tol_m))
  reg("V4 varianza FFBS-MC == suavizador", err_v_rel < tol_v,
      sprintf("(err.rel=%.4f, tol=%.4f)", err_v_rel, tol_v))
  
  ## --- V6: covarianza cruzada lag-1 == B_t C_{t+1|T} ---------------------
  err_cc <- 0
  for (tt in 0:(Tt - 1L)) {
    A <- draws_arr[, tt + 1L, ]; Bm <- draws_arr[, tt + 2L, ]
    cc_emp <- (A %*% t(Bm)) / R_draws -
      outer(emp_m[, tt + 1L], emp_m[, tt + 2L])
    C_t   <- filt$C[, , tt + 1L]
    R_nxt <- simetrizar(C_t + diagv(W_diag[, tt + 1L]))
    B_t   <- C_t %*% solve(R_nxt)
    cc_teo <- B_t %*% sm$C[, , tt + 2L]                   # Cov(theta_t,theta_{t+1}|D_T)
    sc <- sqrt(outer(sm_v[, tt + 1L], sm_v[, tt + 2L]))
    err_cc <- max(err_cc, max(abs(cc_emp - cc_teo) / sc))
  }
  tol_cc <- 7 / sqrt(R_draws)
  reg("V6 cov-cruzada lag-1 == B_t C_{t+1|T}", err_cc < tol_cc,
      sprintf("(err.std=%.4f, tol=%.4f)", err_cc, tol_cc))
  
  ## --- V5: cobertura frecuentista del suavizador -------------------------
  n_sim <- 50L; pp <- 4L; TT <- 8L
  dentro <- 0L; total <- 0L
  m0c <- rep(0, pp); C0c <- diagv(rep(4, pp))
  for (s in 1:n_sim) {
    Wd <- matrix(runif(pp * TT, 0.05, 0.30), pp, TT)
    sm_sim <- simular_dlm_t(TT, pp, N_por_t = 120L, m_0 = m0c, C_0 = C0c,
                            W_diag = Wd, nu = 8, prec_base = 1.0)
    ff <- forward_filter(sm_sim$y, sm_sim$F, sm_sim$omega_inv, Wd, m0c, C0c)
    su <- kalman_smoother(ff, Wd)
    sdv <- sqrt(apply(su$C, 3L, diag))                    # pp x (TT+1)
    lo <- su$m - 1.96 * sdv; hi <- su$m + 1.96 * sdv
    dentro <- dentro + sum(sm_sim$theta_true >= lo & sm_sim$theta_true <= hi)
    total  <- total + length(sm_sim$theta_true)
  }
  cob <- dentro / total
  reg("V5 cobertura bandas 95%", abs(cob - 0.95) < 0.03,
      sprintf("(cobertura=%.3f sobre %d estados)", cob, total))
  
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
# ok_etapa2 <- verificar_etapa2()