#_______________________________________________________________________________
#@  ADAPTADOR MODELO 3 (Normal dinamico)
#@  Traduce el panel sintetico -> variables GLOBALES que el Gibbs M3 consume
#_______________________________________________________________________________
construir_W_diag <- function(tau2_list, bloques_info, q_mu_val, p_total, Tt_val) {
  W <- matrix(NA_real_, nrow = p_total, ncol = Tt_val)
  W[1, ] <- q_mu_val
  for (B in names(bloques_info)) {
    blk <- bloques_info[[B]]
    W[blk$idx, ] <- blk$varsigma2 * tau2_list[[B]]
  }
  W
}

#' @param panel  data.frame largo de simulate_panel()$panel
#' @param truth  objeto truth
#' @param dir_salida  carpeta donde el Gibbs escribira los .bin
#' @param control  lista con n_iter, burn_in, thin
#' @param envir   entorno donde se crearan las globales (default: globalenv)
preparar_globales_M3 <- function(panel, truth, dir_salida,
                                 control = list(n_iter = 2000L, burn_in = 1000L, thin = 1L),
                                 estandarizar = FALSE,
                                 envir = globalenv()) {

  nm_E <- truth$labels$nombres_E
  nm_M <- truth$labels$nombres_M
  nm_D <- truth$labels$nombres_D
  p_E <- length(nm_E); p_M <- length(nm_M); p_D <- length(nm_D)
  p   <- 1L + p_E + p_M + p_D

  # --- Indices secuenciales k, j, t desde el panel ---
  pan <- panel
  deptos  <- sort(unique(pan$k)); m <- length(deptos)
  tiempos <- sort(unique(pan$t)); Tt <- length(tiempos)
  k_idx <- match(pan$k, deptos)
  t_idx <- match(pan$t, tiempos)
  # j local dentro de cada depto (consistente en todos los t)
  j_idx <- integer(nrow(pan))
  n_k <- integer(m); muni_ids <- vector("list", m)
  for (kk in 1:m) {
    sel_k <- which(k_idx == kk)
    munis <- sort(unique(pan$j[sel_k]))
    n_k[kk] <- length(munis)
    j_idx[sel_k] <- match(pan$j[sel_k], munis)
    muni_ids[[kk]] <- as.character(munis)
  }

  # --- Covariables: individuales del panel; contextuales del periodo t ---
  # (M3 usa las contextuales POR PERIODO, no promediadas)
  Xcols_E <- if (p_E > 0) as.matrix(pan[, nm_E, drop = FALSE]) else matrix(0, nrow(pan), 0)
  Xcols_M <- if (p_M > 0) as.matrix(pan[, nm_M, drop = FALSE]) else matrix(0, nrow(pan), 0)
  Xcols_D <- if (p_D > 0) as.matrix(pan[, nm_D, drop = FALSE]) else matrix(0, nrow(pan), 0)
  if (estandarizar) {
    estand <- function(M) { if (ncol(M)==0) return(M); mu<-colMeans(M); sg<-apply(M,2,sd); sg[sg<1e-12]<-1; sweep(sweep(M,2,mu,"-"),2,sg,"/") }
    Xcols_E <- estand(Xcols_E); Xcols_M <- estand(Xcols_M); Xcols_D <- estand(Xcols_D)
  }
  F_mat <- cbind(intercepto = 1, Xcols_E, Xcols_M, Xcols_D)
  colnames(F_mat) <- c("intercepto", nm_E, nm_M, nm_D)
  y_vec <- as.numeric(pan$y)

  # --- Indices de bloque ---
  idx_mu <- 1L
  idx_E <- if (p_E > 0) 2L:(1L + p_E) else integer(0)
  idx_M <- if (p_M > 0) (1L + p_E + 1L):(1L + p_E + p_M) else integer(0)
  idx_D <- if (p_D > 0) (1L + p_E + p_M + 1L):p else integer(0)
  theta_labels <- c("intercepto", nm_E, nm_M, nm_D)
  depto_ids  <- as.character(deptos)
  tiempo_ids <- as.character(tiempos)

  # --- Estructuras anidadas Y[[t]][[k]][[j]], F_lst, N_jkt (orden t,k,j) ---
  Y <- vector("list", Tt); F_lst <- vector("list", Tt)
  N_jkt <- array(0L, dim = c(max(n_k), m, Tt))
  for (tt in 1:Tt) {
    Y[[tt]] <- vector("list", m); F_lst[[tt]] <- vector("list", m)
    for (kk in 1:m) {
      Y[[tt]][[kk]] <- vector("list", n_k[kk]); F_lst[[tt]][[kk]] <- vector("list", n_k[kk])
      for (jj in 1:n_k[kk]) {
        sel <- which(t_idx == tt & k_idx == kk & j_idx == jj)
        Y[[tt]][[kk]][[jj]]     <- y_vec[sel]
        F_lst[[tt]][[kk]][[jj]] <- F_mat[sel, , drop = FALSE]
        N_jkt[jj, kk, tt]       <- length(sel)
      }
    }
  }
  N_jk <- matrix(0L, nrow = max(n_k), ncol = m)
  for (kk in 1:m) for (jj in 1:n_k[kk]) N_jk[jj, kk] <- sum(N_jkt[jj, kk, ])
  N_kt <- matrix(0L, nrow = m, ncol = Tt)
  for (kk in 1:m) for (tt in 1:Tt) N_kt[kk, tt] <- sum(N_jkt[1:n_k[kk], kk, tt])

  # --- Estadisticos globales para inicializacion ---
  y_media_global <- mean(y_vec); y_var_global <- var(y_vec)
  ols_global <- tryCatch(qr.solve(crossprod(F_mat), crossprod(F_mat, y_vec)),
                         error = function(e) rep(0, p))
  names(ols_global) <- theta_labels
  s2_depto <- numeric(m)
  for (kk in 1:m) {
    yk <- y_vec[k_idx == kk]; s2_depto[kk] <- if (length(yk) > 1) var(yk) else y_var_global
  }
  s2_depto[!is.finite(s2_depto) | s2_depto <= 0] <- y_var_global

  # --- Hiperparametros (defaults de la etapa 1 de M3) ---
  q_mu <- 10.0
  varsigma2 <- list(E = 1.0, M = 1.0, D = 1.0)
  a_lambda  <- list(E = 1.0, M = 1.0, D = 1.0)
  b_lambda  <- list(E = 0.1, M = 0.1, D = 0.1)
  a_v <- 1.0; b_v <- 1.0
  # beta de descuento: del config si se pidio dinamica de varianza, si no 0.95
  beta_val <- if (!is.null(truth$config$beta_k_discount)) truth$config$beta_k_discount else 0.95
  beta_discount <- rep(beta_val, m)
  m_0 <- ols_global
  C_0 <- diag(p); C_0[1, 1] <- 4 * y_var_global
  for (ib in list(idx_E, idx_M, idx_D)) if (length(ib) > 0) diag(C_0)[ib] <- 100.0
  n_k0 <- rep(4.0, m); d_k0 <- (n_k0 - 2) * s2_depto
  eps_omega <- 1e-8

  bloques <- list(
    E = list(nombre="Individual",   dim=p_E, idx=idx_E, varsigma2=varsigma2$E, a_lambda=a_lambda$E, b_lambda=b_lambda$E),
    M = list(nombre="Municipal",    dim=p_M, idx=idx_M, varsigma2=varsigma2$M, a_lambda=a_lambda$M, b_lambda=b_lambda$M),
    D = list(nombre="Departamental",dim=p_D, idx=idx_D, varsigma2=varsigma2$D, a_lambda=a_lambda$D, b_lambda=b_lambda$D)
  )

  # --- Estado inicial (*_curr) ---
  theta_curr <- matrix(m_0, nrow = p, ncol = Tt + 1); rownames(theta_curr) <- theta_labels
  tau2_curr <- list()
  for (B in names(bloques)) { blk <- bloques[[B]]; tau2_curr[[B]] <- matrix(2.0/(blk$a_lambda/blk$b_lambda), nrow=blk$dim, ncol=Tt) }
  lambda2_curr <- sapply(bloques, function(blk) blk$a_lambda / blk$b_lambda)
  v_curr <- matrix(NA_real_, nrow = max(n_k), ncol = m)
  for (kk in 1:m) v_curr[1:n_k[kk], kk] <- a_v / b_v
  s_curr <- matrix(rep(s2_depto, Tt), nrow = m, ncol = Tt)
  phi_curr <- 1.0 / s_curr
  m_filt <- matrix(NA_real_, nrow = p, ncol = Tt + 1); m_filt[, 1] <- m_0
  C_filt <- array(NA_real_, dim = c(p, p, Tt + 1)); C_filt[,, 1] <- C_0
  SSE_jkt <- array(0, dim = c(max(n_k), m, Tt))
  n_kt_work <- matrix(NA_real_, nrow = m, ncol = Tt)
  d_kt_work <- matrix(NA_real_, nrow = m, ncol = Tt)

  # --- Control MCMC ---
  n_iter  <- as.integer(control$n_iter)
  burn_in <- as.integer(control$burn_in)
  thin    <- as.integer(control$thin)
  n_save  <- (n_iter - burn_in) %/% thin

  # --- Volcar TODO al entorno destino ---
  vars <- list(
    Y=Y, F_lst=F_lst, N_jkt=N_jkt, N_jk=N_jk, N_kt=N_kt, n_k=n_k, m=m, Tt=Tt,
    p=p, p_E=p_E, p_M=p_M, p_D=p_D, idx_mu=idx_mu, idx_E=idx_E, idx_M=idx_M, idx_D=idx_D,
    theta_labels=theta_labels, depto_ids=depto_ids, tiempo_ids=tiempo_ids, bloques=bloques,
    F_mat=F_mat, y_vec=y_vec, y_var_global=y_var_global, ols_global=ols_global, s2_depto=s2_depto,
    q_mu=q_mu, varsigma2=varsigma2, a_lambda=a_lambda, b_lambda=b_lambda, a_v=a_v, b_v=b_v,
    beta_discount=beta_discount, m_0=m_0, C_0=C_0, n_k0=n_k0, d_k0=d_k0, eps_omega=eps_omega,
    theta_curr=theta_curr, tau2_curr=tau2_curr, lambda2_curr=lambda2_curr,
    v_curr=v_curr, s_curr=s_curr, phi_curr=phi_curr, m_filt=m_filt, C_filt=C_filt,
    SSE_jkt=SSE_jkt, n_kt_work=n_kt_work, d_kt_work=d_kt_work,
    n_iter=n_iter, burn_in=burn_in, thin=thin, n_save=n_save,
    DIR_SALIDA_M3=dir_salida
  )
  for (nm in names(vars)) assign(nm, vars[[nm]], envir = envir)
  invisible(vars)
}
