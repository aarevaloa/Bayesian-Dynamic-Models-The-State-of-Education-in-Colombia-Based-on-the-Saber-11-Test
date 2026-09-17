#_______________________________________________________________________________
#@  ADAPTADOR MODELO 4 (t-Student dinamico)
#@  Traduce el panel sintetico -> listas datos / dims / hip
#_______________________________________________________________________________
adaptar_a_M4 <- function(panel, truth, estandarizar = FALSE,
                         sigma_MH = 0.3) {
  nm_E <- truth$labels$nombres_E
  nm_M <- truth$labels$nombres_M
  nm_D <- truth$labels$nombres_D
  p_E <- length(nm_E); p_M <- length(nm_M); p_D <- length(nm_D)
  p   <- 1L + p_E + p_M + p_D

  # --- Indices secuenciales (mismo enfoque que M3) ---
  deptos  <- sort(unique(panel$k)); m <- length(deptos)
  tiempos <- sort(unique(panel$t)); Tt <- length(tiempos)
  k_idx <- match(panel$k, deptos)
  t_idx <- match(panel$t, tiempos)
  j_idx <- integer(nrow(panel)); n_k <- integer(m)
  for (kk in 1:m) {
    sel_k <- which(k_idx == kk)
    munis <- sort(unique(panel$j[sel_k]))
    n_k[kk] <- length(munis)
    j_idx[sel_k] <- match(panel$j[sel_k], munis)
  }

  # --- Covariables (individuales del panel; contextuales por periodo) ---
  XE <- if (p_E > 0) as.matrix(panel[, nm_E, drop = FALSE]) else matrix(0, nrow(panel), 0)
  XM <- if (p_M > 0) as.matrix(panel[, nm_M, drop = FALSE]) else matrix(0, nrow(panel), 0)
  XD <- if (p_D > 0) as.matrix(panel[, nm_D, drop = FALSE]) else matrix(0, nrow(panel), 0)
  if (estandarizar) {
    est <- function(M){ if(ncol(M)==0) return(M); mu<-colMeans(M); s<-apply(M,2,sd); s[s<1e-12]<-1; sweep(sweep(M,2,mu,"-"),2,s,"/") }
    XE <- est(XE); XM <- est(XM); XD <- est(XD)
  }
  F_mat <- cbind(1, XE, XM, XD)
  colnames(F_mat) <- c("intercepto", nm_E, nm_M, nm_D)
  y_all <- as.numeric(panel$y)

  # --- Indices de bloque ---
  idx_mu <- 1L
  idx_E <- if (p_E > 0) seq.int(2L, length.out = p_E) else integer(0)
  idx_M <- if (p_M > 0) seq.int(2L + p_E, length.out = p_M) else integer(0)
  idx_D <- if (p_D > 0) seq.int(2L + p_E + p_M, length.out = p_D) else integer(0)
  idx_bloques <- list(E = idx_E, M = idx_M, D = idx_D)

  # --- Estructuras por periodo: y_list, F_list, orden_celdas (orden k->j) ---
  y_list <- vector("list", Tt); F_list <- vector("list", Tt)
  orden_celdas <- vector("list", Tt)
  N_jkt <- array(0L, dim = c(max(n_k), m, Tt))
  for (tt in 1:Tt) {
    filas <- list(); blocks_y <- list(); blocks_F <- list()
    for (kk in 1:m) for (jj in 1:n_k[kk]) {
      sel <- which(t_idx == tt & k_idx == kk & j_idx == jj)
      n_ijkt <- length(sel)
      if (n_ijkt == 0L) next
      blocks_y[[length(blocks_y)+1L]] <- y_all[sel]
      blocks_F[[length(blocks_F)+1L]] <- F_mat[sel, , drop = FALSE]
      filas[[length(filas)+1L]] <- c(kk, jj, n_ijkt)
      N_jkt[jj, kk, tt] <- n_ijkt
    }
    y_list[[tt]] <- if (length(blocks_y)>0) do.call(c, blocks_y) else numeric(0)
    F_list[[tt]] <- if (length(blocks_F)>0) do.call(rbind, blocks_F) else matrix(0,0,p)
    orden_celdas[[tt]] <- if (length(filas)>0) do.call(rbind, filas) else matrix(0L,0,3)
  }
  N_jk <- apply(N_jkt, c(1,2), sum)
  N_kt <- apply(N_jkt, c(2,3), sum)
  if (m == 1L) N_kt <- matrix(N_kt, nrow = 1)   # apply colapsa si m=1

  # --- Estadisticos para inicializacion ---
  y_var_global <- var(y_all)
  ols_global <- tryCatch(qr.solve(crossprod(F_mat), crossprod(F_mat, y_all)),
                         error = function(e) rep(0, p))
  s2_depto <- numeric(m)
  for (kk in 1:m) { yk <- y_all[k_idx==kk]; s2_depto[kk] <- if(length(yk)>1) var(yk) else y_var_global }
  s2_depto[!is.finite(s2_depto) | s2_depto<=0] <- y_var_global

  # --- Construir las tres listas ---
  datos <- list(y_list = y_list, F_list = F_list, orden_celdas = orden_celdas,
                N_jk = N_jk, N_kt = N_kt)

  dims <- list(p = p, Tt = Tt, m = m, n_k = n_k, idx_mu = idx_mu,
               idx_bloques = idx_bloques, N_t = sapply(y_list, length))

  beta_val <- if (!is.null(truth$config$beta_k_discount)) truth$config$beta_k_discount else 0.95
  C_0 <- diag(p); C_0[1,1] <- 4 * y_var_global
  for (ib in list(idx_E, idx_M, idx_D)) if (length(ib)>0) diag(C_0)[ib] <- 100.0
  n_k0 <- rep(4.0, m); d_k0 <- (n_k0 - 2) * s2_depto

  hip <- list(
    m_0 = as.numeric(ols_global), C_0 = C_0,
    q_mu = 10.0, a_v = 1.0, b_v = 1.0,
    beta_disc = rep(beta_val, m), n_k0 = n_k0, d_k0 = d_k0,
    a_phi = c(E=3, M=3, D=3), b_phi = c(E=2, M=2, D=2),
    a_lam = c(E=1, M=1, D=1), b_lam = c(E=0.1, M=0.1, D=0.1),
    lambda_nu = 0.1, sigma_MH = sigma_MH
  )

  list(datos = datos, dims = dims, hip = hip,
       theta_labels = c("intercepto", nm_E, nm_M, nm_D))
}
