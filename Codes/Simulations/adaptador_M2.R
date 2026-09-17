#_______________________________________________________________________________
#@  ADAPTADOR MODELO 2
#@  Traduce el panel sintetico -> objeto `model_data`
#_______________________________________________________________________________
adaptar_a_M2 <- function(panel, truth, estandarizar = FALSE) {

  # --- (1) Colapso estatico ---
  est <- colapsar_a_estatico(panel, truth)
  nm_E <- truth$labels$nombres_E
  nm_M <- truth$labels$nombres_M
  nm_D <- truth$labels$nombres_D
  p_E <- length(nm_E); p_M <- length(nm_M); p_D <- length(nm_D)
  p   <- p_E + p_M + p_D

  # --- (2) Orden k -> j y recodificacion de indices ---
  ord <- order(est$k, est$j)
  est <- est[ord, , drop = FALSE]
  N <- nrow(est)
  deptos <- sort(unique(est$k)); K <- length(deptos)
  row_k <- match(est$k, deptos)
  row_j <- integer(N); J_k <- integer(K); muni_ids <- vector("list", K)
  for (kk in 1:K) {
    sel <- which(row_k == kk)
    munis <- sort(unique(est$j[sel]))
    J_k[kk] <- length(munis)
    row_j[sel] <- match(est$j[sel], munis)
    muni_ids[[kk]] <- as.character(munis)
  }
  n_cells <- sum(J_k)
  max_J <- max(J_k)

  # --- (3) Matrices de diseno ---
  X_E <- if (p_E > 0) as.matrix(est[, nm_E, drop = FALSE]) else matrix(0, N, 0)
  X_M <- if (p_M > 0) as.matrix(est[, nm_M, drop = FALSE]) else matrix(0, N, 0)
  X_D <- if (p_D > 0) as.matrix(est[, nm_D, drop = FALSE]) else matrix(0, N, 0)
  y_vec <- as.numeric(est$y)

  est_p <- NULL
  if (estandarizar) {
    estand <- function(M) {
      if (ncol(M) == 0) return(M)
      mu <- colMeans(M); sg <- apply(M, 2, sd); sg[sg < 1e-12] <- 1
      sweep(sweep(M, 2, mu, "-"), 2, sg, "/")
    }
    X_E <- estand(X_E); X_M <- estand(X_M); X_D <- estand(X_D)
  }
  colnames(X_E) <- nm_E; colnames(X_M) <- nm_M; colnames(X_D) <- nm_D

  # X_star = [1, X_E, X_M, X_D]
  X_star <- cbind(1, X_E, X_M, X_D)
  theta_labels <- c("intercepto", nm_E, nm_M, nm_D)
  colnames(X_star) <- theta_labels

  # indices de bloque (1-based, con offset del intercepto)
  idx_mu <- 1L
  idx_E <- if (p_E > 0) (1 + 1):(1 + p_E) else integer(0)
  idx_M <- if (p_M > 0) (1 + p_E + 1):(1 + p_E + p_M) else integer(0)
  idx_D <- if (p_D > 0) (1 + p_E + p_M + 1):(1 + p_E + p_M + p_D) else integer(0)

  # --- (4) Aplanado de celdas + estructuras anidadas ---
  n_jk <- matrix(0L, nrow = max_J, ncol = K)   # MATRIZ (contrato M2)
  idx_obs <- vector("list", K)
  Y <- vector("list", K); X_star_lst <- vector("list", K)
  X_E_lst <- vector("list", K); X_M_lst <- vector("list", K); X_D_lst <- vector("list", K)
  cell_index <- vector("list", K)
  z_jk <- matrix(0, n_cells, p_M); w_k <- matrix(0, K, p_D)
  cc <- 0L
  for (kk in 1:K) {
    idx_obs[[kk]] <- vector("list", J_k[kk])
    Y[[kk]] <- vector("list", J_k[kk]); X_star_lst[[kk]] <- vector("list", J_k[kk])
    X_E_lst[[kk]] <- vector("list", J_k[kk]); X_M_lst[[kk]] <- vector("list", J_k[kk])
    X_D_lst[[kk]] <- vector("list", J_k[kk])
    cell_index[[kk]] <- integer(J_k[kk])
    for (jj in 1:J_k[kk]) {
      cc <- cc + 1L; cell_index[[kk]][jj] <- cc
      sel <- which(row_k == kk & row_j == jj)
      njk <- length(sel)
      n_jk[jj, kk] <- njk
      idx_obs[[kk]][[jj]] <- sel
      Y[[kk]][[jj]] <- y_vec[sel]
      X_star_lst[[kk]][[jj]] <- X_star[sel, , drop = FALSE]
      X_E_lst[[kk]][[jj]] <- X_E[sel, , drop = FALSE]
      X_M_lst[[kk]][[jj]] <- X_M[sel, , drop = FALSE]
      X_D_lst[[kk]][[jj]] <- X_D[sel, , drop = FALSE]
      if (p_M > 0) z_jk[cc, ] <- X_M[sel[1], ]
    }
    if (p_D > 0) w_k[kk, ] <- X_D[idx_obs[[kk]][[1]][1], ]
  }
  if (p_M > 0) colnames(z_jk) <- nm_M
  if (p_D > 0) colnames(w_k) <- nm_D

  # --- (5) Ensamblar (misma estructura que ensamblar_model_data de M2) ---
  model_data <- list(
    N = N, K = K, J_k = J_k, n_jk = n_jk,
    p = p, p_E = p_E, p_M = p_M, p_D = p_D,
    idx_mu = idx_mu, idx_E = idx_E, idx_M = idx_M, idx_D = idx_D,
    row_k = row_k, row_j = row_j,
    y_vec = y_vec, X_E = X_E, X_M = X_M, X_D = X_D, X_star = X_star,
    Y = Y, X_E_lst = X_E_lst, X_M_lst = X_M_lst, X_D_lst = X_D_lst,
    X_star_lst = X_star_lst, idx_obs = idx_obs,
    z_jk = z_jk, w_k = w_k,
    theta_labels = theta_labels,
    depto_ids = as.character(deptos), muni_ids = muni_ids,
    estandar_params = est_p, cfg = truth$config,
    orden_filas = ord    # permutacion cross-section -> orden k->j del objeto
  )
  class(model_data) <- "model_data_m2"
  model_data
}
