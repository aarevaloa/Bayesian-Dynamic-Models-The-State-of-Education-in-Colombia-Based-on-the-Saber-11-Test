#_______________________________________________________________________________
#@  ADAPTADOR MODELO 1
#@  Traduce el panel sintetico -> objeto `datos`
#_______________________________________________________________________________
#' @param panel  data.frame largo de simulate_panel()$panel
#' @param truth  objeto truth de simulate_panel()$truth
#' @param estandarizar  si TRUE, estandariza covariables (como el pipeline real).
#'                      Para validacion de recuperacion conviene FALSE (mantener
#'                      la escala en que se generaron los coeficientes verdaderos).
#' @return objeto `datos` con clase "model1_data"
adaptar_a_M1 <- function(panel, truth, estandarizar = FALSE) {

  # --- (1) Colapso estatico: cross-section con contextuales promediadas ---
  est <- colapsar_a_estatico(panel, truth)   # cols: j, k, y, <E>, <M>, <D>
  nm_E <- truth$labels$nombres_E
  nm_M <- truth$labels$nombres_M
  nm_D <- truth$labels$nombres_D
  p_E <- length(nm_E); p_M <- length(nm_M); p_D <- length(nm_D)
  p   <- p_E + p_M + p_D

  # --- (2) Reordenar filas en orden k -> j (aplanado estable) ---
  ord <- order(est$k, est$j)
  est <- est[ord, , drop = FALSE]
  N <- nrow(est)
  # 'ord' mapea: fila nueva (k->j) -> indice en la cross-section colapsada.
  # 'orden_filas_panel' permite alinear cantidades a nivel obs (p.ej. xi) con el panel.

  # Recodificar k a 1..K consecutivos y j a 1..J_k dentro de cada k
  k_raw <- est$k
  deptos <- sort(unique(k_raw))
  K <- length(deptos)
  k_map <- match(k_raw, deptos)          # 1..K
  row_k <- k_map

  # j local dentro de cada depto (consecutivo 1..J_k)
  row_j <- integer(N)
  muni_ids <- vector("list", K)
  J_k <- integer(K)
  for (kk in 1:K) {
    sel <- which(row_k == kk)
    j_raw_k <- est$j[sel]
    munis <- sort(unique(j_raw_k))
    J_k[kk] <- length(munis)
    row_j[sel] <- match(j_raw_k, munis)
    muni_ids[[kk]] <- as.character(munis)
  }
  n_cells <- sum(J_k)

  # --- (3) Matrices de diseno globales (orden de filas ya es k->j) ---
  X_E <- if (p_E > 0) as.matrix(est[, nm_E, drop = FALSE]) else matrix(0, N, 0)
  X_M <- if (p_M > 0) as.matrix(est[, nm_M, drop = FALSE]) else matrix(0, N, 0)
  X_D <- if (p_D > 0) as.matrix(est[, nm_D, drop = FALSE]) else matrix(0, N, 0)
  colnames(X_E) <- nm_E; colnames(X_M) <- nm_M; colnames(X_D) <- nm_D
  y_vec <- as.numeric(est$y)

  est_p <- NULL
  if (estandarizar) {
    estand <- function(M) {
      if (ncol(M) == 0) return(list(M = M, mean = numeric(0), sd = numeric(0)))
      mu <- colMeans(M); sg <- apply(M, 2, sd); sg[sg < 1e-12] <- 1
      list(M = sweep(sweep(M, 2, mu, "-"), 2, sg, "/"), mean = mu, sd = sg)
    }
    eE <- estand(X_E); eM <- estand(X_M); eD <- estand(X_D)
    X_E <- eE$M; X_M <- eM$M; X_D <- eD$M
    est_p <- list(E = eE[c("mean","sd")], M = eM[c("mean","sd")], D = eD[c("mean","sd")])
  }

  # --- (4) Aplanado de celdas (orden k->j) + mapeos ---
  cell_index <- vector("list", K)
  cell_k <- integer(n_cells); cell_j <- integer(n_cells)
  row_cell <- integer(N)
  n_jk <- vector("list", K)
  n_jk_flat <- integer(n_cells)
  idx_obs <- vector("list", K)
  Y <- vector("list", K); X_E_lst <- vector("list", K)
  cc <- 0L
  for (kk in 1:K) {
    cell_index[[kk]] <- integer(J_k[kk])
    n_jk[[kk]] <- integer(J_k[kk])
    idx_obs[[kk]] <- vector("list", J_k[kk])
    Y[[kk]] <- vector("list", J_k[kk]); X_E_lst[[kk]] <- vector("list", J_k[kk])
    for (jj in 1:J_k[kk]) {
      cc <- cc + 1L
      cell_index[[kk]][jj] <- cc
      cell_k[cc] <- kk; cell_j[cc] <- jj
      sel <- which(row_k == kk & row_j == jj)
      njk <- length(sel)
      n_jk[[kk]][jj] <- njk; n_jk_flat[cc] <- njk
      idx_obs[[kk]][[jj]] <- sel
      row_cell[sel] <- cc
      Y[[kk]][[jj]] <- y_vec[sel]
      X_E_lst[[kk]][[jj]] <- X_E[sel, , drop = FALSE]
    }
  }

  # --- (5) Matrices compactas Z (n_cells x p_M) y W (K x p_D) ---
  Z <- matrix(0, n_cells, p_M)
  if (p_M > 0) for (kk in 1:K) for (jj in 1:J_k[kk]) {
    ccc <- cell_index[[kk]][jj]; s1 <- idx_obs[[kk]][[jj]][1]
    Z[ccc, ] <- X_M[s1, ]
  }
  W <- matrix(0, K, p_D)
  if (p_D > 0) for (kk in 1:K) {
    s1 <- idx_obs[[kk]][[1]][1]; W[kk, ] <- X_D[s1, ]
  }
  if (p_M > 0) colnames(Z) <- nm_M
  if (p_D > 0) colnames(W) <- nm_D

  # --- (6) Indices de bloque dentro de beta = (E, M, D); mu NO incluido ---
  idx_E <- if (p_E > 0) 1:p_E else integer(0)
  idx_M <- if (p_M > 0) (p_E + 1):(p_E + p_M) else integer(0)
  idx_D <- if (p_D > 0) (p_E + p_M + 1):(p_E + p_M + p_D) else integer(0)

  beta_labels  <- c(nm_E, nm_M, nm_D)
  theta_labels <- c("mu", beta_labels)

  # --- (7) Ensamblar el objeto (misma estructura que ensamblar_datos de M1) ---
  datos <- list(
    N = N, K = K, J_k = J_k, n_cells = n_cells,
    n_jk = n_jk, n_jk_flat = n_jk_flat,
    p = p, p_E = p_E, p_M = p_M, p_D = p_D,
    idx_E = idx_E, idx_M = idx_M, idx_D = idx_D,
    cell_index = cell_index, cell_k = cell_k, cell_j = cell_j,
    row_k = row_k, row_j = row_j, row_cell = row_cell,
    y_vec = y_vec, X_E = X_E, X_M = X_M, X_D = X_D,
    Y = Y, X_E_lst = X_E_lst, idx_obs = idx_obs,
    Z = Z, W = W,
    beta_labels = beta_labels, theta_labels = theta_labels,
    nombres_E = nm_E, nombres_M = nm_M, nombres_D = nm_D,
    depto_ids = as.character(deptos), muni_ids = muni_ids,
    estandar_params = est_p, cfg = truth$config,
    orden_filas = ord    # permutacion cross-section -> orden k->j del objeto
  )
  class(datos) <- "model1_data"
  datos
}
