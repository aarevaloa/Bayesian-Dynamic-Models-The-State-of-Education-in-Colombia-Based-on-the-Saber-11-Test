#_______________________________________________________________________________
#@  MODELO 4 - MODELO DINÁMICO BAYESIANO t-STUDENT ROBUSTO
#@  CONDICIONAL COMPLETA DE LAS PRECISIONES MUNICIPALES v_{j,k}
#_______________________________________________________________________________
# ==============================================================================
#  1. SUMA DE CUADRADOS PONDERADA POR MUNICIPIO
# ==============================================================================
acumular_SSW <- function(e_list, xi_list, orden_celdas, phi_curr,
                         n_k, m, Tt) {
  stopifnot(length(e_list) == Tt, length(xi_list) == Tt,
            length(orden_celdas) == Tt, ncol(phi_curr) == Tt, nrow(phi_curr) == m)
  SSW <- matrix(0.0, nrow = max(n_k), ncol = m)
  for (tt in 1:Tt) {
    e_t  <- e_list[[tt]]
    xi_t <- xi_list[[tt]]
    kj   <- orden_celdas[[tt]]
    if (nrow(kj) == 0L) next
    stopifnot(length(e_t) == length(xi_t),
              length(e_t) == sum(kj[, 3L]))
    # Contribución por observación: phi_{k,t} xi e^2. phi es constante dentro de
    # la celda; se aplica por celda. xi y e^2 son por observación.
    pos <- 0L
    for (r in 1:nrow(kj)) {
      kk <- kj[r, 1L]; jj <- kj[r, 2L]; n_ijkt <- kj[r, 3L]
      rng <- (pos + 1L):(pos + n_ijkt)
      # sum_i xi_i e_i^2  dentro de la celda, luego se pondera por phi_{k,t}
      ss_celda <- sum(xi_t[rng] * e_t[rng]^2)
      SSW[jj, kk] <- SSW[jj, kk] + phi_curr[kk, tt] * ss_celda
      pos <- pos + n_ijkt
    }
  }
  SSW
}


# ==============================================================================
#  2. MUESTREO DE LAS PRECISIONES MUNICIPALES 
# ==============================================================================
muestrear_v <- function(e_list, xi_list, orden_celdas, phi_curr,
                        N_jk, n_k, m, Tt, a_v, b_v) {
  stopifnot(a_v > 0, b_v > 0, nrow(N_jk) == max(n_k), ncol(N_jk) == m)
  
  # SSW_{j,k} = sum_{t,i} phi xi e^2
  SSW <- acumular_SSW(e_list, xi_list, orden_celdas, phi_curr, n_k, m, Tt)
  
  v_curr <- matrix(NA_real_, nrow = max(n_k), ncol = m)
  for (kk in 1:m) {
    jj_idx  <- 1:n_k[kk]
    shape_v <- a_v + N_jk[jj_idx, kk] / 2                 # a_v + n_{j,k}/2
    rate_v  <- b_v + SSW[jj_idx, kk] / 2                  # b_v + SSW_{j,k}/2
    # Municipios sin datos (n_{j,k}=0, SSW=0): shape=a_v, rate=b_v -> prior.
    v_curr[jj_idx, kk] <- rgamma(n_k[kk], shape = shape_v, rate = rate_v)
  }
  v_curr
}


# ==============================================================================
#  3. VERIFICACIÓN NUMÉRICA
# ==============================================================================
verificar_etapa4a <- function(semilla = 20250517L, verbose = TRUE) {
  set.seed(semilla)
  ok_global <- TRUE
  reg <- function(nombre, ok, detalle = "") {
    ok_global <<- ok_global && ok
    if (verbose)
      cat(sprintf("  [%s] %-50s %s\n",
                  if (ok) "PASS" else "FALLA", nombre, detalle))
    invisible(ok)
  }
  if (verbose) cat("\n===== VERIFICACIÓN ETAPA 4a - PRECISIONES MUNICIPALES v =====\n\n")
  
  ## --- V1: acumular_SSW sobre ejemplo conocido ---------------------------
  {
    m <- 2L; n_k <- c(1L, 2L); Tt <- 2L
    # t=1: celda (k1,j1) con 2 obs, celda (k2,j1) con 1 obs
    # t=2: celda (k2,j2) con 2 obs
    orden <- list(rbind(c(1L,1L,2L), c(2L,1L,1L)),
                  rbind(c(2L,2L,2L)))
    e_list  <- list(c(1.0, 2.0, 3.0), c(0.5, 1.5))
    xi_list <- list(c(1.0, 0.5, 2.0), c(2.0, 1.0))
    phi_curr <- matrix(c(10, 20, 30, 40), nrow = 2, ncol = 2)  # phi[k,t]
    SSW <- acumular_SSW(e_list, xi_list, orden, phi_curr, n_k, m, Tt)
    # SSW[j=1,k=1]: phi[1,1]=10 * sum(xi*e^2) = 10*(1*1 + 0.5*4) = 10*3 = 30
    # SSW[j=1,k=2]: phi[2,1]=20 * (2*9)       = 20*18 = 360
    # SSW[j=2,k=2]: phi[2,2]=40 * (2*0.25 + 1*2.25) = 40*2.75 = 110
    ok <- abs(SSW[1,1] - 30) < 1e-9 &&
      abs(SSW[1,2] - 360) < 1e-9 &&
      abs(SSW[2,2] - 110) < 1e-9 &&
      SSW[2,1] == 0      # entrada no usada (n_k[1]=1)
    reg("V1 acumular_SSW (suma ponderada phi*xi*e^2)", ok,
        sprintf("(SSW=[%.0f, %.0f, %.0f])", SSW[1,1], SSW[1,2], SSW[2,2]))
  }
  
  ## --- V2: parametrización Gamma shape-rate ------------------------------
  {
    a_v <- 1.0; b_v <- 1.0
    # municipio único, n_jk=8 obs, SSW conocido
    m <- 1L; n_k <- 1L; Tt <- 1L
    n_obs <- 8L
    orden  <- list(rbind(c(1L,1L,n_obs)))
    e_t  <- c(0.4,-0.7,1.1,-0.3,0.9,-1.2,0.6,-0.5)
    xi_t <- c(1.0, 0.8,1.2, 0.6,1.1, 0.9,1.0, 0.7)
    phi_curr <- matrix(2.0, 1, 1)
    SSW_val <- phi_curr[1,1] * sum(xi_t * e_t^2)
    N_jk <- matrix(n_obs, 1, 1)
    R <- 200000L
    draws <- numeric(R)
    a_post <- a_v + n_obs/2
    b_post <- b_v + SSW_val/2
    for (r in 1:R)
      draws[r] <- muestrear_v(list(e_t), list(xi_t), orden, phi_curr,
                              N_jk, n_k, m, Tt, a_v, b_v)[1,1]
    teo_m <- a_post / b_post; teo_v <- a_post / b_post^2
    err_m <- abs(mean(draws) - teo_m)
    err_v <- abs(var(draws) / teo_v - 1)
    ok <- err_m < 5*sqrt(teo_v/R) && err_v < 0.03
    reg("V2 parametrización Gamma shape-rate", ok,
        sprintf("(media=%.4f vs %.4f; var=%.4f vs %.4f)",
                mean(draws), teo_m, var(draws), teo_v))
  }
  
  ## --- V3: caso límite n_{j,k}=0 -> prior --------------------------------
  {
    a_v <- 2.0; b_v <- 3.0
    m <- 1L; n_k <- 1L; Tt <- 1L
    # municipio sin observaciones: orden_celdas vacío
    orden <- list(matrix(integer(0), nrow = 0L, ncol = 3L))
    N_jk  <- matrix(0L, 1, 1)
    phi_curr <- matrix(1.0, 1, 1)
    R <- 200000L
    draws <- numeric(R)
    for (r in 1:R)
      draws[r] <- muestrear_v(list(numeric(0)), list(numeric(0)), orden,
                              phi_curr, N_jk, n_k, m, Tt, a_v, b_v)[1,1]
    # debe ser exactamente la prior G(a_v,b_v): media a/b, var a/b^2
    teo_m <- a_v/b_v; teo_v <- a_v/b_v^2
    ks <- suppressWarnings(ks.test(draws, "pgamma", shape = a_v, rate = b_v))
    ok <- ks$p.value > 0.01 && abs(mean(draws) - teo_m) < 0.01
    reg("V3 caso n_{j,k}=0 -> condicional == prior", ok,
        sprintf("(KS p=%.3f; media=%.4f vs %.4f)",
                ks$p.value, mean(draws), teo_m))
  }
  
  ## --- V4: test conjunto de Geweke ---------------------------------------
  {
    a_v <- 3.0; b_v <- 2.0
    m <- 1L; n_k <- 1L; Tt <- 1L
    n_obs <- 6L
    w <- c(1.0, 2.5, 0.7, 1.8, 3.0, 0.5)        # pesos phi*xi por obs (fijos)
    orden <- list(rbind(c(1L,1L,n_obs)))
    N_jk  <- matrix(n_obs, 1, 1)
    # se factoriza w = phi * xi  (phi escalar de la celda, xi por obs)
    phi_val <- 1.3
    xi_w <- w / phi_val
    phi_curr <- matrix(phi_val, 1, 1)
    R <- 120000L
    v_new <- numeric(R)
    for (r in 1:R) {
      v_star <- rgamma(1, shape = a_v, rate = b_v)        # prior
      y      <- rnorm(n_obs, 0, sd = 1/sqrt(v_star * w))  # y | v*
      # residuos e = y - 0 ; condicional de v dado (y, phi, xi)
      v_new[r] <- muestrear_v(list(y), list(xi_w), orden, phi_curr,
                              N_jk, n_k, m, Tt, a_v, b_v)[1,1]
    }
    ks <- suppressWarnings(ks.test(v_new, "pgamma", shape = a_v, rate = b_v))
    err_m <- abs(mean(v_new) - a_v/b_v)
    ok <- ks$p.value > 0.01 && err_m < 0.02
    reg("V4 test conjunto de Geweke (condicional correcta)", ok,
        sprintf("(KS p=%.3f; media=%.4f vs %.4f)",
                ks$p.value, mean(v_new), a_v/b_v))
  }
  
  ## --- V5: recuperación de parámetros ------------------------------------
  {
    a_v <- 1.0; b_v <- 1.0
    m <- 1L; n_k <- 1L; Tt <- 1L
    n_obs <- 20000L
    v_true <- 2.7
    w <- runif(n_obs, 0.5, 2.0)                  # pesos phi*xi conocidos
    phi_val <- 1.0; xi_w <- w / phi_val
    phi_curr <- matrix(phi_val, 1, 1)
    y <- rnorm(n_obs, 0, sd = 1/sqrt(v_true * w))
    orden <- list(rbind(c(1L,1L,n_obs)))
    N_jk  <- matrix(n_obs, 1, 1)
    R <- 4000L
    draws <- numeric(R)
    for (r in 1:R)
      draws[r] <- muestrear_v(list(y), list(xi_w), orden, phi_curr,
                              N_jk, n_k, m, Tt, a_v, b_v)[1,1]
    err_rel <- abs(mean(draws) - v_true) / v_true
    reg("V5 recuperación de parámetros (n grande)", err_rel < 0.03,
        sprintf("(E[v|.]=%.4f vs v_true=%.2f; err.rel=%.4f)",
                mean(draws), v_true, err_rel))
  }
  
  ## --- V6: ponderación correcta (phi*xi, robustez de xi) -----------------
  {
    m <- 1L; n_k <- 1L; Tt <- 1L
    orden <- list(rbind(c(1L,1L,3L)))
    e_t  <- c(0.5, 0.5, 8.0)                     # 3a obs: atípico grande
    phi_curr <- matrix(1.0, 1, 1)
    xi_normal  <- c(1.0, 1.0, 1.0)               # sin descuento
    xi_robusto <- c(1.0, 1.0, 0.02)              # atípico descontado (xi peq.)
    SSW_norm <- acumular_SSW(list(e_t), list(xi_normal),  orden, phi_curr,
                             n_k, m, Tt)[1,1]
    SSW_rob  <- acumular_SSW(list(e_t), list(xi_robusto), orden, phi_curr,
                             n_k, m, Tt)[1,1]
    # SSW_norm = 0.25+0.25+64 = 64.5 ; SSW_rob = 0.25+0.25+0.02*64 = 1.78
    ok <- abs(SSW_norm - 64.5) < 1e-9 && abs(SSW_rob - 1.78) < 1e-9 &&
      SSW_rob < SSW_norm
    reg("V6 ponderación phi*xi y robustez", ok,
        sprintf("(SSW sin descuento=%.2f, con descuento=%.2f)",
                SSW_norm, SSW_rob))
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
# ok_etapa4a <- verificar_etapa4a()