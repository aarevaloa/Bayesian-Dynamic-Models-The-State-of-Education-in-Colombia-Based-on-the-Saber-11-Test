#_______________________________________________________________________________
#@  MODELO 4 - MODELO DINÁMICO BAYESIANO t-STUDENT ROBUSTO
#@  CONDICIONAL COMPLETA DE LAS ESCALAS LATENTES xi_{i,j,k,t}
#_______________________________________________________________________________
residuos_por_tiempo <- function(y_list, F_list, theta) {
  Tt <- length(y_list)
  stopifnot(length(F_list) == Tt, ncol(theta) == Tt + 1L, Tt >= 1L)
  p <- nrow(theta)
  e_list <- vector("list", Tt)
  for (tt in 1:Tt) {
    yt  <- y_list[[tt]]
    Ft  <- F_list[[tt]]
    N_t <- length(yt)
    if (N_t == 0L) { e_list[[tt]] <- numeric(0); next }
    stopifnot(nrow(Ft) == N_t, ncol(Ft) == p)
    e_list[[tt]] <- yt - as.numeric(Ft %*% theta[, tt + 1L])
  }
  e_list
}


# ==============================================================================
#  2. PRODUCTO v_{j,k} phi_{k,t} POR OBSERVACIÓN
# ==============================================================================
construir_vphi_list <- function(orden_celdas, v_curr, phi_curr, Tt) {
  stopifnot(length(orden_celdas) == Tt, ncol(phi_curr) == Tt)
  vphi_list <- vector("list", Tt)
  for (tt in 1:Tt) {
    kj  <- orden_celdas[[tt]]
    N_t <- if (nrow(kj) > 0L) sum(kj[, 3L]) else 0L
    vphi_t <- numeric(N_t)
    pos <- 0L
    if (nrow(kj) > 0L) for (r in 1:nrow(kj)) {
      kk <- kj[r, 1L]; jj <- kj[r, 2L]; n_ijkt <- kj[r, 3L]
      rng <- (pos + 1L):(pos + n_ijkt)
      vphi_t[rng] <- v_curr[jj, kk] * phi_curr[kk, tt]
      pos <- pos + n_ijkt
    }
    vphi_list[[tt]] <- vphi_t
  }
  vphi_list
}


# ==============================================================================
#  3. MUESTREO DE LAS ESCALAS LATENTES
# ==============================================================================
muestrear_xi <- function(e_list, vphi_list, nu, xi_min = .Machine$double.xmin) {
  Tt <- length(e_list)
  stopifnot(nu > 2, length(vphi_list) == Tt, Tt >= 1L)
  shape_xi <- (nu + 1) / 2                       # a_xi, constante
  xi_list <- vector("list", Tt)
  for (tt in 1:Tt) {
    e_t    <- e_list[[tt]]
    vphi_t <- vphi_list[[tt]]
    N_t    <- length(e_t)
    stopifnot(length(vphi_t) == N_t)
    if (N_t == 0L) { xi_list[[tt]] <- numeric(0); next }
    stopifnot(all(is.finite(e_t)), all(vphi_t > 0))
    # b_xi por observación = nu/2 + v*phi*e^2 / 2
    rate_t <- nu / 2 + 0.5 * vphi_t * e_t^2
    # Muestreo vectorizado: un rgamma para las N_t observaciones del periodo.
    xi_t <- rgamma(N_t, shape = shape_xi, rate = rate_t)
    xi_list[[tt]] <- pmax(xi_t, xi_min)          # piso anti-underflow
  }
  xi_list
}

muestrear_xi_completo <- function(y_list, F_list, theta, orden_celdas,
                                  v_curr, phi_curr, nu) {
  Tt    <- length(y_list)
  e_l   <- residuos_por_tiempo(y_list, F_list, theta)
  vphi  <- construir_vphi_list(orden_celdas, v_curr, phi_curr, Tt)
  muestrear_xi(e_l, vphi, nu)
}


# ==============================================================================
#  4. VERIFICACIÓN NUMÉRICA
# ==============================================================================
verificar_etapa3 <- function(semilla = 20250517L, verbose = TRUE) {
  set.seed(semilla)
  ok_global <- TRUE
  reg <- function(nombre, ok, detalle = "") {
    ok_global <<- ok_global && ok
    if (verbose)
      cat(sprintf("  [%s] %-48s %s\n",
                  if (ok) "PASS" else "FALLA", nombre, detalle))
    invisible(ok)
  }
  if (verbose) cat("\n===== VERIFICACIÓN ETAPA 3 - ESCALAS LATENTES xi =====\n\n")
  
  ## --- V1: parametrización Gamma shape-rate ------------------------------
  {
    nu <- 5; prec <- 2; e_val <- sqrt(1.5)            # r^2 = prec*e^2 = 3
    a_xi <- (nu + 1) / 2                              # 3
    b_xi <- nu / 2 + 0.5 * prec * e_val^2             # 2.5 + 1.5 = 4
    Nobs <- 300000L
    # Se invoca la PROPIA función muestrear_xi (todas las obs con igual e,vphi).
    xi_l <- muestrear_xi(list(rep(e_val, Nobs)), list(rep(prec, Nobs)), nu)
    xi <- xi_l[[1]]
    teo_m <- a_xi / b_xi                              # 0.75
    teo_v <- a_xi / b_xi^2                            # 0.1875
    err_m <- abs(mean(xi) - teo_m)
    err_v <- abs(var(xi) / teo_v - 1)
    # Si se hubiera usado scale en vez de rate, la media sería a*b = 12.
    no_es_scale <- mean(xi) < 1                       # descarta scale-param
    ok <- err_m < 5 * sqrt(teo_v / Nobs) && err_v < 0.03 && no_es_scale
    reg("V1 parametrización Gamma shape-rate", ok,
        sprintf("(media=%.4f vs %.4f; var=%.4f vs %.4f)",
                mean(xi), teo_m, var(xi), teo_v))
  }
  
  ## --- V2: mecanismo de robustez -----------------------------------------
  {
    nu <- 5; prec <- 1
    r2_grid <- c(0, 1, 5, 20, 100)
    Nobs <- 200000L
    emp <- numeric(length(r2_grid)); teo <- (nu + 1) / (nu + r2_grid)
    for (i in seq_along(r2_grid)) {
      e_val <- sqrt(r2_grid[i] / prec)
      xi_l <- muestrear_xi(list(rep(e_val, Nobs)), list(rep(prec, Nobs)), nu)
      emp[i] <- mean(xi_l[[1]])
    }
    err <- max(abs(emp - teo) / teo)
    decreciente <- all(diff(emp) < 0)                 # E[xi] cae con r^2
    reg("V2 mecanismo de robustez E[xi]=(nu+1)/(nu+r2)",
        err < 0.01 && decreciente,
        sprintf("(max err.rel=%.4f; r2=0 -> %.3f, r2=100 -> %.4f)",
                err, emp[1], emp[length(emp)]))
  }
  
  ## --- V3: inducción de la t-Student marginal ----------------------------
  # y | xi ~ N(0, 1/(prec*xi)),  xi ~ G(nu/2, nu/2)  =>  sqrt(prec)*y ~ t_nu.
  {
    nu <- 7; prec <- 1.5; R <- 120000L
    xi   <- rgamma(R, shape = nu / 2, rate = nu / 2)
    y    <- rnorm(R, mean = 0, sd = 1 / sqrt(prec * xi))
    tval <- sqrt(prec) * y
    ks   <- suppressWarnings(ks.test(tval, "pt", df = nu))
    var_teo <- nu / (nu - 2)                          # var de t_nu
    err_v   <- abs(var(tval) / var_teo - 1)
    ok <- ks$p.value > 0.01 && err_v < 0.06
    reg("V3 t-Student marginal (mezcla N-G)", ok,
        sprintf("(KS p=%.3f; var=%.3f vs %.3f)",
                ks$p.value, var(tval), var_teo))
  }
  
  ## --- V4: test conjunto de Geweke ---------------------------------------
  # Si la condicional es correcta, al muestrear  xi* ~ prior,  y | xi*,  y luego
  # xi' ~ condicional(y),  entonces  xi'  es marginalmente G(nu/2, nu/2) (prior).
  {
    nu <- 5; prec <- 2; R <- 150000L
    xi_star <- rgamma(R, shape = nu / 2, rate = nu / 2)
    y       <- rnorm(R, mean = 0, sd = 1 / sqrt(prec * xi_star))
    # Condicional completa de xi dado y (eta = 0): shape (nu+1)/2, rate como en muestrear_xi.
    xi_new  <- muestrear_xi(list(y - 0), list(rep(prec, R)), nu)[[1]]
    ks <- suppressWarnings(ks.test(xi_new, "pgamma",
                                   shape = nu / 2, rate = nu / 2))
    err_m <- abs(mean(xi_new) - 1)                    # E[prior] = 1
    err_v <- abs(var(xi_new) - 2 / nu)                # Var[prior] = 2/nu
    ok <- ks$p.value > 0.01 && err_m < 0.02 && err_v < 0.02
    reg("V4 test conjunto de Geweke (condicional correcta)", ok,
        sprintf("(KS p=%.3f; media=%.4f vs 1; var=%.4f vs %.4f)",
                ks$p.value, mean(xi_new), var(xi_new), 2 / nu))
  }
  
  ## --- V5: vectorizado == bucle observación-a-observación ----------------
  {
    nu <- 6
    e_list   <- list(c(0.3, -1.2, 2.0), c(-0.5, 0.8), numeric(0), c(1.1))
    vphi_list<- list(c(1.0,  2.0, 0.5), c( 3.0, 1.5), numeric(0), c(2.2))
    set.seed(999L)
    xi_vec <- muestrear_xi(e_list, vphi_list, nu)
    # Referencia: mismo orden de consumo del RNG, rgamma(1,.) por observación.
    set.seed(999L)
    xi_loop <- vector("list", length(e_list))
    a_xi <- (nu + 1) / 2
    for (tt in seq_along(e_list)) {
      e_t <- e_list[[tt]]; vphi_t <- vphi_list[[tt]]
      if (length(e_t) == 0L) { xi_loop[[tt]] <- numeric(0); next }
      out <- numeric(length(e_t))
      # rgamma vectorizado consume el stream elemento a elemento: se replica
      # con una sola llamada rgamma(N,...) para igualar exactamente.
      rate_t <- nu / 2 + 0.5 * vphi_t * e_t^2
      out <- rgamma(length(e_t), shape = a_xi, rate = rate_t)
      xi_loop[[tt]] <- pmax(out, .Machine$double.xmin)
    }
    d <- max(abs(unlist(xi_vec) - unlist(xi_loop)))
    reg("V5 muestreo vectorizado reproducible", d == 0,
        sprintf("(max|dif| = %.1e)", d))
  }
  
  ## --- V6: helpers residuos_por_tiempo y construir_vphi_list -------------
  {
    p <- 2L; Tt <- 2L
    theta <- matrix(c(0,0,  1,0,  2,1), nrow = p, ncol = Tt + 1L)  # cols: t=0,1,2
    y_list <- list(c(10, 20), c(5))
    F_list <- list(matrix(c(1,1, 2,3), nrow = 2, ncol = 2, byrow = TRUE),
                   matrix(c(1, 4), nrow = 1, ncol = 2, byrow = TRUE))
    # e_t = y_t - F_t %*% theta_t.  theta_1 = (1,0); theta_2 = (2,1).
    #   t=1: F=[[1,1],[2,3]], theta_1=(1,0) -> Fth=(1,2); e=(10-1,20-2)=(9,18)
    #   t=2: F=[[1,4]],       theta_2=(2,1) -> Fth=(6);   e=(5-6)=(-1)
    e_l <- residuos_por_tiempo(y_list, F_list, theta)
    e_ok <- isTRUE(all.equal(e_l[[1]], c(9, 18))) &&
      isTRUE(all.equal(e_l[[2]], c(-1)))
    # construir_vphi_list
    orden <- list(rbind(c(1L,1L,1L), c(1L,2L,1L)),   # t=1: dos celdas, 1 obs c/u
                  rbind(c(1L,1L,1L)))                # t=2: una celda, 1 obs
    v_curr   <- matrix(c(2, 3), nrow = 2, ncol = 1)  # v[1,1]=2, v[2,1]=3
    phi_curr <- matrix(c(5, 7), nrow = 1, ncol = 2)  # phi[1,1]=5, phi[1,2]=7
    vphi <- construir_vphi_list(orden, v_curr, phi_curr, Tt)
    # t=1: celda(k1,j1): 2*5=10 ; celda(k1,j2): 3*5=15.  t=2: celda(k1,j1): 2*7=14.
    vphi_ok <- isTRUE(all.equal(vphi[[1]], c(10, 15))) &&
      isTRUE(all.equal(vphi[[2]], c(14)))
    reg("V6 helpers residuos / vphi", e_ok && vphi_ok)
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
  ok_etapa3 <- verificar_etapa3()

