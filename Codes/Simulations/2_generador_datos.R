#_______________________________________________________________________________
#@  CODIGO 1 — GENERADOR DE DATOS PANEL PARA VALIDACION DE LOS 4 MUESTREADORES
#@  Modelos:  M1 normal estatico | M2 t estatico | M3 normal dinamico | M4 t dinamico
#_______________________________________________________________________________

suppressMessages(suppressWarnings(library(MASS)))   # mvrnorm (covarianza Toeplitz)


# =============================================================================
# SECCION 0 — UTILIDADES DE PARAMETRIZACION
# =============================================================================

#' Inversa-Gamma via Gamma + reciproco.  X ~ IG(shape, scale) <=> 1/X ~ G(shape, rate=scale).
#' Replica exactamente la convencion de los samplers (rate = escala de la IG).
rinvgamma_rate <- function(n, shape, scale) {
  g <- rgamma(n, shape = shape, rate = scale)
  1 / g
}

#' Media geometrica estable (via media de logaritmos).
media_geometrica <- function(x) exp(mean(log(x)))

#' Matriz de covarianza Toeplitz:  Sigma_{ab} = rho^|a-b|.
toeplitz_cov <- function(p, rho) {
  if (p == 0) return(matrix(0, 0, 0))
  idx <- 0:(p - 1)
  rho ^ abs(outer(idx, idx, "-"))
}


# =============================================================================
# SECCION 1 — EL OBJETO `config`: TODOS LOS HIPERPARAMETROS VARIABLES
# =============================================================================

#' default_config: configuracion de REFERENCIA.
#'   (D,M_k,n_jk)=(15,10,50), balanceado, rho_X=0.3, nu_kappa=5, errores normales,
#'   p_E=5, sin contaminacion, dinamica estatica.  Cada escenario sobreescribe
#'   SOLO los campos que varia.
default_config <- function() {
  list(
    ## --- (1a) Estructura jerarquica y temporal ---
    D            = 15L,          # nº de departamentos
    M_k          = 10L,          # municipios por depto (escalar -> uniforme; o vector long D)
    n_jk         = 50L,          # individuos por municipio-periodo (escalar o estructura)
    n_periodos   = 1L,           # T. =1 cross-section (M1/M2); >1 panel (M3/M4)

    ## --- (1b) Desbalance poblacional (S04-S06) ---
    imbalance_dept   = "balanced",   # "balanced"|"concentrated"|"extreme"
    imbalance_muni   = "balanced",
    n_jk_varia_en_t  = FALSE,        # n_jk,t variable entre cortes
    permitir_celdas_vacias = FALSE,  # TRUE: un muni puede faltar en algun t
    prob_celda_vacia = 0.15,         # si se permite: prob de que (j,k) falte en un t

    ## --- (1c) Covariables ---
    p_E          = 5L,           # continuas individuales
    p_E_bin      = 0L,           # binarias individuales (Bernoulli)
    p_M          = 3L,           # municipales
    p_D          = 2L,           # departamentales
    rho_X        = 0.3,          # correlacion Toeplitz de continuas individuales
    prob_bin     = 0.4,          # p de las binarias individuales
    contextual_drift_sd = 0.0,   # 0 => contextuales fijas en t; >0 => deriva suave

    ## --- (1d) Coeficientes verdaderos y esparsidad ---
    effect_type_E = "mixed",     # "strong"|"weak"|"mixed"|"zero"|"signed"
    effect_type_M = "mixed",
    effect_type_D = "mixed",
    sparsity_E   = 0.0,          # proporcion EXACTA de coefs = 0 en el bloque
    sparsity_M   = 0.0,
    sparsity_D   = 0.0,
    beta_scale_E = 1.0,          # magnitud base de senales del bloque
    beta_scale_M = 1.0,
    beta_scale_D = 1.0,
    mu_true      = 0.0,          # intercepto verdadero base (mu_0)

    ## --- (1e) Dinamica de los coeficientes (T>1) ---
    dynamics       = "static",   # "static"|"rw"|"ar1"|"jump"
    sigma_omega    = 0.1,        # sd de innovaciones RW (lenta ~0.1, rapida ~0.5)
    ar1_phi        = 0.8,        # coef AR(1)
    jump_time      = NULL,       # periodo del salto (default: mitad del horizonte)
    jump_magnitude = 2.0,        # magnitud del salto (en SD de innovacion-equivalente)
    dyn_mu         = FALSE,      # si TRUE, el intercepto mu_t tambien evoluciona

    ## --- (1f) Varianzas: vista estatica = base (S07/S08) ---
    alpha_kappa  = 2.0,          # kappa2_k ~ G(alpha/2, beta/2) => E[kappa2_k]=alpha/beta
    beta_kappa   = 2.0,
    nu_kappa     = 5.0,          # IG municipal; >2 para media finita

    ## --- (1g) Dinamica de varianza (T>1; D07/D08) ---
    sigma2_dynamics = "static",  # "static"|"beta_gamma"|"rw_log"
    beta_k_discount = 0.95,      # factor de descuento (cadena beta-gamma)
    sigma_logs      = 0.2,       # sd del RW en log-escala (si rw_log)

    ## --- (1h) Errores y colas (S14/S15) ---
    error_family  = "normal",    # "normal"|"t"|"mixture"
    nu_error      = 5.0,         # gl de la t verdadera (>2)
    mixture_w     = 0.1,         # peso de la 2a componente (mezcla de normales)
    mixture_scale = 5.0,         # factor de varianza de la 2a componente

    ## --- (1i) Contaminacion por atipicos (S11-S13) ---
    contam_type      = "none",   # "none"|"y_sym"|"y_asym"|"context"|"X"
    contam_frac      = 0.0,      # 0.05 / 0.10 / 0.20
    contam_magnitude = 8.0,      # en SD locales
    contam_col_X     = 1L,       # cual covariable individual contaminar (si "X")

    ## --- (1j) Mala especificacion estructural (S19) ---
    random_intercepts_in_truth = FALSE,  # u_jk, v_k en la MEDIA (no modelados)
    sigma_u = 1.0,               # sd de interceptos aleatorios municipales
    sigma_v = 1.0,               # sd de interceptos aleatorios departamentales

    ## --- (1k) Reproducibilidad ---
    seed = NULL
  )
}


# =============================================================================
# SECCION 2 — VALIDACION DEL config
# =============================================================================

validar_config <- function(cfg) {
  err <- function(msg) stop(sprintf("[config] %s", msg), call. = FALSE)

  if (cfg$D < 1) err("D debe ser >= 1.")
  if (cfg$n_periodos < 1) err("n_periodos debe ser >= 1.")
  for (nm in c("sparsity_E", "sparsity_M", "sparsity_D")) {
    v <- cfg[[nm]]; if (v < 0 || v > 1) err(sprintf("%s debe estar en [0,1].", nm))
  }
  if (cfg$contam_frac < 0 || cfg$contam_frac > 1) err("contam_frac en [0,1].")
  if (cfg$beta_k_discount <= 0 || cfg$beta_k_discount > 1)
    err("beta_k_discount en (0,1].")
  if (cfg$error_family == "t" && cfg$nu_error <= 2)
    err("nu_error debe ser > 2 (varianza finita).")
  if (cfg$nu_kappa <= 2)
    warning("[config] nu_kappa <= 2: E[kappa2_jk] no finita (puede ser intencional).")
  if (!cfg$dynamics %in% c("static", "rw", "ar1", "jump"))
    err("dynamics invalido.")
  if (!cfg$error_family %in% c("normal", "t", "mixture"))
    err("error_family invalido.")
  if (!cfg$sigma2_dynamics %in% c("static", "beta_gamma", "rw_log"))
    err("sigma2_dynamics invalido.")
  if (!cfg$contam_type %in% c("none", "y_sym", "y_asym", "context", "X"))
    err("contam_type invalido.")
  for (nm in c("imbalance_dept", "imbalance_muni")) {
    if (!cfg[[nm]] %in% c("balanced", "concentrated", "extreme"))
      err(sprintf("%s invalido.", nm))
  }
  if (cfg$n_periodos == 1L && cfg$dynamics != "static")
    warning("[config] n_periodos=1 con dynamics!='static': la dinamica se ignora.")
  invisible(TRUE)
}


# =============================================================================
# SECCION 3 — ESTRUCTURA JERARQUICA  (D, M_k, n_jkt)
# =============================================================================

#' reparto_masa: devuelve un vector de tamaños (longitud n_unidades) segun el
#' regimen de balance, con suma aproximada a total_objetivo.
#'   balanced     -> uniforme
#'   concentrated -> proporcional a pesos Gamma(shape=2) decrecientes
#'   extreme      -> 10% de unidades ~70% de la masa; resto al minimo
reparto_masa <- function(n_unidades, total_objetivo, regimen, minimo = 1L) {
  if (n_unidades == 1L) return(as.integer(round(total_objetivo)))
  if (regimen == "balanced") {
    base <- rep(total_objetivo / n_unidades, n_unidades)
  } else if (regimen == "concentrated") {
    w <- sort(rgamma(n_unidades, shape = 2, rate = 1), decreasing = TRUE)
    base <- total_objetivo * w / sum(w)
  } else if (regimen == "extreme") {
    n_grandes <- max(1L, round(0.10 * n_unidades))
    masa_grande <- 0.70 * total_objetivo
    masa_resto  <- 0.30 * total_objetivo
    base <- numeric(n_unidades)
    base[seq_len(n_grandes)] <- masa_grande / n_grandes
    if (n_unidades > n_grandes)
      base[(n_grandes + 1L):n_unidades] <- masa_resto / (n_unidades - n_grandes)
  }
  out <- pmax(minimo, as.integer(round(base)))
  out
}

#' construir_estructura: resuelve M_k por depto, n_jk por municipio, y la
#' presencia de cada celda (j,k) en cada periodo t (matriz de conteos n_jkt).
construir_estructura <- function(cfg) {
  D  <- cfg$D
  Tt <- cfg$n_periodos

  # --- Municipios por departamento ---
  if (length(cfg$M_k) == 1L) {
    M_k <- reparto_masa(D, D * cfg$M_k, cfg$imbalance_dept, minimo = 1L)
  } else {
    stopifnot(length(cfg$M_k) == D); M_k <- as.integer(cfg$M_k)
  }
  n_cells <- sum(M_k)

  # --- Tamaño base por municipio (constante o de la masa) ---
  # n_jk_base[j,k]: individuos por celda-periodo "nominal".
  n_jk_base <- matrix(0L, nrow = max(M_k), ncol = D)
  for (kk in 1:D) {
    if (length(cfg$n_jk) == 1L) {
      tam_k <- reparto_masa(M_k[kk], M_k[kk] * cfg$n_jk, cfg$imbalance_muni, minimo = 1L)
    } else {
      stop("n_jk como estructura no escalar aun no soportado; usar escalar.")
    }
    n_jk_base[1:M_k[kk], kk] <- tam_k
  }

  # --- Conteos por celda-periodo n_jkt (j, k, t) ---
  n_jkt <- array(0L, dim = c(max(M_k), D, Tt))
  for (kk in 1:D) for (jj in 1:M_k[kk]) {
    base <- n_jk_base[jj, kk]
    for (tt in 1:Tt) {
      presente <- TRUE
      if (cfg$permitir_celdas_vacias && Tt > 1L)
        presente <- runif(1) > cfg$prob_celda_vacia
      if (!presente) { n_jkt[jj, kk, tt] <- 0L; next }
      if (cfg$n_jk_varia_en_t && Tt > 1L) {
        # variacion Poisson alrededor del tamaño base (minimo 1)
        n_jkt[jj, kk, tt] <- max(1L, rpois(1, base))
      } else {
        n_jkt[jj, kk, tt] <- base
      }
    }
    # Garantizar que el municipio aparece en AL MENOS un periodo
    if (sum(n_jkt[jj, kk, ]) == 0L)
      n_jkt[jj, kk, sample.int(Tt, 1)] <- base
  }

  # --- Conteos derivados ---
  N_jk <- matrix(0L, nrow = max(M_k), ncol = D)   # por municipio (sum sobre t)
  for (kk in 1:D) for (jj in 1:M_k[kk]) N_jk[jj, kk] <- sum(n_jkt[jj, kk, ])
  N_kt <- matrix(0L, nrow = D, ncol = Tt)         # por depto-periodo
  for (kk in 1:D) for (tt in 1:Tt) N_kt[kk, tt] <- sum(n_jkt[1:M_k[kk], kk, tt])
  N_t <- colSums(N_kt)                             # por periodo
  N_total <- sum(N_t)

  list(D = D, M_k = M_k, Tt = Tt, n_cells = n_cells,
       n_jk_base = n_jk_base, n_jkt = n_jkt,
       N_jk = N_jk, N_kt = N_kt, N_t = N_t, N_total = N_total,
       max_M = max(M_k))
}


# =============================================================================
# SECCION 4 — COVARIABLES
# =============================================================================

#' generar_X_individual: para CADA periodo t (cohortes nuevas), genera la matriz
#' de covariables individuales de las N_t personas de ese corte:
#'   - p_E continuas con covarianza Toeplitz(rho_X), estandarizadas.
#'   - p_E_bin binarias Bernoulli(prob_bin).
#' Devuelve lista de longitud Tt; X_ind[[t]] es matriz (N_t x (p_E+p_E_bin)) en el
#' MISMO orden de apilado que se usara para y (recorriendo k, luego j).
#' Tambien devuelve orden_celdas[[t]] = matriz (k, j, n_ijkt) por fila (clave para
#' los adaptadores dinamicos).
generar_X_individual <- function(cfg, estr) {
  Tt <- estr$Tt; D <- estr$D; M_k <- estr$M_k
  pE <- cfg$p_E; pEb <- cfg$p_E_bin
  Sigma <- toeplitz_cov(pE, cfg$rho_X)
  X_ind <- vector("list", Tt)
  orden_celdas <- vector("list", Tt)
  for (tt in 1:Tt) {
    filas_kj <- list(); blocks <- list(); pos <- 0L
    for (kk in 1:D) for (jj in 1:M_k[kk]) {
      n_ijkt <- estr$n_jkt[jj, kk, tt]
      if (n_ijkt == 0L) next
      Xc <- if (pE > 0) MASS::mvrnorm(n_ijkt, mu = rep(0, pE), Sigma = Sigma) else NULL
      # mvrnorm colapsa a vector cuando n_ijkt==1 (devuelve longitud pE) o cuando
      # pE==1 (devuelve longitud n_ijkt). En ambos casos forzamos matriz n_ijkt x pE.
      if (pE > 0) Xc <- matrix(Xc, nrow = n_ijkt, ncol = pE)
      Xb <- if (pEb > 0) matrix(rbinom(n_ijkt * pEb, 1, cfg$prob_bin), n_ijkt, pEb) else NULL
      Xi <- cbind(Xc, Xb)
      blocks[[length(blocks) + 1L]] <- Xi
      filas_kj[[length(filas_kj) + 1L]] <- c(kk, jj, n_ijkt)
      pos <- pos + n_ijkt
    }
    Xmat <- if (length(blocks) > 0) do.call(rbind, blocks) else matrix(0, 0, pE + pEb)
    # estandarizar continuas (columnas 1:pE) sobre el corte
    if (pE > 0 && nrow(Xmat) > 1) {
      for (c in 1:pE) {
        s <- sd(Xmat[, c]); if (is.finite(s) && s > 1e-12) Xmat[, c] <- (Xmat[, c] - mean(Xmat[, c])) / s
      }
    }
    X_ind[[tt]] <- Xmat
    orden_celdas[[tt]] <- if (length(filas_kj) > 0) do.call(rbind, filas_kj) else matrix(0L, 0, 3)
  }
  list(X_ind = X_ind, orden_celdas = orden_celdas,
       nombres_E = c(if (cfg$p_E > 0) paste0("xE", seq_len(cfg$p_E)),
                     if (cfg$p_E_bin > 0) paste0("xEbin", seq_len(cfg$p_E_bin))))
}

#' generar_Z_municipal: covariables municipales CONSTANTES dentro de (j,k) en un
#' corte. Valor base z_jk; si drift>0, z_jk,t = z_jk + deriva suave. Devuelve el
#' array por periodo z_jkt (max_M x D x p_M x Tt) y el promedio temporal z_bar.
generar_Z_municipal <- function(cfg, estr) {
  D <- estr$D; M_k <- estr$M_k; Tt <- estr$Tt; pM <- cfg$p_M
  if (pM == 0) return(list(z_jkt = NULL, z_bar = NULL, nombres_M = character(0)))
  z_base <- array(rnorm(estr$max_M * D * pM), dim = c(estr$max_M, D, pM))
  z_jkt  <- array(NA_real_, dim = c(estr$max_M, D, pM, Tt))
  for (kk in 1:D) for (jj in 1:M_k[kk]) for (cc in 1:pM) {
    base <- z_base[jj, kk, cc]
    if (cfg$contextual_drift_sd > 0 && Tt > 1) {
      drift <- cumsum(rnorm(Tt, 0, cfg$contextual_drift_sd))
      z_jkt[jj, kk, cc, ] <- base + drift
    } else {
      z_jkt[jj, kk, cc, ] <- base
    }
  }
  # promedio temporal por celda (para el colapso estatico)
  z_bar <- apply(z_jkt, c(1, 2, 3), mean)   # max_M x D x p_M
  list(z_jkt = z_jkt, z_bar = z_bar, nombres_M = paste0("zM", seq_len(pM)))
}

#' generar_W_departamental: analogo a Z pero a nivel departamento (constante en k).
generar_W_departamental <- function(cfg, estr) {
  D <- estr$D; Tt <- estr$Tt; pD <- cfg$p_D
  if (pD == 0) return(list(w_kt = NULL, w_bar = NULL, nombres_D = character(0)))
  w_base <- matrix(rnorm(D * pD), nrow = D, ncol = pD)
  w_kt <- array(NA_real_, dim = c(D, pD, Tt))
  for (kk in 1:D) for (cc in 1:pD) {
    base <- w_base[kk, cc]
    if (cfg$contextual_drift_sd > 0 && Tt > 1) {
      drift <- cumsum(rnorm(Tt, 0, cfg$contextual_drift_sd))
      w_kt[kk, cc, ] <- base + drift
    } else {
      w_kt[kk, cc, ] <- base
    }
  }
  w_bar <- apply(w_kt, c(1, 2), mean)   # D x p_D
  list(w_kt = w_kt, w_bar = w_bar, nombres_D = paste0("wD", seq_len(pD)))
}


# =============================================================================
# SECCION 5 — COEFICIENTES VERDADEROS theta_t = [mu_t, beta_E,t, beta_M,t, beta_D,t]
# =============================================================================

#' construir_bloque_beta: vector de coeficientes base de UN bloque, segun patron y
#' esparsidad. Devuelve list(beta, is_zero).
construir_bloque_beta <- function(p, effect_type, sparsity, scale) {
  if (p == 0) return(list(beta = numeric(0), is_zero = logical(0)))
  beta <- switch(effect_type,
    strong = rep(scale * 2.0, p),
    weak   = rep(scale * 0.3, p),
    mixed  = scale * seq(0.3, 2.0, length.out = p),
    zero   = rep(0, p),
    signed = scale * rep(c(1, -1), length.out = p) * seq(0.5, 2.0, length.out = p)
  )
  # imponer esparsidad EXACTA: fija floor(sparsity*p) coeficientes a 0
  n_cero <- floor(sparsity * p)
  is_zero <- rep(FALSE, p)
  if (n_cero > 0) {
    idx0 <- sample.int(p, n_cero)
    beta[idx0] <- 0
    is_zero[idx0] <- TRUE
  }
  is_zero <- is_zero | (beta == 0)
  list(beta = beta, is_zero = is_zero)
}

#' construir_beta_base: ensambla theta_0 = [mu, beta_E, beta_M, beta_D] y el
#' indicador de esparsidad por componente (para sensibilidad/especificidad LASSO).
construir_beta_base <- function(cfg) {
  bE <- construir_bloque_beta(cfg$p_E + cfg$p_E_bin, cfg$effect_type_E, cfg$sparsity_E, cfg$beta_scale_E)
  bM <- construir_bloque_beta(cfg$p_M, cfg$effect_type_M, cfg$sparsity_M, cfg$beta_scale_M)
  bD <- construir_bloque_beta(cfg$p_D, cfg$effect_type_D, cfg$sparsity_D, cfg$beta_scale_D)
  theta0 <- c(cfg$mu_true, bE$beta, bM$beta, bD$beta)
  is_zero <- c(FALSE, bE$is_zero, bM$is_zero, bD$is_zero)  # intercepto nunca "cero-LASSO"
  list(theta0 = theta0, is_zero = is_zero,
       p = length(theta0), p_E = cfg$p_E + cfg$p_E_bin, p_M = cfg$p_M, p_D = cfg$p_D)
}

#' evolucionar_theta: produce theta_t (p x T) segun el regimen dinamico.
#' El intercepto (fila 1) solo evoluciona si cfg$dyn_mu=TRUE.
evolucionar_theta <- function(theta0, cfg, estr) {
  p <- length(theta0); Tt <- estr$Tt
  theta <- matrix(theta0, nrow = p, ncol = Tt)   # default: estatico (todas las cols = theta0)
  if (Tt == 1L || cfg$dynamics == "static") return(theta)

  idx_dyn <- if (cfg$dyn_mu) 1:p else 2:p   # cuales coordenadas evolucionan
  jt <- if (is.null(cfg$jump_time)) ceiling(Tt / 2) else cfg$jump_time

  for (tt in 2:Tt) {
    prev <- theta[, tt - 1]
    nxt  <- prev
    if (cfg$dynamics == "rw") {
      nxt[idx_dyn] <- prev[idx_dyn] + rnorm(length(idx_dyn), 0, cfg$sigma_omega)
    } else if (cfg$dynamics == "ar1") {
      # AR(1) hacia 0:  theta_t = phi*theta_{t-1} + N(0,sigma^2)
      nxt[idx_dyn] <- cfg$ar1_phi * prev[idx_dyn] + rnorm(length(idx_dyn), 0, cfg$sigma_omega)
    } else if (cfg$dynamics == "jump") {
      # RW suave + salto puntual en jt
      nxt[idx_dyn] <- prev[idx_dyn] + rnorm(length(idx_dyn), 0, cfg$sigma_omega)
      if (tt == jt)
        nxt[idx_dyn] <- nxt[idx_dyn] + cfg$jump_magnitude * cfg$sigma_omega *
                        sample(c(-1, 1), length(idx_dyn), replace = TRUE)
    }
    theta[, tt] <- nxt
  }
  theta
}


# =============================================================================
# SECCION 6 — VARIANZA-VERDAD  (UNA realidad fisica; dos vistas)
# =============================================================================

#' generar_varianza_base_estatica: la jerarquia que consumen M1/M2.
#'   kappa2_k  ~ G(alpha/2, beta/2)                  [longitud D]
#'   kappa2_jk ~ IG(nu_k/2, nu_k*kappa2_k/2)         [por celda (j,k)]
#' Devuelve kappa2_k (vector D) y kappa2_jk (matriz max_M x D, NA en celdas j>M_k).
generar_varianza_base_estatica <- function(cfg, estr) {
  D <- estr$D; M_k <- estr$M_k
  kappa2_k <- rgamma(D, shape = cfg$alpha_kappa / 2, rate = cfg$beta_kappa / 2)
  kappa2_jk <- matrix(NA_real_, nrow = estr$max_M, ncol = D)
  for (kk in 1:D) {
    kappa2_jk[1:M_k[kk], kk] <- rinvgamma_rate(
      M_k[kk], shape = cfg$nu_kappa / 2, scale = cfg$nu_kappa * kappa2_k[kk] / 2)
  }
  list(kappa2_k = kappa2_k, kappa2_jk = kappa2_jk)
}

#' generar_factor_dinamico_varianza: factor d_kt (D x T) con MEDIA ~1 por depto.
#'   static     -> d_kt = 1  (varianza constante en t; rama dinamica == estatica)
#'   beta_gamma -> cadena West&Harrison: phi_kt = phi_{k,t-1}*eta/beta;
#'                 eta~Beta(beta*n/2,(1-beta)*n/2); s_kt=1/phi; d_kt = s_kt/mean_t(s_kt)
#'   rw_log     -> log d_kt = RW(0,sigma_logs) recentrado a media 0 (=> media d ~1)
generar_factor_dinamico_varianza <- function(cfg, estr) {
  D <- estr$D; Tt <- estr$Tt
  d_kt <- matrix(1.0, nrow = D, ncol = Tt)
  if (Tt == 1L || cfg$sigma2_dynamics == "static") return(d_kt)

  if (cfg$sigma2_dynamics == "rw_log") {
    for (kk in 1:D) {
      lg <- cumsum(rnorm(Tt, 0, cfg$sigma_logs))
      lg <- lg - mean(lg)                 # centrar => media geometrica 1
      d_kt[kk, ] <- exp(lg)
    }
  } else if (cfg$sigma2_dynamics == "beta_gamma") {
    beta_k <- cfg$beta_k_discount
    n0 <- 8.0                              # gl nominal del prior (forma)
    for (kk in 1:D) {
      phi <- numeric(Tt)
      phi[1] <- rgamma(1, shape = n0 / 2, rate = n0 / 2)   # E[phi]=1
      if (Tt >= 2) for (tt in 2:Tt) {
        n_prev <- n0
        a_eta <- beta_k * n_prev / 2
        b_eta <- (1 - beta_k) * n_prev / 2
        eta <- if (b_eta > 0) rbeta(1, a_eta, b_eta) else 1.0
        phi[tt] <- phi[tt - 1] * eta / beta_k
      }
      s <- 1 / phi
      d_kt[kk, ] <- s / mean(s)            # normalizar a media 1 (separar nivel de forma)
    }
  }
  d_kt
}

#' componer_kappa2_jkt: VERDAD FISICA. kappa2_jkt = kappa2_jk * d_kt (broadcast).
#' En celdas (j,k,t) sin observaciones (n_jkt=0) se deja el valor (esta definido)
#' pero el truth marcara NA donde no hay datos.
componer_kappa2_jkt <- function(var_base, d_kt, estr) {
  D <- estr$D; M_k <- estr$M_k; Tt <- estr$Tt
  k2 <- array(NA_real_, dim = c(estr$max_M, D, Tt))
  for (kk in 1:D) for (jj in 1:M_k[kk]) for (tt in 1:Tt) {
    k2[jj, kk, tt] <- var_base$kappa2_jk[jj, kk] * d_kt[kk, tt]
  }
  k2
}

derivar_vista_dinamica <- function(var_base, d_kt, estr) {
  D <- estr$D; M_k <- estr$M_k; Tt <- estr$Tt
  v_jk <- matrix(NA_real_, nrow = estr$max_M, ncol = D)
  s_kt <- matrix(NA_real_, nrow = D, ncol = Tt)
  for (kk in 1:D) {
    k2jk_k <- var_base$kappa2_jk[1:M_k[kk], kk]
    v_raw  <- 1 / k2jk_k
    v_bar  <- media_geometrica(v_raw)
    v_jk[1:M_k[kk], kk] <- v_raw / v_bar       # sum_j log v_jk = 0
    s_kt[kk, ] <- d_kt[kk, ] / v_bar           # s_kt/v_jk = kappa2_jk*d_kt = kappa2_jkt
  }
  list(v_jk = v_jk, s_kt = s_kt)
}


# =============================================================================
# SECCION 7 — ERRORES Y RESPUESTA  y_ijkt
# =============================================================================

#' ensamblar_y: recorre periodos y celdas, construye zeta = F'theta_t, genera el
#' error segun error_family, aplica (opcional) interceptos aleatorios y
#' contaminacion, y devuelve un data.frame largo + estructuras de truth (xi,
#' outlier_flag). Apilado consistente con orden_celdas (k, luego j).
ensamblar_y <- function(theta, Xobj, Zobj, Wobj, k2jkt, vista_dyn, cfg, estr) {
  D <- estr$D; M_k <- estr$M_k; Tt <- estr$Tt
  pE_all <- cfg$p_E + cfg$p_E_bin; pM <- cfg$p_M; pD <- cfg$p_D
  idx_E <- if (pE_all > 0) 2:(1 + pE_all) else integer(0)
  idx_M <- if (pM > 0) (1 + pE_all + 1):(1 + pE_all + pM) else integer(0)
  idx_D <- if (pD > 0) (1 + pE_all + pM + 1):(1 + pE_all + pM + pD) else integer(0)

  # interceptos aleatorios en la MEDIA (mala especificacion estructural)
  u_jk <- v_k <- NULL
  if (cfg$random_intercepts_in_truth) {
    u_jk <- matrix(rnorm(estr$max_M * D, 0, cfg$sigma_u), estr$max_M, D)
    v_k  <- rnorm(D, 0, cfg$sigma_v)
  }

  filas <- list(); xi_list <- vector("list", Tt); fila_global <- 0L
  outlier_all <- logical(0)
  for (tt in 1:Tt) {
    oc <- Xobj$orden_celdas[[tt]]
    if (nrow(oc) == 0L) { xi_list[[tt]] <- numeric(0); next }
    Xt <- Xobj$X_ind[[tt]]
    N_t <- sum(oc[, 3L])
    y_t <- numeric(N_t); xi_t <- rep(1.0, N_t); out_t <- logical(N_t)
    k_t <- integer(N_t); j_t <- integer(N_t)
    pos <- 0L
    th  <- theta[, tt]
    for (r in 1:nrow(oc)) {
      kk <- oc[r, 1L]; jj <- oc[r, 2L]; n_ijkt <- oc[r, 3L]
      rng <- (pos + 1L):(pos + n_ijkt)
      # media condicional zeta = mu + x'bE + z'bM + w'bD
      zeta <- rep(th[1], n_ijkt)
      if (pE_all > 0) zeta <- zeta + as.vector(Xt[rng, , drop = FALSE] %*% th[idx_E])
      if (pM > 0) {
        zval <- Zobj$z_jkt[jj, kk, , tt]                 # p_M
        zeta <- zeta + sum(zval * th[idx_M])
      }
      if (pD > 0) {
        wval <- Wobj$w_kt[kk, , tt]
        zeta <- zeta + sum(wval * th[idx_D])
      }
      if (cfg$random_intercepts_in_truth) zeta <- zeta + u_jk[jj, kk] + v_k[kk]
      # error con varianza kappa2_jkt
      k2 <- k2jkt[jj, kk, tt]
      if (cfg$error_family == "normal") {
        eps <- rnorm(n_ijkt, 0, sqrt(k2))
      } else if (cfg$error_family == "t") {
        xi <- rgamma(n_ijkt, shape = cfg$nu_error / 2, rate = cfg$nu_error / 2)
        eps <- rnorm(n_ijkt, 0, sqrt(k2 / xi))
        xi_t[rng] <- xi
      } else if (cfg$error_family == "mixture") {
        comp2 <- runif(n_ijkt) < cfg$mixture_w
        sds <- ifelse(comp2, sqrt(k2 * cfg$mixture_scale), sqrt(k2))
        eps <- rnorm(n_ijkt, 0, sds)
      }
      y_t[rng] <- zeta + eps
      k_t[rng] <- kk; j_t[rng] <- jj
      pos <- pos + n_ijkt
    }
    # --- contaminacion por atipicos (sobre el corte t) ---
    if (cfg$contam_type != "none" && cfg$contam_frac > 0) {
      sd_local <- sd(y_t)
      if (cfg$contam_type %in% c("y_sym", "y_asym")) {
        n_out <- floor(cfg$contam_frac * N_t)
        if (n_out > 0) {
          idx_out <- sample.int(N_t, n_out)
          signo <- if (cfg$contam_type == "y_sym")
            sample(c(-1, 1), n_out, replace = TRUE) else 1
          y_t[idx_out] <- y_t[idx_out] + signo * cfg$contam_magnitude * sd_local
          out_t[idx_out] <- TRUE
        }
      } else if (cfg$contam_type == "context") {
        # municipios COMPLETOS contaminados
        celdas <- unique(cbind(k_t, j_t))
        n_cel_out <- floor(cfg$contam_frac * nrow(celdas))
        if (n_cel_out > 0) {
          sel <- celdas[sample.int(nrow(celdas), n_cel_out), , drop = FALSE]
          for (rr in 1:nrow(sel)) {
            mask <- (k_t == sel[rr, 1]) & (j_t == sel[rr, 2])
            y_t[mask] <- y_t[mask] + cfg$contam_magnitude * sd_local
            out_t[mask] <- TRUE
          }
        }
      }
      # contam_type=="X" se aplica fuera (perturba una covariable); ver nota
    }
    xi_list[[tt]] <- xi_t
    df_t <- data.frame(
      i = seq_len(N_t), j = j_t, k = k_t, t = tt, y = y_t,
      outlier = out_t
    )
    # adjuntar X individual (mismas filas, mismo orden)
    if (pE_all > 0) {
      Xdf <- as.data.frame(Xt); names(Xdf) <- Xobj$nombres_E
      df_t <- cbind(df_t, Xdf)
    }
    filas[[tt]] <- df_t
    outlier_all <- c(outlier_all, out_t)
    fila_global <- fila_global + N_t
  }
  panel <- do.call(rbind, filas)

  # contam_type=="X": contaminacion de COVARIABLE (errores de medicion en un
  # predictor). Se aplica aqui, sobre el panel ya ensamblado, perturbando una
  # fraccion de las filas de la covariable individual contam_col_X en +-magnitud
  # SD locales. Marca esas filas como outlier. Es independiente de la familia de
  # errores (no toca y), evaluando robustez frente a ruido en X.
  if (cfg$contam_type == "X" && cfg$contam_frac > 0 && cfg$p_E > 0) {
    col_idx <- max(1L, min(cfg$contam_col_X, cfg$p_E))
    col_nm  <- Xobj$nombres_E[col_idx]
    if (!is.null(col_nm) && col_nm %in% names(panel)) {
      n_tot <- nrow(panel)
      n_out <- floor(cfg$contam_frac * n_tot)
      if (n_out > 0) {
        idx_out <- sample.int(n_tot, n_out)
        sd_col  <- sd(panel[[col_nm]])
        signo   <- sample(c(-1, 1), n_out, replace = TRUE)  # simetrico en X
        panel[[col_nm]][idx_out] <- panel[[col_nm]][idx_out] +
          signo * cfg$contam_magnitude * sd_col
        panel$outlier[idx_out] <- TRUE
      }
    }
  }

  # adjuntar contextuales por fila (valor del periodo t y promedio temporal)
  if (pM > 0) for (cc in 1:pM) {
    panel[[Zobj$nombres_M[cc]]]            <- Zobj$z_jkt[cbind(panel$j, panel$k, cc, panel$t)]
    panel[[paste0(Zobj$nombres_M[cc], "_bar")]] <- Zobj$z_bar[cbind(panel$j, panel$k, cc)]
  }
  if (pD > 0) for (cc in 1:pD) {
    panel[[Wobj$nombres_D[cc]]]            <- Wobj$w_kt[cbind(panel$k, cc, panel$t)]
    panel[[paste0(Wobj$nombres_D[cc], "_bar")]] <- Wobj$w_bar[cbind(panel$k, cc)]
  }
  list(panel = panel, xi_list = xi_list, u_jk = u_jk, v_k = v_k)
}


# =============================================================================
# SECCION 9 — ARMADO DEL OBJETO `truth`
# =============================================================================

armar_truth <- function(cfg, estr, beta_obj, theta, var_base, d_kt, k2jkt,
                        vista_dyn, ens, Xobj, Zobj, Wobj) {
  theta_bar <- rowMeans(theta)            # verdad estatica equivalente (promedio temporal)
  theta_labels <- c("intercepto", Xobj$nombres_E, Zobj$nombres_M, Wobj$nombres_D)
  rownames(theta) <- theta_labels

  list(
    ## --- coeficientes ---
    theta        = theta,                 # p x T  (verdad por periodo)
    theta_bar    = theta_bar,             # promedio temporal (verdad estatica equiv.)
    theta_T      = theta[, estr$Tt],      # ultimo periodo (referencia alternativa)
    theta0       = beta_obj$theta0,       # base
    beta_is_zero = beta_obj$is_zero,      # indicador de esparsidad (sens/espec LASSO)
    ## --- varianzas: vista estatica (M1/M2) ---
    kappa2_k     = var_base$kappa2_k,     # nivel departamental
    kappa2_jk    = var_base$kappa2_jk,    # nivel municipal
    ## --- varianza fisica comun a los 4 modelos ---
    kappa2_jkt   = k2jkt,                 # s_kt/v_jk (VERDAD comun)
    d_kt         = d_kt,                  # factor dinamico de varianza
    ## --- varianzas: vista dinamica RECENTRADA (M3/M4) ---
    v_jk         = vista_dyn$v_jk,        # precision municipal (geom-centrada)
    s_kt         = vista_dyn$s_kt,        # escala departamental dinamica
    ## --- robustez ---
    nu_error     = if (cfg$error_family == "t") cfg$nu_error else NA_real_,
    xi_list      = ens$xi_list,           # escalas latentes verdaderas (M2/M4)
    ## --- contaminacion / mala especificacion ---
    outlier_flag = ens$panel$outlier,
    ranef        = list(u_jk = ens$u_jk, v_k = ens$v_k),
    ## --- metadatos ---
    sizes  = list(D = estr$D, M_k = estr$M_k, n_jkt = estr$n_jkt, Tt = estr$Tt,
                  N_t = estr$N_t, N_total = estr$N_total,
                  p = beta_obj$p, p_E = beta_obj$p_E, p_M = beta_obj$p_M, p_D = beta_obj$p_D),
    labels = list(theta_labels = theta_labels,
                  nombres_E = Xobj$nombres_E, nombres_M = Zobj$nombres_M,
                  nombres_D = Wobj$nombres_D),
    orden_celdas = Xobj$orden_celdas,     # mapeo (k,j,n) por periodo (para adaptadores)
    config = cfg
  )
}


# =============================================================================
# SECCION 10 — ORQUESTADOR  simulate_panel(cfg)
# =============================================================================

#' simulate_panel: genera el panel canonico + el truth exhaustivo.
#' Devuelve list(panel = <data.frame largo>, truth = <list>).
simulate_panel <- function(cfg = default_config()) {
  validar_config(cfg)
  if (!is.null(cfg$seed)) set.seed(cfg$seed)

  estr      <- construir_estructura(cfg)
  Xobj      <- generar_X_individual(cfg, estr)
  Zobj      <- generar_Z_municipal(cfg, estr)
  Wobj      <- generar_W_departamental(cfg, estr)

  beta_obj  <- construir_beta_base(cfg)
  theta     <- evolucionar_theta(beta_obj$theta0, cfg, estr)

  var_base  <- generar_varianza_base_estatica(cfg, estr)
  d_kt      <- generar_factor_dinamico_varianza(cfg, estr)
  k2jkt     <- componer_kappa2_jkt(var_base, d_kt, estr)
  vista_dyn <- derivar_vista_dinamica(var_base, d_kt, estr)

  ens       <- ensamblar_y(theta, Xobj, Zobj, Wobj, k2jkt, vista_dyn, cfg, estr)
  truth     <- armar_truth(cfg, estr, beta_obj, theta, var_base, d_kt, k2jkt,
                           vista_dyn, ens, Xobj, Zobj, Wobj)

  list(panel = ens$panel, truth = truth)
}


# =============================================================================
# SECCION 11 — COLAPSO TEMPORAL PARA LOS MODELOS ESTATICOS
# =============================================================================

#' colapsar_a_estatico: produce la cross-section que consumen M1/M2.
#' CERO individuos promediados: conserva las N_total filas con su y real; elimina
#' la etiqueta de periodo y usa las CONTEXTUALES PROMEDIADAS (_bar) por celda.
colapsar_a_estatico <- function(panel, truth) {
  nm_E <- truth$labels$nombres_E
  nm_M <- truth$labels$nombres_M
  nm_D <- truth$labels$nombres_D
  out <- data.frame(j = panel$j, k = panel$k, y = panel$y)
  if (length(nm_E) > 0) out <- cbind(out, panel[, nm_E, drop = FALSE])
  # contextuales: usar las versiones _bar (promedio temporal por celda)
  for (v in nm_M) out[[v]] <- panel[[paste0(v, "_bar")]]
  for (v in nm_D) out[[v]] <- panel[[paste0(v, "_bar")]]
  out
}


# =============================================================================
# SECCION 12 — VERIFICACION DE COHERENCIA (auto-chequeo del generador)
# =============================================================================

#' verificar_coherencia: comprueba invariantes clave de un objeto sim.
verificar_coherencia <- function(sim, tol = 1e-8) {
  panel <- sim$panel; tr <- sim$truth; estr_sizes <- tr$sizes
  ok <- TRUE; msg <- function(t, c) { cat(sprintf("  [%s] %s\n", if (c) "OK" else "FALLA", t)); ok <<- ok & c }

  # (a) nº de filas = N_total
  msg("nrow(panel) == N_total", nrow(panel) == estr_sizes$N_total)

  # (b) la descomposicion s_kt/v_jk reproduce kappa2_jkt en celdas observadas
  D <- estr_sizes$D; M_k <- estr_sizes$M_k; Tt <- estr_sizes$Tt
  err_max <- 0
  for (kk in 1:D) for (jj in 1:M_k[kk]) for (tt in 1:Tt) {
    if (estr_sizes$n_jkt[jj, kk, tt] == 0L) next
    recon <- tr$s_kt[kk, tt] / tr$v_jk[jj, kk]
    err_max <- max(err_max, abs(recon - tr$kappa2_jkt[jj, kk, tt]))
  }
  msg(sprintf("s_kt/v_jk == kappa2_jkt (err max=%.2e)", err_max), err_max < tol)

  # (c) recentrado: media geometrica de v_jk = 1 por depto
  err_geom <- 0
  for (kk in 1:D) {
    vg <- media_geometrica(tr$v_jk[1:M_k[kk], kk])
    err_geom <- max(err_geom, abs(vg - 1))
  }
  msg(sprintf("media_geom(v_jk)=1 por depto (err max=%.2e)", err_geom), err_geom < tol)

  # (d) si dinamica estatica, kappa2_jkt constante en t (== kappa2_jk)
  if (tr$config$sigma2_dynamics == "static") {
    err_dt <- 0
    for (kk in 1:D) for (jj in 1:M_k[kk]) {
      vals <- tr$kappa2_jkt[jj, kk, ]; vals <- vals[!is.na(vals)]
      if (length(vals) > 1) err_dt <- max(err_dt, diff(range(vals)))
    }
    msg(sprintf("varianza estatica => kappa2_jkt const en t (err=%.2e)", err_dt), err_dt < tol)
  }

  # (e) esparsidad: nº de coeficientes cero coincide con el indicador
  msg("indicador de esparsidad coherente (longitud p)",
      length(tr$beta_is_zero) == estr_sizes$p)

  # (f) sin NA/Inf en y
  msg("y sin NA/Inf", all(is.finite(panel$y)))

  invisible(ok)
}
