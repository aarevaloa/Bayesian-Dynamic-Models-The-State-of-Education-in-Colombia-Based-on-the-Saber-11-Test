# =============================================================================
#  waic.R  —  Criterios de informacion (WAIC) para los cuatro modelos
# =============================================================================

.log_sum_exp <- function(x) {
  m <- max(x)
  m + log(sum(exp(x - m)))
}

calcular_waic <- function(loglik) {
  stopifnot(is.matrix(loglik))
  S <- nrow(loglik); N <- ncol(loglik)
  lppd_i  <- numeric(N)
  pwaic_i <- numeric(N)
  for (i in 1:N) {
    col <- loglik[, i]
    col <- col[is.finite(col)]
    Si <- length(col)
    if (Si < 2) { lppd_i[i] <- NA_real_; pwaic_i[i] <- NA_real_; next }
    lppd_i[i]  <- .log_sum_exp(col) - log(Si)
    pwaic_i[i] <- var(col)
  }
  lppd   <- sum(lppd_i,  na.rm = TRUE)
  p_waic <- sum(pwaic_i, na.rm = TRUE)
  elpd_waic <- lppd - p_waic
  waic <- -2 * elpd_waic
  list(waic = waic, lppd = lppd, p_waic = p_waic, elpd_waic = elpd_waic,
       elpd_i = lppd_i - pwaic_i, pwaic_i = pwaic_i,
       n_obs = N, n_draws = S)
}


# -----------------------------------------------------------------------------
# [A.1] Densidades log puntuales (normal y t de Student)
# -----------------------------------------------------------------------------
# log N(y | mu, sigma2)
.logdnorm <- function(y, mu, sigma2) {
  -0.5 * (log(2 * pi) + log(sigma2) + (y - mu)^2 / sigma2)
}
# log t_nu(y | mu, sigma2) con sigma2 la VARIANZA de escala (no la varianza de la
# t). La t se parametriza como y = mu + sqrt(sigma2) * t_nu, de modo que la
# densidad es la t de localizacion-escala estandar.
.logdt <- function(y, mu, sigma2, nu) {
  z2 <- (y - mu)^2 / sigma2
  lgamma((nu + 1) / 2) - lgamma(nu / 2) -
    0.5 * (log(nu) + log(pi) + log(sigma2)) -
    (nu + 1) / 2 * log1p(z2 / nu)
}


# -----------------------------------------------------------------------------
# [A.2] Reconstruccion de la prediccion lineal por observacion
# -----------------------------------------------------------------------------
.matriz_diseno <- function(panel, theta_labels) {
  N <- nrow(panel)
  # columnas de covariables presentes en el panel (excluyen i,j,k,t,y,outlier)
  no_cov <- c("i","j","k","t","y","outlier")
  cols_cov <- setdiff(names(panel), no_cov)
  # quitar columnas "_bar" (medias de grupo) que no son covariables del modelo
  cols_cov <- cols_cov[!grepl("_bar$", cols_cov)]
  X <- matrix(1.0, nrow = N, ncol = 1)  # intercepto
  colnames(X) <- "intercepto"
  if (length(cols_cov) > 0) {
    Xc <- as.matrix(panel[, cols_cov, drop = FALSE])
    X <- cbind(X, Xc)
  }
  # alinear al orden de theta_labels si estan disponibles
  if (!is.null(theta_labels) && all(theta_labels %in% colnames(X))) {
    X <- X[, theta_labels, drop = FALSE]
  }
  X
}


# -----------------------------------------------------------------------------
# [A.3] loglik_pointwise(): matriz S x N de log-verosimilitud puntual
# -----------------------------------------------------------------------------
loglik_pointwise <- function(fit, panel, truth = NULL) {
  modelo   <- fit$extra$modelo
  estatico <- isTRUE(fit$extra$estatico)
  theta    <- fit$theta_draws           # p x T x S
  labels   <- fit$theta_labels
  if (is.null(labels)) labels <- fit$extra$theta_labels
  S <- dim(theta)[3]
  Tt <- dim(theta)[2]
  N <- nrow(panel)

  X <- .matriz_diseno(panel, labels)     # N x p
  p <- ncol(X)
  if (dim(theta)[1] != p)
    stop(sprintf("loglik_pointwise: dim theta (%d) != columnas X (%d) para %s",
                 dim(theta)[1], p, modelo))

  y  <- panel$y
  tt <- if ("t" %in% names(panel)) panel$t else rep(1L, N)
  jj <- panel$j; kk <- panel$k
  es_t <- modelo %in% c("M2", "M4")
  nu_draws <- if (es_t) fit$extra$nu else NULL

  loglik <- matrix(NA_real_, nrow = S, ncol = N)

  # --- precomputar indices de periodo (1..T) por observacion ---
  # en estaticos, todas las obs usan el unico periodo (col 1 de theta)
  per_idx <- if (estatico || Tt == 1) rep(1L, N) else as.integer(tt)

  for (s in 1:S) {
    # matriz de coeficientes de esta muestra: p x T
    th_s <- matrix(theta[, , s], nrow = p, ncol = Tt)
    # media por observacion: usar la columna del periodo correspondiente
    mu <- numeric(N)
    # para eficiencia, agrupar por periodo
    for (per in unique(per_idx)) {
      filas <- which(per_idx == per)
      mu[filas] <- as.numeric(X[filas, , drop = FALSE] %*% th_s[, per])
    }
    # varianza por observacion
    sigma2 <- .varianza_obs(fit, modelo, estatico, s, jj, kk, tt, N)

    if (es_t) {
      nu_s <- if (!is.null(nu_draws)) nu_draws[s] else NA_real_
      if (is.na(nu_s)) nu_s <- 1e6   # si no hay nu, ~ normal
      loglik[s, ] <- .logdt(y, mu, sigma2, nu_s)
    } else {
      loglik[s, ] <- .logdnorm(y, mu, sigma2)
    }
  }
  loglik
}


# -----------------------------------------------------------------------------
# [A.4] Varianza por observacion segun el modelo
# -----------------------------------------------------------------------------
.varianza_obs <- function(fit, modelo, estatico, s, jj, kk, tt, N) {
  if (estatico) {
    k2 <- fit$kappa2_jk_draws
    if (is.null(k2)) return(rep(1.0, N))  # fallback defensivo
    # k2 puede ser matriz S x C (C celdas) con columnas nombradas "k_j" o indice
    # Estrategia robusta: si es matriz, tomar la fila s y mapear por celda lineal.
    if (is.matrix(k2)) {
      # El adaptador numera las celdas en orden k=1..K, dentro j=1..J_k, con
      # departamentos y municipios ordenados numericamente (sort(unique)). Hay que
      # reproducir EXACTAMENTE ese orden para mapear cada observacion a su columna.
      cols <- colnames(k2)
      celdas_obs <- paste(kk, jj, sep = "_")
      if (!is.null(cols) && all(celdas_obs %in% cols)) {
        # si las columnas vienen nombradas, usar el nombre directamente
        return(as.numeric(k2[s, celdas_obs]))
      }
      # sin nombres: construir el indice de celda en el orden k->j del adaptador
      deptos <- sort(unique(kk))
      orden_celdas <- character(0)
      for (kd in deptos) {
        munis <- sort(unique(jj[kk == kd]))
        orden_celdas <- c(orden_celdas, paste(kd, munis, sep = "_"))
      }
      # mapa celda -> indice de columna
      idx_col <- match(celdas_obs, orden_celdas)
      idx_col[is.na(idx_col) | idx_col > ncol(k2)] <- ncol(k2)
      return(as.numeric(k2[s, idx_col]))
    }
    # si es vector (una sola celda o promedio), reciclar
    return(rep(as.numeric(k2[s]), N))
  } else {
    # dinamico: la escala de la t marginal difiere entre M3 y M4.
    #   M3: kappa2_jkt = s_kt / v_jk      (s_draws [depto,tiempo,S], v_draws [muni,depto,S])
    #   M4: kappa2_jkt = 1/(v_jk * phi_kt) (v_draws [muni,depto,S], phi_draws [depto,tiempo,S])
    # En M4, kappa2 es la ESCALA BASE de la t (las escalas latentes xi ya estan
    # integradas en la verosimilitud marginal, que es la correcta para el WAIC).
    es_m4 <- identical(modelo, "M4")
    v <- fit$v_draws
    w <- if (es_m4) fit$phi_draws else fit$s_draws   # phi (M4) o s (M3)
    if (is.null(v) || is.null(w)) return(rep(1.0, N))
    sigma2 <- numeric(N)
    for (i in 1:N) {
      vjk <- v[jj[i], kk[i], s]
      wkt <- w[kk[i], tt[i], s]
      if (es_m4) {
        # varianza = 1/(v * phi)
        sigma2[i] <- if (is.finite(vjk) && is.finite(wkt) && vjk > 0 && wkt > 0) 1 / (vjk * wkt) else 1.0
      } else {
        # varianza = s / v
        sigma2[i] <- if (is.finite(vjk) && is.finite(wkt) && vjk > 0) wkt / vjk else 1.0
      }
    }
    sigma2
  }
}


# -----------------------------------------------------------------------------
# [A.5] waic_modelo(): envoltorio que produce el WAIC de un fit
# -----------------------------------------------------------------------------
waic_modelo <- function(fit, panel, truth = NULL) {
  ll <- tryCatch(loglik_pointwise(fit, panel, truth),
                 error = function(e) {
                   message(sprintf("    [WAIC %s] error en loglik: %s",
                                   fit$extra$modelo, conditionMessage(e)))
                   NULL
                 })
  if (is.null(ll)) return(NULL)
  tryCatch(calcular_waic(ll),
           error = function(e) {
             message(sprintf("    [WAIC %s] error en calcular_waic: %s",
                             fit$extra$modelo, conditionMessage(e)))
             NULL
           })
}


# =============================================================================
# [C] VALIDACION
# =============================================================================
tabla_seleccion_waic <- function(tabla) {
  if (!"waic" %in% names(tabla)) {
    warning("la tabla no tiene columna 'waic'; corre el estudio con calcular_waic=TRUE")
    return(invisible(NULL))
  }
  ids <- unique(tabla$escenario)
  filas <- list()
  for (id in ids) {
    sub <- tabla[tabla$escenario == id, ]
    sub <- sub[is.finite(sub$waic), ]
    if (nrow(sub) == 0) next
    # modelo seleccionado por WAIC (menor WAIC)
    mod_waic <- sub$modelo[which.min(sub$waic)]
    # modelo correctamente especificado (etiqueta 'correcta')
    correctos <- sub$modelo[sub$especificacion == "correcta"]
    mod_correcto <- if (length(correctos) >= 1) correctos[1] else NA_character_
    # diferencia de WAIC entre el mejor y el segundo (margen de seleccion)
    waics_ord <- sort(sub$waic)
    margen <- if (length(waics_ord) >= 2) waics_ord[2] - waics_ord[1] else NA_real_
    filas[[length(filas)+1L]] <- data.frame(
      escenario = id,
      modelo_waic = mod_waic,
      modelo_correcto = mod_correcto,
      acierto = !is.na(mod_correcto) && mod_waic == mod_correcto,
      waic_min = waics_ord[1],
      margen_waic = margen,
      stringsAsFactors = FALSE)
  }
  if (length(filas) == 0) { warning("ningun escenario con WAIC y etiqueta valida"); return(invisible(NULL)) }
  out <- do.call(rbind, filas)
  out[order(out$escenario), ]
}

# -----------------------------------------------------------------------------
# [C.2] resumen_seleccion_waic(): tasa global de acierto del WAIC
# -----------------------------------------------------------------------------
resumen_seleccion_waic <- function(tabla, mapa_familia = NULL, verbose = TRUE) {
  sel <- tabla_seleccion_waic(tabla)
  if (is.null(sel)) return(invisible(NULL))
  # filtrar a escenarios con etiqueta 'correcta' definida (donde el acierto tiene sentido)
  sel_val <- sel[!is.na(sel$modelo_correcto), ]
  n <- nrow(sel_val)
  n_ok <- sum(sel_val$acierto)
  if (verbose) {
    cat(sprintf("\n=== Validacion del WAIC como selector de modelo ===\n"))
    cat(sprintf("Escenarios evaluados (con modelo correcto definido): %d\n", n))
    cat(sprintf("WAIC selecciono el modelo correcto en: %d / %d (%.0f%%)\n",
                n_ok, n, 100 * n_ok / max(n, 1)))
    # listar los desaciertos, si hay
    fallos <- sel_val[!sel_val$acierto, ]
    if (nrow(fallos) > 0) {
      cat("\nEscenarios donde el WAIC NO eligio el modelo correcto:\n")
      for (i in seq_len(nrow(fallos))) {
        cat(sprintf("  %s: WAIC eligio %s, correcto era %s (margen %.1f)\n",
                    fallos$escenario[i], fallos$modelo_waic[i],
                    fallos$modelo_correcto[i], fallos$margen_waic[i]))
      }
    } else {
      cat("\nEl WAIC acerto en todos los escenarios evaluados.\n")
    }
  }
  invisible(sel)
}
