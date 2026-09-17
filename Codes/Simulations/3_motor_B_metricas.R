#_______________________________________________________________________________
#@  CODIGO 2 — MOTOR DE SIMULACION
#@  Compara las cadenas de cada modelo contra el `truth` en la escala comun
#_______________________________________________________________________________
#
# Todas las metricas operan sobre el formato comun `fit` de la Parte A:
#   fit$theta_draws : array p x T x n_save
# y el `truth` del generador. Para modelos estaticos (T=1) la "verdad" de
# referencia es theta_bar (promedio temporal); para dinamicos, la trayectoria
# theta_t completa.
#
# Metricas implementadas:
#   - recuperacion de coeficientes: sesgo, sesgo relativo, RMSE, MAE, por bloque
#   - calibracion: cobertura IC95 y amplitud media, por bloque
#   - seleccion LASSO: sensibilidad y especificidad (coef cero vs activo)
#   - recuperacion de varianza: error relativo de kappa2 efectivo
#   - diagnostico: ESS aproximado, tiempo de computo
#_______________________________________________________________________________


# =============================================================================
# UTILIDADES
# =============================================================================

.especificacion_modelo <- function(modelo, truth) {
  cfg <- truth$config
  # caracteristicas del modelo
  modelo_t        <- modelo %in% c("M2", "M4")   # asume errores t
  modelo_dinamico <- modelo %in% c("M3", "M4")   # modela el tiempo
  # caracteristicas del escenario
  datos_t        <- cfg$error_family %in% c("t", "mixture")
  datos_dinamicos<- cfg$dynamics != "static"

  ok_colas    <- (modelo_t == datos_t)
  ok_temporal <- (modelo_dinamico == datos_dinamicos)

  ratio_teorico <- if (!ok_colas && !modelo_t && datos_t && !is.na(truth$nu_error)) {
    truth$nu_error / (truth$nu_error - 2)
  } else if (ok_colas) {
    1.0
  } else NA_real_   # mal especificado en colas pero en direccion no estandar

  estado <- if (ok_colas && ok_temporal) "correcta"
            else paste0("mal:",
                        paste(c(if (!ok_colas) "colas", if (!ok_temporal) "temporal"),
                              collapse = "+"))

  list(ok_colas = ok_colas, ok_temporal = ok_temporal,
       estado = estado, ratio_teorico = ratio_teorico,
       modelo_t = modelo_t, modelo_dinamico = modelo_dinamico,
       datos_t = datos_t, datos_dinamicos = datos_dinamicos)
}

.resumir_draws <- function(x) {
  c(media = mean(x), sd = sd(x),
    q025 = unname(quantile(x, 0.025)), q975 = unname(quantile(x, 0.975)))
}

.ess_aprox <- function(x) {
  n <- length(x); if (n < 8L || sd(x) == 0) return(NA_real_)
  ac <- acf(x, lag.max = min(n - 1L, 200L), plot = FALSE, demean = TRUE)$acf[, 1, 1]
  rho <- ac[-1]; suma <- 0; k <- 1L
  while (2L * k <= length(rho)) {
    par <- rho[2L*k - 1L] + rho[2L*k]; if (par <= 0) break
    suma <- suma + par; k <- k + 1L
  }
  min(n / max(1 + 2 * suma, 1e-8), n)
}


# =============================================================================
# 1. RECUPERACION DE COEFICIENTES  (sesgo, RMSE, MAE, cobertura)
# =============================================================================
metricas_coeficientes <- function(fit, truth, ref_estatico = "theta_bar") {
  td <- fit$theta_draws          # p x T x n_save
  p <- dim(td)[1]; Tt_fit <- dim(td)[2]
  labels <- fit$theta_labels
  es_estatico <- fit$extra$estatico

  # construir la matriz de verdad p x T_ref alineada con el fit
  if (es_estatico) {
    verdad_mat <- matrix(if (ref_estatico == "theta_T") truth$theta_T else truth$theta_bar,
                         ncol = 1)
    Tt_ref <- 1L
  } else {
    verdad_mat <- truth$theta      # p x T
    Tt_ref <- ncol(verdad_mat)
  }
  Tt_use <- min(Tt_fit, Tt_ref)

  filas <- list()
  for (l in 1:p) {
    for (tt in 1:Tt_use) {
      draws_lt <- td[l, tt, ]
      vt <- verdad_mat[l, tt]
      res <- .resumir_draws(draws_lt)
      sesgo <- res["media"] - vt
      filas[[length(filas)+1L]] <- data.frame(
        coef = labels[l], periodo = tt, verdad = vt,
        post_media = unname(res["media"]), post_sd = unname(res["sd"]),
        sesgo = unname(sesgo),
        sesgo_rel = unname(ifelse(abs(vt) > 1e-8, sesgo / vt, NA)),
        ic_lo = unname(res["q025"]), ic_hi = unname(res["q975"]),
        cubre95 = (vt >= res["q025"]) && (vt <= res["q975"]),
        amplitud = unname(res["q975"] - res["q025"]),
        ess = .ess_aprox(draws_lt),
        stringsAsFactors = FALSE)
    }
  }
  det <- do.call(rbind, filas)

  # resumen global
  resumen <- data.frame(
    rmse = sqrt(mean(det$sesgo^2)),
    mae = mean(abs(det$sesgo)),
    sesgo_medio = mean(det$sesgo),
    cobertura95 = mean(det$cubre95),
    amplitud_media = mean(det$amplitud),
    ess_min = min(det$ess, na.rm = TRUE),
    ess_mediana = median(det$ess, na.rm = TRUE),
    cor_verdad_post = cor(det$verdad, det$post_media)
  )
  list(detalle = det, resumen = resumen)
}


# =============================================================================
# 2. SELECCION LASSO  (sensibilidad / especificidad)
# =============================================================================

metricas_seleccion <- function(fit, truth) {
  td <- fit$theta_draws
  p <- dim(td)[1]
  is_zero_true <- truth$beta_is_zero   # longitud p (intercepto = FALSE)
  if (length(is_zero_true) != p) {
    # alinear por si el fit reordena: asumimos mismo orden [mu,E,M,D]
    is_zero_true <- is_zero_true[seq_len(p)]
  }

  # estimar "cero" por coordenada: IC95 (sobre draws agregados en t) contiene 0
  detect_zero <- logical(p)
  for (l in 1:p) {
    draws_l <- as.numeric(td[l, , ])   # agrega todos los periodos
    ic <- quantile(draws_l, c(0.025, 0.975))
    detect_zero[l] <- (ic[1] <= 0) && (ic[2] >= 0)
  }

  # excluir intercepto del conteo
  idx <- 2:p
  zt <- is_zero_true[idx]; dz <- detect_zero[idx]
  n_cero  <- sum(zt); n_activo <- sum(!zt)
  sensibilidad <- if (n_cero > 0)  sum(dz & zt) / n_cero  else NA   # ceros detectados
  especificidad<- if (n_activo > 0) sum(!dz & !zt) / n_activo else NA # activos preservados

  data.frame(n_cero = n_cero, n_activo = n_activo,
             sensibilidad = sensibilidad, especificidad = especificidad,
             tasa_correcta = mean(dz == zt))
}


# =============================================================================
# 3. RECUPERACION DE VARIANZA  (kappa2 efectivo, escala comun)
# =============================================================================
metricas_varianza <- function(fit, truth) {
  modelo <- fit$extra$modelo
  M_k <- truth$sizes$M_k; D <- truth$sizes$D; Tt <- truth$sizes$Tt

  errs <- c()
  if (modelo %in% c("M3", "M4")) {
    if (modelo == "M3") {
      if (is.null(fit$v_draws) || is.null(fit$s_draws)) return(NULL)
      v_post <- apply(fit$v_draws, c(1,2), mean)   # max_nk x m
      s_post <- apply(fit$s_draws, c(1,2), mean)   # m x Tt
      kap_post <- function(jj,kk,tt) s_post[kk,tt] / v_post[jj,kk]
    } else {
      v_post <- fit$v_mean; phi_post <- fit$phi_mean
      kap_post <- function(jj,kk,tt) 1 / (v_post[jj,kk] * phi_post[kk,tt])
    }
    ratios <- c()
    for (kk in 1:D) for (jj in 1:M_k[kk]) for (tt in 1:Tt) {
      k2t <- truth$kappa2_jkt[jj,kk,tt]
      if (is.na(k2t)) next
      errs <- c(errs, abs(k2t - kap_post(jj,kk,tt)) / k2t)
      ratios <- c(ratios, kap_post(jj,kk,tt) / k2t)
    }
    if (length(errs) == 0) return(NULL)
    return(data.frame(error_rel_medio = mean(errs),
                      error_rel_mediana = median(errs),
                      error_rel_max = max(errs),
                      ratio_post_true = mean(ratios)))
  } else {
    # estaticos: kappa2_jk_draws es matriz n_save x n_cells (orden k->j).
    # Comparar media posterior por celda vs verdad (promedio temporal por celda).
    if (is.null(fit$kappa2_jk_draws)) return(NULL)
    k2_draws <- fit$kappa2_jk_draws
    if (is.null(dim(k2_draws))) k2_draws <- matrix(k2_draws, ncol = 1)
    k2_post_cell <- colMeans(k2_draws, na.rm = TRUE)   # longitud n_cells
    # verdad por celda en orden k->j (igual que el aplanado del adaptador)
    k2_true_cell <- c()
    for (kk in 1:D) for (jj in 1:M_k[kk])
      k2_true_cell <- c(k2_true_cell, mean(truth$kappa2_jkt[jj,kk,], na.rm = TRUE))
    n_use <- min(length(k2_post_cell), length(k2_true_cell))
    errs <- abs(k2_true_cell[1:n_use] - k2_post_cell[1:n_use]) / k2_true_cell[1:n_use]
    # ratio sistematico post/true (separa sesgo por mala-especificacion del ruido):
    # con errores t y modelo normal, este ratio ~ nu/(nu-2) > 1.
    ratio_sistematico <- mean(k2_post_cell[1:n_use] / k2_true_cell[1:n_use])
    return(data.frame(error_rel_medio = mean(errs),
                      error_rel_mediana = median(errs),
                      error_rel_max = max(errs),
                      ratio_post_true = ratio_sistematico))
  }
}


# =============================================================================
# 4. RECUPERACION DE nu  (modelos t: M2, M4)
# =============================================================================
metricas_nu <- function(fit, truth) {
  if (is.null(fit$extra$nu) || is.na(truth$nu_error)) return(NULL)
  nu_ch <- fit$extra$nu
  ic <- quantile(nu_ch, c(0.025, 0.975))
  data.frame(nu_verdadero = truth$nu_error, nu_post_media = mean(nu_ch),
             nu_ic_lo = unname(ic[1]), nu_ic_hi = unname(ic[2]),
             nu_cubre = (truth$nu_error >= ic[1]) && (truth$nu_error <= ic[2]),
             nu_error_rel = abs(mean(nu_ch) - truth$nu_error) / truth$nu_error)
}


# =============================================================================
# 5. AGREGADOR DE METRICAS POR MODELO
# =============================================================================
#
# Devuelve una lista con todas las metricas de un fit.
calcular_metricas <- function(fit, truth, ref_estatico = "theta_bar") {
  coef <- metricas_coeficientes(fit, truth, ref_estatico)
  espec <- .especificacion_modelo(fit$extra$modelo, truth)
  list(
    modelo = fit$extra$modelo,
    especificacion = espec,
    coef_detalle = coef$detalle,
    coef_resumen = coef$resumen,
    seleccion = metricas_seleccion(fit, truth),
    varianza = metricas_varianza(fit, truth),
    nu = metricas_nu(fit, truth),
    tiempo_seg = fit$tiempo_seg
  )
}
