#_______________________________________________________________________________
#@  CODIGO 2 — MOTOR DE SIMULACION
#@  Corre escenarios x replicas x modelos y agrega las metricas
#_______________________________________________________________________________


.correr_modelo_seguro <- function(modelo, panel, truth, dir_base, control, ref_estatico) {
  runner <- RUNNERS[[modelo]]
  fit <- tryCatch(
    runner(panel, truth, dir_base, control),
    error = function(e) {
      message(sprintf("    [%s] ERROR en runner: %s", modelo, conditionMessage(e)))
      NULL
    })
  if (is.null(fit)) return(NULL)
  met <- tryCatch(
    calcular_metricas(fit, truth, ref_estatico),
    error = function(e) {
      message(sprintf("    [%s] ERROR en metricas: %s", modelo, conditionMessage(e)))
      NULL
    })
  if (is.null(met)) return(NULL)
  # WAIC: solo si se activa con control$calcular_waic = TRUE y el modulo esta
  # cargado. Se calcula a partir del fit y el panel; si falla, met$waic queda NULL
  # sin abortar el escenario.
  if (isTRUE(control$calcular_waic) && exists("waic_modelo", mode = "function")) {
    w <- tryCatch(waic_modelo(fit, panel, truth), error = function(e) {
      message(sprintf("    [%s] WAIC fallo: %s", modelo, conditionMessage(e))); NULL })
    if (!is.null(w)) {
      met$waic       <- w$waic
      met$waic_lppd  <- w$lppd
      met$waic_pwaic <- w$p_waic
      met$waic_elpd  <- w$elpd_waic
    }
  }
  met
}


# =============================================================================
# CORRER UN ESCENARIO 
# =============================================================================
correr_escenario <- function(cfg, modelos = c("M1","M2"), R = 10L,
                             control = list(n_iter = 1500L, burn_in = 800L, thin = 2L),
                             id_escenario = "S00", semilla_base = 1000L,
                             ref_estatico = "theta_bar",
                             dir_trabajo = tempfile("sim_"),
                             verbose = TRUE) {
  dir.create(dir_trabajo, recursive = TRUE, showWarnings = FALSE)
  if (verbose) cat(sprintf("\n=== Escenario %s | modelos: %s | R=%d ===\n",
                           id_escenario, paste(modelos, collapse=","), R))

  met_por_replica <- vector("list", R)
  for (r in 1:R) {
    cfg_r <- cfg
    cfg_r$seed <- semilla_base + r
    control_r <- control; control_r$semilla <- semilla_base + r
    sim <- simulate_panel(cfg_r)

    dir_r <- file.path(dir_trabajo, sprintf("rep%03d", r))
    dir.create(dir_r, showWarnings = FALSE)

    met_r <- list()
    for (mod in modelos) {
      m <- .correr_modelo_seguro(mod, sim$panel, sim$truth, dir_r, control_r, ref_estatico)
      if (!is.null(m)) met_r[[mod]] <- m
    }
    met_por_replica[[r]] <- met_r
    if (verbose) {
      tiempos <- sapply(met_r, function(x) round(x$tiempo_seg, 1))
      cat(sprintf("  rep %d/%d completada  (%s)\n", r, R,
                  paste(sprintf("%s=%.1fs", names(tiempos), tiempos), collapse=" ")))
    }
    # limpiar .bin de la replica para no acumular disco
    unlink(dir_r, recursive = TRUE)
  }

  # agregar
  agg <- agregar_metricas(met_por_replica, modelos)
  list(id = id_escenario, config = cfg, modelos = modelos, R = R,
       control = control, por_replica = met_por_replica, agregado = agg)
}


# =============================================================================
# AGREGACION DE METRICAS SOBRE REPLICAS
# =============================================================================

agregar_metricas <- function(met_por_replica, modelos) {
  R <- length(met_por_replica)
  out <- list()
  for (mod in modelos) {
    # recolectar resumenes de coeficientes
    res_coef <- lapply(met_por_replica, function(rep) {
      if (!is.null(rep[[mod]])) rep[[mod]]$coef_resumen else NULL
    })
    res_coef <- res_coef[!sapply(res_coef, is.null)]
    if (length(res_coef) == 0) { out[[mod]] <- NULL; next }
    coef_mat <- do.call(rbind, res_coef)

    agg_coef <- data.frame(
      metrica = names(coef_mat),
      media = sapply(coef_mat, mean, na.rm = TRUE),
      sd = sapply(coef_mat, sd, na.rm = TRUE),
      stringsAsFactors = FALSE, row.names = NULL)

    # seleccion LASSO
    sel <- lapply(met_por_replica, function(rep) if (!is.null(rep[[mod]])) rep[[mod]]$seleccion else NULL)
    sel <- sel[!sapply(sel, is.null)]
    agg_sel <- if (length(sel) > 0) {
      sm <- do.call(rbind, sel)
      data.frame(sensibilidad = mean(sm$sensibilidad, na.rm=TRUE),
                 especificidad = mean(sm$especificidad, na.rm=TRUE),
                 tasa_correcta = mean(sm$tasa_correcta, na.rm=TRUE))
    } else NULL

    # varianza
    vr <- lapply(met_por_replica, function(rep) if (!is.null(rep[[mod]])) rep[[mod]]$varianza else NULL)
    vr <- vr[!sapply(vr, is.null)]
    agg_var <- if (length(vr) > 0) {
      vm <- do.call(rbind, vr)
      data.frame(
        ratio_post_true = mean(vm$ratio_post_true, na.rm=TRUE),     # PRINCIPAL: sesgo
        ratio_sd = sd(vm$ratio_post_true, na.rm=TRUE),
        error_rel_mediano = mean(vm$error_rel_mediana, na.rm=TRUE), # SECUNDARIO: precision
        error_rel_mediano_sd = sd(vm$error_rel_mediana, na.rm=TRUE))
    } else NULL

    # nu
    nu <- lapply(met_por_replica, function(rep) if (!is.null(rep[[mod]])) rep[[mod]]$nu else NULL)
    nu <- nu[!sapply(nu, is.null)]
    agg_nu <- if (length(nu) > 0) {
      nm <- do.call(rbind, nu)
      data.frame(nu_verdadero = nm$nu_verdadero[1],
                 nu_post_media = mean(nm$nu_post_media),
                 nu_cobertura = mean(nm$nu_cubre),
                 nu_error_rel = mean(nm$nu_error_rel))
    } else NULL

    # tiempo
    tiempos <- sapply(met_por_replica, function(rep) if (!is.null(rep[[mod]])) rep[[mod]]$tiempo_seg else NA)

    waics  <- sapply(met_por_replica, function(rep) if (!is.null(rep[[mod]]) && !is.null(rep[[mod]]$waic)) rep[[mod]]$waic else NA_real_)
    lppds  <- sapply(met_por_replica, function(rep) if (!is.null(rep[[mod]]) && !is.null(rep[[mod]]$waic_lppd)) rep[[mod]]$waic_lppd else NA_real_)
    pwaics <- sapply(met_por_replica, function(rep) if (!is.null(rep[[mod]]) && !is.null(rep[[mod]]$waic_pwaic)) rep[[mod]]$waic_pwaic else NA_real_)
    agg_waic <- if (any(is.finite(waics))) {
      data.frame(waic = mean(waics, na.rm = TRUE),
                 waic_sd = sd(waics, na.rm = TRUE),
                 lppd = mean(lppds, na.rm = TRUE),
                 p_waic = mean(pwaics, na.rm = TRUE),
                 n_replicas_waic = sum(is.finite(waics)))
    } else NULL

    # especificacion (igual en todas las replicas del mismo escenario; tomar la 1a)
    espec <- NULL
    for (rep in met_por_replica) if (!is.null(rep[[mod]])) { espec <- rep[[mod]]$especificacion; break }

    out[[mod]] <- list(coef = agg_coef, seleccion = agg_sel,
                       varianza = agg_var, nu = agg_nu,
                       waic = agg_waic,
                       especificacion = espec,
                       tiempo_medio_seg = mean(tiempos, na.rm = TRUE),
                       n_replicas_ok = length(res_coef))
  }
  out
}


# =============================================================================
# IMPRESION DE RESULTADOS
# =============================================================================

imprimir_agregado <- function(resultado) {
  cat(sprintf("\n========== RESULTADOS ESCENARIO %s (R=%d) ==========\n",
              resultado$id, resultado$R))
  for (mod in resultado$modelos) {
    a <- resultado$agregado[[mod]]
    if (is.null(a)) { cat(sprintf("\n[%s] sin resultados\n", mod)); next }

    # encabezado con etiqueta de especificacion
    esp <- a$especificacion
    etiqueta <- if (is.null(esp)) "?" else
      switch(esp$estado,
             "correcta" = "especificacion CORRECTA",
             paste0("MAL ESPECIFICADO (", sub("^mal:", "", esp$estado), ")"))
    cat(sprintf("\n--- %s | %s | %d/%d rep OK, %.1fs/rep ---\n",
                mod, etiqueta, a$n_replicas_ok, resultado$R, a$tiempo_medio_seg))

    cr <- a$coef
    get <- function(nm) cr$media[cr$metrica == nm]

    # ---- SECCION 1: RECUPERACION (¿el modelo acierta la verdad?) ----
    cat("  [RECUPERACION]\n")
    cat(sprintf("    Coef:     RMSE=%.4f  MAE=%.4f  sesgo=%.4f  cor(verdad,post)=%.4f\n",
                get("rmse"), get("mae"), get("sesgo_medio"), get("cor_verdad_post")))
    cat(sprintf("    Calibr.:  cobertura95=%.1f%%  amplitud=%.3f  ESS min=%.0f mediana=%.0f\n",
                100*get("cobertura95"), get("amplitud_media"),
                get("ess_min"), get("ess_mediana")))
    if (!is.null(a$varianza)) {
      # ratio cercano a 1 = recupera la escala; lejos de 1 = sesgo (ver robustez)
      cat(sprintf("    Varianza: ratio post/true=%.2f (sd %.2f)  error celda mediano=%.1f%%\n",
                  a$varianza$ratio_post_true, a$varianza$ratio_sd,
                  100*a$varianza$error_rel_mediano))
    }
    if (!is.null(a$nu) && (is.null(esp) || esp$ok_temporal))
      cat(sprintf("    nu:       verdadero=%.1f post=%.2f cobertura=%.1f%% err.rel=%.3f\n",
                  a$nu$nu_verdadero, a$nu$nu_post_media,
                  100*a$nu$nu_cobertura, a$nu$nu_error_rel))

    # ---- SECCION 2: ROBUSTEZ / CONTRASTE (¿cuanto paga el modelo equivocado?) ----
    hay_robustez <- !is.null(esp) && esp$estado != "correcta"
    if (hay_robustez || !is.null(a$seleccion)) {
      cat("  [ROBUSTEZ / CONTRASTE]\n")
      if (!is.null(a$seleccion))
        cat(sprintf("    LASSO:    sensibilidad=%.1f%%  especificidad=%.1f%%\n",
                    100*a$seleccion$sensibilidad, 100*a$seleccion$especificidad))
      if (hay_robustez) {
        # contextualizar el sesgo de varianza con el valor teorico esperado
        if (!is.null(a$varianza) && !is.na(esp$ratio_teorico)) {
          cat(sprintf("    Varianza: ratio observado=%.2f  vs  teorico esperado=%.2f%s\n",
                      a$varianza$ratio_post_true, esp$ratio_teorico,
                      if (!esp$ok_colas) " [normal sobre datos t: nu/(nu-2)]" else ""))
        }
        # contextualizar nu si el modelo es estatico sobre datos dinamicos
        if (!is.null(a$nu) && !esp$ok_temporal)
          cat(sprintf("    nu:       post=%.2f cobertura=%.1f%% -- ESPERADO inflado (estatico sobre datos dinamicos)\n",
                      a$nu$nu_post_media, 100*a$nu$nu_cobertura))
        if (!esp$ok_temporal)
          cat("    (coeficientes: el estatico estima el promedio temporal; sesgo vs theta_t es esperado)\n")
      }
    }
  }
  cat("\n")
}
