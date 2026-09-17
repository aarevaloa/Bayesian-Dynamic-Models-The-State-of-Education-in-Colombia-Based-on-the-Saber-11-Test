#_______________________________________________________________________________
#@  MODELO 4 — CORRIDA DEL MUESTREADOR SOBRE DATOS REALES
#_______________________________________________________________________________

# ---- (1) Carpeta del código (AJUSTAR) ----
RUTA_CODIGO <- "C:/Users/USUARIO/OneDrive - Universidad Nacional de Colombia/Tesis/Escritorio/Códigos/Modelos/Modelo 4"

# ---- (2) Interruptores ----
VERIFICAR_MODULOS  <- FALSE  # TRUE: corre verificar_etapa2..7 al cargar (~1.5 min extra)
CORRER_DIAGNOSTICO <- FALSE  # TRUE: corre la Etapa 8 (diagnóstico) al final, si existe
# ARCHIVO_DIAGNOSTICO <- "10_diagnostico_M4_v2.R"

cargar <- function(f) source(file.path(RUTA_CODIGO, f), encoding = "UTF-8")

# ---- (3) Cargar módulos de funciones (Etapas 2-7 + ensamblador Etapa 7) ----
cat("\n[1/5] Cargando módulos de funciones...\n")
cargar("2_FFBS_theta_v6_M4.R")
cargar("3_Escalas_latentes_xi_v4_M4.R")
cargar("4_precision_municipal_v_v3_M4.R")
cargar("5_Muestreo_retrospectivo_beta_gamma_phi_v7_M4.R")
cargar("6_LASSO_dinamico_tau_lambda_phi_v4_M4.R")
cargar("7_MH_nu_v3_M4.R")
cargar("8_Muestreador_GIBBS_v4_M4.R")
cat("      OK: 7 módulos cargados",
    if (VERIFICAR_MODULOS) "(verificaciones ejecutadas).\n" else "(verificaciones omitidas).\n")

# ---- (4) Preparación de datos (Etapa 1: lee y procesa el CSV real) ----
cat("\n[2/5] Ejecutando Etapa 1 (preparación de matrices)...\n")
cargar("1_Preparación_matrices_v4_M4.R")

# ---- (5) Puente Etapa 1 -> Etapa 7: ensamblaje apilado ----
cat("\n[3/5] Ensamblando datos/dims/hip para el muestreador...\n")
yF <- ensamblar_yF(Y, F_lst, N_jkt, n_k, m, Tt, p)

datos <- list(y_list       = yF$y,
              F_list       = yF$F,
              orden_celdas = yF$orden_celdas,
              N_jk         = N_jk,
              N_kt         = N_kt)

dims <- list(p = p, Tt = Tt, m = m, n_k = n_k,
             idx_mu      = 1L,
             idx_bloques = list(E = idx_E, M = idx_M, D = idx_D),
             N_t         = vapply(yF$y, length, integer(1L)))

hip <- list(m_0       = m_0,
            C_0       = C_0,
            q_mu      = q_mu,
            a_v       = a_v,
            b_v       = b_v,
            beta_disc = beta_discount,
            n_k0      = n_k0,
            d_k0      = d_k0,
            a_phi     = unlist(a_phi)[c("E", "M", "D")],
            b_phi     = unlist(b_phi)[c("E", "M", "D")],
            a_lam     = unlist(a_lambda)[c("E", "M", "D")],
            b_lam     = unlist(b_lambda)[c("E", "M", "D")],
            lambda_nu = lambda_nu,
            sigma_MH  = sigma_MH_init)

# Chequeos de coherencia del ensamblaje (baratos; no tocan la matemática)
stopifnot(
  length(dims$N_t) == Tt, sum(dims$N_t) == N_total,
  all(vapply(datos$orden_celdas, nrow, integer(1L)) > 0L),
  identical(names(hip$a_phi), c("E", "M", "D")),
  identical(names(hip$a_lam), c("E", "M", "D")),
  length(hip$beta_disc) == m, length(hip$n_k0) == m, length(hip$d_k0) == m,
  length(hip$m_0) == p, all(dim(hip$C_0) == c(p, p))
)
cat(sprintf("      OK: N=%d obs, Tt=%d periodos, m=%d deptos, %d municipios, p=%d (E=%d, M=%d, D=%d)\n",
            N_total, Tt, m, sum(n_k), p, p_E, p_M, p_D))

# ---- (6) Configuración MCMC ----
N_ITER    <- 6000L   # muestras GUARDADAS post burn-in
BURN_IN   <- 5000L   # sweeps de calentamiento (descartados)
THIN      <- 5L       # se guarda 1 de cada THIN sweeps post burn-in
LOTE_ADAP <- 100L     # lote de adaptación de sigma_MH durante el burn-in
SEMILLA   <- 2026L    # gibbs_modelo4 hace set.seed(SEMILLA) internamente
PRINT_CADA <- 1L   # frecuencia del reporte de progreso (NULL = automático, ~100 reportes)

TOTAL <- BURN_IN + N_ITER * THIN     # sweeps internos totales (= 100000)
stopifnot(N_ITER > 0L, THIN >= 1L, BURN_IN >= LOTE_ADAP)

cat(sprintf("\n      MCMC: %d guardadas (burn-in=%d, thin=%d) -> %d sweeps totales | semilla=%d\n",
            N_ITER, BURN_IN, THIN, TOTAL, SEMILLA))
cat(sprintf("      theta_save ocupará ~%.2f GB en RAM (%d x %d x %d)\n",
            p * (Tt + 1) * N_ITER * 8 / 1e9, p, Tt + 1L, N_ITER))

DIR_SALIDA <- file.path(RUTA_CODIGO, "mcmc_salida_m4")
if (!dir.exists(DIR_SALIDA)) dir.create(DIR_SALIDA, recursive = TRUE)

# ============================================================================
# PRODUCCIÓN
# ============================================================================
cat("\n################## CORRIDA DE PRODUCCIÓN ##################\n")
cat("[4/5] Ejecutando el muestreador de Gibbs...\n\n")
t_inicio <- Sys.time()

salida <- gibbs_modelo4(datos, dims, hip,
                        n_iter    = N_ITER,
                        burn_in   = BURN_IN,
                        thin      = THIN,
                        lote_adap = LOTE_ADAP,
                        semilla   = SEMILLA,
                        verbose   = TRUE,
                        print_cada = PRINT_CADA)

t_fin <- Sys.time()
dur <- as.numeric(difftime(t_fin, t_inicio, units = "mins"))
cat(sprintf("\n      Muestreador finalizado en %.1f minutos.\n", dur))
cat(sprintf("      Tasa de aceptación MH de nu: %.3f\n", salida$tasa_MH))
cat(sprintf("      sigma_MH tras adaptación:    %.4f\n", salida$sigma_MH_final))

# ---- (7) Guardado de resultados y resumen ----
cat("\n[5/5] Guardando resultados...\n")
saveRDS(salida, file.path(DIR_SALIDA, "salida_gibbs_M4.rds"))
cat(sprintf("      salida_gibbs_M4.rds guardado en %s/\n", DIR_SALIDA))

nu_post <- mean(salida$nu)
nu_ic   <- quantile(salida$nu, c(0.025, 0.975))

# kappa2_{j,k,t} = 1/(v_{j,k} * phi_{k,t}) evaluado en las medias posteriores
# (resumen plug-in de la escala base de la t; el detalle fino va en la Etapa 8)
kap2 <- array(NA_real_, dim = c(max(n_k), m, Tt))
for (kk in 1:m) for (jj in 1:n_k[kk])
  kap2[jj, kk, ] <- 1 / (salida$v_mean[jj, kk] * salida$phi_mean[kk, ])
kap2_v <- kap2[is.finite(kap2)]

resumen_txt <- c(
  "==== RESUMEN DE LA CORRIDA - MODELO 4 (t-Student robusto) ====",
  sprintf("Fecha:            %s", format(Sys.time())),
  sprintf("Duración:         %.1f min", dur),
  sprintf("MCMC:             guardadas=%d, burn_in=%d, thin=%d (total=%d sweeps), semilla=%d",
          N_ITER, BURN_IN, THIN, TOTAL, SEMILLA),
  sprintf("Datos:            N=%d, m=%d, Tt=%d, p=%d (E=%d, M=%d, D=%d)",
          N_total, m, Tt, p, p_E, p_M, p_D),
  "",
  sprintf("nu posterior:     media=%.3f  IC95=[%.3f, %.3f]",
          nu_post, nu_ic[1], nu_ic[2]),
  sprintf("Aceptación MH nu: %.3f", salida$tasa_MH),
  sprintf("sigma_MH final:   %.4f", salida$sigma_MH_final),
  "",
  "phi2 (escalas globales LASSO) - mediana posterior:",
  sprintf("  E=%.4f  M=%.4f  D=%.4f",
          median(salida$phi2[, "E"]), median(salida$phi2[, "M"]),
          median(salida$phi2[, "D"])),
  "lambda2 (regularización) - mediana posterior:",
  sprintf("  E=%.4f  M=%.4f  D=%.4f",
          median(salida$lambda2[, "E"]), median(salida$lambda2[, "M"]),
          median(salida$lambda2[, "D"])),
  "",
  "kappa2_jk = 1/(v*phi) en medias posteriores (escala base de la t):",
  sprintf("  mediana=%.1f  rango=[%.1f, %.1f]",
          median(kap2_v), min(kap2_v), max(kap2_v))
)
writeLines(resumen_txt, file.path(DIR_SALIDA, "resumen_M4.txt"))
cat("      resumen_M4.txt guardado.\n")

# ---- (8) Diagnóstico opcional (Etapa 8) ----
if (CORRER_DIAGNOSTICO) {
  ruta_etapa8 <- file.path(RUTA_CODIGO, ARCHIVO_DIAGNOSTICO)
  if (file.exists(ruta_etapa8)) {
    cat("\n  Ejecutando módulo de diagnóstico (Etapa 8)...\n")
    invisible(capture.output(suppressMessages(
      source(ruta_etapa8, encoding = "UTF-8"))))
    conv  <- diagnostico_convergencia(salida, verbose = TRUE)
    dnu   <- diagnostico_nu(salida, verbose = TRUE)
    lasso <- diagnostico_lasso(salida, dims$idx_bloques, verbose = TRUE)
    ppc   <- posterior_predictivo_t(salida, datos, dims, n_rep = 200L,
                                    verbose = TRUE)
    res   <- resumen_diagnostico(conv, dnu, lasso, ppc, verbose = TRUE)
    saveRDS(list(conv = conv, dnu = dnu, lasso = lasso, ppc = ppc,
                 resumen = res),
            file.path(DIR_SALIDA, "diagnostico_M4.rds"))
    cat("\n      diagnostico_M4.rds guardado.\n")
  } else {
    cat(sprintf("\n      (%s no encontrado; se omite el diagnóstico.)\n",
                ARCHIVO_DIAGNOSTICO))
  }
}
