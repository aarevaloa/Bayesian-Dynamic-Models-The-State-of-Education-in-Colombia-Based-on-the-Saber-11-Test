#_______________________________________________________________________________
#@  MODELO 4 - EXPORTADOR DE RESULTADOS PARA DIAGNOSTICO EN PYTHON
#_______________________________________________________________________________
# ---- (1) Configuración (AJUSTAR) ----
RUTA_CODIGO    <- "C:/Users/USUARIO/OneDrive - Universidad Nacional de Colombia/Tesis/Escritorio/Códigos/Modelos/Modelo 4"
ARCHIVO_RDS    <- file.path(RUTA_CODIGO, "mcmc_salida_m4", "salida_gibbs_M4.rds")
DIR_EXPORT     <- file.path(RUTA_CODIGO, "export_python")
EXPORTAR_DATOS <- TRUE

VERIFICAR_MODULOS <- FALSE
cargar <- function(f) source(file.path(RUTA_CODIGO, f), encoding = "UTF-8")

# ---- (2) Salida del muestreador ----
if (!exists("salida")) {
  cat(sprintf("Leyendo %s ...\n", ARCHIVO_RDS))
  salida <- readRDS(ARCHIVO_RDS)
}

# ---- (3) Datos/dims: reutiliza la sesión o reconstruye el puente ----
if (!exists("datos") || !exists("dims")) {
  cat("Reconstruyendo datos/dims (módulos + Etapa 1 + puente)...\n")
  cargar("2_FFBS_theta_v6_M4.R")
  cargar("3_Escalas_latentes_xi_v4_M4.R")
  cargar("4_precision_municipal_v_v3_M4.R")
  cargar("5_Muestreo_retrospectivo_beta_gamma_phi_v7_M4.R")
  cargar("6_LASSO_dinamico_tau_lambda_phi_v4_M4.R")
  cargar("7_MH_nu_v3_M4.R")
  cargar("8_Muestreador_GIBBS_v4_M4.R")
  cargar("1_Preparación_matrices_v4_M4.R")
  yF <- ensamblar_yF(Y, F_lst, N_jkt, n_k, m, Tt, p)
  datos <- list(y_list = yF$y, F_list = yF$F, orden_celdas = yF$orden_celdas,
                N_jk = N_jk, N_kt = N_kt)
  dims <- list(p = p, Tt = Tt, m = m, n_k = n_k, idx_mu = 1L,
               idx_bloques = list(E = idx_E, M = idx_M, D = idx_D),
               N_t = vapply(yF$y, length, integer(1L)))
}

S  <- dim(salida$theta)[3]
stopifnot(dim(salida$theta)[1] == dims$p,
          dim(salida$theta)[2] == dims$Tt + 1L,
          length(salida$nu) == S)

dir.create(DIR_EXPORT, showWarnings = FALSE, recursive = TRUE)
wD <- function(x, f) { con <- file(file.path(DIR_EXPORT, f), "wb")
                       writeBin(as.numeric(x), con, size = 8); close(con) }
wI <- function(x, f) { con <- file(file.path(DIR_EXPORT, f), "wb")
                       writeBin(as.integer(x), con, size = 4); close(con) }

# ---- (4) Cadenas y medias posteriores (siempre) ----
cat("Exportando cadenas...\n")
wD(salida$theta,    "theta.bin")      # p x (Tt+1) x S, orden columna (R)
wD(salida$nu,       "nu.bin")         # S
wD(salida$phi2,     "phi2.bin")       # S x 3 (E,M,D), orden columna
wD(salida$lambda2,  "lambda2.bin")    # S x 3
wD(salida$v_mean,   "v_mean.bin")     # max(n_k) x m
wD(salida$phi_mean, "phi_mean.bin")   # m x Tt

# ---- (5) Datos por observación (para ajuste y PPC) ----
tiene_datos <- FALSE; N <- 0L
if (EXPORTAR_DATOS) {
  cat("Exportando datos por observación...\n")
  yv <- unlist(datos$y_list, use.names = FALSE)
  Fm <- do.call(rbind, datos$F_list)
  t_obs <- k_obs <- j_obs <- integer(0)
  for (tt in 1:dims$Tt) {
    kj <- datos$orden_celdas[[tt]]
    t_obs <- c(t_obs, rep.int(tt,       sum(kj[, 3L])))
    k_obs <- c(k_obs, rep.int(kj[, 1L], kj[, 3L]))
    j_obs <- c(j_obs, rep.int(kj[, 2L], kj[, 3L]))
  }
  N <- length(yv)
  stopifnot(nrow(Fm) == N, length(t_obs) == N, sum(dims$N_t) == N)
  wD(yv, "y.bin")                     # N
  wD(Fm, "F.bin")                     # N x p, orden columna (R)
  wI(t_obs, "t_obs.bin"); wI(k_obs, "k_obs.bin"); wI(j_obs, "j_obs.bin")
  tiene_datos <- TRUE
}

# ---- (6) Trazas escalares compactas (compartibles: nu, phi2, lambda2, mu_t) ----
mu_tz <- t(salida$theta[dims$idx_mu, , , drop = TRUE])          # S x (Tt+1)
colnames(mu_tz) <- paste0("mu_", 0:dims$Tt)
tz <- data.frame(iter = 1:S, nu = salida$nu,
                 phi2_E = salida$phi2[, "E"], phi2_M = salida$phi2[, "M"],
                 phi2_D = salida$phi2[, "D"],
                 lambda2_E = salida$lambda2[, "E"], lambda2_M = salida$lambda2[, "M"],
                 lambda2_D = salida$lambda2[, "D"], mu_tz, check.names = FALSE)
con <- gzfile(file.path(DIR_EXPORT, "trazas_escalares_M4.csv.gz"), "w")
write.csv(tz, con, row.names = FALSE); close(con)

# ---- (7) meta.json (manual, sin jsonlite) ----
jV <- function(x) paste0("[", paste(x, collapse = ","), "]")
jS <- function(x) paste0("[", paste0('"', gsub('"', "'", x), '"', collapse = ","), "]")
labels <- get0("theta_labels", ifnotfound = paste0("theta_", 1:dims$p))
deptos <- get0("depto_ids",    ifnotfound = 1:dims$m)
meta <- paste0("{",
  '"p":', dims$p, ',"Tt":', dims$Tt, ',"m":', dims$m, ',"S":', S,
  ',"n_k":', jV(dims$n_k), ',"N_t":', jV(dims$N_t),
  ',"idx_mu":', dims$idx_mu,
  ',"idx_E":', jV(dims$idx_bloques$E),
  ',"idx_M":', jV(dims$idx_bloques$M),
  ',"idx_D":', jV(dims$idx_bloques$D),
  ',"theta_labels":', jS(labels),
  ',"depto_ids":', jV(deptos),
  ',"burn_in":', salida$burn_in, ',"thin":', salida$thin,
  ',"tasa_MH":', sprintf("%.6f", salida$tasa_MH),
  ',"sigma_MH_final":', sprintf("%.6f", salida$sigma_MH_final),
  ',"tiene_datos":', tolower(tiene_datos), ',"N":', N, "}")
writeLines(meta, file.path(DIR_EXPORT, "meta.json"))

tam <- sum(file.size(list.files(DIR_EXPORT, full.names = TRUE))) / 1e6
cat(sprintf("\nLISTO. %d archivos en %s (%.1f MB).\n",
            length(list.files(DIR_EXPORT)), DIR_EXPORT, tam))
cat("Siguiente paso:  python 12_diagnosticos_M4.py\n")
cat("(trazas_escalares_M4.csv.gz es pequeño y compartible para revisión externa.)\n")
