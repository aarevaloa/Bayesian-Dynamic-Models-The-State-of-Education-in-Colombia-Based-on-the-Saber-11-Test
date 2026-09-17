# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# @  MODELO 4 - RECONSTRUCCION DE LA POSTERIOR DE LAS VARIANZAS
# _____________________________________________________________________________

import os, sys, time
import numpy as np
from scipy.special import polygamma

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import estilo_M4 as E

# ============================== PARAMETROS ===================================
# Deben coincidir con los de la corrida (Etapa 1, bloques 12d y 12g).
BETA_DISC = 0.95     # verificado: beta_discount = 0.95 en los 33 departamentos
N_K0      = 4.0      # n_k0 <- rep(4.0, m).  Su efecto es despreciable (ver salida)
A_V       = 1.0      # a_v <- 1.0
S_REC     = 6000     # draws a simular
S_SUB     = 400      # submuestra para los objetos municipales
SEMILLA   = 2026

t0 = time.time()
def paso(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

rng = np.random.default_rng(SEMILLA)
D = E.cargar_cache(con_varianzas=False)
N_kt   = D["n_D"].astype(float)          # (m, Tt) conteos por celda depto-periodo
phim   = D["phi_mean"]                   # (m, Tt) media posterior de phi recentrado
v_mean = D["v_jk"]                       # (J,)   media posterior de v recentrado
n_k    = D["n_k"]; muni_depto = D["muni_depto"]; dep = D["depto_ids"]
m, Tt  = phim.shape
J      = v_mean.size

# ===================== 1. GRADOS DE LIBERTAD DEL FILTRO ======================
paso("reproduciendo la recursion n_kt del filtro forward")
def gl_filtro(n_k0):
    n = np.empty((m, Tt)); prev = np.full(m, float(n_k0))
    for t in range(Tt):
        prev = BETA_DISC * prev + N_kt[:, t]
        n[:, t] = prev
    return n

n_ft = gl_filtro(N_K0)
paso(f"   n_kt en t={Tt}: min {n_ft[:,-1].min():.0f}  mediana {np.median(n_ft[:,-1]):.0f}"
     f"  max {n_ft[:,-1].max():.0f}")

# ===================== 2. RECUPERAR d_kt DESDE phi_mean ======================
paso("despejando d_kt de la recursion de medias")
dif = phim[:, :-1] - BETA_DISC * phim[:, 1:]
if (dif <= 0).any():
    malas = np.argwhere(dif <= 0)
    raise RuntimeError(
        f"E[phi_t] - beta E[phi_t+1] <= 0 en {len(malas)} celdas {malas[:5].tolist()}. "
        "La reconstruccion es incoherente: revisar BETA_DISC o el origen de phi_mean.")
paso(f"   coherencia: min(E[phi_t] - beta E[phi_t+1]) = {dif.min():.3e} > 0  en "
     f"{dif.size}/{dif.size} celdas")

d_ft = np.empty((m, Tt))
d_ft[:, -1] = n_ft[:, -1] / phim[:, -1]
d_ft[:, :-1] = (1 - BETA_DISC) * n_ft[:, :-1] / dif

# ===================== 3. MUESTREO RETROSPECTIVO DE phi ======================
paso(f"simulando {S_REC} draws retrospectivos de phi")
PHI = np.empty((m, Tt, S_REC))
PHI[:, -1, :] = rng.gamma(n_ft[:, -1, None] / 2, 2.0 / d_ft[:, -1, None],
                          size=(m, S_REC))
for t in range(Tt - 2, -1, -1):
    sh = (1 - BETA_DISC) * n_ft[:, t, None] / 2
    PHI[:, t, :] = BETA_DISC * PHI[:, t + 1, :] + rng.gamma(
        sh, 2.0 / d_ft[:, t, None], size=(m, S_REC))

err = np.abs(PHI.mean(axis=2) / phim - 1).max()
paso(f"   VERIFICACION media simulada vs phi_mean.bin: error relativo max {err:.2e}")
if err > 0.05:
    raise RuntimeError("La media simulada no reproduce phi_mean.bin. Revisar supuestos.")

SD_kt = 1.0 / np.sqrt(PHI)                                   # (m, Tt, S) puntos

# ===================== 4. APROXIMACION PARA v_{j,k} ==========================
paso("aproximando la posterior de v_jk (lognormal + recentrado)")
N_jk = np.zeros(J)
nM = D["n_M"]                                                 # (J, Tt)
N_jk = nM.sum(axis=1)
shape_v = A_V + N_jk / 2.0
var_log = polygamma(1, shape_v)                               # trigamma
mu_log = np.log(v_mean) - var_log / 2.0                       # casar E[v] = v_mean

V = np.exp(mu_log[:, None] + np.sqrt(var_log)[:, None] *
           rng.standard_normal((J, S_SUB)))
# reimponer sum_j log v_jk = 0 por departamento, como recentrar_identificabilidad
logV = np.log(V)
for k in range(m):
    sel = muni_depto == k
    logV[sel] -= logV[sel].mean(axis=0, keepdims=True)
V = np.exp(logV)
paso(f"   CV(v_jk) implicito: mediana {np.median(np.sqrt(var_log)):.3f}  "
     f"rango [{np.sqrt(var_log).min():.3f}, {np.sqrt(var_log).max():.3f}]")

# dispersion municipal: kappa2_jkt = 1/(v_jk phi_kt); se guardan resumenes
sub = np.linspace(0, S_REC - 1, S_SUB).astype(int)
SD_jkt = 1.0 / np.sqrt(V[:, None, :] * PHI[muni_depto][:, :, sub])   # (J, Tt, S_SUB)

def resu(A, ax=-1):
    return (np.median(A, axis=ax), np.quantile(A, .025, axis=ax),
            np.quantile(A, .975, axis=ax))

sd_kt_med, sd_kt_lo, sd_kt_hi = resu(SD_kt)
sd_jkt_med, sd_jkt_lo, sd_jkt_hi = resu(SD_jkt)

# ===================== 5. CAMBIO EN LA DESIGUALDAD INTERNA ===================
pares = np.where(D["es_par"])[0]
t_ini, t_fin = pares[0], pares[-1]
dsd = SD_kt[:, t_fin, :] - SD_kt[:, t_ini, :]
dsd_med = np.median(dsd, axis=1)
dsd_lo = np.quantile(dsd, .025, axis=1)
dsd_hi = np.quantile(dsd, .975, axis=1)
P_dsd = (dsd > 0).mean(axis=1)

# ===================== 6. GUARDADO ==========================================
paso("guardando varianzas_M4.npz")
np.savez(os.path.join(E.DIR_CACHE, "varianzas_M4.npz"),
         PHI=PHI, SD_kt=SD_kt, V_aprox=V,
         sd_kt_med=sd_kt_med, sd_kt_lo=sd_kt_lo, sd_kt_hi=sd_kt_hi,
         sd_jkt_med=sd_jkt_med, sd_jkt_lo=sd_jkt_lo, sd_jkt_hi=sd_jkt_hi,
         dsd_med=dsd_med, dsd_lo=dsd_lo, dsd_hi=dsd_hi, P_dsd=P_dsd,
         n_ft=n_ft, d_ft=d_ft, N_jk=N_jk, idx_sub_phi=sub,
         t_ini=np.array([t_ini]), t_fin=np.array([t_fin]),
         beta_disc=np.array([BETA_DISC]), n_k0=np.array([N_K0]))
mb = os.path.getsize(os.path.join(E.DIR_CACHE, "varianzas_M4.npz")) / 1e6
paso(f"   varianzas_M4.npz: {mb:.1f} MB")

# ===================== 7. INFORME ===========================================
print("\n" + "=" * 78)
print(f"CAMBIO EN LA DESIGUALDAD INTERNA RESIDUAL  (t = {t_fin+1} menos t = {t_ini+1}, "
      "ambos del mismo semestre)")
print("=" * 78)
print(f"{'depto':>6} {'n total':>9} {'sd ini':>8} {'sd fin':>8} {'Delta':>8} "
      f"{'IC95':>18} {'P(>0)':>7}")
o = np.argsort(-dsd_med)
for i in o:
    print(f"{dep[i]:>6} {int(N_kt[i].sum()):>9,} {sd_kt_med[i,t_ini]:>8.2f} "
          f"{sd_kt_med[i,t_fin]:>8.2f} {dsd_med[i]:>+8.2f} "
          f"[{dsd_lo[i]:>+6.2f}, {dsd_hi[i]:>+6.2f}] {P_dsd[i]:>7.3f}")
print(f"\n  aumento creible  P(Delta>0) > 0.95 : {int((P_dsd>0.95).sum())} de {m}")
print(f"  aumento creible  P(Delta>0) > 0.90 : {int((P_dsd>0.90).sum())} de {m}")
print(f"  descenso creible P(Delta<0) > 0.95 : {int((P_dsd<0.05).sum())} de {m}")
print(f"  amplitud mediana del IC95 de sd_kt : {np.median(sd_kt_hi-sd_kt_lo):.2f} puntos")
print(f"  amplitud mediana del IC95 de sd_jkt: {np.median(sd_jkt_hi-sd_jkt_lo):.2f} puntos")

# ===================== 8. SENSIBILIDAD A n_k0 ===============================
print("\n" + "=" * 78)
print("SENSIBILIDAD AL HIPERPARAMETRO n_k0")
print("=" * 78)

def reconstruir(n_k0, semilla):
    r = np.random.default_rng(semilla)
    nf = gl_filtro(n_k0)
    df = np.empty((m, Tt))
    df[:, -1] = nf[:, -1] / phim[:, -1]
    df[:, :-1] = (1 - BETA_DISC) * nf[:, :-1] / dif
    P = np.empty((m, Tt, S_REC))
    P[:, -1, :] = r.gamma(nf[:, -1, None] / 2, 2.0 / df[:, -1, None], size=(m, S_REC))
    for t in range(Tt - 2, -1, -1):
        P[:, t, :] = BETA_DISC * P[:, t + 1, :] + r.gamma(
            (1 - BETA_DISC) * nf[:, t, None] / 2, 2.0 / df[:, t, None], size=(m, S_REC))
    sdd = 1 / np.sqrt(P[:, t_fin, :]) - 1 / np.sqrt(P[:, t_ini, :])
    return np.median(sdd, axis=1), (sdd > 0).mean(axis=1)

print(f"  {'n_k0':>6} {'n_kt(t=1) min':>14} {'Delta sd medio':>15} {'P>0.95':>8} {'P>0.90':>8}")
for nk0 in (2.0, 4.0, 6.0, 10.0):
    dm, pp = reconstruir(nk0, SEMILLA + int(nk0))
    print(f"  {nk0:>6.0f} {gl_filtro(nk0)[:,0].min():>14.1f} {np.mean(dm):>15.3f} "
          f"{int((pp>0.95).sum()):>8d} {int((pp>0.90).sum()):>8d}")
print("  (el valor usado en la corrida, segun la Etapa 1, es n_k0 = 4)")
paso("LISTO")
