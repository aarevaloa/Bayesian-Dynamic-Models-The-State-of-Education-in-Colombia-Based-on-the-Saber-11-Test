# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# @  MODELO 4 -  PREPARACION: AGREGADOS TERRITORIALES Y NIVEL ESPERADO
# _____________________________________________________________________________

import json, os, sys, time, warnings
import numpy as np


warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import estilo_M4 as E

# ============================== PARAMETROS ===================================
S_SUB = 400          # draws equispaciados para los objetos grandes (convencion de la tesis)
VERIF = True         # imprime las comprobaciones de integridad

t0 = time.time()
def paso(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

# ============================== 1. METADATOS =================================
paso("leyendo meta.json")
with open(os.path.join(E.RUTA_EXPORT, "meta.json"), encoding="utf-8") as fh:
    meta = json.load(fh)

N, p, Tt, S, m = meta["N"], meta["p"], meta["Tt"], meta["S"], meta["m"]
labels = meta["theta_labels"]
depto_ids = np.asarray(meta["depto_ids"], dtype=int)
n_k = np.asarray(meta["n_k"], dtype=int)              # municipios por departamento
N_t = np.asarray(meta["N_t"], dtype=int)
J = int(n_k.sum())                                     # 1107 municipios

# indices de bloque en meta vienen en base 1
idx_bloque = {b: np.asarray(meta[f"idx_{b}"], dtype=int) - 1 for b in ("E", "M", "D")}
idx_mu = meta["idx_mu"] - 1

# mapa municipio global -> (departamento, j local).  offset acumulado de n_k
off = np.concatenate([[0], np.cumsum(n_k)])
muni_depto = np.repeat(np.arange(m), n_k)                       # (J,) indice 0..m-1
muni_j = np.concatenate([np.arange(nk) for nk in n_k])          # (J,) indice local 0..n_k-1

_f_ids = os.path.join(E.RUTA_EXPORT, "muni_ids.csv")
if os.path.exists(_f_ids):
    _tab = np.genfromtxt(_f_ids, delimiter=",", names=True, dtype=int)
    muni_ids = np.full(J, -1, dtype=int)
    muni_ids[off[_tab["k"] - 1] + (_tab["j"] - 1)] = _tab["cod_dane"]
    assert (muni_ids > 0).all(), "muni_ids.csv no cubre los 1107 municipios"
else:
    muni_ids = np.full(J, -1, dtype=int)
    print("  AVISO: no se encontro muni_ids.csv. Los agregados municipales se\n"
          "         calculan igual, pero no podran mapearse a un geojson hasta\n"
          "         exportar desde R la equivalencia (k, j) -> codigo DIVIPOLA.")

_f_per = os.path.join(E.RUTA_EXPORT, "periodo_ids.csv")
if os.path.exists(_f_per):
    _tp = np.genfromtxt(_f_per, delimiter=",", names=True, dtype=int)
    periodo_cod = _tp["periodo"][np.argsort(_tp["t"])]
    et_periodo = np.array([f"{c // 10}-{c % 10}" for c in periodo_cod])
else:
    periodo_cod = np.zeros(Tt, dtype=int)
    et_periodo = np.array([f"t = {t}" for t in range(1, Tt + 1)])
    print("  AVISO: sin periodo_ids.csv; los ejes se rotulan t = 1..16.")

_f_est = os.path.join(E.RUTA_EXPORT, "estandar_params.csv")
if os.path.exists(_f_est):
    _te = np.genfromtxt(_f_est, delimiter=",", names=True, dtype=None,
                        encoding="utf-8")
    est_var = np.array([str(v).strip('"') for v in _te["variable"]], dtype=object)
    est_media = np.asarray(_te["media"], dtype=float)
    est_sd = np.asarray(_te["desvest"], dtype=float)
    est_nna = np.asarray(_te["n_na"], dtype=int)
else:
    est_var = np.array([], dtype=object)
    est_media = est_sd = np.array([]); est_nna = np.array([], dtype=int)
    print("  AVISO: sin estandar_params.csv; los efectos de M y D quedaran en\n"
          "         desviaciones estandar, no en unidades interpretables.")

es_par = (np.arange(1, Tt + 1) % 2 == 0)

# ============================== 2. DATOS =====================================
paso("abriendo F.bin como memmap y leyendo indices")
F = np.memmap(os.path.join(E.RUTA_EXPORT, "F.bin"),
              dtype="float64", mode="r", shape=(N, p), order="F")
y = np.fromfile(os.path.join(E.RUTA_EXPORT, "y.bin"))
k_obs = np.fromfile(os.path.join(E.RUTA_EXPORT, "k_obs.bin"), dtype="int32")
j_obs = np.fromfile(os.path.join(E.RUTA_EXPORT, "j_obs.bin"), dtype="int32")
t_obs = np.fromfile(os.path.join(E.RUTA_EXPORT, "t_obs.bin"), dtype="int32")

assert y.size == N and k_obs.size == N and j_obs.size == N and t_obs.size == N
assert k_obs.min() == 1 and k_obs.max() == m
assert t_obs.min() == 1 and t_obs.max() == Tt

# celdas: departamento-periodo y municipio-periodo
gid = off[k_obs - 1] + (j_obs - 1)                 # id municipal global 0..J-1
cel_D = (k_obs - 1) * Tt + (t_obs - 1)
cel_M = gid * Tt + (t_obs - 1)
nc_D, nc_M = m * Tt, J * Tt

# ============================== 3. AGREGADOS =================================
def agrega(cel, ncel, forma):
    """n, media y sd observadas de y por celda; NaN donde no hay datos."""
    n = np.bincount(cel, minlength=ncel).astype(float)
    s1 = np.bincount(cel, weights=y, minlength=ncel)
    s2 = np.bincount(cel, weights=y * y, minlength=ncel)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = s1 / n
        var = s2 / n - mu ** 2
    mu[n == 0] = np.nan
    var[n < 2] = np.nan
    return (n.reshape(forma), mu.reshape(forma),
            np.sqrt(np.maximum(var, 0)).reshape(forma))

paso("agregando y por celda (departamental y municipal)")
n_D, ybar_D, sd_obs_D = agrega(cel_D, nc_D, (m, Tt))
n_M, ybar_M, sd_obs_M = agrega(cel_M, nc_M, (J, Tt))

paso(f"agregando las {p} covariables (una pasada por columna de F)")
Xbar_D = np.empty((p, m, Tt))
Xbar_M = np.empty((p, J, Tt))
nD_flat = n_D.ravel(); nM_flat = n_M.ravel()
for i in range(p):
    col = np.asarray(F[:, i], dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        Xbar_D[i] = (np.bincount(cel_D, weights=col, minlength=nc_D)
                     / nD_flat).reshape(m, Tt)
        Xbar_M[i] = (np.bincount(cel_M, weights=col, minlength=nc_M)
                     / nM_flat).reshape(J, Tt)
    if (i + 1) % 20 == 0:
        paso(f"   columna {i+1}/{p}")
Xbar_D[:, n_D == 0] = np.nan
Xbar_M[:, n_M == 0] = np.nan
del F

# ============================== 4. VARIANZAS =================================
paso("reconstruyendo dispersion departamental y municipal")
phi_mean = np.fromfile(os.path.join(E.RUTA_EXPORT, "phi_mean.bin")).reshape(
    (m, Tt), order="F")

sd_kt_plugin = np.sqrt(1.0 / phi_mean)                   # (m, Tt) puntos

# v_mean se exporta como matriz rectangular (max n_k) x m con ceros de relleno
v_raw = np.fromfile(os.path.join(E.RUTA_EXPORT, "v_mean.bin"))
n_max = int(v_raw.size // m)
V = v_raw.reshape((n_max, m), order="F")
v_jk = np.array([V[muni_j[g], muni_depto[g]] for g in range(J)])
phi_k = phi_mean.mean(axis=1)                            # perfil temporal promedio
kappa2_jk = 1.0 / (v_jk * phi_k[muni_depto])
sd_jk = np.sqrt(kappa2_jk)                               # (J,) puntos

# dispersion municipio-periodo: v_jk estatico x phi_kt dinamico
sd_jkt_plugin = np.sqrt(1.0 / (v_jk[:, None] * phi_mean[muni_depto, :]))

# escalares
nu = np.fromfile(os.path.join(E.RUTA_EXPORT, "nu.bin"))
lambda2 = np.fromfile(os.path.join(E.RUTA_EXPORT, "lambda2.bin")).reshape(
    (S, 3), order="F").T
phi2 = np.fromfile(os.path.join(E.RUTA_EXPORT, "phi2.bin")).reshape(
    (S, 3), order="F").T

for _nom, _A, _ref in (("phi2", phi2, (1.2909, 1.0956, 1.1907)),
                       ("lambda2", lambda2, (0.1420, 0.5331, 0.0728))):
    _obs = np.median(_A, axis=1)
    if not np.allclose(_obs, _ref, rtol=0.02):
        raise RuntimeError(
            f"{_nom} no coincide con resumen_M4.txt: {np.round(_obs,4).tolist()} "
            f"frente a {list(_ref)}. Revisar el orden de lectura del binario.")

# ============================== 5. NIVEL ESPERADO ============================
paso("leyendo theta.bin y calculando el nivel esperado")
theta = np.fromfile(os.path.join(E.RUTA_EXPORT, "theta.bin")).reshape(
    (p, Tt + 1, S), order="F")
th = theta[:, 1:, :]                                     # (p, Tt, S), t = 1..Tt

idx_sub = np.linspace(0, S - 1, S_SUB).astype(int)

Y_D = E.nivel_esperado(Xbar_D, th)                       # (m, Tt, S) completa
Y_M = E.nivel_esperado(Xbar_M, th[:, :, idx_sub])        # (J, Tt, S_SUB)

Y_bloq = {}
for b in ("E", "M", "D"):
    Y_bloq[b] = E.nivel_esperado(Xbar_D, th[:, :, idx_sub],
                                 bloques=b, idx_bloque=idx_bloque)
Y_D_mu = th[idx_mu]                                      # (Tt, S)

# enmascarar celdas sin datos
Y_D[n_D == 0] = np.nan
Y_M[n_M == 0] = np.nan
for b in Y_bloq:
    Y_bloq[b][n_D == 0] = np.nan

def resu(A):
    return (np.nanmedian(A, axis=-1),
            np.nanquantile(A, .025, axis=-1),
            np.nanquantile(A, .975, axis=-1))

med_D, lo_D, hi_D = resu(Y_D)
med_M, lo_M, hi_M = resu(Y_M)
res_D = ybar_D - med_D                                   # residuo departamental
res_M = ybar_M - med_M

# ============================== 6. GUARDADO ==================================
paso("guardando cache")
np.savez_compressed(
    os.path.join(E.DIR_CACHE, "prep_M4.npz"),
    Xbar_D=Xbar_D, Xbar_M=Xbar_M,
    ybar_D=ybar_D, sd_obs_D=sd_obs_D, n_D=n_D,
    ybar_M=ybar_M, sd_obs_M=sd_obs_M, n_M=n_M,
    sd_kt_plugin=sd_kt_plugin, sd_jk=sd_jk, sd_jkt_plugin=sd_jkt_plugin,
    kappa2_jk=kappa2_jk,
    phi_mean=phi_mean, v_jk=v_jk,
    nu=nu, lambda2=lambda2, phi2=phi2,
    labels=np.array(labels, dtype=object), depto_ids=depto_ids, n_k=n_k,
    muni_depto=muni_depto, muni_j=muni_j, muni_ids=muni_ids,
    N_t=N_t, es_par=es_par, periodo_cod=periodo_cod, et_periodo=et_periodo,
    est_var=est_var, est_media=est_media, est_sd=est_sd, est_nna=est_nna,
    idx_E=idx_bloque["E"], idx_M=idx_bloque["M"], idx_D=idx_bloque["D"],
    idx_mu=np.array([idx_mu]))

np.savez(
    os.path.join(E.DIR_CACHE, "nivel_M4.npz"),
    Y_D=Y_D, Y_M=Y_M, Y_D_mu=Y_D_mu,
    Y_D_E=Y_bloq["E"], Y_D_M=Y_bloq["M"], Y_D_D=Y_bloq["D"],
    med_D=med_D, lo_D=lo_D, hi_D=hi_D,
    med_M=med_M, lo_M=lo_M, hi_M=hi_M,
    res_D=res_D, res_M=res_M, idx_sub=idx_sub)

for f in ("prep_M4.npz", "nivel_M4.npz"):
    mb = os.path.getsize(os.path.join(E.DIR_CACHE, f)) / 1e6
    paso(f"   {f}: {mb:.1f} MB")

# ============================== 7. VERIFICACION ==============================
if VERIF:
    print("\n" + "=" * 74)
    print("VERIFICACION DE INTEGRIDAD")
    print("=" * 74)
    w = n_D / n_D.sum()
    print(f"  N total                        {int(n_D.sum()):>12,}   (meta: {N:,})")
    print(f"  media ponderada de ybar_D      {np.nansum(ybar_D*w):>12.3f}   (y.bin: {y.mean():.3f})")
    print(f"  media ponderada de med_D       {np.nansum(med_D*w):>12.3f}")
    ok = np.isfinite(ybar_D) & np.isfinite(med_D)
    r = np.corrcoef(ybar_D[ok], med_D[ok])[0, 1]
    print(f"  corr(ybar_D, yhat_D)           {r:>12.4f}   R2 entre celdas {r**2:.4f}")
    print(f"  celdas depto-periodo vacias    {int((n_D==0).sum()):>12d} de {m*Tt}")
    print(f"  celdas muni-periodo vacias     {int((n_M==0).sum()):>12d} de {J*Tt}"
          f"   ({100*(n_M==0).mean():.1f}%)")
    print(f"  kappa2_jk mediana              {np.median(kappa2_jk):>12.1f}   "
          f"rango [{kappa2_jk.min():.1f}, {kappa2_jk.max():.1f}]")
    print(f"  sd_jk (puntos)                 mediana {np.median(sd_jk):.1f}   "
          f"rango [{sd_jk.min():.1f}, {sd_jk.max():.1f}]")
    print(f"  amplitud IC95 de yhat_D        mediana {np.nanmedian(hi_D-lo_D):.2f} puntos")
    resw = np.nansum(res_D * w)
    sdw = np.sqrt(np.nansum((res_D - resw) ** 2 * w))
    print(f"  residuo departamental          media {resw:+.2f}   sd ponderada {sdw:.2f}")

    print("\n  ESTRUCTURA SEMESTRAL (los t impares y pares NO son comparables)")
    print(f"  {'periodo':>8} {'N_t':>8} {'ybar':>8} {'%calen_B':>9} {'%rural':>8} {'munis c/dato':>13}")
    icb = labels.index("calen_B"); iru = labels.index("cole_area_ubicacion_RURAL")
    for t in range(Tt):
        wt = n_D[:, t] / n_D[:, t].sum()
        print(f"  {et_periodo[t]:>8} {int(n_D[:,t].sum()):>8,} {np.nansum(ybar_D[:,t]*wt):>8.1f} "
              f"{np.nansum(Xbar_D[icb,:,t]*wt):>9.3f} {np.nansum(Xbar_D[iru,:,t]*wt):>8.3f} "
              f"{int((n_M[:,t]>0).sum()):>13,}")

    print("\n  DESCOMPOSICION DEL NIVEL ESPERADO (promedio y sd ENTRE departamentos)")
    for b, A in (("E (estudiante)", Y_bloq["E"]), ("M (municipal)", Y_bloq["M"]),
                 ("D (departamental)", Y_bloq["D"])):
        med = np.nanmedian(A, axis=-1)
        sde = np.nanmean([np.sqrt(np.nansum((med[:, t] - np.nansum(med[:, t]*n_D[:, t]/n_D[:, t].sum()))**2
                                            * n_D[:, t]/n_D[:, t].sum())) for t in range(Tt)])
        print(f"    {b:<20} media {np.nansum(med*w):>8.2f}   sd entre deptos {sde:>6.2f}")
    print(f"    {'intercepto mu_t':<20} media {np.median(Y_D_mu):>8.2f}")

    print("\n  RESIDUOS DEPARTAMENTALES EXTREMOS (ybar - yhat, ponderado)")
    rw = np.nansum(res_D * n_D / n_D.sum(axis=1, keepdims=True).clip(1), axis=1)
    nD_tot = n_D.sum(axis=1)
    o = np.argsort(rw)
    for i in list(o[:4]) + list(o[-4:]):
        print(f"    depto {depto_ids[i]:>3}   residuo {rw[i]:+7.2f}   "
              f"n = {int(nD_tot[i]):>7,}   sd interna {sd_kt_plugin[i].mean():5.1f}")
    rho = np.corrcoef(np.log(nD_tot), np.abs(rw))[0, 1]
    _lect = ("el residuo grande se concentra en departamentos pequenos"
             if rho < -0.3 else "sin relacion clara con el tamano muestral")
    print(f"    corr(log n, |residuo|) = {rho:+.3f}   -> {_lect}")

    print("\n  DISPERSION: MODELO vs. OBSERVADA (puntos, promedio de periodos)")
    sdo = np.nansum(sd_obs_D * n_D / n_D.sum(axis=1, keepdims=True).clip(1), axis=1)
    print(f"    sd observada  media {np.average(sdo, weights=nD_tot):.2f}")
    print(f"    sd del modelo media {np.average(sd_kt_plugin.mean(axis=1), weights=nD_tot):.2f}"
          f"   (residual: descuenta composicion y contexto)")
    print(f"    razon modelo/observada: mediana {np.median(sd_kt_plugin.mean(axis=1)/sdo):.3f}")

paso("LISTO")
