# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# @  MODELO 4 - (PANORAMA: BLOQUE 0 (ENCUADRE) Y BLOQUE I (TERRITORIO)
# _____________________________________________________________________________

import os, sys, time, warnings
import numpy as np
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import estilo_M4 as E

t0 = time.time()
def paso(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

# ============================== DATOS ========================================
paso("cargando cache")
D = E.cargar_cache()
V = np.load(os.path.join(E.DIR_CACHE, "varianzas_M4.npz"))
P = E.cargar_posterior()

dep, et = D["depto_ids"], D["et_periodo"]
nD, nM = D["n_D"], D["n_M"]
ybD, ybM = D["ybar_D"], D["ybar_M"]
sdoM = D["sd_obs_M"]
YD, YM = D["Y_D"], D["Y_M"]                       # posteriores del nivel esperado
medD, loD, hiD = D["med_D"], D["lo_D"], D["hi_D"]
medM = D["med_M"]
m, Tt = nD.shape
pares = np.where(D["es_par"])[0]                  # 8 cohortes principales
etp = et[pares]

def wmean(x, w, ax=None):
    ok = np.isfinite(x) & (w > 0)
    return np.sum(np.where(ok, x * w, 0), axis=ax) / np.sum(np.where(ok, w, 0), axis=ax)

# ============================== FIG 01: ENCUADRE =============================
paso("fig01 encuadre")
# (a) descomposicion jerarquica de la suma de cuadrados del puntaje observado
ybar_t = wmean(ybD, nD, ax=0)                              # media por periodo
ybar_g = float(wmean(ybD, nD))
SS_est = np.nansum(np.where(np.isfinite(sdoM), nM * sdoM ** 2, 0))
SS_mun = np.nansum(np.where(np.isfinite(ybM), nM * (ybM - ybD[D["muni_depto"]]) ** 2, 0))
SS_dep = np.nansum(np.where(np.isfinite(ybD), nD * (ybD - ybar_t[None, :]) ** 2, 0))
SS_per = float(np.sum(nD.sum(axis=0) * (ybar_t - ybar_g) ** 2))
SS_tot = SS_est + SS_mun + SS_dep + SS_per
comp = np.array([SS_per, SS_dep, SS_mun, SS_est]) / SS_tot
et_comp = ["entre periodos", "entre departamentos", "entre municipios",
           "entre estudiantes"]
col_comp = ["#b0b0b0", E.C["D"], E.C["M"], E.C["E"]]

fig = plt.figure(figsize=(13.6, 8.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.42, wspace=0.24)

ax = fig.add_subplot(gs[0, 0])
izq = 0.0
for f, e, c in zip(comp, et_comp, col_comp):
    ax.barh(0, f, left=izq, color=c, height=0.5, edgecolor="white", lw=1.2)
    if f > 0.03:
        ax.text(izq + f / 2, 0, f"{100*f:.1f}%", ha="center", va="center",
                color="white" if f > 0.1 else "0.2", fontsize=9, fontweight="bold")
    izq += f
ax.set_xlim(0, 1); ax.set_ylim(-.6, .8); ax.set_yticks([])
ax.set_xlabel("proporción de la variación total del puntaje")
ax.set_title("¿Dónde está la desigualdad educativa?")
ax.grid(False)
for e, c in zip(et_comp, col_comp):
    ax.plot([], [], "s", color=c, ms=8, label=e)
ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center",
          bbox_to_anchor=(0.5, 1.02))
ax.spines[["left", "right", "top"]].set_visible(False)

# (b) serie nacional observada, separando semestres
ax = fig.add_subplot(gs[0, 1])
imp = np.setdiff1d(np.arange(Tt), pares)
ax.plot(pares + 1, ybar_t[pares], "o-", color=E.C_SEM["par"], lw=2.4, ms=5,
        label=E.ET_SEM["par"], zorder=3)
ax.plot(imp + 1, ybar_t[imp], "o--", color=E.C_SEM["impar"], lw=1.6, ms=4,
        alpha=0.85, label=E.ET_SEM["impar"])
ax.set_xticks(range(1, Tt + 1)); ax.set_xticklabels(et, rotation=90, fontsize=6.5)
ax.set_ylabel("puntaje global medio")
ax.set_title("La serie nacional alterna entre dos poblaciones")
ax.legend(frameon=False, fontsize=8)

# (c) trayectoria del intercepto mu_t, que NO es la media nacional
ax = fig.add_subplot(gs[1, 0])
mu = P.draws_t("intercepto")
q = np.quantile(mu, [.025, .5, .975], axis=1)
ax.fill_between(np.arange(1, Tt + 1), q[0], q[2], color=E.C_SEM["par"], alpha=0.16)
ax.plot(np.arange(1, Tt + 1), q[1], color=E.C_SEM["par"], lw=2)
E.marcar_semestres(ax, Tt)
ax.set_xticks(range(1, Tt + 1)); ax.set_xticklabels(et, rotation=90, fontsize=6.5)
ax.set_ylabel("puntos")
ax.set_title("Intercepto $\\mu_t$: perfil de referencia, no la media nacional")

# (d) nivel esperado nacional del modelo frente al observado, cohorte principal
ax = fig.add_subplot(gs[1, 1])
w = nD / nD.sum(axis=0, keepdims=True)
Yn = np.einsum("kt,kts->ts", np.nan_to_num(w), np.nan_to_num(YD))   # (Tt, S)
qn = np.quantile(Yn, [.025, .5, .975], axis=1)
ax.fill_between(pares + 1, qn[0][pares], qn[2][pares], color=E.C["E"], alpha=0.2,
                label="nivel esperado por el modelo (IC 95%)")
ax.plot(pares + 1, qn[1][pares], color=E.C["E"], lw=2)
ax.plot(pares + 1, ybar_t[pares], "o", color="0.15", ms=5, label="observado")
ax.set_xticks(pares + 1); ax.set_xticklabels(etp, rotation=45, fontsize=7.5)
ax.set_ylabel("puntaje global")
ax.set_title("Ajuste nacional en la cohorte principal")
ax.legend(frameon=False, fontsize=8)

fig.suptitle("Modelo 4 — encuadre: qué explica el modelo y qué no", y=0.985)
E.guardar(fig, "fig01_encuadre.png", cerrar=True)

# ============================== MAPAS ========================================
paso("cargando geografia")
geo_d = E.cargar_geojson(E.RUTA_GEOJSON, "DPTO_CCDGO", orden=dep, clave=int)
E.fijar_geo(geo_d)

# ---- fig02: nivel esperado por departamento, 8 cohortes ----
paso("fig02 nivel departamental")
nrm = E.norma_robusta(medD[:, pares])
fig, axs = plt.subplots(2, 4, figsize=(13.2, 7.6))
for i, t in enumerate(pares):
    ax = axs[i // 4, i % 4]
    E.eje_mapa(ax, medD[:, t], E.CMAP_NIVEL, nrm, etiq=False)
    ax.set_title(et[t], fontsize=9.5)
sm = plt.cm.ScalarMappable(norm=nrm, cmap=E.CMAP_NIVEL)
fig.colorbar(sm, ax=axs, fraction=0.022, pad=0.01,
             label="puntaje esperado (mediana posterior)")
fig.suptitle("Nivel esperado por el Modelo 4 dada la composición y el contexto\n"
             "cohorte principal (segundo semestre)", y=0.99)
E.guardar(fig, "fig02_nivel_depto.png", cerrar=True)

# ---- fig03: residuo territorial ----
paso("fig03 residuo territorial")
res = ybD - medD
resw = wmean(res[:, pares], nD[:, pares], ax=1)
sdhat = np.nanstd(YD, axis=-1)

sd_post = D["sd_kt_med"] if "sd_kt_med" in D else D["sd_kt_plugin"]
se = np.sqrt(np.where(nD > 0, sd_post ** 2 / np.maximum(nD, 1), np.nan) + sdhat ** 2)
z = res / se
n_ext = np.nansum(np.abs(z[:, pares]) > 2, axis=1)
nT = nD.sum(axis=1)

fig, axs = plt.subplots(1, 2, figsize=(13.4, 7.4), width_ratios=[1, 1.05])
nrm_r = E.norma_divergente(resw)
E.eje_mapa(axs[0], resw, E.CMAP_RESIDUO, nrm_r)
fig.colorbar(plt.cm.ScalarMappable(norm=nrm_r, cmap=E.CMAP_RESIDUO), ax=axs[0],
             fraction=0.035, pad=0.02, label="observado − esperado (puntos)")
axs[0].set_title("Residuo territorial\nnaranja = rinde por encima de su composición")

ax = axs[1]
ax.scatter(nT, resw, s=42, c=[E.CMAP_RESIDUO(nrm_r(v)) for v in resw],
           edgecolors="0.3", linewidths=0.6, zorder=3)
for i in np.argsort(-n_ext * 100 - np.abs(resw))[:9]:
    ax.annotate(str(dep[i]), (nT[i], resw[i]), xytext=(6, 4),
                textcoords="offset points", fontsize=8, color="0.25")
ax.set_xscale("log"); ax.axhline(0, color="k", lw=0.9)
ax.set_xlabel("observaciones en el departamento (escala log)")
ax.set_ylabel("residuo (puntos)")
ax.set_title(f"Etiquetados: los de más celdas discrepantes.  "
             f"{int(np.nansum(np.abs(z[:, pares]) > 2))} de "
             f"{int(np.isfinite(z[:, pares]).sum())} celdas superan 2 se\n"
             f"({100*np.nanmean(np.abs(z[:, pares]) > 2):.1f}%, frente al 5% "
             f"esperado si el modelo fuera correcto)")
fig.suptitle("Lo que el Modelo 4 no explica: desviaciones departamentales sistemáticas")
fig.tight_layout(rect=(0, 0, 1, 0.94))
E.guardar(fig, "fig03_residuo.png", cerrar=True)

# ---- fig04: heatmap departamento x periodo ----
paso("fig04 heatmap")
Hm = medD[:, pares]
orden = np.argsort(wmean(Hm, nD[:, pares], ax=1))
fig, ax = plt.subplots(figsize=(9.2, 9.4))
im = ax.imshow(Hm[orden], aspect="auto", cmap=E.CMAP_NIVEL,
               vmin=np.nanpercentile(Hm, 2), vmax=np.nanpercentile(Hm, 98))
ax.set_xticks(range(len(pares))); ax.set_xticklabels(etp, rotation=45, fontsize=8)
ax.set_yticks(range(m)); ax.set_yticklabels(dep[orden], fontsize=7.5)
ax.set_ylabel("departamento (ordenado por nivel medio)")
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="puntaje esperado")
ax.set_title("Persistencia territorial: el orden apenas cambia en ocho años")
fig.tight_layout()
E.guardar(fig, "fig04_heatmap.png", cerrar=True)

# ---- fig05: convergencia ----
paso("fig05 convergencia")
wp = np.nan_to_num(nD / nD.sum(axis=0, keepdims=True))
mu_s = np.einsum("kt,kts->ts", wp, np.nan_to_num(YD))
var_s = np.einsum("kt,kts->ts", wp, np.nan_to_num(YD) ** 2) - mu_s ** 2
sd_entre = np.sqrt(np.maximum(var_s, 0))                      # (Tt, S)
qs = np.quantile(sd_entre[pares], [.025, .5, .975], axis=1)

ini, fin = pares[0], pares[-1]
x0 = YD[:, ini, :]; dxy = YD[:, fin, :] - YD[:, ini, :]
b_draw = np.empty(YD.shape[2])
for s in range(0, YD.shape[2], 4):                            # submuestra: basta
    ok = np.isfinite(x0[:, s]) & np.isfinite(dxy[:, s])
    b_draw[s] = np.polyfit(x0[ok, s], dxy[ok, s], 1)[0]
b_draw = b_draw[::4]

fig, axs = plt.subplots(1, 2, figsize=(13.2, 5.0))
ax = axs[0]
ax.fill_between(pares + 1, qs[0], qs[2], color=E.C["D"], alpha=0.2)
ax.plot(pares + 1, qs[1], "o-", color=E.C["D"], lw=2.2, ms=5)
ax.set_xticks(pares + 1); ax.set_xticklabels(etp, rotation=45, fontsize=8)
ax.set_ylabel("desviación estándar entre departamentos (puntos)")
ax.set_title("$\\sigma$-convergencia: ¿se cierran las brechas territoriales?")

ax = axs[1]
ax.scatter(np.nanmedian(x0, axis=1), np.nanmedian(dxy, axis=1), s=40,
           color=E.C["D"], edgecolors="0.3", linewidths=0.6, zorder=3)
for i in range(m):
    ax.annotate(str(dep[i]), (np.nanmedian(x0[i]), np.nanmedian(dxy[i])),
                xytext=(4, 3), textcoords="offset points", fontsize=6.5, color="0.4")
xx = np.linspace(np.nanmin(np.nanmedian(x0, axis=1)),
                 np.nanmax(np.nanmedian(x0, axis=1)), 50)
bm = np.median(b_draw)
am = np.nanmedian(np.nanmedian(dxy, axis=1)) - bm * np.nanmedian(np.nanmedian(x0, axis=1))
ax.plot(xx, am + bm * xx, color="0.25", lw=1.8)
ax.axhline(0, color="k", lw=0.9)
ax.set_xlabel(f"nivel esperado en {etp[0]} (puntos)")
ax.set_ylabel(f"cambio hasta {etp[-1]} (puntos)")
ax.set_title(f"$\\beta$-convergencia: pendiente {bm:+.3f}  "
             f"[P(<0) = {np.mean(b_draw < 0):.2f}]")
fig.suptitle("¿Las brechas territoriales aumentan o disminuyen?")
fig.tight_layout(rect=(0, 0, 1, 0.93))
E.guardar(fig, "fig05_convergencia.png", cerrar=True)

# ---- fig06: nivel municipal ----
paso("fig06 nivel municipal")
cods = [str(int(c)).zfill(5) for c in D["muni_ids"]]
geo_m = E.cargar_geojson(E.RUTA_GEOJSON_MUNI, "mpio_cdpmp", orden=cods, clave=str)
vals = wmean(medM[:, pares], nM[:, pares], ax=1)
nrm_m = E.norma_robusta(vals)
fig, ax = plt.subplots(figsize=(8.2, 9.2))
E.eje_mapa(ax, vals, E.CMAP_NIVEL, nrm_m, geo=geo_m, etiq=False, lw=0.12)
fig.colorbar(plt.cm.ScalarMappable(norm=nrm_m, cmap=E.CMAP_NIVEL), ax=ax,
             fraction=0.035, pad=0.02, label="puntaje esperado (puntos)")
n_dib = sum(1 for c in cods if c in geo_m.K_DE_COD and np.isfinite(vals[geo_m.K_DE_COD[c]]))
ax.set_title("Nivel esperado municipal, promedio de las ocho cohortes principales\n"
             f"{n_dib:,} de {len(cods):,} municipios dibujados; gris = sin datos o sin geometría",
             fontsize=11)
E.guardar(fig, "fig06_nivel_muni.png", cerrar=True)

# ============================== INFORME ======================================
print("\n" + "=" * 78)
print("RESULTADOS DEL PANORAMA")
print("=" * 78)
print("  Descomposicion de la variacion del puntaje:")
for e, f in zip(et_comp, comp):
    print(f"    {e:<24} {100*f:>6.2f}%")
print(f"\n  Media nacional: {ybar_t[pares][0]:.1f} en {etp[0]} -> "
      f"{ybar_t[pares][-1]:.1f} en {etp[-1]}  ({ybar_t[pares][-1]-ybar_t[pares][0]:+.1f})")
print(f"  Minimo de la serie: {ybar_t[pares].min():.1f} en {etp[np.argmin(ybar_t[pares])]}")
print(f"\n  sigma entre departamentos: {qs[1][0]:.2f} -> {qs[1][-1]:.2f} puntos")
dsig = sd_entre[fin] - sd_entre[ini]
print(f"    cambio {np.median(dsig):+.2f}  IC95 [{np.quantile(dsig,.025):+.2f}, "
      f"{np.quantile(dsig,.975):+.2f}]  P(<0) = {np.mean(dsig<0):.3f}")
print(f"  beta-convergencia: pendiente {bm:+.4f}  IC95 "
      f"[{np.quantile(b_draw,.025):+.4f}, {np.quantile(b_draw,.975):+.4f}]  "
      f"P(<0) = {np.mean(b_draw<0):.3f}")
print(f"\n  Persistencia: correlacion de rangos entre {etp[0]} y {etp[-1]} = "
      f"{np.corrcoef(np.argsort(np.argsort(medD[:,ini])), np.argsort(np.argsort(medD[:,fin])))[0,1]:.3f}")
print(f"  Rango territorial en {etp[-1]}: {np.nanmin(medD[:,fin]):.1f} "
      f"({dep[np.nanargmin(medD[:,fin])]}) a {np.nanmax(medD[:,fin]):.1f} "
      f"({dep[np.nanargmax(medD[:,fin])]}) = {np.nanmax(medD[:,fin])-np.nanmin(medD[:,fin]):.1f} puntos")
print(f"  Rango municipal: {np.nanmin(vals):.1f} a {np.nanmax(vals):.1f} = "
      f"{np.nanmax(vals)-np.nanmin(vals):.1f} puntos")
paso("LISTO")
