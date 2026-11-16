#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""avance7_tres_semillas_p3.py: recalculo con la media de 3 semillas (2026, 2027, 2028) de las cifras
de `Overleaf/main.tex` que seguian apoyandose en la semilla 2026 (o en una mezcla de semillas).

Informe: avance_7/redaccion/P3_tres_semillas.md
Comando: ~/anaconda3/envs/eeg/python.exe scripts/avance7_tres_semillas_p3.py
Salida:  results/avance_7/analisis/tres_semillas_p3_*.csv

Datos de origen (scripts/avance7_datos.py): within(m) y loso(m, "base") con las semillas 2026, 2027 y 2028
de las cuatro arquitecturas; transferencia("h44_openbmi"|"h44_wbcic", m, "base").

REGLA: antes de calcular nada con 3 semillas se REPRODUCE el valor publicado con el mismo metodo
(`reproducir`). Si alguno no coincide, el script se detiene y lo explica; las excepciones conocidas y
explicadas se marcan en `EXCEPCIONES` y se informan en la columna `nota` de
`tres_semillas_p3_reproduccion_2026.csv` (no se ocultan).

UNIDAD: el sujeto (54; 10 y 51 en transferencia). La media de 3 semillas se toma DENTRO del sujeto (o del
fold, en LOSO, donde un fold = un sujeto); las semillas no se tratan como replicas independientes.
Intervalos: bootstrap, B = 10 000, generador numpy.random.default_rng(2026) nuevo por cantidad
(jerarquico: sujetos y, dentro de cada uno, sus semillas; igual que `avance7_semillas_contrastes.hboot`).

REGIMEN (se declara, no se interpreta): las tres semillas no comparten regimen de optimizacion en LOSO
(2026 sin recorte de gradiente; 2027 y 2028 con recorte, salvo EEGMamba, con recorte siempre) ni en el
intra-sujeto de LaBraM (2026 sin recorte). Hardware mixto (A40 en lo previo; L40S en las semillas 2027 y
2028 salvo intra-sujeto de EEGNet y Conformer, que fue A40).
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
os.chdir(RAIZ)
sys.path.insert(0, str(RAIZ / "scripts"))
import avance7_datos as A  # noqa: E402
import avance7_semillas as X  # noqa: E402  (mat: matriz sujetos x semillas con comprobaciones)
import verificar_cifras as V  # noqa: E402  (rutas y funciones de lo publicado)
from ruido_flotante import wilcoxon_pareado  # noqa: E402

OUT = RAIZ / "results" / "avance_7" / "analisis"
ARQ = A.ARQ
NOM = {"eegnet": "EEGNet", "conformer": "Conformer", "labram": "LaBraM", "eegmamba": "EEGMamba"}
SEEDS = (2026, 2027, 2028)
B = 10_000
GEN = 2026
UMBRAL = 0.70
RESIDUALES = ["sub-06", "sub-29", "sub-32", "sub-37"]
APARTADOS = ["sub-04", "sub-07", "sub-13", "sub-17", "sub-30", "sub-34", "sub-36", "sub-47", "sub-51", "sub-52"]
PFX = "tres_semillas_p3_"

REPRO: list[dict] = []


# ------------------------------------------------------------------ utilidades

def le_u(s):
    return V.le_umbral(s)


def hboot(Xm, nivel=0.95, resta=None):
    """Bootstrap jerarquico de la media sobre sujetos de la media de sus semillas. Xm: (n, k).
    `resta` (n, k) opcional: se remuestrea con los mismos indices y se resta (diferencia pareada)."""
    n, k = Xm.shape
    g = np.random.default_rng(GEN)
    si = g.integers(0, n, (B, n))
    ia = g.integers(0, k, (B, n, k))
    m = Xm[si[:, :, None], ia].mean(axis=2)
    if resta is not None:
        m = m - resta[si[:, :, None], ia].mean(axis=2)
    m = m.mean(axis=1)
    a = (1 - nivel) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


def bsub(x, nivel=0.95):
    x = np.asarray(x, float)
    g = np.random.default_rng(GEN)
    m = x[g.integers(0, len(x), (B, len(x)))].mean(axis=1)
    a = (1 - nivel) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


def repro(item, clave, publicado, calculado, tol, metodo, nota=""):
    ok = abs(float(calculado) - float(publicado)) <= tol + 1e-12
    REPRO.append(dict(item=item, clave=clave, publicado=publicado, reproducido=float(calculado),
                      tolerancia=tol, ok=bool(ok), metodo=metodo, nota=nota))
    return ok


def cm_clase(serie):
    cm = sum(np.array(ast.literal_eval(s)) for s in serie)
    return cm, cm.diagonal() / cm.sum(axis=1)


# ------------------------------------------------------------------ datos
M = {}
for _a in ARQ:
    for _met in ("acc", "kappa", "acc_visto", "kappa_visto"):
        M[("loso", _a, _met)] = X.mat("loso", _a, _met)
    M[("within", _a, "acc")] = X.mat("within", _a, "acc")


def loso_m3(a, met="acc"):
    return M[("loso", a, met)].mean(axis=1)


def loso_26(a, met="acc"):
    return M[("loso", a, met)][2026]


def within_m3(a):
    return M[("within", a, "acc")].mean(axis=1)


def write(df, nombre):
    df.to_csv(OUT / f"{PFX}{nombre}.csv", index=False)
    return df


CIFRAS: dict[str, float] = {}      # id -> valor (para el verificador)


# ================================================================== 1. percentil 5 (tab:robustez)
def p5():
    filas = []
    pub = {"eegnet": 0.572, "conformer": 0.542, "labram": 0.565, "eegmamba": 0.565}
    for a in ARQ:
        x26 = loso_26(a)
        repro("tab:robustez", f"p5_loso_2026.{a}", pub[a], np.percentile(x26, 5), 5e-4 + 1e-9,
              "np.percentile(exactitud LOSO por sujeto, 5), semilla 2026")
        f = dict(arq=a, p5_2026=np.percentile(x26, 5))
        for s in SEEDS:
            f[f"p5_{s}"] = np.percentile(M[("loso", a, "acc")][s], 5)
        f["p5_media3"] = np.percentile(loso_m3(a), 5)
        f["desv_media3"] = loso_m3(a).std(ddof=1)
        filas.append(f)
        CIFRAS[f"p3.p5.{a}"] = f["p5_media3"]
    return write(pd.DataFrame(filas), "robustez_p5")


# ================================================================== 2. IC95 de la exactitud (tab:bootstrap)
def ic_exactitud():
    # reproduccion: bootstrap_ic.py usa UN generador secuencial: 4 LOSO, 4 IDS, 6 diferencias, 4 intra
    rng = np.random.default_rng(GEN)
    pub_loso = {"eegnet": (0.763, 0.730, 0.794), "conformer": (0.752, 0.718, 0.785),
                "labram": (0.754, 0.727, 0.782), "eegmamba": (0.717, 0.691, 0.744)}
    pub_in = {"eegnet": (0.634, 0.606, 0.663), "conformer": (0.602, 0.573, 0.635),
              "labram": (0.665, 0.627, 0.706), "eegmamba": (0.577, 0.552, 0.605)}
    for a in ARQ:
        x = pd.read_csv(V.LOSO[a]).groupby("test_sub").acc.mean().sort_index().to_numpy()
        lo, hi = np.quantile(x[rng.integers(0, 54, (B, 54))].mean(axis=1), [0.025, 0.975])
        for nm, v, p in (("media", x.mean(), pub_loso[a][0]), ("inf", lo, pub_loso[a][1]),
                         ("sup", hi, pub_loso[a][2])):
            repro("tab:bootstrap", f"loso.{a}.{nm}", p, v, 5e-4 + 1e-9,
                  "bootstrap_ic.py: generador secuencial 2026 (4 LOSO, 4 IDS, 6 dif., 4 intra), solo sujetos")
    for _ in range(10):
        rng.integers(0, 54, (B, 54))
    for a in ARQ:
        x = pd.read_csv(V.WITHIN[a]).groupby("sub").acc.mean().sort_index().to_numpy()
        lo, hi = np.quantile(x[rng.integers(0, 54, (B, 54))].mean(axis=1), [0.025, 0.975])
        sem = sorted(pd.read_csv(V.WITHIN[a]).seed.unique())
        nota = "" if len(sem) == 3 or a in ("labram", "eegmamba") else ""
        for nm, v, p in (("media", x.mean(), pub_in[a][0]), ("inf", lo, pub_in[a][1]),
                         ("sup", hi, pub_in[a][2])):
            repro("tab:bootstrap", f"intra.{a}.{nm}", p, v, 5e-4 + 1e-9,
                  f"bootstrap_ic.py (secuencial); archivo previo con semillas {sem}",
                  "el bloque rotulado 'Semilla 2026' mezcla: EEGNet y Conformer intra-sujeto son la media de "
                  "tres semillas del archivo previo" if len(sem) == 3 else "")
    filas = []
    for proto in ("loso", "intra"):
        for a in ARQ:
            Xm = M[("loso", a, "acc")] if proto == "loso" else M[("within", a, "acc")]
            m3 = Xm.mean(axis=1)
            jlo, jhi = hboot(Xm.to_numpy())
            slo, shi = bsub(m3.to_numpy())
            x26 = Xm[2026].to_numpy()
            l26, h26 = bsub(x26)
            filas.append(dict(protocolo=proto, arq=a, media3=m3.mean(), jer_ic95_inf=jlo, jer_ic95_sup=jhi,
                              jer_amplitud=jhi - jlo, sujetos_ic95_inf=slo, sujetos_ic95_sup=shi,
                              sujetos_amplitud=shi - slo, media_2026=x26.mean(), ic95_2026_inf=l26,
                              ic95_2026_sup=h26))
            CIFRAS[f"p3.ic.{proto}.{a}.media"] = m3.mean()
            CIFRAS[f"p3.ic.{proto}.{a}.inf"] = jlo
            CIFRAS[f"p3.ic.{proto}.{a}.sup"] = jhi
    return write(pd.DataFrame(filas), "bootstrap_exactitud")


# ================================================================== 3. interseccion en LOSO
def interseccion_loso():
    filas = []
    pub_conj = {a: set(loso_26(a).index[le_u(loso_26(a))]) for a in ARQ}
    repro("sec:within-vs-loso / discusion", "interseccion_loso_2026", 11, len(set.intersection(*pub_conj.values())),
          0.5, "sujetos con exactitud LOSO <= 0.70 (tolerancia 1e-9) en las cuatro, semilla 2026")
    vistas = {}
    for v in list(SEEDS) + ["media3"]:
        conj = {}
        for a in ARQ:
            x = loso_m3(a) if v == "media3" else M[("loso", a, "acc")][v]
            conj[a] = set(x.index[le_u(x)])
        vistas[v] = conj
        inter = set.intersection(*conj.values())
        bajo = pd.DataFrame({a: pd.Series(1, index=sorted(conj[a])) for a in ARQ}).reindex(
            loso_26("eegnet").index).notna()
        k = bajo.sum(axis=1)
        dist = np.array([1.0])
        for p in bajo.mean():
            dist = np.convolve(dist, [1 - p, p])
        f = dict(semilla=v, interseccion_4=len(inter), n_eegnet=len(conj["eegnet"]), n_conformer=len(conj["conformer"]),
                 n_labram=len(conj["labram"]), n_eegmamba=len(conj["eegmamba"]),
                 esperado_k4=dist[4] * 54)
        for i in range(5):
            f[f"obs_k{i}"] = int((k == i).sum())
        filas.append(f)
        if v == "media3":
            CIFRAS["p3.loso.interseccion"] = len(inter)
            for a in ARQ:
                CIFRAS[f"p3.loso.n_bajo.{a}"] = len(conj[a])
            CIFRAS["p3.loso.esperado_k4"] = dist[4] * 54
    # estabilidad: sujetos que estan en la interseccion con las 3 semillas
    inter3 = [set.intersection(*vistas[s].values()) for s in SEEDS]
    filas.append(dict(semilla="en_las_3_semillas", interseccion_4=len(set.intersection(*inter3))))
    filas.append(dict(semilla="en_alguna_semilla", interseccion_4=len(set.union(*inter3))))
    CIFRAS["p3.loso.interseccion_en_las_3"] = len(set.intersection(*inter3))
    CIFRAS["p3.loso.interseccion_en_alguna"] = len(set.union(*inter3))
    return write(pd.DataFrame(filas), "interseccion_loso")


# ================================================================== 4. distribucion de EEGNet (tab:distribucion)
def distribucion():
    pub = dict(q1=0.670, med=0.770, q3=0.844, mn=0.522, mx=0.993, n_lt=6, n_gt=20)
    x26 = loso_26("eegnet")
    q = x26.quantile([.25, .5, .75]).to_numpy()
    for k, v, t in (("q1", q[0], 5e-4), ("med", q[1], 5e-4), ("q3", q[2], 5e-4), ("mn", x26.min(), 5e-4),
                    ("mx", x26.max(), 5e-4), ("n_lt", (x26 < 0.60).sum(), 0.5), ("n_gt", (x26 > 0.80).sum(), 0.5)):
        repro("tab:distribucion", k, pub[k], v, t + 1e-9, "EEGNet LOSO, semilla 2026; cuartiles por interpolacion lineal")
    filas = []
    for v in list(SEEDS) + ["media3"]:
        x = loso_m3("eegnet") if v == "media3" else M[("loso", "eegnet", "acc")][v]
        qq = x.quantile([.25, .5, .75]).to_numpy()
        filas.append(dict(semilla=v, q1=qq[0], mediana=qq[1], q3=qq[2], minimo=x.min(), sub_minimo=x.idxmin(),
                          maximo=x.max(), sub_maximo=x.idxmax(), n_lt_060=int((x < 0.60).sum()),
                          n_gt_080=int((x > 0.80).sum()), media=x.mean(), sd=x.std(ddof=1)))
        if v == "media3":
            for k, val in (("q1", qq[0]), ("med", qq[1]), ("q3", qq[2]), ("mn", x.min()), ("mx", x.max()),
                           ("n_lt", int((x < 0.60).sum())), ("n_gt", int((x > 0.80).sum()))):
                CIFRAS[f"p3.dist.{k}"] = float(val)
    return write(pd.DataFrame(filas), "distribucion")


# ================================================================== 5. sensibilidad (tab:sensibilidad)
def sensibilidad():
    def grupo(kv, k, ac, idx):
        return dict(n=len(idx), acc=ac.loc[idx].mean(), dsi_kappa=(kv.loc[idx].mean() - k.loc[idx].mean()) / kv.loc[idx].mean())
    pub = {"res": (0.839, -0.180), "resto": (0.756, 0.092)}
    resto = [s for s in M[("loso", "eegnet", "acc")].index if s not in RESIDUALES]
    filas = []
    for v in list(SEEDS) + ["media3"]:
        sel = (lambda met: loso_m3("eegnet", met)) if v == "media3" else (lambda met, v=v: M[("loso", "eegnet", met)][v])
        kv, k, ac = sel("kappa_visto"), sel("kappa"), sel("acc")
        for nm, idx in (("residuales", RESIDUALES), ("resto", resto)):
            g = grupo(kv, k, ac, idx)
            filas.append(dict(semilla=v, grupo=nm, **g))
            if v == 2026:
                p = pub["res" if nm == "residuales" else "resto"]
                repro("tab:sensibilidad", f"{nm}.acc", p[0], g["acc"], 5e-4 + 1e-9, "EEGNet LOSO 2026")
                repro("tab:sensibilidad", f"{nm}.dsi_kappa", p[1], g["dsi_kappa"], 5e-4 + 1e-9,
                      "(kappa_visto - kappa)/kappa_visto con medias del grupo de folds")
            if v == "media3":
                CIFRAS[f"p3.sens.{nm}.acc"] = g["acc"]
                CIFRAS[f"p3.sens.{nm}.dsi_kappa"] = g["dsi_kappa"]
    return write(pd.DataFrame(filas), "sensibilidad")


# ================================================================== 6. matriz de confusion y restriccion
def confusion_restriccion():
    d = A.loso("eegnet")
    filas = []
    for v in list(SEEDS) + ["media3"]:
        sub = d if v == "media3" else d[d.seed == v]
        cm, rec = cm_clase(sub.confusion_matrix)
        filas.append(dict(semilla=v, clase0_acierto=rec[0], clase1_acierto=rec[1], n_clase0=int(cm[0].sum()),
                          n_clase1=int(cm[1].sum()), cm00=int(cm[0, 0]), cm01=int(cm[0, 1]),
                          cm10=int(cm[1, 0]), cm11=int(cm[1, 1])))
        if v == 2026:
            repro("sec:within-vs-loso", "confusion.clase0", 0.764, rec[0], 5e-4 + 1e-9, "suma de matrices de los 54 folds, 2026")
            repro("sec:within-vs-loso", "confusion.clase1", 0.761, rec[1], 5e-4 + 1e-9, "suma de matrices de los 54 folds, 2026")
        # asimetria entre clases por sujeto (fold): acierto de la clase 0 menos el de la clase 1, media de semillas
        r = sub.assign(r0=[ast.literal_eval(c)[0][0] / sum(ast.literal_eval(c)[0]) for c in sub.confusion_matrix],
                       r1=[ast.literal_eval(c)[1][1] / sum(ast.literal_eval(c)[1]) for c in sub.confusion_matrix])
        pf = r.groupby("test_sub")[["r0", "r1"]].mean()
        dd = (pf.r0 - pf.r1).to_numpy()
        pw = float(wilcoxon_pareado(pf.r0, pf.r1).pvalue)
        filas[-1].update(dif_clases_media_por_sujeto=float(dd.mean()), p_wilcoxon_clase0_vs_clase1=pw,
                         n_sujetos_clase0_mayor=int((dd > 0).sum()))
        if v == "media3":
            CIFRAS["p3.conf.clase0"], CIFRAS["p3.conf.clase1"] = float(rec[0]), float(rec[1])
            CIFRAS["p3.conf.dif"], CIFRAS["p3.conf.p"] = float(dd.mean()), pw
            CIFRAS["p3.conf.n_clase0_mayor"] = int((dd > 0).sum())
    write(pd.DataFrame(filas), "confusion_eegnet")
    # restriccion de interpretacion (kappa, EEGNet): desviacion de vistos, de no vistos y r
    filas = []
    for v in list(SEEDS) + ["media3"]:
        sel = (lambda met: loso_m3("eegnet", met)) if v == "media3" else (lambda met, v=v: M[("loso", "eegnet", met)][v])
        kv, k = sel("kappa_visto"), sel("kappa")
        dsi = (kv - k) / kv
        r = float(np.corrcoef(dsi, k)[0, 1])
        filas.append(dict(semilla=v, sd_kappa_visto=kv.std(ddof=1), sd_kappa_no_visto=k.std(ddof=1), r_dsi_kappa=r))
        if v == 2026:
            repro("sec:restriccion", "sd_kappa_visto", 0.022, kv.std(ddof=1), 5e-4 + 1e-9, "EEGNet 2026, ddof=1, 54 folds")
            repro("sec:restriccion", "sd_kappa_no_visto", 0.239, k.std(ddof=1), 5e-4 + 1e-9, "EEGNet 2026, ddof=1, 54 folds")
            repro("sec:restriccion", "r", -0.996, r, 5e-4 + 1e-9, "Pearson entre el DSI de cada fold y el kappa del sujeto de prueba")
        if v == "media3":
            CIFRAS["p3.restr.sd_kv"], CIFRAS["p3.restr.sd_k"], CIFRAS["p3.restr.r"] = kv.std(ddof=1), k.std(ddof=1), r
    return write(pd.DataFrame(filas), "restriccion_kappa")


# ================================================================== 7. +- de vistos (tab:conformer)
def sd_vistos():
    pub = {"eegnet": (0.011, 0.022), "conformer": (0.018, 0.037), "labram": (0.013, 0.026), "eegmamba": (0.012, 0.023)}
    filas = []
    for a in ARQ:
        for v in list(SEEDS) + ["media3"]:
            av = loso_m3(a, "acc_visto") if v == "media3" else M[("loso", a, "acc_visto")][v]
            kv = loso_m3(a, "kappa_visto") if v == "media3" else M[("loso", a, "kappa_visto")][v]
            ac = loso_m3(a, "acc") if v == "media3" else M[("loso", a, "acc")][v]
            filas.append(dict(arq=a, semilla=v, media_acc_visto=av.mean(), sd_acc_visto=av.std(ddof=1),
                              media_kappa_visto=kv.mean(), sd_kappa_visto=kv.std(ddof=1),
                              media_acc_no_visto=ac.mean(), sd_acc_no_visto=ac.std(ddof=1)))
            if v == 2026:
                repro("tab:conformer", f"sd_acc_visto.{a}", pub[a][0], av.std(ddof=1), 5e-4 + 1e-9,
                      "desviacion entre los 54 folds, 2026 (cifra de main_antes_integracion.tex)")
                repro("tab:conformer", f"sd_kappa_visto.{a}", pub[a][1], kv.std(ddof=1), 5e-4 + 1e-9,
                      "desviacion entre los 54 folds, 2026 (cifra de main_antes_integracion.tex)")
            if v == "media3":
                CIFRAS[f"p3.sdv.acc.{a}"] = av.std(ddof=1)
                CIFRAS[f"p3.sdv.kappa.{a}"] = kv.std(ddof=1)
    return write(pd.DataFrame(filas), "sd_vistos")


# ================================================================== 8. Esas / Resto (tab:seleccion-intra)
def seleccion():
    filas = []
    for proto in ("within", "loso"):
        pubfiles = V.WITHIN if proto == "within" else V.LOSO
        for a in ARQ:
            for k, (prop, esas, resto) in {"w": V._SEL["within"], "l": V._SEL["loso"]}[proto[0]].items():
                if k != a:
                    continue
                repro("tab:seleccion-intra (avances anteriores)", f"{proto}.{a}.esas", esas,
                      V._calentamiento(pubfiles[a], "esas"), 5e-4 + 1e-9, "verificar_cifras._calentamiento (archivo previo, sin filtrar semilla)")
                repro("tab:seleccion-intra (avances anteriores)", f"{proto}.{a}.resto", resto,
                      V._calentamiento(pubfiles[a], "resto"), 5e-4 + 1e-9, "idem")
            d = A.within(a) if proto == "within" else A.loso(a, "base")
            for v in list(SEEDS) + ["agrupadas3"]:
                s = d if v == "agrupadas3" else d[d.seed == v]
                m = s.n_epochs == 30
                filas.append(dict(protocolo=proto, arq=a, semilla=v, n=len(s), prop_calentamiento=100 * m.mean(),
                                  n_esas=int(m.sum()), acc_esas=s.acc[m].mean(), acc_resto=s.acc[~m].mean()))
                if v == "agrupadas3":
                    CIFRAS[f"p3.sel.{proto}.{a}.esas"] = s.acc[m].mean()
                    CIFRAS[f"p3.sel.{proto}.{a}.resto"] = s.acc[~m].mean()
                    CIFRAS[f"p3.sel.{proto}.{a}.prop"] = 100 * m.mean()
    df = pd.DataFrame(filas)
    # por semilla: rango (min, max) de las tres
    for proto in ("within", "loso"):
        sub = df[(df.protocolo == proto) & (df.semilla != "agrupadas3")]
        for a in ARQ:
            s = sub[sub.arq == a]
            CIFRAS[f"p3.sel.{proto}.{a}.esas_min"] = s.acc_esas.min()
            CIFRAS[f"p3.sel.{proto}.{a}.esas_max"] = s.acc_esas.max()
            CIFRAS[f"p3.sel.{proto}.{a}.resto_min"] = s.acc_resto.min()
            CIFRAS[f"p3.sel.{proto}.{a}.resto_max"] = s.acc_resto.max()
    return write(df, "seleccion_calentamiento")


# ================================================================== 9. diez apartados frente al resto (LOSO)
def apartados():
    filas = []
    pub = {"eegnet": (0.739, 0.768, 0.33), "conformer": (None, None, 0.456), "labram": (None, None, 0.82),
           "eegmamba": (None, None, 0.35)}
    for a in ARQ:
        for v in list(SEEDS) + ["media3"]:
            x = loso_m3(a) if v == "media3" else M[("loso", a, "acc")][v]
            g, o = x.loc[APARTADOS], x.drop(APARTADOS)
            p = float(stats.mannwhitneyu(g, o, alternative="two-sided").pvalue)
            filas.append(dict(arq=a, semilla=v, media_apartados=g.mean(), media_resto=o.mean(), dif=o.mean() - g.mean(),
                              p_mannwhitney=p, n_apartados=len(g), n_resto=len(o)))
            if v == 2026:
                pb = pub[a]
                if pb[0] is not None:
                    repro("sec:protocolos-extra", f"{a}.apartados", pb[0], g.mean(), 5e-4 + 1e-9, "LOSO 2026")
                    repro("sec:protocolos-extra", f"{a}.resto", pb[1], o.mean(), 5e-4 + 1e-9, "LOSO 2026")
                repro("sec:protocolos-extra", f"{a}.p_MW", pb[2], p, 5e-3 + 1e-9, "Mann-Whitney bilateral, 2026")
            if v == "media3":
                CIFRAS[f"p3.apart.{a}.apartados"] = g.mean()
                CIFRAS[f"p3.apart.{a}.resto"] = o.mean()
                CIFRAS[f"p3.apart.{a}.dif"] = o.mean() - g.mean()
                CIFRAS[f"p3.apart.{a}.p"] = p
    df = pd.DataFrame(filas)
    # diferencias redondeadas de las medias de 2026 (lo publicado: "entre 2.1 y 3.3 puntos")
    d26 = df[df.semilla == 2026]
    repro("sec:protocolos-extra", "dif_min_puntos", 2.1, 100 * d26.dif.min(), 0.05 + 1e-9,
          "diferencia resto menos apartados (puntos) 2026, SIN redondear las medias",
          "NO REPRODUCE: da 2.0; lo publicado sale de restar medias ya redondeadas a tres decimales (0.758 - 0.737)")
    repro("sec:protocolos-extra", "dif_max_puntos", 3.3, 100 * d26.dif.max(), 0.05 + 1e-9,
          "idem", "NO REPRODUCE: da 3.2; lo publicado sale de 0.758 - 0.725")
    return write(df, "apartados_loso")


# ================================================================== 10. desviaciones de transferencia
def sd_transferencia():
    filas = []
    pub = {"persona": (0.098, 0.136), "wbcic": (0.106, 0.132)}
    for tipo, nom in (("h44_openbmi", "persona"), ("h44_wbcic", "wbcic")):
        n_esp = 10 if nom == "persona" else 51
        sds = {v: [] for v in list(SEEDS) + ["media3"]}
        for a in ARQ:
            d = A.transferencia(tipo, a, "base")
            d = d[d.norm_destino == "source"]
            assert set(d.seed.unique()) == set(SEEDS), (a, tipo, d.seed.unique())
            p = d.pivot_table(index="sub", columns="seed", values="acc", aggfunc="mean")
            assert p.shape == (n_esp, 3) and not p.isna().any().any(), (a, tipo, p.shape)
            for v in list(SEEDS) + ["media3"]:
                col = p.mean(axis=1) if v == "media3" else p[v]
                filas.append(dict(conjunto=nom, arq=a, semilla=v, n_sujetos=n_esp, media=col.mean(), sd_entre_sujetos=col.std(ddof=1)))
                sds[v].append(col.std(ddof=1))
                if v == "media3":
                    CIFRAS[f"p3.sdt.{nom}.{a}"] = col.std(ddof=1)
        repro("sec:descomposicion", f"sd_{nom}.min", pub[nom][0], min(sds[2026]), 5e-4 + 1e-9, "politica source, 2026, ddof=1")
        repro("sec:descomposicion", f"sd_{nom}.max", pub[nom][1], max(sds[2026]), 5e-4 + 1e-9, "politica source, 2026, ddof=1")
        CIFRAS[f"p3.sdt.{nom}.min"] = min(sds["media3"])
        CIFRAS[f"p3.sdt.{nom}.max"] = max(sds["media3"])
    return write(pd.DataFrame(filas), "sd_transferencia")


# ================================================================== 11. limite superior de la caida por persona en LOSO
def persona_loso_ic():
    filas = []
    sup26, sup3 = [], []
    for a in ARQ:
        x = (M[("loso", a, "acc_visto")] - M[("loso", a, "acc")])
        x26 = x[2026].to_numpy()
        g = np.random.default_rng(GEN)
        s26 = float(np.quantile(x26[g.integers(0, 54, (B, 54))].mean(1), 0.975))
        sup26.append(s26)
        jlo, jhi = hboot(x.to_numpy())
        slo, shi = bsub(x.mean(axis=1).to_numpy())
        filas.append(dict(arq=a, caida_2026=x26.mean(), ic95_sup_2026=s26, caida_media3=x.mean(axis=1).mean(),
                          jer_ic95_inf=jlo, jer_ic95_sup=jhi, sujetos_ic95_inf=slo, sujetos_ic95_sup=shi))
        sup3.append(jhi)
        CIFRAS[f"p3.persona_loso.{a}.media"] = x.mean(axis=1).mean()
        CIFRAS[f"p3.persona_loso.{a}.jer_sup"] = jhi
    repro("sec:descomposicion", "persona_loso_ic_sup.min", 0.053, min(sup26), 5e-4 + 1e-9, "IC95 percentil, folds, 2026")
    repro("sec:descomposicion", "persona_loso_ic_sup.max", 0.074, max(sup26), 5e-4 + 1e-9, "IC95 percentil, folds, 2026")
    CIFRAS["p3.persona_loso.jer_sup.min"], CIFRAS["p3.persona_loso.jer_sup.max"] = min(sup3), max(sup3)
    return write(pd.DataFrame(filas), "persona_loso_ic")


# ================================================================== 12. EEGMamba intra-sujeto frente al azar
def azar_eegmamba():
    y26 = M[("within", "eegmamba", "acc")][2026].sort_index()
    p_sin = float(stats.wilcoxon(y26 - 0.5).pvalue)
    repro("sec:comparacion / sec:wilcoxon", "p_azar_eegmamba_2026", 1.5e-8, p_sin, 0.05e-8 + 1e-12,
          "stats.wilcoxon(y - 0.5) SIN redondear a 1e-9 (como se publico)",
          "con el redondeo a 1e-9 acordado (ruido_flotante) el mismo contraste da otro valor, ver columna siguiente")
    filas = []
    for v in list(SEEDS) + ["media3"]:
        y = within_m3("eegmamba") if v == "media3" else M[("within", "eegmamba", "acc")][v]
        d = y - 0.5
        p_r = float(wilcoxon_pareado(d).pvalue)
        p_s = float(stats.wilcoxon(d).pvalue)
        filas.append(dict(semilla=v, media=y.mean(), p_redondeado_1e9=p_r, p_sin_redondear=p_s,
                          n_sobre_azar=int((d > 0).sum())))
        if v == 2026:
            CIFRAS["p3.azar.2026_redondeado"] = p_r
        if v == "media3":
            CIFRAS["p3.azar.media3"] = p_r
            CIFRAS["p3.azar.media3_sin_redondear"] = p_s
    # cabezas: la variante de promediado (avgpool) solo existe con la semilla 2026
    av = V._por_sujeto(V.EEGMAMBA_AVGPOOL)
    assert sorted(pd.read_csv(V.EEGMAMBA_AVGPOOL).seed.unique()) == [2026]
    d = (y26 - av)
    p_sin = float(stats.wilcoxon(d).pvalue)
    repro("sec:comparacion / sec:wilcoxon", "p_cabezas_2026", 2.6e-6, p_sin, 0.05e-6 + 1e-12,
          "stats.wilcoxon(flatten - avgpool) sin redondear, ambas semilla 2026")
    z = abs(stats.norm.ppf(p_sin / 2))
    repro("sec:comparacion / sec:wilcoxon", "r_cabezas_2026", 0.64, z / np.sqrt(len(d)), 5e-3 + 1e-9, "r = z / sqrt(n)")
    return write(pd.DataFrame(filas), "azar_eegmamba")


# ================================================================== 13. figuras (valores representados)
def figuras():
    # fig:solapamiento
    filas = []
    acc3 = pd.DataFrame({a: within_m3(a) for a in ARQ})
    bajo = pd.DataFrame({a: le_u(acc3[a]) for a in ARQ})
    k = bajo.sum(axis=1)
    dist = np.array([1.0])
    for p in bajo.mean():
        dist = np.convolve(dist, [1 - p, p])
    # lo publicado (mezcla: EEGNet y Conformer 3 semillas; LaBraM y EEGMamba 2026)
    pubm = V._matriz_bajo()
    for i, v in {0: 5, 1: 3, 2: 4, 3: 8, 4: 34}.items():
        repro("fig:solapamiento", f"obs_k{i}", v, int((pubm.sum(axis=1) == i).sum()), 0.5, "figura_solapamiento.py (modo anterior: mezcla de semillas)")
    for a in ARQ:
        repro("fig:solapamiento", f"n_bajo.{a}", {"eegnet": 41, "conformer": 45, "labram": 37, "eegmamba": 48}[a],
              int(pubm[a].sum()), 0.5, "idem")
    for i in range(5):
        filas.append(dict(k=i, observado=int((k == i).sum()), esperado=dist[i] * 54))
        CIFRAS[f"p3.sol.obs{i}"] = int((k == i).sum())
        CIFRAS[f"p3.sol.esp{i}"] = dist[i] * 54
    for a in ARQ:
        CIFRAS[f"p3.sol.n_bajo.{a}"] = int(bajo[a].sum())
    CIFRAS["p3.sol.interseccion"] = int(bajo.all(axis=1).sum())
    write(pd.DataFrame(filas), "fig_solapamiento")
    # fig:resultados-baseline
    lo3 = loso_m3("eegnet").round(9)       # mismo redondeo a 1e-9 que scripts/figuras_resultados.py --semillas 3
    wi3 = within_m3("eegnet").round(9)
    lo26 = loso_26("eegnet")
    pubw = pd.read_csv(V.WITHIN["eegnet"]).groupby("sub").acc.mean()     # 3 semillas (archivo previo)
    repro("fig:resultados-baseline", "n_sobre_diagonal", 48, int((lo26 > pubw).sum()), 0.5,
          "LOSO 2026 frente a intra-sujeto del archivo previo (3 semillas)")
    repro("fig:resultados-baseline", "mediana_loso", 0.770, lo26.median(), 5e-4 + 1e-9, "mediana LOSO 2026")
    rows = []
    for nom, lo in (("2026", lo26), ("media3", lo3)):
        rows.append(dict(semilla=nom, mediana_loso=lo.median(), n_sobre_diagonal=int((lo > wi3).sum()),
                         n_sujetos=len(lo), n_residuales=len(RESIDUALES), media_loso=lo.mean(),
                         bins_050_a_100_paso005=" ".join(str(int(c)) for c in np.histogram(lo, bins=np.arange(0.50, 1.001, 0.05))[0])))
    CIFRAS["p3.fig_base.mediana"] = lo3.median()
    CIFRAS["p3.fig_base.n_sobre_diag"] = int((lo3 > wi3).sum())
    write(pd.DataFrame(rows), "fig_resultados_baseline")


def main():
    p5()
    ic_exactitud()
    interseccion_loso()
    distribucion()
    sensibilidad()
    confusion_restriccion()
    sd_vistos()
    seleccion()
    apartados()
    sd_transferencia()
    persona_loso_ic()
    azar_eegmamba()
    figuras()
    rep = pd.DataFrame(REPRO)
    write(rep, "reproduccion_2026")
    pd.DataFrame([dict(id=k, valor=v) for k, v in CIFRAS.items()]).to_csv(OUT / f"{PFX}cifras.csv", index=False)
    mal = rep[~rep.ok]
    print(f"[p3] {len(rep)} reproducciones, {int(rep.ok.sum())} ok, {len(mal)} no coinciden")
    if len(mal):
        print(mal[["item", "clave", "publicado", "reproducido", "nota"]].to_string())
    print(f"[p3] {len(CIFRAS)} cifras nuevas en {PFX}cifras.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
