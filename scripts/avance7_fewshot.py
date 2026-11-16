#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""avance7_fewshot.py: analisis del few-shot (punto 3.9 del director), LOSO, OpenBMI.

Entradas (via scripts/avance7_datos.py):
  base: results/avance_7/reports/openbmi/fewshot/baseline_{m}_loso_completo_global{tag}_epochs_cache_ica.csv
        (semillas 2027 y 2028, F5 y F7; con recorte de gradiente)
  ea:   ..._epochs_cache_ica_ea.csv (semilla 2026, F6; con recorte de gradiente)
Salidas: results/avance_7/analisis/fewshot_*.csv

Comando: ~/anaconda3/envs/eeg/python.exe scripts/avance7_fewshot.py

FAMILIAS (declaradas antes de calcular; ver avance_7/analisis/03_FEWSHOT_TRANSFERENCIA.md):
  F-A1 (base): 4 arq x k{10,20} x metodo{stats,cabeza,stats+cabeza} = 24 contrastes de la
       ganancia por sujeto frente a "cero". Holm sobre los 24.
  F-A2 (ea): los mismos 24 sobre las ejecuciones con alineamiento (informada aparte). Holm sobre 24.
  F-A1T / F-A2T: TOST (margen 0.03) de cada familia, Holm sobre 24 propios.
  Exploratorio (sin familia): dependencia del nivel inicial.
Unidad: sujeto (n = 54). Ganancia por sujeto = promedio sobre 5 sorteos, luego sobre semillas.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import avance7_datos as A
import avance7_stats as S

OUT = A.RAIZ / "results" / "avance_7" / "analisis"
OUT.mkdir(parents=True, exist_ok=True)
METODOS = ["stats", "cabeza", "stats+cabeza"]
KS = [10, 20]
UMBRAL = 0.70


def cargar(variante):
    return pd.concat([A.fewshot(m, variante) for m in A.ARQ], ignore_index=True)


def qa(d, variante, semillas):
    filas = []
    for m, g in d.groupby("arq"):
        esperado = 54 * len(semillas) * 2 * 5 * 4
        c = g.groupby(["test_sub", "seed", "k", "rep"]).metodo.nunique()
        filas.append(dict(
            variante=variante, arq=m, filas=len(g), esperadas=esperado,
            sujetos=g.test_sub.nunique(), semillas=sorted(int(s) for s in g.seed.unique()),
            duplicados=int(g.duplicated(["test_sub", "seed", "k", "rep", "metodo"]).sum()),
            metodos_por_sorteo_ok=bool((c == 4).all()),
            nan_acc=int(g.acc.isna().sum()), cfg_hash="|".join(sorted(g.cfg_hash.unique())),
            n_calib_ok=bool((g.n_calib == g.k).all()), n_eval_ok=bool((g.n_eval == 400 - g.k).all()),
            humo=bool(g.run_id.str.contains("smoke").any())))
    return pd.DataFrame(filas)


def ganancias(d):
    """Una fila por (arq, sub, seed, k, metodo): promedio de los 5 sorteos de acc, cero y ganancia."""
    ancho = d.pivot_table(index=["arq", "test_sub", "seed", "k", "rep"], columns="metodo",
                          values="acc").reset_index()
    filas = []
    for m in METODOS:
        t = ancho[["arq", "test_sub", "seed", "k", "rep", "cero", m]].copy()
        t["metodo"] = m
        t = t.rename(columns={m: "acc"})
        t["gan"] = t["acc"] - t["cero"]
        filas.append(t)
    t = pd.concat(filas)
    return (t.groupby(["arq", "test_sub", "seed", "k", "metodo"])
             .agg(cero=("cero", "mean"), acc=("acc", "mean"), gan=("gan", "mean"),
                  sd_sorteos=("gan", "std")).reset_index())


def contrastes(g, familia, tost_fam):
    s = (g.groupby(["arq", "test_sub", "k", "metodo"])
          .agg(cero=("cero", "mean"), acc=("acc", "mean"), gan=("gan", "mean")).reset_index())
    filas = []
    for m in A.ARQ:
        for k in KS:
            for me in METODOS:
                x = s[(s.arq == m) & (s.k == k) & (s.metodo == me)].sort_values("test_sub")
                d = x.gan.to_numpy()
                w = S.wilcoxon_r(d)
                lo, hi = S.boot_ic(d)
                filas.append(dict(
                    familia=familia, arq=m, k=k, metodo=me, n=w["n"],
                    cero=x.cero.mean(), acc_metodo=x.acc.mean(), gan_media=d.mean(),
                    gan_mediana=float(np.median(d)), ic_inf=lo, ic_sup=hi, p=w["p"], r=w["r"],
                    magnitud=S.magnitud(w["r"]), sujetos_mejoran=int((d > 0).sum()),
                    sujetos_empeoran=int((d < 0).sum()), p_tost=S.tost(d)))
    c = pd.DataFrame(filas)
    c["p_holm"] = S.holm(c.p)
    c["sig_holm"] = c.p_holm < 0.05
    c["p_tost_holm"] = S.holm(c.p_tost)
    c["equiv_003"] = c.p_tost_holm < 0.05
    c["familia_tost"] = tost_fam
    return c


def consistencia_semillas(g):
    filas = []
    for m in A.ARQ:
        for k in KS:
            for me in METODOS:
                x = g[(g.arq == m) & (g.k == k) & (g.metodo == me)]
                a = x[x.seed == 2027].set_index("test_sub").gan
                b = x[x.seed == 2028].set_index("test_sub").gan.loc[a.index]
                wa, wb = S.wilcoxon_r(a.to_numpy()), S.wilcoxon_r(b.to_numpy())
                dif = a - b
                filas.append(dict(
                    arq=m, k=k, metodo=me, gan_2027=a.mean(), gan_2028=b.mean(),
                    dif_entre_semillas=a.mean() - b.mean(),
                    p_2027=wa["p"], p_2028=wb["p"],
                    mismo_signo=bool(np.sign(a.mean()) == np.sign(b.mean())),
                    sig_en_ambas_p05=bool(wa["p"] < 0.05 and wb["p"] < 0.05),
                    rho_sujeto_entre_semillas=float(stats.spearmanr(a, b).statistic),
                    sd_dif_por_sujeto=float(dif.std(ddof=1)),
                    sd_sorteos_media=float(x.sd_sorteos.mean())))
    return pd.DataFrame(filas)


def _nivel_fila(m, ref, k, me, gan, nivel):
    bajo = nivel <= UMBRAL
    gb, ga = gan[bajo], gan[~bajo]
    rho = stats.spearmanr(nivel, gan)
    mw = stats.mannwhitneyu(gb, ga, alternative="two-sided").pvalue if len(gb) > 1 and len(ga) > 1 else np.nan
    return dict(arq=m, referencia=ref, k=k, metodo=me, n_bajo=int(bajo.sum()), n_alto=int((~bajo).sum()),
                gan_bajo=gb.mean() if len(gb) else np.nan, gan_alto=ga.mean(),
                dif_bajo_menos_alto=(gb.mean() - ga.mean()) if len(gb) else np.nan,
                p_mannwhitney=mw, rho_spearman=float(rho.statistic), p_rho=float(rho.pvalue),
                exploratorio=True)


def nivel_inicial_base(g):
    """Exploratorio. Ganancia de una semilla frente al nivel (cero, promedio de k) de la OTRA."""
    filas = []
    lvl = (g[g.metodo == "stats"].groupby(["arq", "test_sub", "seed"]).cero.mean()
           .rename("nivel").reset_index())
    for m in A.ARQ:
        for sg, sl in ((2027, 2028), (2028, 2027)):
            L = lvl[(lvl.arq == m) & (lvl.seed == sl)].set_index("test_sub").nivel
            for k in KS:
                for me in METODOS:
                    x = g[(g.arq == m) & (g.seed == sg) & (g.k == k) & (g.metodo == me)]
                    x = x.set_index("test_sub").gan.loc[L.index]
                    filas.append(_nivel_fila(m, f"gan{sg}_nivel{sl}", k, me, x, L))
    return pd.DataFrame(filas)


def nivel_inicial_ea(gea, gbase):
    lvl = (gbase[gbase.metodo == "stats"].groupby(["arq", "test_sub"]).cero.mean()
           .rename("nivel").reset_index())
    lvl_ac = (gea[gea.metodo == "stats"].groupby(["arq", "test_sub"]).cero.mean()
              .rename("nivel").reset_index())
    filas = []
    for m in A.ARQ:
        for nombre, LL in (("nivel_cero_base_2027_28", lvl), ("nivel_cero_ea_acoplado", lvl_ac)):
            L = LL[LL.arq == m].set_index("test_sub").nivel
            for k in KS:
                for me in METODOS:
                    x = gea[(gea.arq == m) & (gea.k == k) & (gea.metodo == me)].set_index("test_sub").gan.loc[L.index]
                    filas.append(_nivel_fila(m, nombre, k, me, x, L))
    return pd.DataFrame(filas)


def resumen_nivel(n, nombre):
    """Resumen de las filas exploratorias por referencia y metodo: cuantas con p < 0.05 sin corregir."""
    r = []
    for (ref, me), x in n.groupby(["referencia", "metodo"]):
        r.append(dict(conjunto=nombre, referencia=ref, metodo=me, filas=len(x),
                      p_mw_menor_005=int((x.p_mannwhitney < 0.05).sum()),
                      p_rho_menor_005=int((x.p_rho < 0.05).sum()),
                      rho_positivo=int((x.rho_spearman > 0).sum()),
                      dif_bajo_menos_alto_media=x.dif_bajo_menos_alto.mean(),
                      exploratorio=True))
    return pd.DataFrame(r)


def main():
    db, de = cargar("base"), cargar("ea")
    q = pd.concat([qa(db, "base", (2027, 2028)), qa(de, "ea", (2026,))])
    q.to_csv(OUT / "fewshot_qa.csv", index=False)
    pd.set_option("display.width", 250)
    print(q.to_string())
    gb, ge = ganancias(db), ganancias(de)
    gb.assign(variante="base").to_csv(OUT / "fewshot_ganancia_sujeto_semilla_base.csv", index=False)
    ge.assign(variante="ea").to_csv(OUT / "fewshot_ganancia_sujeto_ea.csv", index=False)
    cb = contrastes(gb, "F-A1", "F-A1T")
    ce = contrastes(ge, "F-A2", "F-A2T")
    cb.to_csv(OUT / "fewshot_contrastes_base.csv", index=False)
    ce.to_csv(OUT / "fewshot_contrastes_ea.csv", index=False)
    consistencia_semillas(gb).to_csv(OUT / "fewshot_consistencia_semillas.csv", index=False)
    nb, ne = nivel_inicial_base(gb), nivel_inicial_ea(ge, gb)
    nb.to_csv(OUT / "fewshot_nivel_inicial_base_exploratorio.csv", index=False)
    ne.to_csv(OUT / "fewshot_nivel_inicial_ea_exploratorio.csv", index=False)
    rs = pd.concat([resumen_nivel(nb, "base"), resumen_nivel(ne, "ea")])
    rs.to_csv(OUT / "fewshot_nivel_inicial_resumen_exploratorio.csv", index=False)
    print(rs.round(4).to_string())
    cols = ["arq", "k", "metodo", "cero", "gan_media", "ic_inf", "ic_sup", "p", "p_holm", "r", "equiv_003"]
    print(cb[cols].round(4).to_string())
    print(ce[cols].round(4).to_string())


if __name__ == "__main__":
    main()
