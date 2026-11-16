#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""avance7_semillas.py: semillas, cota del confusor (recorte de gradiente) y recalculo con 3 semillas.

Informe: avance_7/analisis/04_SEMILLAS_Y_CONFUSOR.md
Salida:  results/avance_7/analisis/{semillas_reproduccion_2026,confusor_*,semillas_*,tres_semillas_*}.csv
Comando: ~/anaconda3/envs/eeg/python.exe scripts/avance7_semillas.py          (todo)
         ~/anaconda3/envs/eeg/python.exe scripts/avance7_semillas.py --etapa confusor
Etapas:  reproducir (siempre primero; si falla no se calcula nada nuevo), confusor,
         contrastes (scripts/avance7_semillas_contrastes.py),
         derivados (scripts/avance7_semillas_derivados.py)

Datos de origen (via scripts/avance7_datos.py, que fija la ruta de cada CSV):
    within(m)        results/pod_*/ (semilla 2026 y, en eegnet y conformer, 2027-2028) +
                     results/avance_7/reports/openbmi/*_within_* (labram y eegmamba, 2027-2028)
    loso(m, "base")  results/pod_*/ (2026) + results/avance_7/reports/openbmi/*_loso_* (2027, 2028)
    transferencia("h44_openbmi"|"h44_wbcic", m, "base")  results/punto6b/holdout44 (2026) +
                     results/avance_7/transferencia/base (2027, 2028)

Definiciones que NO se mezclan (cada columna lleva el nombre de la suya):
    exactitud por sujeto y semilla : media de las filas del sujeto en esa semilla
                                     (LOSO: 1 pliegue; intra-sujeto: 20 pliegues)
    media_semillas                 : media de las 3 semillas de esa cantidad
    sd_entre_semillas              : desviacion tipica (ddof=1) de los 3 valores por semilla de la
                                     media sobre sujetos
    sd_entre_sujetos               : desviacion tipica (ddof=1) sobre los 54 sujetos
    sd_semilla_dentro_sujeto       : media sobre sujetos de la desviacion (ddof=1) entre las 3 semillas

Generadores: todo remuestreo usa numpy.random.default_rng(2026), con generador NUEVO por cantidad
(el resultado de una cantidad no depende del orden de calculo de las demas), B = 10 000.

Regimen de entrenamiento (verificado por cfg_hash en cada ejecucion del script):
    SIN recorte de gradiente: cfg_hash 53d6a9fc85b4 (eegnet), 728ed36e1f36 (conformer),
    7b96314fb256 (labram); CON recorte 1.0: el resto (221b9bb4d26a, 6f16811d008d, aad8ea7882e6,
    eabca26d5bdf). EEGMamba: recorte en ambos.
"""
from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
os.chdir(RAIZ)
sys.path.insert(0, str(RAIZ / "scripts"))
import avance7_datos as A  # noqa: E402
import avance7_stats as S  # noqa: E402

OUT = RAIZ / "results" / "avance_7" / "analisis"
OUT.mkdir(parents=True, exist_ok=True)

B = 10_000
SEMILLA_GEN = 2026
SEEDS = (2026, 2027, 2028)
ARQ = A.ARQ
NOMBRE = {"eegnet": "EEGNet", "conformer": "Conformer", "labram": "LaBraM", "eegmamba": "EEGMamba"}
PARES = [("eegnet", "conformer"), ("eegnet", "labram"), ("conformer", "labram"),
         ("eegnet", "eegmamba"), ("conformer", "eegmamba"), ("labram", "eegmamba")]   # orden de tab:wilcoxon
UMBRAL = 0.70
MARGEN = 0.03
HASH_SIN_RECORTE = {"53d6a9fc85b4", "728ed36e1f36", "7b96314fb256"}


def rng():
    return np.random.default_rng(SEMILLA_GEN)


# ------------------------------------------------------------------ datos

_CACHE: dict = {}


def tabla(proto: str, arq: str) -> pd.DataFrame:
    k = (proto, arq)
    if k not in _CACHE:
        _CACHE[k] = A.within(arq) if proto == "within" else A.loso(arq, "base")
    return _CACHE[k]


def mat(proto: str, arq: str, metrica: str = "acc") -> pd.DataFrame:
    """Matriz 54 x 3 (sujetos x semillas) de la metrica, promediada dentro de sujeto y semilla."""
    d = tabla(proto, arq)
    col = "sub" if proto == "within" else "test_sub"
    n_esp = 20 if proto == "within" else 1
    cuenta = d.groupby([col, "seed"]).size()
    assert (cuenta == n_esp).all() and len(cuenta) == 54 * 3, (proto, arq)
    p = d.pivot_table(index=col, columns="seed", values=metrica, aggfunc="mean")
    assert p.shape == (54, 3) and not p.isna().any().any()
    return p.sort_index()[list(SEEDS)]


def sin_recorte(proto: str, arq: str, seed: int) -> bool:
    """Regimen declarado por Felipe; `verificar_regimen` lo contrasta con cfg_hash."""
    if arq == "eegmamba":
        return False
    if proto == "loso":
        return seed == 2026
    if arq in ("eegnet", "conformer"):
        return True
    return seed == 2026     # labram intra-sujeto


def verificar_regimen() -> pd.DataFrame:
    filas = []
    for proto in ("loso", "within"):
        for arq in ARQ:
            d = tabla(proto, arq)
            for s in SEEDS:
                hs = set(d[d.seed == s].cfg_hash.unique())
                assert len(hs) == 1, (proto, arq, s, hs)
                h = hs.pop()
                dec = sin_recorte(proto, arq, s)
                assert (h in HASH_SIN_RECORTE) == dec, \
                    f"regimen declarado distinto del cfg_hash: {proto} {arq} {s} {h}"
                filas.append(dict(protocolo=proto, arq=arq, seed=s, cfg_hash=h, sin_recorte=dec,
                                  regimen_fuente=d[d.seed == s].regimen.iloc[0]))
    return pd.DataFrame(filas)


def regimen_str(proto, arq, seeds=SEEDS):
    """S = sin recorte, C = con recorte, una letra por semilla."""
    return "".join("S" if sin_recorte(proto, arq, s) else "C" for s in seeds)


# ------------------------------------------------------------- utilidades estadisticas

def boot_media_ic(x, nivel=0.95, seed=SEMILLA_GEN, b=B):
    x = np.asarray(x, float)
    g = np.random.default_rng(seed)
    m = x[g.integers(0, len(x), (b, len(x)))].mean(axis=1)
    a = (1 - nivel) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


def t_ic(x, nivel=0.95):
    x = np.asarray(x, float)
    se = x.std(ddof=1) / np.sqrt(len(x))
    h = stats.t.ppf(1 - (1 - nivel) / 2, len(x) - 1) * se
    return float(x.mean() - h), float(x.mean() + h)


def p_t(x):
    return float(stats.ttest_1samp(np.asarray(x, float), 0.0).pvalue)


def p_w(x):
    r = S.wilcoxon_r(x)
    return r["p"], r["r"]


def tost_t(d, margen=MARGEN):
    d = np.asarray(d, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    p1 = 1 - stats.t.cdf((d.mean() + margen) / se, n - 1)
    p2 = stats.t.cdf((d.mean() - margen) / se, n - 1)
    return float(max(p1, p2))


# ================================================================== ETAPA 0: reproducir

PUB_WILCOXON = {   # tab:wilcoxon: (dif mediana, p, r); p None = "<0.001"
    ("acc", "loso"): [(+0.006, 0.12, 0.21), (+0.006, 0.27, 0.15), (-0.001, 0.52, 0.09),
                      (+0.044, None, 0.71), (+0.028, None, 0.59), (+0.035, None, 0.67)],
    ("acc", "within"): [(+0.019, None, 0.48), (+0.015, 0.29, 0.14), (-0.035, 0.004, 0.40),
                        (+0.065, None, 0.63), (+0.016, None, 0.45), (+0.041, None, 0.73)],
    ("kappa", "loso"): [(+0.013, 0.12, 0.21), (+0.013, 0.28, 0.15), (-0.003, 0.46, 0.10),
                        (+0.088, None, 0.70), (+0.055, None, 0.59), (+0.070, None, 0.67)],
    ("kappa", "within"): [(+0.038, None, 0.48), (+0.030, 0.31, 0.14), (-0.070, 0.004, 0.39),
                          (+0.130, None, 0.63), (+0.033, None, 0.45), (+0.083, None, 0.73)],
}
PUB_DSI_FOLDS = {"eegnet": 33, "conformer": 34, "labram": 38, "eegmamba": 37}


def reproducir() -> bool:
    import verificar_cifras as V
    filas = []

    def add(item, pub, rec, ok):
        filas.append(dict(item=item, publicado=pub, recalculado=rec, ok=bool(ok)))

    # 0. todo el verificador del proyecto
    fallos = 0
    for ident, decl, donde, fn in V.CIFRAS:
        o = float(fn())
        ok = abs(o - decl) <= V.tolerancia(decl)
        fallos += (not ok)
        if not ok:
            add(f"verificar_cifras:{ident}", decl, o, False)
    add("verificar_cifras: cifras declaradas / discrepancias", len(V.CIFRAS), fallos, fallos == 0)

    # 1. la semilla 2026 cargada con avance7_datos coincide con la que usa el verificador
    for arq in ARQ:
        add(f"loso 2026 acc media {arq}", round(V.acc_loso(arq), 6),
            round(mat("loso", arq)[2026].mean(), 6),
            abs(V.acc_loso(arq) - mat("loso", arq)[2026].mean()) < 1e-9)
    for arq in ("labram", "eegmamba"):
        add(f"within 2026 acc media {arq}", round(V.acc_within(arq), 6),
            round(mat("within", arq)[2026].mean(), 6),
            abs(V.acc_within(arq) - mat("within", arq)[2026].mean()) < 1e-9)
    for arq in ("eegnet", "conformer"):
        v = mat("within", arq).mean(axis=1).mean()
        add(f"within media3 acc {arq}", round(V.acc_within(arq), 6), round(v, 6),
            abs(V.acc_within(arq) - v) < 1e-9)

    # 2. tab:wilcoxon completa (semilla 2026; diferencia mediana, p, r)
    for (metrica, proto), pub in PUB_WILCOXON.items():
        M = {a: mat(proto, a, metrica)[2026] for a in ARQ}
        for (a, b), (d_p, p_p, r_p) in zip(PARES, pub):
            d = M[a] - M[b]
            p, r = p_w(d.to_numpy())
            dm = d.median()
            ok = (abs(dm - d_p) <= 0.0005 + 1e-9 and round(r, 2) == r_p and
                  ((p < 0.001) if p_p is None else round(p, 2 if p_p >= 0.01 else 3) == p_p))
            add(f"tab:wilcoxon {metrica} {proto} {a}-{b} (dif mediana, p, r)",
                f"{d_p:+.3f}, {'<0.001' if p_p is None else p_p}, {r_p}",
                f"{dm:+.3f}, {p:.4g}, {r:.2f}", ok)

    # 3. tab:within-loso (EEGNet): publicada con intra-sujeto de 3 semillas y LOSO 2026
    w3 = mat("within", "eegnet").mean(axis=1)
    l26 = mat("loso", "eegnet")[2026]
    dif = l26 - w3
    p, _ = p_w(dif.to_numpy())
    add("tab:within-loso EEGNet intra (3 semillas) / LOSO 2026 / dif",
        "0.634 / 0.763 / +0.128", f"{w3.mean():.3f} / {l26.mean():.3f} / {dif.mean():+.3f}",
        round(w3.mean(), 3) == 0.634 and round(l26.mean(), 3) == 0.763
        and round(dif.mean(), 3) == 0.128)
    add("tab:within-loso EEGNet sujetos que mejoran / p", "48 / 3e-8",
        f"{int((dif > 0).sum())} / {p:.2g}",
        int((dif > 0).sum()) == 48 and abs(np.log10(p) - np.log10(3e-8)) < 0.5)

    # 4. tab:dsi: folds > 0 con semilla 2026
    for arq in ARQ:
        d = tabla("loso", arq)
        d = d[d.seed == 2026]
        n = int((d.acc_visto > d.acc).sum())
        add(f"tab:dsi folds>0 {arq}", PUB_DSI_FOLDS[arq], n, n == PUB_DSI_FOLDS[arq])

    # 5. bajo umbral: la publicacion mezcla 3 semillas (eegnet, conformer) y 1 (labram, eegmamba)
    pub_b = {"eegnet": 41, "conformer": 45, "labram": 37, "eegmamba": 48}
    for arq in ARQ:
        x = mat("within", arq)
        v = x.mean(axis=1) if arq in ("eegnet", "conformer") else x[2026]
        n = int((v <= UMBRAL + 1e-9).sum())
        add(f"bajo<=0.70 intra {arq} (semillas de la publicacion)", pub_b[arq], n, n == pub_b[arq])

    df = pd.DataFrame(filas)
    df.to_csv(OUT / "semillas_reproduccion_2026.csv", index=False)
    mal = df[~df.ok]
    print(f"[reproducir] {len(df)} comprobaciones, {len(mal)} discrepan")
    if len(mal):
        print(mal.to_string())
    return len(mal) == 0


# ================================================================== ETAPA 1: confusor

def stats_vector(x, prefijo):
    x = np.asarray(x, float)
    lo_b, hi_b = boot_media_ic(x)
    lo_t, hi_t = t_ic(x)
    pw, _ = p_w(x)
    return {f"{prefijo}_media": x.mean(), f"{prefijo}_mediana": np.median(x),
            f"{prefijo}_sd": x.std(ddof=1),
            f"{prefijo}_ic95_t_inf": lo_t, f"{prefijo}_ic95_t_sup": hi_t,
            f"{prefijo}_ic95_boot_inf": lo_b, f"{prefijo}_ic95_boot_sup": hi_b,
            f"{prefijo}_p_wilcoxon": pw, f"{prefijo}_p_t": p_t(x)}


def d_cruce(M):
    return (M[2026] - (M[2027] + M[2028]) / 2).to_numpy()


def d_intra(M):
    return (M[2027] - M[2028]).to_numpy()


def confusor():
    verificar_regimen().to_csv(OUT / "confusor_regimenes_por_semilla.csv", index=False)
    filas, did, cota = [], [], []
    D = {(p, a): mat(p, a) for p in ("loso", "within") for a in ARQ}
    for proto in ("loso", "within"):
        for arq in ARQ:
            M = D[(proto, arq)]
            dc, di = d_cruce(M), d_intra(M)
            cambia = regimen_str(proto, arq) not in ("SSS", "CCC")
            fila = dict(protocolo=proto, arq=arq, regimen_2026_2027_2028=regimen_str(proto, arq),
                        hay_cambio_de_regimen=cambia, n=len(dc))
            fila.update(stats_vector(dc, "cruce"))
            fila.update(stats_vector(di, "intra"))
            for s in (2027, 2028):      # las otras dos diferencias "una semilla contra el resto"
                otros = [t for t in SEEDS if t != s]
                fila[f"loo{s}_media"] = float((M[s] - M[otros].mean(axis=1)).mean())
            # Bajo H0 (sin efecto de regimen; ruido de semilla homogeneo, varianza s2):
            # Var(d_intra) = 2 s2, Var(d_cruce) = 1.5 s2, Cov(d_cruce, d_intra) = 0.
            v_c, v_i = dc.var(ddof=1), di.var(ddof=1)
            razon = v_c / (0.75 * v_i)
            n = len(dc)
            pF = 2 * min(stats.f.cdf(razon, n - 1, n - 1), stats.f.sf(razon, n - 1, n - 1))
            idx = rng().integers(0, n, (B, n))
            vc = dc[idx].var(axis=1, ddof=1)
            vi = di[idx].var(axis=1, ddof=1)
            het = np.sqrt(np.maximum(vc - 0.75 * vi, 0.0))
            fila.update(sd_cruce_esperada_por_ruido=float(np.sqrt(0.75 * v_i)),
                        razon_var_cruce_sobre_esperada=float(razon), p_F_bilateral=float(pF),
                        razon_ic95_boot_inf=float(np.quantile(vc / (0.75 * vi), .025)),
                        razon_ic95_boot_sup=float(np.quantile(vc / (0.75 * vi), .975)),
                        sd_heterogeneidad_efecto=float(np.sqrt(max(v_c - 0.75 * v_i, 0.0))),
                        sd_heterogeneidad_ic95_sup=float(np.quantile(het, .975)),
                        sd_ruido_semilla_sujeto=float(di.std(ddof=1) / np.sqrt(2)))
            # diferencias entre pares de semillas (si el ruido es homogeneo, las tres tienen la misma sd)
            for a_, b_ in ((2026, 2027), (2026, 2028), (2027, 2028)):
                dpar = (M[a_] - M[b_]).to_numpy()
                fila[f"par{a_}_{b_}_media"] = float(dpar.mean())
                fila[f"par{a_}_{b_}_sd"] = float(dpar.std(ddof=1))
            filas.append(fila)
    Tdf = pd.DataFrame(filas)
    Tdf.to_csv(OUT / "confusor_tabla.csv", index=False)
    T = Tdf.set_index(["protocolo", "arq"])

    # diferencia de diferencias frente al control negativo EEGMamba (mismo calculo, sin cambio de regimen)
    for proto in ("loso", "within"):
        ctrl = d_cruce(D[(proto, "eegmamba")])
        for arq in ("eegnet", "conformer", "labram"):
            dd = d_cruce(D[(proto, arq)]) - ctrl
            lo, hi = boot_media_ic(dd)
            lo_t, hi_t = t_ic(dd)
            pw, _ = p_w(dd)
            did.append(dict(protocolo=proto, arq=arq,
                            hay_cambio_de_regimen=bool(T.loc[(proto, arq), "hay_cambio_de_regimen"]),
                            control="eegmamba", media_dd=dd.mean(), sd_dd=dd.std(ddof=1),
                            ic95_boot_inf=lo, ic95_boot_sup=hi, ic95_t_inf=lo_t, ic95_t_sup=hi_t,
                            p_wilcoxon=pw, p_t=p_t(dd)))
    Ddf = pd.DataFrame(did)
    Ddf.to_csv(OUT / "confusor_vs_control.csv", index=False)
    Dd = Ddf.set_index(["protocolo", "arq"])

    # cota: delta = exactitud(sin recorte) - exactitud(con recorte); cota = max(|inf|, |sup|) del IC 95 %
    for proto in ("loso", "within"):
        for arq in ARQ:
            t = T.loc[(proto, arq)]
            if not t.hay_cambio_de_regimen:
                continue
            fila = dict(protocolo=proto, arq=arq, delta_bruto=t.cruce_media,
                        bruto_ic95_inf=t.cruce_ic95_boot_inf, bruto_ic95_sup=t.cruce_ic95_boot_sup,
                        cota_bruta=max(abs(t.cruce_ic95_boot_inf), abs(t.cruce_ic95_boot_sup)))
            if (proto, arq) in Dd.index:
                dd = Dd.loc[(proto, arq)]
                fila.update(delta_neto=dd.media_dd, neto_ic95_inf=dd.ic95_boot_inf,
                            neto_ic95_sup=dd.ic95_boot_sup,
                            cota_neta=max(abs(dd.ic95_boot_inf), abs(dd.ic95_boot_sup)))
            fila["contaminacion_media3_cota_bruta"] = fila["cota_bruta"] / 3
            fila["contaminacion_media3_cota_neta"] = fila.get("cota_neta", np.nan) / 3
            cota.append(fila)
    pd.DataFrame(cota).to_csv(OUT / "confusor_cota.csv", index=False)
    print("[confusor] escrito confusor_tabla, confusor_vs_control, confusor_cota, regimenes")
    return Tdf, Ddf, pd.DataFrame(cota)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa", default="todo",
                    choices=["todo", "reproducir", "confusor", "contrastes", "derivados"])
    ap.add_argument("--continuar", action="store_true",
                    help="seguir aunque la reproduccion de lo publicado falle (necesario tras "
                         "redondear las diferencias a 1e-9 en Wilcoxon: cambian 7 celdas de "
                         "tab:wilcoxon; ver 07_VERIFICACION.md)")
    a = ap.parse_args()
    if not reproducir() and not a.continuar:
        raise SystemExit("La reproduccion de lo publicado fallo: no se calcula nada nuevo.")
    if a.etapa in ("todo", "confusor"):
        confusor()
    if a.etapa in ("todo", "contrastes"):
        import avance7_semillas_contrastes as C
        C.main()
    if a.etapa in ("todo", "derivados"):
        import avance7_semillas_derivados as Dv
        Dv.main()


if __name__ == "__main__":
    main()
