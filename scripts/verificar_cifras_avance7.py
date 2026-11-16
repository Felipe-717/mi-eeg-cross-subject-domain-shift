#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""verificar_cifras_avance7.py: recalcula las cifras clave de los informes 03, 04 y 05 del avance 7.

POR QUE EXISTE
    Es la extension de `verificar_cifras.py` para el avance 7 (mismo patron: una cifra
    declarada junto a la funcion que la recalcula desde los CSV). El objetivo es que NINGUNA
    cifra de los informes `avance_7/analisis/03_FEWSHOT_TRANSFERENCIA.md`,
    `04_SEMILLAS_Y_CONFUSOR.md` y `05_ABLACIONES_EA_CR.md` quede sin un recalculo
    independiente.

INDEPENDENCIA DEL RECALCULO
    Se parte de `scripts/avance7_datos.py` (solo carga de CSV crudos: `within`, `loso`,
    `ablacion`, `fewshot`, `transferencia`) y se reimplementan AQUI las estadisticas
    (Wilcoxon zsplit, r, Holm, TOST, bootstrap, IDS, IDP, cota del confusor). No se importa
    `avance7_stats.py` ni se lee ningun CSV de `results/avance_7/analisis/`: todas las cifras
    salen de los CSV crudos (campo `origen` = "crudo"). Si una cifra solo pudiera salir del CSV
    de analisis, se marcaria con origen "csv_analisis"; hoy no hay ninguna.

COMO SE USA
    ~/anaconda3/envs/eeg/python.exe scripts/verificar_cifras_avance7.py             # solo fallos
    ~/anaconda3/envs/eeg/python.exe scripts/verificar_cifras_avance7.py --verboso   # todas
    ~/anaconda3/envs/eeg/python.exe scripts/verificar_cifras_avance7.py --bloque t4 # un bloque
    ~/anaconda3/envs/eeg/python.exe scripts/verificar_cifras.py --avance7           # ambos

    Codigo de salida 1 si alguna cifra DISCREPA.

ESTADO DE LO DECLARADO
    Todos los valores declarados son PROVISIONALES: se copiaron de las tablas de los informes
    (generados una vez por `DECLARADOS`, mas abajo) y llevan `donde="pendiente"`. Cuando el
    texto de `main.tex` este redactado hay que sustituir `pendiente` por la etiqueta real
    (tab:..., sec:...) con `fijar_donde(prefijo, etiqueta)` o editando `DONDE`.

CONVENCION DE IDENTIFICADORES
    <bloque>.<...>.<campo>; el bloque (primer segmento) elige la funcion de calculo de
    BLOQUES. Un identificador declarado que el calculo no produce se lista aparte como
    "sin recalculo" (no se oculta).

TOLERANCIA
    Media unidad de la ultima cifra DECLARADA en el informe (se guarda con cada valor). Para
    recuentos enteros, 0.5.

CIFRAS QUE NO SE VERIFICAN (y por que)
    - IC jerarquico de 04 (T6, T10 columna "IC95 jerarquico"): el orden de las extracciones
      del generador no esta fijado por el informe; reimplementarlo seria adivinar.
    - 03 sec. 2.6 (nivel inicial, exploratorio) y 05 T5 (concentracion del beneficio):
      exploratorios y sin cifra clave citada en las conclusiones.
    - Cifras de `p < 0.001` ("<0.001") : no son numericas en el informe.
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import avance7_datos as A  # noqa: E402  (solo carga de CSV crudos)
from ruido_flotante import wilcoxon_pareado  # Wilcoxon sobre diferencias redondeadas a 1e-9

try:  # misma tolerancia para el umbral de bajo rendimiento que verificar_cifras.py
    from verificar_cifras import le_umbral, UMBRAL_BAJO_RENDIMIENTO  # noqa: E402
except Exception:  # pragma: no cover
    UMBRAL_BAJO_RENDIMIENTO = 0.70

    def le_umbral(serie):
        return pd.Series(serie).round(9) <= UMBRAL_BAJO_RENDIMIENTO

ARQ = ["eegnet", "conformer", "labram", "eegmamba"]
NOMBRE = {"eegnet": "EEGNet", "conformer": "Conformer", "labram": "LaBraM", "eegmamba": "EEGMamba"}
SEM = (2026, 2027, 2028)
PARES = [("eegnet", "conformer"), ("eegnet", "labram"), ("conformer", "labram"),
         ("eegnet", "eegmamba"), ("conformer", "eegmamba"), ("labram", "eegmamba")]
B = 10_000
SEMILLA_BOOT = 2026
MARGEN = 0.03
KS = (10, 20)
METODOS = ("stats", "cabeza", "stats+cabeza")


# ------------------------------------------------------------------ estadistica propia

def w_zsplit(d):
    """Wilcoxon bilateral zsplit; r = z / sqrt(n) con z desde el p bilateral."""
    d = np.asarray(d, float)
    p = float(wilcoxon_pareado(d, zero_method="zsplit", alternative="two-sided").pvalue)
    z = abs(stats.norm.ppf(p / 2)) if p > 0 else abs(stats.norm.ppf(1e-16))
    return p, float(z / np.sqrt(len(d)))


def w_pareado(x, y):
    """Wilcoxon pareado por defecto de scipy (el de verificar_cifras.p_wilcoxon_ids)."""
    p = float(wilcoxon_pareado(x, y).pvalue)
    return p, float(stats.norm.isf(p / 2) / np.sqrt(len(x)))


def holm(p):
    p = np.asarray(p, float)
    orden = np.argsort(p)
    aj = np.empty(len(p))
    previo = 0.0
    for rango, i in enumerate(orden):
        previo = max(previo, (len(p) - rango) * p[i])
        aj[i] = min(previo, 1.0)
    return aj


def tost(d, margen=MARGEN):
    d = np.asarray(d, float)
    p1 = wilcoxon_pareado(d + margen, zero_method="zsplit", alternative="greater").pvalue
    p2 = wilcoxon_pareado(d - margen, zero_method="zsplit", alternative="less").pvalue
    return float(max(p1, p2))


def boot(d, qs=(0.025, 0.975)):
    """Cuantiles de la media remuestreando sujetos; generador 2026 nuevo en cada llamada."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(SEMILLA_BOOT)
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
    return tuple(float(x) for x in np.quantile(m, qs))


def contraste(var: pd.Series, ref: pd.Series) -> dict:
    """var menos ref por sujeto: medias, dif, IC95, IC90, p, r, mejoran/empeoran."""
    var, ref = var.sort_index(), ref.sort_index()
    assert (var.index == ref.index).all()
    d = (var - ref).to_numpy()
    rng = np.random.default_rng(SEMILLA_BOOT)
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(axis=1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    lo9, hi9 = np.quantile(m, [0.05, 0.95])
    p, r = w_zsplit(d)
    return dict(ref=ref.mean(), var=var.mean(), dif=d.mean(), dif_med=float(np.median(d)),
                ic_inf=float(lo), ic_sup=float(hi), ic90_inf=float(lo9), ic90_sup=float(hi9),
                mejoran=int((d > 0).sum()), empeoran=int((d < 0).sum()), p=p, r=r,
                margen=float(max(abs(lo9), abs(hi9))), d=d)


# ------------------------------------------------------------------ series por sujeto

@lru_cache(None)
def within_sub(arq: str) -> pd.DataFrame:
    """sub x semilla: exactitud media de las particiones intra-sujeto."""
    return A.within(arq).groupby(["sub", "seed"]).acc.mean().unstack().sort_index()


@lru_cache(None)
def within_kappa(arq: str) -> pd.DataFrame:
    return A.within(arq).groupby(["sub", "seed"]).kappa.mean().unstack().sort_index()


@lru_cache(None)
def loso_sub(arq: str, col: str = "acc", variante: str = "base") -> pd.DataFrame:
    """test_sub x semilla."""
    return A.loso(arq, variante).groupby(["test_sub", "seed"])[col].mean().unstack().sort_index()


def sub_prot(prot: str, arq: str, col: str = "acc") -> pd.DataFrame:
    if prot == "within":
        return within_sub(arq) if col == "acc" else within_kappa(arq)
    return loso_sub(arq, col)


@lru_cache(None)
def abl_sub(variante: str, arq: str) -> pd.Series:
    return A.ablacion(variante, arq).groupby("sub").acc.mean().sort_index()


def r3(x):
    return float(x)


# ================================================================== BLOQUE t5: exactitud por semilla (04 T5)

def b_t5() -> dict:
    out = {}
    for prot in ("loso", "within"):
        for m in ARQ:
            for col in ("acc", "kappa"):
                t = sub_prot(prot, m, col)
                por_sem = t.mean()                       # media sobre 54 sujetos, por semilla
                for s in SEM:
                    out[f"t5.{prot}.{m}.{col}.s{s}"] = por_sem[s]
                out[f"t5.{prot}.{m}.{col}.media"] = t.mean(axis=1).mean()
                out[f"t5.{prot}.{m}.{col}.sd_semillas"] = float(por_sem.std(ddof=1))
                if col == "acc":
                    out[f"t5.{prot}.{m}.acc.sd_sujetos"] = float(t.mean(axis=1).std())
                    out[f"t5.{prot}.{m}.acc.sd_dentro"] = float(t.std(axis=1, ddof=1).mean())
    return out


# ================================================================== BLOQUES t1..t4: confusor (04 sec. 1)

def _dcruce(prot, m):
    t = sub_prot(prot, m)
    return (t[2026] - t[[2027, 2028]].mean(axis=1)), (t[2027] - t[2028])


def b_confusor() -> dict:
    out = {}
    dc = {(p, m): _dcruce(p, m)[0] for p in ("loso", "within") for m in ARQ}
    for prot in ("loso", "within"):
        for m in ARQ:
            c, i = _dcruce(prot, m)
            lo, hi = boot(c)
            pw, _ = w_zsplit(c.to_numpy())
            pi, _ = w_zsplit(i.to_numpy())
            k = f"t1.{prot}.{m}"
            out.update({f"{k}.dcruce_media": c.mean(), f"{k}.dcruce_ic_inf": lo,
                        f"{k}.dcruce_ic_sup": hi, f"{k}.dcruce_mediana": float(np.median(c)),
                        f"{k}.dcruce_sd": float(c.std()), f"{k}.dcruce_p": pw,
                        f"{k}.dintra_media": i.mean(), f"{k}.dintra_sd": float(i.std()),
                        f"{k}.dintra_p": pi})
            k2 = f"t2.{prot}.{m}"
            esperada = np.sqrt(0.75) * i.std()
            out[f"{k2}.sd_esperada"] = float(esperada)
            out[f"{k2}.razon"] = float(c.var() / (0.75 * i.var()))
            # diferencia de diferencias frente al control EEGMamba (T3)
            if m != "eegmamba":
                dd = c - dc[(prot, "eegmamba")]
                dlo, dhi = boot(dd)
                pw3, _ = w_zsplit(dd.to_numpy())
                k3 = f"t3.{prot}.{m}"
                out.update({f"{k3}.media": dd.mean(), f"{k3}.ic_inf": dlo,
                            f"{k3}.ic_sup": dhi, f"{k3}.p": pw3})
                # T4: cotas
                bruta = max(abs(lo), abs(hi))
                neta = max(abs(dlo), abs(dhi))
                uso = max(bruta, neta)
                k4 = f"t4.{prot}.{m}"
                out.update({f"{k4}.delta_bruto": c.mean(), f"{k4}.cota_bruta": bruta,
                            f"{k4}.delta_neto": dd.mean(), f"{k4}.cota_neta": neta,
                            f"{k4}.cota_uso": uso, f"{k4}.cota_media3": uso / 3})
    return out


# ================================================================== BLOQUE t6/t7: contrastes entre arquitecturas (04 sec. 2)

def b_t6() -> dict:
    out = {}
    for prot in ("loso", "within"):
        # variantes: serie por sujeto de cada arquitectura
        ser = {"s2026": {m: sub_prot(prot, m)[2026] for m in ARQ},
               "s2027": {m: sub_prot(prot, m)[2027] for m in ARQ},
               "s2028": {m: sub_prot(prot, m)[2028] for m in ARQ},
               "media3": {m: sub_prot(prot, m).mean(axis=1) for m in ARQ}}
        res = {}
        for var, ss in ser.items():
            filas = []
            for a, b in PARES:
                d = (ss[a] - ss[b]).to_numpy()
                p, r = w_zsplit(d)
                lo9, hi9 = boot(d, (0.05, 0.95))
                filas.append(dict(a=a, b=b, dif=d.mean(), med=float(np.median(d)), p=p, r=r,
                                  ic90=(lo9, hi9), tost=tost(d)))
            ph = holm([f["p"] for f in filas])
            th = holm([f["tost"] for f in filas])
            for f, x, y in zip(filas, ph, th):
                f["p_holm"], f["tost_holm"] = float(x), float(y)
            res[var] = filas
        for i, (a, b) in enumerate(PARES):
            k = f"t6.{prot}.{a}-{b}"
            for var in ("s2026", "media3"):
                f = res[var][i]
                out.update({f"{k}.{var}.dif": f["dif"], f"{k}.{var}.mediana": f["med"],
                            f"{k}.{var}.p": f["p"], f"{k}.{var}.r": f["r"],
                            f"{k}.{var}.p_holm": f["p_holm"],
                            f"t7.{prot}.{a}-{b}.{var}.ic90_inf": f["ic90"][0],
                            f"t7.{prot}.{a}-{b}.{var}.ic90_sup": f["ic90"][1],
                            f"t7.{prot}.{a}-{b}.{var}.tost_holm": f["tost_holm"]})
            difs = [res[s][i]["dif"] for s in ("s2026", "s2027", "s2028")]
            for s in ("s2026", "s2027", "s2028"):
                out[f"{k}.dif_{s}"] = res[s][i]["dif"]
                out[f"{k}.p_{s}"] = res[s][i]["p"]
            out[f"{k}.sd_semillas_dif"] = float(np.std(difs, ddof=1))
            # solo 2027-2028 (IC90)
            d2 = (0.5 * (ser["s2027"][a] + ser["s2028"][a]) - 0.5 * (ser["s2027"][b] + ser["s2028"][b])).to_numpy()
            lo9, hi9 = boot(d2, (0.05, 0.95))
            out[f"t7.{prot}.{a}-{b}.m2728.ic90_inf"] = lo9
            out[f"t7.{prot}.{a}-{b}.m2728.ic90_sup"] = hi9
    return out


# ================================================================== BLOQUE t9: caida intra -> LOSO (04 T9)

def b_t9() -> dict:
    out = {}
    for m in ARQ:
        wi, lo = within_sub(m), loso_sub(m)
        previo = A.within(m)
        previo = previo[previo.regimen == "previo"].groupby("sub").acc.mean().sort_index()
        casos = {f"s{s}": (wi[s], lo[s]) for s in SEM}
        casos["media3"] = (wi.mean(axis=1), lo.mean(axis=1))
        casos["publicada"] = (previo, lo[2026])
        for nom, (i, l) in casos.items():
            d = (l - i).to_numpy()
            p, r = w_zsplit(d)
            k = f"t9.{m}.{nom}"
            out.update({f"{k}.intra": i.mean(), f"{k}.loso": l.mean(), f"{k}.dif": d.mean(),
                        f"{k}.mediana": float(np.median(d)), f"{k}.mejoran": int((d > 0).sum()),
                        f"{k}.r": r, f"{k}.p_lt001": float(p < 0.001)})
            lo_, hi_ = boot(d)
            out[f"{k}.ic_inf"], out[f"{k}.ic_sup"] = lo_, hi_
    return out


# ================================================================== BLOQUE t10: IDS por semilla y con 3 semillas (04 T10)

def b_t10() -> dict:
    out = {}
    casos = {}
    for s in SEM:
        casos[f"s{s}"] = {m: (lambda d: d[d.seed == s].sort_values("test_sub"))(A.loso(m, "base")) for m in ARQ}
    media = {}
    for m in ARQ:
        d = A.loso(m, "base")
        media[m] = d.groupby("test_sub")[["acc", "acc_visto", "kappa", "kappa_visto"]].mean().sort_index()
    casos["media3"] = media
    for nom, dd in casos.items():
        ps = {}
        for m in ARQ:
            d = dd[m]
            v, n = d.acc_visto.mean(), d.acc.mean()
            kv, kn = d.kappa_visto.mean(), d.kappa.mean()
            p, r = w_pareado(d.acc_visto.to_numpy(), d.acc.to_numpy())
            ps[m] = p
            k = f"t10.{m}.{nom}"
            out.update({f"{k}.vistos": v, f"{k}.novistos": n, f"{k}.ids_acc": (v - n) / v,
                        f"{k}.ids_kappa": (kv - kn) / kv, f"{k}.folds": int((d.acc_visto > d.acc).sum()),
                        f"{k}.p": p, f"{k}.r": r})
        ph = holm([ps[m] for m in ARQ])
        for m, x in zip(ARQ, ph):
            out[f"t10.{m}.{nom}.p_holm"] = float(x)
    # sd entre semillas del IDS
    for m in ARQ:
        v = [out[f"t10.{m}.s{s}.ids_acc"] for s in SEM]
        out[f"t10.{m}.sd_ids_semillas"] = float(np.std(v, ddof=1))
    return out


# ================================================================== BLOQUE tr: transferencia por semilla (03 parte B) y t11

@lru_cache(None)
def transf_escalares(variante: str):
    """Para cada (arq, semilla): cantidades escalares de la descomposicion. Todo desde crudo."""
    D = {}
    for m in ARQ:
        cz = A.transferencia("cruzado", m, variante)
        ho = A.transferencia("h44_openbmi", m, variante)
        hw = A.transferencia("h44_wbcic", m, variante)
        nat = A.transferencia("nativo", m, variante)
        h44 = A.transferencia("h44", m, variante)
        t54 = A.transferencia("train54", m, variante)
        for s in sorted(cz.seed.unique()):
            e = {}
            for pol in ("source", "per_trial", "per_subject"):
                e[f"trans_{pol}"] = cz[(cz.seed == s) & (cz.norm_destino == pol)].groupby("sub").acc.mean().sort_index()
            e["persona_serie"] = ho[(ho.seed == s) & (ho.norm_destino == "source")].groupby("sub").acc.mean().sort_index()
            e["wbcic_serie"] = hw[(hw.seed == s) & (hw.norm_destino == "source")].groupby("sub").acc.mean().sort_index()
            e["nativo"] = float(nat[nat.seed == s].acc_reservado.iloc[0])
            e["vistos44"] = float(h44[h44.seed == s].acc.iloc[0])
            e["vistos54"] = float(t54[t54.seed == s].acc.iloc[0])
            D[(m, int(s))] = e
    return D


def cantidades(e: dict) -> dict:
    q = {"transferido": e["trans_source"].mean(), "nativo": e["nativo"],
         "vistos44": e["vistos44"], "persona": e["persona_serie"].mean(),
         "wbcic44": e["wbcic_serie"].mean()}
    q["dif_nativo_menos_transferido"] = q["nativo"] - q["transferido"]
    q["caida_persona"] = q["vistos44"] - q["persona"]
    q["caida_protocolo"] = q["persona"] - q["wbcic44"]
    q["protocolo_menos_persona"] = q["caida_protocolo"] - q["caida_persona"]
    q["idp"] = q["caida_protocolo"] / q["persona"]
    return q


CANT = ["transferido", "nativo", "dif_nativo_menos_transferido", "vistos44", "persona", "wbcic44",
        "caida_persona", "caida_protocolo", "protocolo_menos_persona", "idp"]


def b_tr() -> dict:
    out = {}
    D = transf_escalares("base")
    for m in ARQ:
        qs = {s: cantidades(D[(m, s)]) for s in SEM}
        for c in CANT:
            v = np.array([qs[s][c] for s in SEM])
            for s, x in zip(SEM, v):
                out[f"tr.{c}.{m}.s{s}"] = x
            # IDP con 3 semillas: la formula de main.tex (eq. idp) es (m_nuevo - m_protocolo) / m_nuevo
            # sobre MEDIAS, y verificar_cifras.idp la aplica asi a la semilla 2026. Con 3 semillas se
            # aplica a las medias de las 3 semillas = cociente de medias (caida por protocolo media /
            # persona nueva media). El promedio de los 3 IDP por semilla se guarda aparte.
            if c == "idp":
                out[f"tr.{c}.{m}.media"] = (np.mean([qs[s]["caida_protocolo"] for s in SEM])
                                            / np.mean([qs[s]["persona"] for s in SEM]))
                out[f"tr.{c}.{m}.media_de_idp"] = v.mean()
            else:
                out[f"tr.{c}.{m}.media"] = v.mean()
            out[f"tr.{c}.{m}.dt"] = float(v.std(ddof=1))
            out[f"tr.{c}.{m}.min"], out[f"tr.{c}.{m}.max"] = v.min(), v.max()
        # T11 (04): persona nueva, WBCIC, IDP por semilla
        for s in SEM:
            out[f"t11.{m}.s{s}.persona"] = qs[s]["persona"]
            out[f"t11.{m}.s{s}.wbcic"] = qs[s]["wbcic44"]
            out[f"t11.{m}.s{s}.idp"] = qs[s]["idp"]
        out[f"t11.{m}.media3.persona"] = np.mean([qs[s]["persona"] for s in SEM])
        out[f"t11.{m}.media3.wbcic"] = np.mean([qs[s]["wbcic44"] for s in SEM])
        # IDP de la media de 3 semillas = cociente de medias (definicion de 04 T11 y del texto de 03);
        # la tabla 3.3 de 03 ("tr.idp...media") promedia los tres IDP por semilla. Difieren en el
        # tercer decimal en EEGMamba (0.1365 frente a 0.1366): hallazgo H1 de 07_VERIFICACION.md.
        out[f"t11.{m}.media3.idp"] = (np.mean([qs[s]["caida_protocolo"] for s in SEM])
                                      / np.mean([qs[s]["persona"] for s in SEM]))
        out[f"t11.{m}.media3.idp_media_de_idp"] = np.mean([qs[s]["idp"] for s in SEM])
        out[f"t11.{m}.media3.sd"] = float(np.std([qs[s]["idp"] for s in SEM], ddof=1))
        # cociente IDP/IDS (04 sec. 3.3)
        ids3 = b_t10_cache()[f"t10.{m}.media3.ids_acc"]
        out[f"t11.{m}.cociente_idp_ids"] = out[f"t11.{m}.media3.idp"] / ids3
    # politicas de normalizacion (03 3.5, familia F-B1 de 8)
    filas = []
    for m in ARQ:
        ser = {pol: pd.concat([D[(m, s)][f"trans_{pol}"] for s in SEM], axis=1).mean(axis=1)
               for pol in ("source", "per_trial", "per_subject")}
        out[f"pol.{m}.source"], out[f"pol.{m}.per_trial"], out[f"pol.{m}.per_subject"] = (
            ser["source"].mean(), ser["per_trial"].mean(), ser["per_subject"].mean())
        out[f"pol.{m}.amplitud"] = max(ser[p].mean() for p in ser) - min(ser[p].mean() for p in ser)
        for pol in ("per_trial", "per_subject"):
            c = contraste(ser["source"], ser[pol])
            filas.append((m, pol, c))
    ph = holm([c["p"] for _, _, c in filas])
    for (m, pol, c), x in zip(filas, ph):
        k = f"polc.{m}.{pol}"
        out.update({f"{k}.dif": c["dif"], f"{k}.ic_inf": c["ic_inf"], f"{k}.ic_sup": c["ic_sup"],
                    f"{k}.p": c["p"], f"{k}.p_holm": float(x), f"{k}.r": c["r"]})
    # IC de las caidas (03 3.4)
    def mats(variante, semillas, m):
        Dv = transf_escalares(variante)
        P = np.stack([Dv[(m, s)]["persona_serie"].to_numpy() for s in semillas])
        W = np.stack([Dv[(m, s)]["wbcic_serie"].to_numpy() for s in semillas])
        V = np.array([Dv[(m, s)]["vistos44"] for s in semillas])
        return P, W, V
    for m in ARQ:
        for nom, sems, jer in (("s2026", (2026,), False), ("sujetos3", SEM, False),
                               ("jer3", SEM, True), ("s2027", (2027,), False), ("s2028", (2028,), False)):
            P, W, V = mats("base", sems, m)
            ic = boot_caidas(P, W, V, jer)
            for q, (lo, hi) in ic.items():
                out[f"trc.{q}.{m}.{nom}.inf"], out[f"trc.{q}.{m}.{nom}.sup"] = lo, hi
    return out


_T10 = None


def b_t10_cache():
    global _T10
    if _T10 is None:
        _T10 = b_t10()
    return _T10


def boot_caidas(P, W, V, remuestrear_semillas, P2=None, W2=None, V2=None):
    """Bootstrap de las caidas: se remuestrean por separado los 10 sujetos apartados y los 51
    de WBCIC (vistos fijo por semilla); con `remuestrear_semillas` tambien las semillas.
    Orden de extraccion: sujetos de apartados, sujetos de WBCIC, semillas (generador 2026)."""
    rng = np.random.default_rng(SEMILLA_BOOT)
    S_ = P.shape[0]
    pidx = rng.integers(0, P.shape[1], (B, P.shape[1]))
    widx = rng.integers(0, W.shape[1], (B, W.shape[1]))
    sidx = rng.integers(0, S_, (B, S_)) if remuestrear_semillas else np.tile(np.arange(S_), (B, 1))

    def calc(P, W, V):
        p = P[sidx[:, :, None], pidx[:, None, :]].mean(axis=(1, 2))
        w = W[sidx[:, :, None], widx[:, None, :]].mean(axis=(1, 2))
        v = V[sidx].mean(axis=1)
        return v, p, w

    v, p, w = calc(P, W, V)
    cp, cw = v - p, p - w
    res = {"caida_persona": cp, "caida_protocolo": cw, "protocolo_menos_persona": cw - cp, "idp": cw / p}
    if P2 is not None:
        v2, p2, w2 = calc(P2, W2, V2)
        cp2, cw2 = v2 - p2, p2 - w2
        res.update({"ea_caida_persona": cp2 - cp, "ea_caida_protocolo": cw2 - cw,
                    "ea_protocolo_menos_persona": (cw2 - cp2) - (cw - cp),
                    "ea_wbcic44": w2 - w, "ea_persona": p2 - p})
    return {k: (float(np.quantile(x, .025)), float(np.quantile(x, .975))) for k, x in res.items()}


# ================================================================== BLOQUE eat: alineamiento en transferencia (03 parte C)

def b_eat() -> dict:
    out = {}
    Db, De = transf_escalares("base"), transf_escalares("ea")
    filas = []
    for m in ARQ:
        qb, qe = cantidades(Db[(m, 2026)]), cantidades(De[(m, 2026)])
        vs = {c: np.array([cantidades(Db[(m, s)])[c] for s in SEM]) for c in CANT}
        for c in CANT:
            dt = float(vs[c].std(ddof=1))
            k = f"eat.{c}.{m}"
            out.update({f"{k}.base": qb[c], f"{k}.ea": qe[c], f"{k}.dif": qe[c] - qb[c],
                        f"{k}.dt_base": dt, f"{k}.cociente": (qe[c] - qb[c]) / dt,
                        f"{k}.fuera": float(qe[c] < vs[c].min() or qe[c] > vs[c].max())})
        for ev, kb in (("cruzado_51", "trans_source"), ("h44_wbcic_51", "wbcic_serie"),
                       ("h44_openbmi_10", "persona_serie")):
            c = contraste(De[(m, 2026)][kb], Db[(m, 2026)][kb])
            filas.append((m, ev, c))
        # IC pareados de las caidas
        Pb, Wb, Vb = [np.stack([Db[(m, 2026)][k] if not isinstance(Db[(m, 2026)][k], pd.Series)
                                else Db[(m, 2026)][k].to_numpy()]) for k in ("persona_serie", "wbcic_serie")] + \
            [np.array([Db[(m, 2026)]["vistos44"]])]
        Pe, We, Ve = [np.stack([De[(m, 2026)][k].to_numpy()]) for k in ("persona_serie", "wbcic_serie")] + \
            [np.array([De[(m, 2026)]["vistos44"]])]
        ic = boot_caidas(Pb, Wb, Vb, False, Pe, We, Ve)
        for q, base_q in (("ea_caida_persona", "caida_persona"), ("ea_caida_protocolo", "caida_protocolo"),
                          ("ea_protocolo_menos_persona", "protocolo_menos_persona"),
                          ("ea_persona", "persona"), ("ea_wbcic44", "wbcic44")):
            out[f"eati.{m}.{base_q}.est"] = qe[base_q] - qb[base_q]
            out[f"eati.{m}.{base_q}.inf"], out[f"eati.{m}.{base_q}.sup"] = ic[q]
        out[f"eatf.{m}.caida_protocolo"] = (qb["caida_protocolo"] - qe["caida_protocolo"]) / qb["caida_protocolo"]
        out[f"eatf.{m}.brecha"] = (qb["dif_nativo_menos_transferido"] - qe["dif_nativo_menos_transferido"]) \
            / qb["dif_nativo_menos_transferido"]
    ph = holm([c["p"] for _, _, c in filas])
    for (m, ev, c), x in zip(filas, ph):
        k = f"eatc.{m}.{ev}"
        out.update({f"{k}.base": c["ref"], f"{k}.ea": c["var"], f"{k}.dif": c["dif"],
                    f"{k}.ic_inf": c["ic_inf"], f"{k}.ic_sup": c["ic_sup"], f"{k}.p": c["p"],
                    f"{k}.p_holm": float(x), f"{k}.r": c["r"], f"{k}.mejoran": c["mejoran"],
                    f"{k}.empeoran": c["empeoran"]})
    return out


# ================================================================== BLOQUE fs: few-shot (03 parte A)

@lru_cache(None)
def ganancias_fs(variante: str) -> pd.DataFrame:
    """Una fila por (arq, sub, seed, k, metodo): ganancia media entre los 5 sorteos."""
    filas = []
    for m in ARQ:
        d = A.fewshot(m, variante)
        ancho = d.pivot_table(index=["test_sub", "seed", "k", "rep"], columns="metodo", values="acc")
        for me in METODOS:
            g = (ancho[me] - ancho["cero"]).rename("gan").reset_index()
            g["cero"] = ancho["cero"].to_numpy()
            g["metodo"], g["arq"] = me, m
            filas.append(g)
    t = pd.concat(filas)
    return t.groupby(["arq", "test_sub", "seed", "k", "metodo"]).agg(gan=("gan", "mean"),
                                                                     cero=("cero", "mean")).reset_index()


def b_fs() -> dict:
    out = {}
    for var, pref in (("base", "fs_base"), ("ea", "fs_ea")):
        g = ganancias_fs(var)
        filas = []
        for m in ARQ:
            for k in KS:
                for me in METODOS:
                    x = g[(g.arq == m) & (g.k == k) & (g.metodo == me)]
                    s = x.groupby("test_sub").agg(gan=("gan", "mean"), cero=("cero", "mean")).sort_index()
                    d = s.gan.to_numpy()
                    p, r = w_zsplit(d)
                    lo, hi = boot(d)
                    filas.append((m, k, me, dict(cero=s.cero.mean(), gan=d.mean(), ic_inf=lo, ic_sup=hi,
                                                 p=p, r=r, mejoran=int((d > 0).sum()),
                                                 empeoran=int((d < 0).sum()), tost=tost(d), d=d)))
        ph = holm([f["p"] for *_, f in filas])
        th = holm([f["tost"] for *_, f in filas])
        for (m, k, me, f), x, y in zip(filas, ph, th):
            kk = f"{pref}.{m}.{k}.{me}"
            for c in ("cero", "gan", "ic_inf", "ic_sup", "p", "r", "mejoran", "empeoran"):
                out[f"{kk}.{c}"] = f[c]
            out[f"{kk}.p_holm"] = float(x)
            out[f"{kk}.equiv"] = float(y < 0.05)
        if var == "base":
            # por semilla (03 2.4)
            for m in ARQ:
                for k in KS:
                    for me in METODOS:
                        x = g[(g.arq == m) & (g.k == k) & (g.metodo == me)]
                        por = {s: x[x.seed == s].set_index("test_sub").gan.sort_index() for s in (2027, 2028)}
                        kk = f"fs_sem.{m}.{k}.{me}"
                        for s, ser in por.items():
                            out[f"{kk}.gan{s}"] = ser.mean()
                            out[f"{kk}.p{s}"] = w_zsplit(ser.to_numpy())[0]
                        out[f"{kk}.rho"] = float(stats.spearmanr(por[2027], por[2028])[0])
            # medias de las cuatro arquitecturas por metodo y k (03 2.3)
            for k in KS:
                for me in METODOS:
                    out[f"fs_media4.base.{k}.{me}"] = np.mean([out[f"fs_base.{m}.{k}.{me}.gan"] for m in ARQ])
        else:
            for k in KS:
                for me in METODOS:
                    out[f"fs_media4.ea.{k}.{me}"] = np.mean([out[f"fs_ea.{m}.{k}.{me}.gan"] for m in ARQ])
    return out


# ================================================================== BLOQUES t12..t15: bajo rendimiento (04 T12 a T15)

def b_bajo() -> dict:
    out = {}
    W = {m: within_sub(m) for m in ARQ}
    L = {m: loso_sub(m) for m in ARQ}
    for nom in ("s2026", "s2027", "s2028", "media3"):
        sel = (lambda t: t[int(nom[1:])]) if nom.startswith("s") else (lambda t: t.mean(axis=1))
        for m in ARQ:
            out[f"t12.within.{m}.{nom}"] = int(le_umbral(sel(W[m])).sum())
            out[f"t12.loso.{m}.{nom}"] = int(le_umbral(sel(L[m])).sum())
        # solapamiento intra-sujeto (T13)
        bajo = pd.DataFrame({m: le_umbral(sel(W[m])) for m in ARQ})
        n = bajo.sum(axis=1)
        dist = np.array([1.0])
        for pr in bajo.mean():
            dist = np.convolve(dist, [1 - pr, pr])
        for i in range(5):
            out[f"t13.{nom}.k{i}.obs"] = int((n == i).sum())
            out[f"t13.{nom}.k{i}.esp"] = float(dist[i] * len(bajo))
        for m in ARQ:
            w = sel(W[m]); l = sel(L[m])
            out[f"t15.{m}.{nom}.r"] = float(np.corrcoef(w.loc[l.index], l)[0, 1])
    return out


# ================================================================== BLOQUE abl: ablaciones, EA y reflexion (05)

CONF = {"eegnet": True, "conformer": True, "labram": True, "eegmamba": False}
F4_VARS = {"eegnet": A.ABL_TODAS, "conformer": A.ABL_TODAS, "labram": A.ABL_GRANDES, "eegmamba": A.ABL_GRANDES}


def _ref_within(m, sems):
    """Referencia intra-sujeto por sujeto sobre las semillas `sems`.

    Se promedian las FILAS (particiones) de esas semillas, no las medias de cada semilla: son
    iguales salvo ~1e-16, pero ese ruido rompe empates exactos entre diferencias y mueve el p de
    Wilcoxon hasta 0.008 (ver 07_VERIFICACION.md, hallazgo H2). Se usa la definicion del
    informe 05 (`acc_within` de avance7_ablaciones.py: media de las filas)."""
    d = A.within(m)
    return d[d.seed.isin(list(sems))].groupby("sub").acc.mean().sort_index()


def b_abl() -> dict:
    out = {}
    # --- T1 y T2: F4
    todos22 = []
    for lectura, sems in (("f4", (2026,)), ("f4s", (2027, 2028))):
        filas = []
        for m in ARQ:
            if lectura == "f4s" and m not in ("labram", "eegmamba"):
                continue
            ref = _ref_within(m, sems)
            for v in F4_VARS[m]:
                c = contraste(abl_sub(v, m), ref)
                filas.append((m, v, c))
        # Holm por familia (arquitectura)
        for m in ARQ:
            fam = [(mm, v, c) for mm, v, c in filas if mm == m]
            if not fam:
                continue
            ph = holm([c["p"] for *_, c in fam])
            for (mm, v, c), x in zip(fam, ph):
                c["p_holm"] = float(x)
        if lectura == "f4":
            p22 = holm([c["p"] for *_, c in filas])
            for (m, v, c), x in zip(filas, p22):
                c["p_holm22"] = float(x)
        for m, v, c in filas:
            kk = f"{lectura}.{m}.{v}"
            for f in ("ref", "var", "dif", "dif_med", "ic_inf", "ic_sup", "mejoran", "empeoran",
                      "p", "p_holm", "r", "margen"):
                out[f"{kk}.{f}"] = c[f]
            out[f"{kk}.equiv"] = float(c["ic90_inf"] > -MARGEN and c["ic90_sup"] < MARGEN)
            if "p_holm22" in c:
                out[f"{kk}.p_holm22"] = c["p_holm22"]
    # --- T3: F6 y F8
    princ_fam = {}
    for fam, variante in (("f6", "ea"), ("f8", "augcr")):
        for lectura, sems in (("", (2026,)), ("s", (2027, 2028))):
            filas = []
            for m in ARQ:
                var = loso_sub(m, "acc", variante)[2026]
                ref = loso_sub(m, "acc")[list(sems)].mean(axis=1)
                filas.append((m, contraste(var, ref)))
            ph = holm([c["p"] for _, c in filas])
            for (m, c), x in zip(filas, ph):
                c["p_holm"] = float(x)
            if lectura == "":
                princ_fam[fam] = filas
            for m, c in filas:
                kk = f"{fam}{lectura}.{m}"
                for f in ("ref", "var", "dif", "dif_med", "ic_inf", "ic_sup", "mejoran", "empeoran",
                          "p", "p_holm", "r", "margen"):
                    out[f"{kk}.{f}"] = c[f]
                out[f"{kk}.equiv"] = float(c["ic90_inf"] > -MARGEN and c["ic90_sup"] < MARGEN)
    princ = princ_fam["f6"] + princ_fam["f8"]
    p8 = holm([c["p"] for _, c in princ])
    for ((m, c), x), fam in zip(zip(princ, p8), ["f6"] * 4 + ["f8"] * 4):
        out[f"{fam}.{m}.p_holm8"] = float(x)
    # --- T4: EA y caida vistos/no vistos
    for m in ARQ:
        b = A.loso(m, "base"); b = b[b.seed == 2026].set_index("test_sub").sort_index()
        e = A.loso(m, "ea").set_index("test_sub").sort_index()
        gb, ge = b.acc_visto - b.acc, e.acc_visto - e.acc
        c = contraste(ge, gb)
        w = within_sub(m)[2026].mean()
        k = f"t4ea.{m}"
        out.update({f"{k}.nov_base": b.acc.mean(), f"{k}.nov_ea": e.acc.mean(),
                    f"{k}.vis_base": b.acc_visto.mean(), f"{k}.vis_ea": e.acc_visto.mean(),
                    f"{k}.caida_base": gb.mean(), f"{k}.caida_ea": ge.mean(),
                    f"{k}.ids_base": gb.mean() / b.acc_visto.mean(),
                    f"{k}.ids_ea": ge.mean() / e.acc_visto.mean(),
                    f"{k}.dif": c["dif"], f"{k}.ic_inf": c["ic_inf"], f"{k}.ic_sup": c["ic_sup"],
                    f"{k}.p": c["p"], f"{k}.r": c["r"],
                    f"{k}.dsi_base": (w - b.acc.mean()) / w, f"{k}.dsi_ea": (w - e.acc.mean()) / w})
        # reduccion relativa de la caida
        rng = np.random.default_rng(SEMILLA_BOOT)
        idx = rng.integers(0, len(gb), (B, len(gb)))
        red = 1 - ge.to_numpy()[idx].mean(axis=1) / gb.to_numpy()[idx].mean(axis=1)
        out[f"{k}.red_inf"], out[f"{k}.red_sup"] = (float(x) for x in np.quantile(red, [0.025, 0.975]))
    # --- T6: ruido entre semillas de la base
    for prot in ("within", "loso"):
        for m in ARQ:
            t = sub_prot(prot, m)
            for a, b_ in ((2026, 2027), (2026, 2028), (2027, 2028)):
                d = (t[b_] - t[a]).to_numpy()
                lo, hi = boot(d)
                out[f"t6r.{prot}.{m}.{b_}-{a}.dif"] = d.mean()
                out[f"t6r.{prot}.{m}.{b_}-{a}.ic_inf"], out[f"t6r.{prot}.{m}.{b_}-{a}.ic_sup"] = lo, hi
                out[f"t6r.{prot}.{m}.{b_}-{a}.p"] = w_zsplit(d)[0]
    # --- T7: medias por condicion
    for m in ARQ:
        t = within_sub(m)
        for s in SEM:
            out[f"t7m.within.{m}.base.s{s}"] = t[s].mean()
        for v in F4_VARS[m]:
            out[f"t7m.within.{m}.{v}.s2026"] = abl_sub(v, m).mean()
        tl = loso_sub(m)
        for s in SEM:
            out[f"t7m.loso.{m}.base.s{s}"] = tl[s].mean()
        out[f"t7m.loso.{m}.ea.s2026"] = loso_sub(m, "acc", "ea")[2026].mean()
        out[f"t7m.loso.{m}.augcr.s2026"] = loso_sub(m, "acc", "augcr")[2026].mean()
    # --- T8: cotas (las mismas de t4 de 04, citadas por 05)
    cot = valores(b_confusor)
    for prot, m in (("loso", "eegnet"), ("loso", "conformer"), ("loso", "labram"), ("within", "labram")):
        out[f"t8.{prot}.{m}.bruta"] = cot[f"t4.{prot}.{m}.cota_bruta"]
        out[f"t8.{prot}.{m}.neta"] = cot[f"t4.{prot}.{m}.cota_neta"]
        out[f"t8.{prot}.{m}.uso"] = cot[f"t4.{prot}.{m}.cota_uso"]
    # --- T9: parada temprana
    for m in ARQ:
        base = A.within(m); base = base[base.seed == 2026]
        conds = {"base_2026": base}
        for v in F4_VARS[m]:
            conds[v] = A.ablacion(v, m)
        for nom, d in conds.items():
            out[f"t9p.{m}.{nom}.suelo"] = 100 * float((d.n_epochs == 30).mean())
            out[f"t9p.{m}.{nom}.mediana"] = float(d.n_epochs.median())
            out[f"t9p.{m}.{nom}.tope"] = 100 * float((d.n_epochs == 200).mean())
    return out


# ================================================================== registro y ejecucion

BLOQUES = {"t5": b_t5, "t1": b_confusor, "t2": b_confusor, "t3": b_confusor, "t4": b_confusor,
           "t6": b_t6, "t7": b_t6, "t9": b_t9, "t10": b_t10_cache, "tr": b_tr, "t11": b_tr,
           "pol": b_tr, "polc": b_tr, "trc": b_tr, "eat": b_eat, "eati": b_eat, "eatf": b_eat,
           "eatc": b_eat, "fs_base": b_fs, "fs_ea": b_fs, "fs_sem": b_fs, "fs_media4": b_fs,
           "t12": b_bajo, "t13": b_bajo, "t15": b_bajo,
           "f4": b_abl, "f4s": b_abl, "f6": b_abl, "f6s": b_abl, "f8": b_abl, "f8s": b_abl,
           "t4ea": b_abl, "t6r": b_abl, "t7m": b_abl, "t8": b_abl, "t9p": b_abl}
# nota: "t4" lo declara el informe 04 (cota) y "t4ea" el 05 (EA); "t8" es la cota citada en 05.

_CACHE: dict = {}


def valores(bloque_fn) -> dict:
    if bloque_fn not in _CACHE:
        _CACHE[bloque_fn] = bloque_fn()
    return _CACHE[bloque_fn]


ORIGEN = "crudo"      # todas las cifras salen de los CSV crudos de avance7_datos.py
DONDE = "pendiente"   # se cambia a tab:/sec: cuando el texto este redactado

# --------------------------------------------------------------- declarados provisionales
# (ident) -> (valor, tolerancia, informe). Generado una vez desde las tablas de los informes
# 03, 04 y 05 con la tolerancia = media unidad de la ultima cifra escrita. No editar a mano.
DECLARADOS: dict = {}
# ---- inicio bloque generado ----
DECLARADOS.update({
    "fs_base.eegnet.10.stats.cero": (0.76, 0.0005, "03 sec. 2.3"),
    "fs_base.eegnet.10.stats.gan": (-0.015, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.10.stats.ic_inf": (-0.0264, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.10.stats.ic_sup": (-0.0042, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.10.stats.p": (0.0031, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.10.stats.p_holm": (0.0314, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.10.stats.r": (0.4, 0.005, "03 sec. 2.3"),
    "fs_base.eegnet.10.stats.mejoran": (18.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegnet.10.stats.empeoran": (36.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegnet.10.stats.equiv": (1.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegnet.10.cabeza.cero": (0.76, 0.0005, "03 sec. 2.3"),
    "fs_base.eegnet.10.cabeza.gan": (-0.0128, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.10.cabeza.ic_inf": (-0.0205, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.10.cabeza.ic_sup": (-0.0048, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.10.cabeza.p": (0.0009, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.10.cabeza.p_holm": (0.0142, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.10.cabeza.r": (0.45, 0.005, "03 sec. 2.3"),
    "fs_base.eegnet.10.cabeza.mejoran": (16.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegnet.10.cabeza.empeoran": (38.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegnet.10.cabeza.equiv": (1.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegnet.10.stats+cabeza.cero": (0.76, 0.0005, "03 sec. 2.3"),
    "fs_base.eegnet.10.stats+cabeza.gan": (-0.0162, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.10.stats+cabeza.ic_inf": (-0.0257, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.10.stats+cabeza.ic_sup": (-0.0062, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.10.stats+cabeza.p": (0.0002, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.10.stats+cabeza.p_holm": (0.004, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.10.stats+cabeza.r": (0.51, 0.005, "03 sec. 2.3"),
    "fs_base.eegnet.10.stats+cabeza.mejoran": (13.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegnet.10.stats+cabeza.empeoran": (41.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegnet.10.stats+cabeza.equiv": (1.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegnet.20.stats.cero": (0.76, 0.0005, "03 sec. 2.3"),
    "fs_base.eegnet.20.stats.gan": (-0.0149, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.20.stats.ic_inf": (-0.0275, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.20.stats.ic_sup": (-0.0034, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.20.stats.p": (0.0077, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.20.stats.p_holm": (0.0556, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.20.stats.r": (0.36, 0.005, "03 sec. 2.3"),
    "fs_base.eegnet.20.stats.mejoran": (18.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegnet.20.stats.empeoran": (36.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegnet.20.stats.equiv": (1.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegnet.20.cabeza.cero": (0.76, 0.0005, "03 sec. 2.3"),
    "fs_base.eegnet.20.cabeza.gan": (-0.0065, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.20.cabeza.ic_inf": (-0.0143, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.20.cabeza.ic_sup": (0.0019, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.20.cabeza.p": (0.0372, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.20.cabeza.p_holm": (0.1487, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.20.cabeza.r": (0.28, 0.005, "03 sec. 2.3"),
    "fs_base.eegnet.20.cabeza.mejoran": (19.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegnet.20.cabeza.empeoran": (35.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegnet.20.cabeza.equiv": (1.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegnet.20.stats+cabeza.cero": (0.76, 0.0005, "03 sec. 2.3"),
    "fs_base.eegnet.20.stats+cabeza.gan": (-0.0114, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.20.stats+cabeza.ic_inf": (-0.0213, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.20.stats+cabeza.ic_sup": (-0.001, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.20.stats+cabeza.p": (0.0037, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.20.stats+cabeza.p_holm": (0.0329, 5e-05, "03 sec. 2.3"),
    "fs_base.eegnet.20.stats+cabeza.r": (0.4, 0.005, "03 sec. 2.3"),
    "fs_base.eegnet.20.stats+cabeza.mejoran": (16.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegnet.20.stats+cabeza.empeoran": (38.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegnet.20.stats+cabeza.equiv": (1.0, 0.5, "03 sec. 2.3"),
    "fs_base.conformer.10.stats.cero": (0.745, 0.0005, "03 sec. 2.3"),
    "fs_base.conformer.10.stats.gan": (-0.0242, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.10.stats.ic_inf": (-0.0365, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.10.stats.ic_sup": (-0.0121, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.10.stats.p": (0.0006, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.10.stats.p_holm": (0.0104, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.10.stats.r": (0.47, 0.005, "03 sec. 2.3"),
    "fs_base.conformer.10.stats.mejoran": (18.0, 0.5, "03 sec. 2.3"),
    "fs_base.conformer.10.stats.empeoran": (36.0, 0.5, "03 sec. 2.3"),
    "fs_base.conformer.10.stats.equiv": (0.0, 0.5, "03 sec. 2.3"),
    "fs_base.conformer.10.cabeza.cero": (0.745, 0.0005, "03 sec. 2.3"),
    "fs_base.conformer.10.cabeza.gan": (-0.0229, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.10.cabeza.ic_inf": (-0.0313, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.10.cabeza.ic_sup": (-0.0147, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.10.cabeza.p": (0.0, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.10.cabeza.p_holm": (0.0001, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.10.cabeza.r": (0.61, 0.005, "03 sec. 2.3"),
    "fs_base.conformer.10.cabeza.mejoran": (13.0, 0.5, "03 sec. 2.3"),
    "fs_base.conformer.10.cabeza.empeoran": (41.0, 0.5, "03 sec. 2.3"),
    "fs_base.conformer.10.cabeza.equiv": (0.0, 0.5, "03 sec. 2.3"),
    "fs_base.conformer.10.stats+cabeza.cero": (0.745, 0.0005, "03 sec. 2.3"),
    "fs_base.conformer.10.stats+cabeza.gan": (-0.0409, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.10.stats+cabeza.ic_inf": (-0.0534, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.10.stats+cabeza.ic_sup": (-0.0282, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.10.stats+cabeza.p": (0.0, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.10.stats+cabeza.p_holm": (0.0, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.10.stats+cabeza.r": (0.68, 0.005, "03 sec. 2.3"),
    "fs_base.conformer.10.stats+cabeza.mejoran": (11.0, 0.5, "03 sec. 2.3"),
    "fs_base.conformer.10.stats+cabeza.empeoran": (43.0, 0.5, "03 sec. 2.3"),
    "fs_base.conformer.10.stats+cabeza.equiv": (0.0, 0.5, "03 sec. 2.3"),
    "fs_base.conformer.20.stats.cero": (0.744, 0.0005, "03 sec. 2.3"),
    "fs_base.conformer.20.stats.gan": (-0.0226, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.20.stats.ic_inf": (-0.0348, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.20.stats.ic_sup": (-0.0105, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.20.stats.p": (0.0024, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.20.stats.p_holm": (0.0261, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.20.stats.r": (0.41, 0.005, "03 sec. 2.3"),
    "fs_base.conformer.20.stats.mejoran": (20.0, 0.5, "03 sec. 2.3"),
    "fs_base.conformer.20.stats.empeoran": (34.0, 0.5, "03 sec. 2.3"),
    "fs_base.conformer.20.stats.equiv": (0.0, 0.5, "03 sec. 2.3"),
    "fs_base.conformer.20.cabeza.cero": (0.744, 0.0005, "03 sec. 2.3"),
    "fs_base.conformer.20.cabeza.gan": (-0.0039, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.20.cabeza.ic_inf": (-0.0099, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.20.cabeza.ic_sup": (0.002, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.20.cabeza.p": (0.2347, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.20.cabeza.p_holm": (0.7042, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.20.cabeza.r": (0.16, 0.005, "03 sec. 2.3"),
    "fs_base.conformer.20.cabeza.mejoran": (23.0, 0.5, "03 sec. 2.3"),
    "fs_base.conformer.20.cabeza.empeoran": (31.0, 0.5, "03 sec. 2.3"),
    "fs_base.conformer.20.cabeza.equiv": (1.0, 0.5, "03 sec. 2.3"),
    "fs_base.conformer.20.stats+cabeza.cero": (0.744, 0.0005, "03 sec. 2.3"),
    "fs_base.conformer.20.stats+cabeza.gan": (-0.0204, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.20.stats+cabeza.ic_inf": (-0.0314, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.20.stats+cabeza.ic_sup": (-0.009, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.20.stats+cabeza.p": (0.0013, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.20.stats+cabeza.p_holm": (0.0174, 5e-05, "03 sec. 2.3"),
    "fs_base.conformer.20.stats+cabeza.r": (0.44, 0.005, "03 sec. 2.3"),
    "fs_base.conformer.20.stats+cabeza.mejoran": (20.0, 0.5, "03 sec. 2.3"),
    "fs_base.conformer.20.stats+cabeza.empeoran": (34.0, 0.5, "03 sec. 2.3"),
    "fs_base.conformer.20.stats+cabeza.equiv": (0.0, 0.5, "03 sec. 2.3"),
    "fs_base.labram.10.stats.cero": (0.755, 0.0005, "03 sec. 2.3"),
    "fs_base.labram.10.stats.gan": (-0.0104, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.10.stats.ic_inf": (-0.0171, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.10.stats.ic_sup": (-0.0029, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.10.stats.p": (0.0004, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.10.stats.p_holm": (0.0078, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.10.stats.r": (0.48, 0.005, "03 sec. 2.3"),
    "fs_base.labram.10.stats.mejoran": (16.0, 0.5, "03 sec. 2.3"),
    "fs_base.labram.10.stats.empeoran": (38.0, 0.5, "03 sec. 2.3"),
    "fs_base.labram.10.stats.equiv": (1.0, 0.5, "03 sec. 2.3"),
    "fs_base.labram.10.cabeza.cero": (0.755, 0.0005, "03 sec. 2.3"),
    "fs_base.labram.10.cabeza.gan": (-0.0148, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.10.cabeza.ic_inf": (-0.0247, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.10.cabeza.ic_sup": (-0.0039, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.10.cabeza.p": (0.0006, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.10.cabeza.p_holm": (0.0104, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.10.cabeza.r": (0.47, 0.005, "03 sec. 2.3"),
    "fs_base.labram.10.cabeza.mejoran": (18.0, 0.5, "03 sec. 2.3"),
    "fs_base.labram.10.cabeza.empeoran": (36.0, 0.5, "03 sec. 2.3"),
    "fs_base.labram.10.cabeza.equiv": (1.0, 0.5, "03 sec. 2.3"),
    "fs_base.labram.10.stats+cabeza.cero": (0.755, 0.0005, "03 sec. 2.3"),
    "fs_base.labram.10.stats+cabeza.gan": (-0.0178, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.10.stats+cabeza.ic_inf": (-0.0282, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.10.stats+cabeza.ic_sup": (-0.006, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.10.stats+cabeza.p": (0.0001, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.10.stats+cabeza.p_holm": (0.0031, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.10.stats+cabeza.r": (0.52, 0.005, "03 sec. 2.3"),
    "fs_base.labram.10.stats+cabeza.mejoran": (15.0, 0.5, "03 sec. 2.3"),
    "fs_base.labram.10.stats+cabeza.empeoran": (39.0, 0.5, "03 sec. 2.3"),
    "fs_base.labram.10.stats+cabeza.equiv": (0.0, 0.5, "03 sec. 2.3"),
    "fs_base.labram.20.stats.cero": (0.755, 0.0005, "03 sec. 2.3"),
    "fs_base.labram.20.stats.gan": (-0.0108, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.20.stats.ic_inf": (-0.0174, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.20.stats.ic_sup": (-0.0034, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.20.stats.p": (0.0004, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.20.stats.p_holm": (0.0078, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.20.stats.r": (0.48, 0.005, "03 sec. 2.3"),
    "fs_base.labram.20.stats.mejoran": (15.0, 0.5, "03 sec. 2.3"),
    "fs_base.labram.20.stats.empeoran": (39.0, 0.5, "03 sec. 2.3"),
    "fs_base.labram.20.stats.equiv": (1.0, 0.5, "03 sec. 2.3"),
    "fs_base.labram.20.cabeza.cero": (0.755, 0.0005, "03 sec. 2.3"),
    "fs_base.labram.20.cabeza.gan": (-0.0078, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.20.cabeza.ic_inf": (-0.0185, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.20.cabeza.ic_sup": (0.0051, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.20.cabeza.p": (0.0069, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.20.cabeza.p_holm": (0.0556, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.20.cabeza.r": (0.37, 0.005, "03 sec. 2.3"),
    "fs_base.labram.20.cabeza.mejoran": (20.0, 0.5, "03 sec. 2.3"),
    "fs_base.labram.20.cabeza.empeoran": (34.0, 0.5, "03 sec. 2.3"),
    "fs_base.labram.20.cabeza.equiv": (1.0, 0.5, "03 sec. 2.3"),
    "fs_base.labram.20.stats+cabeza.cero": (0.755, 0.0005, "03 sec. 2.3"),
    "fs_base.labram.20.stats+cabeza.gan": (-0.0125, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.20.stats+cabeza.ic_inf": (-0.0236, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.20.stats+cabeza.ic_sup": (0.0002, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.20.stats+cabeza.p": (0.0018, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.20.stats+cabeza.p_holm": (0.0216, 5e-05, "03 sec. 2.3"),
    "fs_base.labram.20.stats+cabeza.r": (0.42, 0.005, "03 sec. 2.3"),
    "fs_base.labram.20.stats+cabeza.mejoran": (16.0, 0.5, "03 sec. 2.3"),
    "fs_base.labram.20.stats+cabeza.empeoran": (38.0, 0.5, "03 sec. 2.3"),
    "fs_base.labram.20.stats+cabeza.equiv": (1.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegmamba.10.stats.cero": (0.715, 0.0005, "03 sec. 2.3"),
    "fs_base.eegmamba.10.stats.gan": (-0.0027, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.10.stats.ic_inf": (-0.0085, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.10.stats.ic_sup": (0.003, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.10.stats.p": (0.3569, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.10.stats.p_holm": (0.7138, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.10.stats.r": (0.13, 0.005, "03 sec. 2.3"),
    "fs_base.eegmamba.10.stats.mejoran": (23.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegmamba.10.stats.empeoran": (31.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegmamba.10.stats.equiv": (1.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegmamba.10.cabeza.cero": (0.715, 0.0005, "03 sec. 2.3"),
    "fs_base.eegmamba.10.cabeza.gan": (-0.0113, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.10.cabeza.ic_inf": (-0.0202, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.10.cabeza.ic_sup": (-0.0024, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.10.cabeza.p": (0.0082, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.10.cabeza.p_holm": (0.0556, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.10.cabeza.r": (0.36, 0.005, "03 sec. 2.3"),
    "fs_base.eegmamba.10.cabeza.mejoran": (16.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegmamba.10.cabeza.empeoran": (38.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegmamba.10.cabeza.equiv": (1.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegmamba.10.stats+cabeza.cero": (0.715, 0.0005, "03 sec. 2.3"),
    "fs_base.eegmamba.10.stats+cabeza.gan": (-0.0175, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.10.stats+cabeza.ic_inf": (-0.0261, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.10.stats+cabeza.ic_sup": (-0.0086, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.10.stats+cabeza.p": (0.0002, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.10.stats+cabeza.p_holm": (0.005, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.10.stats+cabeza.r": (0.5, 0.005, "03 sec. 2.3"),
    "fs_base.eegmamba.10.stats+cabeza.mejoran": (14.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegmamba.10.stats+cabeza.empeoran": (40.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegmamba.10.stats+cabeza.equiv": (1.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegmamba.20.stats.cero": (0.715, 0.0005, "03 sec. 2.3"),
    "fs_base.eegmamba.20.stats.gan": (-0.0007, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.20.stats.ic_inf": (-0.007, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.20.stats.ic_sup": (0.0054, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.20.stats.p": (0.8128, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.20.stats.p_holm": (0.8128, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.20.stats.r": (0.03, 0.005, "03 sec. 2.3"),
    "fs_base.eegmamba.20.stats.mejoran": (25.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegmamba.20.stats.empeoran": (28.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegmamba.20.stats.equiv": (1.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegmamba.20.cabeza.cero": (0.715, 0.0005, "03 sec. 2.3"),
    "fs_base.eegmamba.20.cabeza.gan": (-0.0092, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.20.cabeza.ic_inf": (-0.0183, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.20.cabeza.ic_sup": (0.0001, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.20.cabeza.p": (0.0297, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.20.cabeza.p_holm": (0.1485, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.20.cabeza.r": (0.3, 0.005, "03 sec. 2.3"),
    "fs_base.eegmamba.20.cabeza.mejoran": (21.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegmamba.20.cabeza.empeoran": (33.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegmamba.20.cabeza.equiv": (1.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegmamba.20.stats+cabeza.cero": (0.715, 0.0005, "03 sec. 2.3"),
    "fs_base.eegmamba.20.stats+cabeza.gan": (-0.0148, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.20.stats+cabeza.ic_inf": (-0.0239, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.20.stats+cabeza.ic_sup": (-0.0054, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.20.stats+cabeza.p": (0.0012, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.20.stats+cabeza.p_holm": (0.0166, 5e-05, "03 sec. 2.3"),
    "fs_base.eegmamba.20.stats+cabeza.r": (0.44, 0.005, "03 sec. 2.3"),
    "fs_base.eegmamba.20.stats+cabeza.mejoran": (15.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegmamba.20.stats+cabeza.empeoran": (39.0, 0.5, "03 sec. 2.3"),
    "fs_base.eegmamba.20.stats+cabeza.equiv": (1.0, 0.5, "03 sec. 2.3"),
    "fs_ea.eegnet.10.stats.cero": (0.774, 0.0005, "03 sec. 2.5"),
    "fs_ea.eegnet.10.stats.gan": (-0.003, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.10.stats.ic_inf": (-0.006, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.10.stats.ic_sup": (0.0001, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.10.stats.p": (0.0152, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.10.stats.p_holm": (0.2123, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.10.stats.r": (0.33, 0.005, "03 sec. 2.5"),
    "fs_ea.eegnet.10.stats.mejoran": (18.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegnet.10.stats.empeoran": (36.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegnet.10.stats.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegnet.10.cabeza.cero": (0.774, 0.0005, "03 sec. 2.5"),
    "fs_ea.eegnet.10.cabeza.gan": (-0.0108, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.10.cabeza.ic_inf": (-0.018, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.10.cabeza.ic_sup": (-0.0038, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.10.cabeza.p": (0.0109, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.10.cabeza.p_holm": (0.186, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.10.cabeza.r": (0.35, 0.005, "03 sec. 2.5"),
    "fs_ea.eegnet.10.cabeza.mejoran": (21.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegnet.10.cabeza.empeoran": (33.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegnet.10.cabeza.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegnet.10.stats+cabeza.cero": (0.774, 0.0005, "03 sec. 2.5"),
    "fs_ea.eegnet.10.stats+cabeza.gan": (-0.0149, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.10.stats+cabeza.ic_inf": (-0.0222, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.10.stats+cabeza.ic_sup": (-0.0077, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.10.stats+cabeza.p": (0.0004, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.10.stats+cabeza.p_holm": (0.0084, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.10.stats+cabeza.r": (0.49, 0.005, "03 sec. 2.5"),
    "fs_ea.eegnet.10.stats+cabeza.mejoran": (17.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegnet.10.stats+cabeza.empeoran": (37.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegnet.10.stats+cabeza.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegnet.20.stats.cero": (0.774, 0.0005, "03 sec. 2.5"),
    "fs_ea.eegnet.20.stats.gan": (-0.0018, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.20.stats.ic_inf": (-0.0046, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.20.stats.ic_sup": (0.0011, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.20.stats.p": (0.0811, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.20.stats.p_holm": (0.7296, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.20.stats.r": (0.24, 0.005, "03 sec. 2.5"),
    "fs_ea.eegnet.20.stats.mejoran": (21.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegnet.20.stats.empeoran": (33.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegnet.20.stats.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegnet.20.cabeza.cero": (0.774, 0.0005, "03 sec. 2.5"),
    "fs_ea.eegnet.20.cabeza.gan": (-0.0061, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.20.cabeza.ic_inf": (-0.0148, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.20.cabeza.ic_sup": (0.0026, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.20.cabeza.p": (0.1492, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.20.cabeza.p_holm": (0.8953, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.20.cabeza.r": (0.2, 0.005, "03 sec. 2.5"),
    "fs_ea.eegnet.20.cabeza.mejoran": (21.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegnet.20.cabeza.empeoran": (33.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegnet.20.cabeza.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegnet.20.stats+cabeza.cero": (0.774, 0.0005, "03 sec. 2.5"),
    "fs_ea.eegnet.20.stats+cabeza.gan": (-0.0083, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.20.stats+cabeza.ic_inf": (-0.0171, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.20.stats+cabeza.ic_sup": (0.0005, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.20.stats+cabeza.p": (0.0587, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.20.stats+cabeza.p_holm": (0.6462, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegnet.20.stats+cabeza.r": (0.26, 0.005, "03 sec. 2.5"),
    "fs_ea.eegnet.20.stats+cabeza.mejoran": (20.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegnet.20.stats+cabeza.empeoran": (34.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegnet.20.stats+cabeza.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.conformer.10.stats.cero": (0.769, 0.0005, "03 sec. 2.5"),
    "fs_ea.conformer.10.stats.gan": (-0.0041, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.10.stats.ic_inf": (-0.0077, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.10.stats.ic_sup": (-0.0005, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.10.stats.p": (0.0345, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.10.stats.p_holm": (0.4485, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.10.stats.r": (0.29, 0.005, "03 sec. 2.5"),
    "fs_ea.conformer.10.stats.mejoran": (18.0, 0.5, "03 sec. 2.5"),
    "fs_ea.conformer.10.stats.empeoran": (36.0, 0.5, "03 sec. 2.5"),
    "fs_ea.conformer.10.stats.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.conformer.10.cabeza.cero": (0.769, 0.0005, "03 sec. 2.5"),
    "fs_ea.conformer.10.cabeza.gan": (-0.0147, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.10.cabeza.ic_inf": (-0.0229, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.10.cabeza.ic_sup": (-0.0073, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.10.cabeza.p": (0.0013, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.10.cabeza.p_holm": (0.0277, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.10.cabeza.r": (0.44, 0.005, "03 sec. 2.5"),
    "fs_ea.conformer.10.cabeza.mejoran": (17.0, 0.5, "03 sec. 2.5"),
    "fs_ea.conformer.10.cabeza.empeoran": (36.0, 0.5, "03 sec. 2.5"),
    "fs_ea.conformer.10.cabeza.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.conformer.10.stats+cabeza.cero": (0.769, 0.0005, "03 sec. 2.5"),
    "fs_ea.conformer.10.stats+cabeza.gan": (-0.0199, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.10.stats+cabeza.ic_inf": (-0.0286, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.10.stats+cabeza.ic_sup": (-0.0121, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.10.stats+cabeza.p": (0.0, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.10.stats+cabeza.p_holm": (0.0001, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.10.stats+cabeza.r": (0.63, 0.005, "03 sec. 2.5"),
    "fs_ea.conformer.10.stats+cabeza.mejoran": (12.0, 0.5, "03 sec. 2.5"),
    "fs_ea.conformer.10.stats+cabeza.empeoran": (42.0, 0.5, "03 sec. 2.5"),
    "fs_ea.conformer.10.stats+cabeza.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.conformer.20.stats.cero": (0.769, 0.0005, "03 sec. 2.5"),
    "fs_ea.conformer.20.stats.gan": (-0.0018, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.20.stats.ic_inf": (-0.0045, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.20.stats.ic_sup": (0.0009, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.20.stats.p": (0.1862, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.20.stats.p_holm": (0.9308, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.20.stats.r": (0.18, 0.005, "03 sec. 2.5"),
    "fs_ea.conformer.20.stats.mejoran": (24.0, 0.5, "03 sec. 2.5"),
    "fs_ea.conformer.20.stats.empeoran": (30.0, 0.5, "03 sec. 2.5"),
    "fs_ea.conformer.20.stats.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.conformer.20.cabeza.cero": (0.769, 0.0005, "03 sec. 2.5"),
    "fs_ea.conformer.20.cabeza.gan": (-0.0057, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.20.cabeza.ic_inf": (-0.0114, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.20.cabeza.ic_sup": (-0.0006, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.20.cabeza.p": (0.4358, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.20.cabeza.p_holm": (1.0, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.20.cabeza.r": (0.11, 0.005, "03 sec. 2.5"),
    "fs_ea.conformer.20.cabeza.mejoran": (26.0, 0.5, "03 sec. 2.5"),
    "fs_ea.conformer.20.cabeza.empeoran": (26.0, 0.5, "03 sec. 2.5"),
    "fs_ea.conformer.20.cabeza.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.conformer.20.stats+cabeza.cero": (0.769, 0.0005, "03 sec. 2.5"),
    "fs_ea.conformer.20.stats+cabeza.gan": (-0.0084, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.20.stats+cabeza.ic_inf": (-0.0139, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.20.stats+cabeza.ic_sup": (-0.0032, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.20.stats+cabeza.p": (0.0113, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.20.stats+cabeza.p_holm": (0.186, 5e-05, "03 sec. 2.5"),
    "fs_ea.conformer.20.stats+cabeza.r": (0.34, 0.005, "03 sec. 2.5"),
    "fs_ea.conformer.20.stats+cabeza.mejoran": (20.0, 0.5, "03 sec. 2.5"),
    "fs_ea.conformer.20.stats+cabeza.empeoran": (34.0, 0.5, "03 sec. 2.5"),
    "fs_ea.conformer.20.stats+cabeza.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.labram.10.stats.cero": (0.779, 0.0005, "03 sec. 2.5"),
    "fs_ea.labram.10.stats.gan": (-0.0045, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.10.stats.ic_inf": (-0.008, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.10.stats.ic_sup": (-0.0011, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.10.stats.p": (0.0141, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.10.stats.p_holm": (0.2116, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.10.stats.r": (0.33, 0.005, "03 sec. 2.5"),
    "fs_ea.labram.10.stats.mejoran": (21.0, 0.5, "03 sec. 2.5"),
    "fs_ea.labram.10.stats.empeoran": (33.0, 0.5, "03 sec. 2.5"),
    "fs_ea.labram.10.stats.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.labram.10.cabeza.cero": (0.779, 0.0005, "03 sec. 2.5"),
    "fs_ea.labram.10.cabeza.gan": (-0.0096, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.10.cabeza.ic_inf": (-0.0191, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.10.cabeza.ic_sup": (0.0011, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.10.cabeza.p": (0.0022, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.10.cabeza.p_holm": (0.0463, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.10.cabeza.r": (0.42, 0.005, "03 sec. 2.5"),
    "fs_ea.labram.10.cabeza.mejoran": (16.0, 0.5, "03 sec. 2.5"),
    "fs_ea.labram.10.cabeza.empeoran": (38.0, 0.5, "03 sec. 2.5"),
    "fs_ea.labram.10.cabeza.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.labram.10.stats+cabeza.cero": (0.779, 0.0005, "03 sec. 2.5"),
    "fs_ea.labram.10.stats+cabeza.gan": (-0.0103, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.10.stats+cabeza.ic_inf": (-0.0202, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.10.stats+cabeza.ic_sup": (0.0009, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.10.stats+cabeza.p": (0.0031, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.10.stats+cabeza.p_holm": (0.0611, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.10.stats+cabeza.r": (0.4, 0.005, "03 sec. 2.5"),
    "fs_ea.labram.10.stats+cabeza.mejoran": (19.0, 0.5, "03 sec. 2.5"),
    "fs_ea.labram.10.stats+cabeza.empeoran": (35.0, 0.5, "03 sec. 2.5"),
    "fs_ea.labram.10.stats+cabeza.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.labram.20.stats.cero": (0.779, 0.0005, "03 sec. 2.5"),
    "fs_ea.labram.20.stats.gan": (-0.003, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.20.stats.ic_inf": (-0.0068, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.20.stats.ic_sup": (0.001, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.20.stats.p": (0.0685, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.20.stats.p_holm": (0.6854, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.20.stats.r": (0.25, 0.005, "03 sec. 2.5"),
    "fs_ea.labram.20.stats.mejoran": (20.0, 0.5, "03 sec. 2.5"),
    "fs_ea.labram.20.stats.empeoran": (34.0, 0.5, "03 sec. 2.5"),
    "fs_ea.labram.20.stats.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.labram.20.cabeza.cero": (0.779, 0.0005, "03 sec. 2.5"),
    "fs_ea.labram.20.cabeza.gan": (-0.001, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.20.cabeza.ic_inf": (-0.0132, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.20.cabeza.ic_sup": (0.0133, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.20.cabeza.p": (0.1102, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.20.cabeza.p_holm": (0.8816, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.20.cabeza.r": (0.22, 0.005, "03 sec. 2.5"),
    "fs_ea.labram.20.cabeza.mejoran": (21.0, 0.5, "03 sec. 2.5"),
    "fs_ea.labram.20.cabeza.empeoran": (33.0, 0.5, "03 sec. 2.5"),
    "fs_ea.labram.20.cabeza.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.labram.20.stats+cabeza.cero": (0.779, 0.0005, "03 sec. 2.5"),
    "fs_ea.labram.20.stats+cabeza.gan": (-0.0033, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.20.stats+cabeza.ic_inf": (-0.0156, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.20.stats+cabeza.ic_sup": (0.0115, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.20.stats+cabeza.p": (0.043, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.20.stats+cabeza.p_holm": (0.5162, 5e-05, "03 sec. 2.5"),
    "fs_ea.labram.20.stats+cabeza.r": (0.28, 0.005, "03 sec. 2.5"),
    "fs_ea.labram.20.stats+cabeza.mejoran": (21.0, 0.5, "03 sec. 2.5"),
    "fs_ea.labram.20.stats+cabeza.empeoran": (33.0, 0.5, "03 sec. 2.5"),
    "fs_ea.labram.20.stats+cabeza.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.stats.cero": (0.763, 0.0005, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.stats.gan": (-0.0018, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.stats.ic_inf": (-0.0046, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.stats.ic_sup": (0.001, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.stats.p": (0.1263, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.stats.p_holm": (0.8844, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.stats.r": (0.21, 0.005, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.stats.mejoran": (21.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.stats.empeoran": (32.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.stats.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.cabeza.cero": (0.763, 0.0005, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.cabeza.gan": (-0.0105, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.cabeza.ic_inf": (-0.0228, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.cabeza.ic_sup": (0.0044, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.cabeza.p": (0.0053, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.cabeza.p_holm": (0.0962, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.cabeza.r": (0.38, 0.005, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.cabeza.mejoran": (16.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.cabeza.empeoran": (38.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.cabeza.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.stats+cabeza.cero": (0.763, 0.0005, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.stats+cabeza.gan": (-0.0107, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.stats+cabeza.ic_inf": (-0.0226, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.stats+cabeza.ic_sup": (0.0039, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.stats+cabeza.p": (0.0034, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.stats+cabeza.p_holm": (0.0649, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.stats+cabeza.r": (0.4, 0.005, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.stats+cabeza.mejoran": (17.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.stats+cabeza.empeoran": (37.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegmamba.10.stats+cabeza.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.stats.cero": (0.763, 0.0005, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.stats.gan": (-0.0011, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.stats.ic_inf": (-0.0039, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.stats.ic_sup": (0.0018, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.stats.p": (0.2837, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.stats.p_holm": (1.0, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.stats.r": (0.15, 0.005, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.stats.mejoran": (23.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.stats.empeoran": (29.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.stats.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.cabeza.cero": (0.763, 0.0005, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.cabeza.gan": (-0.003, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.cabeza.ic_inf": (-0.0196, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.cabeza.ic_sup": (0.0152, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.cabeza.p": (0.276, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.cabeza.p_holm": (1.0, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.cabeza.r": (0.15, 0.005, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.cabeza.mejoran": (22.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.cabeza.empeoran": (32.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.cabeza.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.stats+cabeza.cero": (0.763, 0.0005, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.stats+cabeza.gan": (-0.0041, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.stats+cabeza.ic_inf": (-0.0207, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.stats+cabeza.ic_sup": (0.014, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.stats+cabeza.p": (0.2975, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.stats+cabeza.p_holm": (1.0, 5e-05, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.stats+cabeza.r": (0.14, 0.005, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.stats+cabeza.mejoran": (23.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.stats+cabeza.empeoran": (31.0, 0.5, "03 sec. 2.5"),
    "fs_ea.eegmamba.20.stats+cabeza.equiv": (1.0, 0.5, "03 sec. 2.5"),
    "fs_sem.eegnet.10.stats.gan2027": (-0.013, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.10.stats.gan2028": (-0.0171, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.10.stats.p2027": (0.0356, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.10.stats.p2028": (0.0094, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.10.stats.rho": (0.53, 0.005, "03 sec. 2.4"),
    "fs_sem.eegnet.10.cabeza.gan2027": (-0.0144, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.10.cabeza.gan2028": (-0.0111, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.10.cabeza.p2027": (0.0016, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.10.cabeza.p2028": (0.0065, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.10.cabeza.rho": (0.47, 0.005, "03 sec. 2.4"),
    "fs_sem.eegnet.10.stats+cabeza.gan2027": (-0.0168, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.10.stats+cabeza.gan2028": (-0.0156, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.10.stats+cabeza.p2027": (0.0002, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.10.stats+cabeza.p2028": (0.0008, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.10.stats+cabeza.rho": (0.56, 0.005, "03 sec. 2.4"),
    "fs_sem.eegnet.20.stats.gan2027": (-0.0119, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.20.stats.gan2028": (-0.018, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.20.stats.p2027": (0.0797, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.20.stats.p2028": (0.0173, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.20.stats.rho": (0.45, 0.005, "03 sec. 2.4"),
    "fs_sem.eegnet.20.cabeza.gan2027": (-0.0059, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.20.cabeza.gan2028": (-0.0071, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.20.cabeza.p2027": (0.085, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.20.cabeza.p2028": (0.0453, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.20.cabeza.rho": (0.44, 0.005, "03 sec. 2.4"),
    "fs_sem.eegnet.20.stats+cabeza.gan2027": (-0.0094, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.20.stats+cabeza.gan2028": (-0.0133, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.20.stats+cabeza.p2027": (0.0163, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.20.stats+cabeza.p2028": (0.0175, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegnet.20.stats+cabeza.rho": (0.36, 0.005, "03 sec. 2.4"),
    "fs_sem.conformer.10.stats.gan2027": (-0.0229, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.10.stats.gan2028": (-0.0254, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.10.stats.p2027": (0.0028, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.10.stats.p2028": (0.0004, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.10.stats.rho": (0.74, 0.005, "03 sec. 2.4"),
    "fs_sem.conformer.10.cabeza.gan2027": (-0.0236, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.10.cabeza.gan2028": (-0.0222, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.10.cabeza.p2027": (0.0002, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.10.cabeza.p2028": (0.0, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.10.cabeza.rho": (0.61, 0.005, "03 sec. 2.4"),
    "fs_sem.conformer.10.stats+cabeza.gan2027": (-0.0377, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.10.stats+cabeza.gan2028": (-0.044, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.10.stats+cabeza.p2027": (0.0, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.10.stats+cabeza.p2028": (0.0, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.10.stats+cabeza.rho": (0.6, 0.005, "03 sec. 2.4"),
    "fs_sem.conformer.20.stats.gan2027": (-0.0209, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.20.stats.gan2028": (-0.0243, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.20.stats.p2027": (0.0133, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.20.stats.p2028": (0.0006, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.20.stats.rho": (0.74, 0.005, "03 sec. 2.4"),
    "fs_sem.conformer.20.cabeza.gan2027": (-0.0017, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.20.cabeza.gan2028": (-0.0061, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.20.cabeza.p2027": (0.4011, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.20.cabeza.p2028": (0.1444, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.20.cabeza.rho": (0.51, 0.005, "03 sec. 2.4"),
    "fs_sem.conformer.20.stats+cabeza.gan2027": (-0.0151, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.20.stats+cabeza.gan2028": (-0.0257, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.20.stats+cabeza.p2027": (0.0119, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.20.stats+cabeza.p2028": (0.0003, 5e-05, "03 sec. 2.4"),
    "fs_sem.conformer.20.stats+cabeza.rho": (0.74, 0.005, "03 sec. 2.4"),
    "fs_sem.labram.10.stats.gan2027": (-0.0094, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.10.stats.gan2028": (-0.0114, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.10.stats.p2027": (0.0025, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.10.stats.p2028": (0.0059, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.10.stats.rho": (0.3, 0.005, "03 sec. 2.4"),
    "fs_sem.labram.10.cabeza.gan2027": (-0.0155, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.10.cabeza.gan2028": (-0.0141, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.10.cabeza.p2027": (0.0124, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.10.cabeza.p2028": (0.0028, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.10.cabeza.rho": (0.44, 0.005, "03 sec. 2.4"),
    "fs_sem.labram.10.stats+cabeza.gan2027": (-0.0171, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.10.stats+cabeza.gan2028": (-0.0184, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.10.stats+cabeza.p2027": (0.0007, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.10.stats+cabeza.p2028": (0.0016, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.10.stats+cabeza.rho": (0.47, 0.005, "03 sec. 2.4"),
    "fs_sem.labram.20.stats.gan2027": (-0.011, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.20.stats.gan2028": (-0.0106, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.20.stats.p2027": (0.0002, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.20.stats.p2028": (0.0125, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.20.stats.rho": (0.43, 0.005, "03 sec. 2.4"),
    "fs_sem.labram.20.cabeza.gan2027": (-0.0073, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.20.cabeza.gan2028": (-0.0083, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.20.cabeza.p2027": (0.022, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.20.cabeza.p2028": (0.0297, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.20.cabeza.rho": (0.52, 0.005, "03 sec. 2.4"),
    "fs_sem.labram.20.stats+cabeza.gan2027": (-0.0112, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.20.stats+cabeza.gan2028": (-0.0138, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.20.stats+cabeza.p2027": (0.0045, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.20.stats+cabeza.p2028": (0.0107, 5e-05, "03 sec. 2.4"),
    "fs_sem.labram.20.stats+cabeza.rho": (0.56, 0.005, "03 sec. 2.4"),
    "fs_sem.eegmamba.10.stats.gan2027": (-0.0017, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.10.stats.gan2028": (-0.0036, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.10.stats.p2027": (0.6175, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.10.stats.p2028": (0.4748, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.10.stats.rho": (0.25, 0.005, "03 sec. 2.4"),
    "fs_sem.eegmamba.10.cabeza.gan2027": (-0.0121, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.10.cabeza.gan2028": (-0.0106, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.10.cabeza.p2027": (0.0105, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.10.cabeza.p2028": (0.0094, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.10.cabeza.rho": (0.56, 0.005, "03 sec. 2.4"),
    "fs_sem.eegmamba.10.stats+cabeza.gan2027": (-0.0191, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.10.stats+cabeza.gan2028": (-0.0158, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.10.stats+cabeza.p2027": (0.0002, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.10.stats+cabeza.p2028": (0.0008, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.10.stats+cabeza.rho": (0.53, 0.005, "03 sec. 2.4"),
    "fs_sem.eegmamba.20.stats.gan2027": (0.0, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.20.stats.gan2028": (-0.0015, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.20.stats.p2027": (0.7928, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.20.stats.p2028": (0.7598, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.20.stats.rho": (0.5, 0.005, "03 sec. 2.4"),
    "fs_sem.eegmamba.20.cabeza.gan2027": (-0.009, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.20.cabeza.gan2028": (-0.0095, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.20.cabeza.p2027": (0.0349, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.20.cabeza.p2028": (0.0303, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.20.cabeza.rho": (0.68, 0.005, "03 sec. 2.4"),
    "fs_sem.eegmamba.20.stats+cabeza.gan2027": (-0.0133, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.20.stats+cabeza.gan2028": (-0.0162, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.20.stats+cabeza.p2027": (0.0035, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.20.stats+cabeza.p2028": (0.0014, 5e-05, "03 sec. 2.4"),
    "fs_sem.eegmamba.20.stats+cabeza.rho": (0.62, 0.005, "03 sec. 2.4"),
    "fs_media4.base.10.stats": (-0.0131, 5e-05, "03 sec. 2.3 (texto)"),
    "fs_media4.base.10.cabeza": (-0.0154, 5e-05, "03 sec. 2.3 (texto)"),
    "fs_media4.base.10.stats+cabeza": (-0.0231, 5e-05, "03 sec. 2.3 (texto)"),
    "fs_media4.base.20.stats": (-0.0123, 5e-05, "03 sec. 2.3 (texto)"),
    "fs_media4.base.20.cabeza": (-0.0069, 5e-05, "03 sec. 2.3 (texto)"),
    "fs_media4.base.20.stats+cabeza": (-0.0148, 5e-05, "03 sec. 2.3 (texto)"),
    "fs_media4.ea.10.stats": (-0.0034, 5e-05, "03 sec. 2.5 (texto)"),
    "fs_media4.ea.10.cabeza": (-0.0114, 5e-05, "03 sec. 2.5 (texto)"),
    "fs_media4.ea.10.stats+cabeza": (-0.0139, 5e-05, "03 sec. 2.5 (texto)"),
    "fs_media4.ea.20.stats": (-0.0019, 5e-05, "03 sec. 2.5 (texto)"),
    "fs_media4.ea.20.cabeza": (-0.004, 5e-05, "03 sec. 2.5 (texto)"),
    "fs_media4.ea.20.stats+cabeza": (-0.006, 5e-05, "03 sec. 2.5 (texto)"),
    "tr.transferido.eegnet.s2026": (0.626, 0.0005, "03 sec. 3.3"),
    "tr.transferido.eegnet.s2027": (0.617, 0.0005, "03 sec. 3.3"),
    "tr.transferido.eegnet.s2028": (0.617, 0.0005, "03 sec. 3.3"),
    "tr.transferido.eegnet.media": (0.62, 0.0005, "03 sec. 3.3"),
    "tr.transferido.eegnet.dt": (0.005, 0.0005, "03 sec. 3.3"),
    "tr.transferido.eegnet.min": (0.617, 0.0005, "03 sec. 3.3"),
    "tr.transferido.eegnet.max": (0.626, 0.0005, "03 sec. 3.3"),
    "tr.transferido.conformer.s2026": (0.611, 0.0005, "03 sec. 3.3"),
    "tr.transferido.conformer.s2027": (0.618, 0.0005, "03 sec. 3.3"),
    "tr.transferido.conformer.s2028": (0.613, 0.0005, "03 sec. 3.3"),
    "tr.transferido.conformer.media": (0.614, 0.0005, "03 sec. 3.3"),
    "tr.transferido.conformer.dt": (0.003, 0.0005, "03 sec. 3.3"),
    "tr.transferido.conformer.min": (0.611, 0.0005, "03 sec. 3.3"),
    "tr.transferido.conformer.max": (0.618, 0.0005, "03 sec. 3.3"),
    "tr.transferido.labram.s2026": (0.613, 0.0005, "03 sec. 3.3"),
    "tr.transferido.labram.s2027": (0.602, 0.0005, "03 sec. 3.3"),
    "tr.transferido.labram.s2028": (0.615, 0.0005, "03 sec. 3.3"),
    "tr.transferido.labram.media": (0.61, 0.0005, "03 sec. 3.3"),
    "tr.transferido.labram.dt": (0.007, 0.0005, "03 sec. 3.3"),
    "tr.transferido.labram.min": (0.602, 0.0005, "03 sec. 3.3"),
    "tr.transferido.labram.max": (0.615, 0.0005, "03 sec. 3.3"),
    "tr.transferido.eegmamba.s2026": (0.589, 0.0005, "03 sec. 3.3"),
    "tr.transferido.eegmamba.s2027": (0.599, 0.0005, "03 sec. 3.3"),
    "tr.transferido.eegmamba.s2028": (0.601, 0.0005, "03 sec. 3.3"),
    "tr.transferido.eegmamba.media": (0.596, 0.0005, "03 sec. 3.3"),
    "tr.transferido.eegmamba.dt": (0.006, 0.0005, "03 sec. 3.3"),
    "tr.transferido.eegmamba.min": (0.589, 0.0005, "03 sec. 3.3"),
    "tr.transferido.eegmamba.max": (0.601, 0.0005, "03 sec. 3.3"),
    "tr.nativo.eegnet.s2026": (0.808, 0.0005, "03 sec. 3.3"),
    "tr.nativo.eegnet.s2027": (0.817, 0.0005, "03 sec. 3.3"),
    "tr.nativo.eegnet.s2028": (0.811, 0.0005, "03 sec. 3.3"),
    "tr.nativo.eegnet.media": (0.812, 0.0005, "03 sec. 3.3"),
    "tr.nativo.eegnet.dt": (0.004, 0.0005, "03 sec. 3.3"),
    "tr.nativo.eegnet.min": (0.808, 0.0005, "03 sec. 3.3"),
    "tr.nativo.eegnet.max": (0.817, 0.0005, "03 sec. 3.3"),
    "tr.nativo.conformer.s2026": (0.8, 0.0005, "03 sec. 3.3"),
    "tr.nativo.conformer.s2027": (0.797, 0.0005, "03 sec. 3.3"),
    "tr.nativo.conformer.s2028": (0.798, 0.0005, "03 sec. 3.3"),
    "tr.nativo.conformer.media": (0.799, 0.0005, "03 sec. 3.3"),
    "tr.nativo.conformer.dt": (0.002, 0.0005, "03 sec. 3.3"),
    "tr.nativo.conformer.min": (0.797, 0.0005, "03 sec. 3.3"),
    "tr.nativo.conformer.max": (0.8, 0.0005, "03 sec. 3.3"),
    "tr.nativo.labram.s2026": (0.814, 0.0005, "03 sec. 3.3"),
    "tr.nativo.labram.s2027": (0.83, 0.0005, "03 sec. 3.3"),
    "tr.nativo.labram.s2028": (0.822, 0.0005, "03 sec. 3.3"),
    "tr.nativo.labram.media": (0.822, 0.0005, "03 sec. 3.3"),
    "tr.nativo.labram.dt": (0.008, 0.0005, "03 sec. 3.3"),
    "tr.nativo.labram.min": (0.814, 0.0005, "03 sec. 3.3"),
    "tr.nativo.labram.max": (0.83, 0.0005, "03 sec. 3.3"),
    "tr.nativo.eegmamba.s2026": (0.762, 0.0005, "03 sec. 3.3"),
    "tr.nativo.eegmamba.s2027": (0.76, 0.0005, "03 sec. 3.3"),
    "tr.nativo.eegmamba.s2028": (0.775, 0.0005, "03 sec. 3.3"),
    "tr.nativo.eegmamba.media": (0.766, 0.0005, "03 sec. 3.3"),
    "tr.nativo.eegmamba.dt": (0.008, 0.0005, "03 sec. 3.3"),
    "tr.nativo.eegmamba.min": (0.76, 0.0005, "03 sec. 3.3"),
    "tr.nativo.eegmamba.max": (0.775, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.eegnet.s2026": (0.182, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.eegnet.s2027": (0.2, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.eegnet.s2028": (0.195, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.eegnet.media": (0.192, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.eegnet.dt": (0.009, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.eegnet.min": (0.182, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.eegnet.max": (0.2, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.conformer.s2026": (0.19, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.conformer.s2027": (0.179, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.conformer.s2028": (0.185, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.conformer.media": (0.185, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.conformer.dt": (0.005, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.conformer.min": (0.179, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.conformer.max": (0.19, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.labram.s2026": (0.201, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.labram.s2027": (0.228, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.labram.s2028": (0.207, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.labram.media": (0.212, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.labram.dt": (0.014, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.labram.min": (0.201, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.labram.max": (0.228, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.eegmamba.s2026": (0.173, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.eegmamba.s2027": (0.161, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.eegmamba.s2028": (0.174, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.eegmamba.media": (0.169, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.eegmamba.dt": (0.007, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.eegmamba.min": (0.161, 0.0005, "03 sec. 3.3"),
    "tr.dif_nativo_menos_transferido.eegmamba.max": (0.174, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.eegnet.s2026": (0.802, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.eegnet.s2027": (0.783, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.eegnet.s2028": (0.794, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.eegnet.media": (0.793, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.eegnet.dt": (0.009, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.eegnet.min": (0.783, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.eegnet.max": (0.802, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.conformer.s2026": (0.814, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.conformer.s2027": (0.8, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.conformer.s2028": (0.815, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.conformer.media": (0.81, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.conformer.dt": (0.008, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.conformer.min": (0.8, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.conformer.max": (0.815, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.labram.s2026": (0.813, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.labram.s2027": (0.814, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.labram.s2028": (0.805, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.labram.media": (0.811, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.labram.dt": (0.005, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.labram.min": (0.805, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.labram.max": (0.814, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.eegmamba.s2026": (0.753, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.eegmamba.s2027": (0.741, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.eegmamba.s2028": (0.769, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.eegmamba.media": (0.755, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.eegmamba.dt": (0.014, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.eegmamba.min": (0.741, 0.0005, "03 sec. 3.3"),
    "tr.vistos44.eegmamba.max": (0.769, 0.0005, "03 sec. 3.3"),
    "tr.persona.eegnet.s2026": (0.742, 0.0005, "03 sec. 3.3"),
    "tr.persona.eegnet.s2027": (0.729, 0.0005, "03 sec. 3.3"),
    "tr.persona.eegnet.s2028": (0.744, 0.0005, "03 sec. 3.3"),
    "tr.persona.eegnet.media": (0.738, 0.0005, "03 sec. 3.3"),
    "tr.persona.eegnet.dt": (0.008, 0.0005, "03 sec. 3.3"),
    "tr.persona.eegnet.min": (0.729, 0.0005, "03 sec. 3.3"),
    "tr.persona.eegnet.max": (0.744, 0.0005, "03 sec. 3.3"),
    "tr.persona.conformer.s2026": (0.718, 0.0005, "03 sec. 3.3"),
    "tr.persona.conformer.s2027": (0.714, 0.0005, "03 sec. 3.3"),
    "tr.persona.conformer.s2028": (0.735, 0.0005, "03 sec. 3.3"),
    "tr.persona.conformer.media": (0.722, 0.0005, "03 sec. 3.3"),
    "tr.persona.conformer.dt": (0.011, 0.0005, "03 sec. 3.3"),
    "tr.persona.conformer.min": (0.714, 0.0005, "03 sec. 3.3"),
    "tr.persona.conformer.max": (0.735, 0.0005, "03 sec. 3.3"),
    "tr.persona.labram.s2026": (0.745, 0.0005, "03 sec. 3.3"),
    "tr.persona.labram.s2027": (0.739, 0.0005, "03 sec. 3.3"),
    "tr.persona.labram.s2028": (0.734, 0.0005, "03 sec. 3.3"),
    "tr.persona.labram.media": (0.739, 0.0005, "03 sec. 3.3"),
    "tr.persona.labram.dt": (0.006, 0.0005, "03 sec. 3.3"),
    "tr.persona.labram.min": (0.734, 0.0005, "03 sec. 3.3"),
    "tr.persona.labram.max": (0.745, 0.0005, "03 sec. 3.3"),
    "tr.persona.eegmamba.s2026": (0.7, 0.0005, "03 sec. 3.3"),
    "tr.persona.eegmamba.s2027": (0.693, 0.0005, "03 sec. 3.3"),
    "tr.persona.eegmamba.s2028": (0.684, 0.0005, "03 sec. 3.3"),
    "tr.persona.eegmamba.media": (0.692, 0.0005, "03 sec. 3.3"),
    "tr.persona.eegmamba.dt": (0.008, 0.0005, "03 sec. 3.3"),
    "tr.persona.eegmamba.min": (0.684, 0.0005, "03 sec. 3.3"),
    "tr.persona.eegmamba.max": (0.7, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.eegnet.s2026": (0.621, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.eegnet.s2027": (0.619, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.eegnet.s2028": (0.63, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.eegnet.media": (0.623, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.eegnet.dt": (0.006, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.eegnet.min": (0.619, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.eegnet.max": (0.63, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.conformer.s2026": (0.618, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.conformer.s2027": (0.622, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.conformer.s2028": (0.615, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.conformer.media": (0.618, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.conformer.dt": (0.003, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.conformer.min": (0.615, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.conformer.max": (0.622, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.labram.s2026": (0.628, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.labram.s2027": (0.611, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.labram.s2028": (0.608, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.labram.media": (0.616, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.labram.dt": (0.011, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.labram.min": (0.608, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.labram.max": (0.628, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.eegmamba.s2026": (0.598, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.eegmamba.s2027": (0.595, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.eegmamba.s2028": (0.6, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.eegmamba.media": (0.598, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.eegmamba.dt": (0.003, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.eegmamba.min": (0.595, 0.0005, "03 sec. 3.3"),
    "tr.wbcic44.eegmamba.max": (0.6, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.eegnet.s2026": (0.059, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.eegnet.s2027": (0.054, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.eegnet.s2028": (0.05, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.eegnet.media": (0.055, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.eegnet.dt": (0.005, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.eegnet.min": (0.05, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.eegnet.max": (0.059, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.conformer.s2026": (0.096, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.conformer.s2027": (0.086, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.conformer.s2028": (0.08, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.conformer.media": (0.087, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.conformer.dt": (0.008, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.conformer.min": (0.08, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.conformer.max": (0.096, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.labram.s2026": (0.068, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.labram.s2027": (0.075, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.labram.s2028": (0.071, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.labram.media": (0.071, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.labram.dt": (0.004, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.labram.min": (0.068, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.labram.max": (0.075, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.eegmamba.s2026": (0.054, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.eegmamba.s2027": (0.048, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.eegmamba.s2028": (0.085, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.eegmamba.media": (0.062, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.eegmamba.dt": (0.02, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.eegmamba.min": (0.048, 0.0005, "03 sec. 3.3"),
    "tr.caida_persona.eegmamba.max": (0.085, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.eegnet.s2026": (0.121, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.eegnet.s2027": (0.11, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.eegnet.s2028": (0.114, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.eegnet.media": (0.115, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.eegnet.dt": (0.006, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.eegnet.min": (0.11, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.eegnet.max": (0.121, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.conformer.s2026": (0.1, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.conformer.s2027": (0.093, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.conformer.s2028": (0.12, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.conformer.media": (0.104, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.conformer.dt": (0.014, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.conformer.min": (0.093, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.conformer.max": (0.12, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.labram.s2026": (0.117, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.labram.s2027": (0.127, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.labram.s2028": (0.126, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.labram.media": (0.123, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.labram.dt": (0.006, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.labram.min": (0.117, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.labram.max": (0.127, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.eegmamba.s2026": (0.102, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.eegmamba.s2027": (0.098, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.eegmamba.s2028": (0.083, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.eegmamba.media": (0.095, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.eegmamba.dt": (0.01, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.eegmamba.min": (0.083, 0.0005, "03 sec. 3.3"),
    "tr.caida_protocolo.eegmamba.max": (0.102, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.eegnet.s2026": (0.062, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.eegnet.s2027": (0.056, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.eegnet.s2028": (0.063, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.eegnet.media": (0.06, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.eegnet.dt": (0.004, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.eegnet.min": (0.056, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.eegnet.max": (0.063, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.conformer.s2026": (0.003, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.conformer.s2027": (0.007, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.conformer.s2028": (0.041, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.conformer.media": (0.017, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.conformer.dt": (0.021, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.conformer.min": (0.003, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.conformer.max": (0.041, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.labram.s2026": (0.049, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.labram.s2027": (0.052, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.labram.s2028": (0.055, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.labram.media": (0.052, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.labram.dt": (0.003, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.labram.min": (0.049, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.labram.max": (0.055, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.eegmamba.s2026": (0.048, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.eegmamba.s2027": (0.05, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.eegmamba.s2028": (-0.002, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.eegmamba.media": (0.032, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.eegmamba.dt": (0.029, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.eegmamba.min": (-0.002, 0.0005, "03 sec. 3.3"),
    "tr.protocolo_menos_persona.eegmamba.max": (0.05, 0.0005, "03 sec. 3.3"),
    "tr.idp.eegnet.s2026": (0.163, 0.0005, "03 sec. 3.3"),
    "tr.idp.eegnet.s2027": (0.151, 0.0005, "03 sec. 3.3"),
    "tr.idp.eegnet.s2028": (0.153, 0.0005, "03 sec. 3.3"),
    "tr.idp.eegnet.media": (0.156, 0.0005, "03 sec. 3.3"),
    "tr.idp.eegnet.dt": (0.006, 0.0005, "03 sec. 3.3"),
    "tr.idp.eegnet.min": (0.151, 0.0005, "03 sec. 3.3"),
    "tr.idp.eegnet.max": (0.163, 0.0005, "03 sec. 3.3"),
    "tr.idp.conformer.s2026": (0.139, 0.0005, "03 sec. 3.3"),
    "tr.idp.conformer.s2027": (0.13, 0.0005, "03 sec. 3.3"),
    "tr.idp.conformer.s2028": (0.164, 0.0005, "03 sec. 3.3"),
    "tr.idp.conformer.media": (0.144, 0.0005, "03 sec. 3.3"),
    "tr.idp.conformer.dt": (0.018, 0.0005, "03 sec. 3.3"),
    "tr.idp.conformer.min": (0.13, 0.0005, "03 sec. 3.3"),
    "tr.idp.conformer.max": (0.164, 0.0005, "03 sec. 3.3"),
    "tr.idp.labram.s2026": (0.157, 0.0005, "03 sec. 3.3"),
    "tr.idp.labram.s2027": (0.173, 0.0005, "03 sec. 3.3"),
    "tr.idp.labram.s2028": (0.172, 0.0005, "03 sec. 3.3"),
    "tr.idp.labram.media": (0.167, 0.0005, "03 sec. 3.3"),
    "tr.idp.labram.dt": (0.009, 0.0005, "03 sec. 3.3"),
    "tr.idp.labram.min": (0.157, 0.0005, "03 sec. 3.3"),
    "tr.idp.labram.max": (0.173, 0.0005, "03 sec. 3.3"),
    "tr.idp.eegmamba.s2026": (0.146, 0.0005, "03 sec. 3.3"),
    "tr.idp.eegmamba.s2027": (0.142, 0.0005, "03 sec. 3.3"),
    "tr.idp.eegmamba.s2028": (0.122, 0.0005, "03 sec. 3.3"),
    "tr.idp.eegmamba.media": (0.137, 0.0005, "03 sec. 3.3"),
    "tr.idp.eegmamba.dt": (0.013, 0.0005, "03 sec. 3.3"),
    "tr.idp.eegmamba.min": (0.122, 0.0005, "03 sec. 3.3"),
    "tr.idp.eegmamba.max": (0.146, 0.0005, "03 sec. 3.3"),
    "trc.caida_persona.eegnet.s2026.inf": (-0.013, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.eegnet.s2026.sup": (0.131, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.eegnet.sujetos3.inf": (-0.018, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.eegnet.sujetos3.sup": (0.126, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.eegnet.jer3.inf": (-0.019, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.eegnet.jer3.sup": (0.126, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.eegnet.s2027.inf": (-0.021, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.eegnet.s2027.sup": (0.126, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.eegnet.s2028.inf": (-0.024, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.eegnet.s2028.sup": (0.123, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.conformer.s2026.inf": (0.013, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.conformer.s2026.sup": (0.173, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.conformer.sujetos3.inf": (0.007, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.conformer.sujetos3.sup": (0.16, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.conformer.jer3.inf": (0.007, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.conformer.jer3.sup": (0.161, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.conformer.s2027.inf": (0.006, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.conformer.s2027.sup": (0.155, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.conformer.s2028.inf": (0.001, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.conformer.s2028.sup": (0.153, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.labram.s2026.inf": (-0.001, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.labram.s2026.sup": (0.136, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.labram.sujetos3.inf": (0.007, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.labram.sujetos3.sup": (0.135, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.labram.jer3.inf": (0.006, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.labram.jer3.sup": (0.135, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.labram.s2027.inf": (0.016, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.labram.s2027.sup": (0.134, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.labram.s2028.inf": (0.005, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.labram.s2028.sup": (0.137, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.eegmamba.s2026.inf": (-0.008, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.eegmamba.s2026.sup": (0.108, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.eegmamba.sujetos3.inf": (0.001, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.eegmamba.sujetos3.sup": (0.115, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.eegmamba.jer3.inf": (-0.001, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.eegmamba.jer3.sup": (0.12, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.eegmamba.s2027.inf": (-0.009, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.eegmamba.s2027.sup": (0.098, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.eegmamba.s2028.inf": (0.018, 0.0005, "03 sec. 3.4"),
    "trc.caida_persona.eegmamba.s2028.sup": (0.141, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.eegnet.s2026.inf": (0.042, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.eegnet.s2026.sup": (0.201, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.eegnet.sujetos3.inf": (0.036, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.eegnet.sujetos3.sup": (0.195, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.eegnet.jer3.inf": (0.035, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.eegnet.jer3.sup": (0.195, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.eegnet.s2027.inf": (0.03, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.eegnet.s2027.sup": (0.192, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.eegnet.s2028.inf": (0.035, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.eegnet.s2028.sup": (0.195, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.conformer.s2026.inf": (0.014, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.conformer.s2026.sup": (0.191, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.conformer.sujetos3.inf": (0.023, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.conformer.sujetos3.sup": (0.193, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.conformer.jer3.inf": (0.022, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.conformer.jer3.sup": (0.193, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.conformer.s2027.inf": (0.014, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.conformer.s2027.sup": (0.181, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.conformer.s2028.inf": (0.037, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.conformer.s2028.sup": (0.208, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.labram.s2026.inf": (0.04, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.labram.s2026.sup": (0.195, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.labram.sujetos3.inf": (0.051, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.labram.sujetos3.sup": (0.197, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.labram.jer3.inf": (0.05, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.labram.jer3.sup": (0.198, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.labram.s2027.inf": (0.056, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.labram.s2027.sup": (0.199, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.labram.s2028.inf": (0.052, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.labram.s2028.sup": (0.201, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.eegmamba.s2026.inf": (0.04, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.eegmamba.s2026.sup": (0.169, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.eegmamba.sujetos3.inf": (0.035, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.eegmamba.sujetos3.sup": (0.16, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.eegmamba.jer3.inf": (0.034, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.eegmamba.jer3.sup": (0.161, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.eegmamba.s2027.inf": (0.04, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.eegmamba.s2027.sup": (0.162, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.eegmamba.s2028.inf": (0.02, 0.0005, "03 sec. 3.4"),
    "trc.caida_protocolo.eegmamba.s2028.sup": (0.154, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.eegnet.s2026.inf": (-0.084, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.eegnet.s2026.sup": (0.21, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.eegnet.sujetos3.inf": (-0.086, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.eegnet.sujetos3.sup": (0.211, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.eegnet.jer3.inf": (-0.085, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.eegnet.jer3.sup": (0.212, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.eegnet.s2027.inf": (-0.093, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.eegnet.s2027.sup": (0.208, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.eegnet.s2028.inf": (-0.084, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.eegnet.s2028.sup": (0.217, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.conformer.s2026.inf": (-0.155, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.conformer.s2026.sup": (0.175, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.conformer.sujetos3.inf": (-0.133, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.conformer.sujetos3.sup": (0.183, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.conformer.jer3.inf": (-0.134, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.conformer.jer3.sup": (0.183, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.conformer.s2027.inf": (-0.136, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.conformer.s2027.sup": (0.171, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.conformer.s2028.inf": (-0.112, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.conformer.s2028.sup": (0.206, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.labram.s2026.inf": (-0.091, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.labram.s2026.sup": (0.19, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.labram.sujetos3.inf": (-0.081, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.labram.sujetos3.sup": (0.185, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.labram.jer3.inf": (-0.082, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.labram.jer3.sup": (0.187, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.labram.s2027.inf": (-0.072, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.labram.s2027.sup": (0.179, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.labram.s2028.inf": (-0.082, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.labram.s2028.sup": (0.193, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.eegmamba.s2026.inf": (-0.063, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.eegmamba.s2026.sup": (0.173, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.eegmamba.sujetos3.inf": (-0.076, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.eegmamba.sujetos3.sup": (0.158, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.eegmamba.jer3.inf": (-0.082, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.eegmamba.jer3.sup": (0.159, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.eegmamba.s2027.inf": (-0.054, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.eegmamba.s2027.sup": (0.167, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.eegmamba.s2028.inf": (-0.116, 0.0005, "03 sec. 3.4"),
    "trc.protocolo_menos_persona.eegmamba.s2028.sup": (0.133, 0.0005, "03 sec. 3.4"),
    "trc.idp.eegnet.s2026.inf": (0.061, 0.0005, "03 sec. 3.4"),
    "trc.idp.eegnet.s2026.sup": (0.249, 0.0005, "03 sec. 3.4"),
    "trc.idp.eegnet.sujetos3.inf": (0.054, 0.0005, "03 sec. 3.4"),
    "trc.idp.eegnet.sujetos3.sup": (0.243, 0.0005, "03 sec. 3.4"),
    "trc.idp.eegnet.jer3.inf": (0.052, 0.0005, "03 sec. 3.4"),
    "trc.idp.eegnet.jer3.sup": (0.243, 0.0005, "03 sec. 3.4"),
    "trc.idp.eegnet.s2027.inf": (0.046, 0.0005, "03 sec. 3.4"),
    "trc.idp.eegnet.s2027.sup": (0.241, 0.0005, "03 sec. 3.4"),
    "trc.idp.eegnet.s2028.inf": (0.051, 0.0005, "03 sec. 3.4"),
    "trc.idp.eegnet.s2028.sup": (0.24, 0.0005, "03 sec. 3.4"),
    "trc.idp.conformer.s2026.inf": (0.022, 0.0005, "03 sec. 3.4"),
    "trc.idp.conformer.s2026.sup": (0.24, 0.0005, "03 sec. 3.4"),
    "trc.idp.conformer.sujetos3.inf": (0.035, 0.0005, "03 sec. 3.4"),
    "trc.idp.conformer.sujetos3.sup": (0.241, 0.0005, "03 sec. 3.4"),
    "trc.idp.conformer.jer3.inf": (0.034, 0.0005, "03 sec. 3.4"),
    "trc.idp.conformer.jer3.sup": (0.243, 0.0005, "03 sec. 3.4"),
    "trc.idp.conformer.s2027.inf": (0.021, 0.0005, "03 sec. 3.4"),
    "trc.idp.conformer.s2027.sup": (0.23, 0.0005, "03 sec. 3.4"),
    "trc.idp.conformer.s2028.inf": (0.056, 0.0005, "03 sec. 3.4"),
    "trc.idp.conformer.s2028.sup": (0.257, 0.0005, "03 sec. 3.4"),
    "trc.idp.labram.s2026.inf": (0.058, 0.0005, "03 sec. 3.4"),
    "trc.idp.labram.s2026.sup": (0.242, 0.0005, "03 sec. 3.4"),
    "trc.idp.labram.sujetos3.inf": (0.074, 0.0005, "03 sec. 3.4"),
    "trc.idp.labram.sujetos3.sup": (0.248, 0.0005, "03 sec. 3.4"),
    "trc.idp.labram.jer3.inf": (0.073, 0.0005, "03 sec. 3.4"),
    "trc.idp.labram.jer3.sup": (0.249, 0.0005, "03 sec. 3.4"),
    "trc.idp.labram.s2027.inf": (0.082, 0.0005, "03 sec. 3.4"),
    "trc.idp.labram.s2027.sup": (0.252, 0.0005, "03 sec. 3.4"),
    "trc.idp.labram.s2028.inf": (0.076, 0.0005, "03 sec. 3.4"),
    "trc.idp.labram.s2028.sup": (0.254, 0.0005, "03 sec. 3.4"),
    "trc.idp.eegmamba.s2026.inf": (0.061, 0.0005, "03 sec. 3.4"),
    "trc.idp.eegmamba.s2026.sup": (0.223, 0.0005, "03 sec. 3.4"),
    "trc.idp.eegmamba.sujetos3.inf": (0.055, 0.0005, "03 sec. 3.4"),
    "trc.idp.eegmamba.sujetos3.sup": (0.214, 0.0005, "03 sec. 3.4"),
    "trc.idp.eegmamba.jer3.inf": (0.052, 0.0005, "03 sec. 3.4"),
    "trc.idp.eegmamba.jer3.sup": (0.215, 0.0005, "03 sec. 3.4"),
    "trc.idp.eegmamba.s2027.inf": (0.061, 0.0005, "03 sec. 3.4"),
    "trc.idp.eegmamba.s2027.sup": (0.217, 0.0005, "03 sec. 3.4"),
    "trc.idp.eegmamba.s2028.inf": (0.032, 0.0005, "03 sec. 3.4"),
    "trc.idp.eegmamba.s2028.sup": (0.207, 0.0005, "03 sec. 3.4"),
    "pol.eegnet.source": (0.62, 0.0005, "03 sec. 3.5"),
    "pol.eegnet.per_trial": (0.604, 0.0005, "03 sec. 3.5"),
    "pol.eegnet.per_subject": (0.603, 0.0005, "03 sec. 3.5"),
    "pol.eegnet.amplitud": (0.017, 0.0005, "03 sec. 3.5"),
    "pol.conformer.source": (0.614, 0.0005, "03 sec. 3.5"),
    "pol.conformer.per_trial": (0.578, 0.0005, "03 sec. 3.5"),
    "pol.conformer.per_subject": (0.593, 0.0005, "03 sec. 3.5"),
    "pol.conformer.amplitud": (0.036, 0.0005, "03 sec. 3.5"),
    "pol.labram.source": (0.61, 0.0005, "03 sec. 3.5"),
    "pol.labram.per_trial": (0.596, 0.0005, "03 sec. 3.5"),
    "pol.labram.per_subject": (0.606, 0.0005, "03 sec. 3.5"),
    "pol.labram.amplitud": (0.014, 0.0005, "03 sec. 3.5"),
    "pol.eegmamba.source": (0.596, 0.0005, "03 sec. 3.5"),
    "pol.eegmamba.per_trial": (0.592, 0.0005, "03 sec. 3.5"),
    "pol.eegmamba.per_subject": (0.604, 0.0005, "03 sec. 3.5"),
    "pol.eegmamba.amplitud": (0.012, 0.0005, "03 sec. 3.5"),
    "polc.eegnet.per_trial.dif": (0.0158, 5e-05, "03 sec. 3.5"),
    "polc.eegnet.per_trial.ic_inf": (0.007, 5e-05, "03 sec. 3.5"),
    "polc.eegnet.per_trial.ic_sup": (0.0243, 5e-05, "03 sec. 3.5"),
    "polc.eegnet.per_trial.p": (0.0015, 5e-05, "03 sec. 3.5"),
    "polc.eegnet.per_trial.p_holm": (0.0105, 5e-05, "03 sec. 3.5"),
    "polc.eegnet.per_trial.r": (0.44, 0.005, "03 sec. 3.5"),
    "polc.eegnet.per_subject.dif": (0.0172, 5e-05, "03 sec. 3.5"),
    "polc.eegnet.per_subject.ic_inf": (0.0038, 5e-05, "03 sec. 3.5"),
    "polc.eegnet.per_subject.ic_sup": (0.0315, 5e-05, "03 sec. 3.5"),
    "polc.eegnet.per_subject.p": (0.061, 5e-05, "03 sec. 3.5"),
    "polc.eegnet.per_subject.p_holm": (0.2439, 5e-05, "03 sec. 3.5"),
    "polc.eegnet.per_subject.r": (0.26, 0.005, "03 sec. 3.5"),
    "polc.conformer.per_trial.dif": (0.0363, 5e-05, "03 sec. 3.5"),
    "polc.conformer.per_trial.ic_inf": (0.0194, 5e-05, "03 sec. 3.5"),
    "polc.conformer.per_trial.ic_sup": (0.0525, 5e-05, "03 sec. 3.5"),
    "polc.conformer.per_trial.p": (0.0001, 5e-05, "03 sec. 3.5"),
    "polc.conformer.per_trial.p_holm": (0.0011, 5e-05, "03 sec. 3.5"),
    "polc.conformer.per_trial.r": (0.54, 0.005, "03 sec. 3.5"),
    "polc.conformer.per_subject.dif": (0.0212, 5e-05, "03 sec. 3.5"),
    "polc.conformer.per_subject.ic_inf": (0.0057, 5e-05, "03 sec. 3.5"),
    "polc.conformer.per_subject.ic_sup": (0.0364, 5e-05, "03 sec. 3.5"),
    "polc.conformer.per_subject.p": (0.0356, 5e-05, "03 sec. 3.5"),
    "polc.conformer.per_subject.p_holm": (0.1778, 5e-05, "03 sec. 3.5"),
    "polc.conformer.per_subject.r": (0.29, 0.005, "03 sec. 3.5"),
    "polc.labram.per_trial.dif": (0.0144, 5e-05, "03 sec. 3.5"),
    "polc.labram.per_trial.ic_inf": (0.0036, 5e-05, "03 sec. 3.5"),
    "polc.labram.per_trial.ic_sup": (0.0252, 5e-05, "03 sec. 3.5"),
    "polc.labram.per_trial.p": (0.0152, 5e-05, "03 sec. 3.5"),
    "polc.labram.per_trial.p_holm": (0.0914, 5e-05, "03 sec. 3.5"),
    "polc.labram.per_trial.r": (0.34, 0.005, "03 sec. 3.5"),
    "polc.labram.per_subject.dif": (0.0042, 5e-05, "03 sec. 3.5"),
    "polc.labram.per_subject.ic_inf": (-0.008, 5e-05, "03 sec. 3.5"),
    "polc.labram.per_subject.ic_sup": (0.0168, 5e-05, "03 sec. 3.5"),
    "polc.labram.per_subject.p": (0.7178, 5e-05, "03 sec. 3.5"),
    "polc.labram.per_subject.p_holm": (1.0, 5e-05, "03 sec. 3.5"),
    "polc.labram.per_subject.r": (0.05, 0.005, "03 sec. 3.5"),
    "polc.eegmamba.per_trial.dif": (0.0041, 5e-05, "03 sec. 3.5"),
    "polc.eegmamba.per_trial.ic_inf": (-0.0047, 5e-05, "03 sec. 3.5"),
    "polc.eegmamba.per_trial.ic_sup": (0.0136, 5e-05, "03 sec. 3.5"),
    "polc.eegmamba.per_trial.p": (0.7098, 5e-05, "03 sec. 3.5"),
    "polc.eegmamba.per_trial.p_holm": (1.0, 5e-05, "03 sec. 3.5"),
    "polc.eegmamba.per_trial.r": (0.05, 0.005, "03 sec. 3.5"),
    "polc.eegmamba.per_subject.dif": (-0.0075, 5e-05, "03 sec. 3.5"),
    "polc.eegmamba.per_subject.ic_inf": (-0.0206, 5e-05, "03 sec. 3.5"),
    "polc.eegmamba.per_subject.ic_sup": (0.0051, 5e-05, "03 sec. 3.5"),
    "polc.eegmamba.per_subject.p": (0.2559, 5e-05, "03 sec. 3.5"),
    "polc.eegmamba.per_subject.p_holm": (0.7676, 5e-05, "03 sec. 3.5"),
    "polc.eegmamba.per_subject.r": (0.16, 0.005, "03 sec. 3.5"),
    "eat.transferido.eegnet.base": (0.626, 0.0005, "03 sec. 4.1"),
    "eat.transferido.eegnet.ea": (0.632, 0.0005, "03 sec. 4.1"),
    "eat.transferido.eegnet.dif": (0.006, 0.0005, "03 sec. 4.1"),
    "eat.transferido.eegnet.dt_base": (0.005, 0.0005, "03 sec. 4.1"),
    "eat.transferido.eegnet.cociente": (1.1, 0.05, "03 sec. 4.1"),
    "eat.transferido.eegnet.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.transferido.conformer.base": (0.611, 0.0005, "03 sec. 4.1"),
    "eat.transferido.conformer.ea": (0.622, 0.0005, "03 sec. 4.1"),
    "eat.transferido.conformer.dif": (0.011, 0.0005, "03 sec. 4.1"),
    "eat.transferido.conformer.dt_base": (0.003, 0.0005, "03 sec. 4.1"),
    "eat.transferido.conformer.cociente": (3.2, 0.05, "03 sec. 4.1"),
    "eat.transferido.conformer.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.transferido.labram.base": (0.613, 0.0005, "03 sec. 4.1"),
    "eat.transferido.labram.ea": (0.623, 0.0005, "03 sec. 4.1"),
    "eat.transferido.labram.dif": (0.009, 0.0005, "03 sec. 4.1"),
    "eat.transferido.labram.dt_base": (0.007, 0.0005, "03 sec. 4.1"),
    "eat.transferido.labram.cociente": (1.3, 0.05, "03 sec. 4.1"),
    "eat.transferido.labram.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.transferido.eegmamba.base": (0.589, 0.0005, "03 sec. 4.1"),
    "eat.transferido.eegmamba.ea": (0.598, 0.0005, "03 sec. 4.1"),
    "eat.transferido.eegmamba.dif": (0.009, 0.0005, "03 sec. 4.1"),
    "eat.transferido.eegmamba.dt_base": (0.006, 0.0005, "03 sec. 4.1"),
    "eat.transferido.eegmamba.cociente": (1.4, 0.05, "03 sec. 4.1"),
    "eat.transferido.eegmamba.fuera": (0.0, 0.5, "03 sec. 4.1"),
    "eat.nativo.eegnet.base": (0.808, 0.0005, "03 sec. 4.1"),
    "eat.nativo.eegnet.ea": (0.813, 0.0005, "03 sec. 4.1"),
    "eat.nativo.eegnet.dif": (0.005, 0.0005, "03 sec. 4.1"),
    "eat.nativo.eegnet.dt_base": (0.004, 0.0005, "03 sec. 4.1"),
    "eat.nativo.eegnet.cociente": (1.1, 0.05, "03 sec. 4.1"),
    "eat.nativo.eegnet.fuera": (0.0, 0.5, "03 sec. 4.1"),
    "eat.nativo.conformer.base": (0.8, 0.0005, "03 sec. 4.1"),
    "eat.nativo.conformer.ea": (0.798, 0.0005, "03 sec. 4.1"),
    "eat.nativo.conformer.dif": (-0.003, 0.0005, "03 sec. 4.1"),
    "eat.nativo.conformer.dt_base": (0.002, 0.0005, "03 sec. 4.1"),
    "eat.nativo.conformer.cociente": (-1.7, 0.05, "03 sec. 4.1"),
    "eat.nativo.conformer.fuera": (0.0, 0.5, "03 sec. 4.1"),
    "eat.nativo.labram.base": (0.814, 0.0005, "03 sec. 4.1"),
    "eat.nativo.labram.ea": (0.812, 0.0005, "03 sec. 4.1"),
    "eat.nativo.labram.dif": (-0.002, 0.0005, "03 sec. 4.1"),
    "eat.nativo.labram.dt_base": (0.008, 0.0005, "03 sec. 4.1"),
    "eat.nativo.labram.cociente": (-0.2, 0.05, "03 sec. 4.1"),
    "eat.nativo.labram.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.nativo.eegmamba.base": (0.762, 0.0005, "03 sec. 4.1"),
    "eat.nativo.eegmamba.ea": (0.758, 0.0005, "03 sec. 4.1"),
    "eat.nativo.eegmamba.dif": (-0.004, 0.0005, "03 sec. 4.1"),
    "eat.nativo.eegmamba.dt_base": (0.008, 0.0005, "03 sec. 4.1"),
    "eat.nativo.eegmamba.cociente": (-0.5, 0.05, "03 sec. 4.1"),
    "eat.nativo.eegmamba.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.eegnet.base": (0.182, 0.0005, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.eegnet.ea": (0.181, 0.0005, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.eegnet.dif": (-0.001, 0.0005, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.eegnet.dt_base": (0.009, 0.0005, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.eegnet.cociente": (-0.1, 0.05, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.eegnet.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.conformer.base": (0.19, 0.0005, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.conformer.ea": (0.175, 0.0005, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.conformer.dif": (-0.014, 0.0005, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.conformer.dt_base": (0.005, 0.0005, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.conformer.cociente": (-2.8, 0.05, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.conformer.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.labram.base": (0.201, 0.0005, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.labram.ea": (0.19, 0.0005, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.labram.dif": (-0.011, 0.0005, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.labram.dt_base": (0.014, 0.0005, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.labram.cociente": (-0.8, 0.05, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.labram.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.eegmamba.base": (0.173, 0.0005, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.eegmamba.ea": (0.16, 0.0005, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.eegmamba.dif": (-0.013, 0.0005, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.eegmamba.dt_base": (0.007, 0.0005, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.eegmamba.cociente": (-1.9, 0.05, "03 sec. 4.1"),
    "eat.dif_nativo_menos_transferido.eegmamba.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.vistos44.eegnet.base": (0.802, 0.0005, "03 sec. 4.1"),
    "eat.vistos44.eegnet.ea": (0.823, 0.0005, "03 sec. 4.1"),
    "eat.vistos44.eegnet.dif": (0.022, 0.0005, "03 sec. 4.1"),
    "eat.vistos44.eegnet.dt_base": (0.009, 0.0005, "03 sec. 4.1"),
    "eat.vistos44.eegnet.cociente": (2.3, 0.05, "03 sec. 4.1"),
    "eat.vistos44.eegnet.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.vistos44.conformer.base": (0.814, 0.0005, "03 sec. 4.1"),
    "eat.vistos44.conformer.ea": (0.841, 0.0005, "03 sec. 4.1"),
    "eat.vistos44.conformer.dif": (0.027, 0.0005, "03 sec. 4.1"),
    "eat.vistos44.conformer.dt_base": (0.008, 0.0005, "03 sec. 4.1"),
    "eat.vistos44.conformer.cociente": (3.2, 0.05, "03 sec. 4.1"),
    "eat.vistos44.conformer.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.vistos44.labram.base": (0.813, 0.0005, "03 sec. 4.1"),
    "eat.vistos44.labram.ea": (0.84, 0.0005, "03 sec. 4.1"),
    "eat.vistos44.labram.dif": (0.027, 0.0005, "03 sec. 4.1"),
    "eat.vistos44.labram.dt_base": (0.005, 0.0005, "03 sec. 4.1"),
    "eat.vistos44.labram.cociente": (5.2, 0.05, "03 sec. 4.1"),
    "eat.vistos44.labram.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.vistos44.eegmamba.base": (0.753, 0.0005, "03 sec. 4.1"),
    "eat.vistos44.eegmamba.ea": (0.803, 0.0005, "03 sec. 4.1"),
    "eat.vistos44.eegmamba.dif": (0.05, 0.0005, "03 sec. 4.1"),
    "eat.vistos44.eegmamba.dt_base": (0.014, 0.0005, "03 sec. 4.1"),
    "eat.vistos44.eegmamba.cociente": (3.7, 0.05, "03 sec. 4.1"),
    "eat.vistos44.eegmamba.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.persona.eegnet.base": (0.742, 0.0005, "03 sec. 4.1"),
    "eat.persona.eegnet.ea": (0.799, 0.0005, "03 sec. 4.1"),
    "eat.persona.eegnet.dif": (0.057, 0.0005, "03 sec. 4.1"),
    "eat.persona.eegnet.dt_base": (0.008, 0.0005, "03 sec. 4.1"),
    "eat.persona.eegnet.cociente": (7.1, 0.05, "03 sec. 4.1"),
    "eat.persona.eegnet.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.persona.conformer.base": (0.718, 0.0005, "03 sec. 4.1"),
    "eat.persona.conformer.ea": (0.763, 0.0005, "03 sec. 4.1"),
    "eat.persona.conformer.dif": (0.045, 0.0005, "03 sec. 4.1"),
    "eat.persona.conformer.dt_base": (0.011, 0.0005, "03 sec. 4.1"),
    "eat.persona.conformer.cociente": (4.1, 0.05, "03 sec. 4.1"),
    "eat.persona.conformer.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.persona.labram.base": (0.745, 0.0005, "03 sec. 4.1"),
    "eat.persona.labram.ea": (0.748, 0.0005, "03 sec. 4.1"),
    "eat.persona.labram.dif": (0.003, 0.0005, "03 sec. 4.1"),
    "eat.persona.labram.dt_base": (0.006, 0.0005, "03 sec. 4.1"),
    "eat.persona.labram.cociente": (0.5, 0.05, "03 sec. 4.1"),
    "eat.persona.labram.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.persona.eegmamba.base": (0.7, 0.0005, "03 sec. 4.1"),
    "eat.persona.eegmamba.ea": (0.759, 0.0005, "03 sec. 4.1"),
    "eat.persona.eegmamba.dif": (0.06, 0.0005, "03 sec. 4.1"),
    "eat.persona.eegmamba.dt_base": (0.008, 0.0005, "03 sec. 4.1"),
    "eat.persona.eegmamba.cociente": (7.5, 0.05, "03 sec. 4.1"),
    "eat.persona.eegmamba.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.wbcic44.eegnet.base": (0.621, 0.0005, "03 sec. 4.1"),
    "eat.wbcic44.eegnet.ea": (0.61, 0.0005, "03 sec. 4.1"),
    "eat.wbcic44.eegnet.dif": (-0.011, 0.0005, "03 sec. 4.1"),
    "eat.wbcic44.eegnet.dt_base": (0.006, 0.0005, "03 sec. 4.1"),
    "eat.wbcic44.eegnet.cociente": (-1.8, 0.05, "03 sec. 4.1"),
    "eat.wbcic44.eegnet.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.wbcic44.conformer.base": (0.618, 0.0005, "03 sec. 4.1"),
    "eat.wbcic44.conformer.ea": (0.63, 0.0005, "03 sec. 4.1"),
    "eat.wbcic44.conformer.dif": (0.012, 0.0005, "03 sec. 4.1"),
    "eat.wbcic44.conformer.dt_base": (0.003, 0.0005, "03 sec. 4.1"),
    "eat.wbcic44.conformer.cociente": (3.6, 0.05, "03 sec. 4.1"),
    "eat.wbcic44.conformer.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.wbcic44.labram.base": (0.628, 0.0005, "03 sec. 4.1"),
    "eat.wbcic44.labram.ea": (0.595, 0.0005, "03 sec. 4.1"),
    "eat.wbcic44.labram.dif": (-0.033, 0.0005, "03 sec. 4.1"),
    "eat.wbcic44.labram.dt_base": (0.011, 0.0005, "03 sec. 4.1"),
    "eat.wbcic44.labram.cociente": (-3.0, 0.05, "03 sec. 4.1"),
    "eat.wbcic44.labram.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.wbcic44.eegmamba.base": (0.598, 0.0005, "03 sec. 4.1"),
    "eat.wbcic44.eegmamba.ea": (0.6, 0.0005, "03 sec. 4.1"),
    "eat.wbcic44.eegmamba.dif": (0.002, 0.0005, "03 sec. 4.1"),
    "eat.wbcic44.eegmamba.dt_base": (0.003, 0.0005, "03 sec. 4.1"),
    "eat.wbcic44.eegmamba.cociente": (0.7, 0.05, "03 sec. 4.1"),
    "eat.wbcic44.eegmamba.fuera": (0.0, 0.5, "03 sec. 4.1"),
    "eat.caida_persona.eegnet.base": (0.059, 0.0005, "03 sec. 4.1"),
    "eat.caida_persona.eegnet.ea": (0.024, 0.0005, "03 sec. 4.1"),
    "eat.caida_persona.eegnet.dif": (-0.035, 0.0005, "03 sec. 4.1"),
    "eat.caida_persona.eegnet.dt_base": (0.005, 0.0005, "03 sec. 4.1"),
    "eat.caida_persona.eegnet.cociente": (-7.6, 0.05, "03 sec. 4.1"),
    "eat.caida_persona.eegnet.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.caida_persona.conformer.base": (0.096, 0.0005, "03 sec. 4.1"),
    "eat.caida_persona.conformer.ea": (0.078, 0.0005, "03 sec. 4.1"),
    "eat.caida_persona.conformer.dif": (-0.019, 0.0005, "03 sec. 4.1"),
    "eat.caida_persona.conformer.dt_base": (0.008, 0.0005, "03 sec. 4.1"),
    "eat.caida_persona.conformer.cociente": (-2.2, 0.05, "03 sec. 4.1"),
    "eat.caida_persona.conformer.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.caida_persona.labram.base": (0.068, 0.0005, "03 sec. 4.1"),
    "eat.caida_persona.labram.ea": (0.092, 0.0005, "03 sec. 4.1"),
    "eat.caida_persona.labram.dif": (0.024, 0.0005, "03 sec. 4.1"),
    "eat.caida_persona.labram.dt_base": (0.004, 0.0005, "03 sec. 4.1"),
    "eat.caida_persona.labram.cociente": (6.3, 0.05, "03 sec. 4.1"),
    "eat.caida_persona.labram.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.caida_persona.eegmamba.base": (0.054, 0.0005, "03 sec. 4.1"),
    "eat.caida_persona.eegmamba.ea": (0.044, 0.0005, "03 sec. 4.1"),
    "eat.caida_persona.eegmamba.dif": (-0.01, 0.0005, "03 sec. 4.1"),
    "eat.caida_persona.eegmamba.dt_base": (0.02, 0.0005, "03 sec. 4.1"),
    "eat.caida_persona.eegmamba.cociente": (-0.5, 0.05, "03 sec. 4.1"),
    "eat.caida_persona.eegmamba.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.caida_protocolo.eegnet.base": (0.121, 0.0005, "03 sec. 4.1"),
    "eat.caida_protocolo.eegnet.ea": (0.189, 0.0005, "03 sec. 4.1"),
    "eat.caida_protocolo.eegnet.dif": (0.068, 0.0005, "03 sec. 4.1"),
    "eat.caida_protocolo.eegnet.dt_base": (0.006, 0.0005, "03 sec. 4.1"),
    "eat.caida_protocolo.eegnet.cociente": (12.2, 0.05, "03 sec. 4.1"),
    "eat.caida_protocolo.eegnet.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.caida_protocolo.conformer.base": (0.1, 0.0005, "03 sec. 4.1"),
    "eat.caida_protocolo.conformer.ea": (0.133, 0.0005, "03 sec. 4.1"),
    "eat.caida_protocolo.conformer.dif": (0.033, 0.0005, "03 sec. 4.1"),
    "eat.caida_protocolo.conformer.dt_base": (0.014, 0.0005, "03 sec. 4.1"),
    "eat.caida_protocolo.conformer.cociente": (2.3, 0.05, "03 sec. 4.1"),
    "eat.caida_protocolo.conformer.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.caida_protocolo.labram.base": (0.117, 0.0005, "03 sec. 4.1"),
    "eat.caida_protocolo.labram.ea": (0.153, 0.0005, "03 sec. 4.1"),
    "eat.caida_protocolo.labram.dif": (0.036, 0.0005, "03 sec. 4.1"),
    "eat.caida_protocolo.labram.dt_base": (0.006, 0.0005, "03 sec. 4.1"),
    "eat.caida_protocolo.labram.cociente": (6.3, 0.05, "03 sec. 4.1"),
    "eat.caida_protocolo.labram.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.caida_protocolo.eegmamba.base": (0.102, 0.0005, "03 sec. 4.1"),
    "eat.caida_protocolo.eegmamba.ea": (0.16, 0.0005, "03 sec. 4.1"),
    "eat.caida_protocolo.eegmamba.dif": (0.058, 0.0005, "03 sec. 4.1"),
    "eat.caida_protocolo.eegmamba.dt_base": (0.01, 0.0005, "03 sec. 4.1"),
    "eat.caida_protocolo.eegmamba.cociente": (5.8, 0.05, "03 sec. 4.1"),
    "eat.caida_protocolo.eegmamba.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.eegnet.base": (0.062, 0.0005, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.eegnet.ea": (0.164, 0.0005, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.eegnet.dif": (0.103, 0.0005, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.eegnet.dt_base": (0.004, 0.0005, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.eegnet.cociente": (28.3, 0.05, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.eegnet.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.conformer.base": (0.003, 0.0005, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.conformer.ea": (0.055, 0.0005, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.conformer.dif": (0.052, 0.0005, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.conformer.dt_base": (0.021, 0.0005, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.conformer.cociente": (2.5, 0.05, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.conformer.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.labram.base": (0.049, 0.0005, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.labram.ea": (0.061, 0.0005, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.labram.dif": (0.012, 0.0005, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.labram.dt_base": (0.003, 0.0005, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.labram.cociente": (3.9, 0.05, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.labram.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.eegmamba.base": (0.048, 0.0005, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.eegmamba.ea": (0.116, 0.0005, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.eegmamba.dif": (0.068, 0.0005, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.eegmamba.dt_base": (0.029, 0.0005, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.eegmamba.cociente": (2.3, 0.05, "03 sec. 4.1"),
    "eat.protocolo_menos_persona.eegmamba.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.idp.eegnet.base": (0.163, 0.0005, "03 sec. 4.1"),
    "eat.idp.eegnet.ea": (0.236, 0.0005, "03 sec. 4.1"),
    "eat.idp.eegnet.dif": (0.073, 0.0005, "03 sec. 4.1"),
    "eat.idp.eegnet.dt_base": (0.006, 0.0005, "03 sec. 4.1"),
    "eat.idp.eegnet.cociente": (11.3, 0.05, "03 sec. 4.1"),
    "eat.idp.eegnet.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.idp.conformer.base": (0.139, 0.0005, "03 sec. 4.1"),
    "eat.idp.conformer.ea": (0.174, 0.0005, "03 sec. 4.1"),
    "eat.idp.conformer.dif": (0.035, 0.0005, "03 sec. 4.1"),
    "eat.idp.conformer.dt_base": (0.018, 0.0005, "03 sec. 4.1"),
    "eat.idp.conformer.cociente": (2.0, 0.05, "03 sec. 4.1"),
    "eat.idp.conformer.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.idp.labram.base": (0.157, 0.0005, "03 sec. 4.1"),
    "eat.idp.labram.ea": (0.205, 0.0005, "03 sec. 4.1"),
    "eat.idp.labram.dif": (0.048, 0.0005, "03 sec. 4.1"),
    "eat.idp.labram.dt_base": (0.009, 0.0005, "03 sec. 4.1"),
    "eat.idp.labram.cociente": (5.4, 0.05, "03 sec. 4.1"),
    "eat.idp.labram.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eat.idp.eegmamba.base": (0.146, 0.0005, "03 sec. 4.1"),
    "eat.idp.eegmamba.ea": (0.21, 0.0005, "03 sec. 4.1"),
    "eat.idp.eegmamba.dif": (0.065, 0.0005, "03 sec. 4.1"),
    "eat.idp.eegmamba.dt_base": (0.013, 0.0005, "03 sec. 4.1"),
    "eat.idp.eegmamba.cociente": (5.0, 0.05, "03 sec. 4.1"),
    "eat.idp.eegmamba.fuera": (1.0, 0.5, "03 sec. 4.1"),
    "eatc.eegnet.cruzado_51.base": (0.626, 0.0005, "03 sec. 4.2"),
    "eatc.eegnet.cruzado_51.ea": (0.632, 0.0005, "03 sec. 4.2"),
    "eatc.eegnet.cruzado_51.dif": (0.0059, 5e-05, "03 sec. 4.2"),
    "eatc.eegnet.cruzado_51.ic_inf": (-0.0176, 5e-05, "03 sec. 4.2"),
    "eatc.eegnet.cruzado_51.ic_sup": (0.0284, 5e-05, "03 sec. 4.2"),
    "eatc.eegnet.cruzado_51.p": (0.5532, 5e-05, "03 sec. 4.2"),
    "eatc.eegnet.cruzado_51.p_holm": (1.0, 5e-05, "03 sec. 4.2"),
    "eatc.eegnet.cruzado_51.r": (0.08, 0.005, "03 sec. 4.2"),
    "eatc.eegnet.cruzado_51.mejoran": (27.0, 0.5, "03 sec. 4.2"),
    "eatc.eegnet.cruzado_51.empeoran": (20.0, 0.5, "03 sec. 4.2"),
    "eatc.eegnet.h44_wbcic_51.base": (0.621, 0.0005, "03 sec. 4.2"),
    "eatc.eegnet.h44_wbcic_51.ea": (0.61, 0.0005, "03 sec. 4.2"),
    "eatc.eegnet.h44_wbcic_51.dif": (-0.0108, 5e-05, "03 sec. 4.2"),
    "eatc.eegnet.h44_wbcic_51.ic_inf": (-0.0343, 5e-05, "03 sec. 4.2"),
    "eatc.eegnet.h44_wbcic_51.ic_sup": (0.0123, 5e-05, "03 sec. 4.2"),
    "eatc.eegnet.h44_wbcic_51.p": (0.3673, 5e-05, "03 sec. 4.2"),
    "eatc.eegnet.h44_wbcic_51.p_holm": (1.0, 5e-05, "03 sec. 4.2"),
    "eatc.eegnet.h44_wbcic_51.r": (0.13, 0.005, "03 sec. 4.2"),
    "eatc.eegnet.h44_wbcic_51.mejoran": (21.0, 0.5, "03 sec. 4.2"),
    "eatc.eegnet.h44_wbcic_51.empeoran": (26.0, 0.5, "03 sec. 4.2"),
    "eatc.eegnet.h44_openbmi_10.base": (0.742, 0.0005, "03 sec. 4.2"),
    "eatc.eegnet.h44_openbmi_10.ea": (0.799, 0.0005, "03 sec. 4.2"),
    "eatc.eegnet.h44_openbmi_10.dif": (0.0567, 5e-05, "03 sec. 4.2"),
    "eatc.eegnet.h44_openbmi_10.ic_inf": (0.0222, 5e-05, "03 sec. 4.2"),
    "eatc.eegnet.h44_openbmi_10.ic_sup": (0.0932, 5e-05, "03 sec. 4.2"),
    "eatc.eegnet.h44_openbmi_10.p": (0.0156, 5e-05, "03 sec. 4.2"),
    "eatc.eegnet.h44_openbmi_10.p_holm": (0.1719, 5e-05, "03 sec. 4.2"),
    "eatc.eegnet.h44_openbmi_10.r": (0.76, 0.005, "03 sec. 4.2"),
    "eatc.eegnet.h44_openbmi_10.mejoran": (8.0, 0.5, "03 sec. 4.2"),
    "eatc.eegnet.h44_openbmi_10.empeoran": (1.0, 0.5, "03 sec. 4.2"),
    "eatc.conformer.cruzado_51.base": (0.611, 0.0005, "03 sec. 4.2"),
    "eatc.conformer.cruzado_51.ea": (0.622, 0.0005, "03 sec. 4.2"),
    "eatc.conformer.cruzado_51.dif": (0.0113, 5e-05, "03 sec. 4.2"),
    "eatc.conformer.cruzado_51.ic_inf": (-0.0123, 5e-05, "03 sec. 4.2"),
    "eatc.conformer.cruzado_51.ic_sup": (0.0358, 5e-05, "03 sec. 4.2"),
    "eatc.conformer.cruzado_51.p": (0.3715, 5e-05, "03 sec. 4.2"),
    "eatc.conformer.cruzado_51.p_holm": (1.0, 5e-05, "03 sec. 4.2"),
    "eatc.conformer.cruzado_51.r": (0.13, 0.005, "03 sec. 4.2"),
    "eatc.conformer.cruzado_51.mejoran": (23.0, 0.5, "03 sec. 4.2"),
    "eatc.conformer.cruzado_51.empeoran": (19.0, 0.5, "03 sec. 4.2"),
    "eatc.conformer.h44_wbcic_51.base": (0.618, 0.0005, "03 sec. 4.2"),
    "eatc.conformer.h44_wbcic_51.ea": (0.63, 0.0005, "03 sec. 4.2"),
    "eatc.conformer.h44_wbcic_51.dif": (0.0123, 5e-05, "03 sec. 4.2"),
    "eatc.conformer.h44_wbcic_51.ic_inf": (-0.0083, 5e-05, "03 sec. 4.2"),
    "eatc.conformer.h44_wbcic_51.ic_sup": (0.0324, 5e-05, "03 sec. 4.2"),
    "eatc.conformer.h44_wbcic_51.p": (0.2106, 5e-05, "03 sec. 4.2"),
    "eatc.conformer.h44_wbcic_51.p_holm": (1.0, 5e-05, "03 sec. 4.2"),
    "eatc.conformer.h44_wbcic_51.r": (0.18, 0.005, "03 sec. 4.2"),
    "eatc.conformer.h44_wbcic_51.mejoran": (27.0, 0.5, "03 sec. 4.2"),
    "eatc.conformer.h44_wbcic_51.empeoran": (19.0, 0.5, "03 sec. 4.2"),
    "eatc.conformer.h44_openbmi_10.base": (0.718, 0.0005, "03 sec. 4.2"),
    "eatc.conformer.h44_openbmi_10.ea": (0.763, 0.0005, "03 sec. 4.2"),
    "eatc.conformer.h44_openbmi_10.dif": (0.0455, 5e-05, "03 sec. 4.2"),
    "eatc.conformer.h44_openbmi_10.ic_inf": (0.0135, 5e-05, "03 sec. 4.2"),
    "eatc.conformer.h44_openbmi_10.ic_sup": (0.078, 5e-05, "03 sec. 4.2"),
    "eatc.conformer.h44_openbmi_10.p": (0.041, 5e-05, "03 sec. 4.2"),
    "eatc.conformer.h44_openbmi_10.p_holm": (0.3691, 5e-05, "03 sec. 4.2"),
    "eatc.conformer.h44_openbmi_10.r": (0.65, 0.005, "03 sec. 4.2"),
    "eatc.conformer.h44_openbmi_10.mejoran": (7.0, 0.5, "03 sec. 4.2"),
    "eatc.conformer.h44_openbmi_10.empeoran": (3.0, 0.5, "03 sec. 4.2"),
    "eatc.labram.cruzado_51.base": (0.613, 0.0005, "03 sec. 4.2"),
    "eatc.labram.cruzado_51.ea": (0.623, 0.0005, "03 sec. 4.2"),
    "eatc.labram.cruzado_51.dif": (0.0093, 5e-05, "03 sec. 4.2"),
    "eatc.labram.cruzado_51.ic_inf": (-0.0074, 5e-05, "03 sec. 4.2"),
    "eatc.labram.cruzado_51.ic_sup": (0.0265, 5e-05, "03 sec. 4.2"),
    "eatc.labram.cruzado_51.p": (0.358, 5e-05, "03 sec. 4.2"),
    "eatc.labram.cruzado_51.p_holm": (1.0, 5e-05, "03 sec. 4.2"),
    "eatc.labram.cruzado_51.r": (0.13, 0.005, "03 sec. 4.2"),
    "eatc.labram.cruzado_51.mejoran": (20.0, 0.5, "03 sec. 4.2"),
    "eatc.labram.cruzado_51.empeoran": (15.0, 0.5, "03 sec. 4.2"),
    "eatc.labram.h44_wbcic_51.base": (0.628, 0.0005, "03 sec. 4.2"),
    "eatc.labram.h44_wbcic_51.ea": (0.595, 0.0005, "03 sec. 4.2"),
    "eatc.labram.h44_wbcic_51.dif": (-0.0333, 5e-05, "03 sec. 4.2"),
    "eatc.labram.h44_wbcic_51.ic_inf": (-0.0593, 5e-05, "03 sec. 4.2"),
    "eatc.labram.h44_wbcic_51.ic_sup": (-0.0064, 5e-05, "03 sec. 4.2"),
    "eatc.labram.h44_wbcic_51.p": (0.0277, 5e-05, "03 sec. 4.2"),
    "eatc.labram.h44_wbcic_51.p_holm": (0.277, 5e-05, "03 sec. 4.2"),
    "eatc.labram.h44_wbcic_51.r": (0.31, 0.005, "03 sec. 4.2"),
    "eatc.labram.h44_wbcic_51.mejoran": (18.0, 0.5, "03 sec. 4.2"),
    "eatc.labram.h44_wbcic_51.empeoran": (31.0, 0.5, "03 sec. 4.2"),
    "eatc.labram.h44_openbmi_10.base": (0.745, 0.0005, "03 sec. 4.2"),
    "eatc.labram.h44_openbmi_10.ea": (0.748, 0.0005, "03 sec. 4.2"),
    "eatc.labram.h44_openbmi_10.dif": (0.003, 5e-05, "03 sec. 4.2"),
    "eatc.labram.h44_openbmi_10.ic_inf": (-0.03, 5e-05, "03 sec. 4.2"),
    "eatc.labram.h44_openbmi_10.ic_sup": (0.032, 5e-05, "03 sec. 4.2"),
    "eatc.labram.h44_openbmi_10.p": (0.9219, 5e-05, "03 sec. 4.2"),
    "eatc.labram.h44_openbmi_10.p_holm": (1.0, 5e-05, "03 sec. 4.2"),
    "eatc.labram.h44_openbmi_10.r": (0.03, 0.005, "03 sec. 4.2"),
    "eatc.labram.h44_openbmi_10.mejoran": (7.0, 0.5, "03 sec. 4.2"),
    "eatc.labram.h44_openbmi_10.empeoran": (3.0, 0.5, "03 sec. 4.2"),
    "eatc.eegmamba.cruzado_51.base": (0.589, 0.0005, "03 sec. 4.2"),
    "eatc.eegmamba.cruzado_51.ea": (0.598, 0.0005, "03 sec. 4.2"),
    "eatc.eegmamba.cruzado_51.dif": (0.0088, 5e-05, "03 sec. 4.2"),
    "eatc.eegmamba.cruzado_51.ic_inf": (-0.0142, 5e-05, "03 sec. 4.2"),
    "eatc.eegmamba.cruzado_51.ic_sup": (0.0319, 5e-05, "03 sec. 4.2"),
    "eatc.eegmamba.cruzado_51.p": (0.5593, 5e-05, "03 sec. 4.2"),
    "eatc.eegmamba.cruzado_51.p_holm": (1.0, 5e-05, "03 sec. 4.2"),
    "eatc.eegmamba.cruzado_51.r": (0.08, 0.005, "03 sec. 4.2"),
    "eatc.eegmamba.cruzado_51.mejoran": (24.0, 0.5, "03 sec. 4.2"),
    "eatc.eegmamba.cruzado_51.empeoran": (24.0, 0.5, "03 sec. 4.2"),
    "eatc.eegmamba.h44_wbcic_51.base": (0.598, 0.0005, "03 sec. 4.2"),
    "eatc.eegmamba.h44_wbcic_51.ea": (0.6, 0.0005, "03 sec. 4.2"),
    "eatc.eegmamba.h44_wbcic_51.dif": (0.002, 5e-05, "03 sec. 4.2"),
    "eatc.eegmamba.h44_wbcic_51.ic_inf": (-0.024, 5e-05, "03 sec. 4.2"),
    "eatc.eegmamba.h44_wbcic_51.ic_sup": (0.027, 5e-05, "03 sec. 4.2"),
    "eatc.eegmamba.h44_wbcic_51.p": (0.8324, 5e-05, "03 sec. 4.2"),
    "eatc.eegmamba.h44_wbcic_51.p_holm": (1.0, 5e-05, "03 sec. 4.2"),
    "eatc.eegmamba.h44_wbcic_51.r": (0.03, 0.005, "03 sec. 4.2"),
    "eatc.eegmamba.h44_wbcic_51.mejoran": (25.0, 0.5, "03 sec. 4.2"),
    "eatc.eegmamba.h44_wbcic_51.empeoran": (21.0, 0.5, "03 sec. 4.2"),
    "eatc.eegmamba.h44_openbmi_10.base": (0.7, 0.0005, "03 sec. 4.2"),
    "eatc.eegmamba.h44_openbmi_10.ea": (0.759, 0.0005, "03 sec. 4.2"),
    "eatc.eegmamba.h44_openbmi_10.dif": (0.0597, 5e-05, "03 sec. 4.2"),
    "eatc.eegmamba.h44_openbmi_10.ic_inf": (0.033, 5e-05, "03 sec. 4.2"),
    "eatc.eegmamba.h44_openbmi_10.ic_sup": (0.0892, 5e-05, "03 sec. 4.2"),
    "eatc.eegmamba.h44_openbmi_10.p": (0.002, 5e-05, "03 sec. 4.2"),
    "eatc.eegmamba.h44_openbmi_10.p_holm": (0.0234, 5e-05, "03 sec. 4.2"),
    "eatc.eegmamba.h44_openbmi_10.r": (0.98, 0.005, "03 sec. 4.2"),
    "eatc.eegmamba.h44_openbmi_10.mejoran": (10.0, 0.5, "03 sec. 4.2"),
    "eatc.eegmamba.h44_openbmi_10.empeoran": (0.0, 0.5, "03 sec. 4.2"),
    "eati.eegnet.caida_persona.est": (-0.035, 0.0005, "03 sec. 4.3"),
    "eati.eegnet.caida_persona.inf": (-0.072, 0.0005, "03 sec. 4.3"),
    "eati.eegnet.caida_persona.sup": (-0.001, 0.0005, "03 sec. 4.3"),
    "eati.eegnet.caida_protocolo.est": (0.068, 0.0005, "03 sec. 4.3"),
    "eati.eegnet.caida_protocolo.inf": (0.025, 0.0005, "03 sec. 4.3"),
    "eati.eegnet.caida_protocolo.sup": (0.111, 0.0005, "03 sec. 4.3"),
    "eati.eegnet.protocolo_menos_persona.est": (0.103, 0.0005, "03 sec. 4.3"),
    "eati.eegnet.protocolo_menos_persona.inf": (0.03, 0.0005, "03 sec. 4.3"),
    "eati.eegnet.protocolo_menos_persona.sup": (0.18, 0.0005, "03 sec. 4.3"),
    "eati.eegnet.persona.est": (0.057, 0.0005, "03 sec. 4.3"),
    "eati.eegnet.persona.inf": (0.022, 0.0005, "03 sec. 4.3"),
    "eati.eegnet.persona.sup": (0.093, 0.0005, "03 sec. 4.3"),
    "eati.eegnet.wbcic44.est": (-0.011, 0.0005, "03 sec. 4.3"),
    "eati.eegnet.wbcic44.inf": (-0.035, 0.0005, "03 sec. 4.3"),
    "eati.eegnet.wbcic44.sup": (0.013, 0.0005, "03 sec. 4.3"),
    "eati.conformer.caida_persona.est": (-0.019, 0.0005, "03 sec. 4.3"),
    "eati.conformer.caida_persona.inf": (-0.051, 0.0005, "03 sec. 4.3"),
    "eati.conformer.caida_persona.sup": (0.013, 0.0005, "03 sec. 4.3"),
    "eati.conformer.caida_protocolo.est": (0.033, 0.0005, "03 sec. 4.3"),
    "eati.conformer.caida_protocolo.inf": (-0.006, 0.0005, "03 sec. 4.3"),
    "eati.conformer.caida_protocolo.sup": (0.072, 0.0005, "03 sec. 4.3"),
    "eati.conformer.protocolo_menos_persona.est": (0.052, 0.0005, "03 sec. 4.3"),
    "eati.conformer.protocolo_menos_persona.inf": (-0.016, 0.0005, "03 sec. 4.3"),
    "eati.conformer.protocolo_menos_persona.sup": (0.12, 0.0005, "03 sec. 4.3"),
    "eati.conformer.persona.est": (0.045, 0.0005, "03 sec. 4.3"),
    "eati.conformer.persona.inf": (0.013, 0.0005, "03 sec. 4.3"),
    "eati.conformer.persona.sup": (0.078, 0.0005, "03 sec. 4.3"),
    "eati.conformer.wbcic44.est": (0.012, 0.0005, "03 sec. 4.3"),
    "eati.conformer.wbcic44.inf": (-0.009, 0.0005, "03 sec. 4.3"),
    "eati.conformer.wbcic44.sup": (0.033, 0.0005, "03 sec. 4.3"),
    "eati.labram.caida_persona.est": (0.024, 0.0005, "03 sec. 4.3"),
    "eati.labram.caida_persona.inf": (-0.005, 0.0005, "03 sec. 4.3"),
    "eati.labram.caida_persona.sup": (0.057, 0.0005, "03 sec. 4.3"),
    "eati.labram.caida_protocolo.est": (0.036, 0.0005, "03 sec. 4.3"),
    "eati.labram.caida_protocolo.inf": (-0.007, 0.0005, "03 sec. 4.3"),
    "eati.labram.caida_protocolo.sup": (0.076, 0.0005, "03 sec. 4.3"),
    "eati.labram.protocolo_menos_persona.est": (0.012, 0.0005, "03 sec. 4.3"),
    "eati.labram.protocolo_menos_persona.inf": (-0.06, 0.0005, "03 sec. 4.3"),
    "eati.labram.protocolo_menos_persona.sup": (0.075, 0.0005, "03 sec. 4.3"),
    "eati.labram.persona.est": (0.003, 0.0005, "03 sec. 4.3"),
    "eati.labram.persona.inf": (-0.03, 0.0005, "03 sec. 4.3"),
    "eati.labram.persona.sup": (0.032, 0.0005, "03 sec. 4.3"),
    "eati.labram.wbcic44.est": (-0.033, 0.0005, "03 sec. 4.3"),
    "eati.labram.wbcic44.inf": (-0.06, 0.0005, "03 sec. 4.3"),
    "eati.labram.wbcic44.sup": (-0.006, 0.0005, "03 sec. 4.3"),
    "eati.eegmamba.caida_persona.est": (-0.01, 0.0005, "03 sec. 4.3"),
    "eati.eegmamba.caida_persona.inf": (-0.039, 0.0005, "03 sec. 4.3"),
    "eati.eegmamba.caida_persona.sup": (0.017, 0.0005, "03 sec. 4.3"),
    "eati.eegmamba.caida_protocolo.est": (0.058, 0.0005, "03 sec. 4.3"),
    "eati.eegmamba.caida_protocolo.inf": (0.021, 0.0005, "03 sec. 4.3"),
    "eati.eegmamba.caida_protocolo.sup": (0.097, 0.0005, "03 sec. 4.3"),
    "eati.eegmamba.protocolo_menos_persona.est": (0.068, 0.0005, "03 sec. 4.3"),
    "eati.eegmamba.protocolo_menos_persona.inf": (0.008, 0.0005, "03 sec. 4.3"),
    "eati.eegmamba.protocolo_menos_persona.sup": (0.133, 0.0005, "03 sec. 4.3"),
    "eati.eegmamba.persona.est": (0.06, 0.0005, "03 sec. 4.3"),
    "eati.eegmamba.persona.inf": (0.033, 0.0005, "03 sec. 4.3"),
    "eati.eegmamba.persona.sup": (0.089, 0.0005, "03 sec. 4.3"),
    "eati.eegmamba.wbcic44.est": (0.002, 0.0005, "03 sec. 4.3"),
    "eati.eegmamba.wbcic44.inf": (-0.024, 0.0005, "03 sec. 4.3"),
    "eati.eegmamba.wbcic44.sup": (0.027, 0.0005, "03 sec. 4.3"),
    "eatf.eegnet.caida_protocolo": (-0.557, 0.0005, "03 sec. 4.3"),
    "eatf.eegnet.brecha": (0.005, 0.0005, "03 sec. 4.3"),
    "eatf.conformer.caida_protocolo": (-0.334, 0.0005, "03 sec. 4.3"),
    "eatf.conformer.brecha": (0.075, 0.0005, "03 sec. 4.3"),
    "eatf.labram.caida_protocolo": (-0.311, 0.0005, "03 sec. 4.3"),
    "eatf.labram.brecha": (0.056, 0.0005, "03 sec. 4.3"),
    "eatf.eegmamba.caida_protocolo": (-0.567, 0.0005, "03 sec. 4.3"),
    "eatf.eegmamba.brecha": (0.076, 0.0005, "03 sec. 4.3"),
    "t5.loso.eegnet.acc.s2026": (0.763, 0.0005, "04 T5"),
    "t5.loso.eegnet.acc.s2027": (0.76, 0.0005, "04 T5"),
    "t5.loso.eegnet.acc.s2028": (0.76, 0.0005, "04 T5"),
    "t5.loso.eegnet.acc.media": (0.761, 0.0005, "04 T5"),
    "t5.loso.eegnet.acc.sd_semillas": (0.0017, 5e-05, "04 T5"),
    "t5.loso.eegnet.acc.sd_sujetos": (0.119, 0.0005, "04 T5"),
    "t5.loso.eegnet.acc.sd_dentro": (0.021, 0.0005, "04 T5"),
    "t5.loso.eegnet.kappa.s2026": (0.525, 0.0005, "04 T5"),
    "t5.loso.eegnet.kappa.s2027": (0.519, 0.0005, "04 T5"),
    "t5.loso.eegnet.kappa.s2028": (0.519, 0.0005, "04 T5"),
    "t5.loso.eegnet.kappa.media": (0.521, 0.0005, "04 T5"),
    "t5.loso.eegnet.kappa.sd_semillas": (0.0034, 5e-05, "04 T5"),
    "t5.loso.conformer.acc.s2026": (0.752, 0.0005, "04 T5"),
    "t5.loso.conformer.acc.s2027": (0.744, 0.0005, "04 T5"),
    "t5.loso.conformer.acc.s2028": (0.745, 0.0005, "04 T5"),
    "t5.loso.conformer.acc.media": (0.747, 0.0005, "04 T5"),
    "t5.loso.conformer.acc.sd_semillas": (0.0042, 5e-05, "04 T5"),
    "t5.loso.conformer.acc.sd_sujetos": (0.121, 0.0005, "04 T5"),
    "t5.loso.conformer.acc.sd_dentro": (0.025, 0.0005, "04 T5"),
    "t5.loso.conformer.kappa.s2026": (0.503, 0.0005, "04 T5"),
    "t5.loso.conformer.kappa.s2027": (0.489, 0.0005, "04 T5"),
    "t5.loso.conformer.kappa.s2028": (0.489, 0.0005, "04 T5"),
    "t5.loso.conformer.kappa.media": (0.494, 0.0005, "04 T5"),
    "t5.loso.conformer.kappa.sd_semillas": (0.0083, 5e-05, "04 T5"),
    "t5.loso.labram.acc.s2026": (0.754, 0.0005, "04 T5"),
    "t5.loso.labram.acc.s2027": (0.753, 0.0005, "04 T5"),
    "t5.loso.labram.acc.s2028": (0.758, 0.0005, "04 T5"),
    "t5.loso.labram.acc.media": (0.755, 0.0005, "04 T5"),
    "t5.loso.labram.acc.sd_semillas": (0.0025, 5e-05, "04 T5"),
    "t5.loso.labram.acc.sd_sujetos": (0.107, 0.0005, "04 T5"),
    "t5.loso.labram.acc.sd_dentro": (0.022, 0.0005, "04 T5"),
    "t5.loso.labram.kappa.s2026": (0.508, 0.0005, "04 T5"),
    "t5.loso.labram.kappa.s2027": (0.506, 0.0005, "04 T5"),
    "t5.loso.labram.kappa.s2028": (0.515, 0.0005, "04 T5"),
    "t5.loso.labram.kappa.media": (0.51, 0.0005, "04 T5"),
    "t5.loso.labram.kappa.sd_semillas": (0.005, 5e-05, "04 T5"),
    "t5.loso.eegmamba.acc.s2026": (0.717, 0.0005, "04 T5"),
    "t5.loso.eegmamba.acc.s2027": (0.714, 0.0005, "04 T5"),
    "t5.loso.eegmamba.acc.s2028": (0.715, 0.0005, "04 T5"),
    "t5.loso.eegmamba.acc.media": (0.715, 0.0005, "04 T5"),
    "t5.loso.eegmamba.acc.sd_semillas": (0.0015, 5e-05, "04 T5"),
    "t5.loso.eegmamba.acc.sd_sujetos": (0.103, 0.0005, "04 T5"),
    "t5.loso.eegmamba.acc.sd_dentro": (0.019, 0.0005, "04 T5"),
    "t5.loso.eegmamba.kappa.s2026": (0.434, 0.0005, "04 T5"),
    "t5.loso.eegmamba.kappa.s2027": (0.428, 0.0005, "04 T5"),
    "t5.loso.eegmamba.kappa.s2028": (0.431, 0.0005, "04 T5"),
    "t5.loso.eegmamba.kappa.media": (0.431, 0.0005, "04 T5"),
    "t5.loso.eegmamba.kappa.sd_semillas": (0.003, 5e-05, "04 T5"),
    "t5.within.eegnet.acc.s2026": (0.633, 0.0005, "04 T5"),
    "t5.within.eegnet.acc.s2027": (0.631, 0.0005, "04 T5"),
    "t5.within.eegnet.acc.s2028": (0.639, 0.0005, "04 T5"),
    "t5.within.eegnet.acc.media": (0.634, 0.0005, "04 T5"),
    "t5.within.eegnet.acc.sd_semillas": (0.0043, 5e-05, "04 T5"),
    "t5.within.eegnet.acc.sd_sujetos": (0.107, 0.0005, "04 T5"),
    "t5.within.eegnet.acc.sd_dentro": (0.025, 0.0005, "04 T5"),
    "t5.within.eegnet.kappa.s2026": (0.266, 0.0005, "04 T5"),
    "t5.within.eegnet.kappa.s2027": (0.261, 0.0005, "04 T5"),
    "t5.within.eegnet.kappa.s2028": (0.278, 0.0005, "04 T5"),
    "t5.within.eegnet.kappa.media": (0.269, 0.0005, "04 T5"),
    "t5.within.eegnet.kappa.sd_semillas": (0.0085, 5e-05, "04 T5"),
    "t5.within.conformer.acc.s2026": (0.607, 0.0005, "04 T5"),
    "t5.within.conformer.acc.s2027": (0.6, 0.0005, "04 T5"),
    "t5.within.conformer.acc.s2028": (0.601, 0.0005, "04 T5"),
    "t5.within.conformer.acc.media": (0.602, 0.0005, "04 T5"),
    "t5.within.conformer.acc.sd_semillas": (0.0038, 5e-05, "04 T5"),
    "t5.within.conformer.acc.sd_sujetos": (0.117, 0.0005, "04 T5"),
    "t5.within.conformer.acc.sd_dentro": (0.02, 0.0005, "04 T5"),
    "t5.within.conformer.kappa.s2026": (0.214, 0.0005, "04 T5"),
    "t5.within.conformer.kappa.s2027": (0.2, 0.0005, "04 T5"),
    "t5.within.conformer.kappa.s2028": (0.201, 0.0005, "04 T5"),
    "t5.within.conformer.kappa.media": (0.205, 0.0005, "04 T5"),
    "t5.within.conformer.kappa.sd_semillas": (0.0077, 5e-05, "04 T5"),
    "t5.within.labram.acc.s2026": (0.665, 0.0005, "04 T5"),
    "t5.within.labram.acc.s2027": (0.663, 0.0005, "04 T5"),
    "t5.within.labram.acc.s2028": (0.662, 0.0005, "04 T5"),
    "t5.within.labram.acc.media": (0.663, 0.0005, "04 T5"),
    "t5.within.labram.acc.sd_semillas": (0.0013, 5e-05, "04 T5"),
    "t5.within.labram.acc.sd_sujetos": (0.153, 0.0005, "04 T5"),
    "t5.within.labram.acc.sd_dentro": (0.016, 0.0005, "04 T5"),
    "t5.within.labram.kappa.s2026": (0.329, 0.0005, "04 T5"),
    "t5.within.labram.kappa.s2027": (0.326, 0.0005, "04 T5"),
    "t5.within.labram.kappa.s2028": (0.324, 0.0005, "04 T5"),
    "t5.within.labram.kappa.media": (0.327, 0.0005, "04 T5"),
    "t5.within.labram.kappa.sd_semillas": (0.0027, 5e-05, "04 T5"),
    "t5.within.eegmamba.acc.s2026": (0.577, 0.0005, "04 T5"),
    "t5.within.eegmamba.acc.s2027": (0.571, 0.0005, "04 T5"),
    "t5.within.eegmamba.acc.s2028": (0.564, 0.0005, "04 T5"),
    "t5.within.eegmamba.acc.media": (0.571, 0.0005, "04 T5"),
    "t5.within.eegmamba.acc.sd_semillas": (0.0062, 5e-05, "04 T5"),
    "t5.within.eegmamba.acc.sd_sujetos": (0.095, 0.0005, "04 T5"),
    "t5.within.eegmamba.acc.sd_dentro": (0.022, 0.0005, "04 T5"),
    "t5.within.eegmamba.kappa.s2026": (0.154, 0.0005, "04 T5"),
    "t5.within.eegmamba.kappa.s2027": (0.142, 0.0005, "04 T5"),
    "t5.within.eegmamba.kappa.s2028": (0.129, 0.0005, "04 T5"),
    "t5.within.eegmamba.kappa.media": (0.142, 0.0005, "04 T5"),
    "t5.within.eegmamba.kappa.sd_semillas": (0.0124, 5e-05, "04 T5"),
    "t1.loso.eegnet.dcruce_media": (0.003, 5e-05, "04 T1"),
    "t1.loso.eegnet.dcruce_ic_inf": (-0.0042, 5e-05, "04 T1"),
    "t1.loso.eegnet.dcruce_ic_sup": (0.0104, 5e-05, "04 T1"),
    "t1.loso.eegnet.dcruce_mediana": (-0.0, 5e-05, "04 T1"),
    "t1.loso.eegnet.dcruce_sd": (0.0277, 5e-05, "04 T1"),
    "t1.loso.eegnet.dcruce_p": (0.6, 0.005, "04 T1"),
    "t1.loso.eegnet.dintra_media": (-0.0001, 5e-05, "04 T1"),
    "t1.loso.eegnet.dintra_sd": (0.0387, 5e-05, "04 T1"),
    "t1.loso.eegnet.dintra_p": (0.25, 0.005, "04 T1"),
    "t1.loso.conformer.dcruce_media": (0.0072, 5e-05, "04 T1"),
    "t1.loso.conformer.dcruce_ic_inf": (-0.0043, 5e-05, "04 T1"),
    "t1.loso.conformer.dcruce_ic_sup": (0.0198, 5e-05, "04 T1"),
    "t1.loso.conformer.dcruce_mediana": (0.0019, 5e-05, "04 T1"),
    "t1.loso.conformer.dcruce_sd": (0.0454, 5e-05, "04 T1"),
    "t1.loso.conformer.dcruce_p": (0.42, 0.005, "04 T1"),
    "t1.loso.conformer.dintra_media": (-0.0004, 5e-05, "04 T1"),
    "t1.loso.conformer.dintra_sd": (0.071, 5e-05, "04 T1"),
    "t1.loso.conformer.dintra_p": (0.41, 0.005, "04 T1"),
    "t1.loso.labram.dcruce_media": (-0.0016, 5e-05, "04 T1"),
    "t1.loso.labram.dcruce_ic_inf": (-0.0097, 5e-05, "04 T1"),
    "t1.loso.labram.dcruce_ic_sup": (0.0063, 5e-05, "04 T1"),
    "t1.loso.labram.dcruce_mediana": (0.0025, 5e-05, "04 T1"),
    "t1.loso.labram.dcruce_sd": (0.0307, 5e-05, "04 T1"),
    "t1.loso.labram.dcruce_p": (0.91, 0.005, "04 T1"),
    "t1.loso.labram.dintra_media": (-0.0046, 5e-05, "04 T1"),
    "t1.loso.labram.dintra_sd": (0.0356, 5e-05, "04 T1"),
    "t1.loso.labram.dintra_p": (0.33, 0.005, "04 T1"),
    "t1.loso.eegmamba.dcruce_media": (0.0022, 5e-05, "04 T1"),
    "t1.loso.eegmamba.dcruce_ic_inf": (-0.0045, 5e-05, "04 T1"),
    "t1.loso.eegmamba.dcruce_ic_sup": (0.0089, 5e-05, "04 T1"),
    "t1.loso.eegmamba.dcruce_mediana": (-0.0012, 5e-05, "04 T1"),
    "t1.loso.eegmamba.dcruce_sd": (0.0252, 5e-05, "04 T1"),
    "t1.loso.eegmamba.dcruce_p": (0.72, 0.005, "04 T1"),
    "t1.loso.eegmamba.dintra_media": (-0.0015, 5e-05, "04 T1"),
    "t1.loso.eegmamba.dintra_sd": (0.0325, 5e-05, "04 T1"),
    "t1.loso.eegmamba.dintra_p": (0.67, 0.005, "04 T1"),
    "t1.within.eegnet.dcruce_media": (-0.0017, 5e-05, "04 T1"),
    "t1.within.eegnet.dcruce_ic_inf": (-0.0106, 5e-05, "04 T1"),
    "t1.within.eegnet.dcruce_ic_sup": (0.0065, 5e-05, "04 T1"),
    "t1.within.eegnet.dcruce_mediana": (-0.0031, 5e-05, "04 T1"),
    "t1.within.eegnet.dcruce_sd": (0.0319, 5e-05, "04 T1"),
    "t1.within.eegnet.dcruce_p": (0.88, 0.005, "04 T1"),
    "t1.within.eegnet.dintra_media": (-0.0083, 5e-05, "04 T1"),
    "t1.within.eegnet.dintra_sd": (0.0545, 5e-05, "04 T1"),
    "t1.within.eegnet.dintra_p": (0.02, 0.005, "04 T1"),
    "t1.within.conformer.dcruce_media": (0.0066, 5e-05, "04 T1"),
    "t1.within.conformer.dcruce_ic_inf": (-0.0002, 5e-05, "04 T1"),
    "t1.within.conformer.dcruce_ic_sup": (0.0133, 5e-05, "04 T1"),
    "t1.within.conformer.dcruce_mediana": (0.0081, 5e-05, "04 T1"),
    "t1.within.conformer.dcruce_sd": (0.0259, 5e-05, "04 T1"),
    "t1.within.conformer.dcruce_p": (0.04, 0.005, "04 T1"),
    "t1.within.conformer.dintra_media": (-0.0006, 5e-05, "04 T1"),
    "t1.within.conformer.dintra_sd": (0.0352, 5e-05, "04 T1"),
    "t1.within.conformer.dintra_p": (0.57, 0.005, "04 T1"),
    "t1.within.labram.dcruce_media": (0.0021, 5e-05, "04 T1"),
    "t1.within.labram.dcruce_ic_inf": (-0.0051, 5e-05, "04 T1"),
    "t1.within.labram.dcruce_ic_sup": (0.0092, 5e-05, "04 T1"),
    "t1.within.labram.dcruce_mediana": (0.0044, 5e-05, "04 T1"),
    "t1.within.labram.dcruce_sd": (0.0265, 5e-05, "04 T1"),
    "t1.within.labram.dcruce_p": (0.37, 0.005, "04 T1"),
    "t1.within.labram.dintra_media": (0.0012, 5e-05, "04 T1"),
    "t1.within.labram.dintra_sd": (0.0219, 5e-05, "04 T1"),
    "t1.within.labram.dintra_p": (0.88, 0.005, "04 T1"),
    "t1.within.eegmamba.dcruce_media": (0.0092, 5e-05, "04 T1"),
    "t1.within.eegmamba.dcruce_ic_inf": (0.0003, 5e-05, "04 T1"),
    "t1.within.eegmamba.dcruce_ic_sup": (0.0178, 5e-05, "04 T1"),
    "t1.within.eegmamba.dcruce_mediana": (0.0062, 5e-05, "04 T1"),
    "t1.within.eegmamba.dcruce_sd": (0.0332, 5e-05, "04 T1"),
    "t1.within.eegmamba.dcruce_p": (0.05, 0.005, "04 T1"),
    "t1.within.eegmamba.dintra_media": (0.0065, 5e-05, "04 T1"),
    "t1.within.eegmamba.dintra_sd": (0.0286, 5e-05, "04 T1"),
    "t1.within.eegmamba.dintra_p": (0.11, 0.005, "04 T1"),
    "t2.loso.eegnet.sd_esperada": (0.0335, 5e-05, "04 T2"),
    "t2.loso.eegnet.razon": (0.68, 0.005, "04 T2"),
    "t2.loso.conformer.sd_esperada": (0.0615, 5e-05, "04 T2"),
    "t2.loso.conformer.razon": (0.54, 0.005, "04 T2"),
    "t2.loso.labram.sd_esperada": (0.0308, 5e-05, "04 T2"),
    "t2.loso.labram.razon": (0.99, 0.005, "04 T2"),
    "t2.loso.eegmamba.sd_esperada": (0.0281, 5e-05, "04 T2"),
    "t2.loso.eegmamba.razon": (0.81, 0.005, "04 T2"),
    "t2.within.eegnet.sd_esperada": (0.0472, 5e-05, "04 T2"),
    "t2.within.eegnet.razon": (0.46, 0.005, "04 T2"),
    "t2.within.conformer.sd_esperada": (0.0305, 5e-05, "04 T2"),
    "t2.within.conformer.razon": (0.72, 0.005, "04 T2"),
    "t2.within.labram.sd_esperada": (0.0189, 5e-05, "04 T2"),
    "t2.within.labram.razon": (1.96, 0.005, "04 T2"),
    "t2.within.eegmamba.sd_esperada": (0.0248, 5e-05, "04 T2"),
    "t2.within.eegmamba.razon": (1.8, 0.005, "04 T2"),
    "t3.loso.eegnet.media": (0.0008, 5e-05, "04 T3"),
    "t3.loso.eegnet.ic_inf": (-0.0082, 5e-05, "04 T3"),
    "t3.loso.eegnet.ic_sup": (0.01, 5e-05, "04 T3"),
    "t3.loso.eegnet.p": (0.96, 0.005, "04 T3"),
    "t3.loso.conformer.media": (0.005, 5e-05, "04 T3"),
    "t3.loso.conformer.ic_inf": (-0.0068, 5e-05, "04 T3"),
    "t3.loso.conformer.ic_sup": (0.0178, 5e-05, "04 T3"),
    "t3.loso.conformer.p": (0.56, 0.005, "04 T3"),
    "t3.loso.labram.media": (-0.0038, 5e-05, "04 T3"),
    "t3.loso.labram.ic_inf": (-0.0142, 5e-05, "04 T3"),
    "t3.loso.labram.ic_sup": (0.0065, 5e-05, "04 T3"),
    "t3.loso.labram.p": (0.57, 0.005, "04 T3"),
    "t3.within.eegnet.media": (-0.0109, 5e-05, "04 T3"),
    "t3.within.eegnet.ic_inf": (-0.0243, 5e-05, "04 T3"),
    "t3.within.eegnet.ic_sup": (0.0022, 5e-05, "04 T3"),
    "t3.within.eegnet.p": (0.13, 0.005, "04 T3"),
    "t3.within.conformer.media": (-0.0025, 5e-05, "04 T3"),
    "t3.within.conformer.ic_inf": (-0.0138, 5e-05, "04 T3"),
    "t3.within.conformer.ic_sup": (0.0085, 5e-05, "04 T3"),
    "t3.within.conformer.p": (0.99, 0.005, "04 T3"),
    "t3.within.labram.media": (-0.0071, 5e-05, "04 T3"),
    "t3.within.labram.ic_inf": (-0.0182, 5e-05, "04 T3"),
    "t3.within.labram.ic_sup": (0.0044, 5e-05, "04 T3"),
    "t3.within.labram.p": (0.29, 0.005, "04 T3"),
    "t4.loso.eegnet.delta_bruto": (0.003, 5e-05, "04 T4"),
    "t4.loso.eegnet.cota_bruta": (0.0104, 5e-05, "04 T4"),
    "t4.loso.eegnet.delta_neto": (0.0008, 5e-05, "04 T4"),
    "t4.loso.eegnet.cota_neta": (0.01, 5e-05, "04 T4"),
    "t4.loso.eegnet.cota_uso": (0.0104, 5e-05, "04 T4"),
    "t4.loso.eegnet.cota_media3": (0.0035, 5e-05, "04 T4"),
    "t4.loso.conformer.delta_bruto": (0.0072, 5e-05, "04 T4"),
    "t4.loso.conformer.cota_bruta": (0.0198, 5e-05, "04 T4"),
    "t4.loso.conformer.delta_neto": (0.005, 5e-05, "04 T4"),
    "t4.loso.conformer.cota_neta": (0.0178, 5e-05, "04 T4"),
    "t4.loso.conformer.cota_uso": (0.0198, 5e-05, "04 T4"),
    "t4.loso.conformer.cota_media3": (0.0066, 5e-05, "04 T4"),
    "t4.loso.labram.delta_bruto": (-0.0016, 5e-05, "04 T4"),
    "t4.loso.labram.cota_bruta": (0.0097, 5e-05, "04 T4"),
    "t4.loso.labram.delta_neto": (-0.0038, 5e-05, "04 T4"),
    "t4.loso.labram.cota_neta": (0.0142, 5e-05, "04 T4"),
    "t4.loso.labram.cota_uso": (0.0142, 5e-05, "04 T4"),
    "t4.loso.labram.cota_media3": (0.0047, 5e-05, "04 T4"),
    "t4.within.labram.delta_bruto": (0.0021, 5e-05, "04 T4"),
    "t4.within.labram.cota_bruta": (0.0092, 5e-05, "04 T4"),
    "t4.within.labram.delta_neto": (-0.0071, 5e-05, "04 T4"),
    "t4.within.labram.cota_neta": (0.0182, 5e-05, "04 T4"),
    "t4.within.labram.cota_uso": (0.0182, 5e-05, "04 T4"),
    "t4.within.labram.cota_media3": (0.0061, 5e-05, "04 T4"),
    "t6.loso.eegnet-conformer.s2026.mediana": (0.006, 0.0005, "04 T6"),
    "t6.loso.eegnet-conformer.s2026.dif": (0.011, 0.0005, "04 T6"),
    "t6.loso.eegnet-conformer.s2026.p": (0.12, 0.005, "04 T6"),
    "t6.loso.eegnet-conformer.s2026.r": (0.21, 0.005, "04 T6"),
    "t6.loso.eegnet-conformer.s2026.p_holm": (0.35, 0.005, "04 T6"),
    "t6.loso.eegnet-conformer.media3.dif": (0.014, 0.0005, "04 T6"),
    "t6.loso.eegnet-conformer.media3.mediana": (0.006, 0.0005, "04 T6"),
    "t6.loso.eegnet-conformer.media3.p": (0.07, 0.005, "04 T6"),
    "t6.loso.eegnet-conformer.media3.r": (0.24, 0.005, "04 T6"),
    "t6.loso.eegnet-conformer.media3.p_holm": (0.22, 0.005, "04 T6"),
    "t6.loso.eegnet-conformer.dif_s2026": (0.011, 0.0005, "04 T6"),
    "t6.loso.eegnet-conformer.dif_s2027": (0.015, 0.0005, "04 T6"),
    "t6.loso.eegnet-conformer.dif_s2028": (0.015, 0.0005, "04 T6"),
    "t6.loso.eegnet-conformer.sd_semillas_dif": (0.0025, 5e-05, "04 T6"),
    "t6.loso.eegnet-conformer.p_s2026": (0.12, 0.005, "04 T6"),
    "t6.loso.eegnet-conformer.p_s2027": (0.06, 0.005, "04 T6"),
    "t6.loso.eegnet-conformer.p_s2028": (0.49, 0.005, "04 T6"),
    "t6.loso.eegnet-labram.s2026.mediana": (0.006, 0.0005, "04 T6"),
    "t6.loso.eegnet-labram.s2026.dif": (0.009, 0.0005, "04 T6"),
    "t6.loso.eegnet-labram.s2026.p": (0.27, 0.005, "04 T6"),
    "t6.loso.eegnet-labram.s2026.r": (0.15, 0.005, "04 T6"),
    "t6.loso.eegnet-labram.s2026.p_holm": (0.55, 0.005, "04 T6"),
    "t6.loso.eegnet-labram.media3.dif": (0.006, 0.0005, "04 T6"),
    "t6.loso.eegnet-labram.media3.mediana": (0.002, 0.0005, "04 T6"),
    "t6.loso.eegnet-labram.media3.p": (0.42, 0.005, "04 T6"),
    "t6.loso.eegnet-labram.media3.r": (0.11, 0.005, "04 T6"),
    "t6.loso.eegnet-labram.media3.p_holm": (0.42, 0.005, "04 T6"),
    "t6.loso.eegnet-labram.dif_s2026": (0.009, 0.0005, "04 T6"),
    "t6.loso.eegnet-labram.dif_s2027": (0.006, 0.0005, "04 T6"),
    "t6.loso.eegnet-labram.dif_s2028": (0.002, 0.0005, "04 T6"),
    "t6.loso.eegnet-labram.sd_semillas_dif": (0.0035, 5e-05, "04 T6"),
    "t6.loso.eegnet-labram.p_s2026": (0.27, 0.005, "04 T6"),
    "t6.loso.eegnet-labram.p_s2027": (0.41, 0.005, "04 T6"),
    "t6.loso.eegnet-labram.p_s2028": (0.88, 0.005, "04 T6"),
    "t6.loso.conformer-labram.s2026.mediana": (-0.001, 0.0005, "04 T6"),
    "t6.loso.conformer-labram.s2026.dif": (-0.002, 0.0005, "04 T6"),
    "t6.loso.conformer-labram.s2026.p": (0.49, 0.005, "04 T6"),
    "t6.loso.conformer-labram.s2026.r": (0.09, 0.005, "04 T6"),
    "t6.loso.conformer-labram.s2026.p_holm": (0.55, 0.005, "04 T6"),
    "t6.loso.conformer-labram.media3.dif": (-0.008, 0.0005, "04 T6"),
    "t6.loso.conformer-labram.media3.mediana": (-0.008, 0.0005, "04 T6"),
    "t6.loso.conformer-labram.media3.p": (0.19, 0.005, "04 T6"),
    "t6.loso.conformer-labram.media3.r": (0.18, 0.005, "04 T6"),
    "t6.loso.conformer-labram.media3.p_holm": (0.38, 0.005, "04 T6"),
    "t6.loso.conformer-labram.dif_s2026": (-0.002, 0.0005, "04 T6"),
    "t6.loso.conformer-labram.dif_s2027": (-0.009, 0.0005, "04 T6"),
    "t6.loso.conformer-labram.dif_s2028": (-0.013, 0.0005, "04 T6"),
    "t6.loso.conformer-labram.sd_semillas_dif": (0.0055, 5e-05, "04 T6"),
    "t6.loso.conformer-labram.p_s2026": (0.49, 0.005, "04 T6"),
    "t6.loso.conformer-labram.p_s2027": (0.32, 0.005, "04 T6"),
    "t6.loso.conformer-labram.p_s2028": (0.5, 0.005, "04 T6"),
    "t6.loso.eegnet-eegmamba.s2026.mediana": (0.044, 0.0005, "04 T6"),
    "t6.loso.eegnet-eegmamba.s2026.dif": (0.046, 0.0005, "04 T6"),
    "t6.loso.eegnet-eegmamba.s2026.r": (0.71, 0.005, "04 T6"),
    "t6.loso.eegnet-eegmamba.media3.dif": (0.045, 0.0005, "04 T6"),
    "t6.loso.eegnet-eegmamba.media3.mediana": (0.045, 0.0005, "04 T6"),
    "t6.loso.eegnet-eegmamba.media3.r": (0.73, 0.005, "04 T6"),
    "t6.loso.eegnet-eegmamba.dif_s2026": (0.046, 0.0005, "04 T6"),
    "t6.loso.eegnet-eegmamba.dif_s2027": (0.046, 0.0005, "04 T6"),
    "t6.loso.eegnet-eegmamba.dif_s2028": (0.044, 0.0005, "04 T6"),
    "t6.loso.eegnet-eegmamba.sd_semillas_dif": (0.0008, 5e-05, "04 T6"),
    "t6.loso.conformer-eegmamba.s2026.mediana": (0.027, 0.0005, "04 T6"),
    "t6.loso.conformer-eegmamba.s2026.dif": (0.035, 0.0005, "04 T6"),
    "t6.loso.conformer-eegmamba.s2026.r": (0.59, 0.005, "04 T6"),
    "t6.loso.conformer-eegmamba.media3.dif": (0.032, 0.0005, "04 T6"),
    "t6.loso.conformer-eegmamba.media3.mediana": (0.036, 0.0005, "04 T6"),
    "t6.loso.conformer-eegmamba.media3.r": (0.54, 0.005, "04 T6"),
    "t6.loso.conformer-eegmamba.dif_s2026": (0.035, 0.0005, "04 T6"),
    "t6.loso.conformer-eegmamba.dif_s2027": (0.03, 0.0005, "04 T6"),
    "t6.loso.conformer-eegmamba.dif_s2028": (0.029, 0.0005, "04 T6"),
    "t6.loso.conformer-eegmamba.sd_semillas_dif": (0.003, 5e-05, "04 T6"),
    "t6.loso.labram-eegmamba.s2026.mediana": (0.035, 0.0005, "04 T6"),
    "t6.loso.labram-eegmamba.s2026.dif": (0.037, 0.0005, "04 T6"),
    "t6.loso.labram-eegmamba.s2026.r": (0.67, 0.005, "04 T6"),
    "t6.loso.labram-eegmamba.media3.dif": (0.04, 0.0005, "04 T6"),
    "t6.loso.labram-eegmamba.media3.mediana": (0.039, 0.0005, "04 T6"),
    "t6.loso.labram-eegmamba.media3.r": (0.74, 0.005, "04 T6"),
    "t6.loso.labram-eegmamba.dif_s2026": (0.037, 0.0005, "04 T6"),
    "t6.loso.labram-eegmamba.dif_s2027": (0.039, 0.0005, "04 T6"),
    "t6.loso.labram-eegmamba.dif_s2028": (0.042, 0.0005, "04 T6"),
    "t6.loso.labram-eegmamba.sd_semillas_dif": (0.0027, 5e-05, "04 T6"),
    "t6.within.eegnet-conformer.s2026.mediana": (0.019, 0.0005, "04 T6"),
    "t6.within.eegnet-conformer.s2026.dif": (0.026, 0.0005, "04 T6"),
    "t6.within.eegnet-conformer.s2026.r": (0.48, 0.005, "04 T6"),
    "t6.within.eegnet-conformer.s2026.p_holm": (0.002, 0.0005, "04 T6"),
    "t6.within.eegnet-conformer.media3.dif": (0.032, 0.0005, "04 T6"),
    "t6.within.eegnet-conformer.media3.mediana": (0.033, 0.0005, "04 T6"),
    "t6.within.eegnet-conformer.media3.r": (0.59, 0.005, "04 T6"),
    "t6.within.eegnet-conformer.dif_s2026": (0.026, 0.0005, "04 T6"),
    "t6.within.eegnet-conformer.dif_s2027": (0.031, 0.0005, "04 T6"),
    "t6.within.eegnet-conformer.dif_s2028": (0.038, 0.0005, "04 T6"),
    "t6.within.eegnet-conformer.sd_semillas_dif": (0.0062, 5e-05, "04 T6"),
    "t6.within.eegnet-labram.s2026.mediana": (0.015, 0.0005, "04 T6"),
    "t6.within.eegnet-labram.s2026.dif": (-0.032, 0.0005, "04 T6"),
    "t6.within.eegnet-labram.s2026.p": (0.3, 0.005, "04 T6"),
    "t6.within.eegnet-labram.s2026.r": (0.14, 0.005, "04 T6"),
    "t6.within.eegnet-labram.s2026.p_holm": (0.3, 0.005, "04 T6"),
    "t6.within.eegnet-labram.media3.dif": (-0.029, 0.0005, "04 T6"),
    "t6.within.eegnet-labram.media3.mediana": (0.018, 0.0005, "04 T6"),
    "t6.within.eegnet-labram.media3.p": (0.44, 0.005, "04 T6"),
    "t6.within.eegnet-labram.media3.r": (0.11, 0.005, "04 T6"),
    "t6.within.eegnet-labram.media3.p_holm": (0.44, 0.005, "04 T6"),
    "t6.within.eegnet-labram.dif_s2026": (-0.032, 0.0005, "04 T6"),
    "t6.within.eegnet-labram.dif_s2027": (-0.033, 0.0005, "04 T6"),
    "t6.within.eegnet-labram.dif_s2028": (-0.023, 0.0005, "04 T6"),
    "t6.within.eegnet-labram.sd_semillas_dif": (0.0052, 5e-05, "04 T6"),
    "t6.within.eegnet-labram.p_s2026": (0.3, 0.005, "04 T6"),
    "t6.within.eegnet-labram.p_s2027": (0.31, 0.005, "04 T6"),
    "t6.within.eegnet-labram.p_s2028": (0.58, 0.005, "04 T6"),
    "t6.within.conformer-labram.s2026.mediana": (-0.035, 0.0005, "04 T6"),
    "t6.within.conformer-labram.s2026.dif": (-0.058, 0.0005, "04 T6"),
    "t6.within.conformer-labram.s2026.p": (0.004, 0.0005, "04 T6"),
    "t6.within.conformer-labram.s2026.r": (0.39, 0.005, "04 T6"),
    "t6.within.conformer-labram.s2026.p_holm": (0.008, 0.0005, "04 T6"),
    "t6.within.conformer-labram.media3.dif": (-0.061, 0.0005, "04 T6"),
    "t6.within.conformer-labram.media3.mediana": (-0.034, 0.0005, "04 T6"),
    "t6.within.conformer-labram.media3.p": (0.001, 0.0005, "04 T6"),
    "t6.within.conformer-labram.media3.r": (0.44, 0.005, "04 T6"),
    "t6.within.conformer-labram.media3.p_holm": (0.002, 0.0005, "04 T6"),
    "t6.within.conformer-labram.dif_s2026": (-0.058, 0.0005, "04 T6"),
    "t6.within.conformer-labram.dif_s2027": (-0.063, 0.0005, "04 T6"),
    "t6.within.conformer-labram.dif_s2028": (-0.061, 0.0005, "04 T6"),
    "t6.within.conformer-labram.sd_semillas_dif": (0.0028, 5e-05, "04 T6"),
    "t6.within.conformer-labram.p_s2026": (0.004, 0.0005, "04 T6"),
    "t6.within.conformer-labram.p_s2027": (0.001, 0.0005, "04 T6"),
    "t6.within.eegnet-eegmamba.s2026.mediana": (0.065, 0.0005, "04 T6"),
    "t6.within.eegnet-eegmamba.s2026.dif": (0.056, 0.0005, "04 T6"),
    "t6.within.eegnet-eegmamba.s2026.r": (0.63, 0.005, "04 T6"),
    "t6.within.eegnet-eegmamba.media3.dif": (0.063, 0.0005, "04 T6"),
    "t6.within.eegnet-eegmamba.media3.mediana": (0.064, 0.0005, "04 T6"),
    "t6.within.eegnet-eegmamba.media3.r": (0.73, 0.005, "04 T6"),
    "t6.within.eegnet-eegmamba.dif_s2026": (0.056, 0.0005, "04 T6"),
    "t6.within.eegnet-eegmamba.dif_s2027": (0.06, 0.0005, "04 T6"),
    "t6.within.eegnet-eegmamba.dif_s2028": (0.074, 0.0005, "04 T6"),
    "t6.within.eegnet-eegmamba.sd_semillas_dif": (0.0097, 5e-05, "04 T6"),
    "t6.within.conformer-eegmamba.s2026.mediana": (0.016, 0.0005, "04 T6"),
    "t6.within.conformer-eegmamba.s2026.dif": (0.03, 0.0005, "04 T6"),
    "t6.within.conformer-eegmamba.s2026.r": (0.45, 0.005, "04 T6"),
    "t6.within.conformer-eegmamba.s2026.p_holm": (0.003, 0.0005, "04 T6"),
    "t6.within.conformer-eegmamba.media3.dif": (0.032, 0.0005, "04 T6"),
    "t6.within.conformer-eegmamba.media3.mediana": (0.027, 0.0005, "04 T6"),
    "t6.within.conformer-eegmamba.media3.r": (0.54, 0.005, "04 T6"),
    "t6.within.conformer-eegmamba.dif_s2026": (0.03, 0.0005, "04 T6"),
    "t6.within.conformer-eegmamba.dif_s2027": (0.029, 0.0005, "04 T6"),
    "t6.within.conformer-eegmamba.dif_s2028": (0.036, 0.0005, "04 T6"),
    "t6.within.conformer-eegmamba.sd_semillas_dif": (0.0038, 5e-05, "04 T6"),
    "t6.within.labram-eegmamba.s2026.mediana": (0.041, 0.0005, "04 T6"),
    "t6.within.labram-eegmamba.s2026.dif": (0.088, 0.0005, "04 T6"),
    "t6.within.labram-eegmamba.s2026.r": (0.73, 0.005, "04 T6"),
    "t6.within.labram-eegmamba.media3.dif": (0.093, 0.0005, "04 T6"),
    "t6.within.labram-eegmamba.media3.mediana": (0.044, 0.0005, "04 T6"),
    "t6.within.labram-eegmamba.media3.r": (0.8, 0.005, "04 T6"),
    "t6.within.labram-eegmamba.dif_s2026": (0.088, 0.0005, "04 T6"),
    "t6.within.labram-eegmamba.dif_s2027": (0.092, 0.0005, "04 T6"),
    "t6.within.labram-eegmamba.dif_s2028": (0.098, 0.0005, "04 T6"),
    "t6.within.labram-eegmamba.sd_semillas_dif": (0.0049, 5e-05, "04 T6"),
    "t7.loso.eegnet-conformer.s2026.ic90_inf": (0.002, 0.0005, "04 T7"),
    "t7.loso.eegnet-conformer.s2026.ic90_sup": (0.02, 0.0005, "04 T7"),
    "t7.loso.eegnet-conformer.media3.ic90_inf": (0.004, 0.0005, "04 T7"),
    "t7.loso.eegnet-conformer.media3.ic90_sup": (0.025, 0.0005, "04 T7"),
    "t7.loso.eegnet-conformer.m2728.ic90_inf": (0.002, 0.0005, "04 T7"),
    "t7.loso.eegnet-conformer.m2728.ic90_sup": (0.029, 0.0005, "04 T7"),
    "t7.loso.eegnet-labram.s2026.ic90_inf": (-0.002, 0.0005, "04 T7"),
    "t7.loso.eegnet-labram.s2026.ic90_sup": (0.02, 0.0005, "04 T7"),
    "t7.loso.eegnet-labram.media3.ic90_inf": (-0.003, 0.0005, "04 T7"),
    "t7.loso.eegnet-labram.media3.ic90_sup": (0.015, 0.0005, "04 T7"),
    "t7.loso.eegnet-labram.m2728.ic90_inf": (-0.006, 0.0005, "04 T7"),
    "t7.loso.eegnet-labram.m2728.ic90_sup": (0.014, 0.0005, "04 T7"),
    "t7.loso.eegnet-labram.s2026.tost_holm": (0.005, 0.0005, "04 T7"),
    "t7.loso.conformer-labram.s2026.ic90_inf": (-0.013, 0.0005, "04 T7"),
    "t7.loso.conformer-labram.s2026.ic90_sup": (0.009, 0.0005, "04 T7"),
    "t7.loso.conformer-labram.media3.ic90_inf": (-0.018, 0.0005, "04 T7"),
    "t7.loso.conformer-labram.media3.ic90_sup": (0.002, 0.0005, "04 T7"),
    "t7.loso.conformer-labram.m2728.ic90_inf": (-0.023, 0.0005, "04 T7"),
    "t7.loso.conformer-labram.m2728.ic90_sup": (0.001, 0.0005, "04 T7"),
    "t7.loso.conformer-labram.s2026.tost_holm": (0.001, 0.0005, "04 T7"),
    "t7.loso.conformer-labram.media3.tost_holm": (0.003, 0.0005, "04 T7"),
    "t7.loso.eegnet-eegmamba.s2026.ic90_inf": (0.035, 0.0005, "04 T7"),
    "t7.loso.eegnet-eegmamba.s2026.ic90_sup": (0.057, 0.0005, "04 T7"),
    "t7.loso.eegnet-eegmamba.media3.ic90_inf": (0.035, 0.0005, "04 T7"),
    "t7.loso.eegnet-eegmamba.media3.ic90_sup": (0.056, 0.0005, "04 T7"),
    "t7.loso.eegnet-eegmamba.m2728.ic90_inf": (0.034, 0.0005, "04 T7"),
    "t7.loso.eegnet-eegmamba.m2728.ic90_sup": (0.056, 0.0005, "04 T7"),
    "t7.loso.eegnet-eegmamba.s2026.tost_holm": (1.0, 0.005, "04 T7"),
    "t7.loso.eegnet-eegmamba.media3.tost_holm": (1.0, 0.005, "04 T7"),
    "t7.loso.conformer-eegmamba.s2026.ic90_inf": (0.023, 0.0005, "04 T7"),
    "t7.loso.conformer-eegmamba.s2026.ic90_sup": (0.046, 0.0005, "04 T7"),
    "t7.loso.conformer-eegmamba.media3.ic90_inf": (0.018, 0.0005, "04 T7"),
    "t7.loso.conformer-eegmamba.media3.ic90_sup": (0.044, 0.0005, "04 T7"),
    "t7.loso.conformer-eegmamba.m2728.ic90_inf": (0.014, 0.0005, "04 T7"),
    "t7.loso.conformer-eegmamba.m2728.ic90_sup": (0.044, 0.0005, "04 T7"),
    "t7.loso.conformer-eegmamba.s2026.tost_holm": (1.0, 0.005, "04 T7"),
    "t7.loso.conformer-eegmamba.media3.tost_holm": (1.0, 0.005, "04 T7"),
    "t7.loso.labram-eegmamba.s2026.ic90_inf": (0.028, 0.0005, "04 T7"),
    "t7.loso.labram-eegmamba.s2026.ic90_sup": (0.046, 0.0005, "04 T7"),
    "t7.loso.labram-eegmamba.media3.ic90_inf": (0.03, 0.0005, "04 T7"),
    "t7.loso.labram-eegmamba.media3.ic90_sup": (0.048, 0.0005, "04 T7"),
    "t7.loso.labram-eegmamba.m2728.ic90_inf": (0.031, 0.0005, "04 T7"),
    "t7.loso.labram-eegmamba.m2728.ic90_sup": (0.05, 0.0005, "04 T7"),
    "t7.loso.labram-eegmamba.s2026.tost_holm": (1.0, 0.005, "04 T7"),
    "t7.loso.labram-eegmamba.media3.tost_holm": (1.0, 0.005, "04 T7"),
    "t7.within.eegnet-conformer.s2026.ic90_inf": (0.014, 0.0005, "04 T7"),
    "t7.within.eegnet-conformer.s2026.ic90_sup": (0.038, 0.0005, "04 T7"),
    "t7.within.eegnet-conformer.media3.ic90_inf": (0.021, 0.0005, "04 T7"),
    "t7.within.eegnet-conformer.media3.ic90_sup": (0.042, 0.0005, "04 T7"),
    "t7.within.eegnet-conformer.m2728.ic90_inf": (0.023, 0.0005, "04 T7"),
    "t7.within.eegnet-conformer.m2728.ic90_sup": (0.045, 0.0005, "04 T7"),
    "t7.within.eegnet-conformer.s2026.tost_holm": (1.0, 0.005, "04 T7"),
    "t7.within.eegnet-conformer.media3.tost_holm": (1.0, 0.005, "04 T7"),
    "t7.within.eegnet-labram.s2026.ic90_inf": (-0.062, 0.0005, "04 T7"),
    "t7.within.eegnet-labram.s2026.ic90_sup": (-0.002, 0.0005, "04 T7"),
    "t7.within.eegnet-labram.media3.ic90_inf": (-0.059, 0.0005, "04 T7"),
    "t7.within.eegnet-labram.media3.ic90_sup": (-0.001, 0.0005, "04 T7"),
    "t7.within.eegnet-labram.m2728.ic90_inf": (-0.058, 0.0005, "04 T7"),
    "t7.within.eegnet-labram.m2728.ic90_sup": (0.001, 0.0005, "04 T7"),
    "t7.within.eegnet-labram.s2026.tost_holm": (1.0, 0.005, "04 T7"),
    "t7.within.eegnet-labram.media3.tost_holm": (1.0, 0.005, "04 T7"),
    "t7.within.conformer-labram.s2026.ic90_inf": (-0.085, 0.0005, "04 T7"),
    "t7.within.conformer-labram.s2026.ic90_sup": (-0.032, 0.0005, "04 T7"),
    "t7.within.conformer-labram.media3.ic90_inf": (-0.087, 0.0005, "04 T7"),
    "t7.within.conformer-labram.media3.ic90_sup": (-0.036, 0.0005, "04 T7"),
    "t7.within.conformer-labram.m2728.ic90_inf": (-0.089, 0.0005, "04 T7"),
    "t7.within.conformer-labram.m2728.ic90_sup": (-0.037, 0.0005, "04 T7"),
    "t7.within.conformer-labram.s2026.tost_holm": (1.0, 0.005, "04 T7"),
    "t7.within.conformer-labram.media3.tost_holm": (1.0, 0.005, "04 T7"),
    "t7.within.eegnet-eegmamba.s2026.ic90_inf": (0.04, 0.0005, "04 T7"),
    "t7.within.eegnet-eegmamba.s2026.ic90_sup": (0.073, 0.0005, "04 T7"),
    "t7.within.eegnet-eegmamba.media3.ic90_inf": (0.049, 0.0005, "04 T7"),
    "t7.within.eegnet-eegmamba.media3.ic90_sup": (0.078, 0.0005, "04 T7"),
    "t7.within.eegnet-eegmamba.m2728.ic90_inf": (0.053, 0.0005, "04 T7"),
    "t7.within.eegnet-eegmamba.m2728.ic90_sup": (0.082, 0.0005, "04 T7"),
    "t7.within.eegnet-eegmamba.s2026.tost_holm": (1.0, 0.005, "04 T7"),
    "t7.within.eegnet-eegmamba.media3.tost_holm": (1.0, 0.005, "04 T7"),
    "t7.within.conformer-eegmamba.s2026.ic90_inf": (0.017, 0.0005, "04 T7"),
    "t7.within.conformer-eegmamba.s2026.ic90_sup": (0.044, 0.0005, "04 T7"),
    "t7.within.conformer-eegmamba.media3.ic90_inf": (0.02, 0.0005, "04 T7"),
    "t7.within.conformer-eegmamba.media3.ic90_sup": (0.044, 0.0005, "04 T7"),
    "t7.within.conformer-eegmamba.m2728.ic90_inf": (0.021, 0.0005, "04 T7"),
    "t7.within.conformer-eegmamba.m2728.ic90_sup": (0.045, 0.0005, "04 T7"),
    "t7.within.conformer-eegmamba.s2026.tost_holm": (1.0, 0.005, "04 T7"),
    "t7.within.conformer-eegmamba.media3.tost_holm": (1.0, 0.005, "04 T7"),
    "t7.within.labram-eegmamba.s2026.ic90_inf": (0.063, 0.0005, "04 T7"),
    "t7.within.labram-eegmamba.s2026.ic90_sup": (0.113, 0.0005, "04 T7"),
    "t7.within.labram-eegmamba.media3.ic90_inf": (0.068, 0.0005, "04 T7"),
    "t7.within.labram-eegmamba.media3.ic90_sup": (0.118, 0.0005, "04 T7"),
    "t7.within.labram-eegmamba.m2728.ic90_inf": (0.07, 0.0005, "04 T7"),
    "t7.within.labram-eegmamba.m2728.ic90_sup": (0.121, 0.0005, "04 T7"),
    "t7.within.labram-eegmamba.s2026.tost_holm": (1.0, 0.005, "04 T7"),
    "t7.within.labram-eegmamba.media3.tost_holm": (1.0, 0.005, "04 T7"),
    "t9.eegnet.s2026.intra": (0.633, 0.0005, "04 T9"),
    "t9.eegnet.s2026.loso": (0.763, 0.0005, "04 T9"),
    "t9.eegnet.s2026.dif": (0.129, 0.0005, "04 T9"),
    "t9.eegnet.s2026.mediana": (0.119, 0.0005, "04 T9"),
    "t9.eegnet.s2026.ic_inf": (0.093, 0.0005, "04 T9"),
    "t9.eegnet.s2026.ic_sup": (0.164, 0.0005, "04 T9"),
    "t9.eegnet.s2026.mejoran": (47.0, 0.5, "04 T9"),
    "t9.eegnet.s2026.r": (0.74, 0.005, "04 T9"),
    "t9.eegnet.s2026.p_lt001": (1.0, 0.5, "04 T9"),
    "t9.eegnet.s2027.intra": (0.631, 0.0005, "04 T9"),
    "t9.eegnet.s2027.loso": (0.76, 0.0005, "04 T9"),
    "t9.eegnet.s2027.dif": (0.129, 0.0005, "04 T9"),
    "t9.eegnet.s2027.mediana": (0.107, 0.0005, "04 T9"),
    "t9.eegnet.s2027.ic_inf": (0.093, 0.0005, "04 T9"),
    "t9.eegnet.s2027.ic_sup": (0.163, 0.0005, "04 T9"),
    "t9.eegnet.s2027.mejoran": (48.0, 0.5, "04 T9"),
    "t9.eegnet.s2027.r": (0.75, 0.005, "04 T9"),
    "t9.eegnet.s2027.p_lt001": (1.0, 0.5, "04 T9"),
    "t9.eegnet.s2028.intra": (0.639, 0.0005, "04 T9"),
    "t9.eegnet.s2028.loso": (0.76, 0.0005, "04 T9"),
    "t9.eegnet.s2028.dif": (0.121, 0.0005, "04 T9"),
    "t9.eegnet.s2028.mediana": (0.107, 0.0005, "04 T9"),
    "t9.eegnet.s2028.ic_inf": (0.086, 0.0005, "04 T9"),
    "t9.eegnet.s2028.ic_sup": (0.155, 0.0005, "04 T9"),
    "t9.eegnet.s2028.mejoran": (47.0, 0.5, "04 T9"),
    "t9.eegnet.s2028.r": (0.72, 0.005, "04 T9"),
    "t9.eegnet.s2028.p_lt001": (1.0, 0.5, "04 T9"),
    "t9.eegnet.media3.intra": (0.634, 0.0005, "04 T9"),
    "t9.eegnet.media3.loso": (0.761, 0.0005, "04 T9"),
    "t9.eegnet.media3.dif": (0.126, 0.0005, "04 T9"),
    "t9.eegnet.media3.mediana": (0.109, 0.0005, "04 T9"),
    "t9.eegnet.media3.ic_inf": (0.092, 0.0005, "04 T9"),
    "t9.eegnet.media3.ic_sup": (0.159, 0.0005, "04 T9"),
    "t9.eegnet.media3.mejoran": (46.0, 0.5, "04 T9"),
    "t9.eegnet.media3.r": (0.75, 0.005, "04 T9"),
    "t9.eegnet.media3.p_lt001": (1.0, 0.5, "04 T9"),
    "t9.eegnet.publicada.intra": (0.634, 0.0005, "04 T9"),
    "t9.eegnet.publicada.loso": (0.763, 0.0005, "04 T9"),
    "t9.eegnet.publicada.dif": (0.128, 0.0005, "04 T9"),
    "t9.eegnet.publicada.mediana": (0.111, 0.0005, "04 T9"),
    "t9.eegnet.publicada.ic_inf": (0.093, 0.0005, "04 T9"),
    "t9.eegnet.publicada.ic_sup": (0.162, 0.0005, "04 T9"),
    "t9.eegnet.publicada.mejoran": (48.0, 0.5, "04 T9"),
    "t9.eegnet.publicada.r": (0.75, 0.005, "04 T9"),
    "t9.eegnet.publicada.p_lt001": (1.0, 0.5, "04 T9"),
    "t9.conformer.s2026.intra": (0.607, 0.0005, "04 T9"),
    "t9.conformer.s2026.loso": (0.752, 0.0005, "04 T9"),
    "t9.conformer.s2026.dif": (0.145, 0.0005, "04 T9"),
    "t9.conformer.s2026.mediana": (0.154, 0.0005, "04 T9"),
    "t9.conformer.s2026.ic_inf": (0.107, 0.0005, "04 T9"),
    "t9.conformer.s2026.ic_sup": (0.181, 0.0005, "04 T9"),
    "t9.conformer.s2026.mejoran": (50.0, 0.5, "04 T9"),
    "t9.conformer.s2026.r": (0.74, 0.005, "04 T9"),
    "t9.conformer.s2026.p_lt001": (1.0, 0.5, "04 T9"),
    "t9.conformer.s2027.intra": (0.6, 0.0005, "04 T9"),
    "t9.conformer.s2027.loso": (0.744, 0.0005, "04 T9"),
    "t9.conformer.s2027.dif": (0.144, 0.0005, "04 T9"),
    "t9.conformer.s2027.mediana": (0.149, 0.0005, "04 T9"),
    "t9.conformer.s2027.ic_inf": (0.105, 0.0005, "04 T9"),
    "t9.conformer.s2027.ic_sup": (0.181, 0.0005, "04 T9"),
    "t9.conformer.s2027.mejoran": (48.0, 0.5, "04 T9"),
    "t9.conformer.s2027.r": (0.74, 0.005, "04 T9"),
    "t9.conformer.s2027.p_lt001": (1.0, 0.5, "04 T9"),
    "t9.conformer.s2028.intra": (0.601, 0.0005, "04 T9"),
    "t9.conformer.s2028.loso": (0.745, 0.0005, "04 T9"),
    "t9.conformer.s2028.dif": (0.144, 0.0005, "04 T9"),
    "t9.conformer.s2028.mediana": (0.154, 0.0005, "04 T9"),
    "t9.conformer.s2028.ic_inf": (0.1, 0.0005, "04 T9"),
    "t9.conformer.s2028.ic_sup": (0.185, 0.0005, "04 T9"),
    "t9.conformer.s2028.mejoran": (47.0, 0.5, "04 T9"),
    "t9.conformer.s2028.r": (0.68, 0.005, "04 T9"),
    "t9.conformer.s2028.p_lt001": (1.0, 0.5, "04 T9"),
    "t9.conformer.media3.intra": (0.602, 0.0005, "04 T9"),
    "t9.conformer.media3.loso": (0.747, 0.0005, "04 T9"),
    "t9.conformer.media3.dif": (0.144, 0.0005, "04 T9"),
    "t9.conformer.media3.mediana": (0.155, 0.0005, "04 T9"),
    "t9.conformer.media3.ic_inf": (0.106, 0.0005, "04 T9"),
    "t9.conformer.media3.ic_sup": (0.18, 0.0005, "04 T9"),
    "t9.conformer.media3.mejoran": (48.0, 0.5, "04 T9"),
    "t9.conformer.media3.r": (0.73, 0.005, "04 T9"),
    "t9.conformer.media3.p_lt001": (1.0, 0.5, "04 T9"),
    "t9.conformer.publicada.intra": (0.602, 0.0005, "04 T9"),
    "t9.conformer.publicada.loso": (0.752, 0.0005, "04 T9"),
    "t9.conformer.publicada.dif": (0.149, 0.0005, "04 T9"),
    "t9.conformer.publicada.mediana": (0.155, 0.0005, "04 T9"),
    "t9.conformer.publicada.ic_inf": (0.113, 0.0005, "04 T9"),
    "t9.conformer.publicada.ic_sup": (0.184, 0.0005, "04 T9"),
    "t9.conformer.publicada.mejoran": (50.0, 0.5, "04 T9"),
    "t9.conformer.publicada.r": (0.75, 0.005, "04 T9"),
    "t9.conformer.publicada.p_lt001": (1.0, 0.5, "04 T9"),
    "t9.labram.s2026.intra": (0.665, 0.0005, "04 T9"),
    "t9.labram.s2026.loso": (0.754, 0.0005, "04 T9"),
    "t9.labram.s2026.dif": (0.089, 0.0005, "04 T9"),
    "t9.labram.s2026.mediana": (0.089, 0.0005, "04 T9"),
    "t9.labram.s2026.ic_inf": (0.059, 0.0005, "04 T9"),
    "t9.labram.s2026.ic_sup": (0.118, 0.0005, "04 T9"),
    "t9.labram.s2026.mejoran": (41.0, 0.5, "04 T9"),
    "t9.labram.s2026.r": (0.64, 0.005, "04 T9"),
    "t9.labram.s2026.p_lt001": (1.0, 0.5, "04 T9"),
    "t9.labram.s2027.intra": (0.663, 0.0005, "04 T9"),
    "t9.labram.s2027.loso": (0.753, 0.0005, "04 T9"),
    "t9.labram.s2027.dif": (0.09, 0.0005, "04 T9"),
    "t9.labram.s2027.mediana": (0.105, 0.0005, "04 T9"),
    "t9.labram.s2027.ic_inf": (0.055, 0.0005, "04 T9"),
    "t9.labram.s2027.ic_sup": (0.121, 0.0005, "04 T9"),
    "t9.labram.s2027.mejoran": (42.0, 0.5, "04 T9"),
    "t9.labram.s2027.r": (0.62, 0.005, "04 T9"),
    "t9.labram.s2027.p_lt001": (1.0, 0.5, "04 T9"),
    "t9.labram.s2028.intra": (0.662, 0.0005, "04 T9"),
    "t9.labram.s2028.loso": (0.758, 0.0005, "04 T9"),
    "t9.labram.s2028.dif": (0.096, 0.0005, "04 T9"),
    "t9.labram.s2028.mediana": (0.102, 0.0005, "04 T9"),
    "t9.labram.s2028.ic_inf": (0.065, 0.0005, "04 T9"),
    "t9.labram.s2028.ic_sup": (0.124, 0.0005, "04 T9"),
    "t9.labram.s2028.mejoran": (44.0, 0.5, "04 T9"),
    "t9.labram.s2028.r": (0.67, 0.005, "04 T9"),
    "t9.labram.s2028.p_lt001": (1.0, 0.5, "04 T9"),
    "t9.labram.media3.intra": (0.663, 0.0005, "04 T9"),
    "t9.labram.media3.loso": (0.755, 0.0005, "04 T9"),
    "t9.labram.media3.dif": (0.092, 0.0005, "04 T9"),
    "t9.labram.media3.mediana": (0.098, 0.0005, "04 T9"),
    "t9.labram.media3.ic_inf": (0.061, 0.0005, "04 T9"),
    "t9.labram.media3.ic_sup": (0.12, 0.0005, "04 T9"),
    "t9.labram.media3.mejoran": (44.0, 0.5, "04 T9"),
    "t9.labram.media3.r": (0.66, 0.005, "04 T9"),
    "t9.labram.media3.p_lt001": (1.0, 0.5, "04 T9"),
    "t9.labram.publicada.intra": (0.665, 0.0005, "04 T9"),
    "t9.labram.publicada.loso": (0.754, 0.0005, "04 T9"),
    "t9.labram.publicada.dif": (0.089, 0.0005, "04 T9"),
    "t9.labram.publicada.mediana": (0.089, 0.0005, "04 T9"),
    "t9.labram.publicada.ic_inf": (0.059, 0.0005, "04 T9"),
    "t9.labram.publicada.ic_sup": (0.118, 0.0005, "04 T9"),
    "t9.labram.publicada.mejoran": (41.0, 0.5, "04 T9"),
    "t9.labram.publicada.r": (0.64, 0.005, "04 T9"),
    "t9.labram.publicada.p_lt001": (1.0, 0.5, "04 T9"),
    "t9.eegmamba.s2026.intra": (0.577, 0.0005, "04 T9"),
    "t9.eegmamba.s2026.loso": (0.717, 0.0005, "04 T9"),
    "t9.eegmamba.s2026.dif": (0.14, 0.0005, "04 T9"),
    "t9.eegmamba.s2026.mediana": (0.147, 0.0005, "04 T9"),
    "t9.eegmamba.s2026.ic_inf": (0.112, 0.0005, "04 T9"),
    "t9.eegmamba.s2026.ic_sup": (0.166, 0.0005, "04 T9"),
    "t9.eegmamba.s2026.mejoran": (49.0, 0.5, "04 T9"),
    "t9.eegmamba.s2026.r": (0.8, 0.005, "04 T9"),
    "t9.eegmamba.s2026.p_lt001": (1.0, 0.5, "04 T9"),
    "t9.eegmamba.s2027.intra": (0.571, 0.0005, "04 T9"),
    "t9.eegmamba.s2027.loso": (0.714, 0.0005, "04 T9"),
    "t9.eegmamba.s2027.dif": (0.143, 0.0005, "04 T9"),
    "t9.eegmamba.s2027.mediana": (0.141, 0.0005, "04 T9"),
    "t9.eegmamba.s2027.ic_inf": (0.115, 0.0005, "04 T9"),
    "t9.eegmamba.s2027.ic_sup": (0.17, 0.0005, "04 T9"),
    "t9.eegmamba.s2027.mejoran": (50.0, 0.5, "04 T9"),
    "t9.eegmamba.s2027.r": (0.83, 0.005, "04 T9"),
    "t9.eegmamba.s2027.p_lt001": (1.0, 0.5, "04 T9"),
    "t9.eegmamba.s2028.intra": (0.564, 0.0005, "04 T9"),
    "t9.eegmamba.s2028.loso": (0.715, 0.0005, "04 T9"),
    "t9.eegmamba.s2028.dif": (0.151, 0.0005, "04 T9"),
    "t9.eegmamba.s2028.mediana": (0.146, 0.0005, "04 T9"),
    "t9.eegmamba.s2028.ic_inf": (0.121, 0.0005, "04 T9"),
    "t9.eegmamba.s2028.ic_sup": (0.178, 0.0005, "04 T9"),
    "t9.eegmamba.s2028.mejoran": (50.0, 0.5, "04 T9"),
    "t9.eegmamba.s2028.r": (0.8, 0.005, "04 T9"),
    "t9.eegmamba.s2028.p_lt001": (1.0, 0.5, "04 T9"),
    "t9.eegmamba.media3.intra": (0.571, 0.0005, "04 T9"),
    "t9.eegmamba.media3.loso": (0.715, 0.0005, "04 T9"),
    "t9.eegmamba.media3.dif": (0.145, 0.0005, "04 T9"),
    "t9.eegmamba.media3.mediana": (0.148, 0.0005, "04 T9"),
    "t9.eegmamba.media3.ic_inf": (0.117, 0.0005, "04 T9"),
    "t9.eegmamba.media3.ic_sup": (0.17, 0.0005, "04 T9"),
    "t9.eegmamba.media3.mejoran": (51.0, 0.5, "04 T9"),
    "t9.eegmamba.media3.r": (0.81, 0.005, "04 T9"),
    "t9.eegmamba.media3.p_lt001": (1.0, 0.5, "04 T9"),
    "t9.eegmamba.publicada.intra": (0.577, 0.0005, "04 T9"),
    "t9.eegmamba.publicada.loso": (0.717, 0.0005, "04 T9"),
    "t9.eegmamba.publicada.dif": (0.14, 0.0005, "04 T9"),
    "t9.eegmamba.publicada.mediana": (0.147, 0.0005, "04 T9"),
    "t9.eegmamba.publicada.ic_inf": (0.112, 0.0005, "04 T9"),
    "t9.eegmamba.publicada.ic_sup": (0.166, 0.0005, "04 T9"),
    "t9.eegmamba.publicada.mejoran": (49.0, 0.5, "04 T9"),
    "t9.eegmamba.publicada.r": (0.8, 0.005, "04 T9"),
    "t9.eegmamba.publicada.p_lt001": (1.0, 0.5, "04 T9"),
    "t10.eegnet.s2026.vistos": (0.783, 0.0005, "04 T10"),
    "t10.eegnet.s2026.novistos": (0.763, 0.0005, "04 T10"),
    "t10.eegnet.s2026.ids_acc": (0.026, 0.0005, "04 T10"),
    "t10.eegnet.s2026.ids_kappa": (0.071, 0.0005, "04 T10"),
    "t10.eegnet.s2026.folds": (33.0, 0.5, "04 T10"),
    "t10.eegnet.s2026.p": (0.23, 0.005, "04 T10"),
    "t10.eegnet.s2026.p_holm": (0.23, 0.005, "04 T10"),
    "t10.eegnet.s2026.r": (0.16, 0.005, "04 T10"),
    "t10.conformer.s2026.vistos": (0.786, 0.0005, "04 T10"),
    "t10.conformer.s2026.novistos": (0.752, 0.0005, "04 T10"),
    "t10.conformer.s2026.ids_acc": (0.044, 0.0005, "04 T10"),
    "t10.conformer.s2026.ids_kappa": (0.121, 0.0005, "04 T10"),
    "t10.conformer.s2026.folds": (34.0, 0.5, "04 T10"),
    "t10.conformer.s2026.p": (0.05, 0.005, "04 T10"),
    "t10.conformer.s2026.p_holm": (0.11, 0.005, "04 T10"),
    "t10.conformer.s2026.r": (0.26, 0.005, "04 T10"),
    "t10.labram.s2026.vistos": (0.798, 0.0005, "04 T10"),
    "t10.labram.s2026.novistos": (0.754, 0.0005, "04 T10"),
    "t10.labram.s2026.ids_acc": (0.055, 0.0005, "04 T10"),
    "t10.labram.s2026.ids_kappa": (0.148, 0.0005, "04 T10"),
    "t10.labram.s2026.folds": (38.0, 0.5, "04 T10"),
    "t10.labram.s2026.p": (0.006, 0.0005, "04 T10"),
    "t10.labram.s2026.p_holm": (0.03, 0.005, "04 T10"),
    "t10.labram.s2026.r": (0.37, 0.005, "04 T10"),
    "t10.eegmamba.s2026.vistos": (0.742, 0.0005, "04 T10"),
    "t10.eegmamba.s2026.novistos": (0.717, 0.0005, "04 T10"),
    "t10.eegmamba.s2026.ids_acc": (0.034, 0.0005, "04 T10"),
    "t10.eegmamba.s2026.ids_kappa": (0.105, 0.0005, "04 T10"),
    "t10.eegmamba.s2026.folds": (37.0, 0.5, "04 T10"),
    "t10.eegmamba.s2026.p": (0.04, 0.005, "04 T10"),
    "t10.eegmamba.s2026.p_holm": (0.11, 0.005, "04 T10"),
    "t10.eegmamba.s2026.r": (0.28, 0.005, "04 T10"),
    "t10.eegnet.s2027.vistos": (0.78, 0.0005, "04 T10"),
    "t10.eegnet.s2027.novistos": (0.76, 0.0005, "04 T10"),
    "t10.eegnet.s2027.ids_acc": (0.026, 0.0005, "04 T10"),
    "t10.eegnet.s2027.ids_kappa": (0.072, 0.0005, "04 T10"),
    "t10.eegnet.s2027.folds": (31.0, 0.5, "04 T10"),
    "t10.eegnet.s2027.p": (0.19, 0.005, "04 T10"),
    "t10.eegnet.s2027.p_holm": (0.37, 0.005, "04 T10"),
    "t10.eegnet.s2027.r": (0.18, 0.005, "04 T10"),
    "t10.conformer.s2027.vistos": (0.768, 0.0005, "04 T10"),
    "t10.conformer.s2027.novistos": (0.744, 0.0005, "04 T10"),
    "t10.conformer.s2027.ids_acc": (0.03, 0.0005, "04 T10"),
    "t10.conformer.s2027.ids_kappa": (0.087, 0.0005, "04 T10"),
    "t10.conformer.s2027.folds": (32.0, 0.5, "04 T10"),
    "t10.conformer.s2027.p": (0.21, 0.005, "04 T10"),
    "t10.conformer.s2027.p_holm": (0.37, 0.005, "04 T10"),
    "t10.conformer.s2027.r": (0.17, 0.005, "04 T10"),
    "t10.labram.s2027.vistos": (0.794, 0.0005, "04 T10"),
    "t10.labram.s2027.novistos": (0.753, 0.0005, "04 T10"),
    "t10.labram.s2027.ids_acc": (0.051, 0.0005, "04 T10"),
    "t10.labram.s2027.ids_kappa": (0.139, 0.0005, "04 T10"),
    "t10.labram.s2027.folds": (37.0, 0.5, "04 T10"),
    "t10.labram.s2027.p": (0.02, 0.005, "04 T10"),
    "t10.labram.s2027.p_holm": (0.07, 0.005, "04 T10"),
    "t10.labram.s2027.r": (0.32, 0.005, "04 T10"),
    "t10.eegmamba.s2027.vistos": (0.741, 0.0005, "04 T10"),
    "t10.eegmamba.s2027.novistos": (0.714, 0.0005, "04 T10"),
    "t10.eegmamba.s2027.ids_acc": (0.037, 0.0005, "04 T10"),
    "t10.eegmamba.s2027.ids_kappa": (0.114, 0.0005, "04 T10"),
    "t10.eegmamba.s2027.folds": (37.0, 0.5, "04 T10"),
    "t10.eegmamba.s2027.p": (0.02, 0.005, "04 T10"),
    "t10.eegmamba.s2027.p_holm": (0.07, 0.005, "04 T10"),
    "t10.eegmamba.s2027.r": (0.32, 0.005, "04 T10"),
    "t10.eegnet.s2028.vistos": (0.779, 0.0005, "04 T10"),
    "t10.eegnet.s2028.novistos": (0.76, 0.0005, "04 T10"),
    "t10.eegnet.s2028.ids_acc": (0.025, 0.0005, "04 T10"),
    "t10.eegnet.s2028.ids_kappa": (0.071, 0.0005, "04 T10"),
    "t10.eegnet.s2028.folds": (30.0, 0.5, "04 T10"),
    "t10.eegnet.s2028.p": (0.26, 0.005, "04 T10"),
    "t10.eegnet.s2028.p_holm": (0.26, 0.005, "04 T10"),
    "t10.eegnet.s2028.r": (0.15, 0.005, "04 T10"),
    "t10.conformer.s2028.vistos": (0.772, 0.0005, "04 T10"),
    "t10.conformer.s2028.novistos": (0.745, 0.0005, "04 T10"),
    "t10.conformer.s2028.ids_acc": (0.035, 0.0005, "04 T10"),
    "t10.conformer.s2028.ids_kappa": (0.1, 0.0005, "04 T10"),
    "t10.conformer.s2028.folds": (34.0, 0.5, "04 T10"),
    "t10.conformer.s2028.p": (0.11, 0.005, "04 T10"),
    "t10.conformer.s2028.p_holm": (0.23, 0.005, "04 T10"),
    "t10.conformer.s2028.r": (0.22, 0.005, "04 T10"),
    "t10.labram.s2028.vistos": (0.797, 0.0005, "04 T10"),
    "t10.labram.s2028.novistos": (0.758, 0.0005, "04 T10"),
    "t10.labram.s2028.ids_acc": (0.049, 0.0005, "04 T10"),
    "t10.labram.s2028.ids_kappa": (0.132, 0.0005, "04 T10"),
    "t10.labram.s2028.folds": (36.0, 0.5, "04 T10"),
    "t10.labram.s2028.p": (0.01, 0.005, "04 T10"),
    "t10.labram.s2028.p_holm": (0.05, 0.005, "04 T10"),
    "t10.labram.s2028.r": (0.34, 0.005, "04 T10"),
    "t10.eegmamba.s2028.vistos": (0.742, 0.0005, "04 T10"),
    "t10.eegmamba.s2028.novistos": (0.715, 0.0005, "04 T10"),
    "t10.eegmamba.s2028.ids_acc": (0.036, 0.0005, "04 T10"),
    "t10.eegmamba.s2028.ids_kappa": (0.11, 0.0005, "04 T10"),
    "t10.eegmamba.s2028.folds": (33.0, 0.5, "04 T10"),
    "t10.eegmamba.s2028.p": (0.05, 0.005, "04 T10"),
    "t10.eegmamba.s2028.p_holm": (0.14, 0.005, "04 T10"),
    "t10.eegmamba.s2028.r": (0.27, 0.005, "04 T10"),
    "t10.eegnet.media3.vistos": (0.781, 0.0005, "04 T10"),
    "t10.eegnet.media3.novistos": (0.761, 0.0005, "04 T10"),
    "t10.eegnet.media3.ids_acc": (0.026, 0.0005, "04 T10"),
    "t10.eegnet.media3.ids_kappa": (0.071, 0.0005, "04 T10"),
    "t10.eegnet.media3.folds": (30.0, 0.5, "04 T10"),
    "t10.eegnet.media3.p": (0.25, 0.005, "04 T10"),
    "t10.eegnet.media3.p_holm": (0.25, 0.005, "04 T10"),
    "t10.eegnet.media3.r": (0.16, 0.005, "04 T10"),
    "t10.eegnet.sd_ids_semillas": (0.0002, 5e-05, "04 T10"),
    "t10.conformer.media3.vistos": (0.775, 0.0005, "04 T10"),
    "t10.conformer.media3.novistos": (0.747, 0.0005, "04 T10"),
    "t10.conformer.media3.ids_acc": (0.037, 0.0005, "04 T10"),
    "t10.conformer.media3.ids_kappa": (0.103, 0.0005, "04 T10"),
    "t10.conformer.media3.folds": (34.0, 0.5, "04 T10"),
    "t10.conformer.media3.p": (0.1, 0.005, "04 T10"),
    "t10.conformer.media3.p_holm": (0.2, 0.005, "04 T10"),
    "t10.conformer.media3.r": (0.22, 0.005, "04 T10"),
    "t10.conformer.sd_ids_semillas": (0.0069, 5e-05, "04 T10"),
    "t10.labram.media3.vistos": (0.796, 0.0005, "04 T10"),
    "t10.labram.media3.novistos": (0.755, 0.0005, "04 T10"),
    "t10.labram.media3.ids_acc": (0.052, 0.0005, "04 T10"),
    "t10.labram.media3.ids_kappa": (0.14, 0.0005, "04 T10"),
    "t10.labram.media3.folds": (37.0, 0.5, "04 T10"),
    "t10.labram.media3.p": (0.008, 0.0005, "04 T10"),
    "t10.labram.media3.p_holm": (0.03, 0.005, "04 T10"),
    "t10.labram.media3.r": (0.36, 0.005, "04 T10"),
    "t10.labram.sd_ids_semillas": (0.0031, 5e-05, "04 T10"),
    "t10.eegmamba.media3.vistos": (0.742, 0.0005, "04 T10"),
    "t10.eegmamba.media3.novistos": (0.715, 0.0005, "04 T10"),
    "t10.eegmamba.media3.ids_acc": (0.036, 0.0005, "04 T10"),
    "t10.eegmamba.media3.ids_kappa": (0.11, 0.0005, "04 T10"),
    "t10.eegmamba.media3.folds": (37.0, 0.5, "04 T10"),
    "t10.eegmamba.media3.p": (0.03, 0.005, "04 T10"),
    "t10.eegmamba.media3.p_holm": (0.08, 0.005, "04 T10"),
    "t10.eegmamba.media3.r": (0.3, 0.005, "04 T10"),
    "t10.eegmamba.sd_ids_semillas": (0.0014, 5e-05, "04 T10"),
    "t11.eegnet.s2026.persona": (0.742, 0.0005, "04 T11"),
    "t11.eegnet.s2026.wbcic": (0.621, 0.0005, "04 T11"),
    "t11.eegnet.s2026.idp": (0.163, 0.0005, "04 T11"),
    "t11.eegnet.s2027.persona": (0.729, 0.0005, "04 T11"),
    "t11.eegnet.s2027.wbcic": (0.619, 0.0005, "04 T11"),
    "t11.eegnet.s2027.idp": (0.151, 0.0005, "04 T11"),
    "t11.eegnet.s2028.persona": (0.744, 0.0005, "04 T11"),
    "t11.eegnet.s2028.wbcic": (0.63, 0.0005, "04 T11"),
    "t11.eegnet.s2028.idp": (0.153, 0.0005, "04 T11"),
    "t11.eegnet.media3.persona": (0.738, 0.0005, "04 T11"),
    "t11.eegnet.media3.wbcic": (0.623, 0.0005, "04 T11"),
    "t11.eegnet.media3.idp": (0.156, 0.0005, "04 T11"),
    "t11.eegnet.media3.sd": (0.0065, 5e-05, "04 T11"),
    "t11.conformer.s2026.persona": (0.718, 0.0005, "04 T11"),
    "t11.conformer.s2026.wbcic": (0.618, 0.0005, "04 T11"),
    "t11.conformer.s2026.idp": (0.139, 0.0005, "04 T11"),
    "t11.conformer.s2027.persona": (0.714, 0.0005, "04 T11"),
    "t11.conformer.s2027.wbcic": (0.622, 0.0005, "04 T11"),
    "t11.conformer.s2027.idp": (0.13, 0.0005, "04 T11"),
    "t11.conformer.s2028.persona": (0.735, 0.0005, "04 T11"),
    "t11.conformer.s2028.wbcic": (0.615, 0.0005, "04 T11"),
    "t11.conformer.s2028.idp": (0.164, 0.0005, "04 T11"),
    "t11.conformer.media3.persona": (0.722, 0.0005, "04 T11"),
    "t11.conformer.media3.wbcic": (0.618, 0.0005, "04 T11"),
    "t11.conformer.media3.idp": (0.144, 0.0005, "04 T11"),
    "t11.conformer.media3.sd": (0.0176, 5e-05, "04 T11"),
    "t11.labram.s2026.persona": (0.745, 0.0005, "04 T11"),
    "t11.labram.s2026.wbcic": (0.628, 0.0005, "04 T11"),
    "t11.labram.s2026.idp": (0.157, 0.0005, "04 T11"),
    "t11.labram.s2027.persona": (0.739, 0.0005, "04 T11"),
    "t11.labram.s2027.wbcic": (0.611, 0.0005, "04 T11"),
    "t11.labram.s2027.idp": (0.173, 0.0005, "04 T11"),
    "t11.labram.s2028.persona": (0.734, 0.0005, "04 T11"),
    "t11.labram.s2028.wbcic": (0.608, 0.0005, "04 T11"),
    "t11.labram.s2028.idp": (0.172, 0.0005, "04 T11"),
    "t11.labram.media3.persona": (0.739, 0.0005, "04 T11"),
    "t11.labram.media3.wbcic": (0.616, 0.0005, "04 T11"),
    "t11.labram.media3.idp": (0.167, 0.0005, "04 T11"),
    "t11.labram.media3.sd": (0.0089, 5e-05, "04 T11"),
    "t11.eegmamba.s2026.persona": (0.7, 0.0005, "04 T11"),
    "t11.eegmamba.s2026.wbcic": (0.598, 0.0005, "04 T11"),
    "t11.eegmamba.s2026.idp": (0.146, 0.0005, "04 T11"),
    "t11.eegmamba.s2027.persona": (0.693, 0.0005, "04 T11"),
    "t11.eegmamba.s2027.wbcic": (0.595, 0.0005, "04 T11"),
    "t11.eegmamba.s2027.idp": (0.142, 0.0005, "04 T11"),
    "t11.eegmamba.s2028.persona": (0.684, 0.0005, "04 T11"),
    "t11.eegmamba.s2028.wbcic": (0.6, 0.0005, "04 T11"),
    "t11.eegmamba.s2028.idp": (0.122, 0.0005, "04 T11"),
    "t11.eegmamba.media3.persona": (0.692, 0.0005, "04 T11"),
    "t11.eegmamba.media3.wbcic": (0.598, 0.0005, "04 T11"),
    "t11.eegmamba.media3.idp": (0.137, 0.0005, "04 T11"),
    "t11.eegmamba.media3.sd": (0.0129, 5e-05, "04 T11"),
    "t11.eegnet.cociente_idp_ids": (6.1, 0.05, "04 sec. 3.3 (texto)"),
    "t11.conformer.cociente_idp_ids": (3.9, 0.05, "04 sec. 3.3 (texto)"),
    "t11.labram.cociente_idp_ids": (3.2, 0.05, "04 sec. 3.3 (texto)"),
    "t11.eegmamba.cociente_idp_ids": (3.8, 0.05, "04 sec. 3.3 (texto)"),
    "t12.within.eegnet.s2026": (42.0, 0.5, "04 T12"),
    "t12.loso.eegnet.s2026": (17.0, 0.5, "04 T12"),
    "t12.within.conformer.s2026": (44.0, 0.5, "04 T12"),
    "t12.loso.conformer.s2026": (18.0, 0.5, "04 T12"),
    "t12.within.labram.s2026": (37.0, 0.5, "04 T12"),
    "t12.loso.labram.s2026": (15.0, 0.5, "04 T12"),
    "t12.within.eegmamba.s2026": (48.0, 0.5, "04 T12"),
    "t12.loso.eegmamba.s2026": (28.0, 0.5, "04 T12"),
    "t12.within.eegnet.s2027": (41.0, 0.5, "04 T12"),
    "t12.loso.eegnet.s2027": (17.0, 0.5, "04 T12"),
    "t12.within.conformer.s2027": (46.0, 0.5, "04 T12"),
    "t12.loso.conformer.s2027": (20.0, 0.5, "04 T12"),
    "t12.within.labram.s2027": (36.0, 0.5, "04 T12"),
    "t12.loso.labram.s2027": (16.0, 0.5, "04 T12"),
    "t12.within.eegmamba.s2027": (49.0, 0.5, "04 T12"),
    "t12.loso.eegmamba.s2027": (27.0, 0.5, "04 T12"),
    "t12.within.eegnet.s2028": (39.0, 0.5, "04 T12"),
    "t12.loso.eegnet.s2028": (20.0, 0.5, "04 T12"),
    "t12.within.conformer.s2028": (43.0, 0.5, "04 T12"),
    "t12.loso.conformer.s2028": (19.0, 0.5, "04 T12"),
    "t12.within.labram.s2028": (38.0, 0.5, "04 T12"),
    "t12.loso.labram.s2028": (16.0, 0.5, "04 T12"),
    "t12.within.eegmamba.s2028": (49.0, 0.5, "04 T12"),
    "t12.loso.eegmamba.s2028": (28.0, 0.5, "04 T12"),
    "t12.within.eegnet.media3": (41.0, 0.5, "04 T12"),
    "t12.loso.eegnet.media3": (17.0, 0.5, "04 T12"),
    "t12.within.conformer.media3": (45.0, 0.5, "04 T12"),
    "t12.loso.conformer.media3": (19.0, 0.5, "04 T12"),
    "t12.within.labram.media3": (37.0, 0.5, "04 T12"),
    "t12.loso.labram.media3": (15.0, 0.5, "04 T12"),
    "t12.within.eegmamba.media3": (49.0, 0.5, "04 T12"),
    "t12.loso.eegmamba.media3": (26.0, 0.5, "04 T12"),
    "t13.s2026.k0.obs": (4.0, 0.5, "04 T13"),
    "t13.s2026.k0.esp": (0.1, 0.05, "04 T13"),
    "t13.s2026.k1.obs": (4.0, 0.5, "04 T13"),
    "t13.s2026.k1.esp": (1.4, 0.05, "04 T13"),
    "t13.s2026.k2.obs": (5.0, 0.5, "04 T13"),
    "t13.s2026.k2.esp": (8.8, 0.05, "04 T13"),
    "t13.s2026.k3.obs": (7.0, 0.5, "04 T13"),
    "t13.s2026.k3.esp": (22.9, 0.05, "04 T13"),
    "t13.s2026.k4.obs": (34.0, 0.5, "04 T13"),
    "t13.s2026.k4.esp": (20.8, 0.05, "04 T13"),
    "t13.s2027.k0.obs": (5.0, 0.5, "04 T13"),
    "t13.s2027.k0.esp": (0.1, 0.05, "04 T13"),
    "t13.s2027.k1.obs": (1.0, 0.5, "04 T13"),
    "t13.s2027.k1.esp": (1.2, 0.05, "04 T13"),
    "t13.s2027.k2.obs": (6.0, 0.5, "04 T13"),
    "t13.s2027.k2.esp": (8.5, 0.05, "04 T13"),
    "t13.s2027.k3.obs": (9.0, 0.5, "04 T13"),
    "t13.s2027.k3.esp": (23.1, 0.05, "04 T13"),
    "t13.s2027.k4.obs": (33.0, 0.5, "04 T13"),
    "t13.s2027.k4.esp": (21.1, 0.05, "04 T13"),
    "t13.s2028.k0.obs": (4.0, 0.5, "04 T13"),
    "t13.s2028.k0.esp": (0.1, 0.05, "04 T13"),
    "t13.s2028.k1.obs": (5.0, 0.5, "04 T13"),
    "t13.s2028.k1.esp": (1.6, 0.05, "04 T13"),
    "t13.s2028.k2.obs": (4.0, 0.5, "04 T13"),
    "t13.s2028.k2.esp": (9.4, 0.05, "04 T13"),
    "t13.s2028.k3.obs": (8.0, 0.5, "04 T13"),
    "t13.s2028.k3.esp": (23.1, 0.05, "04 T13"),
    "t13.s2028.k4.obs": (33.0, 0.5, "04 T13"),
    "t13.s2028.k4.esp": (19.8, 0.05, "04 T13"),
    "t13.media3.k0.obs": (5.0, 0.5, "04 T13"),
    "t13.media3.k0.esp": (0.1, 0.05, "04 T13"),
    "t13.media3.k1.obs": (2.0, 0.5, "04 T13"),
    "t13.media3.k1.esp": (1.3, 0.05, "04 T13"),
    "t13.media3.k2.obs": (5.0, 0.5, "04 T13"),
    "t13.media3.k2.esp": (8.5, 0.05, "04 T13"),
    "t13.media3.k3.obs": (8.0, 0.5, "04 T13"),
    "t13.media3.k3.esp": (22.9, 0.05, "04 T13"),
    "t13.media3.k4.obs": (34.0, 0.5, "04 T13"),
    "t13.media3.k4.esp": (21.2, 0.05, "04 T13"),
    "t15.eegnet.s2026.r": (0.317, 0.0005, "04 T15"),
    "t15.conformer.s2026.r": (0.354, 0.0005, "04 T15"),
    "t15.labram.s2026.r": (0.686, 0.0005, "04 T15"),
    "t15.eegmamba.s2026.r": (0.509, 0.0005, "04 T15"),
    "t15.eegnet.s2027.r": (0.381, 0.0005, "04 T15"),
    "t15.conformer.s2027.r": (0.324, 0.0005, "04 T15"),
    "t15.labram.s2027.r": (0.619, 0.0005, "04 T15"),
    "t15.eegmamba.s2027.r": (0.485, 0.0005, "04 T15"),
    "t15.eegnet.s2028.r": (0.342, 0.0005, "04 T15"),
    "t15.conformer.s2028.r": (0.179, 0.0005, "04 T15"),
    "t15.labram.s2028.r": (0.688, 0.0005, "04 T15"),
    "t15.eegmamba.s2028.r": (0.398, 0.0005, "04 T15"),
    "t15.eegnet.media3.r": (0.364, 0.0005, "04 T15"),
    "t15.conformer.media3.r": (0.309, 0.0005, "04 T15"),
    "t15.labram.media3.r": (0.679, 0.0005, "04 T15"),
    "t15.eegmamba.media3.r": (0.485, 0.0005, "04 T15"),
    "f4.eegnet.mu.ref": (0.6331, 5e-05, "05 T1"),
    "f4.eegnet.mu.var": (0.6031, 5e-05, "05 T1"),
    "f4.eegnet.mu.dif": (-0.03, 5e-05, "05 T1"),
    "f4.eegnet.mu.dif_med": (-0.0438, 5e-05, "05 T1"),
    "f4.eegnet.mu.ic_inf": (-0.0635, 5e-05, "05 T1"),
    "f4.eegnet.mu.ic_sup": (0.0032, 5e-05, "05 T1"),
    "f4.eegnet.mu.mejoran": (20.0, 0.5, "05 T1"),
    "f4.eegnet.mu.empeoran": (33.0, 0.5, "05 T1"),
    "f4.eegnet.mu.p": (0.097, 0.0005, "05 T1"),
    "f4.eegnet.mu.p_holm": (0.487, 0.0005, "05 T1"),
    "f4.eegnet.mu.r": (0.23, 0.005, "05 T1"),
    "f4.eegnet.mu.margen": (0.058, 0.0005, "05 T1"),
    "f4.eegnet.mu.equiv": (0.0, 0.5, "05 T1"),
    "f4.eegnet.mu.p_holm22": (0.974, 0.0005, "05 T1"),
    "f4.eegnet.beta.ref": (0.6331, 5e-05, "05 T1"),
    "f4.eegnet.beta.var": (0.5875, 5e-05, "05 T1"),
    "f4.eegnet.beta.dif": (-0.0456, 5e-05, "05 T1"),
    "f4.eegnet.beta.dif_med": (-0.0587, 5e-05, "05 T1"),
    "f4.eegnet.beta.ic_inf": (-0.0788, 5e-05, "05 T1"),
    "f4.eegnet.beta.ic_sup": (-0.0098, 5e-05, "05 T1"),
    "f4.eegnet.beta.mejoran": (15.0, 0.5, "05 T1"),
    "f4.eegnet.beta.empeoran": (39.0, 0.5, "05 T1"),
    "f4.eegnet.beta.p": (0.001, 0.0005, "05 T1"),
    "f4.eegnet.beta.p_holm": (0.009, 0.0005, "05 T1"),
    "f4.eegnet.beta.r": (0.43, 0.005, "05 T1"),
    "f4.eegnet.beta.margen": (0.073, 0.0005, "05 T1"),
    "f4.eegnet.beta.equiv": (0.0, 0.5, "05 T1"),
    "f4.eegnet.beta.p_holm22": (0.025, 0.0005, "05 T1"),
    "f4.eegnet.mubeta.ref": (0.6331, 5e-05, "05 T1"),
    "f4.eegnet.mubeta.var": (0.6271, 5e-05, "05 T1"),
    "f4.eegnet.mubeta.dif": (-0.006, 5e-05, "05 T1"),
    "f4.eegnet.mubeta.dif_med": (-0.04, 5e-05, "05 T1"),
    "f4.eegnet.mubeta.ic_inf": (-0.045, 5e-05, "05 T1"),
    "f4.eegnet.mubeta.ic_sup": (0.0342, 5e-05, "05 T1"),
    "f4.eegnet.mubeta.mejoran": (22.0, 0.5, "05 T1"),
    "f4.eegnet.mubeta.empeoran": (31.0, 0.5, "05 T1"),
    "f4.eegnet.mubeta.p": (0.449, 0.0005, "05 T1"),
    "f4.eegnet.mubeta.p_holm": (0.897, 0.0005, "05 T1"),
    "f4.eegnet.mubeta.r": (0.1, 0.005, "05 T1"),
    "f4.eegnet.mubeta.margen": (0.039, 0.0005, "05 T1"),
    "f4.eegnet.mubeta.equiv": (0.0, 0.5, "05 T1"),
    "f4.eegnet.mubeta.p_holm22": (1.0, 0.0005, "05 T1"),
    "f4.eegnet.w02.ref": (0.6331, 5e-05, "05 T1"),
    "f4.eegnet.w02.var": (0.6748, 5e-05, "05 T1"),
    "f4.eegnet.w02.dif": (0.0417, 5e-05, "05 T1"),
    "f4.eegnet.w02.dif_med": (0.0475, 5e-05, "05 T1"),
    "f4.eegnet.w02.ic_inf": (0.0294, 5e-05, "05 T1"),
    "f4.eegnet.w02.ic_sup": (0.0541, 5e-05, "05 T1"),
    "f4.eegnet.w02.mejoran": (46.0, 0.5, "05 T1"),
    "f4.eegnet.w02.empeoran": (8.0, 0.5, "05 T1"),
    "f4.eegnet.w02.p": (5.4e-07, 5e-09, "05 T1"),
    "f4.eegnet.w02.p_holm": (4.3e-06, 5e-08, "05 T1"),
    "f4.eegnet.w02.r": (0.68, 0.005, "05 T1"),
    "f4.eegnet.w02.margen": (0.052, 0.0005, "05 T1"),
    "f4.eegnet.w02.equiv": (0.0, 0.5, "05 T1"),
    "f4.eegnet.w02.p_holm22": (1.2e-05, 5.000000000000001e-07, "05 T1"),
    "f4.eegnet.w03.ref": (0.6331, 5e-05, "05 T1"),
    "f4.eegnet.w03.var": (0.6376, 5e-05, "05 T1"),
    "f4.eegnet.w03.dif": (0.0045, 5e-05, "05 T1"),
    "f4.eegnet.w03.dif_med": (0.005, 5e-05, "05 T1"),
    "f4.eegnet.w03.ic_inf": (-0.0054, 5e-05, "05 T1"),
    "f4.eegnet.w03.ic_sup": (0.0145, 5e-05, "05 T1"),
    "f4.eegnet.w03.mejoran": (30.0, 0.5, "05 T1"),
    "f4.eegnet.w03.empeoran": (22.0, 0.5, "05 T1"),
    "f4.eegnet.w03.p": (0.276, 0.0005, "05 T1"),
    "f4.eegnet.w03.p_holm": (0.827, 0.0005, "05 T1"),
    "f4.eegnet.w03.r": (0.15, 0.005, "05 T1"),
    "f4.eegnet.w03.margen": (0.013, 0.0005, "05 T1"),
    "f4.eegnet.w03.equiv": (1.0, 0.5, "05 T1"),
    "f4.eegnet.w03.p_holm22": (1.0, 0.0005, "05 T1"),
    "f4.eegnet.augnone.ref": (0.6331, 5e-05, "05 T1"),
    "f4.eegnet.augnone.var": (0.6405, 5e-05, "05 T1"),
    "f4.eegnet.augnone.dif": (0.0073, 5e-05, "05 T1"),
    "f4.eegnet.augnone.dif_med": (0.01, 5e-05, "05 T1"),
    "f4.eegnet.augnone.ic_inf": (-0.0046, 5e-05, "05 T1"),
    "f4.eegnet.augnone.ic_sup": (0.02, 5e-05, "05 T1"),
    "f4.eegnet.augnone.mejoran": (31.0, 0.5, "05 T1"),
    "f4.eegnet.augnone.empeoran": (21.0, 0.5, "05 T1"),
    "f4.eegnet.augnone.p": (0.198, 0.0005, "05 T1"),
    "f4.eegnet.augnone.p_holm": (0.791, 0.0005, "05 T1"),
    "f4.eegnet.augnone.r": (0.18, 0.005, "05 T1"),
    "f4.eegnet.augnone.margen": (0.018, 0.0005, "05 T1"),
    "f4.eegnet.augnone.equiv": (1.0, 0.5, "05 T1"),
    "f4.eegnet.augnone.p_holm22": (1.0, 0.0005, "05 T1"),
    "f4.eegnet.augcr.ref": (0.6331, 5e-05, "05 T1"),
    "f4.eegnet.augcr.var": (0.6591, 5e-05, "05 T1"),
    "f4.eegnet.augcr.dif": (0.026, 5e-05, "05 T1"),
    "f4.eegnet.augcr.dif_med": (0.0238, 5e-05, "05 T1"),
    "f4.eegnet.augcr.ic_inf": (0.012, 5e-05, "05 T1"),
    "f4.eegnet.augcr.ic_sup": (0.042, 5e-05, "05 T1"),
    "f4.eegnet.augcr.mejoran": (35.0, 0.5, "05 T1"),
    "f4.eegnet.augcr.empeoran": (19.0, 0.5, "05 T1"),
    "f4.eegnet.augcr.p": (0.0007400000000000001, 5e-06, "05 T1"),
    "f4.eegnet.augcr.p_holm": (0.005, 0.0005, "05 T1"),
    "f4.eegnet.augcr.r": (0.46, 0.005, "05 T1"),
    "f4.eegnet.augcr.margen": (0.039, 0.0005, "05 T1"),
    "f4.eegnet.augcr.equiv": (0.0, 0.5, "05 T1"),
    "f4.eegnet.augcr.p_holm22": (0.014, 0.0005, "05 T1"),
    "f4.eegnet.augfft.ref": (0.6331, 5e-05, "05 T1"),
    "f4.eegnet.augfft.var": (0.635, 5e-05, "05 T1"),
    "f4.eegnet.augfft.dif": (0.0019, 5e-05, "05 T1"),
    "f4.eegnet.augfft.dif_med": (0.0, 5e-05, "05 T1"),
    "f4.eegnet.augfft.ic_inf": (-0.0035, 5e-05, "05 T1"),
    "f4.eegnet.augfft.ic_sup": (0.0074, 5e-05, "05 T1"),
    "f4.eegnet.augfft.mejoran": (26.0, 0.5, "05 T1"),
    "f4.eegnet.augfft.empeoran": (21.0, 0.5, "05 T1"),
    "f4.eegnet.augfft.p": (0.561, 0.0005, "05 T1"),
    "f4.eegnet.augfft.p_holm": (0.897, 0.0005, "05 T1"),
    "f4.eegnet.augfft.r": (0.08, 0.005, "05 T1"),
    "f4.eegnet.augfft.margen": (0.007, 0.0005, "05 T1"),
    "f4.eegnet.augfft.equiv": (1.0, 0.5, "05 T1"),
    "f4.eegnet.augfft.p_holm22": (1.0, 0.0005, "05 T1"),
    "f4.conformer.mu.ref": (0.6069, 5e-05, "05 T1"),
    "f4.conformer.mu.var": (0.6528, 5e-05, "05 T1"),
    "f4.conformer.mu.dif": (0.0459, 5e-05, "05 T1"),
    "f4.conformer.mu.dif_med": (0.0225, 5e-05, "05 T1"),
    "f4.conformer.mu.ic_inf": (0.0084, 5e-05, "05 T1"),
    "f4.conformer.mu.ic_sup": (0.085, 5e-05, "05 T1"),
    "f4.conformer.mu.mejoran": (29.0, 0.5, "05 T1"),
    "f4.conformer.mu.empeoran": (25.0, 0.5, "05 T1"),
    "f4.conformer.mu.p": (0.056, 0.0005, "05 T1"),
    "f4.conformer.mu.p_holm": (0.233, 0.0005, "05 T1"),
    "f4.conformer.mu.r": (0.26, 0.005, "05 T1"),
    "f4.conformer.mu.margen": (0.078, 0.0005, "05 T1"),
    "f4.conformer.mu.equiv": (0.0, 0.5, "05 T1"),
    "f4.conformer.mu.p_holm22": (0.671, 0.0005, "05 T1"),
    "f4.conformer.beta.ref": (0.6069, 5e-05, "05 T1"),
    "f4.conformer.beta.var": (0.6424, 5e-05, "05 T1"),
    "f4.conformer.beta.dif": (0.0355, 5e-05, "05 T1"),
    "f4.conformer.beta.dif_med": (0.0113, 5e-05, "05 T1"),
    "f4.conformer.beta.ic_inf": (0.0008, 5e-05, "05 T1"),
    "f4.conformer.beta.ic_sup": (0.0723, 5e-05, "05 T1"),
    "f4.conformer.beta.mejoran": (29.0, 0.5, "05 T1"),
    "f4.conformer.beta.empeoran": (24.0, 0.5, "05 T1"),
    "f4.conformer.beta.p": (0.15, 0.0005, "05 T1"),
    "f4.conformer.beta.p_holm": (0.451, 0.0005, "05 T1"),
    "f4.conformer.beta.r": (0.2, 0.005, "05 T1"),
    "f4.conformer.beta.margen": (0.066, 0.0005, "05 T1"),
    "f4.conformer.beta.equiv": (0.0, 0.5, "05 T1"),
    "f4.conformer.beta.p_holm22": (1.0, 0.0005, "05 T1"),
    "f4.conformer.mubeta.ref": (0.6069, 5e-05, "05 T1"),
    "f4.conformer.mubeta.var": (0.685, 5e-05, "05 T1"),
    "f4.conformer.mubeta.dif": (0.0781, 5e-05, "05 T1"),
    "f4.conformer.mubeta.dif_med": (0.04, 5e-05, "05 T1"),
    "f4.conformer.mubeta.ic_inf": (0.0407, 5e-05, "05 T1"),
    "f4.conformer.mubeta.ic_sup": (0.1165, 5e-05, "05 T1"),
    "f4.conformer.mubeta.mejoran": (39.0, 0.5, "05 T1"),
    "f4.conformer.mubeta.empeoran": (13.0, 0.5, "05 T1"),
    "f4.conformer.mubeta.p": (0.00027000000000000006, 5e-06, "05 T1"),
    "f4.conformer.mubeta.p_holm": (0.002, 0.0005, "05 T1"),
    "f4.conformer.mubeta.r": (0.5, 0.005, "05 T1"),
    "f4.conformer.mubeta.margen": (0.111, 0.0005, "05 T1"),
    "f4.conformer.mubeta.equiv": (0.0, 0.5, "05 T1"),
    "f4.conformer.mubeta.p_holm22": (0.005, 0.0005, "05 T1"),
    "f4.conformer.w02.ref": (0.6069, 5e-05, "05 T1"),
    "f4.conformer.w02.var": (0.6025, 5e-05, "05 T1"),
    "f4.conformer.w02.dif": (-0.0044, 5e-05, "05 T1"),
    "f4.conformer.w02.dif_med": (0.005, 5e-05, "05 T1"),
    "f4.conformer.w02.ic_inf": (-0.0193, 5e-05, "05 T1"),
    "f4.conformer.w02.ic_sup": (0.009, 5e-05, "05 T1"),
    "f4.conformer.w02.mejoran": (30.0, 0.5, "05 T1"),
    "f4.conformer.w02.empeoran": (23.0, 0.5, "05 T1"),
    "f4.conformer.w02.p": (0.737, 0.0005, "05 T1"),
    "f4.conformer.w02.p_holm": (1.0, 0.0005, "05 T1"),
    "f4.conformer.w02.r": (0.05, 0.005, "05 T1"),
    "f4.conformer.w02.margen": (0.017, 0.0005, "05 T1"),
    "f4.conformer.w02.equiv": (1.0, 0.5, "05 T1"),
    "f4.conformer.w02.p_holm22": (1.0, 0.0005, "05 T1"),
    "f4.conformer.w03.ref": (0.6069, 5e-05, "05 T1"),
    "f4.conformer.w03.var": (0.5972, 5e-05, "05 T1"),
    "f4.conformer.w03.dif": (-0.0097, 5e-05, "05 T1"),
    "f4.conformer.w03.dif_med": (-0.0112, 5e-05, "05 T1"),
    "f4.conformer.w03.ic_inf": (-0.0175, 5e-05, "05 T1"),
    "f4.conformer.w03.ic_sup": (-0.0022, 5e-05, "05 T1"),
    "f4.conformer.w03.mejoran": (18.0, 0.5, "05 T1"),
    "f4.conformer.w03.empeoran": (33.0, 0.5, "05 T1"),
    "f4.conformer.w03.p": (0.021, 0.0005, "05 T1"),
    "f4.conformer.w03.p_holm": (0.127, 0.0005, "05 T1"),
    "f4.conformer.w03.r": (0.31, 0.005, "05 T1"),
    "f4.conformer.w03.margen": (0.016, 0.0005, "05 T1"),
    "f4.conformer.w03.equiv": (1.0, 0.5, "05 T1"),
    "f4.conformer.w03.p_holm22": (0.297, 0.0005, "05 T1"),
    "f4.conformer.augnone.ref": (0.6069, 5e-05, "05 T1"),
    "f4.conformer.augnone.var": (0.6001, 5e-05, "05 T1"),
    "f4.conformer.augnone.dif": (-0.0068, 5e-05, "05 T1"),
    "f4.conformer.augnone.dif_med": (-0.005, 5e-05, "05 T1"),
    "f4.conformer.augnone.ic_inf": (-0.0126, 5e-05, "05 T1"),
    "f4.conformer.augnone.ic_sup": (-0.0009, 5e-05, "05 T1"),
    "f4.conformer.augnone.mejoran": (21.0, 0.5, "05 T1"),
    "f4.conformer.augnone.empeoran": (29.0, 0.5, "05 T1"),
    "f4.conformer.augnone.p": (0.047, 0.0005, "05 T1"),
    "f4.conformer.augnone.p_holm": (0.233, 0.0005, "05 T1"),
    "f4.conformer.augnone.r": (0.27, 0.005, "05 T1"),
    "f4.conformer.augnone.margen": (0.012, 0.0005, "05 T1"),
    "f4.conformer.augnone.equiv": (1.0, 0.5, "05 T1"),
    "f4.conformer.augnone.p_holm22": (0.605, 0.0005, "05 T1"),
    "f4.conformer.augcr.ref": (0.6069, 5e-05, "05 T1"),
    "f4.conformer.augcr.var": (0.6288, 5e-05, "05 T1"),
    "f4.conformer.augcr.dif": (0.0219, 5e-05, "05 T1"),
    "f4.conformer.augcr.dif_med": (0.0213, 5e-05, "05 T1"),
    "f4.conformer.augcr.ic_inf": (0.012, 5e-05, "05 T1"),
    "f4.conformer.augcr.ic_sup": (0.0318, 5e-05, "05 T1"),
    "f4.conformer.augcr.mejoran": (37.0, 0.5, "05 T1"),
    "f4.conformer.augcr.empeoran": (16.0, 0.5, "05 T1"),
    "f4.conformer.augcr.p": (0.00024, 5e-06, "05 T1"),
    "f4.conformer.augcr.p_holm": (0.002, 0.0005, "05 T1"),
    "f4.conformer.augcr.r": (0.5, 0.005, "05 T1"),
    "f4.conformer.augcr.margen": (0.03, 0.0005, "05 T1"),
    "f4.conformer.augcr.equiv": (0.0, 0.5, "05 T1"),
    "f4.conformer.augcr.p_holm22": (0.005, 0.0005, "05 T1"),
    "f4.conformer.augfft.ref": (0.6069, 5e-05, "05 T1"),
    "f4.conformer.augfft.var": (0.6079, 5e-05, "05 T1"),
    "f4.conformer.augfft.dif": (0.001, 5e-05, "05 T1"),
    "f4.conformer.augfft.dif_med": (-0.0025, 5e-05, "05 T1"),
    "f4.conformer.augfft.ic_inf": (-0.0049, 5e-05, "05 T1"),
    "f4.conformer.augfft.ic_sup": (0.0071, 5e-05, "05 T1"),
    "f4.conformer.augfft.mejoran": (25.0, 0.5, "05 T1"),
    "f4.conformer.augfft.empeoran": (28.0, 0.5, "05 T1"),
    "f4.conformer.augfft.p": (0.952, 0.0005, "05 T1"),
    "f4.conformer.augfft.p_holm": (1.0, 0.0005, "05 T1"),
    "f4.conformer.augfft.r": (0.01, 0.005, "05 T1"),
    "f4.conformer.augfft.margen": (0.006, 0.0005, "05 T1"),
    "f4.conformer.augfft.equiv": (1.0, 0.5, "05 T1"),
    "f4.conformer.augfft.p_holm22": (1.0, 0.0005, "05 T1"),
    "f4.labram.w02.ref": (0.6647, 5e-05, "05 T1"),
    "f4.labram.w02.var": (0.6431, 5e-05, "05 T1"),
    "f4.labram.w02.dif": (-0.0217, 5e-05, "05 T1"),
    "f4.labram.w02.dif_med": (-0.0112, 5e-05, "05 T1"),
    "f4.labram.w02.ic_inf": (-0.0406, 5e-05, "05 T1"),
    "f4.labram.w02.ic_sup": (-0.003, 5e-05, "05 T1"),
    "f4.labram.w02.mejoran": (24.0, 0.5, "05 T1"),
    "f4.labram.w02.empeoran": (30.0, 0.5, "05 T1"),
    "f4.labram.w02.p": (0.06, 0.0005, "05 T1"),
    "f4.labram.w02.p_holm": (0.121, 0.0005, "05 T1"),
    "f4.labram.w02.r": (0.26, 0.005, "05 T1"),
    "f4.labram.w02.margen": (0.038, 0.0005, "05 T1"),
    "f4.labram.w02.equiv": (0.0, 0.5, "05 T1"),
    "f4.labram.w02.p_holm22": (0.671, 0.0005, "05 T1"),
    "f4.labram.augcr.ref": (0.6647, 5e-05, "05 T1"),
    "f4.labram.augcr.var": (0.6508, 5e-05, "05 T1"),
    "f4.labram.augcr.dif": (-0.0139, 5e-05, "05 T1"),
    "f4.labram.augcr.dif_med": (-0.0187, 5e-05, "05 T1"),
    "f4.labram.augcr.ic_inf": (-0.0265, 5e-05, "05 T1"),
    "f4.labram.augcr.ic_sup": (-0.0011, 5e-05, "05 T1"),
    "f4.labram.augcr.mejoran": (19.0, 0.5, "05 T1"),
    "f4.labram.augcr.empeoran": (35.0, 0.5, "05 T1"),
    "f4.labram.augcr.p": (0.014, 0.0005, "05 T1"),
    "f4.labram.augcr.p_holm": (0.043, 0.0005, "05 T1"),
    "f4.labram.augcr.r": (0.33, 0.005, "05 T1"),
    "f4.labram.augcr.margen": (0.024, 0.0005, "05 T1"),
    "f4.labram.augcr.equiv": (1.0, 0.5, "05 T1"),
    "f4.labram.augcr.p_holm22": (0.217, 0.0005, "05 T1"),
    "f4.labram.augfft.ref": (0.6647, 5e-05, "05 T1"),
    "f4.labram.augfft.var": (0.6616, 5e-05, "05 T1"),
    "f4.labram.augfft.dif": (-0.0031, 5e-05, "05 T1"),
    "f4.labram.augfft.dif_med": (-0.005, 5e-05, "05 T1"),
    "f4.labram.augfft.ic_inf": (-0.0098, 5e-05, "05 T1"),
    "f4.labram.augfft.ic_sup": (0.0038, 5e-05, "05 T1"),
    "f4.labram.augfft.mejoran": (20.0, 0.5, "05 T1"),
    "f4.labram.augfft.empeoran": (33.0, 0.5, "05 T1"),
    "f4.labram.augfft.p": (0.25, 0.0005, "05 T1"),
    "f4.labram.augfft.p_holm": (0.25, 0.0005, "05 T1"),
    "f4.labram.augfft.r": (0.16, 0.005, "05 T1"),
    "f4.labram.augfft.margen": (0.009, 0.0005, "05 T1"),
    "f4.labram.augfft.equiv": (1.0, 0.5, "05 T1"),
    "f4.labram.augfft.p_holm22": (1.0, 0.0005, "05 T1"),
    "f4.eegmamba.w02.ref": (0.5769, 5e-05, "05 T1"),
    "f4.eegmamba.w02.var": (0.6011, 5e-05, "05 T1"),
    "f4.eegmamba.w02.dif": (0.0242, 5e-05, "05 T1"),
    "f4.eegmamba.w02.dif_med": (0.035, 5e-05, "05 T1"),
    "f4.eegmamba.w02.ic_inf": (0.0089, 5e-05, "05 T1"),
    "f4.eegmamba.w02.ic_sup": (0.039, 5e-05, "05 T1"),
    "f4.eegmamba.w02.mejoran": (40.0, 0.5, "05 T1"),
    "f4.eegmamba.w02.empeoran": (14.0, 0.5, "05 T1"),
    "f4.eegmamba.w02.p": (0.001, 0.0005, "05 T1"),
    "f4.eegmamba.w02.p_holm": (0.004, 0.0005, "05 T1"),
    "f4.eegmamba.w02.r": (0.44, 0.005, "05 T1"),
    "f4.eegmamba.w02.margen": (0.037, 0.0005, "05 T1"),
    "f4.eegmamba.w02.equiv": (0.0, 0.5, "05 T1"),
    "f4.eegmamba.w02.p_holm22": (0.023, 0.0005, "05 T1"),
    "f4.eegmamba.augcr.ref": (0.5769, 5e-05, "05 T1"),
    "f4.eegmamba.augcr.var": (0.6001, 5e-05, "05 T1"),
    "f4.eegmamba.augcr.dif": (0.0232, 5e-05, "05 T1"),
    "f4.eegmamba.augcr.dif_med": (0.0113, 5e-05, "05 T1"),
    "f4.eegmamba.augcr.ic_inf": (0.009, 5e-05, "05 T1"),
    "f4.eegmamba.augcr.ic_sup": (0.0381, 5e-05, "05 T1"),
    "f4.eegmamba.augcr.mejoran": (34.0, 0.5, "05 T1"),
    "f4.eegmamba.augcr.empeoran": (20.0, 0.5, "05 T1"),
    "f4.eegmamba.augcr.p": (0.007, 0.0005, "05 T1"),
    "f4.eegmamba.augcr.p_holm": (0.015, 0.0005, "05 T1"),
    "f4.eegmamba.augcr.r": (0.36, 0.005, "05 T1"),
    "f4.eegmamba.augcr.margen": (0.036, 0.0005, "05 T1"),
    "f4.eegmamba.augcr.equiv": (0.0, 0.5, "05 T1"),
    "f4.eegmamba.augcr.p_holm22": (0.118, 0.0005, "05 T1"),
    "f4.eegmamba.augfft.ref": (0.5769, 5e-05, "05 T1"),
    "f4.eegmamba.augfft.var": (0.57, 5e-05, "05 T1"),
    "f4.eegmamba.augfft.dif": (-0.0069, 5e-05, "05 T1"),
    "f4.eegmamba.augfft.dif_med": (-0.0088, 5e-05, "05 T1"),
    "f4.eegmamba.augfft.ic_inf": (-0.015, 5e-05, "05 T1"),
    "f4.eegmamba.augfft.ic_sup": (0.0012, 5e-05, "05 T1"),
    "f4.eegmamba.augfft.mejoran": (20.0, 0.5, "05 T1"),
    "f4.eegmamba.augfft.empeoran": (33.0, 0.5, "05 T1"),
    "f4.eegmamba.augfft.p": (0.118, 0.0005, "05 T1"),
    "f4.eegmamba.augfft.p_holm": (0.118, 0.0005, "05 T1"),
    "f4.eegmamba.augfft.r": (0.21, 0.005, "05 T1"),
    "f4.eegmamba.augfft.margen": (0.014, 0.0005, "05 T1"),
    "f4.eegmamba.augfft.equiv": (1.0, 0.5, "05 T1"),
    "f4.eegmamba.augfft.p_holm22": (1.0, 0.0005, "05 T1"),
    "f4s.labram.w02.ref": (0.6626, 5e-05, "05 T2"),
    "f4s.labram.w02.var": (0.6431, 5e-05, "05 T2"),
    "f4s.labram.w02.dif": (-0.0196, 5e-05, "05 T2"),
    "f4s.labram.w02.dif_med": (-0.0031, 5e-05, "05 T2"),
    "f4s.labram.w02.ic_inf": (-0.0391, 5e-05, "05 T2"),
    "f4s.labram.w02.ic_sup": (-0.0007, 5e-05, "05 T2"),
    "f4s.labram.w02.mejoran": (24.0, 0.5, "05 T2"),
    "f4s.labram.w02.empeoran": (29.0, 0.5, "05 T2"),
    "f4s.labram.w02.p": (0.171, 0.0005, "05 T2"),
    "f4s.labram.w02.p_holm": (0.342, 0.0005, "05 T2"),
    "f4s.labram.w02.r": (0.19, 0.005, "05 T2"),
    "f4s.labram.w02.margen": (0.036, 0.0005, "05 T2"),
    "f4s.labram.w02.equiv": (0.0, 0.5, "05 T2"),
    "f4s.labram.augcr.ref": (0.6626, 5e-05, "05 T2"),
    "f4s.labram.augcr.var": (0.6508, 5e-05, "05 T2"),
    "f4s.labram.augcr.dif": (-0.0118, 5e-05, "05 T2"),
    "f4s.labram.augcr.dif_med": (-0.0106, 5e-05, "05 T2"),
    "f4s.labram.augcr.ic_inf": (-0.0224, 5e-05, "05 T2"),
    "f4s.labram.augcr.ic_sup": (-0.0012, 5e-05, "05 T2"),
    "f4s.labram.augcr.mejoran": (20.0, 0.5, "05 T2"),
    "f4s.labram.augcr.empeoran": (34.0, 0.5, "05 T2"),
    "f4s.labram.augcr.p": (0.03, 0.0005, "05 T2"),
    "f4s.labram.augcr.p_holm": (0.091, 0.0005, "05 T2"),
    "f4s.labram.augcr.r": (0.29, 0.005, "05 T2"),
    "f4s.labram.augcr.margen": (0.021, 0.0005, "05 T2"),
    "f4s.labram.augcr.equiv": (1.0, 0.5, "05 T2"),
    "f4s.labram.augfft.ref": (0.6626, 5e-05, "05 T2"),
    "f4s.labram.augfft.var": (0.6616, 5e-05, "05 T2"),
    "f4s.labram.augfft.dif": (-0.001, 5e-05, "05 T2"),
    "f4s.labram.augfft.dif_med": (0.0044, 5e-05, "05 T2"),
    "f4s.labram.augfft.ic_inf": (-0.0077, 5e-05, "05 T2"),
    "f4s.labram.augfft.ic_sup": (0.0056, 5e-05, "05 T2"),
    "f4s.labram.augfft.mejoran": (29.0, 0.5, "05 T2"),
    "f4s.labram.augfft.empeoran": (25.0, 0.5, "05 T2"),
    "f4s.labram.augfft.p": (0.928, 0.0005, "05 T2"),
    "f4s.labram.augfft.p_holm": (0.928, 0.0005, "05 T2"),
    "f4s.labram.augfft.r": (0.01, 0.005, "05 T2"),
    "f4s.labram.augfft.margen": (0.007, 0.0005, "05 T2"),
    "f4s.labram.augfft.equiv": (1.0, 0.5, "05 T2"),
    "f4s.eegmamba.w02.ref": (0.5677, 5e-05, "05 T2"),
    "f4s.eegmamba.w02.var": (0.6011, 5e-05, "05 T2"),
    "f4s.eegmamba.w02.dif": (0.0334, 5e-05, "05 T2"),
    "f4s.eegmamba.w02.dif_med": (0.0406, 5e-05, "05 T2"),
    "f4s.eegmamba.w02.ic_inf": (0.0203, 5e-05, "05 T2"),
    "f4s.eegmamba.w02.ic_sup": (0.0459, 5e-05, "05 T2"),
    "f4s.eegmamba.w02.mejoran": (43.0, 0.5, "05 T2"),
    "f4s.eegmamba.w02.empeoran": (11.0, 0.5, "05 T2"),
    "f4s.eegmamba.w02.p": (1.5000000000000002e-05, 5.000000000000001e-07, "05 T2"),
    "f4s.eegmamba.w02.p_holm": (4.5e-05, 5.000000000000001e-07, "05 T2"),
    "f4s.eegmamba.w02.r": (0.59, 0.005, "05 T2"),
    "f4s.eegmamba.w02.margen": (0.044, 0.0005, "05 T2"),
    "f4s.eegmamba.w02.equiv": (0.0, 0.5, "05 T2"),
    "f4s.eegmamba.augcr.ref": (0.5677, 5e-05, "05 T2"),
    "f4s.eegmamba.augcr.var": (0.6001, 5e-05, "05 T2"),
    "f4s.eegmamba.augcr.dif": (0.0324, 5e-05, "05 T2"),
    "f4s.eegmamba.augcr.dif_med": (0.025, 5e-05, "05 T2"),
    "f4s.eegmamba.augcr.ic_inf": (0.0187, 5e-05, "05 T2"),
    "f4s.eegmamba.augcr.ic_sup": (0.0463, 5e-05, "05 T2"),
    "f4s.eegmamba.augcr.mejoran": (40.0, 0.5, "05 T2"),
    "f4s.eegmamba.augcr.empeoran": (14.0, 0.5, "05 T2"),
    "f4s.eegmamba.augcr.p": (1.8e-05, 5.000000000000001e-07, "05 T2"),
    "f4s.eegmamba.augcr.p_holm": (4.5e-05, 5.000000000000001e-07, "05 T2"),
    "f4s.eegmamba.augcr.r": (0.58, 0.005, "05 T2"),
    "f4s.eegmamba.augcr.margen": (0.044, 0.0005, "05 T2"),
    "f4s.eegmamba.augcr.equiv": (0.0, 0.5, "05 T2"),
    "f4s.eegmamba.augfft.ref": (0.5677, 5e-05, "05 T2"),
    "f4s.eegmamba.augfft.var": (0.57, 5e-05, "05 T2"),
    "f4s.eegmamba.augfft.dif": (0.0022, 5e-05, "05 T2"),
    "f4s.eegmamba.augfft.dif_med": (0.0063, 5e-05, "05 T2"),
    "f4s.eegmamba.augfft.ic_inf": (-0.0049, 5e-05, "05 T2"),
    "f4s.eegmamba.augfft.ic_sup": (0.0091, 5e-05, "05 T2"),
    "f4s.eegmamba.augfft.mejoran": (32.0, 0.5, "05 T2"),
    "f4s.eegmamba.augfft.empeoran": (22.0, 0.5, "05 T2"),
    "f4s.eegmamba.augfft.p": (0.324, 0.0005, "05 T2"),
    "f4s.eegmamba.augfft.p_holm": (0.324, 0.0005, "05 T2"),
    "f4s.eegmamba.augfft.r": (0.13, 0.005, "05 T2"),
    "f4s.eegmamba.augfft.margen": (0.008, 0.0005, "05 T2"),
    "f4s.eegmamba.augfft.equiv": (1.0, 0.5, "05 T2"),
    "f6.eegnet.ref": (0.7625, 5e-05, "05 T3"),
    "f6.eegnet.var": (0.7735, 5e-05, "05 T3"),
    "f6.eegnet.dif": (0.0109, 5e-05, "05 T3"),
    "f6.eegnet.dif_med": (0.0175, 5e-05, "05 T3"),
    "f6.eegnet.ic_inf": (-0.007, 5e-05, "05 T3"),
    "f6.eegnet.ic_sup": (0.0261, 5e-05, "05 T3"),
    "f6.eegnet.mejoran": (34.0, 0.5, "05 T3"),
    "f6.eegnet.empeoran": (19.0, 0.5, "05 T3"),
    "f6.eegnet.p": (0.013, 0.0005, "05 T3"),
    "f6.eegnet.p_holm": (0.013, 0.0005, "05 T3"),
    "f6.eegnet.r": (0.34, 0.005, "05 T3"),
    "f6.eegnet.margen": (0.024, 0.0005, "05 T3"),
    "f6.eegnet.equiv": (1.0, 0.5, "05 T3"),
    "f6.eegnet.p_holm8": (0.051, 0.0005, "05 T3"),
    "f6.conformer.ref": (0.7517, 5e-05, "05 T3"),
    "f6.conformer.var": (0.7689, 5e-05, "05 T3"),
    "f6.conformer.dif": (0.0173, 5e-05, "05 T3"),
    "f6.conformer.dif_med": (0.025, 5e-05, "05 T3"),
    "f6.conformer.ic_inf": (-0.0035, 5e-05, "05 T3"),
    "f6.conformer.ic_sup": (0.0363, 5e-05, "05 T3"),
    "f6.conformer.mejoran": (37.0, 0.5, "05 T3"),
    "f6.conformer.empeoran": (17.0, 0.5, "05 T3"),
    "f6.conformer.p": (0.001, 0.0005, "05 T3"),
    "f6.conformer.p_holm": (0.003, 0.0005, "05 T3"),
    "f6.conformer.r": (0.43, 0.005, "05 T3"),
    "f6.conformer.margen": (0.034, 0.0005, "05 T3"),
    "f6.conformer.equiv": (0.0, 0.5, "05 T3"),
    "f6.conformer.p_holm8": (0.008, 0.0005, "05 T3"),
    "f6.labram.ref": (0.7538, 5e-05, "05 T3"),
    "f6.labram.var": (0.7787, 5e-05, "05 T3"),
    "f6.labram.dif": (0.0249, 5e-05, "05 T3"),
    "f6.labram.dif_med": (0.0275, 5e-05, "05 T3"),
    "f6.labram.ic_inf": (0.013, 5e-05, "05 T3"),
    "f6.labram.ic_sup": (0.0363, 5e-05, "05 T3"),
    "f6.labram.mejoran": (41.0, 0.5, "05 T3"),
    "f6.labram.empeoran": (12.0, 0.5, "05 T3"),
    "f6.labram.p": (0.00015, 5e-06, "05 T3"),
    "f6.labram.p_holm": (0.00046, 5e-06, "05 T3"),
    "f6.labram.r": (0.52, 0.005, "05 T3"),
    "f6.labram.margen": (0.035, 0.0005, "05 T3"),
    "f6.labram.equiv": (0.0, 0.5, "05 T3"),
    "f6.labram.p_holm8": (0.0011, 5e-05, "05 T3"),
    "f6.eegmamba.ref": (0.7168, 5e-05, "05 T3"),
    "f6.eegmamba.var": (0.7631, 5e-05, "05 T3"),
    "f6.eegmamba.dif": (0.0463, 5e-05, "05 T3"),
    "f6.eegmamba.dif_med": (0.045, 5e-05, "05 T3"),
    "f6.eegmamba.ic_inf": (0.0309, 5e-05, "05 T3"),
    "f6.eegmamba.ic_sup": (0.061, 5e-05, "05 T3"),
    "f6.eegmamba.mejoran": (46.0, 0.5, "05 T3"),
    "f6.eegmamba.empeoran": (6.0, 0.5, "05 T3"),
    "f6.eegmamba.p": (1.9e-07, 5e-09, "05 T3"),
    "f6.eegmamba.p_holm": (7.7e-07, 5e-09, "05 T3"),
    "f6.eegmamba.r": (0.71, 0.005, "05 T3"),
    "f6.eegmamba.margen": (0.059, 0.0005, "05 T3"),
    "f6.eegmamba.equiv": (0.0, 0.5, "05 T3"),
    "f6.eegmamba.p_holm8": (1.5e-06, 5e-08, "05 T3"),
    "f6s.eegnet.ref": (0.7596, 5e-05, "05 T3"),
    "f6s.eegnet.var": (0.7735, 5e-05, "05 T3"),
    "f6s.eegnet.dif": (0.0139, 5e-05, "05 T3"),
    "f6s.eegnet.dif_med": (0.0156, 5e-05, "05 T3"),
    "f6s.eegnet.ic_inf": (-0.0021, 5e-05, "05 T3"),
    "f6s.eegnet.ic_sup": (0.028, 5e-05, "05 T3"),
    "f6s.eegnet.mejoran": (36.0, 0.5, "05 T3"),
    "f6s.eegnet.empeoran": (18.0, 0.5, "05 T3"),
    "f6s.eegnet.p": (0.006, 0.0005, "05 T3"),
    "f6s.eegnet.p_holm": (0.006, 0.0005, "05 T3"),
    "f6s.eegnet.r": (0.37, 0.005, "05 T3"),
    "f6s.eegnet.margen": (0.026, 0.0005, "05 T3"),
    "f6s.eegnet.equiv": (1.0, 0.5, "05 T3"),
    "f6s.conformer.ref": (0.7444, 5e-05, "05 T3"),
    "f6s.conformer.var": (0.7689, 5e-05, "05 T3"),
    "f6s.conformer.dif": (0.0245, 5e-05, "05 T3"),
    "f6s.conformer.dif_med": (0.0237, 5e-05, "05 T3"),
    "f6s.conformer.ic_inf": (0.0026, 5e-05, "05 T3"),
    "f6s.conformer.ic_sup": (0.0457, 5e-05, "05 T3"),
    "f6s.conformer.mejoran": (37.0, 0.5, "05 T3"),
    "f6s.conformer.empeoran": (17.0, 0.5, "05 T3"),
    "f6s.conformer.p": (0.00055, 5e-06, "05 T3"),
    "f6s.conformer.p_holm": (0.001, 0.0005, "05 T3"),
    "f6s.conformer.r": (0.47, 0.005, "05 T3"),
    "f6s.conformer.margen": (0.043, 0.0005, "05 T3"),
    "f6s.conformer.equiv": (0.0, 0.5, "05 T3"),
    "f6s.labram.ref": (0.7554, 5e-05, "05 T3"),
    "f6s.labram.var": (0.7787, 5e-05, "05 T3"),
    "f6s.labram.dif": (0.0232, 5e-05, "05 T3"),
    "f6s.labram.dif_med": (0.0294, 5e-05, "05 T3"),
    "f6s.labram.ic_inf": (0.0133, 5e-05, "05 T3"),
    "f6s.labram.ic_sup": (0.0329, 5e-05, "05 T3"),
    "f6s.labram.mejoran": (39.0, 0.5, "05 T3"),
    "f6s.labram.empeoran": (15.0, 0.5, "05 T3"),
    "f6s.labram.p": (2.5e-05, 5.000000000000001e-07, "05 T3"),
    "f6s.labram.p_holm": (7.5e-05, 5.000000000000001e-07, "05 T3"),
    "f6s.labram.r": (0.57, 0.005, "05 T3"),
    "f6s.labram.margen": (0.031, 0.0005, "05 T3"),
    "f6s.labram.equiv": (0.0, 0.5, "05 T3"),
    "f6s.eegmamba.ref": (0.7146, 5e-05, "05 T3"),
    "f6s.eegmamba.var": (0.7631, 5e-05, "05 T3"),
    "f6s.eegmamba.dif": (0.0485, 5e-05, "05 T3"),
    "f6s.eegmamba.dif_med": (0.0431, 5e-05, "05 T3"),
    "f6s.eegmamba.ic_inf": (0.0331, 5e-05, "05 T3"),
    "f6s.eegmamba.ic_sup": (0.0631, 5e-05, "05 T3"),
    "f6s.eegmamba.mejoran": (45.0, 0.5, "05 T3"),
    "f6s.eegmamba.empeoran": (9.0, 0.5, "05 T3"),
    "f6s.eegmamba.p": (5.5e-08, 5e-10, "05 T3"),
    "f6s.eegmamba.p_holm": (2.2e-07, 5e-09, "05 T3"),
    "f6s.eegmamba.r": (0.74, 0.005, "05 T3"),
    "f6s.eegmamba.margen": (0.061, 0.0005, "05 T3"),
    "f6s.eegmamba.equiv": (0.0, 0.5, "05 T3"),
    "f8.eegnet.ref": (0.7625, 5e-05, "05 T3"),
    "f8.eegnet.var": (0.76, 5e-05, "05 T3"),
    "f8.eegnet.dif": (-0.0025, 5e-05, "05 T3"),
    "f8.eegnet.dif_med": (-0.005, 5e-05, "05 T3"),
    "f8.eegnet.ic_inf": (-0.0111, 5e-05, "05 T3"),
    "f8.eegnet.ic_sup": (0.0055, 5e-05, "05 T3"),
    "f8.eegnet.mejoran": (26.0, 0.5, "05 T3"),
    "f8.eegnet.empeoran": (28.0, 0.5, "05 T3"),
    "f8.eegnet.p": (0.773, 0.0005, "05 T3"),
    "f8.eegnet.p_holm": (0.773, 0.0005, "05 T3"),
    "f8.eegnet.r": (0.04, 0.005, "05 T3"),
    "f8.eegnet.margen": (0.01, 0.0005, "05 T3"),
    "f8.eegnet.equiv": (1.0, 0.5, "05 T3"),
    "f8.eegnet.p_holm8": (0.773, 0.0005, "05 T3"),
    "f8.conformer.ref": (0.7517, 5e-05, "05 T3"),
    "f8.conformer.var": (0.7661, 5e-05, "05 T3"),
    "f8.conformer.dif": (0.0144, 5e-05, "05 T3"),
    "f8.conformer.dif_med": (0.0125, 5e-05, "05 T3"),
    "f8.conformer.ic_inf": (0.0029, 5e-05, "05 T3"),
    "f8.conformer.ic_sup": (0.0256, 5e-05, "05 T3"),
    "f8.conformer.mejoran": (39.0, 0.5, "05 T3"),
    "f8.conformer.empeoran": (15.0, 0.5, "05 T3"),
    "f8.conformer.p": (0.005, 0.0005, "05 T3"),
    "f8.conformer.p_holm": (0.019, 0.0005, "05 T3"),
    "f8.conformer.r": (0.38, 0.005, "05 T3"),
    "f8.conformer.margen": (0.024, 0.0005, "05 T3"),
    "f8.conformer.equiv": (1.0, 0.5, "05 T3"),
    "f8.conformer.p_holm8": (0.024, 0.0005, "05 T3"),
    "f8.labram.ref": (0.7538, 5e-05, "05 T3"),
    "f8.labram.var": (0.7644, 5e-05, "05 T3"),
    "f8.labram.dif": (0.0106, 5e-05, "05 T3"),
    "f8.labram.dif_med": (0.0137, 5e-05, "05 T3"),
    "f8.labram.ic_inf": (0.0004, 5e-05, "05 T3"),
    "f8.labram.ic_sup": (0.0206, 5e-05, "05 T3"),
    "f8.labram.mejoran": (34.0, 0.5, "05 T3"),
    "f8.labram.empeoran": (18.0, 0.5, "05 T3"),
    "f8.labram.p": (0.045, 0.0005, "05 T3"),
    "f8.labram.p_holm": (0.134, 0.0005, "05 T3"),
    "f8.labram.r": (0.27, 0.005, "05 T3"),
    "f8.labram.margen": (0.019, 0.0005, "05 T3"),
    "f8.labram.equiv": (1.0, 0.5, "05 T3"),
    "f8.labram.p_holm8": (0.134, 0.0005, "05 T3"),
    "f8.eegmamba.ref": (0.7168, 5e-05, "05 T3"),
    "f8.eegmamba.var": (0.7244, 5e-05, "05 T3"),
    "f8.eegmamba.dif": (0.0076, 5e-05, "05 T3"),
    "f8.eegmamba.dif_med": (0.0062, 5e-05, "05 T3"),
    "f8.eegmamba.ic_inf": (-0.0005, 5e-05, "05 T3"),
    "f8.eegmamba.ic_sup": (0.0156, 5e-05, "05 T3"),
    "f8.eegmamba.mejoran": (34.0, 0.5, "05 T3"),
    "f8.eegmamba.empeoran": (20.0, 0.5, "05 T3"),
    "f8.eegmamba.p": (0.071, 0.0005, "05 T3"),
    "f8.eegmamba.p_holm": (0.142, 0.0005, "05 T3"),
    "f8.eegmamba.r": (0.25, 0.005, "05 T3"),
    "f8.eegmamba.margen": (0.014, 0.0005, "05 T3"),
    "f8.eegmamba.equiv": (1.0, 0.5, "05 T3"),
    "f8.eegmamba.p_holm8": (0.142, 0.0005, "05 T3"),
    "f8s.eegnet.ref": (0.7596, 5e-05, "05 T3"),
    "f8s.eegnet.var": (0.76, 5e-05, "05 T3"),
    "f8s.eegnet.dif": (0.0004, 5e-05, "05 T3"),
    "f8s.eegnet.dif_med": (0.0012, 5e-05, "05 T3"),
    "f8s.eegnet.ic_inf": (-0.0087, 5e-05, "05 T3"),
    "f8s.eegnet.ic_sup": (0.0096, 5e-05, "05 T3"),
    "f8s.eegnet.mejoran": (28.0, 0.5, "05 T3"),
    "f8s.eegnet.empeoran": (23.0, 0.5, "05 T3"),
    "f8s.eegnet.p": (0.737, 0.0005, "05 T3"),
    "f8s.eegnet.p_holm": (0.737, 0.0005, "05 T3"),
    "f8s.eegnet.r": (0.05, 0.005, "05 T3"),
    "f8s.eegnet.margen": (0.008, 0.0005, "05 T3"),
    "f8s.eegnet.equiv": (1.0, 0.5, "05 T3"),
    "f8s.conformer.ref": (0.7444, 5e-05, "05 T3"),
    "f8s.conformer.var": (0.7661, 5e-05, "05 T3"),
    "f8s.conformer.dif": (0.0216, 5e-05, "05 T3"),
    "f8s.conformer.dif_med": (0.0212, 5e-05, "05 T3"),
    "f8s.conformer.ic_inf": (0.0092, 5e-05, "05 T3"),
    "f8s.conformer.ic_sup": (0.0347, 5e-05, "05 T3"),
    "f8s.conformer.mejoran": (38.0, 0.5, "05 T3"),
    "f8s.conformer.empeoran": (16.0, 0.5, "05 T3"),
    "f8s.conformer.p": (0.00096, 5e-06, "05 T3"),
    "f8s.conformer.p_holm": (0.004, 0.0005, "05 T3"),
    "f8s.conformer.r": (0.45, 0.005, "05 T3"),
    "f8s.conformer.margen": (0.032, 0.0005, "05 T3"),
    "f8s.conformer.equiv": (0.0, 0.5, "05 T3"),
    "f8s.labram.ref": (0.7554, 5e-05, "05 T3"),
    "f8s.labram.var": (0.7644, 5e-05, "05 T3"),
    "f8s.labram.dif": (0.009, 5e-05, "05 T3"),
    "f8s.labram.dif_med": (0.0112, 5e-05, "05 T3"),
    "f8s.labram.ic_inf": (0.0001, 5e-05, "05 T3"),
    "f8s.labram.ic_sup": (0.0175, 5e-05, "05 T3"),
    "f8s.labram.mejoran": (37.0, 0.5, "05 T3"),
    "f8s.labram.empeoran": (17.0, 0.5, "05 T3"),
    "f8s.labram.p": (0.014, 0.0005, "05 T3"),
    "f8s.labram.p_holm": (0.039, 0.0005, "05 T3"),
    "f8s.labram.r": (0.33, 0.005, "05 T3"),
    "f8s.labram.margen": (0.016, 0.0005, "05 T3"),
    "f8s.labram.equiv": (1.0, 0.5, "05 T3"),
    "f8s.eegmamba.ref": (0.7146, 5e-05, "05 T3"),
    "f8s.eegmamba.var": (0.7244, 5e-05, "05 T3"),
    "f8s.eegmamba.dif": (0.0098, 5e-05, "05 T3"),
    "f8s.eegmamba.dif_med": (0.0112, 5e-05, "05 T3"),
    "f8s.eegmamba.ic_inf": (0.0028, 5e-05, "05 T3"),
    "f8s.eegmamba.ic_sup": (0.0172, 5e-05, "05 T3"),
    "f8s.eegmamba.mejoran": (38.0, 0.5, "05 T3"),
    "f8s.eegmamba.empeoran": (16.0, 0.5, "05 T3"),
    "f8s.eegmamba.p": (0.013, 0.0005, "05 T3"),
    "f8s.eegmamba.p_holm": (0.039, 0.0005, "05 T3"),
    "f8s.eegmamba.r": (0.34, 0.005, "05 T3"),
    "f8s.eegmamba.margen": (0.016, 0.0005, "05 T3"),
    "f8s.eegmamba.equiv": (1.0, 0.5, "05 T3"),
    "t4ea.eegnet.nov_base": (0.7625, 5e-05, "05 T4"),
    "t4ea.eegnet.nov_ea": (0.7735, 5e-05, "05 T4"),
    "t4ea.eegnet.vis_base": (0.7827, 5e-05, "05 T4"),
    "t4ea.eegnet.vis_ea": (0.7982, 5e-05, "05 T4"),
    "t4ea.eegnet.caida_base": (0.0201, 5e-05, "05 T4"),
    "t4ea.eegnet.caida_ea": (0.0247, 5e-05, "05 T4"),
    "t4ea.eegnet.ids_base": (0.0257, 5e-05, "05 T4"),
    "t4ea.eegnet.ids_ea": (0.031, 5e-05, "05 T4"),
    "t4ea.eegnet.dif": (0.0046, 5e-05, "05 T4"),
    "t4ea.eegnet.ic_inf": (-0.0078, 5e-05, "05 T4"),
    "t4ea.eegnet.ic_sup": (0.0173, 5e-05, "05 T4"),
    "t4ea.eegnet.p": (0.624, 0.0005, "05 T4"),
    "t4ea.eegnet.r": (0.07, 0.005, "05 T4"),
    "t4ea.eegnet.dsi_base": (-0.204, 0.0005, "05 T4"),
    "t4ea.eegnet.dsi_ea": (-0.222, 0.0005, "05 T4"),
    "t4ea.conformer.nov_base": (0.7517, 5e-05, "05 T4"),
    "t4ea.conformer.nov_ea": (0.7689, 5e-05, "05 T4"),
    "t4ea.conformer.vis_base": (0.7862, 5e-05, "05 T4"),
    "t4ea.conformer.vis_ea": (0.7981, 5e-05, "05 T4"),
    "t4ea.conformer.caida_base": (0.0345, 5e-05, "05 T4"),
    "t4ea.conformer.caida_ea": (0.0292, 5e-05, "05 T4"),
    "t4ea.conformer.ids_base": (0.0439, 5e-05, "05 T4"),
    "t4ea.conformer.ids_ea": (0.0366, 5e-05, "05 T4"),
    "t4ea.conformer.dif": (-0.0053, 5e-05, "05 T4"),
    "t4ea.conformer.ic_inf": (-0.0207, 5e-05, "05 T4"),
    "t4ea.conformer.ic_sup": (0.0109, 5e-05, "05 T4"),
    "t4ea.conformer.p": (0.462, 0.0005, "05 T4"),
    "t4ea.conformer.r": (0.1, 0.005, "05 T4"),
    "t4ea.conformer.dsi_base": (-0.239, 0.0005, "05 T4"),
    "t4ea.conformer.dsi_ea": (-0.267, 0.0005, "05 T4"),
    "t4ea.labram.nov_base": (0.7538, 5e-05, "05 T4"),
    "t4ea.labram.nov_ea": (0.7787, 5e-05, "05 T4"),
    "t4ea.labram.vis_base": (0.798, 5e-05, "05 T4"),
    "t4ea.labram.vis_ea": (0.8249, 5e-05, "05 T4"),
    "t4ea.labram.caida_base": (0.0442, 5e-05, "05 T4"),
    "t4ea.labram.caida_ea": (0.0462, 5e-05, "05 T4"),
    "t4ea.labram.ids_base": (0.0554, 5e-05, "05 T4"),
    "t4ea.labram.ids_ea": (0.056, 5e-05, "05 T4"),
    "t4ea.labram.dif": (0.002, 5e-05, "05 T4"),
    "t4ea.labram.ic_inf": (-0.0096, 5e-05, "05 T4"),
    "t4ea.labram.ic_sup": (0.0143, 5e-05, "05 T4"),
    "t4ea.labram.p": (0.942, 0.0005, "05 T4"),
    "t4ea.labram.r": (0.01, 0.005, "05 T4"),
    "t4ea.labram.dsi_base": (-0.134, 0.0005, "05 T4"),
    "t4ea.labram.dsi_ea": (-0.171, 0.0005, "05 T4"),
    "t4ea.eegmamba.nov_base": (0.7168, 5e-05, "05 T4"),
    "t4ea.eegmamba.nov_ea": (0.7631, 5e-05, "05 T4"),
    "t4ea.eegmamba.vis_base": (0.7423, 5e-05, "05 T4"),
    "t4ea.eegmamba.vis_ea": (0.7809, 5e-05, "05 T4"),
    "t4ea.eegmamba.caida_base": (0.0255, 5e-05, "05 T4"),
    "t4ea.eegmamba.caida_ea": (0.0178, 5e-05, "05 T4"),
    "t4ea.eegmamba.ids_base": (0.0343, 5e-05, "05 T4"),
    "t4ea.eegmamba.ids_ea": (0.0228, 5e-05, "05 T4"),
    "t4ea.eegmamba.dif": (-0.0077, 5e-05, "05 T4"),
    "t4ea.eegmamba.ic_inf": (-0.0223, 5e-05, "05 T4"),
    "t4ea.eegmamba.ic_sup": (0.0074, 5e-05, "05 T4"),
    "t4ea.eegmamba.p": (0.328, 0.0005, "05 T4"),
    "t4ea.eegmamba.r": (0.13, 0.005, "05 T4"),
    "t4ea.eegmamba.dsi_base": (-0.243, 0.0005, "05 T4"),
    "t4ea.eegmamba.dsi_ea": (-0.323, 0.0005, "05 T4"),
    "t4ea.eegnet.red_inf": (-3.58, 0.005, "05 sec. 3.5 (texto)"),
    "t4ea.eegnet.red_sup": (2.96, 0.005, "05 sec. 3.5 (texto)"),
    "t6r.within.eegnet.2027-2026.dif": (-0.0025, 5e-05, "05 T6"),
    "t6r.within.eegnet.2027-2026.ic_inf": (-0.0133, 5e-05, "05 T6"),
    "t6r.within.eegnet.2027-2026.ic_sup": (0.0112, 5e-05, "05 T6"),
    "t6r.within.eegnet.2027-2026.p": (0.221, 0.0005, "05 T6"),
    "t6r.within.eegnet.2028-2026.dif": (0.0058, 5e-05, "05 T6"),
    "t6r.within.eegnet.2028-2026.ic_inf": (-0.004, 5e-05, "05 T6"),
    "t6r.within.eegnet.2028-2026.ic_sup": (0.0155, 5e-05, "05 T6"),
    "t6r.within.eegnet.2028-2026.p": (0.23, 0.0005, "05 T6"),
    "t6r.within.eegnet.2028-2027.dif": (0.0083, 5e-05, "05 T6"),
    "t6r.within.eegnet.2028-2027.ic_inf": (-0.0073, 5e-05, "05 T6"),
    "t6r.within.eegnet.2028-2027.ic_sup": (0.0213, 5e-05, "05 T6"),
    "t6r.within.eegnet.2028-2027.p": (0.015, 0.0005, "05 T6"),
    "t6r.within.conformer.2027-2026.dif": (-0.0069, 5e-05, "05 T6"),
    "t6r.within.conformer.2027-2026.ic_inf": (-0.0132, 5e-05, "05 T6"),
    "t6r.within.conformer.2027-2026.ic_sup": (-0.0007, 5e-05, "05 T6"),
    "t6r.within.conformer.2027-2026.p": (0.042, 0.0005, "05 T6"),
    "t6r.within.conformer.2028-2026.dif": (-0.0063, 5e-05, "05 T6"),
    "t6r.within.conformer.2028-2026.ic_inf": (-0.0161, 5e-05, "05 T6"),
    "t6r.within.conformer.2028-2026.ic_sup": (0.0037, 5e-05, "05 T6"),
    "t6r.within.conformer.2028-2026.p": (0.069, 0.0005, "05 T6"),
    "t6r.within.conformer.2028-2027.dif": (0.0006, 5e-05, "05 T6"),
    "t6r.within.conformer.2028-2027.ic_inf": (-0.0085, 5e-05, "05 T6"),
    "t6r.within.conformer.2028-2027.ic_sup": (0.01, 5e-05, "05 T6"),
    "t6r.within.conformer.2028-2027.p": (0.573, 0.0005, "05 T6"),
    "t6r.within.labram.2027-2026.dif": (-0.0015, 5e-05, "05 T6"),
    "t6r.within.labram.2027-2026.ic_inf": (-0.0093, 5e-05, "05 T6"),
    "t6r.within.labram.2027-2026.ic_sup": (0.0061, 5e-05, "05 T6"),
    "t6r.within.labram.2027-2026.p": (0.705, 0.0005, "05 T6"),
    "t6r.within.labram.2028-2026.dif": (-0.0027, 5e-05, "05 T6"),
    "t6r.within.labram.2028-2026.ic_inf": (-0.0102, 5e-05, "05 T6"),
    "t6r.within.labram.2028-2026.ic_sup": (0.0052, 5e-05, "05 T6"),
    "t6r.within.labram.2028-2026.p": (0.24, 0.0005, "05 T6"),
    "t6r.within.labram.2028-2027.dif": (-0.0012, 5e-05, "05 T6"),
    "t6r.within.labram.2028-2027.ic_inf": (-0.0069, 5e-05, "05 T6"),
    "t6r.within.labram.2028-2027.ic_sup": (0.0047, 5e-05, "05 T6"),
    "t6r.within.labram.2028-2027.p": (0.884, 0.0005, "05 T6"),
    "t6r.within.eegmamba.2027-2026.dif": (-0.0059, 5e-05, "05 T6"),
    "t6r.within.eegmamba.2027-2026.ic_inf": (-0.0154, 5e-05, "05 T6"),
    "t6r.within.eegmamba.2027-2026.ic_sup": (0.0037, 5e-05, "05 T6"),
    "t6r.within.eegmamba.2027-2026.p": (0.375, 0.0005, "05 T6"),
    "t6r.within.eegmamba.2028-2026.dif": (-0.0124, 5e-05, "05 T6"),
    "t6r.within.eegmamba.2028-2026.ic_inf": (-0.0218, 5e-05, "05 T6"),
    "t6r.within.eegmamba.2028-2026.ic_sup": (-0.0031, 5e-05, "05 T6"),
    "t6r.within.eegmamba.2028-2026.p": (0.018, 0.0005, "05 T6"),
    "t6r.within.eegmamba.2028-2027.dif": (-0.0065, 5e-05, "05 T6"),
    "t6r.within.eegmamba.2028-2027.ic_inf": (-0.0139, 5e-05, "05 T6"),
    "t6r.within.eegmamba.2028-2027.ic_sup": (0.001, 5e-05, "05 T6"),
    "t6r.within.eegmamba.2028-2027.p": (0.108, 0.0005, "05 T6"),
    "t6r.loso.eegnet.2027-2026.dif": (-0.003, 5e-05, "05 T6"),
    "t6r.loso.eegnet.2027-2026.ic_inf": (-0.0135, 5e-05, "05 T6"),
    "t6r.loso.eegnet.2027-2026.ic_sup": (0.0067, 5e-05, "05 T6"),
    "t6r.loso.eegnet.2027-2026.p": (0.914, 0.0005, "05 T6"),
    "t6r.loso.eegnet.2028-2026.dif": (-0.0029, 5e-05, "05 T6"),
    "t6r.loso.eegnet.2028-2026.ic_inf": (-0.0107, 5e-05, "05 T6"),
    "t6r.loso.eegnet.2028-2026.ic_sup": (0.0049, 5e-05, "05 T6"),
    "t6r.loso.eegnet.2028-2026.p": (0.698, 0.0005, "05 T6"),
    "t6r.loso.eegnet.2028-2027.dif": (0.0001, 5e-05, "05 T6"),
    "t6r.loso.eegnet.2028-2027.ic_inf": (-0.0097, 5e-05, "05 T6"),
    "t6r.loso.eegnet.2028-2027.ic_sup": (0.0109, 5e-05, "05 T6"),
    "t6r.loso.eegnet.2028-2027.p": (0.248, 0.0005, "05 T6"),
    "t6r.loso.conformer.2027-2026.dif": (-0.0074, 5e-05, "05 T6"),
    "t6r.loso.conformer.2027-2026.ic_inf": (-0.02, 5e-05, "05 T6"),
    "t6r.loso.conformer.2027-2026.ic_sup": (0.0034, 5e-05, "05 T6"),
    "t6r.loso.conformer.2027-2026.p": (0.297, 0.0005, "05 T6"),
    "t6r.loso.conformer.2028-2026.dif": (-0.007, 5e-05, "05 T6"),
    "t6r.loso.conformer.2028-2026.ic_inf": (-0.027, 5e-05, "05 T6"),
    "t6r.loso.conformer.2028-2026.ic_sup": (0.0087, 5e-05, "05 T6"),
    "t6r.loso.conformer.2028-2026.p": (0.945, 0.0005, "05 T6"),
    "t6r.loso.conformer.2028-2027.dif": (0.0004, 5e-05, "05 T6"),
    "t6r.loso.conformer.2028-2027.ic_inf": (-0.0206, 5e-05, "05 T6"),
    "t6r.loso.conformer.2028-2027.ic_sup": (0.0176, 5e-05, "05 T6"),
    "t6r.loso.conformer.2028-2027.p": (0.413, 0.0005, "05 T6"),
    "t6r.loso.labram.2027-2026.dif": (-0.0006, 5e-05, "05 T6"),
    "t6r.loso.labram.2027-2026.ic_inf": (-0.0102, 5e-05, "05 T6"),
    "t6r.loso.labram.2027-2026.ic_sup": (0.0086, 5e-05, "05 T6"),
    "t6r.loso.labram.2027-2026.p": (0.763, 0.0005, "05 T6"),
    "t6r.loso.labram.2028-2026.dif": (0.0039, 5e-05, "05 T6"),
    "t6r.loso.labram.2028-2026.ic_inf": (-0.0051, 5e-05, "05 T6"),
    "t6r.loso.labram.2028-2026.ic_sup": (0.0134, 5e-05, "05 T6"),
    "t6r.loso.labram.2028-2026.p": (0.477, 0.0005, "05 T6"),
    "t6r.loso.labram.2028-2027.dif": (0.0046, 5e-05, "05 T6"),
    "t6r.loso.labram.2028-2027.ic_inf": (-0.0048, 5e-05, "05 T6"),
    "t6r.loso.labram.2028-2027.ic_sup": (0.0142, 5e-05, "05 T6"),
    "t6r.loso.labram.2028-2027.p": (0.335, 0.0005, "05 T6"),
    "t6r.loso.eegmamba.2027-2026.dif": (-0.003, 5e-05, "05 T6"),
    "t6r.loso.eegmamba.2027-2026.ic_inf": (-0.0111, 5e-05, "05 T6"),
    "t6r.loso.eegmamba.2027-2026.ic_sup": (0.0051, 5e-05, "05 T6"),
    "t6r.loso.eegmamba.2027-2026.p": (0.779, 0.0005, "05 T6"),
    "t6r.loso.eegmamba.2028-2026.dif": (-0.0014, 5e-05, "05 T6"),
    "t6r.loso.eegmamba.2028-2026.ic_inf": (-0.0091, 5e-05, "05 T6"),
    "t6r.loso.eegmamba.2028-2026.ic_sup": (0.0064, 5e-05, "05 T6"),
    "t6r.loso.eegmamba.2028-2026.p": (0.692, 0.0005, "05 T6"),
    "t6r.loso.eegmamba.2028-2027.dif": (0.0015, 5e-05, "05 T6"),
    "t6r.loso.eegmamba.2028-2027.ic_inf": (-0.0072, 5e-05, "05 T6"),
    "t6r.loso.eegmamba.2028-2027.ic_sup": (0.01, 5e-05, "05 T6"),
    "t6r.loso.eegmamba.2028-2027.p": (0.667, 0.0005, "05 T6"),
    "t7m.loso.conformer.augcr.s2026": (0.7661, 5e-05, "05 T7"),
    "t7m.loso.conformer.base.s2026": (0.7517, 5e-05, "05 T7"),
    "t7m.loso.conformer.base.s2027": (0.7443, 5e-05, "05 T7"),
    "t7m.loso.conformer.base.s2028": (0.7446, 5e-05, "05 T7"),
    "t7m.loso.conformer.ea.s2026": (0.7689, 5e-05, "05 T7"),
    "t7m.loso.eegmamba.augcr.s2026": (0.7244, 5e-05, "05 T7"),
    "t7m.loso.eegmamba.base.s2026": (0.7168, 5e-05, "05 T7"),
    "t7m.loso.eegmamba.base.s2027": (0.7138, 5e-05, "05 T7"),
    "t7m.loso.eegmamba.base.s2028": (0.7154, 5e-05, "05 T7"),
    "t7m.loso.eegmamba.ea.s2026": (0.7631, 5e-05, "05 T7"),
    "t7m.loso.eegnet.augcr.s2026": (0.76, 5e-05, "05 T7"),
    "t7m.loso.eegnet.base.s2026": (0.7625, 5e-05, "05 T7"),
    "t7m.loso.eegnet.base.s2027": (0.7595, 5e-05, "05 T7"),
    "t7m.loso.eegnet.base.s2028": (0.7596, 5e-05, "05 T7"),
    "t7m.loso.eegnet.ea.s2026": (0.7735, 5e-05, "05 T7"),
    "t7m.loso.labram.augcr.s2026": (0.7644, 5e-05, "05 T7"),
    "t7m.loso.labram.base.s2026": (0.7538, 5e-05, "05 T7"),
    "t7m.loso.labram.base.s2027": (0.7531, 5e-05, "05 T7"),
    "t7m.loso.labram.base.s2028": (0.7577, 5e-05, "05 T7"),
    "t7m.loso.labram.ea.s2026": (0.7787, 5e-05, "05 T7"),
    "t7m.within.conformer.augcr.s2026": (0.6288, 5e-05, "05 T7"),
    "t7m.within.conformer.augfft.s2026": (0.6079, 5e-05, "05 T7"),
    "t7m.within.conformer.augnone.s2026": (0.6001, 5e-05, "05 T7"),
    "t7m.within.conformer.base.s2026": (0.6069, 5e-05, "05 T7"),
    "t7m.within.conformer.base.s2027": (0.6, 5e-05, "05 T7"),
    "t7m.within.conformer.base.s2028": (0.6006, 5e-05, "05 T7"),
    "t7m.within.conformer.beta.s2026": (0.6424, 5e-05, "05 T7"),
    "t7m.within.conformer.mu.s2026": (0.6528, 5e-05, "05 T7"),
    "t7m.within.conformer.mubeta.s2026": (0.685, 5e-05, "05 T7"),
    "t7m.within.conformer.w02.s2026": (0.6025, 5e-05, "05 T7"),
    "t7m.within.conformer.w03.s2026": (0.5972, 5e-05, "05 T7"),
    "t7m.within.eegmamba.augcr.s2026": (0.6001, 5e-05, "05 T7"),
    "t7m.within.eegmamba.augfft.s2026": (0.57, 5e-05, "05 T7"),
    "t7m.within.eegmamba.base.s2026": (0.5769, 5e-05, "05 T7"),
    "t7m.within.eegmamba.base.s2027": (0.571, 5e-05, "05 T7"),
    "t7m.within.eegmamba.base.s2028": (0.5645, 5e-05, "05 T7"),
    "t7m.within.eegmamba.w02.s2026": (0.6011, 5e-05, "05 T7"),
    "t7m.within.eegnet.augcr.s2026": (0.6591, 5e-05, "05 T7"),
    "t7m.within.eegnet.augfft.s2026": (0.635, 5e-05, "05 T7"),
    "t7m.within.eegnet.augnone.s2026": (0.6405, 5e-05, "05 T7"),
    "t7m.within.eegnet.base.s2026": (0.6331, 5e-05, "05 T7"),
    "t7m.within.eegnet.base.s2027": (0.6307, 5e-05, "05 T7"),
    "t7m.within.eegnet.base.s2028": (0.639, 5e-05, "05 T7"),
    "t7m.within.eegnet.beta.s2026": (0.5875, 5e-05, "05 T7"),
    "t7m.within.eegnet.mu.s2026": (0.6031, 5e-05, "05 T7"),
    "t7m.within.eegnet.mubeta.s2026": (0.6271, 5e-05, "05 T7"),
    "t7m.within.eegnet.w02.s2026": (0.6748, 5e-05, "05 T7"),
    "t7m.within.eegnet.w03.s2026": (0.6376, 5e-05, "05 T7"),
    "t7m.within.labram.augcr.s2026": (0.6508, 5e-05, "05 T7"),
    "t7m.within.labram.augfft.s2026": (0.6616, 5e-05, "05 T7"),
    "t7m.within.labram.base.s2026": (0.6647, 5e-05, "05 T7"),
    "t7m.within.labram.base.s2027": (0.6632, 5e-05, "05 T7"),
    "t7m.within.labram.base.s2028": (0.662, 5e-05, "05 T7"),
    "t7m.within.labram.w02.s2026": (0.6431, 5e-05, "05 T7"),
    "t8.within.labram.bruta": (0.0092, 5e-05, "05 T8"),
    "t8.within.labram.neta": (0.0182, 5e-05, "05 T8"),
    "t8.within.labram.uso": (0.0182, 5e-05, "05 T8"),
    "t8.loso.eegnet.bruta": (0.0104, 5e-05, "05 T8"),
    "t8.loso.eegnet.neta": (0.01, 5e-05, "05 T8"),
    "t8.loso.eegnet.uso": (0.0104, 5e-05, "05 T8"),
    "t8.loso.conformer.bruta": (0.0198, 5e-05, "05 T8"),
    "t8.loso.conformer.neta": (0.0178, 5e-05, "05 T8"),
    "t8.loso.conformer.uso": (0.0198, 5e-05, "05 T8"),
    "t8.loso.labram.bruta": (0.0097, 5e-05, "05 T8"),
    "t8.loso.labram.neta": (0.0142, 5e-05, "05 T8"),
    "t8.loso.labram.uso": (0.0142, 5e-05, "05 T8"),
    "t9p.eegnet.base_2026.suelo": (16.6, 0.05, "05 T9"),
    "t9p.eegnet.base_2026.mediana": (61.0, 0.5, "05 T9"),
    "t9p.eegnet.base_2026.tope": (8.1, 0.05, "05 T9"),
    "t9p.eegnet.mu.suelo": (22.4, 0.05, "05 T9"),
    "t9p.eegnet.mu.mediana": (56.0, 0.5, "05 T9"),
    "t9p.eegnet.mu.tope": (11.7, 0.05, "05 T9"),
    "t9p.eegnet.beta.suelo": (31.4, 0.05, "05 T9"),
    "t9p.eegnet.beta.mediana": (44.0, 0.5, "05 T9"),
    "t9p.eegnet.beta.tope": (14.1, 0.05, "05 T9"),
    "t9p.eegnet.mubeta.suelo": (22.8, 0.05, "05 T9"),
    "t9p.eegnet.mubeta.mediana": (58.0, 0.5, "05 T9"),
    "t9p.eegnet.mubeta.tope": (14.4, 0.05, "05 T9"),
    "t9p.eegnet.w02.suelo": (7.6, 0.05, "05 T9"),
    "t9p.eegnet.w02.mediana": (71.0, 0.5, "05 T9"),
    "t9p.eegnet.w02.tope": (7.0, 0.05, "05 T9"),
    "t9p.eegnet.w03.suelo": (13.8, 0.05, "05 T9"),
    "t9p.eegnet.w03.mediana": (63.0, 0.5, "05 T9"),
    "t9p.eegnet.w03.tope": (8.0, 0.05, "05 T9"),
    "t9p.eegnet.augnone.suelo": (12.0, 0.05, "05 T9"),
    "t9p.eegnet.augnone.mediana": (64.0, 0.5, "05 T9"),
    "t9p.eegnet.augnone.tope": (12.0, 0.05, "05 T9"),
    "t9p.eegnet.augcr.suelo": (12.2, 0.05, "05 T9"),
    "t9p.eegnet.augcr.mediana": (86.0, 0.5, "05 T9"),
    "t9p.eegnet.augcr.tope": (12.8, 0.05, "05 T9"),
    "t9p.eegnet.augfft.suelo": (15.2, 0.05, "05 T9"),
    "t9p.eegnet.augfft.mediana": (61.0, 0.5, "05 T9"),
    "t9p.eegnet.augfft.tope": (7.7, 0.05, "05 T9"),
    "t9p.conformer.base_2026.suelo": (53.1, 0.05, "05 T9"),
    "t9p.conformer.base_2026.mediana": (30.0, 0.5, "05 T9"),
    "t9p.conformer.base_2026.tope": (0.0, 0.05, "05 T9"),
    "t9p.conformer.mu.suelo": (29.2, 0.05, "05 T9"),
    "t9p.conformer.mu.mediana": (34.0, 0.5, "05 T9"),
    "t9p.conformer.mu.tope": (0.0, 0.05, "05 T9"),
    "t9p.conformer.beta.suelo": (28.4, 0.05, "05 T9"),
    "t9p.conformer.beta.mediana": (35.0, 0.5, "05 T9"),
    "t9p.conformer.beta.tope": (0.0, 0.05, "05 T9"),
    "t9p.conformer.mubeta.suelo": (21.9, 0.05, "05 T9"),
    "t9p.conformer.mubeta.mediana": (36.0, 0.5, "05 T9"),
    "t9p.conformer.mubeta.tope": (0.0, 0.05, "05 T9"),
    "t9p.conformer.w02.suelo": (63.1, 0.05, "05 T9"),
    "t9p.conformer.w02.mediana": (30.0, 0.5, "05 T9"),
    "t9p.conformer.w02.tope": (0.0, 0.05, "05 T9"),
    "t9p.conformer.w03.suelo": (61.9, 0.05, "05 T9"),
    "t9p.conformer.w03.mediana": (30.0, 0.5, "05 T9"),
    "t9p.conformer.w03.tope": (0.0, 0.05, "05 T9"),
    "t9p.conformer.augnone.suelo": (58.9, 0.05, "05 T9"),
    "t9p.conformer.augnone.mediana": (30.0, 0.5, "05 T9"),
    "t9p.conformer.augnone.tope": (0.0, 0.05, "05 T9"),
    "t9p.conformer.augcr.suelo": (28.1, 0.05, "05 T9"),
    "t9p.conformer.augcr.mediana": (41.0, 0.5, "05 T9"),
    "t9p.conformer.augcr.tope": (0.0, 0.05, "05 T9"),
    "t9p.conformer.augfft.suelo": (55.3, 0.05, "05 T9"),
    "t9p.conformer.augfft.mediana": (30.0, 0.5, "05 T9"),
    "t9p.conformer.augfft.tope": (0.0, 0.05, "05 T9"),
    "t9p.labram.base_2026.suelo": (21.1, 0.05, "05 T9"),
    "t9p.labram.base_2026.mediana": (43.0, 0.5, "05 T9"),
    "t9p.labram.base_2026.tope": (0.0, 0.05, "05 T9"),
    "t9p.labram.w02.suelo": (28.5, 0.05, "05 T9"),
    "t9p.labram.w02.mediana": (37.0, 0.5, "05 T9"),
    "t9p.labram.w02.tope": (0.0, 0.05, "05 T9"),
    "t9p.labram.augcr.suelo": (26.9, 0.05, "05 T9"),
    "t9p.labram.augcr.mediana": (44.0, 0.5, "05 T9"),
    "t9p.labram.augcr.tope": (0.0, 0.05, "05 T9"),
    "t9p.labram.augfft.suelo": (25.4, 0.05, "05 T9"),
    "t9p.labram.augfft.mediana": (40.0, 0.5, "05 T9"),
    "t9p.labram.augfft.tope": (0.0, 0.05, "05 T9"),
    "t9p.eegmamba.base_2026.suelo": (85.6, 0.05, "05 T9"),
    "t9p.eegmamba.base_2026.mediana": (30.0, 0.5, "05 T9"),
    "t9p.eegmamba.base_2026.tope": (0.0, 0.05, "05 T9"),
    "t9p.eegmamba.w02.suelo": (79.4, 0.05, "05 T9"),
    "t9p.eegmamba.w02.mediana": (30.0, 0.5, "05 T9"),
    "t9p.eegmamba.w02.tope": (0.0, 0.05, "05 T9"),
    "t9p.eegmamba.augcr.suelo": (58.4, 0.05, "05 T9"),
    "t9p.eegmamba.augcr.mediana": (30.0, 0.5, "05 T9"),
    "t9p.eegmamba.augcr.tope": (0.0, 0.05, "05 T9"),
    "t9p.eegmamba.augfft.suelo": (87.2, 0.05, "05 T9"),
    "t9p.eegmamba.augfft.mediana": (30.0, 0.5, "05 T9"),
    "t9p.eegmamba.augfft.tope": (0.0, 0.05, "05 T9"),
})
# ---- fin bloque generado ----

# Cifras cuyo valor declarado se actualizo al redondear las diferencias a 1e-9 antes de Wilcoxon/TOST
# (ruido_flotante.py): id -> valor que traian los informes 03, 04 y 05 antes del cambio.
ANTERIOR_AL_REDONDEO = {
    "fs_base.eegnet.10.stats.p": 0.0033,
    "fs_base.eegnet.10.stats.p_holm": 0.0332,
    "fs_base.eegnet.10.cabeza.p_holm": 0.014,
    "fs_base.eegnet.10.stats+cabeza.p_holm": 0.0038,
    "fs_base.eegnet.20.stats.p": 0.0079,
    "fs_base.eegnet.20.stats.p_holm": 0.0563,
    "fs_base.eegnet.20.cabeza.p": 0.0376,
    "fs_base.eegnet.20.cabeza.p_holm": 0.1503,
    "fs_base.eegnet.20.stats+cabeza.p_holm": 0.0332,
    "fs_base.conformer.10.stats.p_holm": 0.0102,
    "fs_base.conformer.10.cabeza.r": 0.62,
    "fs_base.conformer.20.stats.p": 0.0023,
    "fs_base.conformer.20.stats.p_holm": 0.0257,
    "fs_base.conformer.20.cabeza.p": 0.2364,
    "fs_base.conformer.20.cabeza.p_holm": 0.7093,
    "fs_base.conformer.20.stats+cabeza.p": 0.0014,
    "fs_base.conformer.20.stats+cabeza.p_holm": 0.0179,
    "fs_base.labram.10.cabeza.p_holm": 0.0102,
    "fs_base.labram.20.cabeza.p": 0.007,
    "fs_base.labram.20.cabeza.p_holm": 0.0563,
    "fs_base.eegmamba.10.stats.p": 0.3502,
    "fs_base.eegmamba.10.stats.p_holm": 0.7093,
    "fs_base.eegmamba.10.cabeza.p": 0.0081,
    "fs_base.eegmamba.10.cabeza.p_holm": 0.0563,
    "fs_base.eegmamba.10.stats+cabeza.p_holm": 0.0048,
    "fs_base.eegmamba.20.stats+cabeza.p_holm": 0.0161,
    "fs_ea.eegnet.10.stats.p": 0.0143,
    "fs_ea.eegnet.10.stats.p_holm": 0.2069,
    "fs_ea.eegnet.10.cabeza.p": 0.0104,
    "fs_ea.eegnet.10.cabeza.p_holm": 0.1771,
    "fs_ea.eegnet.10.stats+cabeza.p_holm": 0.0081,
    "fs_ea.eegnet.20.stats.p": 0.0783,
    "fs_ea.eegnet.20.stats.p_holm": 0.7043,
    "fs_ea.eegnet.20.cabeza.p": 0.1517,
    "fs_ea.eegnet.20.cabeza.p_holm": 0.9463,
    "fs_ea.eegnet.20.stats+cabeza.p": 0.0571,
    "fs_ea.eegnet.20.stats+cabeza.p_holm": 0.6276,
    "fs_ea.conformer.10.stats.p": 0.0349,
    "fs_ea.conformer.10.stats.p_holm": 0.4537,
    "fs_ea.conformer.10.cabeza.p_holm": 0.0295,
    "fs_ea.conformer.10.stats+cabeza.r": 0.62,
    "fs_ea.conformer.20.stats.p": 0.1863,
    "fs_ea.conformer.20.stats.p_holm": 0.9463,
    "fs_ea.conformer.20.cabeza.p": 0.4512,
    "fs_ea.conformer.20.cabeza.r": 0.1,
    "fs_ea.conformer.20.stats+cabeza.p": 0.0118,
    "fs_ea.conformer.20.stats+cabeza.p_holm": 0.1885,
    "fs_ea.labram.10.stats.p": 0.0138,
    "fs_ea.labram.10.stats.p_holm": 0.2069,
    "fs_ea.labram.10.stats.r": 0.34,
    "fs_ea.labram.10.cabeza.p": 0.0021,
    "fs_ea.labram.10.cabeza.p_holm": 0.0444,
    "fs_ea.labram.10.stats+cabeza.p_holm": 0.062,
    "fs_ea.labram.20.stats.p": 0.0673,
    "fs_ea.labram.20.stats.p_holm": 0.6729,
    "fs_ea.labram.20.cabeza.p_holm": 0.8818,
    "fs_ea.labram.20.stats+cabeza.p": 0.0435,
    "fs_ea.labram.20.stats+cabeza.p_holm": 0.5217,
    "fs_ea.labram.20.stats+cabeza.r": 0.27,
    "fs_ea.eegmamba.10.stats.p": 0.1352,
    "fs_ea.eegmamba.10.stats.p_holm": 0.9463,
    "fs_ea.eegmamba.10.stats.r": 0.2,
    "fs_ea.eegmamba.10.cabeza.p": 0.0052,
    "fs_ea.eegmamba.10.cabeza.p_holm": 0.0937,
    "fs_ea.eegmamba.10.stats+cabeza.p": 0.0036,
    "fs_ea.eegmamba.10.stats+cabeza.p_holm": 0.0677,
    "fs_ea.eegmamba.20.stats.p": 0.2742,
    "fs_ea.eegmamba.20.cabeza.p": 0.2742,
    "fs_ea.eegmamba.20.stats+cabeza.p": 0.2935,
    "fs_sem.eegnet.10.stats.p2027": 0.038,
    "fs_sem.eegnet.10.stats.p2028": 0.0093,
    "fs_sem.eegnet.10.cabeza.p2028": 0.0061,
    "fs_sem.eegnet.20.stats.p2027": 0.0812,
    "fs_sem.eegnet.20.cabeza.p2027": 0.0858,
    "fs_sem.eegnet.20.cabeza.p2028": 0.0467,
    "fs_sem.eegnet.20.stats+cabeza.p2027": 0.0167,
    "fs_sem.conformer.10.stats.p2027": 0.0029,
    "fs_sem.conformer.10.stats.p2028": 0.0003,
    "fs_sem.conformer.20.cabeza.p2027": 0.394,
    "fs_sem.conformer.20.cabeza.p2028": 0.1363,
    "fs_sem.labram.10.stats.p2027": 0.0024,
    "fs_sem.labram.10.stats.p2028": 0.0063,
    "fs_sem.labram.10.cabeza.p2027": 0.0121,
    "fs_sem.labram.20.cabeza.p2027": 0.0223,
    "fs_sem.labram.20.cabeza.p2028": 0.031,
    "fs_sem.labram.20.stats+cabeza.p2027": 0.0044,
    "fs_sem.labram.20.stats+cabeza.p2028": 0.0109,
    "fs_sem.eegmamba.10.stats.p2027": 0.6114,
    "fs_sem.eegmamba.10.stats.p2028": 0.4695,
    "fs_sem.eegmamba.10.cabeza.p2028": 0.0093,
    "fs_sem.eegmamba.20.stats.p2027": 0.7829,
    "fs_sem.eegmamba.20.stats.p2028": 0.7664,
    "fs_sem.eegmamba.20.cabeza.p2027": 0.0353,
    "fs_sem.eegmamba.20.cabeza.p2028": 0.0297,
    "fs_sem.eegmamba.20.stats+cabeza.p2027": 0.0036,
    "tr.idp.eegmamba.media": 0.136,
    "polc.eegnet.per_trial.p": 0.0013,
    "polc.eegnet.per_trial.p_holm": 0.0091,
    "polc.eegnet.per_trial.r": 0.45,
    "polc.eegnet.per_subject.p": 0.0511,
    "polc.eegnet.per_subject.p_holm": 0.2046,
    "polc.eegnet.per_subject.r": 0.27,
    "polc.conformer.per_subject.p": 0.0321,
    "polc.conformer.per_subject.p_holm": 0.1607,
    "polc.conformer.per_subject.r": 0.3,
    "polc.labram.per_trial.p": 0.0184,
    "polc.labram.per_trial.p_holm": 0.1102,
    "polc.labram.per_trial.r": 0.33,
    "polc.labram.per_subject.p": 0.7111,
    "polc.eegmamba.per_trial.p": 0.6027,
    "polc.eegmamba.per_trial.r": 0.07,
    "polc.eegmamba.per_subject.p": 0.2603,
    "polc.eegmamba.per_subject.p_holm": 0.781,
    "eatc.eegnet.cruzado_51.p": 0.4584,
    "eatc.eegnet.cruzado_51.r": 0.1,
    "eatc.eegnet.h44_wbcic_51.p": 0.3769,
    "eatc.eegnet.h44_wbcic_51.r": 0.12,
    "eatc.conformer.cruzado_51.p": 0.4245,
    "eatc.conformer.cruzado_51.r": 0.11,
    "eatc.conformer.h44_wbcic_51.p": 0.2134,
    "eatc.conformer.h44_wbcic_51.r": 0.17,
    "eatc.labram.cruzado_51.p": 0.3251,
    "eatc.labram.cruzado_51.r": 0.14,
    "eatc.labram.h44_wbcic_51.p": 0.0186,
    "eatc.labram.h44_wbcic_51.p_holm": 0.1863,
    "eatc.labram.h44_wbcic_51.r": 0.33,
    "eatc.eegmamba.cruzado_51.p": 0.5763,
    "eatc.eegmamba.h44_wbcic_51.p": 0.7819,
    "eatc.eegmamba.h44_wbcic_51.r": 0.04,
    "t1.loso.conformer.dcruce_p": 0.41,
    "t1.loso.conformer.dintra_p": 0.42,
    "t1.loso.labram.dcruce_p": 0.9,
    "t1.loso.labram.dintra_p": 0.32,
    "t1.loso.eegmamba.dintra_p": 0.65,
    "t1.within.eegnet.dcruce_p": 0.89,
    "t1.within.eegnet.dintra_p": 0.01,
    "t1.within.conformer.dintra_p": 0.54,
    "t1.within.labram.dintra_p": 0.89,
    "t3.loso.eegnet.p": 0.97,
    "t3.loso.labram.p": 0.56,
    "t3.within.eegnet.p": 0.12,
    "t6.loso.eegnet-conformer.s2026.p_holm": 0.36,
    "t6.loso.eegnet-conformer.p_s2028": 0.48,
    "t6.loso.eegnet-labram.s2026.p_holm": 0.54,
    "t6.loso.eegnet-labram.p_s2028": 0.87,
    "t6.loso.conformer-labram.s2026.p": 0.52,
    "t6.loso.conformer-labram.s2026.p_holm": 0.54,
    "t6.loso.conformer-labram.p_s2026": 0.52,
    "t6.loso.conformer-labram.p_s2027": 0.31,
    "t6.loso.conformer-labram.p_s2028": 0.49,
    "t6.within.eegnet-labram.s2026.p": 0.29,
    "t6.within.eegnet-labram.s2026.p_holm": 0.29,
    "t6.within.eegnet-labram.p_s2026": 0.29,
    "t6.within.eegnet-labram.p_s2028": 0.59,
    "t6.within.conformer-labram.s2026.r": 0.4,
    "t6.within.conformer-labram.s2026.p_holm": 0.007,
    "t6.within.conformer-labram.media3.r": 0.45,
    "t6.within.eegnet-eegmamba.media3.r": 0.72,
    "t10.conformer.s2028.p_holm": 0.22,
    "f4.eegnet.mubeta.p": 0.443,
    "f4.eegnet.mubeta.p_holm": 0.887,
    "f4.eegnet.w02.p": 6e-07,
    "f4.eegnet.w02.p_holm": 4.8e-06,
    "f4.eegnet.w02.p_holm22": 1.3000000000000001e-05,
    "f4.eegnet.w03.p": 0.272,
    "f4.eegnet.w03.p_holm": 0.817,
    "f4.eegnet.augnone.p": 0.196,
    "f4.eegnet.augnone.p_holm": 0.786,
    "f4.eegnet.augfft.p": 0.564,
    "f4.eegnet.augfft.p_holm": 0.887,
    "f4.conformer.mu.p": 0.055,
    "f4.conformer.mu.p_holm": 0.243,
    "f4.conformer.mu.p_holm22": 0.665,
    "f4.conformer.beta.p": 0.153,
    "f4.conformer.beta.p_holm": 0.459,
    "f4.conformer.beta.r": 0.19,
    "f4.conformer.w02.p": 0.721,
    "f4.conformer.w03.p": 0.019,
    "f4.conformer.w03.p_holm": 0.116,
    "f4.conformer.w03.r": 0.32,
    "f4.conformer.w03.p_holm22": 0.271,
    "f4.conformer.augnone.p": 0.049,
    "f4.conformer.augnone.p_holm": 0.243,
    "f4.conformer.augnone.p_holm22": 0.631,
    "f4.conformer.augcr.p": 0.00025,
    "f4.conformer.augfft.p": 0.948,
    "f4.labram.w02.p": 0.059,
    "f4.labram.w02.p_holm": 0.119,
    "f4.labram.w02.p_holm22": 0.665,
    "f4.labram.augcr.p": 0.013,
    "f4.labram.augcr.p_holm": 0.04,
    "f4.labram.augcr.r": 0.34,
    "f4.labram.augcr.p_holm22": 0.202,
    "f4.labram.augfft.p": 0.257,
    "f4.labram.augfft.p_holm": 0.257,
    "f4.labram.augfft.r": 0.15,
    "f4.eegmamba.w02.p_holm22": 0.024,
    "f4.eegmamba.augcr.p": 0.008,
    "f4.eegmamba.augcr.p_holm": 0.016,
    "f4.eegmamba.augcr.p_holm22": 0.126,
    "f4.eegmamba.augfft.p": 0.121,
    "f4.eegmamba.augfft.p_holm": 0.121,
    "f4s.labram.w02.p": 0.172,
    "f4s.labram.w02.p_holm": 0.345,
    "f4s.labram.augcr.p": 0.031,
    "f4s.labram.augcr.p_holm": 0.094,
    "f4s.labram.augfft.p": 0.925,
    "f4s.labram.augfft.p_holm": 0.925,
    "f4s.eegmamba.w02.p_holm": 4.4000000000000006e-05,
    "f4s.eegmamba.augcr.p_holm": 4.4000000000000006e-05,
    "f4s.eegmamba.augfft.p": 0.32,
    "f4s.eegmamba.augfft.p_holm": 0.32,
    "f4s.eegmamba.augfft.r": 0.14,
    "f6.eegnet.p": 0.012,
    "f6.eegnet.p_holm": 0.012,
    "f6.eegnet.p_holm8": 0.048,
    "f6.conformer.r": 0.44,
    "f6.labram.p": 0.00014,
    "f6.labram.p_holm": 0.00042,
    "f6.labram.p_holm8": 0.00099,
    "f6.eegmamba.p": 1.8e-07,
    "f6.eegmamba.p_holm": 7.2e-07,
    "f6.eegmamba.p_holm8": 1.4e-06,
    "f6s.conformer.p": 0.00053,
    "f6s.labram.p": 2.6000000000000002e-05,
    "f6s.labram.p_holm": 7.900000000000001e-05,
    "f6s.eegmamba.p": 5.7e-08,
    "f6s.eegmamba.p_holm": 2.2999999999999997e-07,
    "f8.eegnet.p": 0.75,
    "f8.eegnet.p_holm": 0.75,
    "f8.eegnet.p_holm8": 0.75,
    "f8.conformer.p_holm": 0.02,
    "f8.conformer.p_holm8": 0.025,
    "f8.labram.p": 0.044,
    "f8.labram.p_holm": 0.133,
    "f8.labram.p_holm8": 0.133,
    "f8.eegmamba.p": 0.074,
    "f8.eegmamba.p_holm": 0.148,
    "f8.eegmamba.r": 0.24,
    "f8.eegmamba.p_holm8": 0.148,
    "f8s.conformer.p": 0.0009699999999999999,
    "f8s.labram.p": 0.015,
    "t4ea.eegnet.p": 0.627,
    "t6r.within.eegnet.2027-2026.p": 0.205,
    "t6r.within.eegnet.2028-2026.p": 0.228,
    "t6r.within.eegnet.2028-2027.p": 0.014,
    "t6r.within.conformer.2027-2026.p": 0.046,
    "t6r.within.conformer.2028-2026.p": 0.07,
    "t6r.within.conformer.2028-2027.p": 0.538,
    "t6r.within.labram.2028-2026.p": 0.235,
    "t6r.within.labram.2028-2027.p": 0.89,
    "t6r.within.eegmamba.2027-2026.p": 0.366,
    "t6r.within.eegmamba.2028-2026.p": 0.016,
    "t6r.within.eegmamba.2028-2027.p": 0.107,
    "t6r.loso.eegnet.2027-2026.p": 0.904,
    "t6r.loso.eegnet.2028-2026.p": 0.701,
    "t6r.loso.eegnet.2028-2027.p": 0.25,
    "t6r.loso.conformer.2027-2026.p": 0.303,
    "t6r.loso.conformer.2028-2027.p": 0.418,
    "t6r.loso.labram.2027-2026.p": 0.773,
    "t6r.loso.labram.2028-2026.p": 0.467,
    "t6r.loso.labram.2028-2027.p": 0.32,
    "t6r.loso.eegmamba.2027-2026.p": 0.78,
    "t6r.loso.eegmamba.2028-2026.p": 0.698,
    "t6r.loso.eegmamba.2028-2027.p": 0.651,
}



# ================================================================== BLOQUES cnt, camb, lect, anom: afirmaciones de prosa

def _cat_r(r):
    return 3 if r >= 0.5 else 2 if r >= 0.3 else 1 if r >= 0.1 else 0


def b_prosa() -> dict:
    """Recuentos y extremos que los informes citan en el texto (no en una tabla)."""
    out = {}
    fb, fe, fsem = valores(b_fs), None, None
    for var, pref in (("base", "fs_base"), ("ea", "fs_ea")):
        g = [fb[f"{pref}.{m}.{k}.{me}.gan"] for m in ARQ for k in KS for me in METODOS]
        out[f"cnt.{pref}.n_neg"] = sum(x < 0 for x in g)
        out[f"cnt.{pref}.n_holm"] = sum(fb[f"{pref}.{m}.{k}.{me}.p_holm"] < 0.05
                                         for m in ARQ for k in KS for me in METODOS)
        out[f"cnt.{pref}.gan_min"], out[f"cnt.{pref}.gan_max"] = min(g), max(g)
        out[f"cnt.{pref}.max_ic_sup"] = max(fb[f"{pref}.{m}.{k}.{me}.ic_sup"]
                                            for m in ARQ for k in KS for me in METODOS)
        out[f"cnt.{pref}.n_equiv"] = sum(fb[f"{pref}.{m}.{k}.{me}.equiv"]
                                         for m in ARQ for k in KS for me in METODOS)
    ks = [(m, k, me) for m in ARQ for k in KS for me in METODOS]
    out["cnt.fs_sem.n_mismo_signo"] = sum(
        np.sign(fb[f"fs_sem.{m}.{k}.{me}.gan2027"]) == np.sign(fb[f"fs_sem.{m}.{k}.{me}.gan2028"])
        for m, k, me in ks)
    out["cnt.fs_sem.n_p05_ambas"] = sum(
        fb[f"fs_sem.{m}.{k}.{me}.p2027"] < 0.05 and fb[f"fs_sem.{m}.{k}.{me}.p2028"] < 0.05
        for m, k, me in ks)
    out["cnt.fs_sem.max_dif_semillas"] = max(
        abs(fb[f"fs_sem.{m}.{k}.{me}.gan2027"] - fb[f"fs_sem.{m}.{k}.{me}.gan2028"]) for m, k, me in ks)
    rho = [fb[f"fs_sem.{m}.{k}.{me}.rho"] for m, k, me in ks]
    out["cnt.fs_sem.rho_min"], out["cnt.fs_sem.rho_max"] = min(rho), max(rho)
    # intervalos de las caidas que excluyen el cero (03 sec. 3.4)
    tr = valores(b_tr)
    for q in ("caida_persona", "caida_protocolo", "protocolo_menos_persona"):
        for m in ARQ + [None]:
            ms = ARQ if m is None else [m]
            n = sum(tr[f"trc.{q}.{mm}.{v}.inf"] > 0 or tr[f"trc.{q}.{mm}.{v}.sup"] < 0
                    for mm in ms for v in ("s2026", "sujetos3", "jer3", "s2027", "s2028"))
            out[f"cnt.trc.{q}.{m or 'todas'}.excluyen"] = n
    # cambios de conclusion de la semilla 2026 a la media de 3 semillas (04 sec. 2.5)
    t6 = valores(b_t6)
    n05 = nholm = ncat = 0
    dd = []
    for prot in ("loso", "within"):
        for a, b in PARES:
            k = f"t6.{prot}.{a}-{b}"
            n05 += (t6[f"{k}.s2026.p"] < 0.05) != (t6[f"{k}.media3.p"] < 0.05)
            nholm += (t6[f"{k}.s2026.p_holm"] < 0.05) != (t6[f"{k}.media3.p_holm"] < 0.05)
            ncat += _cat_r(t6[f"{k}.s2026.r"]) != _cat_r(t6[f"{k}.media3.r"])
            m2728 = 0.5 * (t6[f"{k}.dif_s2027"] + t6[f"{k}.dif_s2028"])
            dd.append(t6[f"{k}.dif_s2026"] - m2728)
    out["camb.n_p05"], out["camb.n_holm"], out["camb.n_categoria_r"] = n05, nholm, ncat
    out["camb.dif2026_menos_2728_min"], out["camb.dif2026_menos_2728_max"] = min(dd), max(dd)
    # lectura frente a la cota (05 T8): 1 si |dif| supera la cota de uso
    ab, co = valores(b_abl), valores(b_confusor)
    for fam, m in (("f6", "eegnet"), ("f8", "eegnet"), ("f6", "conformer"), ("f8", "conformer"),
                   ("f6", "labram"), ("f8", "labram")):
        out[f"lect.{fam}.{m}.supera"] = float(abs(ab[f"{fam}.{m}.dif"]) > co[f"t4.loso.{m}.cota_uso"])
    for v in ("w02", "augcr", "augfft"):
        out[f"lect.f4.labram.{v}.supera"] = float(abs(ab[f"f4.labram.{v}.dif"]) > co["t4.within.labram.cota_uso"])
    # anomalias de una sola semilla y caso de coma flotante (04 sec. 1.2 y 0)
    lo, wi = loso_sub("conformer"), within_sub("eegnet")
    for sub, tab, nom in (("sub-21", lo, "conformer.loso"), ("sub-52", lo, "conformer.loso"),
                          ("sub-21", wi, "eegnet.within")):
        for sd in SEM:
            out[f"anom.{nom}.{sub}.s{sd}"] = float(tab.loc[sub, sd])
    out["anom.eegnet.within.sub-31.media3"] = float(wi.loc["sub-31"].mean())
    return out


BLOQUES.update({"cnt": b_prosa, "camb": b_prosa, "lect": b_prosa, "anom": b_prosa})

DECL_MANUAL = {
    # 03 resumen y secciones 2.3 a 2.5, 3.4
    "cnt.fs_base.n_neg": (24, 0.5, "03 sec. 0"), "cnt.fs_base.n_holm": (16, 0.5, "03 sec. 0"),
    "cnt.fs_base.gan_min": (-0.0409, 5e-5, "03 sec. 0"), "cnt.fs_base.gan_max": (-0.0007, 5e-5, "03 sec. 0"),
    "cnt.fs_base.max_ic_sup": (0.0054, 5e-5, "03 sec. 0"), "cnt.fs_base.n_equiv": (18, 0.5, "03 sec. 2.3 (24 menos 6)"),
    "cnt.fs_ea.n_neg": (24, 0.5, "03 sec. 0"), "cnt.fs_ea.n_holm": (4, 0.5, "03 sec. 0"),
    "cnt.fs_ea.gan_min": (-0.0199, 5e-5, "03 sec. 0"), "cnt.fs_ea.gan_max": (-0.0010, 5e-5, "03 sec. 0"),
    "cnt.fs_ea.n_equiv": (24, 0.5, "03 sec. 2.5"),
    "cnt.fs_sem.n_mismo_signo": (23, 0.5, "03 sec. 2.4"), "cnt.fs_sem.n_p05_ambas": (19, 0.5, "03 sec. 2.4"),
    "cnt.fs_sem.max_dif_semillas": (0.0106, 5e-5, "03 sec. 2.4"),
    "cnt.fs_sem.rho_min": (0.25, 0.005, "03 sec. 2.4"), "cnt.fs_sem.rho_max": (0.74, 0.005, "03 sec. 2.4"),
    "cnt.trc.caida_protocolo.todas.excluyen": (20, 0.5, "03 sec. 3.4"),
    "cnt.trc.protocolo_menos_persona.todas.excluyen": (0, 0.5, "03 sec. 3.4"),
    "cnt.trc.caida_persona.conformer.excluyen": (5, 0.5, "03 sec. 3.4"),
    "cnt.trc.caida_persona.eegnet.excluyen": (0, 0.5, "03 sec. 3.4"),
    "cnt.trc.caida_persona.labram.excluyen": (4, 0.5, "03 sec. 3.4"),
    "cnt.trc.caida_persona.eegmamba.excluyen": (2, 0.5, "03 sec. 3.4"),
    # 04 sec. 2.5
    "camb.n_p05": (0, 0.5, "04 sec. 2.5"), "camb.n_holm": (0, 0.5, "04 sec. 2.5"),
    "camb.n_categoria_r": (3, 0.5, "04 sec. 2.5"),
    "camb.dif2026_menos_2728_min": (-0.011, 5e-4, "04 sec. 2.5"),
    "camb.dif2026_menos_2728_max": (0.009, 5e-4, "04 sec. 2.5"),
    "anom.conformer.loso.sub-21.s2026": (0.950, 5e-4, "04 sec. 1.2"),
    "anom.conformer.loso.sub-21.s2027": (0.940, 5e-4, "04 sec. 1.2"),
    "anom.conformer.loso.sub-21.s2028": (0.542, 5e-4, "04 sec. 1.2"),
    "anom.conformer.loso.sub-52.s2026": (0.810, 5e-4, "04 sec. 1.2"),
    "anom.conformer.loso.sub-52.s2027": (0.590, 5e-4, "04 sec. 1.2"),
    "anom.conformer.loso.sub-52.s2028": (0.845, 5e-4, "04 sec. 1.2"),
    "anom.eegnet.within.sub-21.s2026": (0.658, 5e-4, "04 sec. 1.2"),
    "anom.eegnet.within.sub-21.s2027": (0.910, 5e-4, "04 sec. 1.2"),
    "anom.eegnet.within.sub-21.s2028": (0.640, 5e-4, "04 sec. 1.2"),
    "anom.eegnet.within.sub-31.media3": (0.700, 1e-9, "04 sec. 0 (caso de coma flotante)"),
    # 05 T8, columna "lectura"
    "lect.f6.eegnet.supera": (1, 0.5, "05 T8"), "lect.f8.eegnet.supera": (0, 0.5, "05 T8"),
    "lect.f6.conformer.supera": (0, 0.5, "05 T8"), "lect.f8.conformer.supera": (0, 0.5, "05 T8"),
    "lect.f6.labram.supera": (1, 0.5, "05 T8"), "lect.f8.labram.supera": (0, 0.5, "05 T8"),
    "lect.f4.labram.w02.supera": (1, 0.5, "05 T8"), "lect.f4.labram.augcr.supera": (0, 0.5, "05 T8"),
    "lect.f4.labram.augfft.supera": (0, 0.5, "05 T8"),
}
DECLARADOS.update(DECL_MANUAL)




# ================================================================== BLOQUE p3: media de 3 semillas (P3_tres_semillas.md)
# Cifras de main.tex que seguian apoyandose en la semilla 2026 y se recalculan con la media de las tres
# semillas por sujeto (o por fold en LOSO). Recalculo independiente (aqui, desde los CSV crudos de
# avance7_datos.py); la salida de scripts/avance7_tres_semillas_p3.py son los CSV tres_semillas_p3_*.csv.
# Los valores declarados se copiaron de results/avance_7/analisis/tres_semillas_p3_cifras.csv.

def _hboot_media(Xm, nivel=0.95):
    """IC jerarquico (sujetos y, dentro, semillas) de la media de las medias de semillas; generador 2026."""
    n, k = Xm.shape
    g = np.random.default_rng(SEMILLA_BOOT)
    si = g.integers(0, n, (B, n))
    ia = g.integers(0, k, (B, n, k))
    m = Xm[si[:, :, None], ia].mean(axis=2).mean(axis=1)
    return float(np.quantile(m, (1 - nivel) / 2)), float(np.quantile(m, 1 - (1 - nivel) / 2))


def b_p3() -> dict:
    import ast
    out = {}
    L = {m: {c: loso_sub(m, c) for c in ("acc", "kappa", "acc_visto", "kappa_visto")} for m in ARQ}
    W = {m: within_sub(m) for m in ARQ}
    m3 = lambda m, c="acc": L[m][c][list(SEM)].mean(axis=1)
    resid = ["sub-06", "sub-29", "sub-32", "sub-37"]
    apart = ["sub-04", "sub-07", "sub-13", "sub-17", "sub-30", "sub-34", "sub-36", "sub-47", "sub-51", "sub-52"]
    # percentil 5, recuento bajo umbral e IC jerarquico de la exactitud
    conj = {}
    for m in ARQ:
        out[f"p3.p5.{m}"] = float(np.percentile(m3(m), 5))
        conj[m] = set(m3(m).index[le_umbral(m3(m))])
        out[f"p3.loso.n_bajo.{m}"] = len(conj[m])
        for prot, T in (("loso", L[m]["acc"]), ("intra", W[m])):
            Xm = T[list(SEM)].to_numpy()
            lo, hi = _hboot_media(Xm)
            out[f"p3.ic.{prot}.{m}.media"] = float(Xm.mean())
            out[f"p3.ic.{prot}.{m}.inf"], out[f"p3.ic.{prot}.{m}.sup"] = lo, hi
        # limite superior de la caida por persona (visto - no visto) en LOSO
        x = (L[m]["acc_visto"] - L[m]["acc"])[list(SEM)]
        out[f"p3.persona_loso.{m}.media"] = float(x.mean(axis=1).mean())
        out[f"p3.persona_loso.{m}.jer_sup"] = _hboot_media(x.to_numpy())[1]
    out["p3.persona_loso.jer_sup.min"] = min(out[f"p3.persona_loso.{m}.jer_sup"] for m in ARQ)
    out["p3.persona_loso.jer_sup.max"] = max(out[f"p3.persona_loso.{m}.jer_sup"] for m in ARQ)
    out["p3.loso.interseccion"] = len(set.intersection(*conj.values()))
    bj = pd.DataFrame({m: m3(m).index.isin(sorted(conj[m])) for m in ARQ}, index=m3("eegnet").index)
    dist = np.array([1.0])
    for p in bj.mean():
        dist = np.convolve(dist, [1 - p, p])
    out["p3.loso.esperado_k4"] = float(dist[4] * 54)
    inter = [set.intersection(*[set(L[m]["acc"][s].index[le_umbral(L[m]["acc"][s])]) for m in ARQ]) for s in SEM]
    out["p3.loso.interseccion_en_las_3"] = len(set.intersection(*inter))
    out["p3.loso.interseccion_en_alguna"] = len(set.union(*inter))
    # tab:distribucion (EEGNet LOSO)
    x = m3("eegnet")
    q = x.quantile([.25, .5, .75]).to_numpy()
    out.update({"p3.dist.q1": float(q[0]), "p3.dist.med": float(q[1]), "p3.dist.q3": float(q[2]),
                "p3.dist.mn": float(x.min()), "p3.dist.mx": float(x.max()),
                "p3.dist.n_lt": int((x < 0.60).sum()), "p3.dist.n_gt": int((x > 0.80).sum())})
    # tab:sensibilidad
    resto = [s for s in x.index if s not in resid]
    kv, k, ac = m3("eegnet", "kappa_visto"), m3("eegnet", "kappa"), m3("eegnet")
    for nom, idx in (("residuales", resid), ("resto", resto)):
        out[f"p3.sens.{nom}.acc"] = float(ac.loc[idx].mean())
        out[f"p3.sens.{nom}.dsi_kappa"] = float((kv.loc[idx].mean() - k.loc[idx].mean()) / kv.loc[idx].mean())
    # restriccion de interpretacion
    dsi = (kv - k) / kv
    out["p3.restr.sd_kv"], out["p3.restr.sd_k"] = float(kv.std()), float(k.std())
    out["p3.restr.r"] = float(np.corrcoef(dsi, k)[0, 1])
    # matriz de confusion agregada (suma sobre los 54 folds y las 3 semillas)
    d = A.loso("eegnet")
    cm = sum(np.array(ast.literal_eval(c)) for c in d.confusion_matrix)
    out["p3.conf.clase0"], out["p3.conf.clase1"] = float(cm[0, 0] / cm[0].sum()), float(cm[1, 1] / cm[1].sum())
    r0 = d.assign(r0=[ast.literal_eval(c)[0][0] / sum(ast.literal_eval(c)[0]) for c in d.confusion_matrix],
                  r1=[ast.literal_eval(c)[1][1] / sum(ast.literal_eval(c)[1]) for c in d.confusion_matrix]
                  ).groupby("test_sub")[["r0", "r1"]].mean()
    out["p3.conf.dif"] = float((r0.r0 - r0.r1).mean())
    out["p3.conf.p"] = float(wilcoxon_pareado(r0.r0, r0.r1).pvalue)
    out["p3.conf.n_clase0_mayor"] = int(((r0.r0 - r0.r1) > 0).sum())
    # +- de los sujetos vistos
    for m in ARQ:
        out[f"p3.sdv.acc.{m}"] = float(m3(m, "acc_visto").std())
        out[f"p3.sdv.kappa.{m}"] = float(m3(m, "kappa_visto").std())
    # Esas / Resto del calentamiento (n_epochs == 30)
    for prot in ("within", "loso"):
        for m in ARQ:
            dd = A.within(m) if prot == "within" else A.loso(m, "base")
            f = dd.n_epochs == 30
            out[f"p3.sel.{prot}.{m}.esas"] = float(dd.acc[f].mean())
            out[f"p3.sel.{prot}.{m}.resto"] = float(dd.acc[~f].mean())
            out[f"p3.sel.{prot}.{m}.prop"] = float(100 * f.mean())
            ps = [(dd.acc[(dd.seed == s) & f].mean(), dd.acc[(dd.seed == s) & ~f].mean()) for s in SEM]
            out[f"p3.sel.{prot}.{m}.esas_min"], out[f"p3.sel.{prot}.{m}.esas_max"] = min(a for a, _ in ps), max(a for a, _ in ps)
            out[f"p3.sel.{prot}.{m}.resto_min"], out[f"p3.sel.{prot}.{m}.resto_max"] = min(b for _, b in ps), max(b for _, b in ps)
    # diez apartados frente al resto (LOSO)
    for m in ARQ:
        g, o = m3(m).loc[apart], m3(m).drop(apart)
        out[f"p3.apart.{m}.apartados"], out[f"p3.apart.{m}.resto"] = float(g.mean()), float(o.mean())
        out[f"p3.apart.{m}.dif"] = float(o.mean() - g.mean())
        out[f"p3.apart.{m}.p"] = float(stats.mannwhitneyu(g, o, alternative="two-sided").pvalue)
    # desviacion entre sujetos de persona nueva y WBCIC-SHU (politica source, media de semillas)
    for nom, tipo in (("persona", "h44_openbmi"), ("wbcic", "h44_wbcic")):
        sds = []
        for m in ARQ:
            t = A.transferencia(tipo, m, "base")
            t = t[t.norm_destino == "source"].groupby(["sub", "seed"]).acc.mean().unstack()
            out[f"p3.sdt.{nom}.{m}"] = float(t.mean(axis=1).std())
            sds.append(out[f"p3.sdt.{nom}.{m}"])
        out[f"p3.sdt.{nom}.min"], out[f"p3.sdt.{nom}.max"] = min(sds), max(sds)
    # EEGMamba intra-sujeto frente al azar
    y26 = W["eegmamba"][2026]
    out["p3.azar.2026_redondeado"] = float(wilcoxon_pareado(y26 - 0.5).pvalue)
    y3 = W["eegmamba"][list(SEM)].mean(axis=1)
    out["p3.azar.media3"] = float(wilcoxon_pareado(y3 - 0.5).pvalue)
    out["p3.azar.media3_sin_redondear"] = float(stats.wilcoxon(y3 - 0.5).pvalue)
    # fig:solapamiento (intra-sujeto, media de tres semillas) y fig:resultados-baseline
    bajo = pd.DataFrame({m: le_umbral(W[m][list(SEM)].mean(axis=1)) for m in ARQ})
    kk = bajo.sum(axis=1)
    dist = np.array([1.0])
    for p in bajo.mean():
        dist = np.convolve(dist, [1 - p, p])
    for i in range(5):
        out[f"p3.sol.obs{i}"], out[f"p3.sol.esp{i}"] = int((kk == i).sum()), float(dist[i] * 54)
    for m in ARQ:
        out[f"p3.sol.n_bajo.{m}"] = int(bajo[m].sum())
    out["p3.sol.interseccion"] = int(bajo.all(axis=1).sum())
    lo3, wi3 = m3("eegnet").round(9), W["eegnet"][list(SEM)].mean(axis=1).round(9)
    out["p3.fig_base.mediana"] = float(lo3.median())
    out["p3.fig_base.n_sobre_diag"] = int((lo3 > wi3).sum())
    return out


BLOQUES["p3"] = b_p3

# Declarados del bloque p3 (copiados de results/avance_7/analisis/tres_semillas_p3_cifras.csv)
# id -> (valor, tolerancia, informe). Tolerancia: 5e-5 (cifras con 4 decimales), 0.5 (recuentos),
# 0.5 % del valor (p muy pequeños) y 0.005 (porcentajes del calentamiento).
DECL_P3 = {
    "p3.p5.eegnet": (0.5620833333333333, 5e-05, "P3 tres semillas"),
    "p3.p5.conformer": (0.5553750000000001, 5e-05, "P3 tres semillas"),
    "p3.p5.labram": (0.5805, 5e-05, "P3 tres semillas"),
    "p3.p5.eegmamba": (0.5522916666666666, 5e-05, "P3 tres semillas"),
    "p3.ic.loso.eegnet.media": (0.760570987654321, 5e-05, "P3 tres semillas"),
    "p3.ic.loso.eegnet.inf": (0.7282218364197529, 5e-05, "P3 tres semillas"),
    "p3.ic.loso.eegnet.sup": (0.7919297839506173, 5e-05, "P3 tres semillas"),
    "p3.ic.loso.conformer.media": (0.746851851851852, 5e-05, "P3 tres semillas"),
    "p3.ic.loso.conformer.inf": (0.7139351851851852, 5e-05, "P3 tres semillas"),
    "p3.ic.loso.conformer.sup": (0.780108024691358, 5e-05, "P3 tres semillas"),
    "p3.ic.loso.labram.media": (0.754891975308642, 5e-05, "P3 tres semillas"),
    "p3.ic.loso.labram.inf": (0.7258942901234569, 5e-05, "P3 tres semillas"),
    "p3.ic.loso.labram.sup": (0.783025848765432, 5e-05, "P3 tres semillas"),
    "p3.ic.loso.eegmamba.media": (0.7153395061728394, 5e-05, "P3 tres semillas"),
    "p3.ic.loso.eegmamba.inf": (0.6878703703703705, 5e-05, "P3 tres semillas"),
    "p3.ic.loso.eegmamba.sup": (0.7430104166666667, 5e-05, "P3 tres semillas"),
    "p3.ic.intra.eegnet.media": (0.6342746913580245, 5e-05, "P3 tres semillas"),
    "p3.ic.intra.eegnet.inf": (0.606249614197531, 5e-05, "P3 tres semillas"),
    "p3.ic.intra.eegnet.sup": (0.6637966820987656, 5e-05, "P3 tres semillas"),
    "p3.ic.intra.conformer.media": (0.6024691358024694, 5e-05, "P3 tres semillas"),
    "p3.ic.intra.conformer.inf": (0.5724992283950616, 5e-05, "P3 tres semillas"),
    "p3.ic.intra.conformer.sup": (0.6346149691358024, 5e-05, "P3 tres semillas"),
    "p3.ic.intra.labram.media": (0.6633179012345678, 5e-05, "P3 tres semillas"),
    "p3.ic.intra.labram.inf": (0.623564814814815, 5e-05, "P3 tres semillas"),
    "p3.ic.intra.labram.sup": (0.7058487654320987, 5e-05, "P3 tres semillas"),
    "p3.ic.intra.eegmamba.media": (0.570787037037037, 5e-05, "P3 tres semillas"),
    "p3.ic.intra.eegmamba.inf": (0.5472064043209878, 5e-05, "P3 tres semillas"),
    "p3.ic.intra.eegmamba.sup": (0.5976855709876543, 5e-05, "P3 tres semillas"),
    "p3.loso.interseccion": (13.0, 0.5, "P3 tres semillas"),
    "p3.loso.n_bajo.eegnet": (17.0, 0.5, "P3 tres semillas"),
    "p3.loso.n_bajo.conformer": (19.0, 0.5, "P3 tres semillas"),
    "p3.loso.n_bajo.labram": (15.0, 0.5, "P3 tres semillas"),
    "p3.loso.n_bajo.eegmamba": (26.0, 0.5, "P3 tres semillas"),
    "p3.loso.esperado_k4": (0.7999923792104862, 5e-05, "P3 tres semillas"),
    "p3.loso.interseccion_en_las_3": (9.0, 0.5, "P3 tres semillas"),
    "p3.loso.interseccion_en_alguna": (15.0, 0.5, "P3 tres semillas"),
    "p3.dist.q1": (0.6672916666666666, 5e-05, "P3 tres semillas"),
    "p3.dist.med": (0.7641666666666667, 5e-05, "P3 tres semillas"),
    "p3.dist.q3": (0.8322916666666667, 5e-05, "P3 tres semillas"),
    "p3.dist.mn": (0.5433333333333333, 5e-05, "P3 tres semillas"),
    "p3.dist.mx": (0.9891666666666666, 5e-05, "P3 tres semillas"),
    "p3.dist.n_lt": (6.0, 0.5, "P3 tres semillas"),
    "p3.dist.n_gt": (20.0, 0.5, "P3 tres semillas"),
    "p3.sens.residuales.acc": (0.8406250000000001, 5e-05, "P3 tres semillas"),
    "p3.sens.residuales.dsi_kappa": (-0.1895149786019972, 5e-05, "P3 tres semillas"),
    "p3.sens.resto.acc": (0.7541666666666665, 5e-05, "P3 tres semillas"),
    "p3.sens.resto.dsi_kappa": (0.0925055132259081, 5e-05, "P3 tres semillas"),
    "p3.conf.clase0": (0.777962962962963, 5e-05, "P3 tres semillas"),
    "p3.conf.clase1": (0.743179012345679, 5e-05, "P3 tres semillas"),
    "p3.conf.dif": (0.0347839506172839, 5e-05, "P3 tres semillas"),
    "p3.conf.p": (0.4909261065873447, 5e-05, "P3 tres semillas"),
    "p3.conf.n_clase0_mayor": (29.0, 0.5, "P3 tres semillas"),
    "p3.restr.sd_kv": (0.0285397983650671, 5e-05, "P3 tres semillas"),
    "p3.restr.sd_k": (0.2373755492262167, 5e-05, "P3 tres semillas"),
    "p3.restr.r": (-0.9925638064066102, 5e-05, "P3 tres semillas"),
    "p3.sdv.acc.eegnet": (0.0142698991825335, 5e-05, "P3 tres semillas"),
    "p3.sdv.kappa.eegnet": (0.0285397983650671, 5e-05, "P3 tres semillas"),
    "p3.sdv.acc.conformer": (0.0322047491123449, 5e-05, "P3 tres semillas"),
    "p3.sdv.kappa.conformer": (0.0644094982246899, 5e-05, "P3 tres semillas"),
    "p3.sdv.acc.labram": (0.0098739546605026, 5e-05, "P3 tres semillas"),
    "p3.sdv.kappa.labram": (0.0197479093210052, 5e-05, "P3 tres semillas"),
    "p3.sdv.acc.eegmamba": (0.0093136663942782, 5e-05, "P3 tres semillas"),
    "p3.sdv.kappa.eegmamba": (0.0186273327885564, 5e-05, "P3 tres semillas"),
    "p3.sel.within.eegnet.esas": (0.5111336032388664, 5e-05, "P3 tres semillas"),
    "p3.sel.within.eegnet.resto": (0.6564275309541151, 5e-05, "P3 tres semillas"),
    "p3.sel.within.eegnet.prop": (15.246913580246911, 0.005, "P3 tres semillas"),
    "p3.sel.within.conformer.esas": (0.5205361930294906, 5e-05, "P3 tres semillas"),
    "p3.sel.within.conformer.resto": (0.7135999999999999, 5e-05, "P3 tres semillas"),
    "p3.sel.within.conformer.prop": (57.56172839506173, 0.005, "P3 tres semillas"),
    "p3.sel.within.labram.esas": (0.5210625, 5e-05, "P3 tres semillas"),
    "p3.sel.within.labram.resto": (0.7099590163934426, 5e-05, "P3 tres semillas"),
    "p3.sel.within.labram.prop": (24.69135802469136, 0.005, "P3 tres semillas"),
    "p3.sel.within.eegmamba.esas": (0.5410625444207534, 5e-05, "P3 tres semillas"),
    "p3.sel.within.eegmamba.resto": (0.7671361502347418, 5e-05, "P3 tres semillas"),
    "p3.sel.within.eegmamba.prop": (86.85185185185185, 0.005, "P3 tres semillas"),
    "p3.sel.loso.eegnet.esas": (0.724, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.eegnet.resto": (0.7629769736842105, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.eegnet.prop": (6.172839506172839, 0.005, "P3 tres semillas"),
    "p3.sel.loso.conformer.esas": (0.6852419354838708, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.conformer.resto": (0.7614312977099237, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.conformer.prop": (19.1358024691358, 0.005, "P3 tres semillas"),
    "p3.sel.loso.labram.esas": (0.7072777777777778, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.labram.resto": (0.7732051282051282, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.labram.prop": (27.77777777777778, 0.005, "P3 tres semillas"),
    "p3.sel.loso.eegmamba.esas": (0.7029166666666666, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.eegmamba.resto": (0.7252777777777777, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.eegmamba.prop": (44.44444444444444, 0.005, "P3 tres semillas"),
    "p3.sel.within.eegnet.esas_min": (0.5013812154696132, 5e-05, "P3 tres semillas"),
    "p3.sel.within.eegnet.esas_max": (0.520949720670391, 5e-05, "P3 tres semillas"),
    "p3.sel.within.eegnet.resto_min": (0.6554384017758046, 5e-05, "P3 tres semillas"),
    "p3.sel.within.eegnet.resto_max": (0.6570824524312897, 5e-05, "P3 tres semillas"),
    "p3.sel.within.conformer.esas_min": (0.5156682027649769, 5e-05, "P3 tres semillas"),
    "p3.sel.within.conformer.esas_max": (0.5256968641114983, 5e-05, "P3 tres semillas"),
    "p3.sel.within.conformer.resto_min": (0.6990118577075098, 5e-05, "P3 tres semillas"),
    "p3.sel.within.conformer.resto_max": (0.7293706293706294, 5e-05, "P3 tres semillas"),
    "p3.sel.within.labram.esas_min": (0.5126865671641792, 5e-05, "P3 tres semillas"),
    "p3.sel.within.labram.esas_max": (0.5291666666666667, 5e-05, "P3 tres semillas"),
    "p3.sel.within.labram.resto_min": (0.7009976525821596, 5e-05, "P3 tres semillas"),
    "p3.sel.within.labram.resto_max": (0.7183634020618558, 5e-05, "P3 tres semillas"),
    "p3.sel.within.eegmamba.esas_min": (0.5356223175965665, 5e-05, "P3 tres semillas"),
    "p3.sel.within.eegmamba.esas_max": (0.5442006269592476, 5e-05, "P3 tres semillas"),
    "p3.sel.within.eegmamba.resto_min": (0.7462837837837838, 5e-05, "P3 tres semillas"),
    "p3.sel.within.eegmamba.resto_max": (0.779268292682927, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.eegnet.esas_min": (0.620625, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.eegnet.esas_max": (0.8275, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.eegnet.resto_min": (0.7573500000000002, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.eegnet.resto_max": (0.77065, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.conformer.esas_min": (0.62775, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.conformer.esas_max": (0.7264285714285714, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.conformer.resto_min": (0.7505000000000001, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.conformer.resto_max": (0.771193181818182, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.labram.esas_min": (0.7065277777777779, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.labram.esas_max": (0.7080357142857144, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.labram.resto_min": (0.7698124999999999, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.labram.resto_max": (0.7764583333333333, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.eegmamba.esas_min": (0.6940999999999999, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.eegmamba.esas_max": (0.7121999999999999, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.eegmamba.resto_min": (0.7207758620689656, 5e-05, "P3 tres semillas"),
    "p3.sel.loso.eegmamba.resto_max": (0.7308620689655172, 5e-05, "P3 tres semillas"),
    "p3.apart.eegnet.apartados": (0.74225, 5e-05, "P3 tres semillas"),
    "p3.apart.eegnet.resto": (0.7647348484848483, 5e-05, "P3 tres semillas"),
    "p3.apart.eegnet.dif": (0.0224848484848483, 5e-05, "P3 tres semillas"),
    "p3.apart.eegnet.p": (0.5551131549100907, 5e-05, "P3 tres semillas"),
    "p3.apart.conformer.apartados": (0.7212500000000001, 5e-05, "P3 tres semillas"),
    "p3.apart.conformer.resto": (0.7526704545454546, 5e-05, "P3 tres semillas"),
    "p3.apart.conformer.dif": (0.0314204545454545, 5e-05, "P3 tres semillas"),
    "p3.apart.conformer.p": (0.3439109968142269, 5e-05, "P3 tres semillas"),
    "p3.apart.labram.apartados": (0.7349166666666667, 5e-05, "P3 tres semillas"),
    "p3.apart.labram.resto": (0.759431818181818, 5e-05, "P3 tres semillas"),
    "p3.apart.labram.dif": (0.0245151515151513, 5e-05, "P3 tres semillas"),
    "p3.apart.labram.p": (0.5777164025229793, 5e-05, "P3 tres semillas"),
    "p3.apart.eegmamba.apartados": (0.6941666666666666, 5e-05, "P3 tres semillas"),
    "p3.apart.eegmamba.resto": (0.7201515151515151, 5e-05, "P3 tres semillas"),
    "p3.apart.eegmamba.dif": (0.0259848484848485, 5e-05, "P3 tres semillas"),
    "p3.apart.eegmamba.p": (0.3851178745087243, 5e-05, "P3 tres semillas"),
    "p3.sdt.persona.eegnet": (0.1219020110935089, 5e-05, "P3 tres semillas"),
    "p3.sdt.persona.conformer": (0.1290000717771836, 5e-05, "P3 tres semillas"),
    "p3.sdt.persona.labram": (0.108388483113313, 5e-05, "P3 tres semillas"),
    "p3.sdt.persona.eegmamba": (0.0953488154354225, 5e-05, "P3 tres semillas"),
    "p3.sdt.persona.min": (0.0953488154354225, 5e-05, "P3 tres semillas"),
    "p3.sdt.persona.max": (0.1290000717771836, 5e-05, "P3 tres semillas"),
    "p3.sdt.wbcic.eegnet": (0.1179137364240179, 5e-05, "P3 tres semillas"),
    "p3.sdt.wbcic.conformer": (0.1234988290807395, 5e-05, "P3 tres semillas"),
    "p3.sdt.wbcic.labram": (0.1239764191507668, 5e-05, "P3 tres semillas"),
    "p3.sdt.wbcic.eegmamba": (0.0961956068188242, 5e-05, "P3 tres semillas"),
    "p3.sdt.wbcic.min": (0.0961956068188242, 5e-05, "P3 tres semillas"),
    "p3.sdt.wbcic.max": (0.1239764191507668, 5e-05, "P3 tres semillas"),
    "p3.persona_loso.eegnet.media": (0.019969438392641, 5e-05, "P3 tres semillas"),
    "p3.persona_loso.eegnet.jer_sup": (0.0531486095981602, 5e-05, "P3 tres semillas"),
    "p3.persona_loso.conformer.media": (0.0283206245461147, 5e-05, "P3 tres semillas"),
    "p3.persona_loso.conformer.jer_sup": (0.0610460844831759, 5e-05, "P3 tres semillas"),
    "p3.persona_loso.labram.media": (0.0414012950859356, 5e-05, "P3 tres semillas"),
    "p3.persona_loso.labram.jer_sup": (0.0704565556160735, 5e-05, "P3 tres semillas"),
    "p3.persona_loso.eegmamba.media": (0.0265147664003873, 5e-05, "P3 tres semillas"),
    "p3.persona_loso.eegmamba.jer_sup": (0.0546059670781893, 5e-05, "P3 tres semillas"),
    "p3.persona_loso.jer_sup.min": (0.0531486095981602, 5e-05, "P3 tres semillas"),
    "p3.persona_loso.jer_sup.max": (0.0704565556160735, 5e-05, "P3 tres semillas"),
    "p3.azar.2026_redondeado": (1.6554105972853872e-08, 8.277052986426937e-11, "P3 tres semillas"),
    "p3.azar.media3": (1.175975818692556e-09, 5.87987909346278e-12, "P3 tres semillas"),
    "p3.azar.media3_sin_redondear": (1.1761827039651087e-09, 5.880913519825543e-12, "P3 tres semillas"),
    "p3.sol.obs0": (5.0, 0.5, "P3 tres semillas"),
    "p3.sol.esp0": (0.0631572930955646, 5e-05, "P3 tres semillas"),
    "p3.sol.obs1": (2.0, 0.5, "P3 tres semillas"),
    "p3.sol.esp1": (1.2713763145861905, 5e-05, "P3 tres semillas"),
    "p3.sol.obs2": (5.0, 0.5, "P3 tres semillas"),
    "p3.sol.esp2": (8.51063100137174, 5e-05, "P3 tres semillas"),
    "p3.sol.obs3": (8.0, 0.5, "P3 tres semillas"),
    "p3.sol.esp3": (22.91197988111568, 5e-05, "P3 tres semillas"),
    "p3.sol.obs4": (34.0, 0.5, "P3 tres semillas"),
    "p3.sol.esp4": (21.24285550983082, 5e-05, "P3 tres semillas"),
    "p3.sol.n_bajo.eegnet": (41.0, 0.5, "P3 tres semillas"),
    "p3.sol.n_bajo.conformer": (45.0, 0.5, "P3 tres semillas"),
    "p3.sol.n_bajo.labram": (37.0, 0.5, "P3 tres semillas"),
    "p3.sol.n_bajo.eegmamba": (49.0, 0.5, "P3 tres semillas"),
    "p3.sol.interseccion": (34.0, 0.5, "P3 tres semillas"),
    "p3.fig_base.mediana": (0.764166667, 5e-05, "P3 tres semillas"),
    "p3.fig_base.n_sobre_diag": (46.0, 0.5, "P3 tres semillas"),
}
DECLARADOS.update(DECL_P3)



def construir_cifras():
    """Lista de 4-tuplas compatibles con verificar_cifras.CIFRAS: (id, declarado, donde, fn).

    La tolerancia propia se consulta con TOL[id]; verificar_cifras.tolerancia() no sirve aqui
    (los p en notacion cientifica y los porcentajes de una cifra requieren la tolerancia
    guardada con cada valor)."""
    cifras = []
    for ident, (valor, tol, informe) in DECLARADOS.items():
        bloque = ident.split(".")[0]
        fn_bloque = BLOQUES.get(bloque)
        def fn(ident=ident, fn_bloque=fn_bloque):
            if fn_bloque is None:
                raise KeyError(ident)
            return float(valores(fn_bloque)[ident])
        cifras.append((ident, valor, DONDE, fn))
    return cifras


TOL = {k: v[1] for k, v in DECLARADOS.items()}
INFORME = {k: v[2] for k, v in DECLARADOS.items()}


def ejecutar(verboso: bool = False, bloque: str | None = None) -> int:
    fallos, sin_calculo, n_ok, n = [], [], 0, 0
    for ident, declarado, donde, fn in construir_cifras():
        if bloque and not ident.startswith(bloque + "."):
            continue
        n += 1
        try:
            obtenido = fn()
        except KeyError:
            sin_calculo.append(ident)
            continue
        ok = abs(obtenido - declarado) <= TOL[ident] + 1e-12
        n_ok += ok
        if not ok:
            fallos.append((ident, declarado, obtenido, INFORME[ident]))
        if verboso or not ok:
            marca = "ok      " if ok else "DISCREPA"
            print(f"{marca} {ident:52s} declarado {declarado:>11.6g} obtenido {obtenido:>12.6g}  "
                  f"[{INFORME[ident]}; {donde}]")
    print(f"\n{n} cifras del avance 7, {n_ok} ok, {len(fallos)} discrepan, "
          f"{len(sin_calculo)} sin recalculo (origen: {ORIGEN})")
    for i in sin_calculo[:50]:
        print(f"  sin recalculo  {i}")
    if len(sin_calculo) > 50:
        print(f"  ... y {len(sin_calculo) - 50} mas")
    return 1 if fallos else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verboso", action="store_true")
    ap.add_argument("--bloque", default=None, help="solo identificadores con este prefijo")
    a = ap.parse_args()
    return ejecutar(a.verboso, a.bloque)


if __name__ == "__main__":
    raise SystemExit(main())
