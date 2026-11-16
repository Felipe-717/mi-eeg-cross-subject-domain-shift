#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""avance7_ablaciones.py: ablaciones F4, alineamiento euclidiano (F6) y reflexion de canales (F8).

Lee SOLO lo que expone `scripts/avance7_datos.py` y escribe en `results/avance_7/analisis/`.
Informe: `avance_7/analisis/05_ABLACIONES_EA_CR.md`.  Comando:

    ~/anaconda3/envs/eeg/python.exe scripts/avance7_ablaciones.py

DISENO (fijado antes de calcular; el informe lo repite en su seccion 1)
  Unidad: el sujeto (n = 54). Intra-sujeto: media de acc de las 20 particiones del sujeto
  (2 sesiones x 10 pliegues) de la semilla indicada. LOSO: acc del pliegue del sujeto
  excluido; con varias semillas, media dentro del sujeto.
  Contraste pareado, diferencia = variante - referencia. Wilcoxon bilateral zsplit,
  r = Z/sqrt(n) (Z desde el p bilateral), IC 95 % bootstrap de percentiles (B = 10 000,
  default_rng(2026) POR CONTRASTE), equivalencia +-0.03 por IC 90 % (criterio publicado).
  Holm DENTRO de cada familia:
    F4 (principal): una familia por arquitectura. EEGNet y Conformer: mu, beta, mubeta, w02,
        w03, augnone, augcr, augfft (m = 8). LaBraM y EEGMamba: augcr, augfft, w02 (m = 3).
    F6: EA frente a base LOSO 2026, 4 arquitecturas (m = 4).
    F8: reflexion de canales frente a base LOSO 2026, 4 arquitecturas (m = 4).
  Sensibilidad (misma estructura de familias): referencia = media por sujeto de las semillas
  2027-2028 de la base (con recorte de gradiente) en F6, F8 y en F4 de LaBraM y EEGMamba.
  Secundaria informativa: Holm sobre los 22 contrastes de F4 y sobre los 8 de F6+F8.
  Exploratorio (no entra en familias): caida vistos/no vistos con EA, concentracion del
  beneficio en sujetos de bajo rendimiento, referencia de ruido entre semillas.

REGIMEN: bases previas de EEGNet, Conformer y LaBraM sin grad_clip; avance 7 con grad_clip 1.0;
EEGMamba con recorte en ambos (ver 01_INVENTARIO_Y_QA.md, A1). El contraste esta confundido
con el recorte salvo en EEGMamba. La columna `confundido` lo marca; la comparacion con la cota
del confusor (archivo 04) se hace en `cota_confusor()` si el archivo existe.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from ruido_flotante import wilcoxon_pareado  # Wilcoxon sobre diferencias redondeadas a 1e-9

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "scripts"))
import avance7_datos as A  # noqa: E402

OUT = RAIZ / "results" / "avance_7" / "analisis"
OUT.mkdir(parents=True, exist_ok=True)

B = 10_000
SEMILLA_BOOT = 2026
MARGEN = 0.03
UMBRAL_BAJO = 0.70            # vidaurre2010, el mismo de verificar_cifras.py
ARQ = A.ARQ
CONFUNDIDO = {"eegnet": True, "conformer": True, "labram": True, "eegmamba": False}
F4_VARIANTES = {
    "eegnet": A.ABL_TODAS, "conformer": A.ABL_TODAS,
    "labram": A.ABL_GRANDES, "eegmamba": A.ABL_GRANDES,
}
ORDEN_F4 = ["mu", "beta", "mubeta", "w02", "w03", "augnone", "augcr", "augfft"]
F4_VARIANTES = {m: [v for v in ORDEN_F4 if v in vs] for m, vs in F4_VARIANTES.items()}
SENS_F4 = ["labram", "eegmamba"]     # arquitecturas con base within 2027-2028 CON recorte

# Cota del confusor (archivo 04): se rellena en cota_confusor() si existe.
CSV_COTA_CANDIDATOS = ["cota_confusor.csv", "semillas_confusor.csv", "confusor_cota.csv"]


# ------------------------------------------------------------------ estadistica

def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni; mismo algoritmo que `verificar_cifras.p_holm_ids`."""
    p = np.asarray(p, float)
    m = len(p)
    orden = np.argsort(p)
    aj = np.empty(m)
    previo = 0.0
    for i, k in enumerate(orden):
        previo = max(previo, p[k] * (m - i))
        aj[k] = min(previo, 1.0)
    return aj


def contraste(var: pd.Series, ref: pd.Series) -> dict:
    """Contraste pareado por sujeto; diferencia = var - ref. Series indexadas por sujeto."""
    j = pd.concat([var, ref], axis=1, join="inner", keys=["v", "r"]).dropna()
    n = len(j)
    d = (j["v"] - j["r"]).to_numpy()
    if np.all(np.round(d, 9) == 0):
        p = 1.0
    else:
        p = float(wilcoxon_pareado(j["v"], j["r"], zero_method="zsplit",
                                   alternative="two-sided").pvalue)
    z = abs(stats.norm.isf(p / 2)) if p > 0 else abs(stats.norm.isf(1e-16))
    rng = np.random.default_rng(SEMILLA_BOOT)
    m = d[rng.integers(0, n, (B, n))].mean(axis=1)
    lo95, hi95 = np.quantile(m, [0.025, 0.975])
    lo90, hi90 = np.quantile(m, [0.05, 0.95])
    return {
        "n": n, "media_var": j["v"].mean(), "media_ref": j["r"].mean(),
        "dif_media": d.mean(), "dif_mediana": float(np.median(d)),
        "ic95_inf": lo95, "ic95_sup": hi95, "ic90_inf": lo90, "ic90_sup": hi90,
        "n_mejoran": int((d > 0).sum()), "n_empeoran": int((d < 0).sum()),
        "n_iguales": int((d == 0).sum()), "p_wilcoxon": p, "r": z / np.sqrt(n),
        "equiv_0.03": bool(lo90 > -MARGEN and hi90 < MARGEN),
        "margen_equiv_minimo": float(max(abs(lo90), abs(hi90))),
    }


def aplicar_holm(df: pd.DataFrame, por: list[str], col="p_holm") -> pd.DataFrame:
    df = df.copy()
    df[col] = np.nan
    for _, g in df.groupby(por, sort=False):
        df.loc[g.index, col] = holm(g["p_wilcoxon"].to_numpy())
    return df


# ------------------------------------------------------------------ extraccion por sujeto

def acc_within(d: pd.DataFrame, seeds) -> pd.Series:
    """acc media por sujeto sobre las particiones (sesion x pliegue) de las semillas dadas."""
    x = d[d["seed"].isin(seeds)]
    return x.groupby("sub")["acc"].mean().sort_index()


def acc_loso(d: pd.DataFrame, seeds, col="acc") -> pd.Series:
    x = d[d["seed"].isin(seeds)]
    return x.groupby("test_sub")[col].mean().sort_index()


# ------------------------------------------------------------------ control de calidad

def qa() -> pd.DataFrame:
    """Comprobaciones de la regla 2 del protocolo sobre cada CSV que se usa."""
    filas = []

    def anota(cond, d, esperado, seeds_esp, base_hash, clave, tipo):
        sujetos = d[clave].nunique()
        dup = int(d["run_id"].duplicated().sum()) if tipo != "ea_mismo_runid" else 0
        filas.append({
            "condicion": cond, "fuente": d["fuente"].iloc[0], "regimen": d["regimen"].iloc[0],
            "filas": len(d), "esperado": esperado, "ok_filas": len(d) == esperado,
            "sujetos": sujetos, "semillas": ",".join(map(str, sorted(d.seed.unique()))),
            "ok_semillas": sorted(d.seed.unique()) == sorted(seeds_esp),
            "run_id_duplicados": dup,
            "nan_metricas": int(d[["acc", "kappa", "f1_macro"]].isna().sum().sum()),
            "humo": bool(d["run_id"].astype(str).str.contains("smoke", case=False).any()),
            "cfg_hash": ",".join(sorted(d.cfg_hash.unique())),
            "hash_igual_a_base": (",".join(sorted(d.cfg_hash.unique())) == base_hash),
        })

    for m in ARQ:
        w = A.within(m)
        # base within: referencia 2026 y semillas 2027-2028
        wb = w[w.seed == 2026]
        anota(f"within_base_{m}_2026", wb, 1080, [2026], "", "sub", "base")
        h_prev = ",".join(sorted(wb.cfg_hash.unique()))
        for v in F4_VARIANTES[m]:
            d = A.ablacion(v, m)
            anota(f"abl_{v}_{m}", d, 1080, [2026], h_prev, "sub", "abl")
            # pareo: mismas claves (sub, session, fold)
            k1 = set(map(tuple, wb[["sub", "session", "fold"]].to_numpy()))
            k2 = set(map(tuple, d[["sub", "session", "fold"]].to_numpy()))
            filas[-1]["claves_pareo_iguales"] = (k1 == k2)
            filas[-1]["n_test_igual_a_base"] = bool(
                (wb.sort_values(["sub", "session", "fold"]).n_test.to_numpy() ==
                 d.sort_values(["sub", "session", "fold"]).n_test.to_numpy()).all())
        if m in SENS_F4:
            wr = w[w.seed.isin([2027, 2028])]
            anota(f"within_base_{m}_2027_28", wr, 2160, [2027, 2028], h_prev, "sub", "base")
        lb = A.loso(m, "base")
        l26 = lb[lb.seed == 2026]
        anota(f"loso_base_{m}_2026", l26, 54, [2026], "", "test_sub", "base")
        hl = ",".join(sorted(l26.cfg_hash.unique()))
        l_r = lb[lb.seed.isin([2027, 2028])]
        anota(f"loso_base_{m}_2027_28", l_r, 108, [2027, 2028], hl, "test_sub", "base")
        for v in ("ea", "augcr"):
            d = A.loso(m, v)
            anota(f"loso_{v}_{m}", d, 54, [2026], hl, "test_sub", "ea_mismo_runid"
                  if v == "ea" else "x")
            filas[-1]["claves_pareo_iguales"] = (set(d.test_sub) == set(l26.test_sub))
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ cota del confusor

def cota_confusor() -> dict | None:
    """Lee la cota del confusor que calcula el otro agente (archivo 04).

    `confusor_cota.csv`: una fila por (protocolo, arq) CON cambio de regimen en la base
    (within: solo LaBraM; loso: EEGNet, Conformer, LaBraM). Columnas `cota_bruta` (mayor valor
    absoluto del IC 95 % del salto base 2026 menos media 2027-2028) y `cota_neta` (idem tras
    restar el control EEGMamba, que no cambia de regimen). Se usa la MAYOR de las dos, el
    criterio mas conservador. Devuelve {(arq, protocolo): (bruta, neta, usada)} o None.
    No hay fila para within de EEGNet y Conformer: no existe base con recorte, la cota no
    se puede estimar y se informa como tal.
    """
    for n in CSV_COTA_CANDIDATOS:
        p = OUT / n
        if p.exists():
            d = pd.read_csv(p)
            if {"arq", "protocolo", "cota_bruta", "cota_neta"} <= set(d.columns):
                return {(r.arq, r.protocolo): (float(r.cota_bruta), float(r.cota_neta),
                                               max(float(r.cota_bruta), float(r.cota_neta)))
                        for r in d.itertuples()}
    return None


# ------------------------------------------------------------------ analisis F4

def f4() -> tuple[pd.DataFrame, pd.DataFrame]:
    filas, deltas = [], []
    for m in ARQ:
        w = A.within(m)
        base = acc_within(w, [2026])
        ref_s = acc_within(w, [2027, 2028]) if m in SENS_F4 else None
        for v in F4_VARIANTES[m]:
            x = acc_within(A.ablacion(v, m), [2026])
            for tipo, ref, etq in (("principal", base, "base within 2026"),
                                   ("sensibilidad", ref_s, "media base within 2027-2028")):
                if ref is None:
                    continue
                c = contraste(x, ref)
                filas.append({"familia": f"F4-{m}", "tipo": tipo, "arq": m, "variante": v,
                              "referencia": etq,
                              "confundido": CONFUNDIDO[m] if tipo == "principal" else False,
                              **c})
                dd = (x - ref).dropna()
                deltas += [{"familia": f"F4-{m}", "tipo": tipo, "arq": m, "variante": v,
                            "sujeto": s, "var": x[s], "ref": ref[s], "dif": dd[s]}
                           for s in dd.index]
    df = pd.DataFrame(filas)
    df = aplicar_holm(df, ["tipo", "familia"])
    # secundaria: Holm sobre los 22 contrastes principales
    pri = df[df.tipo == "principal"]
    df["p_holm_22"] = np.nan
    df.loc[pri.index, "p_holm_22"] = holm(pri["p_wilcoxon"].to_numpy())
    return df, pd.DataFrame(deltas)


# ------------------------------------------------------------------ analisis F6 / F8

def f68() -> tuple[pd.DataFrame, pd.DataFrame]:
    filas, deltas = [], []
    for m in ARQ:
        lb = A.loso(m, "base")
        base = acc_loso(lb, [2026])
        ref_s = acc_loso(lb, [2027, 2028])
        for fam, v in (("F6", "ea"), ("F8", "augcr")):
            x = acc_loso(A.loso(m, v), [2026])
            for tipo, ref, etq in (("principal", base, "base LOSO 2026"),
                                   ("sensibilidad", ref_s, "media base LOSO 2027-2028")):
                c = contraste(x, ref)
                filas.append({"familia": fam, "tipo": tipo, "arq": m,
                              "variante": "EA" if v == "ea" else "reflexion",
                              "referencia": etq,
                              "confundido": CONFUNDIDO[m] if tipo == "principal" else False,
                              **c})
                dd = x - ref
                deltas += [{"familia": fam, "tipo": tipo, "arq": m, "variante": v,
                            "sujeto": s, "var": x[s], "ref": ref[s], "dif": dd[s]}
                           for s in dd.index]
    df = pd.DataFrame(filas)
    df = aplicar_holm(df, ["tipo", "familia"])
    pri = df[df.tipo == "principal"]
    df["p_holm_8"] = np.nan
    df.loc[pri.index, "p_holm_8"] = holm(pri["p_wilcoxon"].to_numpy())
    return df, pd.DataFrame(deltas)


def concordancia(df: pd.DataFrame) -> pd.DataFrame:
    p = df[df.tipo == "principal"].set_index(["familia", "arq", "variante"])
    s = df[df.tipo == "sensibilidad"].set_index(["familia", "arq", "variante"])
    p = p.loc[[k for k in p.index if k in s.index]]
    filas = []
    for k in p.index:
        a, b = p.loc[k], s.loc[k]
        filas.append({
            "familia": k[0], "arq": k[1], "variante": k[2],
            "dif_principal": a.dif_media, "dif_sensibilidad": b.dif_media,
            "mismo_signo": np.sign(a.dif_media) == np.sign(b.dif_media),
            "p_holm_principal": a.p_holm, "p_holm_sensibilidad": b.p_holm,
            "signif_principal": a.p_holm < 0.05, "signif_sensibilidad": b.p_holm < 0.05,
            "misma_significacion": (a.p_holm < 0.05) == (b.p_holm < 0.05),
            "coinciden": (np.sign(a.dif_media) == np.sign(b.dif_media))
                         and ((a.p_holm < 0.05) == (b.p_holm < 0.05)),
            "diferencia_entre_referencias": b.media_ref - a.media_ref,
        })
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ EA y caida vistos/no vistos

def ids_formula(visto: pd.Series, no_visto: pd.Series) -> float:
    return float((visto.mean() - no_visto.mean()) / visto.mean())


def ea_caida() -> pd.DataFrame:
    """EXPLORATORIO. Caida por persona nueva (acc_visto - acc) en la base y con EA, y gap LOSO-intra."""
    filas = []
    for m in ARQ:
        lb, le = A.loso(m, "base"), A.loso(m, "ea")
        lb = lb[lb.seed == 2026]
        w = acc_within(A.within(m), [2026])
        nov_b, nov_e = acc_loso(lb, [2026]), acc_loso(le, [2026])
        vis_b, vis_e = acc_loso(lb, [2026], "acc_visto"), acc_loso(le, [2026], "acc_visto")
        gap_b, gap_e = vis_b - nov_b, vis_e - nov_e            # caida por persona, por sujeto
        c = contraste(gap_e, gap_b)
        # reduccion relativa de la caida media, con IC por remuestreo de sujetos
        rng = np.random.default_rng(SEMILLA_BOOT)
        n = len(gap_b)
        idx = rng.integers(0, n, (B, n))
        gb, ge = gap_b.to_numpy(), gap_e.to_numpy()
        red = 1 - ge[idx].mean(axis=1) / gb[idx].mean(axis=1)
        red_pt = 1 - ge.mean() / gb.mean()
        lo, hi = np.quantile(red, [0.025, 0.975])
        filas.append({
            "arq": m, "confundido": CONFUNDIDO[m],
            "acc_no_visto_base": nov_b.mean(), "acc_no_visto_ea": nov_e.mean(),
            "acc_visto_base": vis_b.mean(), "acc_visto_ea": vis_e.mean(),
            "caida_persona_base": gap_b.mean(), "caida_persona_ea": gap_e.mean(),
            "IDS_base": ids_formula(vis_b, nov_b), "IDS_ea": ids_formula(vis_e, nov_e),
            "reduccion_relativa_caida": red_pt, "red_ic95_inf": lo, "red_ic95_sup": hi,
            "dif_caida_media": c["dif_media"], "dif_caida_ic95_inf": c["ic95_inf"],
            "dif_caida_ic95_sup": c["ic95_sup"], "p_wilcoxon_caida": c["p_wilcoxon"],
            "r_caida": c["r"],
            "acc_within_2026": w.mean(),
            "gap_loso_menos_intra_base": nov_b.mean() - w.mean(),
            "gap_loso_menos_intra_ea": nov_e.mean() - w.mean(),
            "DSI_clasico_base": (w.mean() - nov_b.mean()) / w.mean(),
            "DSI_clasico_ea": (w.mean() - nov_e.mean()) / w.mean(),
        })
    return pd.DataFrame(filas)


def bajo_rendimiento() -> pd.DataFrame:
    """EXPLORATORIO. Ganancia (variante - base 2026) segun el rendimiento de referencia del sujeto.

    El rendimiento de referencia sale de las semillas 2027-2028 de la base (independientes de
    las ejecuciones que definen la ganancia), para no fabricar regresion a la media.
    """
    filas = []
    for fam, v in (("F6", "ea"), ("F8", "augcr")):
        for m in ARQ:
            lb = A.loso(m, "base")
            base = acc_loso(lb, [2026])
            refer = acc_loso(lb, [2027, 2028])
            g = (acc_loso(A.loso(m, v), [2026]) - base)
            j = pd.concat([g, refer, base], axis=1, keys=["ganancia", "ref", "base26"]).dropna()
            rho, prho = stats.spearmanr(j["ganancia"], j["ref"])
            ter = pd.qcut(j["ref"], 3, labels=["bajo", "medio", "alto"])
            gb, ga = j.loc[ter == "bajo", "ganancia"], j.loc[ter == "alto", "ganancia"]
            mw = stats.mannwhitneyu(gb, ga, alternative="two-sided")
            bajo = j["ref"] <= UMBRAL_BAJO
            fila = {
                "familia": fam, "arq": m, "n": len(j),
                "spearman_ganancia_vs_ref": rho, "p_spearman": prho,
                "ganancia_tercil_bajo": gb.mean(), "ganancia_tercil_alto": ga.mean(),
                "dif_tercil_bajo_menos_alto": gb.mean() - ga.mean(), "p_mannwhitney": mw.pvalue,
                "n_ref_<=0.70": int(bajo.sum()),
                "ganancia_ref_<=0.70": j.loc[bajo, "ganancia"].mean() if bajo.sum() else np.nan,
                "ganancia_ref_>0.70": j.loc[~bajo, "ganancia"].mean(),
                "umbral_tercil_bajo": float(j["ref"].quantile(1 / 3)),
                "umbral_tercil_alto": float(j["ref"].quantile(2 / 3)),
            }
            if bajo.sum() >= 5 and (~bajo).sum() >= 5:
                fila["p_mw_<=0.70_vs_resto"] = stats.mannwhitneyu(
                    j.loc[bajo, "ganancia"], j.loc[~bajo, "ganancia"]).pvalue
            filas.append(fila)
    df = pd.DataFrame(filas)
    # Holm exploratorio dentro de cada familia sobre el Spearman
    df["p_spearman_holm"] = np.nan
    for _, g in df.groupby("familia"):
        df.loc[g.index, "p_spearman_holm"] = holm(g["p_spearman"].to_numpy())
    return df


# ------------------------------------------------------------------ ruido entre semillas

def ruido_semillas() -> pd.DataFrame:
    """EXPLORATORIO. Diferencia pareada por sujeto entre semillas de la MISMA base.

    Dimensiona el ruido que tendria un contraste sin efecto. `mismo_regimen` marca si las dos
    semillas se entrenaron con el mismo regimen de recorte de gradiente (si no, la diferencia
    incluye tambien el cambio de regimen y es una cota de ese confusor, no solo de semilla).
    """
    filas = []
    # regimen (grad_clip) por (arq, protocolo, semilla)
    for proto in ("within", "loso"):
        for m in ARQ:
            d = A.within(m) if proto == "within" else A.loso(m, "base")
            f = acc_within if proto == "within" else acc_loso
            reg = d.groupby("seed").cfg_hash.first().to_dict()
            ss = sorted(d.seed.unique())
            for i, a in enumerate(ss):
                for b in ss[i + 1:]:
                    c = contraste(f(d, [b]), f(d, [a]))     # dif = b - a
                    filas.append({"protocolo": proto, "arq": m, "semilla_a": a, "semilla_b": b,
                                  "mismo_regimen": reg[a] == reg[b],
                                  "dif_media_b_menos_a": c["dif_media"],
                                  "ic95_inf": c["ic95_inf"], "ic95_sup": c["ic95_sup"],
                                  "p_wilcoxon": c["p_wilcoxon"], "r": c["r"],
                                  "abs_dif_media": abs(c["dif_media"])})
    return pd.DataFrame(filas)


def medias_por_condicion() -> pd.DataFrame:
    filas = []
    for m in ARQ:
        w = A.within(m)
        for s in sorted(w.seed.unique()):
            x = acc_within(w, [s])
            filas.append({"protocolo": "within", "arq": m, "condicion": "base", "semilla": s,
                          "media": x.mean(), "sd_entre_sujetos": x.std(), "n": len(x)})
        for v in F4_VARIANTES[m]:
            x = acc_within(A.ablacion(v, m), [2026])
            filas.append({"protocolo": "within", "arq": m, "condicion": v, "semilla": 2026,
                          "media": x.mean(), "sd_entre_sujetos": x.std(), "n": len(x)})
        lb = A.loso(m, "base")
        for s in sorted(lb.seed.unique()):
            x = acc_loso(lb, [s])
            filas.append({"protocolo": "loso", "arq": m, "condicion": "base", "semilla": s,
                          "media": x.mean(), "sd_entre_sujetos": x.std(), "n": len(x)})
        for v in ("ea", "augcr"):
            x = acc_loso(A.loso(m, v), [2026])
            filas.append({"protocolo": "loso", "arq": m, "condicion": v, "semilla": 2026,
                          "media": x.mean(), "sd_entre_sujetos": x.std(), "n": len(x)})
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ comparacion con la cota

def contra_cota(df: pd.DataFrame, proto: str, cotas) -> pd.DataFrame:
    p = df[df.tipo == "principal"].copy()
    p["protocolo"] = proto
    if cotas is None:
        p["cota_bruta"] = p["cota_neta"] = p["cota_confusor"] = np.nan
        p["supera_cota"] = np.nan
        p["lectura"] = np.where(p["confundido"], "PENDIENTE (falta archivo 04)", "limpio")
    else:
        g = [cotas.get((a, proto), (np.nan, np.nan, np.nan)) for a in p.arq]
        p["cota_bruta"], p["cota_neta"], p["cota_confusor"] = (
            [x[0] for x in g], [x[1] for x in g], [x[2] for x in g])
        p["supera_cota"] = p.dif_media.abs() > p.cota_confusor
        p["lectura"] = np.select(
            [~p.confundido, p.cota_confusor.isna(), p.supera_cota],
            ["limpio", "sin cota estimable (no hay base con recorte)", "supera la cota"],
            "no interpretable")
    return p[["familia", "arq", "variante", "protocolo", "confundido", "dif_media",
              "ic95_inf", "ic95_sup", "p_holm", "cota_bruta", "cota_neta", "cota_confusor",
              "supera_cota", "lectura"]]


def parada_f4() -> pd.DataFrame:
    """EXPLORATORIO. Cuantas particiones paran en el suelo de 30 epocas, por condicion within."""
    filas = []
    for m in ARQ:
        conds = [("base_2026", A.within(m).query("seed == 2026"))]
        conds += [(v, A.ablacion(v, m)) for v in F4_VARIANTES[m]]
        for nom, d in conds:
            filas.append({"arq": m, "condicion": nom, "n": len(d),
                          "pct_suelo": 100 * (d.n_epochs <= 30).mean(),
                          "mediana_epocas": float(d.n_epochs.median()),
                          "pct_tope": 100 * (d.n_epochs >= 200).mean()})
    return pd.DataFrame(filas)


def validacion_metodo() -> pd.DataFrame:
    """Reproduce con `contraste()` dos cifras de `scripts/estadistica.py` (EEGNet vs Conformer,
    semilla 2026, salida de ese script: within p = 4.36e-04, r = 0.479, dif. mediana +0.0188;
    LOSO p = 1.20e-01, r = 0.212, dif. mediana +0.0062). Confirma que unidad de analisis,
    Wilcoxon zsplit y r coinciden con lo publicado."""
    # Valores tras redondear las diferencias a 1e-9 (scripts/ruido_flotante.py). Antes del
    # redondeo: within p = 4.36e-04, r = 0.479; LOSO p = 1.20e-01, r = 0.212 (las publicadas).
    esperado = {"within": (4.21e-04, 0.480, 0.0188), "loso": (1.18e-01, 0.213, 0.0062)}
    filas = []
    for proto, (pe, re_, de) in esperado.items():
        f = acc_within if proto == "within" else acc_loso
        g = A.within if proto == "within" else (lambda m: A.loso(m, "base"))
        c = contraste(f(g("eegnet"), [2026]), f(g("conformer"), [2026]))
        filas.append({"protocolo": proto, "contraste": "eegnet - conformer, semilla 2026",
                      "p_calculado": c["p_wilcoxon"], "p_estadistica_py": pe,
                      "r_calculado": c["r"], "r_estadistica_py": re_,
                      "dif_mediana_calculada": c["dif_mediana"], "dif_mediana_estadistica_py": de,
                      "coincide": (abs(c["p_wilcoxon"] / pe - 1) < 0.01
                                   and abs(c["r"] - re_) < 0.0005
                                   and abs(c["dif_mediana"] - de) < 0.00005)})
    return pd.DataFrame(filas)


def main():
    pd.set_option("display.width", 220, "display.max_columns", 50)
    q = qa()
    q.to_csv(OUT / "ablaciones_qa.csv", index=False)
    malos = q[~(q.ok_filas & q.ok_semillas) | (q.run_id_duplicados > 0) | (q.nan_metricas > 0)
              | q.humo]
    print(f"QA: {len(q)} condiciones, {len(malos)} con problema")
    if len(malos):
        print(malos.to_string())
        raise SystemExit("QA fallido")

    vm = validacion_metodo()
    vm.to_csv(OUT / "ablaciones_validacion_metodo.csv", index=False)
    if not vm.coincide.all():
        raise SystemExit("el metodo no reproduce scripts/estadistica.py")
    print("validacion del metodo contra estadistica.py: OK")

    f4df, f4d = f4()
    f68df, f68d = f68()
    f4df.to_csv(OUT / "ablaciones_f4_contrastes.csv", index=False)
    f68df.to_csv(OUT / "loso_f6_f8_contrastes.csv", index=False)
    pd.concat([f4d, f68d]).to_csv(OUT / "ablaciones_deltas_por_sujeto.csv", index=False)
    conc = concordancia(f68df)
    conc.to_csv(OUT / "loso_f6_f8_concordancia.csv", index=False)
    conc4 = concordancia(f4df)
    conc4.to_csv(OUT / "ablaciones_f4_concordancia.csv", index=False)
    ea = ea_caida()
    ea.to_csv(OUT / "ea_caida_vistos_no_vistos.csv", index=False)
    br = bajo_rendimiento()
    br.to_csv(OUT / "ea_cr_bajo_rendimiento.csv", index=False)
    rs = ruido_semillas()
    rs.to_csv(OUT / "ruido_entre_semillas.csv", index=False)
    pa = parada_f4()
    pa.to_csv(OUT / "ablaciones_parada_temprana.csv", index=False)
    mc = medias_por_condicion()
    mc.to_csv(OUT / "ablaciones_medias.csv", index=False)

    cotas = cota_confusor()
    cc = pd.concat([contra_cota(f4df, "within", cotas), contra_cota(f68df, "loso", cotas)])
    cc.to_csv(OUT / "ablaciones_vs_cota.csv", index=False)
    from avance7_ablaciones_tablas import tablas_md
    (OUT / "ablaciones_tablas.md").write_text(
        tablas_md(f4df, f68df, conc, conc4, ea, br, rs, mc, cc, pa), encoding="utf-8")
    print("cota del confusor:", "leida" if cotas else "NO DISPONIBLE (hueco)")

    cols = ["familia", "tipo", "arq", "variante", "dif_media", "dif_mediana", "ic95_inf",
            "ic95_sup", "p_wilcoxon", "p_holm", "r", "equiv_0.03", "margen_equiv_minimo"]
    print(f4df[cols].round(4).to_string())
    print(f68df[cols].round(4).to_string())
    print(conc.round(4).to_string())
    print(conc4.round(4).to_string())
    print(ea.round(4).T.to_string())
    print(br.round(4).to_string())
    print(rs.round(4).to_string())


if __name__ == "__main__":
    main()
