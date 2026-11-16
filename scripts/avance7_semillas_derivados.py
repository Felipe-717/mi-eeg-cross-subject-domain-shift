#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""avance7_semillas_derivados.py: recalculo con 3 semillas de lo que depende de la exactitud por sujeto.

Se ejecuta desde scripts/avance7_semillas.py (--etapa derivados). Salida en results/avance_7/analisis/:
    tres_semillas_caida_within_loso.csv   LOSO - intra-sujeto por sujeto, por semilla y con 3 semillas
    tres_semillas_ids.csv                 IDS (exactitud y kappa), folds > 0, Wilcoxon visto/no visto y Holm
    tres_semillas_idp.csv                 IDP por semilla y con 3 semillas
    tres_semillas_bajo_umbral.csv         recuento de sujetos con exactitud <= 0.70, intra y LOSO
    tres_semillas_solapamiento.csv        reparto en k arquitecturas y esperado bajo independencia
    tres_semillas_estabilidad_bajo.csv    estabilidad entre semillas de quien queda bajo el umbral
    tres_semillas_correlacion.csv         correlacion intra-sujeto frente a LOSO por arquitectura

Notas de definicion (se repiten en las columnas):
    caida := exactitud LOSO - exactitud intra-sujeto (positiva si LOSO es mayor, como en la publicada
             tab:within-loso); sujeto = unidad; semilla k de un protocolo se aparea con la semilla k
             del otro (las inicializaciones son independientes, el indice solo etiqueta).
    IDS   := (media(acc_visto) - media(acc)) / media(acc_visto) sobre los 54 folds (formula sobre las
             medias, como verificar_cifras.ids). Con 3 semillas, la media de cada fold se toma sobre
             semillas antes de la formula (ids_media3) y se informa tambien la media de los 3 IDS por
             semilla (ids_media_de_ids) y su desviacion entre semillas.
    IDP   := (hold_persona - hold_wbcic) / hold_persona, normalizacion de destino "source" (verificar_cifras.idp).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from ruido_flotante import wilcoxon_pareado  # Wilcoxon sobre diferencias redondeadas a 1e-9

import avance7_semillas as X
import avance7_datos as A
from avance7_semillas import (ARQ, B, OUT, SEEDS, UMBRAL, boot_media_ic, mat, p_w, regimen_str,
                              sin_recorte)
import avance7_stats as S


def ic_q(m, nivel=.95):
    a = (1 - nivel) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


# --------------------------------------------------------------- a. caida intra -> LOSO

def caida():
    filas = []
    for arq in ARQ:
        W, L = mat("within", arq), mat("loso", arq)
        for s in list(SEEDS) + ["media3", "publicada_mixta"]:
            if s == "media3":
                w, l = W.mean(axis=1), L.mean(axis=1)
                reg = f"intra {regimen_str('within', arq)} / LOSO {regimen_str('loso', arq)}"
            elif s == "publicada_mixta":      # intra 3 semillas (eegnet, conformer) o 1 (labram, eegmamba); LOSO 2026
                w = W.mean(axis=1) if arq in ("eegnet", "conformer") else W[2026]
                l = L[2026]
                reg = "como en tab:within-loso"
            else:
                w, l = W[s], L[s]
                reg = f"intra {regimen_str('within', arq, [s])} / LOSO {regimen_str('loso', arq, [s])}"
            d = (l - w).to_numpy()
            p, r = p_w(d)
            lo, hi = boot_media_ic(d)
            filas.append(dict(arq=arq, semilla=s, regimen_intra_loso=reg, intra=w.mean(), loso=l.mean(),
                              caida_loso_menos_intra=d.mean(), mediana=float(np.median(d)),
                              ic95_inf=lo, ic95_sup=hi, n_mejoran_con_loso=int((d > 0).sum()),
                              p_wilcoxon=p, r=r))
        # bootstrap jerarquico de la media de 3 semillas (semillas independientes entre protocolos)
        from avance7_semillas_contrastes import hboot
        m = hboot(L.to_numpy(), W.to_numpy(), comun=False)
        lo, hi = ic_q(m)
        for f in filas:
            if f["arq"] == arq and f["semilla"] == "media3":
                f["jer_ic95_inf"], f["jer_ic95_sup"] = lo, hi
        sd = np.std([f["caida_loso_menos_intra"] for f in filas
                     if f["arq"] == arq and f["semilla"] in SEEDS], ddof=1)
        for f in filas:
            if f["arq"] == arq and f["semilla"] == "media3":
                f["sd_caida_entre_semillas"] = float(sd)
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "tres_semillas_caida_within_loso.csv", index=False)
    return df


# ------------------------------------------------------------------------ b. IDS

def ids_formula(v, n):
    return (np.mean(v) - np.mean(n)) / np.mean(v)


def cuatro(vs, ns):
    """vs, ns: arrays (folds, k). IDS jerarquico: remuestrea folds y semillas (indice comun v/n)."""
    n, k = vs.shape
    g = np.random.default_rng(X.SEMILLA_GEN)
    fi = g.integers(0, n, (B, n))
    si = g.integers(0, k, (B, n, k))
    mv = vs[fi[:, :, None], si].mean(axis=2).mean(axis=1)
    mn = ns[fi[:, :, None], si].mean(axis=2).mean(axis=1)
    return (mv - mn) / mv


def ids():
    filas = []
    por = {}
    for arq in ARQ:
        d = X.tabla("loso", arq)
        piv = {c: d.pivot_table(index="test_sub", columns="seed", values=c, aggfunc="mean")[list(SEEDS)]
               for c in ("acc", "acc_visto", "kappa", "kappa_visto")}
        por[arq] = piv
    for variante in list(SEEDS) + ["media3"]:
        sub = []
        for arq in ARQ:
            piv = por[arq]
            cols = [variante] if variante != "media3" else list(SEEDS)
            v = piv["acc_visto"][cols].mean(axis=1).to_numpy()
            nn = piv["acc"][cols].mean(axis=1).to_numpy()
            kv = piv["kappa_visto"][cols].mean(axis=1).to_numpy()
            kn = piv["kappa"][cols].mean(axis=1).to_numpy()
            p = float(wilcoxon_pareado(v, nn).pvalue)      # como verificar_cifras.p_wilcoxon_ids
            r = float(stats.norm.isf(p / 2) / np.sqrt(len(v)))
            fila = dict(arq=arq, semilla=variante,
                        regimen_loso=regimen_str("loso", arq, cols),
                        acc_visto=v.mean(), acc_no_visto=nn.mean(),
                        ids_acc=ids_formula(v, nn), ids_kappa=ids_formula(kv, kn),
                        folds_mayor_0=int((v > nn).sum()), p_wilcoxon=p, r=r)
            if variante == "media3":
                V3, N3 = piv["acc_visto"].to_numpy(), piv["acc"].to_numpy()
                KV3, KN3 = piv["kappa_visto"].to_numpy(), piv["kappa"].to_numpy()
                ids_s = [ids_formula(V3[:, i], N3[:, i]) for i in range(3)]
                idk_s = [ids_formula(KV3[:, i], KN3[:, i]) for i in range(3)]
                lo, hi = ic_q(cuatro(V3, N3))
                lok, hik = ic_q(cuatro(KV3, KN3))
                fila.update(ids_acc_media_de_ids=float(np.mean(ids_s)),
                            ids_acc_sd_entre_semillas=float(np.std(ids_s, ddof=1)),
                            ids_kappa_media_de_ids=float(np.mean(idk_s)),
                            ids_kappa_sd_entre_semillas=float(np.std(idk_s, ddof=1)),
                            ids_acc_jer_ic95_inf=lo, ids_acc_jer_ic95_sup=hi,
                            ids_kappa_jer_ic95_inf=lok, ids_kappa_jer_ic95_sup=hik)
            sub.append(fila)
        ph = S.holm([f["p_wilcoxon"] for f in sub])      # familia: los 4 contrastes de esa variante
        for f, h in zip(sub, ph):
            f["p_holm"] = float(h)
            f["declara_degradacion"] = bool(f["ids_acc"] > 0 and h < .05)
        filas += sub
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "tres_semillas_ids.csv", index=False)
    return df


# ------------------------------------------------------------------------ c. IDP

def hold_matriz(tipo, arq):
    """Exactitud 'source' por sujeto y semilla, sujetos x semillas."""
    d = A.transferencia(tipo, arq, "base")
    d = d[d.norm_destino == "source"]
    p = d.pivot_table(index="sub", columns="seed", values="acc", aggfunc="mean")[list(SEEDS)]
    assert not p.isna().any().any()
    return p


def idp():
    import verificar_cifras as V
    filas = []
    for arq in ARQ:
        P, W = hold_matriz("h44_openbmi", arq), hold_matriz("h44_wbcic", arq)
        assert P.shape[0] == 10 and W.shape[0] == 51
        por_s = []
        for s in list(SEEDS) + ["media3"]:
            cols = [s] if s != "media3" else list(SEEDS)
            pn, pw = P[cols].mean(axis=1).mean(), W[cols].mean(axis=1).mean()
            f = dict(arq=arq, semilla=s, hold_persona=pn, hold_wbcic=pw, idp=(pn - pw) / pn,
                     caida_protocolo=pn - pw)
            if s == 2026:
                assert abs(f["idp"] - V.idp(arq)) < 1e-9, "IDP 2026 no coincide con verificar_cifras"
            if s in SEEDS:
                por_s.append(f["idp"])
            filas.append(f)
        # jerarquico: sujetos de cada conjunto por separado; semillas independientes por sujeto
        g = np.random.default_rng(X.SEMILLA_GEN)
        Pn, Wn = P.to_numpy(), W.to_numpy()
        ip, iw = g.integers(0, 10, (B, 10)), g.integers(0, 51, (B, 51))
        sp, sw = g.integers(0, 3, (B, 10, 3)), g.integers(0, 3, (B, 51, 3))
        mp = Pn[ip[:, :, None], sp].mean(axis=2).mean(axis=1)
        mw = Wn[iw[:, :, None], sw].mean(axis=2).mean(axis=1)
        lo, hi = ic_q((mp - mw) / mp)
        for f in filas:
            if f["arq"] == arq and f["semilla"] == "media3":
                f.update(idp_sd_entre_semillas=float(np.std(por_s, ddof=1)),
                         idp_media_de_idps=float(np.mean(por_s)),
                         jer_ic95_inf=lo, jer_ic95_sup=hi)
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "tres_semillas_idp.csv", index=False)
    return df


# --------------------------------------------------- d. bajo rendimiento y solapamiento

def reparto(bajo: pd.DataFrame):
    """bajo: sujetos x 4 booleanos. Devuelve (observado[0..4], esperado[0..4])."""
    obs = [int((bajo.sum(axis=1) == i).sum()) for i in range(5)]
    dist = np.array([1.0])
    for pr in bajo.mean():
        dist = np.convolve(dist, [1 - pr, pr])
    return obs, [float(x * len(bajo)) for x in dist]


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else np.nan


def bajo_rendimiento():
    cnt, sol, est, cor = [], [], [], []
    M = {(p, a): mat(p, a) for p in ("within", "loso") for a in ARQ}
    variantes = list(SEEDS) + ["media3", "publicada_mixta"]

    def vec(proto, arq, v):
        m = M[(proto, arq)]
        if v == "media3":
            return m.mean(axis=1)
        if v == "publicada_mixta":
            if proto == "within":
                return m.mean(axis=1) if arq in ("eegnet", "conformer") else m[2026]
            return m[2026]
        return m[v]

    for v in variantes:
        bajo_w = pd.DataFrame({a: vec("within", a, v) <= UMBRAL + 1e-9 for a in ARQ})
        bajo_l = pd.DataFrame({a: vec("loso", a, v) <= UMBRAL + 1e-9 for a in ARQ})
        for a in ARQ:
            cnt.append(dict(semilla=v, arq=a, bajo_intra=int(bajo_w[a].sum()),
                            bajo_loso=int(bajo_l[a].sum()),
                            regimen_intra=("mixta" if v == "publicada_mixta" else
                                           regimen_str("within", a, SEEDS if v == "media3" else [v])),
                            regimen_loso=("2026" if v == "publicada_mixta" else
                                          regimen_str("loso", a, SEEDS if v == "media3" else [v]))))
        obs, esp = reparto(bajo_w)
        fila = dict(semilla=v, interseccion_4=int(bajo_w.all(axis=1).sum()))
        for i in range(5):
            fila[f"obs_k{i}"] = obs[i]
            fila[f"esp_k{i}"] = esp[i]
        sol.append(fila)
        for a in ARQ:
            x, y = vec("within", a, v), vec("loso", a, v)
            r_ = float(np.corrcoef(x, y)[0, 1])
            idx = np.random.default_rng(X.SEMILLA_GEN).integers(0, 54, (B, 54))
            xx, yy = x.to_numpy()[idx], y.to_numpy()[idx]
            rb = ((xx - xx.mean(1, keepdims=True)) * (yy - yy.mean(1, keepdims=True))).sum(1) / np.sqrt(
                ((xx - xx.mean(1, keepdims=True)) ** 2).sum(1) * ((yy - yy.mean(1, keepdims=True)) ** 2).sum(1))
            lo, hi = ic_q(rb)
            cor.append(dict(semilla=v, arq=a, r_pearson_intra_loso=r_, ic95_inf=lo, ic95_sup=hi))

    # estabilidad entre semillas (exactitud por semilla, umbral <= 0.70, intra-sujeto)
    for proto in ("within", "loso"):
        sets = {}
        for a in ARQ:
            S_ = {s: set(M[(proto, a)].index[M[(proto, a)][s] <= UMBRAL + 1e-9]) for s in SEEDS}
            sets[a] = S_
            pares = [jaccard(S_[x], S_[y]) for x in SEEDS for y in SEEDS if x < y]
            todos = set.intersection(*S_.values())
            alguno = set.union(*S_.values())
            m3 = set(M[(proto, a)].index[M[(proto, a)].mean(axis=1) <= UMBRAL + 1e-9])
            est.append(dict(protocolo=proto, arq=a, n_2026=len(S_[2026]), n_2027=len(S_[2027]),
                            n_2028=len(S_[2028]), n_en_las_3=len(todos), n_en_alguna=len(alguno),
                            n_en_2_o_mas=int(sum(sum(s in S_[t] for t in SEEDS) >= 2 for s in alguno)),
                            jaccard_medio_pares=float(np.mean(pares)),
                            jaccard_min=float(np.min(pares)), jaccard_max=float(np.max(pares)),
                            n_media3=len(m3), jaccard_media3_vs_2026=jaccard(m3, S_[2026]),
                            sujetos_inestables=" ".join(sorted(alguno - todos))))
        if proto == "within":
            I = {s: set.intersection(*[sets[a][s] for a in ARQ]) for s in SEEDS}
            todos = set.intersection(*I.values())
            alguno = set.union(*I.values())
            m3 = set.intersection(*[set(M[(proto, a)].index[M[(proto, a)].mean(axis=1) <= UMBRAL + 1e-9])
                                    for a in ARQ])
            pares = [jaccard(I[x], I[y]) for x in SEEDS for y in SEEDS if x < y]
            est.append(dict(protocolo=proto, arq="interseccion_4", n_2026=len(I[2026]), n_2027=len(I[2027]),
                            n_2028=len(I[2028]), n_en_las_3=len(todos), n_en_alguna=len(alguno),
                            n_en_2_o_mas=int(sum(sum(s in I[t] for t in SEEDS) >= 2 for s in alguno)),
                            jaccard_medio_pares=float(np.mean(pares)), jaccard_min=float(np.min(pares)),
                            jaccard_max=float(np.max(pares)), n_media3=len(m3),
                            jaccard_media3_vs_2026=jaccard(m3, I[2026]),
                            sujetos_inestables=" ".join(sorted(alguno - todos))))
    pd.DataFrame(cnt).to_csv(OUT / "tres_semillas_bajo_umbral.csv", index=False)
    pd.DataFrame(sol).to_csv(OUT / "tres_semillas_solapamiento.csv", index=False)
    pd.DataFrame(est).to_csv(OUT / "tres_semillas_estabilidad_bajo.csv", index=False)
    pd.DataFrame(cor).to_csv(OUT / "tres_semillas_correlacion.csv", index=False)
    return pd.DataFrame(cnt), pd.DataFrame(sol), pd.DataFrame(est), pd.DataFrame(cor)


def main():
    caida()
    ids()
    idp()
    bajo_rendimiento()
    print("[derivados] escritos tres_semillas_*.csv")
