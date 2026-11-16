#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""avance7_transferencia.py: transferencia OpenBMI -> WBCIC-SHU, nativo y 44 + 10 con 3 semillas
(parte B) y efecto del alineamiento euclidiano frente a la base de la semilla 2026 (parte C).

Entradas (via scripts/avance7_datos.py; rutas completas en su docstring):
  transferencia("cruzado"|"h44_openbmi"|"h44_wbcic"|"nativo"|"train54"|"h44", arq, "base"|"ea")
  Base: semilla 2026 (results/punto6*, A40) + 2027 y 2028 (results/avance_7, L40S). Todas con
  recorte de gradiente (cfg_hash identico por arquitectura; se verifica abajo).
  EA: semilla 2026 (avance_7), vista comun de 45 canales (`_ea45`).
Salidas: results/avance_7/analisis/transf_*.csv
Comando: ~/anaconda3/envs/eeg/python.exe scripts/avance7_transferencia.py

Definiciones (las de scripts/verificar_cifras.py):
  transferido    media sobre los 51 sujetos de WBCIC de `acc` en `cruzado` (40 segmentos
                 reservados por sujeto; politica de normalizacion de destino `source`).
  nativo         `acc_reservado` del modelo entrenado en WBCIC (2 040 segmentos reservados).
  vistos44       `acc` de la fila `all` del modelo de 44 sujetos: exactitud en el conjunto de
                 VALIDACION (n_test = 0), que decide la parada temprana; optimista.
  persona nueva  media sobre los 10 sujetos apartados (`holdout_openbmi`, 400 ensayos c/u).
  wbcic44        media sobre los 51 sujetos de `holdout_wbcic` (mismo modelo de 44 sujetos).
  caida persona = vistos44 - persona; caida protocolo = persona - wbcic44; IDP = protocolo/persona.

FAMILIAS (declaradas antes de calcular):
  F-B1 (politicas, parte B): Wilcoxon pareado por sujeto (51), `source` frente a `per_trial` y
       `source` frente a `per_subject`, por arquitectura, sobre la exactitud promediada entre
       semillas dentro del sujeto: 8 contrastes, Holm sobre 8.
  F-C  (EA frente a base 2026, parte C): Wilcoxon pareado por sujeto de EA - base en 3
       evaluaciones x 4 arquitecturas = 12 contrastes (cruzado/51, h44_wbcic/51 y
       h44_openbmi/10, politica `source`), Holm sobre 12.
  Sin familia: estimaciones con IC (caidas, diferencias entre caidas, nativo - transferido).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import avance7_datos as A
import avance7_stats as S

OUT = A.RAIZ / "results" / "avance_7" / "analisis"
OUT.mkdir(parents=True, exist_ok=True)
POL = ["source", "per_trial", "per_subject"]
SEM = (2026, 2027, 2028)
PUBLICADO = {  # main.tex cap. 4 / verificar_cifras.py (semilla 2026)
    "transferido": {"eegnet": .626, "conformer": .611, "labram": .613, "eegmamba": .589},
    "nativo": {"eegnet": .808, "conformer": .800, "labram": .814, "eegmamba": .762},
    "vistos44": {"eegnet": .802, "conformer": .814, "labram": .813, "eegmamba": .753},
    "persona": {"eegnet": .742, "conformer": .718, "labram": .745, "eegmamba": .700},
    "wbcic44": {"eegnet": .621, "conformer": .618, "labram": .628, "eegmamba": .598},
    "caida_persona": {"eegnet": .059, "conformer": .096, "labram": .068, "eegmamba": .054},
    "caida_protocolo": {"eegnet": .121, "conformer": .100, "labram": .117, "eegmamba": .102},
    "idp": {"eegnet": .163, "conformer": .139, "labram": .157, "eegmamba": .146},
    "transf_per_trial": {"eegnet": .610, "conformer": .586, "labram": .599, "eegmamba": .594},
    "transf_per_subject": {"eegnet": .609, "conformer": .600, "labram": .612, "eegmamba": .607},
}
PUBLICADO_IC = {  # tab:descomposicion-ic (semilla 2026): (est, lo, hi) persona, protocolo, dif
    "eegnet": [(.059, -.013, .131), (.121, .042, .201), (.062, -.084, .210)],
    "conformer": [(.096, .012, .173), (.100, .015, .190), (.003, -.154, .174)],
    "labram": [(.068, .002, .135), (.117, .039, .192), (.049, -.092, .185)],
    "eegmamba": [(.054, -.006, .109), (.102, .040, .167), (.048, -.065, .170)],
}


# ------------------------------------------------------------------ carga y QA

def esperadas(variante):
    return SEM if variante == "base" else (2026,)


def cargar_todo():
    """D[(arq, variante)][seed] = dict con series por sujeto y escalares."""
    D, qa = {}, []
    for m in A.ARQ:
        for v in ("base", "ea"):
            T = {t: A.transferencia(t, m, v) for t in
                 ("cruzado", "h44_openbmi", "h44_wbcic", "nativo", "train54", "h44")}
            hashes = set()
            for t, d in T.items():
                hashes |= set(d.cfg_hash.unique())
                ok_seed = sorted(int(s) for s in d.seed.unique()) == list(esperadas(v))
                qa.append(dict(arq=m, variante=v, tabla=t, filas=len(d),
                               semillas=sorted(int(s) for s in d.seed.unique()),
                               semillas_ok=ok_seed, nan_acc=int(d.acc.isna().sum()),
                               cfg_hash="|".join(sorted(d.cfg_hash.unique())),
                               regimen="|".join(sorted(d.regimen.unique())),
                               duplicados=int(d.duplicated(
                                   [c for c in ("seed", "sub", "norm_destino") if c in d.columns]
                                   if "sub" in d.columns else ["seed"]).sum()),
                               sujetos=int(d["sub"].nunique()) if "sub" in d.columns else
                               int(d.n_subjects.iloc[0]),
                               n_epochs="|".join(str(int(x)) for x in d.sort_values("seed").n_epochs)
                               if "n_epochs" in d.columns else ""))
            assert len(hashes) == 1, (m, v, hashes)
            D[(m, v)] = {}
            for s in esperadas(v):
                e = {}
                for pol in POL:
                    c = T["cruzado"]; c = c[(c.seed == s) & (c.norm_destino == pol)]
                    e[f"trans_{pol}"] = c.set_index("sub").acc.sort_index()
                    h = T["h44_openbmi"]; h = h[(h.seed == s) & (h.norm_destino == pol)]
                    e[f"persona_{pol}"] = h.set_index("sub").acc.sort_index()
                    w = T["h44_wbcic"]; w = w[(w.seed == s) & (w.norm_destino == pol)]
                    e[f"wbcic44_{pol}"] = w.set_index("sub").acc.sort_index()
                n = T["nativo"]; n = n[n.seed == s].iloc[0]
                e["nativo"] = float(n.acc_reservado); e["nativo_acc_noreservado"] = float(n.acc)
                e["vistos54"] = float(T["train54"][T["train54"].seed == s].acc.iloc[0])
                e["vistos44"] = float(T["h44"][T["h44"].seed == s].acc.iloc[0])
                D[(m, v)][s] = e
    return D, pd.DataFrame(qa)


def escalares(e):
    """Cantidades escalares de una (arq, variante, semilla)."""
    q = {"transferido": e["trans_source"].mean(),
         "transf_per_trial": e["trans_per_trial"].mean(),
         "transf_per_subject": e["trans_per_subject"].mean(),
         "nativo": e["nativo"], "vistos54": e["vistos54"], "vistos44": e["vistos44"],
         "persona": e["persona_source"].mean(), "wbcic44": e["wbcic44_source"].mean()}
    q["dif_nativo_menos_transferido"] = q["nativo"] - q["transferido"]
    q["caida_persona"] = q["vistos44"] - q["persona"]
    q["caida_protocolo"] = q["persona"] - q["wbcic44"]
    q["protocolo_menos_persona"] = q["caida_protocolo"] - q["caida_persona"]
    q["idp"] = q["caida_protocolo"] / q["persona"]
    q["cociente_protocolo_persona"] = q["caida_protocolo"] / q["caida_persona"]
    q["vistos54_menos_transferido"] = q["vistos54"] - q["transferido"]
    return q


# ------------------------------------------------------------------ bootstrap de las caidas

def _caidas(v, p, w):
    cp, cw = v - p, p - w
    return cp, cw, cw - cp


def boot_caidas(pers, wb, vis, remuestrear_semillas, b=S.B, seed=S.SEMILLA_BOOT,
                pers2=None, wb2=None, vis2=None):
    """Bootstrap de las caidas. pers: (S,10), wb: (S,51), vis: (S,). Se remuestrean por
    separado los 10 sujetos apartados y los 51 de WBCIC (como en tab:descomposicion-ic); con
    `remuestrear_semillas` se remuestrean ademas las S semillas (bootstrap jerarquico). Los
    mismos indices se aplican a (pers2, wb2, vis2) si se dan, para contrastar dos condiciones
    sobre los mismos sujetos remuestreados."""
    rng = np.random.default_rng(seed)
    S_ = pers.shape[0]
    pidx = rng.integers(0, pers.shape[1], size=(b, pers.shape[1]))
    widx = rng.integers(0, wb.shape[1], size=(b, wb.shape[1]))
    sidx = (rng.integers(0, S_, size=(b, S_)) if remuestrear_semillas
            else np.tile(np.arange(S_), (b, 1)))

    def calc(P, W, V):
        p = P[sidx[:, :, None], pidx[:, None, :]].mean(axis=(1, 2))
        w = W[sidx[:, :, None], widx[:, None, :]].mean(axis=(1, 2))
        v = V[sidx].mean(axis=1)
        return _caidas(v, p, w), p, w, v

    (cp, cw, d), p, w, v = calc(pers, wb, vis)
    out = {"caida_persona": cp, "caida_protocolo": cw, "protocolo_menos_persona": d,
           "idp": cw / p}
    if pers2 is not None:
        (cp2, cw2, d2), p2, w2, v2 = calc(pers2, wb2, vis2)
        out.update({"ea_menos_base_caida_persona": cp2 - cp,
                    "ea_menos_base_caida_protocolo": cw2 - cw,
                    "ea_menos_base_protocolo_menos_persona": d2 - d,
                    "ea_menos_base_wbcic44": w2 - w, "ea_menos_base_persona": p2 - p})
    return {k: (float(np.quantile(x, .025)), float(np.quantile(x, .975))) for k, x in out.items()}


def mats(D, m, v, semillas):
    P = np.stack([D[(m, v)][s]["persona_source"].to_numpy() for s in semillas])
    W = np.stack([D[(m, v)][s]["wbcic44_source"].to_numpy() for s in semillas])
    V = np.array([D[(m, v)][s]["vistos44"] for s in semillas])
    return P, W, V


# ------------------------------------------------------------------ parte B

def parte_b(D):
    por_semilla, resumen, repro, ic_rows, conclus = [], [], [], [], []
    for m in A.ARQ:
        Q = {s: escalares(D[(m, "base")][s]) for s in SEM}
        for s in SEM:
            for k, x in Q[s].items():
                por_semilla.append(dict(arq=m, variante="base", semilla=s, cantidad=k, valor=x))
        for k in Q[2026]:
            vals = np.array([Q[s][k] for s in SEM])
            resumen.append(dict(arq=m, cantidad=k, media_3sem=vals.mean(), sd_entre_semillas=vals.std(ddof=1),
                                min=vals.min(), max=vals.max(), rango=vals.max() - vals.min(),
                                v2026=vals[0], v2027=vals[1], v2028=vals[2],
                                signo_estable=bool(np.all(np.sign(vals) == np.sign(vals[0])))))
        for k, pub in PUBLICADO.items():
            kk = {"transferido": "transferido"}.get(k, k)
            val = Q[2026][kk]
            repro.append(dict(arq=m, cantidad=k, publicado=pub[m], recalculado_2026=val,
                              coincide_3dec=bool(round(val, 3) == pub[m]),
                              diferencia=round(val - pub[m], 4)))
        # IC: reproduccion 2026, condicional 3 semillas, jerarquico 3 semillas, por semilla
        P, W, V = mats(D, m, "base", SEM)
        for nombre, sel, rs in (("2026_reproduccion", [0], False), ("3sem_sujetos", [0, 1, 2], False),
                                ("3sem_jerarquico", [0, 1, 2], True),
                                ("solo_2027", [1], False), ("solo_2028", [2], False)):
            ic = boot_caidas(P[sel], W[sel], V[sel], rs)
            est = {"caida_persona": np.mean(V[sel]) - P[sel].mean(),
                   "caida_protocolo": P[sel].mean() - W[sel].mean()}
            est["protocolo_menos_persona"] = est["caida_protocolo"] - est["caida_persona"]
            est["idp"] = est["caida_protocolo"] / P[sel].mean()
            for q in ("caida_persona", "caida_protocolo", "protocolo_menos_persona", "idp"):
                lo, hi = ic[q]
                ic_rows.append(dict(arq=m, metodo=nombre, cantidad=q, estimacion=est[q], ic_inf=lo, ic_sup=hi,
                                    excluye_cero=bool(lo > 0 or hi < 0)))
    ic = pd.DataFrame(ic_rows)
    # reproduccion de tab:descomposicion-ic
    rep = []
    for m, tri in PUBLICADO_IC.items():
        for q, (e, lo, hi) in zip(("caida_persona", "caida_protocolo", "protocolo_menos_persona"), tri):
            r = ic[(ic.arq == m) & (ic.metodo == "2026_reproduccion") & (ic.cantidad == q)].iloc[0]
            rep.append(dict(arq=m, cantidad=q, pub_est=e, pub_inf=lo, pub_sup=hi, rec_est=r.estimacion,
                            rec_inf=r.ic_inf, rec_sup=r.ic_sup,
                            max_dif_limite=max(abs(r.ic_inf - lo), abs(r.ic_sup - hi))))
    # promedio de las cuatro arquitecturas (fila "Promedio" publicada)
    # politicas: F-B1
    pol = []
    for m in A.ARQ:
        med = {p: pd.concat([D[(m, "base")][s][f"trans_{p}"] for s in SEM], axis=1).mean(axis=1) for p in POL}
        for p in ("per_trial", "per_subject"):
            d = (med["source"] - med[p]).to_numpy()
            w = S.wilcoxon_r(d); lo, hi = S.boot_ic(d)
            pol.append(dict(familia="F-B1", arq=m, contraste=f"source - {p}", n=w["n"],
                            media_source=med["source"].mean(), media_otra=med[p].mean(),
                            dif_media=d.mean(), ic_inf=lo, ic_sup=hi, p=w["p"], r=w["r"]))
        # exactitud de cada politica por semilla (sin contraste)
    pol = pd.DataFrame(pol)
    pol["p_holm"] = S.holm(pol.p)
    pol["sig_holm"] = pol.p_holm < .05
    return (pd.DataFrame(por_semilla), pd.DataFrame(resumen), pd.DataFrame(repro), ic,
            pd.DataFrame(rep), pol)


# ------------------------------------------------------------------ parte C

def parte_c(D, resumen):
    filas, cont, rec = [], [], []
    for m in A.ARQ:
        eb, ee = D[(m, "base")][2026], D[(m, "ea")][2026]
        qb, qe = escalares(eb), escalares(ee)
        rb = resumen[resumen.arq == m].set_index("cantidad")
        for k in ("transferido", "nativo", "dif_nativo_menos_transferido", "vistos44", "persona", "wbcic44",
                  "caida_persona", "caida_protocolo", "protocolo_menos_persona", "idp",
                  "transf_per_trial", "transf_per_subject"):
            sd = rb.loc[k, "sd_entre_semillas"]
            lo, hi = rb.loc[k, "min"], rb.loc[k, "max"]
            # dispersion de la base SIN la semilla 2026 no es posible con 2 semillas: se usa la de 3
            filas.append(dict(arq=m, cantidad=k, base_2026=qb[k], ea_2026=qe[k], ea_menos_base=qe[k] - qb[k],
                              base_media_3sem=rb.loc[k, "media_3sem"], base_sd_3sem=sd, base_min=lo, base_max=hi,
                              cociente_cambio_sobre_sd=(qe[k] - qb[k]) / sd if sd > 0 else np.nan,
                              ea_fuera_rango_base=bool(qe[k] < lo or qe[k] > hi)))
        # contrastes pareados por sujeto
        for ev, kb in (("cruzado_51", "trans_source"), ("h44_wbcic_51", "wbcic44_source"),
                       ("h44_openbmi_10", "persona_source")):
            d = (ee[kb] - eb[kb]).to_numpy()
            w = S.wilcoxon_r(d); lo, hi = S.boot_ic(d)
            cont.append(dict(familia="F-C", arq=m, evaluacion=ev, n=w["n"], base=eb[kb].mean(), ea=ee[kb].mean(),
                             dif_media=d.mean(), dif_mediana=float(np.median(d)), ic_inf=lo, ic_sup=hi,
                             p=w["p"], r=w["r"], mejoran=int((d > 0).sum()), empeoran=int((d < 0).sum())))
        # bootstrap pareado de las caidas
        Pb, Wb, Vb = mats(D, m, "base", (2026,))
        Pe, We, Ve = mats(D, m, "ea", (2026,))
        ic = boot_caidas(Pb, Wb, Vb, False, pers2=Pe, wb2=We, vis2=Ve)
        for q in ("ea_menos_base_caida_persona", "ea_menos_base_caida_protocolo",
                  "ea_menos_base_protocolo_menos_persona", "ea_menos_base_persona", "ea_menos_base_wbcic44"):
            base_q = q.replace("ea_menos_base_", "")
            est = qe[base_q] - qb[base_q]
            rec.append(dict(arq=m, cantidad=q, estimacion=est, ic_inf=ic[q][0], ic_sup=ic[q][1],
                            excluye_cero=bool(ic[q][0] > 0 or ic[q][1] < 0)))
        # fraccion recuperada de la caida por protocolo y de la brecha frente al nativo
        rec.append(dict(arq=m, cantidad="fraccion_recuperada_caida_protocolo",
                        estimacion=(qb["caida_protocolo"] - qe["caida_protocolo"]) / qb["caida_protocolo"],
                        ic_inf=np.nan, ic_sup=np.nan, excluye_cero=np.nan))
        rec.append(dict(arq=m, cantidad="fraccion_recuperada_brecha_nativo_menos_transferido",
                        estimacion=(qb["dif_nativo_menos_transferido"] - qe["dif_nativo_menos_transferido"])
                        / qb["dif_nativo_menos_transferido"], ic_inf=np.nan, ic_sup=np.nan, excluye_cero=np.nan))
    cont = pd.DataFrame(cont)
    cont["p_holm"] = S.holm(cont.p)
    cont["sig_holm"] = cont.p_holm < .05
    return pd.DataFrame(filas), cont, pd.DataFrame(rec)


# ------------------------------------------------------------------ curvas (tope de 200 epocas)

def curvas():
    rutas = {
        ("train54", "base", 2026): "results/punto6/reports/openbmi/curvas/baseline_{m}_all_comun_wbcic_global{tag}_epochs_cache_ica.csv",
        ("h44", "base", 2026): "results/punto6b/openbmi/curvas/baseline_{m}_all_comun_wbcic_global{tag}_hold10s2026_epochs_cache_ica.csv",
        ("train54", "base", 0): "results/avance_7/reports/openbmi/curvas/baseline_{m}_all_comun_wbcic_global{tag}_epochs_cache_ica.csv",
        ("h44", "base", 1): "results/avance_7/reports/openbmi/curvas/baseline_{m}_all_comun_wbcic_global{tag}_hold10s2026_epochs_cache_ica.csv",
        ("train54", "ea", 0): "results/avance_7/reports/openbmi/curvas/baseline_{m}_all_comun_wbcic_global{tag}_epochs_cache_ica_ea45.csv",
        ("h44", "ea", 1): "results/avance_7/reports/openbmi/curvas/baseline_{m}_all_comun_wbcic_global{tag}_hold10s2026_epochs_cache_ica_ea45.csv",
    }
    filas = []
    for m in A.ARQ:
        for (t, v, _), pl in rutas.items():
            f = A.RAIZ / pl.format(m=m, tag=A.TAG[m])
            if not f.exists():
                continue
            d = pd.read_csv(f)
            for rid, g in d.groupby("run_id"):
                g = g.reset_index(drop=True)
                i = int(g.val_loss.idxmin())
                n = len(g)
                filas.append(dict(
                    arq=m, tabla=t, variante=v, run_id=rid, semilla=int(rid.split("seed")[1].split("_")[0]),
                    n_epocas=n, epoca_minima=int(g.epoch.iloc[i]), val_loss_min=g.val_loss.min(),
                    val_loss_final=g.val_loss.iloc[-1],
                    mejora_ultimas_50=g.val_loss.iloc[:-50].min() - g.val_loss.min() if n > 60 else np.nan,
                    en_tope_200=bool(n >= 200)))
    return pd.DataFrame(filas)


def main():
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    D, qa = cargar_todo()
    qa.to_csv(OUT / "transf_qa.csv", index=False)
    por_sem, res, repro, ic, rep_ic, pol = parte_b(D)
    por_sem.to_csv(OUT / "transf_por_semilla.csv", index=False)
    res.to_csv(OUT / "transf_resumen_3semillas.csv", index=False)
    repro.to_csv(OUT / "transf_reproduccion_2026.csv", index=False)
    ic.to_csv(OUT / "transf_ic_caidas.csv", index=False)
    rep_ic.to_csv(OUT / "transf_reproduccion_ic_2026.csv", index=False)
    pol.to_csv(OUT / "transf_politicas_contrastes.csv", index=False)
    comp, cont, rec = parte_c(D, res)
    comp.to_csv(OUT / "transf_ea_comparacion.csv", index=False)
    cont.to_csv(OUT / "transf_ea_contrastes.csv", index=False)
    rec.to_csv(OUT / "transf_ea_caidas_ic.csv", index=False)
    cv = curvas()
    cv.to_csv(OUT / "transf_curvas_epocas.csv", index=False)
    print(qa[["arq", "variante", "tabla", "filas", "semillas_ok", "cfg_hash", "regimen", "n_epochs"]].to_string())
    print(repro.to_string()); print(rep_ic.round(3).to_string())


if __name__ == "__main__":
    main()
