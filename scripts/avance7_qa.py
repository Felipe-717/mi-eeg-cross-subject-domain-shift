#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""avance7_qa.py: control de calidad de los resultados del avance 7 (y de la base previa).

Uso:
    ~/anaconda3/envs/eeg/python.exe scripts/avance7_qa.py

Lee solo los CSV que expone `avance7_datos.py`, los fragmentos `__wN` (para comprobar que el
consolidado los contiene), `results/avance_7/estado.json` y `results/avance_7/logs/*.log`.
Escribe `results/avance_7/analisis/qa_condiciones.csv` (una fila por condicion) y
`results/avance_7/analisis/qa_fragmentos.csv`, y imprime las anomalias.

Comprobaciones por condicion: filas esperadas vs observadas, sujetos unicos, pliegues,
semillas, `run_id` duplicados, NaN en metricas, `cfg_hash` contra el esperado, `n_epochs`
(el minimo posible es 30 = 10 de calentamiento + 20 de paciencia; `qa_parada_temprana.csv`
da el % de pliegues en ese suelo por condicion, regimen y semilla; `ep_<=25` debe ser 0) y
exactitud media por semilla (solo descriptiva, para detectar saltos de
regimen; no es un contraste).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import avance7_datos as A  # noqa: E402

RAIZ = A.RAIZ
SALIDA = RAIZ / "results" / "avance_7" / "analisis"
SALIDA.mkdir(parents=True, exist_ok=True)

# cfg_hash de la base previa (ESTADO.md §2) y lr/batch de cada arquitectura (avance_7.py)
HASH_PREVIO = {"eegnet": "53d6a9fc85b4", "conformer": "728ed36e1f36",
               "labram": "7b96314fb256", "eegmamba": "eabca26d5bdf"}
LRB = {"eegnet": (2.8e-3, 256), "conformer": (1e-3, 256), "labram": (1e-4, 32),
       "eegmamba": (1e-3, 32)}
# Hash de las ejecuciones `all` previas: se calcularon con el codigo actual (con grad_clip)
UMBRAL_PARADA = 25
METRICAS = ["acc", "kappa", "f1_macro"]


def hash_esperado(arq: str, aug: str = "base") -> str:
    """cfg_hash que produce el codigo ACTUAL (src/training.py + variantes) para `arq`."""
    import copy
    import training as tr
    cfg = copy.deepcopy(tr.DEFAULT_TRAIN_CFG)
    if aug != "base":
        import variantes as va
        cfg["augment"].update(va.PRESETS_AUG[aug])
    lr, b = LRB[arq]
    cfg.update(lr=lr, batch_size_physical=b, batch_size_effective=b)
    cfg = {k: v for k, v in cfg.items() if k not in {"ckpt_cada", "log_cada"}}
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def hash_sin_clip(arq: str) -> str:
    import copy
    import training as tr
    cfg = copy.deepcopy(tr.DEFAULT_TRAIN_CFG)
    lr, b = LRB[arq]
    cfg.update(lr=lr, batch_size_physical=b, batch_size_effective=b)
    cfg = {k: v for k, v in cfg.items() if k not in {"ckpt_cada", "log_cada", "grad_clip"}}
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def aug_de(nombre: str) -> str:
    for a in ("augcr", "augfft", "augnone"):
        if a in nombre:
            return {"augcr": "cr", "augfft": "fft", "augnone": "none"}[a]
    return "base"


def qa_condicion(nombre, f, filas_esp, sem_esp):
    d = f()
    arq = d.arq.iloc[0]
    kind = nombre.split("_")[0]
    r = {"condicion": nombre, "arq": arq, "fuente": " | ".join(sorted(d.fuente.unique())),
         "filas": len(d), "filas_esperadas": filas_esp,
         "semillas": ",".join(str(s) for s in sorted(d.seed.unique())),
         "semillas_esperadas": ",".join(str(s) for s in sem_esp)}
    notas = []
    # sujetos y pliegues
    sc = "sub" if "sub" in d.columns else ("test_sub" if "test_sub" in d.columns else None)
    r["sujetos"] = d[sc].nunique() if sc else np.nan
    r["pliegues"] = d["fold"].nunique() if "fold" in d.columns else np.nan
    # run_id duplicado (solo si hay run_id): cada fila de entrenamiento es unica
    if "run_id" in d.columns and kind in ("within", "loso", "abl", "transf") and "run_id" in d:
        dup = int(d.duplicated("run_id").sum())
        r["run_id_duplicados"] = dup
        if dup:
            notas.append(f"{dup} run_id duplicados")
    # fewshot: clave propia
    if kind == "fewshot":
        clave = ["run_id", "k", "rep", "metodo"]
        r["run_id_duplicados"] = int(d.duplicated(clave).sum())
        r["k"] = ",".join(str(k) for k in sorted(d.k.unique()))
        r["metodos"] = ",".join(sorted(d.metodo.unique()))
        r["reps"] = d.rep.nunique()
        por = d.groupby(["seed", "test_sub"]).size()
        r["filas_por_pliegue"] = ",".join(str(x) for x in sorted(por.unique()))
        r["pliegues_fewshot"] = len(por)
    # NaN
    cols = [c for c in METRICAS + ["acc_reservado"] if c in d.columns]
    nn = int(d[cols].isna().sum().sum())
    r["nan_metricas"] = nn
    if nn:
        notas.append(f"{nn} NaN en metricas")
    # cfg_hash
    hs = sorted(d.cfg_hash.unique())
    r["cfg_hash"] = ",".join(hs)
    # esperado: previo vs avance_7
    esperados = set()
    for reg, g in d.groupby("regimen"):
        if reg == "previo":
            if nombre.startswith(("within", "loso")):
                esperados.add(HASH_PREVIO[arq])
            else:
                esperados.add(hash_esperado(arq))     # punto6 'all': codigo con grad_clip
        else:
            esperados.add(hash_esperado(arq, aug_de(nombre)))
    r["cfg_hash_esperado"] = ",".join(sorted(esperados))
    r["cfg_hash_ok"] = set(hs) == esperados
    if not r["cfg_hash_ok"]:
        notas.append(f"cfg_hash {hs} != esperado {sorted(esperados)}")
    # n_epochs
    if "n_epochs" in d.columns and kind != "fewshot":
        e = d.n_epochs
        r["ep_min"], r["ep_mediana"], r["ep_max"] = int(e.min()), float(e.median()), int(e.max())
        r["ep_<=25"] = int((e <= UMBRAL_PARADA).sum())
        r["ep_<=25_pct"] = round(100 * (e <= UMBRAL_PARADA).mean(), 2)
        r["ep_=200"] = int((e >= 200).sum())
    if "elapsed_s" in d.columns and kind != "fewshot":
        r["elapsed_mediana_s"] = float(d.elapsed_s.median())
    # exactitud media por semilla (descriptiva)
    if "acc" in d.columns:
        g = d.groupby("seed").acc.mean().round(4)
        r["acc_media_por_semilla"] = ";".join(f"{s}:{v}" for s, v in g.items())
    # filas esperadas
    if filas_esp is not None and len(d) != filas_esp:
        notas.append(f"filas {len(d)} != {filas_esp}")
    if set(d.seed.unique()) != set(sem_esp):
        notas.append("semillas distintas de las esperadas")
    # sujetos / pliegues esperados por tipo
    if kind in ("within", "abl"):
        ok = (d.groupby("seed")["sub"].nunique() == 54).all() and \
             (d.groupby(["seed", "sub"]).size() == 20).all()
        r["diseno_ok"] = bool(ok)
        if not ok:
            notas.append("diseno within incompleto (54 sujetos x 20 por semilla)")
    elif kind == "loso":
        ok = (d.groupby("seed")["test_sub"].nunique() == 54).all() and \
             (d.groupby("seed").size() == 54).all()
        r["diseno_ok"] = bool(ok)
        if not ok:
            notas.append("diseno LOSO incompleto")
    elif kind == "fewshot":
        ok = (d.groupby("seed")["test_sub"].nunique() == 54).all()
        r["diseno_ok"] = bool(ok)
        if not ok:
            notas.append("few-shot sin los 54 pliegues")
    # cruzado / holdout: sujetos por politica
    if "norm_destino" in d.columns:
        por = d.groupby(["seed", "norm_destino"])["sub"].nunique()
        r["sujetos_por_politica"] = ",".join(str(x) for x in sorted(por.unique()))
        r["politicas"] = ",".join(sorted(d.norm_destino.unique()))
        if d.duplicated(["seed", "sub", "norm_destino"]).any():
            notas.append("duplicados (seed, sub, norm_destino)")
        r["f1_cero"] = int((d.f1_macro == 0).sum())
        r["n_trials_min_max"] = f"{d.n_trials.min()}-{d.n_trials.max()}"
    r["notas"] = "; ".join(notas)
    r["estado"] = "OK" if not notas else "ANOMALIA"
    return r, d


def qa_fragmentos():
    """Cada consolidado contiene la union de sus fragmentos __wN (por run_id y por metricas)."""
    rutas = set()
    for nombre, (f, _, _) in A.CONDICIONES.items():
        for fu in f().fuente.unique():
            rutas.add(fu)
    # tambien curvas y fewshot (misma regla de fragmentos)
    extra = []
    for fu in sorted(rutas):
        if "/reports/openbmi/" in fu and "_all_" not in fu:
            extra.append(fu.replace("/reports/openbmi/", "/reports/openbmi/fewshot/")
                         if "_loso_" in fu else None)
    rutas |= {e for e in extra if e and (RAIZ / e).exists()}
    filas = []
    for fu in sorted(rutas):
        fr = A.fragmentos(fu)
        if not fr:
            continue
        cons = pd.read_csv(RAIZ / fu)
        clave = ["run_id", "k", "rep", "metodo"] if "/fewshot/" in fu else ["run_id"]
        u = pd.concat([pd.read_csv(p) for p in fr], ignore_index=True)
        n_u = len(u)
        u_d = u.drop_duplicates(clave)
        kc = set(map(tuple, cons[clave].values.tolist()))
        ku = set(map(tuple, u_d[clave].values.tolist()))
        # coincidencia de la metrica
        m = cons.merge(u_d, on=clave, suffixes=("", "_f"))
        col = "acc"
        difm = float((m[col] - m[col + "_f"]).abs().max()) if len(m) and col in m else np.nan
        filas.append({"consolidado": fu, "n_fragmentos": len(fr), "filas_fragmentos": n_u,
                      "filas_unicas_fragmentos": len(u_d), "filas_consolidado": len(cons),
                      "en_fragmentos_no_en_consolidado": len(ku - kc),
                      "en_consolidado_no_en_fragmentos": len(kc - ku),
                      "max_dif_acc": difm,
                      "estado": "OK" if (ku == kc and (np.isnan(difm) or difm < 1e-12)) else "ANOMALIA"})
    return pd.DataFrame(filas)


def qa_pareo_y_regimen():
    """(1) Pareo de cada ablacion/EA/augcr con la base de la MISMA semilla (2026): mismas claves
    y mismos n_train/n_val/n_test. (2) Parada temprana: % de pliegues en el suelo de 30
    epocas (warmup 10 + paciencia 20) por condicion, regimen y semilla. (3) Columnas.
    """
    filas, par = [], []
    for arq in A.ARQ:
        b = A.within(arq)
        b26 = b[b.seed == 2026].set_index(["sub", "session", "fold"])
        for v, m in A.ablaciones_disponibles():
            if m != arq:
                continue
            d = A.ablacion(v, arq).set_index(["sub", "session", "fold"])
            comun = b26.index.intersection(d.index)
            dn = int((b26.loc[comun, ["n_train", "n_val", "n_test"]].values !=
                      d.loc[comun, ["n_train", "n_val", "n_test"]].values).any(axis=1).sum())
            par.append({"comparacion": f"abl_{v}_{arq} vs within_base_{arq}_2026",
                        "claves_comunes": len(comun), "claves_solo_base": len(b26) - len(comun),
                        "claves_solo_var": len(d) - len(comun), "filas_con_n_distinto": dn,
                        "run_id_coincide_con_base": int(d.run_id.isin(b26.run_id).sum())})
        lb = A.loso(arq, "base")
        l26 = lb[lb.seed == 2026].set_index("test_sub")
        for v in ("ea", "augcr"):
            d = A.loso(arq, v).set_index("test_sub")
            comun = l26.index.intersection(d.index)
            dn = int((l26.loc[comun, ["n_train", "n_val", "n_test"]].values !=
                      d.loc[comun, ["n_train", "n_val", "n_test"]].values).any(axis=1).sum())
            par.append({"comparacion": f"loso_{v}_{arq} vs loso_base_{arq}_2026",
                        "claves_comunes": len(comun), "claves_solo_base": len(l26) - len(comun),
                        "claves_solo_var": len(d) - len(comun), "filas_con_n_distinto": dn,
                        "run_id_coincide_con_base": int(d.run_id.isin(l26.run_id).sum())})
        for nom in (f"within_base_{arq}", f"loso_base_{arq}"):
            d = A.cargar(nom)
            for (reg, s), g in d.groupby(["regimen", "seed"]):
                filas.append({"condicion": nom, "regimen": reg, "seed": s, "n": len(g),
                              "ep_mediana": g.n_epochs.median(),
                              "pct_en_suelo_30": round(100 * (g.n_epochs <= 30).mean(), 1),
                              "pct_200": round(100 * (g.n_epochs >= 200).mean(), 1),
                              "elapsed_mediana_s": g.elapsed_s.median()})
        for v, m in A.ablaciones_disponibles():
            if m == arq:
                g = A.ablacion(v, arq)
                filas.append({"condicion": f"abl_{v}_{arq}", "regimen": "avance_7", "seed": 2026,
                              "n": len(g), "ep_mediana": g.n_epochs.median(),
                              "pct_en_suelo_30": round(100 * (g.n_epochs <= 30).mean(), 1),
                              "pct_200": round(100 * (g.n_epochs >= 200).mean(), 1),
                              "elapsed_mediana_s": g.elapsed_s.median()})
    return pd.DataFrame(par), pd.DataFrame(filas)


def qa_columnas():
    out = []
    for arq in A.ARQ:
        for nom in (f"within_base_{arq}", f"loso_base_{arq}", f"transf_train54_base_{arq}",
                    f"transf_nativo_base_{arq}", f"transf_h44_base_{arq}",
                    f"transf_cruzado_base_{arq}"):
            d = A.cargar(nom)
            if not (d.regimen == "avance_7").any() or not (d.regimen == "previo").any():
                continue
            cp = set(c for c in pd.read_csv(RAIZ / d[d.regimen == "previo"].fuente.iloc[0]).columns)
            ca = set(c for c in pd.read_csv(RAIZ / d[d.regimen == "avance_7"].fuente.iloc[0]).columns)
            if cp != ca:
                out.append({"condicion": nom, "solo_previo": sorted(cp - ca),
                            "solo_avance_7": sorted(ca - cp)})
    return pd.DataFrame(out)


PATRONES = {"warning": r"warn", "nan": r"\bnan\b", "oom": r"out of memory|OOM|cuda error",
            "traceback": r"Traceback|Error\b|Exception", "reintento": r"reintent|retry|retrying",
            "reanuda": r"reanud|resum", "killed": r"Killed|killed|SIGKILL|died"}


def qa_logs():
    filas, ejemplos = [], {}
    rutas = sorted((RAIZ / "results" / "avance_7" / "logs").glob("*.log")) +         sorted((RAIZ / "avance_7" / "extraido" / "logs").glob("*.log"))
    for p in rutas:
        t = p.read_text(encoding="utf-8", errors="replace")
        lin = t.splitlines()
        r = {"log": p.name, "lineas": len(lin),
             "ejecuciones": sum(1 for l in lin if l.startswith("=== "))}
        for k, pat in PATRONES.items():
            hits = [l for l in lin if re.search(pat, l, re.I if k != "oom" else 0)]
            r[k] = len(hits)
            if hits:
                ejemplos.setdefault((p.name, k), hits[:3])
        filas.append(r)
    return pd.DataFrame(filas), ejemplos


def main():
    filas, dfs = [], {}
    for nombre, (f, fe, se) in A.CONDICIONES.items():
        r, d = qa_condicion(nombre, f, fe, se)
        filas.append(r)
        dfs[nombre] = d
    q = pd.DataFrame(filas)
    q.to_csv(SALIDA / "qa_condiciones.csv", index=False)
    par, ep = qa_pareo_y_regimen()
    par.to_csv(SALIDA / "qa_pareo.csv", index=False)
    ep.to_csv(SALIDA / "qa_parada_temprana.csv", index=False)
    cc = qa_columnas()
    cc.to_csv(SALIDA / "qa_columnas.csv", index=False)
    fr = qa_fragmentos()
    fr.to_csv(SALIDA / "qa_fragmentos.csv", index=False)
    lg, ej = qa_logs()
    lg.to_csv(SALIDA / "qa_logs.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_rows", 500,
                  "display.max_colwidth", 120)
    print("== condiciones con anomalia:")
    print(q[q.estado != "OK"][["condicion", "filas", "filas_esperadas", "cfg_hash",
                                "cfg_hash_esperado", "notas"]].to_string())
    print("\n== resumen n_epochs <=25 / =200 (solo within/loso/abl):")
    cols = ["condicion", "ep_min", "ep_mediana", "ep_max", "ep_<=25", "ep_<=25_pct", "ep_=200",
            "acc_media_por_semilla"]
    print(q[q.condicion.str.match(r"(within|loso|abl)")][cols].to_string())
    print("\n== pareo con la base 2026:")
    print(par.to_string())
    print("\n== parada temprana por regimen y semilla:")
    print(ep.to_string())
    print("\n== columnas distintas previo vs avance_7:")
    print(cc.to_string())
    print("\n== fragmentos:")
    print(fr.to_string())
    print("\n== logs con hallazgos:")
    cl = [c for c in PATRONES]
    print(lg[(lg[cl].sum(axis=1) > 0)].to_string())
    print("\n== ejemplos de lineas:")
    for (n, k), l in sorted(ej.items()):
        print(n, k)
        for x in l:
            print("    ", x[:200])
    est = json.loads((RAIZ / "results" / "avance_7" / "estado.json").read_text(encoding="utf-8"))
    print("\n== estado.json:", len(est), "trabajos; ok =", sum(v["ok"] for v in est.values()),
          "; errores:", {k: v["error"] for k, v in est.items() if not v["ok"]},
          "; horas totales =", round(sum(v["horas"] for v in est.values()), 1))


if __name__ == "__main__":
    main()
