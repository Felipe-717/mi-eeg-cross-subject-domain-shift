#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""avance7_datos.py: carga de los resultados del avance 7 y de la base previa, por condicion.

Cada funcion devuelve un DataFrame CONSOLIDADO listo para analisis (nunca los fragmentos
`__wN`, que solo se leen en `scripts/avance7_qa.py` para comprobar que el consolidado los
contiene). Las filas de humo (variante `smoke*`) no se cargan jamas: sus archivos no estan
en ninguna de las rutas de abajo y ademas `_cargar` rechaza cualquier fila cuyo `run_id` o
`ckpt_path` contenga "smoke".

Columnas anadidas a todo DataFrame:
    arq        arquitectura (eegnet, conformer, labram, eegmamba)
    condicion  nombre de la condicion (ver CONDICIONES)
    fuente     ruta del CSV de origen, relativa a la raiz del repositorio
    regimen    "previo" (A40, base de septiembre/octubre) o "avance_7" (L40S, noviembre 2026)

RUTAS (relativas a la raiz; {m}=arquitectura, {tag}=sufijo de configuracion del modelo)
    tag: eegnet y conformer ""; labram "_full_uv1_lr0.0001"; eegmamba "_flatten_uv1_lr0.001"

    WITHIN base (1080 filas por semilla = 54 sujetos x 2 sesiones x 10 pliegues)
        previo   eegnet    results/pod_baseline/baseline_eegnet_within_..._porsesion_epochs_cache_ica.csv (2026-2028)
                 conformer results/pod_conformer/openbmi/baseline_conformer_within_..._porsesion_epochs_cache_ica.csv (2026-2028)
                 labram    results/pod_labram/baseline_labram_within_..._porsesion_full_uv1_lr0.0001_epochs_cache_ica.csv (2026)
                 eegmamba  results/pod_eegmamba_flatten/baseline_eegmamba_within_..._porsesion_flatten_uv1_lr0.001_epochs_cache_ica.csv (2026)
        avance_7 (F2, solo labram y eegmamba, semillas 2027 y 2028)
                 results/avance_7/reports/openbmi/baseline_{m}_within_completo_global_porsesion{tag}_epochs_cache_ica.csv

    LOSO base (54 filas por semilla)
        previo   results/pod_baseline/ (eegnet), results/pod_conformer/openbmi/, results/pod_labram/,
                 results/pod_eegmamba/  (semilla 2026; mismas rutas que LOSO en verificar_cifras.py)
        avance_7 (F5 semilla 2027 + F7 semilla 2028, en el MISMO CSV de 108 filas)
                 results/avance_7/reports/openbmi/baseline_{m}_loso_completo_global{tag}_epochs_cache_ica.csv
    LOSO EA (F6, semilla 2026)  ..._loso_completo_global{tag}_epochs_cache_ica_ea.csv
    LOSO augcr (F8, semilla 2026) ..._loso_completo_global{tag}_vaugcr_epochs_cache_ica.csv

    ABLACIONES F4 (within, semilla 2026, 1080 filas), results/avance_7/reports/openbmi/
        mu, beta, mubeta   baseline_{m}_within_completo_global_porsesion{tag}_epochs_cache_ica_{mu|beta|mubeta}.csv
        w02, w03, augnone, augcr, augfft
                           baseline_{m}_within_completo_global_porsesion{tag}_v{variante}_epochs_cache_ica.csv
        eegnet y conformer: las 8; labram y eegmamba: solo augcr, augfft, w02 (F4 recortado)

    FEW-SHOT (LOSO; k, rep, metodo por pliegue), results/avance_7/reports/openbmi/fewshot/
        base: semillas 2027 y 2028 (F5, F7) en baseline_{m}_loso_completo_global{tag}_epochs_cache_ica.csv
        ea:   semilla 2026 (F6) en ..._epochs_cache_ica_ea.csv
        No existe few-shot de la semilla 2026 base (los modelos LOSO base no se conservaron),
        ni de augcr (F8 se lanzo sin --fewshot).

    TRANSFERENCIA / NATIVO / 44+10 (--mode all; una fila por semilla y modelo)
        entrenamiento 54 sujetos de OpenBMI
            previo   results/punto6/reports/openbmi/baseline_{m}_all_comun_wbcic_global{tag}_epochs_cache_ica.csv
            avance_7 results/avance_7/reports/openbmi/baseline_{m}_all_comun_wbcic_global{tag}_epochs_cache_ica[_ea45].csv
        nativo WBCIC (51 sujetos, 20 % reservado; columnas acc_reservado)
            previo   results/punto6/reports/wbcic_2c/baseline_{m}_all_...csv
            avance_7 results/avance_7/reports/wbcic_2c/baseline_{m}_all_...csv[_ea45]
        entrenamiento 44 sujetos (hold10s2026)
            previo   results/punto6b/openbmi/baseline_{m}_all_comun_wbcic_global{tag}_hold10s2026_epochs_cache_ica.csv
            avance_7 results/avance_7/reports/openbmi/..._hold10s2026_epochs_cache_ica[_ea45].csv
        evaluacion cruzada 54 -> WBCIC (153 filas = 51 sujetos x 3 politicas de normalizacion)
            previo   results/punto6/reports/cruzado/cruzado_{m}.csv
            avance_7 results/avance_7/transferencia/{base|ea}/cruzado_{m}_s{semilla}.csv
        44 + 10, evaluacion en los 10 apartados de OpenBMI (30 filas = 10 sujetos x 3 politicas)
            previo   results/punto6b/holdout44/holdout_openbmi_{m}.csv
            avance_7 results/avance_7/transferencia/{base|ea}/holdout_openbmi_{m}_s{semilla}.csv
        44 + 10, evaluacion en WBCIC (153 filas)
            previo   results/punto6b/holdout44/holdout_wbcic_{m}.csv
            avance_7 results/avance_7/transferencia/{base|ea}/holdout_wbcic_{m}_s{semilla}.csv

    Variante "ea" de transferencia: solo semilla 2026 y vista comun de 45 canales (cache `_ea45`).

Uso:
    import sys; sys.path.insert(0, "scripts")
    import avance7_datos as A
    d = A.within("labram")                    # seeds 2026 (previo) + 2027, 2028 (F2)
    d = A.loso("eegnet", "base")              # seeds 2026, 2027, 2028
    d = A.loso("conformer", "ea")             # F6
    d = A.ablacion("w02", "eegmamba")         # F4
    d = A.fewshot("labram", "base")           # k, rep, metodo, acc por pliegue
    d = A.transferencia("cruzado", "eegnet", "base")
    d = A.cargar("loso_ea_conformer")         # por nombre del registro CONDICIONES
    A.CONDICIONES                             # dict nombre -> (funcion, filas esperadas, semillas)
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ARQ = ["eegnet", "conformer", "labram", "eegmamba"]
TAG = {"eegnet": "", "conformer": "", "labram": "_full_uv1_lr0.0001",
       "eegmamba": "_flatten_uv1_lr0.001"}
SEMILLAS_BASE = (2026, 2027, 2028)
N_SUJ = 54
FILAS_WITHIN_SEMILLA = 54 * 2 * 10
A7 = Path("results/avance_7")
A7_OB = A7 / "reports" / "openbmi"
A7_WB = A7 / "reports" / "wbcic_2c"

# Rutas previas (semilla 2026 y, en within de eegnet/conformer, 2026-2028). Las mismas que
# scripts/verificar_cifras.py (diccionarios LOSO y WITHIN) y los de punto6/punto6b.
PREVIO_LOSO = {
    "eegnet": "results/pod_baseline/baseline_eegnet_loso_completo_global_epochs_cache_ica.csv",
    "conformer": "results/pod_conformer/openbmi/baseline_conformer_loso_completo_global_epochs_cache_ica.csv",
    "labram": "results/pod_labram/baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica.csv",
    "eegmamba": "results/pod_eegmamba/baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica.csv",
}
PREVIO_WITHIN = {
    "eegnet": "results/pod_baseline/baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica.csv",
    "conformer": "results/pod_conformer/openbmi/baseline_conformer_within_completo_global_porsesion_epochs_cache_ica.csv",
    "labram": "results/pod_labram/baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica.csv",
    "eegmamba": "results/pod_eegmamba_flatten/baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica.csv",
}
PREVIO_ALL = "results/punto6/reports/openbmi/baseline_{m}_all_comun_wbcic_global{tag}_epochs_cache_ica.csv"
PREVIO_NATIVO = "results/punto6/reports/wbcic_2c/baseline_{m}_all_comun_wbcic_global{tag}_epochs_cache_ica.csv"
PREVIO_H44 = "results/punto6b/openbmi/baseline_{m}_all_comun_wbcic_global{tag}_hold10s2026_epochs_cache_ica.csv"
PREVIO_CRUZADO = "results/punto6/reports/cruzado/cruzado_{m}.csv"
PREVIO_HOLD_OB = "results/punto6b/holdout44/holdout_openbmi_{m}.csv"
PREVIO_HOLD_WB = "results/punto6b/holdout44/holdout_wbcic_{m}.csv"

ABL_CACHE = {"mu": "_mu", "beta": "_beta", "mubeta": "_mubeta"}       # sufijo del cache
ABL_V = ["w02", "w03", "augnone", "augcr", "augfft"]                  # --variante
ABL_TODAS = list(ABL_CACHE) + ABL_V
ABL_GRANDES = ["augcr", "augfft", "w02"]                              # F4 recortado


# ------------------------------------------------------------------ utilidades

def _cargar(ruta, arq, condicion, regimen, **extra) -> pd.DataFrame:
    ruta = str(ruta).replace("\\", "/")
    d = pd.read_csv(RAIZ / ruta)
    for c in ("run_id", "ckpt_path"):
        if c in d.columns and d[c].astype(str).str.contains("smoke", case=False).any():
            raise ValueError(f"{ruta}: filas de humo en {c}")
    d = d.assign(arq=arq, condicion=condicion, fuente=ruta, regimen=regimen, **extra)
    return d


def fragmentos(ruta) -> list[Path]:
    """Fragmentos `__wN` del CSV consolidado `ruta` (solo para control de calidad)."""
    p = RAIZ / str(ruta)
    return sorted(p.parent.glob(p.stem + "__w*.csv"),
                  key=lambda x: int(x.stem.rsplit("__w", 1)[1]))


def _a7(sub, nombre):
    return (sub / nombre).as_posix()


def ruta_within(arq: str, variante: str | None = None) -> str:
    """CSV avance_7 de within: variante None = base (F2), o una de ABL_TODAS (F4)."""
    t = TAG[arq]
    if variante is None:
        n = f"baseline_{arq}_within_completo_global_porsesion{t}_epochs_cache_ica.csv"
    elif variante in ABL_CACHE:
        n = f"baseline_{arq}_within_completo_global_porsesion{t}_epochs_cache_ica{ABL_CACHE[variante]}.csv"
    else:
        n = f"baseline_{arq}_within_completo_global_porsesion{t}_v{variante}_epochs_cache_ica.csv"
    return _a7(A7_OB, n)


def ruta_loso(arq: str, variante: str = "base") -> str:
    t = TAG[arq]
    suf = {"base": ("", ""), "ea": ("", "_ea"), "augcr": ("_vaugcr", "")}[variante]
    return _a7(A7_OB, f"baseline_{arq}_loso_completo_global{t}{suf[0]}_epochs_cache_ica{suf[1]}.csv")


def ruta_fewshot(arq: str, variante: str = "base") -> str:
    return ruta_loso(arq, variante).replace("/openbmi/", "/openbmi/fewshot/")


def ruta_all(arq: str, tipo: str, variante: str = "base") -> str:
    """tipo: 'train54' (OpenBMI), 'nativo' (WBCIC), 'h44' (44 sujetos). variante base|ea."""
    t = TAG[arq]
    hold = "_hold10s2026" if tipo == "h44" else ""
    ea = "_ea45" if variante == "ea" else ""
    carpeta = A7_WB if tipo == "nativo" else A7_OB
    return _a7(carpeta, f"baseline_{arq}_all_comun_wbcic_global{t}{hold}_epochs_cache_ica{ea}.csv")


# ------------------------------------------------------------------ within / LOSO

def within(arq: str) -> pd.DataFrame:
    """Within base de `arq` con todas las semillas disponibles (previa + F2)."""
    cond = f"within_base_{arq}"
    partes = [_cargar(PREVIO_WITHIN[arq], arq, cond, "previo")]
    if arq in ("labram", "eegmamba"):
        partes.append(_cargar(ruta_within(arq), arq, cond, "avance_7"))
    return pd.concat(partes, ignore_index=True)


def loso(arq: str, variante: str = "base") -> pd.DataFrame:
    """LOSO. base: 2026 (previo) + 2027 (F5) + 2028 (F7). ea: 2026 (F6). augcr: 2026 (F8)."""
    cond = f"loso_{variante}_{arq}"
    partes = []
    if variante == "base":
        partes.append(_cargar(PREVIO_LOSO[arq], arq, cond, "previo"))
    partes.append(_cargar(ruta_loso(arq, variante), arq, cond, "avance_7"))
    return pd.concat(partes, ignore_index=True)


def ablacion(variante: str, arq: str) -> pd.DataFrame:
    """Ablacion F4 intra-sujeto, semilla 2026 (compararla con within(arq) de la semilla 2026)."""
    if variante not in ABL_TODAS:
        raise ValueError(variante)
    return _cargar(ruta_within(arq, variante), arq, f"abl_{variante}_{arq}", "avance_7",
                   variante=variante)


def ablaciones_disponibles() -> list[tuple[str, str]]:
    """Pares (variante, arq) realmente ejecutados en F4."""
    out = []
    for v in ABL_TODAS:
        for m in ARQ:
            if m in ("labram", "eegmamba") and v not in ABL_GRANDES:
                continue
            out.append((v, m))
    return out


def fewshot(arq: str, variante: str = "base") -> pd.DataFrame:
    """Calibracion con k ensayos (LOSO). base: semillas 2027 y 2028. ea: semilla 2026."""
    return _cargar(ruta_fewshot(arq, variante), arq, f"fewshot_{variante}_{arq}", "avance_7")


# ------------------------------------------------------------------ transferencia

def _prev_all(plantilla, arq):
    return plantilla.format(m=arq, tag=TAG[arq])


def transferencia(tipo: str, arq: str, variante: str = "base") -> pd.DataFrame:
    """tipo en: train54, nativo, h44 (filas de --mode all) o cruzado, h44_openbmi, h44_wbcic
    (evaluaciones por sujeto y politica de normalizacion).

    variante "base": semilla 2026 (previo) + 2027, 2028 (avance_7). variante "ea": 2026.
    """
    cond = f"transf_{tipo}_{variante}_{arq}"
    partes = []
    if tipo in ("train54", "nativo", "h44"):
        plant = {"train54": PREVIO_ALL, "nativo": PREVIO_NATIVO, "h44": PREVIO_H44}[tipo]
        if variante == "base":
            partes.append(_cargar(_prev_all(plant, arq), arq, cond, "previo"))
        partes.append(_cargar(ruta_all(arq, tipo, variante), arq, cond, "avance_7"))
    else:
        plant = {"cruzado": PREVIO_CRUZADO, "h44_openbmi": PREVIO_HOLD_OB,
                 "h44_wbcic": PREVIO_HOLD_WB}[tipo]
        pref = {"cruzado": "cruzado", "h44_openbmi": "holdout_openbmi",
                "h44_wbcic": "holdout_wbcic"}[tipo]
        if variante == "base":
            partes.append(_cargar(plant.format(m=arq), arq, cond, "previo"))
            semillas = (2027, 2028)
        else:
            semillas = (2026,)
        for s in semillas:
            partes.append(_cargar(_a7(A7 / "transferencia" / variante, f"{pref}_{arq}_s{s}.csv"),
                                  arq, cond, "avance_7"))
    return pd.concat(partes, ignore_index=True)


# ------------------------------------------------------------------ registro

def _reg():
    """CONDICIONES[nombre] = (funcion, filas_esperadas, semillas_esperadas)."""
    R = {}
    for m in ARQ:
        sem = SEMILLAS_BASE
        R[f"within_base_{m}"] = (lambda m=m: within(m), FILAS_WITHIN_SEMILLA * 3, sem)
        R[f"loso_base_{m}"] = (lambda m=m: loso(m, "base"), 54 * 3, sem)
        R[f"loso_ea_{m}"] = (lambda m=m: loso(m, "ea"), 54, (2026,))
        R[f"loso_augcr_{m}"] = (lambda m=m: loso(m, "augcr"), 54, (2026,))
        R[f"fewshot_base_{m}"] = (lambda m=m: fewshot(m, "base"), None, (2027, 2028))
        R[f"fewshot_ea_{m}"] = (lambda m=m: fewshot(m, "ea"), None, (2026,))
        for v, mm in ablaciones_disponibles():
            if mm == m:
                R[f"abl_{v}_{m}"] = (lambda v=v, m=m: ablacion(v, m),
                                     FILAS_WITHIN_SEMILLA, (2026,))
        for tipo, base_n, ea_n in (("train54", 3, 1), ("nativo", 3, 1), ("h44", 3, 1),
                                   ("cruzado", 153 * 3, 153), ("h44_openbmi", 30 * 3, 30),
                                   ("h44_wbcic", 153 * 3, 153)):
            R[f"transf_{tipo}_base_{m}"] = (lambda t=tipo, m=m: transferencia(t, m, "base"),
                                            base_n, sem)
            R[f"transf_{tipo}_ea_{m}"] = (lambda t=tipo, m=m: transferencia(t, m, "ea"),
                                          ea_n, (2026,))
    return R


CONDICIONES = _reg()


def cargar(nombre: str) -> pd.DataFrame:
    return CONDICIONES[nombre][0]()


if __name__ == "__main__":
    for n, (f, filas, sem) in CONDICIONES.items():
        d = f()
        print(f"{n:34s} filas={len(d):5d} esperadas={filas}  semillas={sorted(d.seed.unique())}")
