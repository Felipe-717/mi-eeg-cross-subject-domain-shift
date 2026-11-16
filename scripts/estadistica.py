#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
estadistica.py — Wilcoxon pareado con TAMANO DE EFECTO r entre las 4 arquitecturas.

Responde al comentario del director: reportar r junto al valor p, para distinguir una
diferencia no significativa que es realmente pequena de una que solo carece de potencia.

Emparejamiento: por SUJETO. En LOSO cada arquitectura tiene una medida por sujeto excluido
(54 folds); en within se promedian los folds y sesiones de cada sujeto. Es la misma particion
para todas las arquitecturas, asi que el contraste es pareado y n = 54 en ambos protocolos.

    r = |Z| / sqrt(n)     (Rosenthal)

con Z reconstruido desde el p bilateral exacto de scipy. Referencia de Cohen: 0.1 pequeno,
0.3 mediano, 0.5 grande.

Entorno: usar el env `eeg` de conda, que trae scipy/numpy/pandas.

    ~/anaconda3/envs/eeg/python.exe scripts/estadistica.py
    ~/anaconda3/envs/eeg/python.exe scripts/estadistica.py --metrica kappa
"""
from __future__ import annotations

import argparse
import os
from itertools import combinations

import pandas as pd
from scipy import stats
from ruido_flotante import wilcoxon_pareado  # Wilcoxon sobre diferencias redondeadas a 1e-9

SEMILLA = 2026            # unica semilla presente en LOSO para las 4 arquitecturas

# Un fichero por arquitectura y protocolo. Explicitos y no por glob porque en results/
# conviven variantes descartadas (EEGMamba con cabeza avgpool, LaBraM linear, ficheros
# parciales de 16 sujetos) que no deben entrar en la comparacion.
FUENTES = {
    "loso": {
        "EEGNet":    "results/pod_baseline/baseline_eegnet_loso_completo_global_epochs_cache_ica.csv",
        "Conformer": "results/pod_conformer/openbmi/baseline_conformer_loso_completo_global_epochs_cache_ica.csv",
        "LaBraM":    "results/pod_labram/baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica.csv",
        "EEGMamba":  "results/pod_eegmamba/baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica.csv",
    },
    "within": {
        "EEGNet":    "results/pod_baseline/baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica.csv",
        "Conformer": "results/pod_conformer/openbmi/baseline_conformer_within_completo_global_porsesion_epochs_cache_ica.csv",
        "LaBraM":    "results/pod_labram/baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica.csv",
        "EEGMamba":  "results/pod_eegmamba_flatten/baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica.csv",
        # Se usa la variante `flatten`, no la `avgpool`: aquella quedo en azar (0.519) porque
        # el promediado global elimina la lateralizacion. Con `flatten` sube a 0.577, por
        # encima del azar (p = 1.5e-08) y por encima de `avgpool` con efecto grande (r = 0.64).
    },
}


def cargar(ruta: str, metrica: str) -> pd.Series:
    """Serie {sujeto: metrica} de la semilla base, promediando folds y sesiones del sujeto."""
    if not os.path.exists(ruta):
        return pd.Series(dtype=float)
    df = pd.read_csv(ruta)
    df = df[df["seed"] == SEMILLA]
    col = "test_sub" if "test_sub" in df.columns else "sub"
    return df.groupby(col)[metrica].mean()


def magnitud(r: float) -> str:
    return "grande" if r >= 0.5 else "mediano" if r >= 0.3 else "pequeno" if r >= 0.1 else "trivial"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrica", default="acc", choices=["acc", "kappa", "f1_macro"])
    a = ap.parse_args()

    for protocolo, rutas in FUENTES.items():
        datos = {m: s for m, p in rutas.items() if not (s := cargar(p, a.metrica)).empty}
        if len(datos) < 2:
            print(f"\n### {protocolo.upper()}: datos insuficientes")
            continue

        print(f"\n{'=' * 76}")
        print(f" {protocolo.upper()}  |  metrica: {a.metrica}  |  semilla {SEMILLA}")
        print(f"{'=' * 76}")
        print(f"{'modelo':<11} {'media':>7} {'sd':>7} {'mediana':>8} {'sujetos':>8}")
        for m, s in datos.items():
            print(f"{m:<11} {s.mean():>7.4f} {s.std():>7.4f} {s.median():>8.4f} {len(s):>8}")

        print(f"\n{'comparacion':<24} {'dif. mediana':>13} {'p':>10} {'r':>7}  magnitud")
        print("-" * 72)
        for x, y in combinations(datos, 2):
            par = pd.concat([datos[x], datos[y]], axis=1, join="inner", keys=["a", "b"]).dropna()
            # `zsplit` reparte los empates exactos en vez de descartarlos, que es lo correcto
            # cuando se quiere un p conservador; scipy elige el metodo exacto o normal segun n.
            res = wilcoxon_pareado(par["a"], par["b"], zero_method="zsplit", alternative="two-sided")
            n = len(par)
            # r desde el p bilateral: z = Phi^-1(1 - p/2). Es equivalente a |Z|/sqrt(n) y no
            # depende de si scipy uso la via exacta o la normal.
            z = abs(stats.norm.ppf(res.pvalue / 2)) if res.pvalue > 0 else abs(stats.norm.ppf(1e-16))
            r = z / (n ** 0.5)
            dm = (par["a"] - par["b"]).median()
            sig = "*" if res.pvalue < 0.05 else " "
            print(f"{x + ' vs ' + y:<24} {dm:>+13.4f} {res.pvalue:>9.2e}{sig} "
                  f"{r:>7.3f}  {magnitud(r)}   (n={n})")
        print("\n  * p < 0.05.  r: 0.1 pequeno, 0.3 mediano, 0.5 grande.")
        print("  La diferencia mediana es del primer modelo menos el segundo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
