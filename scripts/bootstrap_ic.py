#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""bootstrap_ic.py — intervalos de confianza por remuestreo, sin entrenar ningun modelo.

QUE HACE Y POR QUE
    El director propuso remuestrear con reemplazo los valores por sujeto que ya estan
    calculados, para obtener intervalos de confianza de la media sin gastar computo. Eso
    es lo que hace este script sobre tres cantidades:

      1. Exactitud LOSO por arquitectura (54 valores, uno por sujeto).
      2. IDS por arquitectura. Se remuestrean los FOLDS y se recalcula el indicador con la
         formula sobre las medias, igual que en el informe:
             IDS = (media(acc_visto) - media(acc)) / media(acc_visto)
         No se remuestrea el indicador por fold: eso estimaria otra cantidad.
      3. Diferencia pareada entre arquitecturas. El remuestreo es POR SUJETO y comun a las
         dos arquitecturas, de modo que se conserva el emparejamiento; remuestrear cada
         una por separado destruiria la correlacion entre ellas e inflaria el intervalo.

    Tambien la exactitud intra-sujeto, promediada antes por sujeto para que cada persona
    pese lo mismo con independencia de cuantas semillas se le ejecutaron.

METODO
    Percentil simple sobre B remuestreos. Se declara la semilla para que el resultado sea
    reproducible: dos ejecuciones dan exactamente las mismas cifras.

Uso:
    ~/anaconda3/envs/eeg/python.exe scripts/bootstrap_ic.py
    ~/anaconda3/envs/eeg/python.exe scripts/bootstrap_ic.py --b 50000 --csv salida.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA = 2026
B_POR_DEFECTO = 10_000
ALFA = 0.05

LOSO = {
    "EEGNet": "results/pod_baseline/baseline_eegnet_loso_completo_global_epochs_cache_ica.csv",
    "Conformer": "results/pod_conformer/openbmi/baseline_conformer_loso_completo_global_epochs_cache_ica.csv",
    "LaBraM": "results/pod_labram/baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica.csv",
    "EEGMamba": "results/pod_eegmamba/baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica.csv",
}

WITHIN = {
    "EEGNet": "results/pod_baseline/baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica.csv",
    "Conformer": "results/pod_conformer/openbmi/baseline_conformer_within_completo_global_porsesion_epochs_cache_ica.csv",
    "LaBraM": "results/pod_labram/baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica.csv",
    "EEGMamba": "results/pod_eegmamba_flatten/baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica.csv",
}


def ic(muestras: np.ndarray) -> tuple[float, float]:
    """Intervalo de percentil al 95 %."""
    return (float(np.quantile(muestras, ALFA / 2)),
            float(np.quantile(muestras, 1 - ALFA / 2)))


def indices(rng: np.random.Generator, n: int, b: int) -> np.ndarray:
    """Matriz b x n de indices remuestreados con reemplazo."""
    return rng.integers(0, n, size=(b, n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=B_POR_DEFECTO, help="numero de remuestreos")
    ap.add_argument("--csv", type=Path, default=None, help="guardar la tabla en este fichero")
    args = ap.parse_args()

    rng = np.random.default_rng(SEMILLA)
    filas: list[dict] = []

    # ------------------------------------------------------ exactitud LOSO
    accs: dict[str, np.ndarray] = {}
    vistos: dict[str, np.ndarray] = {}
    print(f"Exactitud LOSO. B = {args.b}, semilla {SEMILLA}, percentil al 95 %\n")
    print(f"{'arquitectura':12s} {'media':>7s} {'IC 95 %':>18s} {'amplitud':>9s}")
    for arq, ruta in LOSO.items():
        d = pd.read_csv(ruta).sort_values("test_sub")
        a = d.groupby("test_sub").acc.mean().to_numpy()
        accs[arq] = a
        vistos[arq] = d.groupby("test_sub").acc_visto.mean().to_numpy()
        idx = indices(rng, len(a), args.b)
        lo, hi = ic(a[idx].mean(axis=1))
        print(f"{arq:12s} {a.mean():7.3f} [{lo:.3f}, {hi:.3f}] {hi - lo:9.3f}")
        filas.append({"cantidad": "acc_loso", "arquitectura": arq, "n": len(a),
                      "estimacion": a.mean(), "ic_inf": lo, "ic_sup": hi})

    # ------------------------------------------------------------- IDS
    print("\nIDS. Se remuestrean los folds y se recalcula la formula sobre las medias\n")
    print(f"{'arquitectura':12s} {'IDS':>7s} {'IC 95 %':>18s} {'cruza 0':>8s}")
    for arq in LOSO:
        v, a = vistos[arq], accs[arq]
        idx = indices(rng, len(a), args.b)
        mv, ma = v[idx].mean(axis=1), a[idx].mean(axis=1)
        lo, hi = ic((mv - ma) / mv)
        est = (v.mean() - a.mean()) / v.mean()
        print(f"{arq:12s} {est:+7.3f} [{lo:+.3f}, {hi:+.3f}] {'si' if lo <= 0 <= hi else 'NO':>8s}")
        filas.append({"cantidad": "ids", "arquitectura": arq, "n": len(a),
                      "estimacion": est, "ic_inf": lo, "ic_sup": hi})

    # ------------------------------------ diferencias pareadas entre arquitecturas
    print("\nDiferencia de exactitud LOSO entre arquitecturas, pareada por sujeto.")
    print("El remuestreo de sujetos es COMUN a las dos, para conservar el emparejamiento\n")
    print(f"{'par':24s} {'dif':>7s} {'IC 95 %':>18s} {'cruza 0':>8s}")
    nombres = list(LOSO)
    for i, x in enumerate(nombres):
        for y in nombres[i + 1:]:
            dif = accs[x] - accs[y]
            idx = indices(rng, len(dif), args.b)
            lo, hi = ic(dif[idx].mean(axis=1))
            print(f"{x + ' - ' + y:24s} {dif.mean():+7.3f} [{lo:+.3f}, {hi:+.3f}] "
                  f"{'si' if lo <= 0 <= hi else 'NO':>8s}")
            filas.append({"cantidad": "dif_loso", "arquitectura": f"{x}-{y}", "n": len(dif),
                          "estimacion": dif.mean(), "ic_inf": lo, "ic_sup": hi})

    # -------------------------------------------------- exactitud intra-sujeto
    print("\nExactitud intra-sujeto, promediada antes por sujeto\n")
    print(f"{'arquitectura':12s} {'media':>7s} {'IC 95 %':>18s}")
    for arq, ruta in WITHIN.items():
        d = pd.read_csv(ruta)
        a = d.groupby("sub").acc.mean().sort_index().to_numpy()
        idx = indices(rng, len(a), args.b)
        lo, hi = ic(a[idx].mean(axis=1))
        print(f"{arq:12s} {a.mean():7.3f} [{lo:.3f}, {hi:.3f}]")
        filas.append({"cantidad": "acc_within", "arquitectura": arq, "n": len(a),
                      "estimacion": a.mean(), "ic_inf": lo, "ic_sup": hi})

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(filas).to_csv(args.csv, index=False)
        print(f"\nescrito {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
