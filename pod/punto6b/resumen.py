# -*- coding: utf-8 -*-
"""Resumen del experimento 44 + 10.

Las tres cifras de cada fila salen del MISMO modelo, con los mismos pesos, evaluado sobre
tres conjuntos distintos. Por eso las diferencias entre columnas son restables: no mezclan
regimenes de entrenamiento. Comparar contra el promedio LOSO, en cambio, no lo seria.

Se reporta la politica `source` (los estadisticos del entrenamiento en el origen), que es la
mas estricta y la unica identica en los tres conjuntos.
"""
import csv
import glob
import os
import statistics as st

REP = "/workspace/tesis/results/reports"
OUT = f"{REP}/holdout44"
MODELOS = ["eegnet", "conformer", "labram", "eegmamba"]
POL = "source"


def media(patron):
    fs = glob.glob(patron)
    if not fs:
        return None, None, None
    filas = [r for r in csv.DictReader(open(fs[0], encoding="utf-8"))
             if r["norm_destino"] == POL]
    if not filas:
        return None, None, None
    acc = [float(r["acc"]) for r in filas]
    kap = [float(r["kappa"]) for r in filas]
    return st.mean(acc), (st.stdev(acc) if len(acc) > 1 else 0.0), st.mean(kap)


def visto(mo):
    """Validacion propia del modelo: personas VISTAS, protocolo conocido."""
    fs = [f for f in glob.glob(f"{REP}/openbmi/baseline_{mo}_all_*hold*")
          if "curvas" not in f]
    if not fs:
        return None
    filas = list(csv.DictReader(open(fs[0], encoding="utf-8")))
    return float(filas[-1]["acc"]) if filas else None


print(f"Politica de normalizacion: {POL}. Azar 0.50. Semilla 2026.")
print()
print(f"{'modelo':<10}{'1 visto':>10}{'2 nuevo':>10}{'3 WBCIC':>10}"
      f"{'persona':>10}{'protocolo':>11}")
print(f"{'':<10}{'OpenBMI':>10}{'OpenBMI':>10}{'':>10}{'(1-2)':>10}{'(2-3)':>11}")
print("-" * 61)
for mo in MODELOS:
    v = visto(mo)
    n, _, _ = media(f"{OUT}/holdout_openbmi_{mo}.csv")
    w, _, _ = media(f"{OUT}/holdout_wbcic_{mo}.csv")
    if v is None or n is None or w is None:
        print(f"{mo:<10}{'(incompleto)':>10}")
        continue
    print(f"{mo:<10}{v:>10.3f}{n:>10.3f}{w:>10.3f}{v - n:>10.3f}{n - w:>11.3f}")
print()
print("Columna 'persona': lo que cuesta una persona nueva del MISMO protocolo.")
print("Columna 'protocolo': lo que cuesta ademas cambiar de conjunto de datos.")
print()
print("Recordatorio: una sola semilla (2026) y 10 sujetos apartados, asi que la columna")
print("'persona' se estima sobre 10 personas y no sobre 54. Es una estimacion, no un censo.")
