# -*- coding: utf-8 -*-
"""Resumen del punto 6: transferencia de disparo cero frente al techo nativo, ambos
medidos sobre LOS MISMOS segmentos (el 20% reservado de WBCIC).

El CSV de eval_cruzado.py trae una fila por sujeto y politica, asi que hay que promediar
por politica; imprimirlo crudo serian 153 lineas por arquitectura."""
import csv, glob, os, statistics as st

CRUZ = "/workspace/tesis/results/reports/cruzado/cruzado_{}.csv"
NAT = "/workspace/tesis/results/reports/wbcic_2c/baseline_{}_all_*.csv"
MODELOS = ["eegnet", "conformer", "labram", "eegmamba"]

def nativo(mo):
    fs = [f for f in glob.glob(NAT.format(mo)) if "curvas" not in f]
    if not fs:
        return None
    filas = list(csv.DictReader(open(fs[0], encoding="utf-8")))
    return filas[-1] if filas else None

print(f"{'modelo':<10} {'politica':<12} {'acc':>8} {'sd':>7} {'kappa':>8}   (azar 0.50)")
print("-" * 60)
for mo in MODELOS:
    f = CRUZ.format(mo)
    if os.path.exists(f):
        filas = list(csv.DictReader(open(f, encoding="utf-8")))
        pols = {}
        for r in filas:
            pols.setdefault(r["norm_destino"], []).append((float(r["acc"]), float(r["kappa"])))
        for pol, v in pols.items():
            accs = [a for a, _ in v]
            sd = st.stdev(accs) if len(accs) > 1 else 0.0
            print(f"{mo:<10} {pol:<12} {st.mean(accs):>8.4f} {sd:>7.4f} "
                  f"{st.mean([k for _, k in v]):>8.4f}")
    else:
        print(f"{mo:<10} {'(sin cruzado)':<12}")
    r = nativo(mo)
    if r:
        print(f"{mo:<10} {'NATIVO':<12} {float(r['acc_reservado']):>8.4f} {'':>7} "
              f"{float(r['kappa_reservado']):>8.4f}   <- techo, sujetos conocidos")
    print()
