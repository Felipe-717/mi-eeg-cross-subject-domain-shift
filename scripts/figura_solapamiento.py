#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""figura_solapamiento.py: solapamiento de participantes de bajo rendimiento.

QUE MUESTRA Y POR QUE
    El director pidio un analisis post-hoc del solapamiento entre los participantes que
    quedan bajo el umbral del 70 % en las cuatro arquitecturas, para sostener el argumento
    del segundo limite: que hay participantes cuya senal no es discriminable con
    independencia del modelo.

    Panel (a): matriz de 54 participantes por 4 arquitecturas, coloreada por exactitud
    intra-sujeto y ordenada por la media del participante. El borde marca las celdas bajo
    umbral. Se ve que las columnas coinciden.

    Panel (b): en cuantas arquitecturas cae cada participante bajo umbral, frente a lo que
    cabria esperar si las cuatro fallaran de forma independiente. La referencia de
    independencia conserva las tasas marginales de cada arquitectura, de modo que la unica
    diferencia entre ambas distribuciones es si los fallos estan o no asociados.

POR QUE EL PROTOCOLO INTRA-SUJETO
    El bajo rendimiento es una propiedad de la senal del participante. Medirlo sobre
    sujetos no vistos confundiria dos cosas: que la senal no sea discriminable, y que el
    modelo no haya generalizado a esa persona. Aqui el modelo se entreno con la misma
    persona que evalua.

UMBRAL
    Vidaurre: se considera de bajo rendimiento a quien NO SUPERA el 70 %, de modo que la
    comparacion es `<=`. Hay participantes con exactitud exactamente 0.700.

SEMILLAS (avance 7)
    --semillas mixta   (por defecto, modo anterior) EEGNet y Conformer con las tres semillas del archivo
                       previo y LaBraM y EEGMamba con la semilla 2026: es la figura publicada hasta el avance 6.
    --semillas 3       las cuatro arquitecturas con la media de las tres semillas (2026, 2027 y 2028) por
                       participante, leidas con scripts/avance7_datos.py. Es la figura del avance 7.

Uso:
    ~/anaconda3/envs/eeg/python.exe scripts/figura_solapamiento.py
    ~/anaconda3/envs/eeg/python.exe scripts/figura_solapamiento.py --semillas 3 --salida Overleaf --sin-png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

SALIDA = Path("tesis/figuras")
UMBRAL = 0.70

WITHIN = {
    "EEGNet": "results/pod_baseline/baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica.csv",
    "Conformer": "results/pod_conformer/openbmi/baseline_conformer_within_completo_global_porsesion_epochs_cache_ica.csv",
    "LaBraM": "results/pod_labram/baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica.csv",
    "EEGMamba": "results/pod_eegmamba_flatten/baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica.csv",
}


def exactitudes(modo: str = "mixta") -> pd.DataFrame:
    """Exactitud intra-sujeto promediada por participante, 54 x 4.

    Se promedia por `sub` antes de nada para que cada persona pese lo mismo. Modo "mixta" (anterior):
    EEGNet y el Conformer tienen 3 semillas y LaBraM y EEGMamba 1. Modo "3": las cuatro con las tres
    semillas (2026, 2027 y 2028), via avance7_datos.within.
    """
    if modo == "3":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import avance7_datos as A
        nombre = {"eegnet": "EEGNet", "conformer": "Conformer", "labram": "LaBraM", "eegmamba": "EEGMamba"}
        out = {}
        for m, k in nombre.items():
            d = A.within(m)
            assert sorted(d.seed.unique()) == [2026, 2027, 2028], (m, d.seed.unique())
            out[k] = d.groupby("sub").acc.mean()
        return pd.DataFrame(out)
    return pd.DataFrame({k: pd.read_csv(f).groupby("sub").acc.mean()
                         for k, f in WITHIN.items()})


def esperado_independiente(bajo: pd.DataFrame) -> np.ndarray:
    """Distribucion esperada de k bajo independencia, conservando las marginales.

    Se convoluciona la distribucion de Bernoulli de cada arquitectura con su propia tasa
    de bajo rendimiento. El resultado es el numero esperado de participantes que caerian
    bajo umbral en exactamente k arquitecturas si los fallos no estuvieran asociados.
    """
    dist = np.array([1.0])
    for p in bajo.mean():
        dist = np.convolve(dist, [1 - p, p])
    return dist * len(bajo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", choices=["mixta", "3"], default="mixta",
                    help="mixta = figura anterior; 3 = media de tres semillas en las cuatro arquitecturas")
    ap.add_argument("--salida", default=str(SALIDA), help="carpeta de salida")
    ap.add_argument("--sin-png", action="store_true", help="no escribir el PNG")
    args = ap.parse_args()
    salida = Path(args.salida)
    acc = exactitudes(args.semillas)
    # umbral con la misma tolerancia de coma flotante que verificar_cifras.le_umbral (1e-9)
    bajo = acc.round(9) <= UMBRAL
    orden = acc.mean(axis=1).sort_values().index
    acc, bajo = acc.loc[orden], bajo.loc[orden]

    k = bajo.sum(axis=1)
    obs = np.array([(k == i).sum() for i in range(5)])
    esp = esperado_independiente(bajo)

    fig = plt.figure(figsize=(11.0, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.45, 1.0], wspace=0.28)

    # ------------------------------------------------------------ panel (a)
    ax = fig.add_subplot(gs[0])
    # Escala divergente centrada en el umbral, de modo que el color por si solo separa a
    # quien lo supera de quien no. RdBu es legible tambien en impresion en gris y para
    # daltonismo rojo-verde, que RdYlGn no lo es.
    norma = TwoSlopeNorm(vmin=0.40, vcenter=UMBRAL, vmax=0.90)
    im = ax.imshow(acc.T.to_numpy(), aspect="auto", cmap="RdBu",
                   norm=norma, interpolation="nearest")
    for j, arq in enumerate(acc.columns):
        for i, sub in enumerate(acc.index):
            if bajo.iloc[i, j]:
                ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False,
                                       edgecolor="black", lw=0.7))
    ax.set_yticks(range(len(acc.columns)))
    ax.set_yticklabels(acc.columns, fontsize=8)
    ax.set_xticks(range(0, len(acc.index), 4))
    ax.set_xticklabels([acc.index[i].replace("sub-", "") for i in
                        range(0, len(acc.index), 4)], fontsize=6.5)
    ax.set_xlabel("participantes, ordenados por exactitud media", fontsize=8.5, labelpad=44)
    ax.set_title("(a) Exactitud intra-sujeto por participante. El recuadro marca las "
                 "celdas que no superan el 70 %", fontsize=9, pad=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("exactitud", fontsize=8)
    cb.ax.axhline(UMBRAL, color="black", lw=1.0)

    # Separador tras los participantes que caen bajo umbral con las cuatro, rotulado
    # bajo el eje para no invadir ni el titulo ni la propia matriz.
    corte = int((k == 4).sum())
    ax.axvline(corte - 0.5, color="black", lw=1.6)
    tx = ax.get_xaxis_transform()
    ax.plot([-0.5, corte - 0.5], [-0.17, -0.17], color="black", lw=1.0,
            transform=tx, clip_on=False)
    ax.text((corte - 1) / 2, -0.20,
            f"{corte} participantes no superan el umbral con las cuatro",
            ha="center", va="top", fontsize=7.5, transform=tx)

    # ------------------------------------------------------------ panel (b)
    ax2 = fig.add_subplot(gs[1])
    x = np.arange(5)
    ax2.bar(x - 0.2, obs, width=0.4, color="#2c3e50", label="observado")
    ax2.bar(x + 0.2, esp, width=0.4, color="#bdc3c7", edgecolor="#7f8c8d",
            label="esperado si fueran\nindependientes")
    for xi, v in zip(x, obs):
        ax2.text(xi - 0.2, v + 0.7, str(int(v)), ha="center", fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xlabel("arquitecturas en que el participante\nno supera el umbral",
                   fontsize=10)
    ax2.set_ylabel("participantes", fontsize=10)
    ax2.set_title("(b) Concentración en los extremos", fontsize=10.5)
    ax2.tick_params(labelsize=8.5)
    ax2.legend(fontsize=8, frameon=False, loc="upper left")
    ax2.grid(axis="y", alpha=0.25, lw=0.5)
    ax2.set_axisbelow(True)

    fig.subplots_adjust(left=0.06, right=0.97, top=0.86, bottom=0.30)
    salida.mkdir(parents=True, exist_ok=True)
    for ext in (("pdf",) if args.sin_png else ("pdf", "png")):
        ruta = salida / f"solapamiento_bajo_rendimiento.{ext}"
        fig.savefig(ruta, dpi=200, bbox_inches="tight")
        print(f"escrito {ruta}")

    print("\nDistribucion de k:")
    for i in range(5):
        print(f"  k={i}: observado {obs[i]:2d}, esperado {esp[i]:5.1f}")
    print(f"\nMarginales: {dict(bajo.sum())}")
    print(f"Interseccion de las cuatro: {obs[4]} de {len(acc)} "
          f"({100 * obs[4] / len(acc):.0f} %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
