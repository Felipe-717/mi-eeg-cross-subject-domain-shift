#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""figuras_curvas.py — curvas de pérdida de las cuatro arquitecturas, dos paneles.

QUE DIBUJA Y DE DONDE SALE
    Las curvas por época solo existen para los protocolos que se ejecutaron DESPUES de que
    la salida de curvas pasara a ser permanente. Los protocolos intra-sujeto y LOSO son
    anteriores: sus CSV guardan `n_epochs` y las metricas finales, pero no la traza por
    epoca. Por eso los dos paneles son:

      (a) OpenBMI, protocolo global: 19.440 segmentos de entrenamiento, 54 sujetos.
      (b) WBCIC-SHU, modelo nativo:   7.342 segmentos de entrenamiento, 51 sujetos.

    El contraste entre ambos es el del volumen de datos, que es justamente el factor que
    domina la diferencia entre protocolos (seccion sec:within-vs-loso del documento).

    Fuente: results/punto6/reports/{openbmi,wbcic_2c}/curvas/*.csv, columnas
    run_id, epoch, train_loss, val_loss.

Uso:
    ~/anaconda3/envs/eeg/python.exe scripts/figuras_curvas.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path("results/punto6/reports")
SALIDA = Path("tesis/figuras")

PANELES = [
    ("openbmi", "(a) OpenBMI, protocolo global\n19 440 segmentos de entrenamiento"),
    ("wbcic_2c", "(b) WBCIC-SHU, modelo nativo\n7 342 segmentos de entrenamiento"),
]

MODELOS = [("eegnet", "EEGNet", "#1f77b4"),
           ("conformer", "Conformer", "#d62728"),
           ("labram", "LaBraM", "#2ca02c"),
           ("eegmamba", "EEGMamba", "#9467bd")]


def curva(dataset: str, modelo: str) -> pd.DataFrame | None:
    pat = f"{RAIZ}/{dataset}/curvas/baseline_{modelo}_all_*.csv"
    fs = [f for f in glob.glob(pat) if "hold" not in f]
    if not fs:
        return None
    return pd.read_csv(fs[0])


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, (ds, titulo) in zip(axes, PANELES):
        for mo, etq, color in MODELOS:
            d = curva(ds, mo)
            if d is None:
                print(f"  aviso: falta la curva de {mo} en {ds}")
                continue
            ax.plot(d.epoch, d.train_loss, color=color, lw=1.3, label=f"{etq}, entrenamiento")
            ax.plot(d.epoch, d.val_loss, color=color, lw=1.3, ls="--", alpha=0.75,
                    label=f"{etq}, validación")
            # La parada temprana se queda con el minimo de validacion, no con la ultima
            # epoca: marcarlo evita leer el final de la curva como el modelo elegido.
            i = int(d.val_loss.idxmin())
            ax.plot(d.epoch[i], d.val_loss[i], marker="o", ms=4.5, color=color,
                    mec="white", mew=0.8, zorder=5)
        ax.set_title(titulo, fontsize=9)
        ax.set_xlabel("época")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("pérdida (entropía cruzada)")

    manejadores, etiquetas = axes[0].get_legend_handles_labels()
    fig.legend(manejadores, etiquetas, loc="lower center", ncol=4, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Curvas de pérdida por época. El punto marca el mínimo de validación, "
                 "que es el modelo que conserva la parada temprana.", fontsize=8.5, y=1.0)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))

    SALIDA.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        ruta = SALIDA / f"curvas_perdida.{ext}"
        fig.savefig(ruta, dpi=200, bbox_inches="tight")
        print(f"escrito {ruta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
