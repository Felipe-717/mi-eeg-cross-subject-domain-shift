#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
figura_curvas.py — curva de perdida por epoca, LOSO frente a intra-sujeto.

Muestra por que el protocolo intra-sujeto satura: con 144 ensayos de entrenamiento la
perdida de entrenamiento se desploma mientras la de validacion apenas se mueve, y la
brecha se abre desde la epoca 5. En LOSO, con 18.360 ensayos, las dos bajan juntas.
Es el argumento de regimen de datos, no de arquitectura.

Lee dos formatos:
  * los `_curvas.csv` que ahora escribe train_baseline.py (run_id, epoch, train_loss, val_loss)
  * los .log antiguos del pod, que es lo unico que sobrevive de la base de 2026

Entorno: usar el env `eeg` de conda.

    ~/anaconda3/envs/eeg/python.exe scripts/figura_curvas.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SALIDA = REPO / "tesis/figuras/curvas_perdida.pdf"

# Fuentes por protocolo: primero el CSV nuevo, y si no existe el .log antiguo.
FUENTES = {
    "Inter-sujeto (LOSO)": [
        REPO / "results/pod_baseline/baseline_eegnet_loso_completo_global_epochs_cache_ica_curvas.csv",
        REPO / "results/pod_baseline/loso.log",
    ],
    "Intra-sujeto": [
        REPO / "results/pod_baseline/baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica_curvas.csv",
        REPO / "results/pod_baseline/within.log",
    ],
}

# Paleta validada, la misma de figuras_resultados.py
AZUL, NARANJA = "#2a78d6", "#eb6834"
TINTA, MUTED, REJILLA, EJE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"

RE_EP = re.compile(r"ep\s+(\d+)/\d+ \| train ([\d.]+) \| val ([\d.]+)")
H = 40                      # horizonte: mas alla quedan pocas curvas vivas y el promedio enganya


def _de_log(ruta: Path) -> list[list[tuple]]:
    """Curvas desde un .log del pod. Una curva nueva empieza cada vez que epoch vuelve a 0."""
    curvas, cur = [], []
    with open(ruta, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            m = RE_EP.search(ln)
            if not m:
                continue
            e, t, v = int(m.group(1)), float(m.group(2)), float(m.group(3))
            if e == 0 and cur:
                curvas.append(cur)
                cur = []
            cur.append((e, t, v))
    if cur:
        curvas.append(cur)
    return curvas


def _de_csv(ruta: Path) -> list[list[tuple]]:
    df = pd.read_csv(ruta)
    return [[(int(r.epoch), float(r.train_loss), float(r.val_loss))
             for r in g.itertuples()] for _, g in df.groupby("run_id", sort=False)]


def cargar(rutas: list[Path]) -> list[list[tuple]]:
    for r in rutas:
        if r.exists():
            return _de_csv(r) if r.suffix == ".csv" else _de_log(r)
    return []


def matrices(curvas: list[list[tuple]]):
    """(train, val) de forma (n_curvas, H) con NaN donde la curva ya termino."""
    tr = np.full((len(curvas), H), np.nan)
    va = np.full((len(curvas), H), np.nan)
    for i, c in enumerate(curvas):
        for e, t, v in c:
            if e < H:
                tr[i, e], va[i, e] = t, v
    return tr, va


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)

    for ax, (titulo, rutas) in zip(axes, FUENTES.items()):
        curvas = cargar(rutas)
        if not curvas:
            ax.text(0.5, 0.5, "sin datos", ha="center", transform=ax.transAxes, color=MUTED)
            continue
        tr, va = matrices(curvas)
        x = np.arange(H)
        for M, color, etq in ((tr, AZUL, "Entrenamiento"), (va, NARANJA, "Validación")):
            med = np.nanmedian(M, axis=0)
            q1 = np.nanpercentile(M, 25, axis=0)
            q3 = np.nanpercentile(M, 75, axis=0)
            ax.plot(x, med, color=color, lw=1.8, label=etq, zorder=3)
            ax.fill_between(x, q1, q3, color=color, alpha=0.18, lw=0, zorder=2)

        # Cuantas curvas siguen activas al final del horizonte. Sin este dato la cola
        # parece tan fiable como el principio, cuando se promedia sobre muchas menos.
        vivas = int(np.sum(~np.isnan(tr[:, H - 1])))
        ax.set_title(titulo, fontsize=10, color=TINTA, pad=14)
        ax.annotate(f"{len(curvas)} ejecuciones, {vivas} activas en la \u00e9poca {H - 1}",
                    xy=(0.5, 1.02), xycoords="axes fraction", ha="center",
                    fontsize=7.5, color=MUTED)
        ax.set_xlabel("\u00c9poca", fontsize=9, color=MUTED)
        ax.grid(True, color=REJILLA, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(EJE)
        ax.tick_params(colors=MUTED, labelsize=8)

    axes[0].set_ylabel("Pérdida", fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=TINTA, loc="upper right")

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA, bbox_inches="tight")
    fig.savefig(SALIDA.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"escrito {SALIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
