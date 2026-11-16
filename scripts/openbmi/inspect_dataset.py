"""PASO 1 — Inspección de OpenBMI/Lee2019 vía MOABB, ANTES de cachear nada.

Por qué existe este script: el diseño del caché depende de cómo MOABB exponga las
2 sesiones y las fases offline/online, de los nombres REALES de los canales y del
mapeo de etiquetas. Todo eso está ahora mismo SUPUESTO en `src/datasets/openbmi.py`
(los nombres de `CH_SETS` vienen del montaje 10-20 estándar, no de los datos).
Cachear sobre suposiciones es como se pierden días.

Descarga UN sujeto (MOABB lo guarda en ~/mne_data) y reporta la estructura.

Uso:  python scripts/openbmi/inspect_dataset.py [--subject 1]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import datasets

import mne
mne.set_log_level("ERROR")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", type=int, default=1)
    args = ap.parse_args()

    ds_cfg = datasets.get("openbmi")
    from moabb.datasets import Lee2019_MI

    print("=" * 78)
    print(f"INSPECCIÓN OpenBMI / Lee2019_MI — sujeto {args.subject}")
    print("=" * 78)

    ds = Lee2019_MI()
    print(f"\nSujetos declarados por MOABB: {len(ds.subject_list)} -> {ds.subject_list[:8]}...")
    print(f"event_id de MOABB: {getattr(ds, 'event_id', 'n/d')}")
    print(f"intervalo: {getattr(ds, 'interval', 'n/d')}  |  código: {getattr(ds, 'code', 'n/d')}")

    print("\nDescargando (la primera vez tarda)...", flush=True)
    data = ds.get_data(subjects=[args.subject])[args.subject]

    # ---------------- estructura sesión / run ----------------
    print("\n--- ESTRUCTURA sesión -> run ---")
    for ses, runs in data.items():
        print(f"  sesión {ses!r}: runs = {list(runs)}")

    ses0 = list(data)[0]
    run0 = list(data[ses0])[0]
    raw = data[ses0][run0]

    # ---------------- canales ----------------
    print(f"\n--- CANALES (sesión {ses0!r}, run {run0!r}) ---")
    print(f"  n = {len(raw.ch_names)}  |  sfreq = {raw.info['sfreq']} Hz  "
          f"|  duración = {raw.n_times / raw.info['sfreq']:.0f} s")
    print(f"  tipos: {set(raw.get_channel_types())}")
    print(f"  nombres: {raw.ch_names}")

    # ¿coinciden con lo que asumimos en src/datasets/openbmi.py?
    print("\n--- ¿COINCIDEN NUESTRAS SUPOSICIONES? ---")
    reales = {c.upper() for c in raw.ch_names}
    for nombre, lista in ds_cfg.CH_SETS.items():
        if lista is None:
            continue
        faltan = [c for c in lista if c.upper() not in reales]
        estado = "OK" if not faltan else f"FALTAN {faltan}"
        print(f"  {nombre:14s} ({len(lista):2d} canales): {estado}")

    # ---------------- eventos ----------------
    print("\n--- EVENTOS / ANOTACIONES ---")
    if raw.annotations is not None and len(raw.annotations):
        vals, cnt = np.unique(raw.annotations.description, return_counts=True)
        for v, c in zip(vals, cnt):
            print(f"  anotación {v!r}: {c}")
    try:
        ev, ev_id = mne.events_from_annotations(raw, verbose="ERROR")
        print(f"  events_from_annotations -> event_id = {ev_id}, n_eventos = {len(ev)}")
    except Exception as e:
        print(f"  events_from_annotations falló: {type(e).__name__}: {e}")

    # ---------------- paradigma (la vía recomendada) ----------------
    print("\n--- VÍA PARADIGMA (LeftRightImagery) ---")
    try:
        from moabb.paradigms import LeftRightImagery
        # sin filtrar ni remuestrear aquí: solo para ver forma, etiquetas y metadatos
        par = LeftRightImagery(fmin=ds_cfg.AUTHORS_CFG["l_freq"],
                               fmax=ds_cfg.AUTHORS_CFG["h_freq"],
                               resample=ds_cfg.SFREQ)
        X, y, meta = par.get_data(dataset=ds, subjects=[args.subject])
        print(f"  X = {X.shape}   (trials, canales, muestras)")
        print(f"  y: {dict(zip(*np.unique(y, return_counts=True)))}")
        print(f"  columnas de metadata: {list(meta.columns)}")
        for col in meta.columns:
            u = meta[col].unique()
            print(f"    {col}: {u[:6]}{'...' if len(u) > 6 else ''}  ({len(u)} únicos)")
        print(f"\n  trials por sesión: {meta.groupby('session').size().to_dict()}")
        if "run" in meta.columns:
            print(f"  trials por (sesión, run): {meta.groupby(['session','run']).size().to_dict()}")
    except Exception as e:
        print(f"  FALLÓ: {type(e).__name__}: {e}")

    print("\n" + "=" * 78)
    print("QUÉ MIRAR:")
    print("  1. ¿Los 20 canales de 'motor_paper' existen? -> si no, corregir CH_SETS")
    print("  2. ¿Cuántas sesiones y qué runs? -> los runs suelen ser offline/online")
    print("  3. ¿400 trials por sujeto (200 por sesión) y balanceados?")
    print("  4. ¿Qué nombres tienen las clases? -> ajustar CLASS_ORDER si no es")
    print(f"     {ds_cfg.CLASS_ORDER}")
    print("=" * 78)


if __name__ == "__main__":
    main()
