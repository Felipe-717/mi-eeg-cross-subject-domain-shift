#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
expandir.py — deshace el consolidado de `exportar.py` en un .npz por sujeto.

`splits.load_subject`, y por tanto todo `train_baseline.py`, espera un fichero por sujeto
en el directorio de cache. Este script reconstruye esa forma en el pod tras bajar el
consolidado de Hugging Face, igual que `pod/expandir_cache.py` hace con OpenBMI.

El fp16 del consolidado se reescribe en float32, que es lo que espera el resto del codigo.

Uso, dentro del pod:
    hf download Felipe717/wbcic-2c-ica --repo-type dataset --local-dir /workspace/cache_wbcic
    python scripts/wbcic/expandir.py --src /workspace/cache_wbcic/wbcic_2c_ica_fp16.npz \\
                                     --dest /workspace/cache_wbcic/epochs_cache_ica
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path,
                    default=Path("data/processed/wbcic/wbcic_2c_ica_fp16.npz"))
    ap.add_argument("--dest", type=Path,
                    default=Path("data/processed/wbcic/epochs_cache_ica"))
    ap.add_argument("--forzar", action="store_true", help="reescribe los .npz existentes")
    a = ap.parse_args()

    if not a.src.exists():
        raise SystemExit(f"x No existe {a.src}\n"
                         f"  Bajalo antes:  hf download <repo> --repo-type dataset "
                         f"--local-dir {a.src.parent}")
    a.dest.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo {a.src} ({a.src.stat().st_size / 1e9:.2f} GB)...", flush=True)
    d = np.load(a.src, allow_pickle=False)
    X, y, ses, sid = d["X"], d["y"], d["session"], d["subject"]
    chs, sf, tmin = d["ch_names"], float(d["sfreq"]), float(d["tmin"])
    print(f"  {X.shape} {X.dtype} | sfreq {sf} | tmin {tmin} | cfg {str(d['cfg_hash'])}")

    n_esc = n_sal = 0
    for sub in sorted(set(str(s) for s in sid)):
        salida = a.dest / f"{sub}_full.npz"
        if salida.exists() and not a.forzar:
            n_sal += 1
            continue
        m = sid == sub
        np.savez_compressed(salida, X=X[m].astype(np.float32), y=y[m],
                            ch_names=chs, sfreq=sf, tmin=tmin, session=ses[m])
        n_esc += 1
        print(f"  {sub}: {int(m.sum())} trials", flush=True)

    print(f"\n{n_esc} escritos, {n_sal} ya estaban -> {a.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
