"""Reexpande el caché consolidado de Hugging Face a los 54 .npz por sujeto.

POR QUE HACE FALTA
    La fase `export` consolida los 54 sujetos en UN archivo fp16 (2.68 GB) para subirlo
    fuera del pod: un solo fichero viaja mucho mejor que 54 comprimidos. Pero
    `splits.load_subject` —y por tanto todo `train_baseline.py`— espera un `.npz` por
    sujeto en `epochs_cache_ica/`. Este script deshace la consolidación.

    Así una sesión nueva arranca en ~3 min (bajar 2.68 GB + expandir) en vez de ~1 h
    (descargar 66 GB de crudo + ICA), y sin depender de que MOABB y GigaDB respondan.

QUE CAMBIA RESPECTO AL CACHE ORIGINAL (y por que da igual)
    * dtype: el consolidado es fp16; aquí se reescribe en float32, que es lo que espera el
      resto del código. La pérdida de fp16 se midió: la log-varianza por canal se desvía
      3.5e-04 y los estadísticos de z-score ~1e-04. El z-score se recalcula por fold de
      todos modos y el entrenamiento usa AMP en fp16.
    * ventana: el consolidado va de -0.5 a 4.5 s en vez de -2 a 5 s. La ventana de
      modelado (0.0-4.0 s) queda dentro con medio segundo de margen a cada lado, que es
      justo lo que permite desplazarla (p. ej. 0.5-4.0 s para excluir la sacada del cue)
      sin reprocesar.

Uso:
  python pod/expandir_cache.py                                   # rutas por defecto
  python pod/expandir_cache.py --src /workspace/cache/openbmi_ica_fp16.npz
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    raiz = Path(os.environ.get("OPENBMI_PROC_ROOT", "D:/Tesis/data/processed/openbmi"))
    ap.add_argument("--src", default=str(raiz / "openbmi_ica_fp16.npz"))
    ap.add_argument("--dest", default=str(raiz / "epochs_cache_ica"))
    ap.add_argument("--forzar", action="store_true",
                    help="reescribe los .npz aunque ya existan")
    args = ap.parse_args()

    src, dest = Path(args.src), Path(args.dest)
    if not src.exists():
        sys.exit(f"No existe {src}\n"
                 f"Bajalo antes:  hf download Felipe717/openbmi-ica "
                 f"--repo-type dataset --local-dir {src.parent}")
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo {src} ({src.stat().st_size/1e9:.2f} GB)...", flush=True)
    d = np.load(src, allow_pickle=False)
    X, y, ses, run = d["X"], d["y"], d["session"], d["run"]
    subs = [str(s) for s in d["subjects"]]
    chs, sfreq, tmin = d["ch_names"], float(d["sfreq"]), float(d["tmin"])
    cfg_hash = str(d["cfg_hash"])
    print(f"  {X.shape} {X.dtype} | sfreq {sfreq} | tmin {tmin} | cfg {cfg_hash}")

    n_escritos = n_saltados = 0
    for i, sub in enumerate(subs):
        f = dest / f"{sub}_full.npz"
        if f.exists() and not args.forzar:
            n_saltados += 1
            continue
        np.savez_compressed(
            f,
            # a float32: es lo que espera el resto del codigo, y evita que numpy acumule
            # en fp16 al calcular medias y desviaciones de la normalizacion
            X=X[i].astype(np.float32), y=y[i], session=ses[i], run=run[i],
            ch_names=chs, sfreq=sfreq, tmin=tmin, subject=sub, cfg_hash=cfg_hash,
        )
        n_escritos += 1
        if n_escritos % 10 == 0:
            print(f"  {n_escritos} escritos...", flush=True)

    print(f"\n{n_escritos} escritos, {n_saltados} ya estaban -> {dest}")

    # --- verificacion: lo que espera train_baseline ---
    print("\n=== verificacion ===")
    fs = sorted(dest.glob("sub-*_full.npz"))
    print(f"  archivos: {len(fs)}")
    malos = []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        Xi, yi = z["X"], z["y"]
        if Xi.shape[0] != 400 or Xi.shape[1] != 62 or not np.isfinite(Xi).all():
            malos.append(f"{f.name}: forma {Xi.shape}")
        elif sorted(np.unique(yi, return_counts=True)[1].tolist()) != [200, 200]:
            malos.append(f"{f.name}: desbalance")
    print(f"  {'PROBLEMAS: ' + str(malos) if malos else f'ok — {len(fs)} x 400 x 62, balanceados, sin NaN'}")
    if not malos and len(fs) != 54:
        print(f"  AVISO: hay {len(fs)} sujetos, se esperaban 54")

    # La ventana de modelado tiene que poder recortarse: es lo que hace load_subject.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import splits as sp                      # noqa: E402
    import datasets.openbmi as DS            # noqa: E402
    Xw, yw, cw = sp.load_subject(fs[0].name.replace("_full.npz", ""), dest,
                                 window=DS.WINDOW, ch_names=None)
    print(f"  recorte a la ventana de modelado {DS.WINDOW}: {Xw.shape} ({len(cw)} canales)")
    if Xw.shape[2] != int((DS.WINDOW[1] - DS.WINDOW[0]) * DS.SFREQ) + 1:
        sys.exit("  FALLO: el recorte no da las muestras esperadas")
    print("\nListo para entrenar.")


if __name__ == "__main__":
    main()
