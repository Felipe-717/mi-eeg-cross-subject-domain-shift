"""Valida que el pipeline en GPU da los MISMOS resultados que el de CPU.

CONTEXTO (2026-07-27). Para bajar el coste de la Fase 3 se cambiaron dos cosas del
entrenamiento:
  * la augmentation pasó de numpy/CPU (float64, bucles por trial) a torch/GPU (float32,
    vectorizada)  -> 7.0x
  * el lote fisico de 32 a 256, con la lr escalada por sqrt(8)                 -> 1.5x mas
Total 10.6x. Pero ambos cambios ALTERAN los numeros: distinta precision en la FFT,
distinto generador aleatorio y distinta dinamica de optimizacion.

Este script compara, sujeto a sujeto y con la MISMA semilla, el within-subject obtenido
con cada pipeline. Es una comparacion PAREADA: lo que importa no es que cada sujeto
coincida (el ruido de inicializacion ya movia +-0.08 entre semillas) sino que no haya
un sesgo SISTEMATICO en una direccion.

Uso (en local, tras traerse el CSV del pod):
  python pod/validar_pipeline.py --nuevo results/pod/baseline_eegnet_within_*_ica.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
REF = REPO / "results" / "reports" / "openbmi" / \
    "baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica.csv"
SEED = 2026


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nuevo", required=True, help="CSV producido con el pipeline en GPU")
    ap.add_argument("--ref", default=str(REF), help="CSV de referencia (pipeline en CPU)")
    args = ap.parse_args()

    agg = lambda f: (pd.read_csv(f).query("seed == @SEED")
                     .groupby("sub").agg(acc=("acc", "mean"), kappa=("kappa", "mean"),
                                         ep=("n_epochs", "mean"), n=("acc", "size")))
    ref, nue = agg(args.ref), agg(args.nuevo)
    subs = sorted(set(ref.index) & set(nue.index))
    if not subs:
        raise SystemExit("No hay sujetos en comun entre los dos CSV")

    t = pd.DataFrame({
        "cpu": ref.acc[subs], "gpu": nue.acc[subs],
        "d_acc": (nue.acc[subs] - ref.acc[subs]),
        "d_kappa": (nue.kappa[subs] - ref.kappa[subs]),
        "ep_cpu": ref.ep[subs], "ep_gpu": nue.ep[subs],
    }).sort_values("d_acc")

    print(f"Sujetos emparejados: {len(subs)} | semilla {SEED}\n")
    print(t.round(4).to_string())

    d = t.d_acc
    print(f"\nmedia   {d.mean():+.4f}   mediana {d.median():+.4f}   sd {d.std():.4f}")
    print(f"suben {int((d > 0).sum())} / bajan {int((d < 0).sum())}")
    print(f"|d| max {d.abs().max():.4f}  (el rango entre semillas ya medido era hasta 0.083)")

    if len(d) >= 6:
        p = wilcoxon(d).pvalue
        print(f"\nWilcoxon pareado (H0: el cambio de pipeline no altera la exactitud): p={p:.4f}")
        if p >= 0.05:
            print("-> NO se detecta sesgo sistematico. El cambio es aceptable.")
        else:
            print("-> HAY sesgo sistematico. Revisar lote y lr antes de gastar en la Fase 3.")
    print("\nOJO: 'no se detecta sesgo' no es 'son equivalentes'. Con esta n el test")
    print("descarta sesgos grandes, no diferencias pequenas.")


if __name__ == "__main__":
    main()
