"""DIAGNOSTICO DE ATRIBUCION: el confundidor frontal, ¿es OCULAR o MUSCULAR?

CONTEXTO (2026-07-24). El Control A midio que 9/54 sujetos tienen informacion de clase
significativa en el proxy `Fp2-Fp1`, y lo llamamos "ocular". Pero `Fp1`/`Fp2` estan sobre
el musculo frontal y cerca del temporal, asi que ese proxy tambien recoge EMG facial: el
nombre del electrodo no demuestra el origen de la señal.

El diagnostico de ICA sobre sub-29 lo puso en duda. Ningun componente correlaciona con el
HEOG (ni siquiera el que ICLabel etiqueta 'eye blink' con 0.85: el parpadeo es VERTICAL y
`Fp2-Fp1` lo cancela por construccion), y los que decodifican estan etiquetados
'muscle artifact' y 'other'.

COMO SE DISTINGUEN
  Los dos artefactos viven en bandas distintas y no se solapan:

      movimiento ocular  -> LENTO,  0.5-8 Hz   (la sacada es casi un escalon)
      EMG facial         -> RAPIDO, >20 Hz     (el musculo es banda ancha alta)

  Asi que basta ver EN QUE BANDA se concentra la decodificacion del proxy.

LECTURA
  * Pico en 0.5-4 / 4-8 Hz   -> OCULAR. Tratamiento: ICA de componentes oculares.
  * Pico en 20-38 Hz         -> MUSCULAR. Tratamiento: bajar el umbral de
                                'muscle artifact' (hoy 0.9, muy estricto).
  * Plano o pico en 8-30 Hz  -> ninguno de los dos: seria señal NEURAL frontal legitima,
                                y entonces no hay nada que limpiar.

  Este ultimo caso importa: si la decodificacion del proxy vive en mu/beta, no es un
  artefacto, es actividad cerebral que los frontopolares captan por conduccion de volumen,
  y "limpiarla" seria borrar señal.

Uso:
  python scripts/openbmi/diag_banda.py --subjects sub-29 sub-04 sub-35 sub-01
  python scripts/openbmi/diag_banda.py --cache D:/.../epochs_cache_ica --subjects sub-29
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from mne.filter import filter_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "openbmi"))
import splits as sp                       # noqa: E402
import datasets.openbmi as DS             # noqa: E402
from control_ocular import build_eog_proxy, BANDS   # noqa: E402


def decode_banda(proxy, y, sfreq, banda, seed=2026):
    """Decodifica la clase usando el proxy filtrado a UNA sola banda."""
    lo, hi = banda
    Pb = filter_data(proxy.astype(np.float64), sfreq, lo, hi, verbose="ERROR")
    F = np.log(np.var(Pb, axis=2) + 1e-20)          # (n_trials, 2)
    clf = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, F, y, cv=cv, scoring="accuracy").mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="por defecto, TODOS los del cache")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None, help="CSV de salida")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    cache = Path(args.cache) if args.cache else DS.CACHE_DIR
    if not cache.exists():
        sys.exit(f"No existe el cache {cache}")
    if args.subjects is None:
        args.subjects = sorted({f.name.split("_")[0]
                                for f in cache.glob("sub-*_full.npz")})
    out = Path(args.out) if args.out else (
        REPO_ROOT / "results" / "reports" / "openbmi" / f"diag_banda_{cache.name}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Cache: {cache}\nBanda del proxy Fp2-Fp1 que lleva la informacion de clase\n")
    cab = "sujeto   " + " ".join(f"{lo:.0f}-{hi:.0f}".rjust(7) for lo, hi in BANDS) + "   tipo"
    print(cab)
    print("-" * len(cab))

    filas = []
    for sub in args.subjects:
        X, y, chs = sp.load_subject(sub, cache, window=DS.WINDOW, ch_names=None)
        y = np.asarray(y)
        proxy = build_eog_proxy(X, chs)
        if proxy is None:
            print(f"{sub}: faltan Fp1/Fp2")
            continue
        accs = [decode_banda(proxy, y, DS.SFREQ, b, args.seed) for b in BANDS]
        i = int(np.argmax(accs))
        lo, hi = BANDS[i]
        # Solo se atribuye si el pico despega del azar; si no, no hay nada que atribuir.
        if accs[i] < 0.55:
            tipo = "limpio"
        elif hi <= 8:
            tipo = "OCULAR"
        elif lo >= 20:
            tipo = "MUSCULAR"
        else:
            tipo = "mu/beta"      # posible señal neural frontal, NO es artefacto
        print(f"{sub}  " + " ".join(f"{a:7.3f}" for a in accs) +
              f"   {tipo} ({lo:.0f}-{hi:.0f} Hz, {accs[i]:.3f})")
        filas.append({"sub": sub, "tipo": tipo, "pico_hz": f"{lo:.0f}-{hi:.0f}",
                      "acc_pico": round(accs[i], 4),
                      **{f"b_{a:.0f}_{b:.0f}": round(v, 4)
                         for (a, b), v in zip(BANDS, accs)}})


    if filas:
        import pandas as pd
        d = pd.DataFrame(filas)
        d.to_csv(out, index=False)
        print("\n--- ATRIBUCION ---")
        for t, n in d["tipo"].value_counts().items():
            subs = ", ".join(d.loc[d.tipo == t, "sub"])
            print(f"  {t:9s} {n:>3}  {subs}")
        print("\n  OCULAR   -> lo trata el ICA de componentes oculares.")
        print("  MUSCULAR -> pide bajar el umbral de 'muscle artifact' (hoy 0.9).")
        print("  mu/beta  -> NO es artefacto: senal neural captada por conduccion de")
        print("              volumen en los frontopolares. Limpiarla borraria senal.")
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
