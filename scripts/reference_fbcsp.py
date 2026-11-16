"""REFERENCIA CLÁSICA INTER-SUJETO: FBCSP + LDA bajo el protocolo LOSO.

¿Por qué este script? El baseline EEGNet se queda en el azar en LOSO (val_loss ~ ln(4)).
Antes de culpar al modelo hay que responder una pregunta más básica:

    ¿Es MIND decodificable ENTRE SUJETOS con CUALQUIER método?

FBCSP+LDA es el método en el que ya confiamos (0.353 within-subject sobre canales motores,
`scripts/control_eog_proxy.py`) y es el que usan los autores del dataset. Si FBCSP TAMBIÉN
fracasa en LOSO, el problema es la tarea y no la arquitectura -> eso es un HALLAZGO de la
tesis (el cambio de dominio es severo en este paradigma), no un fallo de implementación.
Si FBCSP sí decodifica, entonces el problema está en cómo entrenamos la red.

Diferencia clave con `control_ocular_vs_motor.py`: allí el CV era WITHIN-subject (train y
test del mismo sujeto). Aquí el CSP se ajusta con 29 sujetos y se evalúa en uno NUNCA VISTO.

Uso:  python scripts/reference_fbcsp_loso.py [--folds 6] [--channels motor_ext] [--norm per_trial]
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, cohen_kappa_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import datasets
import splits as sp

mne.set_log_level("ERROR")

DS = datasets.get("mind")     # este script es la referencia clásica sobre MIND
CACHE = DS.CACHE_DIR
OUT = REPO_ROOT / "results" / "reports" / "mind" / "reference_fbcsp_loso.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Mismo banco de bandas que scripts/control_ocular_vs_motor.py (y que los autores)
BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 38)]
WINDOW = (0.0, 4.0)
SFREQ = 200.0
CHANCE = 0.25


def filter_bank(X, sfreq=SFREQ):
    """Devuelve una lista con X filtrado en cada banda. El filtrado es NO supervisado."""
    return [mne.filter.filter_data(X.astype(np.float64), sfreq, lo, hi, verbose="ERROR")
            for lo, hi in BANDS]


def fbcsp_loso_fold(X_tr, y_tr, X_te, y_te, n_comp=4):
    """FBCSP ajustado SOLO con los sujetos de train; evaluado en el sujeto excluido."""
    n_comp = min(n_comp, X_tr.shape[1] - 1)
    Xb_tr, Xb_te = filter_bank(X_tr), filter_bank(X_te)
    ftr, fte = [], []
    for Xi_tr, Xi_te in zip(Xb_tr, Xb_te):
        # rank="info" es obligatorio: el CAR + interpolación dejan la covarianza deficiente
        # de rango y el blanqueo interno de CSP produce inf/NaN (trampa #1 del HANDOFF).
        csp = CSP(n_components=n_comp, reg="ledoit_wolf", log=True,
                  norm_trace=False, rank="info")
        csp.fit(Xi_tr, y_tr)
        ftr.append(csp.transform(Xi_tr))
        fte.append(csp.transform(Xi_te))
    clf = LDA(solver="lsqr", shrinkage="auto")
    clf.fit(np.hstack(ftr), y_tr)
    y_pred = clf.predict(np.hstack(fte))
    return accuracy_score(y_te, y_pred), cohen_kappa_score(y_te, y_pred)


# Los 6 sujetos de scripts/control_eog_proxy.py, para que la comparación sea directa.
SUBS_CONTROL = ["sub-01", "sub-03", "sub-08", "sub-15", "sub-20", "sub-26"]
REF_MOTOR_PRE = 0.353     # media "motor pre" publicada en HANDOFF.md §3 sobre esos 6 sujetos


def within_on_cache(channels: str, norm, n_folds: int = 5):
    """CONTROL DE VALIDEZ DEL CACHÉ (cierra la Actividad 2.5/2.8).

    Todos los números de decodificabilidad en los que confiamos (0.333 / 0.353 / 0.369)
    salieron de `scripts/control_*.py`, que epocan AL VUELO desde BIDS y con
    do_autoreject=False. La Fase 3 entera consume el CACHÉ, generado con autoreject=True.
    Son dos rutas distintas y nunca se comprobó que la segunda conserve la señal.

    Este control repite FBCSP within-subject sobre el CACHÉ con los mismos 6 sujetos,
    canales y ventana. Si reproduce ~0.353, el caché está sano y el fracaso en LOSO es real.
    Si sale en el azar, el caché degradó la señal y hay que revisar el preprocesamiento
    ANTES de sacar cualquier conclusión sobre el dataset.
    """
    from sklearn.model_selection import StratifiedKFold

    print(f"CONTROL within-subject SOBRE EL CACHÉ | canales={channels} | norm={norm}")
    print(f"Referencia a reproducir (procesado al vuelo, HANDOFF §3): {REF_MOTOR_PRE:.3f}\n",
          flush=True)
    rows = []
    for sub in SUBS_CONTROL:
        t0 = time.time()
        X, y, _ = sp.load_subject(sub, CACHE, window=WINDOW, ch_names=DS.CH_SETS[channels])
        if norm == "per_trial":
            X = sp.normalize_per_trial(X)
        cv = StratifiedKFold(n_folds, shuffle=True, random_state=2026)
        accs, kappas = [], []
        for tr_i, te_i in cv.split(X, y):
            a, k = fbcsp_loso_fold(X[tr_i], y[tr_i], X[te_i], y[te_i])
            accs.append(a); kappas.append(k)
        row = {"sub": sub, "n_trials": len(y), "acc": round(float(np.mean(accs)), 4),
               "kappa": round(float(np.mean(kappas)), 4), "channels": channels,
               "norm": str(norm), "segundos": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"  {sub}: acc={row['acc']:.3f} kappa={row['kappa']:.3f} "
              f"({row['n_trials']} trials, {row['segundos']}s)", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(REPO_ROOT / "results" / "reports" / "mind" / "control_cache_within_fbcsp.csv", index=False)
    media = d["acc"].mean()
    print(f"\n===== CONTROL DEL CACHÉ ({len(d)} sujetos, azar={CHANCE}) =====")
    print(f"  accuracy media: {media:.3f} ± {d['acc'].std():.3f}   (referencia {REF_MOTOR_PRE:.3f})")
    print(f"  kappa media   : {d['kappa'].mean():.3f}")
    print(f"  sobre azar en : {int((d['acc'] > CHANCE).sum())}/{len(d)} sujetos")
    if media >= REF_MOTOR_PRE - 0.05:
        print("\n  ✓ El caché REPRODUCE la referencia -> está sano.")
        print("    => El fracaso en LOSO es real: el cambio de dominio es el problema.")
    elif media > CHANCE + 0.03:
        print("\n  ~ Decodifica, pero por debajo de la referencia -> el caché perdió algo")
        print("    (¿autoreject demasiado agresivo?). Revisar antes de concluir sobre LOSO.")
    else:
        print("\n  ⚠ El caché NO reproduce la decodificabilidad within-subject.")
        print("    => TODO lo medido en Fase 3 está sobre un caché degradado. Hay que")
        print("       revisar el preprocesamiento ANTES de concluir nada sobre el dataset.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["loso", "check-cache"], default="loso")
    ap.add_argument("--folds", type=int, default=6, help="cuántos folds LOSO correr")
    ap.add_argument("--channels", choices=list(DS.CH_SETS.keys()), default="motor_ext")
    ap.add_argument("--norm", choices=["global", "per_subject", "per_trial", "none"],
                    default="per_trial")
    args = ap.parse_args()

    if args.mode == "check-cache":
        within_on_cache(args.channels, None if args.norm == "none" else args.norm)
        return

    subs = sorted(p.name.replace("_full.npz", "") for p in CACHE.glob("sub-*_full.npz"))
    folds = sp.make_loso_folds(subs)[:args.folds]
    norm = None if args.norm == "none" else args.norm
    ch_view = DS.CH_SETS[args.channels]
    print(f"FBCSP LOSO: {len(folds)} folds | canales={args.channels} | norm={args.norm}",
          flush=True)

    rows = []
    for f in folds:
        t0 = time.time()
        # El CSP se ajusta con inner_train + inner_val (no hay early stopping que proteger);
        # el sujeto de test sigue siendo intocable.
        built = sp.build_fold(f, CACHE, window=WINDOW, ch_names=ch_view, normalize=norm)
        X_tr = np.concatenate([built["X_train"], built["X_val"]])
        y_tr = np.concatenate([built["y_train"], built["y_val"]])
        acc, kappa = fbcsp_loso_fold(X_tr, y_tr, built["X_test"], built["y_test"])
        row = {"fold": f.idx, "test_sub": f.test, "n_train": len(y_tr),
               "n_test": len(built["y_test"]), "acc": round(acc, 4), "kappa": round(kappa, 4),
               "channels": args.channels, "norm": args.norm, "segundos": round(time.time() - t0, 1)}
        rows.append(row)
        pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"  fold {f.idx} (test={f.test}): acc={acc:.3f} kappa={kappa:.3f} "
              f"({row['segundos']}s)", flush=True)

    d = pd.DataFrame(rows)
    print(f"\n===== FBCSP LOSO ({len(d)} folds, azar={CHANCE}) =====")
    print(f"  accuracy: media {d['acc'].mean():.3f} ± {d['acc'].std():.3f} "
          f"| sobre azar en {int((d['acc'] > CHANCE).sum())}/{len(d)}")
    print(f"  kappa   : media {d['kappa'].mean():.3f} ± {d['kappa'].std():.3f}")
    print("\nLECTURA:")
    print("  Si FBCSP tampoco supera el azar -> el cambio de dominio en MIND es severo para")
    print("  cualquier método. Es un HALLAZGO de la tesis, no un fallo de EEGNet.")
    print("  Si FBCSP sí decodifica -> el problema está en cómo entrenamos la red.")


if __name__ == "__main__":
    main()
