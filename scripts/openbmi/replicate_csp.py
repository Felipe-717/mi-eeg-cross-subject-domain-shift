"""🚪 PUERTA 1 — Réplica del baseline CSP+LDA de Lee 2019 sobre el caché de OpenBMI.

Objetivo: reproducir el CSP-cv del paper (sesión 1 ≈ 70.1 %, sesión 2 ≈ 72.2 %).
Si se reproduce, el cargador, los eventos, las etiquetas, los canales y los splits
están bien -> cualquier fallo posterior es del entrenamiento profundo, no del dato.
CSP+LDA no tiene partes móviles (sin GPU, sin hiperparámetros): es el control más
barato y limpio para validar el DATO antes de gastar cómputo. (Lección de MIND.)

Método del paper (VERIFICADO en Lee 2019, sección MI):
  * CSP de BANDA ÚNICA 8-30 Hz (Butterworth orden 5) — NO es FBCSP.
  * 20 canales motores (`motor_paper`).
  * Ventana 1000-3500 ms.
  * CSP-cv = 10 repeticiones de 10-fold CV, mezclando offline+online DENTRO de la sesión.
  * Modelos POR SUJETO Y POR SESIÓN; las sesiones NO se mezclan.

Diferencias con el paper (menores, documentadas):
  * Trabajamos desde el caché (0.5-40 Hz, 200 Hz) y RE-filtramos 8-30 encima. El
    prefiltro 0.5-40 es inocuo porque 8-30 está dentro. 200 Hz vs 100 Hz del paper
    es irrelevante para contenido < 40 Hz.

Uso:
  python scripts/openbmi/replicate_csp.py                 # los 54 sujetos
  python scripts/openbmi/replicate_csp.py --subjects sub-01 sub-02 --reps 3
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import datasets
import splits as sp

mne.set_log_level("ERROR")

DS = datasets.get("openbmi")
OUT = REPO_ROOT / "results" / "reports" / "openbmi" / "puerta1_csp_replica.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Config de los AUTORES (no la nuestra): es una RÉPLICA, no el estudio.
BAND = (DS.AUTHORS_CFG["l_freq"], DS.AUTHORS_CFG["h_freq"])   # 8-30 Hz
WIN = DS.AUTHORS_CFG["window"]                                # 1.0-3.5 s
CHANNELS = "motor_paper"                                      # los 20 del paper
REF = DS.AUTHORS_REF                                          # {'session_1':0.701, ...}


def csp_cv(X, y, n_folds=10, n_reps=10, n_comp=8, seed=2026):
    """CSP-cv: media de accuracy/kappa sobre n_reps x n_folds, CSP ajustado dentro del fold."""
    accs, kappas = [], []
    for rep in range(n_reps):
        cv = StratifiedKFold(n_folds, shuffle=True, random_state=seed + rep)
        for tr, te in cv.split(X, y):
            csp = CSP(n_components=min(n_comp, X.shape[1] - 1), reg="ledoit_wolf",
                      log=True, norm_trace=False)
            f_tr = csp.fit_transform(X[tr], y[tr])
            f_te = csp.transform(X[te])
            clf = LDA(solver="lsqr", shrinkage="auto")
            clf.fit(f_tr, y[tr])
            pred = clf.predict(f_te)
            accs.append(accuracy_score(y[te], pred))
            kappas.append(cohen_kappa_score(y[te], pred))
    return float(np.mean(accs)), float(np.mean(kappas))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=10, help="repeticiones de CV (paper: 10)")
    args = ap.parse_args()

    subs = args.subjects or sorted(
        p.name.replace("_full.npz", "") for p in DS.CACHE_DIR.glob("sub-*_full.npz"))
    print(f"PUERTA 1 - replica CSP+LDA | {len(subs)} sujetos | banda {BAND} Hz | "
          f"ventana {WIN} s | {CHANNELS} | {args.reps}x10-fold", flush=True)
    print(f"Objetivo: sesion 1 ~ {REF['session_1']:.3f} | sesion 2 ~ {REF['session_2']:.3f}\n",
          flush=True)

    filas = []
    for i, sub in enumerate(subs, 1):
        t0 = time.time()
        # Ventana del paper (1.0-3.5 s) recortada del caché; 20 canales motores.
        X, y_txt, chs = sp.load_subject(sub, DS.CACHE_DIR, window=WIN,
                                        ch_names=DS.CH_SETS[CHANNELS])
        y = np.array([DS.CLASS_ORDER.index(str(v)) for v in y_txt])
        d = np.load(DS.CACHE_DIR / f"{sub}_full.npz", allow_pickle=True)
        sess = d["session"]

        # Banda 8-30 Hz (no supervisado -> fuera del CV)
        Xf = mne.filter.filter_data(X.astype(np.float64), DS.SFREQ, *BAND, verbose="ERROR")

        fila = {"sub": sub}
        for s_key, s_name in DS.SESSION_KEYS.items():   # '0'->session_1, '1'->session_2
            m = sess == s_key
            acc, kappa = csp_cv(Xf[m], y[m], n_reps=args.reps)
            fila[f"{s_name}_acc"] = round(acc, 4)
            fila[f"{s_name}_kappa"] = round(kappa, 4)
        fila["segundos"] = round(time.time() - t0, 1)
        filas.append(fila)
        pd.DataFrame([fila]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        print(f"[{i}/{len(subs)}] {sub}: "
              f"s1={fila['session_1_acc']:.3f} s2={fila['session_2_acc']:.3f} "
              f"({fila['segundos']}s)", flush=True)

    d = pd.DataFrame(filas)
    print(f"\n===== PUERTA 1 - REPLICA ({len(d)} sujetos, azar 0.50) =====")
    for s_name, ref in [("session_1", REF["session_1"]), ("session_2", REF["session_2"])]:
        m, sd = d[f"{s_name}_acc"].mean(), d[f"{s_name}_acc"].std()
        delta = m - ref
        marca = "OK REPRODUCE" if abs(delta) <= 0.03 else ("~ cerca" if abs(delta) <= 0.05 else "!! LEJOS")
        print(f"  {s_name}: {m:.3f} +/- {sd:.3f}   (paper {ref:.3f}, delta {delta:+.3f})  {marca}")
    print("\nSi REPRODUCE (+/-0.03): el dato esta bien -> seguir con EEGNet (Puerta 2).")
    print("Si esta LEJOS: revisar canales/ventana/banda ANTES de tocar nada profundo.")


if __name__ == "__main__":
    main()
