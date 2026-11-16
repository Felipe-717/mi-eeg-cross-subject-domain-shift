"""PRUEBA DEFINITIVA: ¿la señal motora sobrevive al eliminar las SACADAS direccionales?

El ICA (ICLabel) solo marcó componentes `eye blink`. Un parpadeo es igual en las 4 clases,
pero una SACADA (movimiento ocular direccional) NO: seguir con los ojos la trayectoria
imaginada produce un artefacto ESPECÍFICO DE CLASE. Ese es el confundidor que importa.

Proxy de EOG construido desde los canales frontales (las dos dimensiones de las 4 clases):
    HEOG (horizontal) ~ Fp2 - Fp1
    VEOG (vertical)   ~ Fpz   (o media de Fp1, Fp2)

Tres pruebas:
  1. ¿El proxy SOLO ya decodifica las clases?            -> confirma que hay info ocular direccional
  2. Decodificación motora ANTES de la regresión          -> línea base
  3. Decodificación motora DESPUÉS de regresar fuera el proxy -> ¿sobrevive?

Interpretación:
  motor sobrevive (caída pequeña) -> la señal motora es REAL. Defendible ante el jurado.
  motor se desploma               -> era artefacto ocular propagado. Replantear.

Nota anti-fuga: la regresión de EOG es NO supervisada (no usa las etiquetas), así que puede
ajustarse sobre todos los trials sin introducir fuga de clase.

Uso:  conda run -n eeg python scripts/control_eog_proxy.py
"""
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from datasets import mind as mp

mne.set_log_level("ERROR")

BIDS_ROOT = Path(os.environ.get("MIND_BIDS_ROOT", r"D:/Tesis/data/MIND_BIDS"))
OUT = REPO_ROOT / "results" / "reports" / "mind" / "control_eog_proxy.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

CFG = dict(mp.DEFAULT_CFG)
BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 38)]
WIN = (0.0, 4.0)
CHANCE = 0.25
MOTOR = ["C3", "C1", "CZ", "C2", "C4", "FC3", "FCZ", "FC4", "CP3", "CPZ", "CP4", "C5", "C6"]

SUBJECTS = os.environ.get("MIND_CTRL_SUBJECTS",
                          "sub-01,sub-03,sub-08,sub-15,sub-20,sub-26").split(",")


def up(ep):
    return {c.upper(): c for c in ep.ch_names}


def build_eog_proxy(ep):
    """Devuelve (n_trials, 2, n_times): HEOG = Fp2-Fp1, VEOG = Fpz."""
    m = up(ep)
    need = ["FP1", "FP2"]
    if not all(n in m for n in need):
        return None
    X = ep.get_data()
    i1, i2 = ep.ch_names.index(m["FP1"]), ep.ch_names.index(m["FP2"])
    heog = X[:, i2, :] - X[:, i1, :]
    if "FPZ" in m:
        veog = X[:, ep.ch_names.index(m["FPZ"]), :]
    else:
        veog = (X[:, i1, :] + X[:, i2, :]) / 2.0
    return np.stack([heog, veog], axis=1)


def regress_out(X, proxy):
    """Elimina de cada canal la parte explicable linealmente por el proxy de EOG.

    Se ajusta un modelo por canal sobre TODOS los trials concatenados en el tiempo
    (no usa etiquetas -> sin fuga de clase). X: (n_ep, n_ch, n_t); proxy: (n_ep, 2, n_t).
    """
    n_ep, n_ch, n_t = X.shape
    P = proxy.transpose(0, 2, 1).reshape(-1, proxy.shape[1])      # (n_ep*n_t, 2)
    P = np.column_stack([P, np.ones(len(P))])                     # + intercepto
    out = np.empty_like(X)
    for c in range(n_ch):
        y = X[:, c, :].reshape(-1)                                # (n_ep*n_t,)
        beta, *_ = np.linalg.lstsq(P, y, rcond=None)
        out[:, c, :] = (y - P @ beta).reshape(n_ep, n_t)
    return out


def fbcsp_from_arrays(Xb, y, n_comp=4, seed=CFG["seed"]):
    """FBCSP+LDA sobre una lista de arrays ya filtrados por banda. CSP dentro del fold."""
    accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xb[0], y):
        ftr, fte = [], []
        for Xi in Xb:
            csp = CSP(n_components=min(n_comp, Xi.shape[1] - 1), reg="ledoit_wolf",
                      log=True, norm_trace=False, rank="info")
            csp.fit(Xi[tr], y[tr])
            ftr.append(csp.transform(Xi[tr]))
            fte.append(csp.transform(Xi[te]))
        clf = LDA(solver="lsqr", shrinkage="auto")
        clf.fit(np.hstack(ftr), y[tr])
        accs.append((clf.predict(np.hstack(fte)) == y[te]).mean())
    return float(np.mean(accs))


def bandpass_arrays(X, sfreq, bands=BANDS):
    return [mne.filter.filter_data(X, sfreq, lo, hi, verbose="ERROR") for lo, hi in bands]


def decode_proxy_alone(proxy, y, sfreq):
    """¿El proxy de EOG (2 señales) decodifica las clases por sí solo?
    Con 2 canales, CSP no aporta: se usan log-varianzas por banda como features."""
    feats = []
    for lo, hi in BANDS:
        Pb = mne.filter.filter_data(proxy, sfreq, lo, hi, verbose="ERROR")
        feats.append(np.log(np.var(Pb, axis=2) + 1e-20))          # (n_ep, 2)
    F = np.hstack(feats)
    clf = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    cv = StratifiedKFold(5, shuffle=True, random_state=CFG["seed"])
    return float(cross_val_score(clf, F, y, cv=cv, scoring="accuracy").mean())


def main():
    for i, sub in enumerate(SUBJECTS, 1):
        try:
            ep = mp.process_subject_epochs(sub, BIDS_ROOT, CFG, do_autoreject=False)
            ep = ep.copy().crop(*WIN)
            sf = ep.info["sfreq"]
            y = ep.events[:, 2]
            proxy = build_eog_proxy(ep)
            if proxy is None:
                print(f"[{i}] {sub}: sin canales Fp -> se omite", flush=True)
                continue

            # (1) proxy solo
            acc_proxy = decode_proxy_alone(proxy, y, sf)

            # (2) motor antes
            m = up(ep)
            picks = [m[c] for c in MOTOR if c in m]
            Xm = ep.copy().pick(picks).get_data()
            acc_pre = fbcsp_from_arrays(bandpass_arrays(Xm, sf), y)

            # (3) motor después de regresar fuera el EOG
            Xm_clean = regress_out(Xm, proxy)
            acc_post = fbcsp_from_arrays(bandpass_arrays(Xm_clean, sf), y)

            row = {"sub": sub, "n_trials": len(y),
                   "acc_proxy_EOG": round(acc_proxy, 4),
                   "acc_motor_pre": round(acc_pre, 4),
                   "acc_motor_post": round(acc_post, 4),
                   "caida": round(acc_pre - acc_post, 4)}
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            print(f"[{i}/{len(SUBJECTS)}] {sub}: proxy EOG {acc_proxy:.3f} | motor pre {acc_pre:.3f} "
                  f"-> post {acc_post:.3f} (caída {acc_pre-acc_post:+.3f})", flush=True)
        except Exception as e:
            print(f"[{i}/{len(SUBJECTS)}] {sub}: ERROR {type(e).__name__}: {e}", flush=True)

    if OUT.exists():
        d = pd.read_csv(OUT)
        print("\n===== PRUEBA DEL PROXY DE EOG =====", flush=True)
        print(f"Proxy EOG solo : {d['acc_proxy_EOG'].mean():.3f}  "
              f"(sobre azar en {int((d['acc_proxy_EOG']>CHANCE).sum())}/{len(d)})")
        print(f"Motor ANTES    : {d['acc_motor_pre'].mean():.3f}")
        print(f"Motor DESPUÉS  : {d['acc_motor_post'].mean():.3f}")
        print(f"Caída media    : {d['caida'].mean():+.3f}")
        print("\n--- VEREDICTO ---")
        if d["acc_motor_post"].mean() < CHANCE + 0.02:
            print("La señal motora NO sobrevive -> era artefacto ocular propagado. Replantear.")
        elif d["caida"].mean() > 0.05:
            print("La señal motora sobrevive pero CAE de forma apreciable -> parte era ocular.")
            print("Reportar SIEMPRE los resultados con el EOG regresado fuera.")
        else:
            print("La señal motora SOBREVIVE sin caída apreciable -> es REAL.")
            print("Argumento sólido: la decodificación no depende de los movimientos oculares.")
        print(f"-> {OUT}")


if __name__ == "__main__":
    main()
