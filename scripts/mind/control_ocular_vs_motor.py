"""CONTROL DECISIVO: ¿la señal decodificable es MOTORA u OCULAR?

MIND es un paradigma direccional (L→R, T→B, diagonales). Si los sujetos mueven los ojos
siguiendo la trayectoria imaginada, el artefacto ocular sería ESPECÍFICO DE CADA CLASE y un
clasificador podría acertar leyendo los ojos en vez de la imaginación motora.

Prueba: decodificar (FBCSP + LDA, CSP ajustado dentro de cada fold) usando por separado
  (a) solo canales FRONTALES  (proxy de EOG: Fp1/Fp2/Fpz/AF3/AF4/F7/F8)
  (b) solo canales CENTRALES/MOTORES (C/FC/CP)
  (c) todos los canales

Interpretación:
  frontal >> central  -> la señal es OCULAR (confundidor grave para la tesis)
  central >= frontal  -> hay señal motora genuina
  ambos ~ azar        -> no hay información decodificable

Uso:  conda run -n eeg python scripts/control_ocular_vs_motor.py
"""
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from datasets import mind as mp

mne.set_log_level("ERROR")

BIDS_ROOT = Path(os.environ.get("MIND_BIDS_ROOT", r"D:/Tesis/data/MIND_BIDS"))
OUT = REPO_ROOT / "results" / "reports" / "mind" / "control_ocular_vs_motor.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

CFG = dict(mp.DEFAULT_CFG)
BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 38)]
WIN = (0.0, 4.0)
CHANCE = 0.25

FRONTAL = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F8", "F5", "F6"]
MOTOR = ["C3", "C1", "CZ", "C2", "C4", "FC3", "FCZ", "FC4", "CP3", "CPZ", "CP4", "C5", "C6"]

SUBJECTS = os.environ.get("MIND_CTRL_SUBJECTS",
                          "sub-01,sub-03,sub-08,sub-15,sub-20,sub-26").split(",")


def pick_present(ep, names):
    up = {c.upper(): c for c in ep.ch_names}
    return [up[n] for n in names if n in up]


def fbcsp_cv(ep, picks, n_comp=4, seed=CFG["seed"]):
    """FBCSP + LDA con CSP ajustado SOLO en el train de cada fold."""
    if len(picks) < 4:
        return np.nan, np.nan
    e0 = ep.copy().pick(picks).crop(*WIN)
    y = e0.events[:, 2]
    n_comp = min(n_comp, len(picks) - 1)
    # el filtrado por banda es NO supervisado -> seguro fuera del CV
    Xb = [e0.copy().filter(lo, hi, verbose="ERROR").get_data() for lo, hi in BANDS]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    accs = []
    for tr, te in cv.split(Xb[0], y):
        ftr, fte = [], []
        for Xi in Xb:
            # rank="info": el CAR y la interpolación dejan la covarianza deficiente de rango
            # (autovalores ~1e-28, incluso negativos) y el blanqueo de CSP produce inf/NaN.
            csp = CSP(n_components=n_comp, reg="ledoit_wolf", log=True,
                      norm_trace=False, rank="info")
            csp.fit(Xi[tr], y[tr])
            ftr.append(csp.transform(Xi[tr]))
            fte.append(csp.transform(Xi[te]))
        clf = LDA(solver="lsqr", shrinkage="auto")
        clf.fit(np.hstack(ftr), y[tr])
        accs.append((clf.predict(np.hstack(fte)) == y[te]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def main():
    rows = []
    for i, sub in enumerate(SUBJECTS, 1):
        try:
            ep = mp.process_subject_epochs(sub, BIDS_ROOT, CFG, do_autoreject=False)
            f_ch, m_ch = pick_present(ep, FRONTAL), pick_present(ep, MOTOR)
            a_f, _ = fbcsp_cv(ep, f_ch)
            a_m, _ = fbcsp_cv(ep, m_ch)
            a_all, _ = fbcsp_cv(ep, ep.ch_names)
            row = {"sub": sub, "n_epocas": len(ep),
                   "acc_frontal": round(a_f, 4), "n_frontal": len(f_ch),
                   "acc_motor": round(a_m, 4), "n_motor": len(m_ch),
                   "acc_todos": round(a_all, 4), "n_todos": len(ep.ch_names)}
            rows.append(row)
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            print(f"[{i}/{len(SUBJECTS)}] {sub}: frontal {a_f:.3f} | motor {a_m:.3f} | "
                  f"todos {a_all:.3f}  (azar {CHANCE})", flush=True)
        except Exception as e:
            print(f"[{i}/{len(SUBJECTS)}] {sub}: ERROR {type(e).__name__}: {e}", flush=True)

    if rows:
        d = pd.DataFrame(rows)
        print("\n===== CONTROL OCULAR vs MOTOR =====")
        print(f"Frontal : media {d['acc_frontal'].mean():.3f}  (sobre azar: {int((d['acc_frontal']>CHANCE).sum())}/{len(d)})")
        print(f"Motor   : media {d['acc_motor'].mean():.3f}  (sobre azar: {int((d['acc_motor']>CHANCE).sum())}/{len(d)})")
        print(f"Todos   : media {d['acc_todos'].mean():.3f}  (sobre azar: {int((d['acc_todos']>CHANCE).sum())}/{len(d)})")
        diff = d["acc_frontal"].mean() - d["acc_motor"].mean()
        print(f"\nDiferencia frontal - motor: {diff:+.3f}")
        if d["acc_todos"].mean() < CHANCE + 0.02:
            print("VEREDICTO: sin información decodificable clara (todo cerca del azar).")
        elif diff > 0.05:
            print("VEREDICTO: señal dominada por canales FRONTALES -> sospecha de CONFUNDIDOR OCULAR.")
        else:
            print("VEREDICTO: la señal no está dominada por los frontales -> compatible con origen motor.")
        print(f"-> {OUT}")


if __name__ == "__main__":
    main()
