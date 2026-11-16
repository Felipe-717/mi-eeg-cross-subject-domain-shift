"""EXPERIMENTO DECISIVO: ¿desaparece la ventaja frontal al eliminar los componentes oculares?

Repite el control frontal-vs-motor PERO aplicando antes ICA + ICLabel para retirar componentes
clasificados como oculares (y opcionalmente musculares).

Desenlaces:
  (a) la ventaja frontal desaparece y el motor se mantiene -> había confundidor ocular, ya limpio
  (b) todo cae cerca del azar                              -> la señal era ocular
  (c) nada cambia                                          -> la señal frontal es neural

Uso:  conda run -n eeg python scripts/control_ica_ocular.py
"""
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from datasets import mind as mp

mne.set_log_level("ERROR")

BIDS_ROOT = Path(os.environ.get("MIND_BIDS_ROOT", r"D:/Tesis/data/MIND_BIDS"))
OUT = REPO_ROOT / "results" / "reports" / "mind" / "control_ica_ocular.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

CFG = dict(mp.DEFAULT_CFG)
BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 38)]
WIN = (0.0, 4.0)
CHANCE = 0.25
FRONTAL = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F8", "F5", "F6"]
MOTOR = ["C3", "C1", "CZ", "C2", "C4", "FC3", "FCZ", "FC4", "CP3", "CPZ", "CP4", "C5", "C6"]
# Etiquetas de ICLabel que consideramos artefacto ocular (y muscular, configurable)
DROP_LABELS = {"eye blink", "eye"}
DROP_MUSCLE = os.environ.get("DROP_MUSCLE", "0") == "1"
PROB_MIN = 0.8

SUBJECTS = os.environ.get("MIND_CTRL_SUBJECTS",
                          "sub-01,sub-03,sub-08,sub-15,sub-20,sub-26").split(",")


def pick_present(ep, names):
    up = {c.upper(): c for c in ep.ch_names}
    return [up[n] for n in names if n in up]


def fbcsp_cv(ep, picks, n_comp=4, seed=CFG["seed"]):
    if len(picks) < 4:
        return np.nan
    e0 = ep.copy().pick(picks).crop(*WIN)
    y = e0.events[:, 2]
    n_comp = min(n_comp, len(picks) - 1)
    Xb = [e0.copy().filter(lo, hi, verbose="ERROR").get_data() for lo, hi in BANDS]
    accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xb[0], y):
        ftr, fte = [], []
        for Xi in Xb:
            csp = CSP(n_components=n_comp, reg="ledoit_wolf", log=True,
                      norm_trace=False, rank="info")
            csp.fit(Xi[tr], y[tr])
            ftr.append(csp.transform(Xi[tr]))
            fte.append(csp.transform(Xi[te]))
        clf = LDA(solver="lsqr", shrinkage="auto")
        clf.fit(np.hstack(ftr), y[tr])
        accs.append((clf.predict(np.hstack(fte)) == y[te]).mean())
    return float(np.mean(accs))


def clean_with_ica(raw, seed=CFG["seed"]):
    """Ajusta ICA (infomax extendido, requisito de ICLabel) y retira componentes oculares.
    Devuelve (raw_limpio, n_excluidos, etiquetas)."""
    from mne_icalabel import label_components
    # ICLabel espera datos con referencia promedio (ya la tenemos) y paso alto >=1 Hz.
    raw_ica = raw.copy().filter(1.0, None, verbose="ERROR")
    n_comp = min(30, len(raw.ch_names) - 2)
    ica = ICA(n_components=n_comp, method="infomax", fit_params=dict(extended=True),
              max_iter="auto", random_state=seed)
    ica.fit(raw_ica, verbose="ERROR")
    res = label_components(raw_ica, ica, method="iclabel")
    labels, probs = res["labels"], np.asarray(res["y_pred_proba"])
    targets = set(DROP_LABELS) | ({"muscle artifact", "muscle"} if DROP_MUSCLE else set())
    excl = [i for i, (lab, p) in enumerate(zip(labels, probs))
            if lab.lower() in targets and p >= PROB_MIN]
    ica.exclude = excl
    out = ica.apply(raw.copy(), verbose="ERROR")
    return out, len(excl), [labels[i] for i in excl]


def epochs_with_ica(sub):
    ep_list, n_excl_tot, labs = [], 0, []
    for run in mp.RUNS:
        raw, _ = mp.load_run(sub, run, BIDS_ROOT, CFG)
        rc = mp.preprocess_core(raw, CFG)
        rc, n_ex, lb = clean_with_ica(rc)
        n_excl_tot += n_ex
        labs += lb
        ep_list.append(mp.reject_gross(mp.make_epochs(rc, CFG), CFG))
    return mne.concatenate_epochs(ep_list, verbose="ERROR"), n_excl_tot, labs


def main():
    for i, sub in enumerate(SUBJECTS, 1):
        try:
            ep, n_ex, labs = epochs_with_ica(sub)
            f_ch, m_ch = pick_present(ep, FRONTAL), pick_present(ep, MOTOR)
            row = {"sub": sub, "n_epocas": len(ep), "n_comp_excluidos": n_ex,
                   "etiquetas": ";".join(sorted(set(labs))),
                   "acc_frontal_ICA": round(fbcsp_cv(ep, f_ch), 4),
                   "acc_motor_ICA": round(fbcsp_cv(ep, m_ch), 4),
                   "acc_todos_ICA": round(fbcsp_cv(ep, ep.ch_names), 4)}
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            print(f"[{i}/{len(SUBJECTS)}] {sub}: ICs fuera {n_ex} | frontal {row['acc_frontal_ICA']:.3f} "
                  f"| motor {row['acc_motor_ICA']:.3f} | todos {row['acc_todos_ICA']:.3f}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(SUBJECTS)}] {sub}: ERROR {type(e).__name__}: {e}", flush=True)

    if OUT.exists():
        d = pd.read_csv(OUT)
        prev_f = REPO_ROOT / "results" / "reports" / "mind" / "control_ocular_vs_motor.csv"
        print("\n===== ICA: ¿desaparece la ventaja frontal? =====")
        print(f"CON ICA -> frontal {d['acc_frontal_ICA'].mean():.3f} | "
              f"motor {d['acc_motor_ICA'].mean():.3f} | todos {d['acc_todos_ICA'].mean():.3f}")
        dif_ica = d["acc_frontal_ICA"].mean() - d["acc_motor_ICA"].mean()
        print(f"Diferencia frontal - motor CON ICA: {dif_ica:+.3f}")
        if prev_f.exists():
            p = pd.read_csv(prev_f)
            dif_sin = p["acc_frontal"].mean() - p["acc_motor"].mean()
            print(f"SIN ICA -> frontal {p['acc_frontal'].mean():.3f} | "
                  f"motor {p['acc_motor'].mean():.3f} | todos {p['acc_todos'].mean():.3f}")
            print(f"Diferencia frontal - motor SIN ICA: {dif_sin:+.3f}")
            print("\n--- VEREDICTO ---")
            if d["acc_motor_ICA"].mean() < CHANCE + 0.03:
                print("(b) Todo cae cerca del azar -> la señal era mayormente OCULAR. Replantear.")
            elif dif_ica < dif_sin - 0.03:
                print("(a) La ventaja frontal SE REDUCE y el motor se mantiene -> había confundidor")
                print("    ocular y el ICA lo limpia. Recomendación: ICA entra al NÚCLEO del pipeline.")
            else:
                print("(c) La ventaja frontal PERSISTE -> probablemente es actividad neural frontal,")
                print("    no artefacto ocular. Documentar y continuar.")
        print(f"-> {OUT}")


if __name__ == "__main__":
    main()
