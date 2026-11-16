"""EDA de calidad de señal + ERD/ERS sobre los 30 sujetos de MIND.

Batch largo (~1 h). Escribe el CSV de forma INCREMENTAL: si se interrumpe, lo ya calculado
se conserva y al relanzar se reanuda (salta los sujetos ya presentes en el CSV).

Uso:  conda run -n eeg python scripts/run_eda_all_subjects.py
"""
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import mne

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from datasets import mind as mp

mne.set_log_level("ERROR")

BIDS_ROOT = Path(os.environ.get("MIND_BIDS_ROOT", r"D:/Tesis/data/MIND_BIDS"))
OUT = REPO_ROOT / "results" / "reports" / "mind" / "eda_calidad_30sujetos.csv"
# ERD mu por CANAL y sujeto: evidencia descriptiva de dónde vive la señal de MI.
# OJO: es caracterización, NO selección de canales. Elegir canales mirando todos los sujetos
# y luego evaluar con LOSO sería FUGA. La selección debe ser anatómica a priori (plan §A.4)
# o calcularse dentro de cada fold de entrenamiento.
OUT_CH = REPO_ROOT / "results" / "reports" / "mind" / "eda_erd_por_canal.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

CFG = dict(mp.DEFAULT_CFG)
MOTOR = ["C3", "Cz", "C4"]
POSTERIOR = ["O1", "OZ", "Oz", "O2", "PO7", "PO8", "Pz", "PZ"]
BANDS = {"mu": (8, 13), "beta": (13, 30)}


def band_envelope(epochs, band):
    pw = epochs.copy().filter(*band, verbose="ERROR").apply_hilbert(envelope=True)
    return pw.get_data() * 1e6, pw.times, list(pw.ch_names)   # µV


def erd_pct(data, times, chs, picks, base=(-2, 0), win=(0.5, 3.5)):
    b = (times >= base[0]) & (times < base[1])
    w = (times >= win[0]) & (times < win[1])
    out = {}
    for c in picks:
        if c in chs:
            i = chs.index(c)
            bv = data[:, i, b].mean()
            wv = data[:, i, w].mean()
            out[c] = 100 * (wv - bv) / bv if bv > 0 else np.nan
    return out


def analyse(sub):
    # canales malos (antes de interpolar), sobre los 3 runs
    bads = set()
    for run in mp.RUNS:
        raw, _ = mp.load_run(sub, run, BIDS_ROOT, CFG)
        r = mp.set_montage_1020(raw.copy(), CFG)
        bads |= set(mp.detect_bad_channels(r, CFG))

    ep = mp.process_subject_epochs(sub, BIDS_ROOT, CFG, do_autoreject=False)
    row = {"sub": sub, "n_epocas": len(ep), "n_canales": len(ep.ch_names),
           "n_canales_malos": len(bads), "canales_malos": ";".join(sorted(bads))}

    for bname, band in BANDS.items():
        d, t, chs = band_envelope(ep, band)
        e = erd_pct(d, t, chs, MOTOR)
        for c in MOTOR:
            row[f"ERD{bname}_{c}"] = round(float(e.get(c, np.nan)), 1)
        row[f"ERD{bname}_motor_medio"] = round(float(np.nanmean(list(e.values()))), 1)
        if bname == "mu":
            # Bandera de alfa posterior (diagnóstico de sub-04: ojos cerrados / somnolencia)
            post = [c for c in POSTERIOR if c in chs]
            cen = [c for c in MOTOR if c in chs]
            pa = d[:, [chs.index(c) for c in post], :].mean() if post else np.nan
            ca = d[:, [chs.index(c) for c in cen], :].mean() if cen else np.nan
            row["alfa_posterior_uV"] = round(float(pa), 2)
            row["alfa_central_uV"] = round(float(ca), 2)
            row["ratio_post_central"] = round(float(pa / ca), 2) if ca else np.nan
            ep_post = erd_pct(d, t, chs, post)
            row["ERDmu_posterior_medio"] = round(float(np.nanmean(list(ep_post.values()))), 1)

    row["ERD_plausible"] = bool(row.get("ERDmu_motor_medio", 0) < 0)

    # ERD mu de TODOS los canales (caracterización espacial, ver nota de fuga arriba)
    d, t, chs = band_envelope(ep, BANDS["mu"])
    per_ch = erd_pct(d, t, chs, chs)
    ch_row = {"sub": sub, **{c: round(float(v), 1) for c, v in per_ch.items()}}
    return row, ch_row


def main():
    subs = sorted([p.name for p in BIDS_ROOT.glob("sub-*") if p.is_dir()])
    done = set()
    if OUT.exists():
        try:
            done = set(pd.read_csv(OUT)["sub"].tolist())
        except Exception:
            done = set()
    print(f"Sujetos: {len(subs)} | ya procesados: {len(done)}", flush=True)

    for i, s in enumerate(subs, 1):
        if s in done:
            print(f"[{i}/{len(subs)}] {s}: ya estaba, se salta", flush=True)
            continue
        try:
            row, ch_row = analyse(s)
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            pd.DataFrame([ch_row]).to_csv(OUT_CH, mode="a", header=not OUT_CH.exists(), index=False)
            print(f"[{i}/{len(subs)}] {s}: {row['n_epocas']} ép | ERDmu motor "
                  f"{row['ERDmu_motor_medio']:+.1f}% | ratio post/cen {row['ratio_post_central']} "
                  f"| {'OK' if row['ERD_plausible'] else 'SIN ERD'}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(subs)}] {s}: ERROR -> {type(e).__name__}: {e}", flush=True)

    if OUT.exists():
        res = pd.read_csv(OUT)
        print("\n===== RESUMEN =====", flush=True)
        print(f"Sujetos analizados: {len(res)}")
        print(f"Con ERD mu motor negativo (plausible): {int(res['ERD_plausible'].sum())}/{len(res)}")
        print(f"ERD mu motor: mediana {res['ERDmu_motor_medio'].median():.1f}% "
              f"| rango [{res['ERDmu_motor_medio'].min():.1f}, {res['ERDmu_motor_medio'].max():.1f}]")
        print(f"Canales malos por sujeto: mediana {res['n_canales_malos'].median():.0f} "
              f"| max {res['n_canales_malos'].max()}")
        print(f"-> {OUT}")

    if OUT_CH.exists():
        ch = pd.read_csv(OUT_CH).set_index("sub")
        med = ch.median(axis=0).sort_values()   # más negativo = más desincronización = más señal MI
        print("\n===== ERD mu por canal (mediana entre sujetos) =====")
        print("Top-12 canales con MAYOR desincronización (candidatos a portar señal MI):")
        print(med.head(12).round(1).to_string())
        print("\nTop-5 con MENOR desincronización (menos informativos):")
        print(med.tail(5).round(1).to_string())
        print("\n[NOTA] Esto es caracterización descriptiva. NO usar para seleccionar canales")
        print("       de cara a LOSO sin disciplina de folds: sería fuga de datos.")
        print(f"-> {OUT_CH}")


if __name__ == "__main__":
    main()
