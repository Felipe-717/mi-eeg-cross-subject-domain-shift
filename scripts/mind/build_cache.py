"""Genera el CACHÉ DE TENSORES de los 30 sujetos de MIND: el entregable de la fase de datos.

Pipeline aplicado (todo automático y determinista, mismos parámetros para todos):
  BrainVision -> montaje 10-20 (60 ch) -> canales malos interpolados -> filtro 0.5-40 + notch 50/100
  -> CAR -> 200 Hz -> ICA+ICLabel (quita componentes oculares) -> épocas -2..10 s
  -> rechazo grosero -> autoreject

Decisiones de diseño:
  * VENTANA AMPLIA (-2..10 s): cualquier ventana de modelado es luego un simple RECORTE del caché,
    así que la decisión de ventana no obliga a reprocesar (horas de cómputo).
  * MONTAJE COMPLETO (60 canales): la selección de canales (motor_ext, etc.) se aplica después como
    VISTA (mp.CH_SETS), lo que habilita el ablation de nº de canales del plan §A.4.
  * SIN NORMALIZAR: el z-score se ajusta por fold LOSO, solo con train (anti-fuga).

Salida por sujeto: <sub>_full-epo.fif (épocas MNE) y <sub>_full.npz (tensor + metadatos).
Escritura incremental: si se interrumpe, al relanzar salta los sujetos ya hechos.

Uso:  conda run -n eeg python scripts/build_tensor_cache.py
"""
import sys, os, json, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import mne

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from datasets import mind as mp

mne.set_log_level("ERROR")

BIDS_ROOT = Path(os.environ.get("MIND_BIDS_ROOT", r"D:/Tesis/data/MIND_BIDS"))
CACHE = Path(os.environ.get("MIND_PROC_ROOT", r"D:/Tesis/data/processed/mind")) / "epochs_cache"
CACHE.mkdir(parents=True, exist_ok=True)
LOG = REPO_ROOT / "results" / "reports" / "mind" / "tensor_cache_log.csv"
LOG.parent.mkdir(parents=True, exist_ok=True)

CFG = dict(mp.DEFAULT_CFG)
CFG_HASH = hashlib.sha256(json.dumps(CFG, sort_keys=True, default=str).encode()).hexdigest()[:12]


def build(sub):
    t0 = time.time()
    ep, info = mp.process_subject_epochs(sub, BIDS_ROOT, CFG, do_autoreject=True, return_info=True)
    ep.save(CACHE / f"{sub}_full-epo.fif", overwrite=True, verbose="ERROR")

    X = ep.get_data() * 1e6                      # µV, [trial, canal, tiempo] SIN normalizar
    inv = {v: k for k, v in ep.event_id.items()}
    y = np.array([inv[c] for c in ep.events[:, 2]])
    np.savez_compressed(
        CACHE / f"{sub}_full.npz",
        X=X.astype(np.float32), y=y,
        ch_names=np.array(ep.ch_names), sfreq=ep.info["sfreq"],
        tmin=ep.tmin, tmax=ep.tmax, subject=sub, cfg_hash=CFG_HASH,
    )
    counts = {k: int(np.sum(y == k)) for k in sorted(set(y.tolist()))}
    por_run = info.get("n_ica_por_run", [])
    return {
        "sub": sub, "n_trials": len(ep), "n_canales": len(ep.ch_names),
        "sfreq": ep.info["sfreq"], "tmin": ep.tmin, "tmax": ep.tmax,
        # OJO: el total suma los 3 runs. Se registra el desglose para no malinterpretarlo.
        "ica_total_3runs": info.get("n_ica_total", 0),
        "ica_por_run": json.dumps(por_run),
        "ica_media_por_run": round(float(np.mean(por_run)), 1) if por_run else 0,
        "ica_n_componentes": info.get("n_componentes_ica", 0),
        "ica_etiquetas": json.dumps(info.get("etiquetas_ica", {})),
        "clases": json.dumps(counts), "cfg_hash": CFG_HASH,
        "segundos": round(time.time() - t0, 1),
    }, X


def main():
    subs = sorted([p.name for p in BIDS_ROOT.glob("sub-*") if p.is_dir()])
    done = set()
    if LOG.exists():
        try:
            done = set(pd.read_csv(LOG)["sub"].tolist())
        except Exception:
            pass
    print(f"Sujetos: {len(subs)} | ya cacheados: {len(done)} | cfg {CFG_HASH}", flush=True)
    print(f"ICA activo: {CFG['use_ica']} | ventana {CFG['epoch_tmin']}..{CFG['epoch_tmax']}s "
          f"| {CFG['resample_hz']} Hz", flush=True)

    for i, s in enumerate(subs, 1):
        if s in done and (CACHE / f"{s}_full.npz").exists():
            print(f"[{i}/{len(subs)}] {s}: ya cacheado, se salta", flush=True)
            continue
        try:
            row, X = build(s)
            pd.DataFrame([row]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
            print(f"[{i}/{len(subs)}] {s}: X={X.shape} | ICA {row['ica_media_por_run']}/"
                  f"{row['ica_n_componentes']} por run | {row['segundos']}s | {row['clases']}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(subs)}] {s}: ERROR {type(e).__name__}: {e}", flush=True)

    if LOG.exists():
        d = pd.read_csv(LOG)
        print("\n===== CACHÉ DE TENSORES =====", flush=True)
        print(f"Sujetos cacheados: {len(d)}/{len(subs)}")
        print(f"Trials por sujeto: min {d['n_trials'].min()} | mediana {d['n_trials'].median():.0f} "
              f"| max {d['n_trials'].max()}  (total {int(d['n_trials'].sum())})")
        print(f"Canales: {d['n_canales'].unique().tolist()} | {d['sfreq'].unique().tolist()} Hz")
        print(f"Componentes ICA eliminados POR RUN: mediana {d['ica_media_por_run'].median():.1f} "
              f"| max {d['ica_media_por_run'].max():.1f} (de {d['ica_n_componentes'].iloc[0]})")
        print(f"Tiempo total: {d['segundos'].sum()/60:.1f} min")
        print(f"-> {CACHE}")
        print("\nSIGUIENTE: splits LOSO + normalización por fold (solo train) + test anti-fuga.")


if __name__ == "__main__":
    main()
