"""Dataset MIND 2026 — pipeline canónico e interfaz del contrato de `datasets/__init__.py`.

Extraído de `notebooks/01_mind_descarga_preprocesamiento_eeg.ipynb` (verificado sobre datos reales).
Carga BrainVision + eventos `events.tsv` BIDS -> anotaciones -> preprocesamiento -> épocas.
Los archivos originales (.vhdr/.eeg a 1000 Hz) nunca se modifican.

⚠️ ESTADO (2026-07-20): MIND queda como **dataset de contraste**, no principal.
El baseline EEGNet da acc 0.272 ± 0.036 (azar 0.25) en monousuario y **azar en LOSO**,
igual que FBCSP -> el cambio de dominio es infranqueable en este paradigma direccional.
Ver `HANDOFF.md` §9.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import mne

# --- Constantes del dataset (events.tsv BIDS) ---
MI_CLASSES = {"mi_l2r": "horizontal_L2R", "mi_t2b": "vertical_T2B",
              "mi_ul2lr": "diag_UL2LR", "mi_ur2ll": "diag_UR2LL"}

# --------------------------------------------------------------------------- #
# Interfaz del contrato (ver src/datasets/__init__.py)
# --------------------------------------------------------------------------- #
NAME = "mind"
# El caché guarda el VALOR de MI_CLASSES; el orden se fija por la clave canónica
# (mi_l2r, mi_t2b, mi_ul2lr, mi_ur2ll) para que el índice de clase sea estable.
CLASS_ORDER = [MI_CLASSES[k] for k in sorted(MI_CLASSES)]
N_CLASSES = len(CLASS_ORDER)
CHANCE = 1.0 / N_CLASSES
SFREQ = 200.0
WINDOW = (0.0, 4.0)
SESSIONS = None          # sesión única (3 runs del mismo día, no son sesiones distintas)
CACHE_DIR = Path(os.environ.get("MIND_PROC_ROOT",
                                r"D:/Tesis/data/processed/mind")) / "epochs_cache"
REST_TYPES = {"rest_open": "rest_EO", "rest_closed": "rest_EC"}
RUNS = ["run-01", "run-02", "run-03"]

# --- Conjuntos de canales (plan §A.4). Se aplican como VISTA sobre el tensor completo,
# nunca al cachear: así se conserva el montaje completo y se habilita el ablation. ---
CH_SETS = {
    "core_motor": ["C3", "Cz", "C4"],
    "motor_ext": ["C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                  "FC3", "FCz", "FC4", "CP3", "CPz", "CP4"],
    "central_amplio": ["F3", "Fz", "F4", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
                       "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                       "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "P3", "Pz", "P4"],
    "completo": None,   # los 60 canales
}


# Canales frontopolares/frontales: los más contaminados por artefacto ocular.
# Se EXCLUYEN del promedio de referencia (no se borran) para no repartir señal ocular
# al resto de canales vía el CAR. Ver docs/COMPARACION_PIPELINE_AUTORES.md.
FRONTAL_CH = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F8", "F5", "F6"]


def pick_channel_set(inst, name):
    """Devuelve una copia con el subconjunto de canales indicado (insensible a mayúsculas)."""
    names = CH_SETS[name]
    if names is None:
        return inst.copy()
    up = {c.upper(): c for c in inst.ch_names}
    present = [up[n.upper()] for n in names if n.upper() in up]
    return inst.copy().pick(present)

# --- Configuración por defecto (misma que el notebook 01) ---
DEFAULT_CFG = {
    "l_freq": 0.5, "h_freq": 40.0,
    "notch": [50.0, 100.0],
    "resample_hz": 200,
    "reference": "average",
    # El promedio de referencia se calcula EXCLUYENDO los canales frontales (FRONTAL_CH):
    # así el artefacto ocular que el ICA no captura (p.ej. sacadas direccionales) no se
    # reparte al resto de canales al restar la media. Los frontales SÍ se conservan.
    "ref_exclude_frontal": True,
    # CB1/CB2: sin coordenada en el 10-20 (romperían interpolación/autoreject).
    # M1/M2: mastoides; se marcaban como "malos" en casi todos los sujetos e interpolarlos desde
    # el cuero cabelludo no tiene sentido fisiológico. -> montaje final de 60 canales de scalp.
    "drop_channels": ["CB1", "CB2", "M1", "M2"],
    "mi_per_class_run": 10,
    "epoch_tmin": -2.0, "epoch_tmax": 10.0,
    "baseline_tmin": -2.0, "baseline_tmax": 0.0,
    "mi_tmin": 0.0, "mi_tmax": 4.0,
    "amp_reject_uv": 800.0,
    "flat_uv": 1e-1,
    "seed": 2026,
    # --- ICA para artefactos (2026-07-19: entra al NÚCLEO) ---
    # Control experimental: en sub-01 la ventaja de los canales frontales (0.470 vs 0.411 motores)
    # se invierte tras ICA (0.292 vs 0.351) -> había confundidor ocular y el ICA lo limpia.
    # Umbrales ALINEADOS con los autores (EEGPreprocess.m -> pop_icflag):
    #   Muscle 0.9-1 | Eye 0.8-1 | Channel noise 0.5-1 | Brain/Heart/Line/Other: no se rechazan.
    "use_ica": True,
    "ica_n_components": 30,
    "ica_drop_thresholds": {
        "eye blink": 0.8,
        "muscle artifact": 0.9,
        "channel noise": 0.5,
    },
}


def load_run(sub, run, bids_root, cfg=DEFAULT_CFG):
    """Carga un run EEG (BrainVision) con las anotaciones MI del events.tsv, recortadas a
    cfg['mi_per_class_run'] trials/clase (primeros por onset). Devuelve (raw, events_df)."""
    bids_root = Path(bids_root)
    stem = bids_root / sub / "eeg" / f"{sub}_task-mi2d_{run}"
    raw = mne.io.read_raw_brainvision(str(stem) + "_eeg.vhdr", preload=True, verbose="ERROR")
    ev = pd.read_csv(str(stem) + "_events.tsv", sep="\t")
    mi = ev[ev["trial_type"].isin(MI_CLASSES)].sort_values("onset")
    n = cfg.get("mi_per_class_run")
    if n:
        mi = mi.groupby("trial_type", group_keys=False).head(n).sort_values("onset")
    raw.set_annotations(mne.Annotations(onset=mi["onset"].values, duration=mi["duration"].values,
                                        description=mi["trial_type"].values))
    return raw, ev


def set_montage_1020(raw, cfg=DEFAULT_CFG):
    montage = mne.channels.make_standard_montage("standard_1020")
    std = {c.lower(): c for c in montage.ch_names}
    rename = {ch: std[ch.lower()] for ch in raw.ch_names if ch.lower() in std and std[ch.lower()] != ch}
    if rename:
        raw.rename_channels(rename)
    drop = [c for c in cfg.get("drop_channels", []) if c in raw.ch_names]
    if drop:
        raw.drop_channels(drop)
    raw.set_montage(montage, on_missing="ignore")
    pos = raw.get_montage().get_positions()["ch_pos"]
    no_pos = [c for c in raw.ch_names if c not in pos or not np.isfinite(np.asarray(list(pos[c]))).all()]
    if no_pos:
        raw.drop_channels(no_pos)
    return raw


def detect_bad_channels(raw, cfg=DEFAULT_CFG, z=4.0):
    data = raw.get_data() * 1e6
    sd = data.std(axis=1)
    bads = [raw.ch_names[i] for i in range(len(sd)) if sd[i] < cfg["flat_uv"]]
    med = np.median(sd)
    mad = np.median(np.abs(sd - med)) + 1e-9
    bads += [raw.ch_names[i] for i in range(len(sd)) if abs(sd[i] - med) / (1.4826 * mad) > z]
    return sorted(set(bads))


def preprocess_core(raw, cfg=DEFAULT_CFG):
    """Montaje -> canales malos -> filtro cero-fase -> notch -> CAR -> (remuestreo opcional)."""
    raw = set_montage_1020(raw.copy(), cfg)
    raw.info["bads"] = detect_bad_channels(raw, cfg)
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=True)
    raw.filter(cfg["l_freq"], cfg["h_freq"], method="fir", phase="zero", fir_design="firwin", verbose="ERROR")
    raw.notch_filter(cfg["notch"], verbose="ERROR")
    # Referencia. Si ref_exclude_frontal=True, el promedio se calcula SOLO con canales no frontales
    # (los frontales se conservan en los datos, pero no contaminan la referencia del resto).
    if cfg.get("ref_exclude_frontal") and cfg["reference"] == "average":
        frontal = {c.upper() for c in FRONTAL_CH}
        ref_ch = [c for c in raw.ch_names if c.upper() not in frontal]
        raw.set_eeg_reference(ref_channels=ref_ch, verbose="ERROR")
    else:
        raw.set_eeg_reference(cfg["reference"], verbose="ERROR")
    if cfg["resample_hz"]:
        raw.resample(cfg["resample_hz"])
    return raw


def remove_artifact_ica(raw, cfg=DEFAULT_CFG):
    """Elimina componentes de artefacto con ICA + ICLabel (automático, sin inspección manual).

    Justificación (control experimental, 2026-07-19): sin ICA los canales frontales decodificaban
    MEJOR que los motores (0.410 vs 0.347) -> el clasificador podía estar leyendo los ojos en vez
    del cerebro, un confundidor grave en un paradigma DIRECCIONAL. Tras el ICA la ventaja frontal
    se invierte y la señal motora se mantiene.

    Umbrales por clase alineados con los autores del dataset (ver
    docs/COMPARACION_PIPELINE_AUTORES.md): ojo 0.8, músculo 0.9, ruido de canal 0.5.

    Devuelve (raw_limpio, n_componentes_excluidos, etiquetas).
    """
    from mne.preprocessing import ICA
    try:
        from mne_icalabel import label_components
    except ImportError:
        print("  [AVISO] mne-icalabel no instalado -> se omite el ICA.")
        return raw, 0, []
    # ICLabel exige referencia promedio (ya aplicada) y paso alto >= 1 Hz para el ajuste.
    raw_fit = raw.copy().filter(1.0, None, verbose="ERROR")
    n_comp = min(cfg.get("ica_n_components", 30), len(raw.ch_names) - 2)
    ica = ICA(n_components=n_comp, method="infomax", fit_params=dict(extended=True),
              max_iter="auto", random_state=cfg["seed"])
    ica.fit(raw_fit, verbose="ERROR")
    res = label_components(raw_fit, ica, method="iclabel")
    labels = res["labels"]
    probs = np.asarray(res["y_pred_proba"])
    # Umbral POR CLASE (dict etiqueta -> probabilidad mínima). Insensible a mayúsculas.
    thr = {k.lower(): v for k, v in cfg.get("ica_drop_thresholds", {}).items()}
    excl = [i for i, (lab, p) in enumerate(zip(labels, probs))
            if lab.lower() in thr and p >= thr[lab.lower()]]
    ica.exclude = excl
    return ica.apply(raw.copy(), verbose="ERROR"), len(excl), [labels[i] for i in excl]


def make_epochs(raw_clean, cfg=DEFAULT_CFG):
    events, ann_id = mne.events_from_annotations(raw_clean, verbose="ERROR")
    event_id = {MI_CLASSES[k]: v for k, v in ann_id.items() if k in MI_CLASSES}
    return mne.Epochs(raw_clean, events, event_id=event_id,
                      tmin=cfg["epoch_tmin"], tmax=cfg["epoch_tmax"],
                      baseline=None, preload=True, reject=None, verbose="ERROR")


def reject_gross(epochs, cfg=DEFAULT_CFG):
    ep = epochs.copy().drop_bad(reject={"eeg": cfg["amp_reject_uv"] * 1e-6},
                                flat={"eeg": cfg["flat_uv"] * 1e-6}, verbose="ERROR")
    return ep


def maybe_autoreject(epochs, cfg=DEFAULT_CFG):
    import importlib
    if importlib.util.find_spec("autoreject") is None:
        return epochs
    from autoreject import AutoReject
    ar = AutoReject(random_state=cfg["seed"], n_jobs=1, verbose=False)
    return ar.fit_transform(epochs)


def process_subject_epochs(sub, bids_root, cfg=DEFAULT_CFG, runs=RUNS, do_autoreject=True,
                           return_info=False):
    """Procesa los runs de un sujeto y concatena las épocas MI limpias (sin guardar).

    Cadena: carga -> montaje/canales malos -> filtro/notch/CAR -> remuestreo
            -> [ICA ocular si cfg['use_ica']] -> épocas -> rechazo -> [autoreject].
    """
    ep_list, n_ica, labs, por_run = [], 0, [], []
    for run in runs:
        raw, _ = load_run(sub, run, bids_root, cfg)
        rc = preprocess_core(raw, cfg)
        if cfg.get("use_ica"):
            rc, n_ex, lb = remove_artifact_ica(rc, cfg)
            n_ica += n_ex          # OJO: total sumado sobre los runs, no por run
            por_run.append(n_ex)   # desglose por run (evita malinterpretar el total)
            labs += lb
        ep = reject_gross(make_epochs(rc, cfg), cfg)
        ep_list.append(maybe_autoreject(ep, cfg) if do_autoreject else ep)
    epochs = mne.concatenate_epochs(ep_list)
    if return_info:
        from collections import Counter
        return epochs, {"n_ica_total": n_ica, "n_ica_por_run": por_run,
                        "n_componentes_ica": cfg.get("ica_n_components", 30),
                        "etiquetas_ica": dict(Counter(labs))}
    return epochs
