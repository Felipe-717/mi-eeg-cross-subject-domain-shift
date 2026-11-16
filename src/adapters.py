"""Actividad 2.4 — Representaciones de entrada por arquitectura.

Punto de partida ÚNICO para todos los modelos: el tensor canónico del caché
    X : (n_trials, n_canales, n_muestras)  en µV, SIN normalizar, 60 ch @ 200 Hz

Cada adaptador es un paso **fijo y documentado**, NO un hiperparámetro que se ajusta.
Así se preserva la comparabilidad exigida por el anteproyecto ("mismo preprocesamiento
para todos los modelos"); lo único que cambia es la transformación final de entrada.

ORDEN CORRECTO respecto a la normalización:
    caché (µV, sin normalizar)
      -> split LOSO
      -> z-score por fold (estadísticos SOLO de train)   <- src/splits.py
      -> adaptador de la arquitectura                    <- este módulo
      -> modelo
"""
from __future__ import annotations
import numpy as np

# --------------------------------------------------------------------------- #
# LaBraM
# --------------------------------------------------------------------------- #
# Vocabulario de canales copiado VERBATIM de LaBraM/utils.py (repo oficial
# 935963004/LaBraM). El modelo indexa los canales con standard_1020.index(nombre)+1,
# por lo que un nombre ausente lanza ValueError.
LABRAM_STANDARD_1020 = [
    'FP1', 'FPZ', 'FP2',
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]

LABRAM_SFREQ = 200        # verificado: raw.resample(200) en dataset_maker/make_TUAB.py
LABRAM_PATCH = 200        # 1 segundo por parche (input_size=200 en run_class_finetuning.py)

# ✅ VERIFICADO en el repo oficial (LaBraM/engine_for_finetuning.py:64-65):
#     samples = samples.float().to(device, non_blocking=True) / 100
#     samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
# Los datos llegan en µV (dataset_maker: raw.get_data(units='uV')) y se dividen entre 100
# -> unidades de 0.1 mV. La forma final es (B, canales, parches, 200), que es justo la
# que produce adapt_labram().
LABRAM_UV_SCALE = 100.0


def labram_input_chans(ch_names) -> list[int]:
    """Réplica exacta de LaBraM/utils.py::get_input_chans.

    Antepone 0 (token CLS) y mapea cada canal a standard_1020.index(nombre)+1.
    Los nombres se pasan a MAYÚSCULAS: MNE devuelve 'Cz'/'FCz'/'Fp1' pero el
    vocabulario de LaBraM usa 'CZ'/'FCZ'/'FP1' -> sin .upper() lanza ValueError
    (verificado: 10 de nuestros 60 canales fallan sin esta conversión).
    """
    chans = [0]
    faltan = [c for c in ch_names if c.upper() not in LABRAM_STANDARD_1020]
    if faltan:
        raise ValueError(f"Canales fuera del vocabulario de LaBraM: {faltan}")
    for c in ch_names:
        chans.append(LABRAM_STANDARD_1020.index(c.upper()) + 1)
    return chans


def adapt_labram(X, ch_names, sfreq=LABRAM_SFREQ, patch=LABRAM_PATCH,
                 uv_scale=LABRAM_UV_SCALE):
    """Convierte (n_trials, n_ch, n_muestras) -> (n_trials, n_ch, n_parches, patch).

    Devuelve (X_parches, input_chans). La señal debe llegar a 200 Hz.
    Si el nº de muestras no es múltiplo del parche, se recorta el sobrante final
    (mejor recortar de forma explícita que rellenar con ceros inventados).
    """
    assert sfreq == LABRAM_SFREQ, f"LaBraM espera {LABRAM_SFREQ} Hz y llegó {sfreq}"
    X = np.asarray(X, dtype=np.float32)
    n_tr, n_ch, n_t = X.shape
    assert n_ch == len(ch_names), "n_canales no coincide con ch_names"

    n_patches = n_t // patch
    if n_patches < 1:
        raise ValueError(f"La ventana ({n_t} muestras) es menor que un parche ({patch})")
    sobra = n_t - n_patches * patch
    if sobra:
        X = X[:, :, :n_patches * patch]

    if uv_scale:
        X = X / float(uv_scale)
    Xp = X.reshape(n_tr, n_ch, n_patches, patch)
    return Xp, labram_input_chans(ch_names)


# --------------------------------------------------------------------------- #
# EEGNet / EEG Conformer (braindecode) — señal cruda
# --------------------------------------------------------------------------- #

def adapt_raw(X, ch_names=None, add_channel_dim=True):
    """Entrada cruda C×T para EEGNet y Conformer.

    braindecode espera (batch, n_canales, n_muestras); algunas implementaciones
    añaden una dimensión de "canal de imagen": (batch, 1, n_canales, n_muestras).
    """
    X = np.asarray(X, dtype=np.float32)
    return X[:, None, :, :] if add_channel_dim else X


# --------------------------------------------------------------------------- #
# EEGMamba — pendiente
# --------------------------------------------------------------------------- #

EEGMAMBA_SFREQ = 200      # verificado: in_dim=200 (parche de 1 s) en model_for_bciciv2a.py
EEGMAMBA_PATCH = 200
EEGMAMBA_UV_SCALE = 100.0  # verificado: datasets/bciciv2a_dataset.py -> "return data/100"


def adapt_eegmamba(X, ch_names=None, sfreq=EEGMAMBA_SFREQ, patch=EEGMAMBA_PATCH,
                   uv_scale=EEGMAMBA_UV_SCALE):
    """Convierte (n_trials, n_ch, n_muestras) -> (n_trials, n_ch, n_parches, patch).

    ✅ VERIFICADO en el repo oficial (wjq-learning/EEGMamba):
      - models/eegmamba.py:43  ->  `bz, ch_num, seq_len, patch_size = x.shape`
        es decir, la MISMA forma que LaBraM: (B, canales, parches, 200).
      - models/model_for_bciciv2a.py -> EEGMamba(in_dim=200, ...) y clasificador
        `nn.Linear(22 * 4 * 200, ...)`: 22 canales × 4 parches × 200 muestras,
        o sea 4 s a 200 Hz (justo nuestra ventana de modelado).
      - datasets/bciciv2a_dataset.py -> `return data/100` (misma escala que LaBraM).

    Diferencia con LaBraM: EEGMamba NO necesita lista de índices de canal — su módulo
    ST-Adaptive acepta cualquier número de canales. Por eso devuelve solo el tensor.
    """
    assert sfreq == EEGMAMBA_SFREQ, f"EEGMamba espera {EEGMAMBA_SFREQ} Hz y llegó {sfreq}"
    X = np.asarray(X, dtype=np.float32)
    n_tr, n_ch, n_t = X.shape
    n_patches = n_t // patch
    if n_patches < 1:
        raise ValueError(f"La ventana ({n_t} muestras) es menor que un parche ({patch})")
    if n_t - n_patches * patch:
        X = X[:, :, :n_patches * patch]
    if uv_scale:
        X = X / float(uv_scale)
    return X.reshape(n_tr, n_ch, n_patches, patch)


# --------------------------------------------------------------------------- #
# Registro
# --------------------------------------------------------------------------- #
ADAPTERS = {
    "eegnet": adapt_raw,
    "conformer": adapt_raw,
    "labram": adapt_labram,
    "eegmamba": adapt_eegmamba,
}


def describe():
    """Resumen legible de los requisitos de entrada por arquitectura."""
    return {
        "eegnet":    {"fs": 200, "forma": "(B, 1, C, T)", "escala": "z-score por fold", "estado": "listo"},
        "conformer": {"fs": 200, "forma": "(B, 1, C, T)", "escala": "z-score por fold", "estado": "listo"},
        "labram":    {"fs": 200, "forma": "(B, C, n_parches, 200)",
                      "escala": f"z-score por fold + /{LABRAM_UV_SCALE:.0f}",
                      "canales": "MAYÚSCULAS, mapeados a standard_1020 + token CLS",
                      "estado": "listo (verificado contra el repo)"},
        "eegmamba":  {"fs": 200, "forma": "(B, C, n_parches, 200)",
                      "escala": f"z-score por fold + /{EEGMAMBA_UV_SCALE:.0f}",
                      "canales": "libres (ST-Adaptive)",
                      "estado": "listo (verificado contra el repo)"},
    }
