"""Constantes compartidas del dataset WBCIC-MI 2025 (subconjuntos 2C y 3C).

Referencia: *A multi-day and high-quality EEG dataset for motor imagery brain-computer
interface*, Scientific Data, 23-mar-2025. DOI 10.1038/s41597-025-04826-y
Datos: Figshare `10.25452/figshare.plus.22671172` · CC BY 4.0 · formato EEG-BIDS
Origen: 2019 World Robot Conference Contest — BCI Robot Contest

⚠️ **NO CONFUNDIR con el `SHU Dataset`** (Ma et al. 2022: 25 sujetos, 2 clases, 32 ch),
que está PROHIBIDO porque EEGMamba lo usa. Comparten las siglas de Shanghai University y
son conjuntos distintos. `PLAN_MAESTRO_SOFTWARE_TESIS_V3.md` lo llama "WBCIC-SHU".
**Identificar siempre por DOI.**

VERIFICACIÓN ANTI-FUGA (hecha en PLAN_MAESTRO_V3 §133-135, reconfirmada 2026-07-20):
  * No figura en el **Apéndice D de LaBraM** (que sí lista BCI Competition IV-1 y
    EEG Motor Movement/Imagery Dataset entre sus 17 datasets de preentrenamiento).
  * No figura en el `pretrain_main.py` de EEGMamba (TUEG, PhysioNet 2018, Raw EEG Data,
    Siena, SZBD) ni entre sus datasets de MI downstream (BCI-IV-2a, Shu).
  * Publicado DESPUÉS de LaBraM (may-2024) y EEGMamba (jul-2024).
  Formulación defendible: "no se encontró en las fuentes declaradas". NUNCA "garantizado".

⚠️ SUS BASELINES SON WITHIN-SUJETO / WITHIN-SESIÓN (10-fold CV, modelos por sujeto,
sesiones evaluadas por separado y promediadas). **No reporta cross-subject.** Es el
escenario más fácil que existe; no es comparable con LOSO.
"""
from __future__ import annotations
import os
from pathlib import Path

DOI_ARTICULO = "10.1038/s41597-025-04826-y"
DOI_DATOS = "10.25452/figshare.plus.22671172"
LICENCIA = "CC BY 4.0"

SFREQ = 200.0                 # 1000 Hz originales; 200 Hz por exigencia de LaBraM
WINDOW = (0.0, 4.0)
SESSIONS = ["session_1", "session_2", "session_3"]   # 3 días distintos -> eje temporal

_ROOT = Path(os.environ.get("WBCIC_PROC_ROOT", r"D:/Tesis/data/processed/wbcic"))

# 59 canales EEG (los 64 totales incluyen ECG/EOG, que se excluyen). La validación
# técnica del paper usa 58. CONFIRMAR nombres reales al descargar.
CH_SETS = {
    "core_motor": ["C3", "Cz", "C4"],
    "motor_ext": ["C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                  "FC3", "FCz", "FC4", "CP3", "CPz", "CP4"],
    # Vista COMPARTIDA con el otro dataset (punto 6: validacion cruzada).
    # Interseccion real de los dos montajes, calculada el 2026-09-06: 45 de los
    # 62 de OpenBMI y de los 59 de WBCIC-SHU. El ORDEN es el de OpenBMI y tiene
    # que ser identico en ambos ficheros: la primera capa de las cuatro
    # arquitecturas es espacial, asi que la posicion de cada canal es parte del
    # modelo. Fuera quedan los 17 de OpenBMI sin equivalente (TP9/TP10, PO9/PO10,
    # CPz, P1/P2, la serie FTT/TPP/TTP de alta densidad y F9/F10) y los 14 de
    # WBCIC sin equivalente (Fpz, F1/F2/F5/F6, FCz, FT7/FT8, P5/P6, PO5-PO8).
    "comun_wbcic": [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3",
        "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "O1",
        "Oz", "O2", "FC3", "FC4", "C5", "C1", "C2", "C6", "CP3", "CP4", "POz", "TP7",
        "TP8", "AF7", "AF3", "AF4", "AF8", "PO3", "PO4"
    ],
    # Subconjunto motor de la anterior. Es `motor_ext` de OpenBMI SIN CPz, que
    # no existe en WBCIC (alli hay FCz, que a su vez no existe en OpenBMI).
    "motor_comun_wbcic": [
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "FC3", "FC4", "CP3", "CP4"
    ],
    "completo": None,
}

# Configuración de los autores, solo para la puerta de validación (no es la del estudio)
AUTHORS_CFG = {"resample_hz": 250, "window_s": 4.0, "n_channels": 58}

# ⚠️ El sujeto mejora entre sesiones (efecto de aprendizaje BCI): la sesión 3 rinde más.
# Eso hace el leave-one-session-out ASIMÉTRICO -> hay que rotar y reportar POR SESIÓN,
# no solo el promedio. Ver docs/PLAN_MIGRACION_WBCIC.md §3.4.
EFECTO_APRENDIZAJE = True
