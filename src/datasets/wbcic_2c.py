"""WBCIC-MI 2025, subconjunto **2C**: 51 sujetos, mano izquierda vs derecha.

Ver `_wbcic_base.py` para la ficha del dataset y la verificación anti-fuga.

    51 sujetos · 3 sesiones · 200 trials/sesión = 600 trials/sujeto (300 por clase)

BASELINE PUBLICADO (within-sujeto / within-sesión, 10-fold CV):
    EEGNet 85.31 %  ·  deepConvNet 84.47 %  ·  FBCNet 78.40 %      (azar 50 %)

POR QUÉ 2C Y NO 3C (ver docs/PLAN_MIGRACION_WBCIC.md §2):
la unidad de inferencia estadística es el SUJETO. Con k=4 arquitecturas, la diferencia
crítica de Nemenyi es 0.66 con 51 sujetos frente a 1.41 con los 11 del subconjunto 3C
—es decir, con 3C el test post-hoc probablemente no detectaría ninguna diferencia—.
Además el LOSO entrena con 3,3x más datos. Los trials POR CLASE empatan (~90 en ambos).
"""
from __future__ import annotations
from ._wbcic_base import (      # noqa: F401  (se reexportan a propósito)
    DOI_ARTICULO, DOI_DATOS, LICENCIA, SFREQ, WINDOW, SESSIONS, CH_SETS,
    AUTHORS_CFG, EFECTO_APRENDIZAJE, _ROOT,
)

NAME = "wbcic_2c"
CLASS_ORDER = ["left_hand", "right_hand"]
N_CLASSES = len(CLASS_ORDER)
CHANCE = 1.0 / N_CLASSES                 # 0.50
CACHE_DIR = _ROOT / "2c" / "epochs_cache"

N_SUBJECTS = 51
TRIALS_POR_SESION = 200
AUTHORS_REF = {"eegnet": 0.8531, "deepconvnet": 0.8447, "fbcnet": 0.7840}
