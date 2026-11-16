"""WBCIC-MI 2025, subconjunto **3C**: 11 sujetos, mano izq / mano der / pie.

Ver `_wbcic_base.py` para la ficha del dataset y la verificación anti-fuga.

    11 sujetos · 3 sesiones · 300 trials/sesión = 900 trials/sujeto (300 por clase)

BASELINE PUBLICADO (within-sujeto / within-sesión, 10-fold CV):
    deepConvNet 76.90 %  ·  EEGNet 75.34 %  ·  FBCNet 74.77 %      (azar 33.3 %)

⚠️ **Era el dataset principal del `PLAN_MAESTRO_SOFTWARE_TESIS_V3.md`** (que descartaba
explícitamente el 2C) hasta que MIND lo desplazó el 2026-07-18.

⚠️ **LIMITACIÓN CRÍTICA — solo 11 sujetos.** La unidad de inferencia estadística es el
sujeto: con k=4 arquitecturas la diferencia crítica de Nemenyi es **1.41** (frente a 0.66
con los 51 del 2C), y arquitecturas parecidas suelen separarse 0.5-1.0 puestos de rango
-> **el post-hoc probablemente no detectaría nada**. El propio plan V3 §166 ya avisaba:
"WBCIC-SHU 3C aporta 11 observaciones pareadas por modelo".
Recomendación: usar 3C como **réplica secundaria**, no como principal.
"""
from __future__ import annotations
from ._wbcic_base import (      # noqa: F401  (se reexportan a propósito)
    DOI_ARTICULO, DOI_DATOS, LICENCIA, SFREQ, WINDOW, SESSIONS, CH_SETS,
    AUTHORS_CFG, EFECTO_APRENDIZAJE, _ROOT,
)

NAME = "wbcic_3c"
CLASS_ORDER = ["left_hand", "right_hand", "foot"]
N_CLASSES = len(CLASS_ORDER)
CHANCE = 1.0 / N_CLASSES                 # 0.333
CACHE_DIR = _ROOT / "3c" / "epochs_cache"

N_SUBJECTS = 11
TRIALS_POR_SESION = 300
AUTHORS_REF = {"deepconvnet": 0.7690, "eegnet": 0.7534, "fbcnet": 0.7477}
