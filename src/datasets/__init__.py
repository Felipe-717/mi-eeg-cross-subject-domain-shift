"""Registro de datasets — CONTRATO que permite reutilizar todo el código de entrenamiento.

Cada módulo de dataset (`mind.py`, `openbmi.py`, ...) expone la misma interfaz, de modo
que `src/splits.py`, `src/training.py` y los scripts genéricos funcionan con cualquiera
sin cambios. Lo ÚNICO específico de cada dataset vive en su módulo.

Esto no es cosmética: la "comparación justa" de la tesis exige que el código de
entrenamiento sea IDÉNTICO entre datasets y arquitecturas. Si se bifurcara en copias,
los resultados dejarían de ser comparables y cada bug habría que arreglarlo dos veces.

INTERFAZ OBLIGATORIA de un módulo de dataset:

    NAME         str        identificador corto ("mind", "openbmi")
    CLASS_ORDER  list[str]  etiquetas en orden FIJO -> define el índice de cada clase
    N_CLASSES    int        len(CLASS_ORDER)
    CHANCE       float      1 / N_CLASSES
    SFREQ        float      frecuencia de muestreo del caché
    WINDOW       tuple      (tmin, tmax) de la ventana de modelado, en segundos
    CH_SETS      dict       vistas de canales {nombre: [canales] | None para todos}
    CACHE_DIR    Path       directorio del caché de tensores
    SESSIONS     list|None  identificadores de sesión, o None si el dataset es de sesión única

Uso:
    from datasets import get
    ds = get("openbmi")
    X, y, chs = splits.load_subject(sub, ds.CACHE_DIR, window=ds.WINDOW)
"""
from __future__ import annotations
import importlib

_DISPONIBLES = ("openbmi", "wbcic_2c", "wbcic_3c", "mind")
#                 ^principal   ^réplica    ^réplica 2ª  ^contraste (DSI no discrimina)

_ATRIBUTOS_REQUERIDOS = (
    "NAME", "CLASS_ORDER", "N_CLASSES", "CHANCE",
    "SFREQ", "WINDOW", "CH_SETS", "CACHE_DIR", "SESSIONS",
)


def get(name: str):
    """Devuelve el módulo del dataset, verificando que cumpla el contrato."""
    if name not in _DISPONIBLES:
        raise ValueError(f"dataset desconocido: {name!r}. Disponibles: {_DISPONIBLES}")
    mod = importlib.import_module(f"datasets.{name}")
    faltan = [a for a in _ATRIBUTOS_REQUERIDOS if not hasattr(mod, a)]
    if faltan:
        raise AttributeError(
            f"El módulo de dataset '{name}' no cumple el contrato: faltan {faltan}. "
            f"Ver la documentación en src/datasets/__init__.py"
        )
    if mod.N_CLASSES != len(mod.CLASS_ORDER):
        raise ValueError(f"'{name}': N_CLASSES={mod.N_CLASSES} no coincide con "
                         f"len(CLASS_ORDER)={len(mod.CLASS_ORDER)}")
    return mod


def disponibles() -> tuple[str, ...]:
    return _DISPONIBLES
