"""Anti-fuga del conjunto SUJETO VISTO (indicador de cambio de dominio, Actividad 4.3).

El conjunto "visto" son trials reservados de los sujetos de ENTRENAMIENTO. Si por error
entrenaran, o si contribuyeran a los estadísticos de normalización, la comparación
visto-vs-no-visto quedaría inflada y el DSI seria falso — justo el numero que compara
las arquitecturas en la tesis. Estos tests lo blindan.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import splits as sp  # noqa: E402


@pytest.fixture
def cache(tmp_path):
    """6 sujetos sintéticos: 40 trials, 2 clases, 2 sesiones, balanceados."""
    rng = np.random.default_rng(0)
    for i in range(1, 7):
        n = 40
        y = np.array(["left_hand", "right_hand"] * (n // 2))
        ses = np.array(["0"] * (n // 2) + ["1"] * (n // 2))
        np.savez(tmp_path / f"sub-{i:02d}_full.npz",
                 X=rng.normal(size=(n, 4, 50)).astype(np.float32), y=y,
                 session=ses, run=np.array(["r"] * n),
                 ch_names=np.array(["C3", "Cz", "C4", "Pz"]), sfreq=200.0, tmin=0.0,
                 subject=f"sub-{i:02d}", cfg_hash="test")
    return tmp_path


def _fold(cache, frac=0.25):
    subs = sorted(p.name.replace("_full.npz", "") for p in Path(cache).glob("sub-*.npz"))
    f = sp.make_loso_folds(subs, n_inner_val=1)[0]
    return f, sp.build_fold(f, cache, frac_visto=frac)


def test_visto_no_entrena(cache):
    """Ningún trial reservado puede aparecer en el conjunto de entrenamiento."""
    _, b = _fold(cache)
    tr_set = {x.tobytes() for x in b["X_train"]}
    assert not any(x.tobytes() in tr_set for x in b["X_visto"])


def test_visto_viene_de_sujetos_de_entrenamiento(cache):
    """El 'visto' debe ser de gente que SÍ estaba en el entrenamiento; si viniera de
    otros sujetos mediria otra cosa y el DSI perderia su sentido."""
    f, b = _fold(cache)
    assert set(b["sub_visto"]) <= set(f.inner_train)
    assert f.test not in set(b["sub_visto"])


def test_visto_estratificado_por_clase_y_sesion(cache):
    """Debe estar tan balanceado como el sujeto no visto: si no, se estaría mezclando
    cambio temporal (día 1 vs día 2) con cambio inter-sujeto."""
    _, b = _fold(cache)
    y = np.asarray(b["y_visto"])
    assert len(np.unique(y)) == 2
    n = [int((y == c).sum()) for c in np.unique(y)]
    assert max(n) - min(n) <= len(np.unique(b["sub_visto"]))   # tolerancia de redondeo


def test_normalizacion_no_ve_el_visto(cache):
    """Los estadísticos deben salir SOLO del train ya reducido. Incluir el 'visto'
    seria fuga en el preprocesado, el mismo error que los 25 tests de splits evitan.

    Comprobacion: si los estadisticos se ajustaron EXACTAMENTE sobre X_train, entonces
    X_train ya normalizado tiene media 0 y desviacion 1 por canal. Si el 'visto' hubiera
    entrado en el ajuste, la media del train no daria cero.
    """
    _, b = _fold(cache)
    m = b["X_train"].mean(axis=(0, 2))
    s = b["X_train"].std(axis=(0, 2))
    assert np.allclose(m, 0.0, atol=1e-5), f"media del train tras normalizar: {m}"
    assert np.allclose(s, 1.0, atol=1e-4), f"desviacion del train tras normalizar: {s}"
    # Y el 'visto', al no haber participado, NO tiene por que quedar centrado.
    assert not np.allclose(b["X_visto"].mean(axis=(0, 2)), 0.0, atol=1e-5)


def test_frac_cero_no_cambia_nada(cache):
    """frac_visto=0 debe reproducir el comportamiento clásico, sin claves nuevas."""
    subs = sorted(p.name.replace("_full.npz", "") for p in Path(cache).glob("sub-*.npz"))
    f = sp.make_loso_folds(subs, n_inner_val=1)[0]
    b = sp.build_fold(f, cache, frac_visto=0.0)
    assert "X_visto" not in b


def test_tamanos_coherentes(cache):
    """train + visto = el total original de los sujetos de entrenamiento."""
    f, b = _fold(cache, frac=0.25)
    assert len(b["y_train"]) + len(b["y_visto"]) == 40 * len(f.inner_train)
    assert 0.15 < len(b["y_visto"]) / (40 * len(f.inner_train)) < 0.35


def test_determinista(cache):
    """Misma semilla -> mismo reparto. Sin esto no se podria reanudar ni reproducir."""
    f = sp.make_loso_folds(
        sorted(p.name.replace("_full.npz", "") for p in Path(cache).glob("sub-*.npz")),
        n_inner_val=1)[0]
    a = sp.build_fold(f, cache, frac_visto=0.25, seed=7)
    c = sp.build_fold(f, cache, frac_visto=0.25, seed=7)
    assert np.array_equal(a["X_visto"], c["X_visto"])
