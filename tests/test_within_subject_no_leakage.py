"""Tests anti-fuga del protocolo within-subject (5-fold CV, Anexo B.1).

Si alguno de estos tests falla, los resultados within-subject de la tesis NO son válidos.
Ejecutar:  conda run -n eeg python -m pytest tests/ -v
           (o directamente: conda run -n eeg python tests/test_within_subject_no_leakage.py)
"""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import splits as sp

RNG = np.random.default_rng(0)
N_TRIALS = 118            # típico de un sujeto MIND (mediana 114-120)
Y = RNG.integers(0, 4, size=N_TRIALS)


# --------------------------------------------------------------------------- #
# 1. Disyunción por trial
# --------------------------------------------------------------------------- #

def test_hay_un_fold_por_n_folds():
    folds = sp.make_within_subject_folds(Y, n_folds=5)
    assert len(folds) == 5


def test_test_nunca_aparece_en_entrenamiento_ni_validacion():
    """La regla más importante: los índices de test son intocables."""
    for f in sp.make_within_subject_folds(Y, n_folds=5):
        s_te = set(f.test_idx.tolist())
        assert not (s_te & set(f.inner_train_idx.tolist())), f"FUGA: test en inner_train (fold {f.idx})"
        assert not (s_te & set(f.inner_val_idx.tolist())), f"FUGA: test en inner_val (fold {f.idx})"


def test_particiones_disjuntas_y_completas():
    for f in sp.make_within_subject_folds(Y, n_folds=5):
        f.assert_disjoint()
        todos = set(f.test_idx.tolist()) | set(f.inner_val_idx.tolist()) | set(f.inner_train_idx.tolist())
        assert todos == set(range(N_TRIALS))
        assert len(f.test_idx) + len(f.inner_val_idx) + len(f.inner_train_idx) == N_TRIALS


def test_los_5_test_folds_cubren_todos_los_trials_sin_solape():
    """Unión de los test_idx de los 5 folds = todos los trials, sin repetir (CV clásica)."""
    folds = sp.make_within_subject_folds(Y, n_folds=5)
    vistos = []
    for f in folds:
        vistos.extend(f.test_idx.tolist())
    assert sorted(vistos) == list(range(N_TRIALS))


# --------------------------------------------------------------------------- #
# 2. Determinismo
# --------------------------------------------------------------------------- #

def test_misma_semilla_mismos_folds():
    a = sp.make_within_subject_folds(Y, n_folds=5, seed=2026)
    b = sp.make_within_subject_folds(Y, n_folds=5, seed=2026)
    for x, y in zip(a, b):
        assert np.array_equal(x.test_idx, y.test_idx)
        assert np.array_equal(x.inner_train_idx, y.inner_train_idx)
        assert np.array_equal(x.inner_val_idx, y.inner_val_idx)


def test_semillas_distintas_cambian_los_folds():
    a = sp.make_within_subject_folds(Y, n_folds=5, seed=1)
    b = sp.make_within_subject_folds(Y, n_folds=5, seed=2)
    assert any(not np.array_equal(x.test_idx, y.test_idx) for x, y in zip(a, b))


# --------------------------------------------------------------------------- #
# 3. Estratificación (balance de clases)
# --------------------------------------------------------------------------- #

def test_estratificacion_por_clase():
    """Cada fold de test debe conservar aprox. la proporción de clases global."""
    prop_global = np.bincount(Y, minlength=4) / len(Y)
    for f in sp.make_within_subject_folds(Y, n_folds=5):
        prop_fold = np.bincount(Y[f.test_idx], minlength=4) / len(f.test_idx)
        assert np.abs(prop_fold - prop_global).max() < 0.25, f"fold {f.idx} muy desbalanceado"


# --------------------------------------------------------------------------- #
# 3.bis Within POR SESIÓN — el fold NO puede cruzar el eje temporal (OpenBMI, §10)
# --------------------------------------------------------------------------- #

# Sujeto OpenBMI sintético: 400 trials, sesiones '0'/'1' de 200 cada una, 2 clases balanceadas.
_SES = np.array(["0"] * 200 + ["1"] * 200)
_Y2 = np.array(([0, 1] * 100) + ([0, 1] * 100))


def test_por_sesion_genera_n_folds_por_cada_sesion():
    folds = sp.make_within_session_folds(_Y2, _SES, n_folds=10)
    assert len(folds) == 20                       # 10 folds x 2 sesiones
    assert {f.session for f in folds} == {"0", "1"}
    assert sum(f.session == "0" for f in folds) == 10


def test_ningun_fold_mezcla_sesiones():
    """El test clave: todos los índices de un fold pertenecen a UNA sola sesión."""
    for f in sp.make_within_session_folds(_Y2, _SES, n_folds=10):
        idx = np.concatenate([f.test_idx, f.inner_val_idx, f.inner_train_idx])
        sesiones = set(_SES[idx].tolist())
        assert sesiones == {f.session}, f"fold {f.idx} cruza sesiones: {sesiones}"


def test_por_sesion_disjunto_y_cubre_cada_sesion():
    folds = sp.make_within_session_folds(_Y2, _SES, n_folds=10)
    for ses in ("0", "1"):
        abs_ses = set(np.where(_SES == ses)[0].tolist())
        test_union = []
        for f in folds:
            if f.session != ses:
                continue
            f.assert_disjoint()
            todos = set(f.test_idx.tolist()) | set(f.inner_val_idx.tolist()) | set(f.inner_train_idx.tolist())
            assert todos == abs_ses, f"fold {f.idx} no cubre exactamente la sesión {ses}"
            test_union.extend(f.test_idx.tolist())
        assert sorted(test_union) == sorted(abs_ses), f"los test de la sesión {ses} no cubren sin solape"


def test_por_sesion_determinista():
    a = sp.make_within_session_folds(_Y2, _SES, n_folds=10, seed=2026)
    b = sp.make_within_session_folds(_Y2, _SES, n_folds=10, seed=2026)
    for x, y in zip(a, b):
        assert x.session == y.session
        assert np.array_equal(x.test_idx, y.test_idx)


# --------------------------------------------------------------------------- #
# 4. Normalización sin fuga (build_within_fold)
# --------------------------------------------------------------------------- #

def test_build_within_fold_normaliza_solo_con_inner_train():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (N_TRIALS, 6, 50)).astype(np.float32)
    folds = sp.make_within_subject_folds(Y, n_folds=5, seed=2026)
    out = sp.build_within_fold(folds[0], X, Y)

    st_directo = sp.fit_normalizer(X[folds[0].inner_train_idx])
    assert np.allclose(out["stats"]["mean"], st_directo["mean"])
    assert np.allclose(out["stats"]["std"], st_directo["std"])

    Ztr = out["X_train"]
    assert np.allclose(Ztr.mean(axis=(0, 2)), 0, atol=1e-5)
    assert np.allclose(Ztr.std(axis=(0, 2)), 1, atol=1e-5)


if __name__ == "__main__":
    fallos = 0
    for nombre, fn in sorted(globals().items()):
        if nombre.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {nombre}")
            except AssertionError as e:
                fallos += 1
                print(f"  FAIL  {nombre}: {e}")
    print("\n" + ("TODOS LOS TESTS ANTI-FUGA PASARON" if not fallos
                  else f"{fallos} TEST(S) FALLARON - protocolo NO válido"))
    sys.exit(1 if fallos else 0)
