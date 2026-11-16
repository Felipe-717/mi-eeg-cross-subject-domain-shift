"""Tests anti-fuga del protocolo LOSO.

Si alguno de estos tests falla, los resultados de la tesis NO son válidos.
Ejecutar:  conda run -n eeg python -m pytest tests/ -v
           (o directamente: conda run -n eeg python tests/test_splits_no_leakage.py)
"""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import splits as sp

SUBS = [f"sub-{i:02d}" for i in range(1, 31)]


# --------------------------------------------------------------------------- #
# 1. Disyunción por sujeto
# --------------------------------------------------------------------------- #

def test_hay_un_fold_por_sujeto():
    folds = sp.make_loso_folds(SUBS)
    assert len(folds) == len(SUBS)
    assert sorted(f.test for f in folds) == sorted(SUBS)


def test_test_nunca_aparece_en_entrenamiento_ni_validacion():
    """La regla más importante: el sujeto de test es intocable."""
    for f in sp.make_loso_folds(SUBS):
        assert f.test not in f.inner_train, f"FUGA: {f.test} en inner_train"
        assert f.test not in f.inner_val, f"FUGA: {f.test} en inner_val"


def test_particiones_disjuntas_y_completas():
    for f in sp.make_loso_folds(SUBS):
        f.assert_disjoint()
        # las tres particiones cubren exactamente todos los sujetos, sin repetir
        todos = set(f.inner_train) | set(f.inner_val) | {f.test}
        assert todos == set(SUBS)
        assert len(f.inner_train) + len(f.inner_val) + 1 == len(SUBS)


def test_validacion_interna_rota_entre_sujetos():
    """Ningún sujeto debe quedar siempre como validación (sesgo de selección)."""
    folds = sp.make_loso_folds(SUBS, n_inner_val=2)
    veces = {s: 0 for s in SUBS}
    for f in folds:
        for s in f.inner_val:
            veces[s] += 1
    # con rotación determinista nadie debería acaparar la validación
    assert max(veces.values()) <= 4, f"validación mal repartida: {veces}"


# --------------------------------------------------------------------------- #
# 2. Determinismo
# --------------------------------------------------------------------------- #

def test_misma_semilla_mismos_folds():
    a = sp.make_loso_folds(SUBS, seed=2026)
    b = sp.make_loso_folds(SUBS, seed=2026)
    for x, y in zip(a, b):
        assert x.test == y.test and x.inner_val == y.inner_val


def test_semillas_distintas_cambian_la_validacion():
    a = sp.make_loso_folds(SUBS, seed=1)
    b = sp.make_loso_folds(SUBS, seed=2)
    assert any(x.inner_val != y.inner_val for x, y in zip(a, b))


# --------------------------------------------------------------------------- #
# 3. Normalización sin fuga
# --------------------------------------------------------------------------- #

def test_normalizador_solo_usa_train():
    """Los estadísticos deben depender EXCLUSIVAMENTE de los datos de train."""
    rng = np.random.default_rng(0)
    Xtr = rng.normal(0, 1, (40, 5, 100))
    Xte_a = rng.normal(50, 10, (10, 5, 100))   # test muy distinto
    Xte_b = rng.normal(-90, 3, (10, 5, 100))   # otro test aún más distinto

    s1 = sp.fit_normalizer(Xtr)
    s2 = sp.fit_normalizer(Xtr)                # mismo train -> mismos stats
    assert np.allclose(s1["mean"], s2["mean"])
    assert np.allclose(s1["std"], s2["std"])

    # Aplicar a tests distintos no puede alterar los estadísticos
    _ = sp.apply_normalizer(Xte_a, s1)
    _ = sp.apply_normalizer(Xte_b, s1)
    assert np.allclose(s1["mean"], s2["mean"]), "FUGA: los stats cambiaron con el test"


def test_train_normalizado_queda_centrado():
    rng = np.random.default_rng(1)
    X = rng.normal(7, 3, (50, 4, 200))
    st = sp.fit_normalizer(X)
    Z = sp.apply_normalizer(X, st)
    assert np.allclose(Z.mean(axis=(0, 2)), 0, atol=1e-6)
    assert np.allclose(Z.std(axis=(0, 2)), 1, atol=1e-6)


def test_test_no_queda_centrado_a_la_fuerza():
    """Si el test quedara perfectamente centrado, sería señal de que se usaron sus stats."""
    rng = np.random.default_rng(2)
    Xtr = rng.normal(0, 1, (60, 4, 150))
    Xte = rng.normal(20, 5, (15, 4, 150))
    st = sp.fit_normalizer(Xtr)
    Zte = sp.apply_normalizer(Xte, st)
    assert not np.allclose(Zte.mean(axis=(0, 2)), 0, atol=0.5), \
        "FUGA: el test quedó centrado -> se usaron sus propios estadísticos"


def test_normalizador_soporta_canal_constante():
    X = np.ones((10, 3, 50))          # varianza cero
    st = sp.fit_normalizer(X)
    Z = sp.apply_normalizer(X, st)
    assert np.isfinite(Z).all(), "canal constante produjo NaN/inf"


# --------------------------------------------------------------------------- #
# 3.bis Estrategias alternativas de normalización
# --------------------------------------------------------------------------- #

def test_per_trial_no_mira_otros_trials():
    """La garantía de 'per_trial': cambiar unos trials no altera cómo se normaliza otro."""
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (20, 5, 80))
    Z1 = sp.normalize_per_trial(X)
    X2 = X.copy()
    X2[10:] *= 50.0                      # se alteran otros trials
    Z2 = sp.normalize_per_trial(X2)
    assert np.allclose(Z1[:10], Z2[:10]), "FUGA: un trial se normalizó mirando a otros"


def test_per_trial_preserva_el_patron_espacial():
    """Escala con UN escalar por trial -> la relación de amplitud entre canales se conserva."""
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (8, 4, 60))
    X[:, 0] *= 10.0                       # canal 0 con mucha más amplitud
    Z = sp.normalize_per_trial(X)
    ratio_antes = X[:, 0].std(axis=-1) / X[:, 1].std(axis=-1)
    ratio_despues = Z[:, 0].std(axis=-1) / Z[:, 1].std(axis=-1)
    assert np.allclose(ratio_antes, ratio_despues), \
        "per_trial destruyó la relación entre canales (la señal que usan CSP/EEGNet)"


def test_per_subject_normaliza_cada_sujeto_por_separado():
    rng = np.random.default_rng(6)
    X = np.concatenate([rng.normal(0, 1, (10, 3, 40)), rng.normal(30, 20, (10, 3, 40))])
    sid = np.array(["sub-01"] * 10 + ["sub-02"] * 10)
    Z = sp.normalize_per_subject(X, sid)
    for s in ("sub-01", "sub-02"):
        m = sid == s
        assert np.allclose(Z[m].mean(axis=(0, 2)), 0, atol=1e-6)
        assert np.allclose(Z[m].std(axis=(0, 2)), 1, atol=1e-6)


def test_per_subject_no_mezcla_sujetos():
    """Alterar la escala de un sujeto no puede cambiar el resultado de otro."""
    rng = np.random.default_rng(7)
    X = rng.normal(0, 1, (20, 3, 40))
    sid = np.array(["sub-01"] * 10 + ["sub-02"] * 10)
    Z1 = sp.normalize_per_subject(X, sid)
    X2 = X.copy(); X2[10:] = X2[10:] * 100 + 5
    Z2 = sp.normalize_per_subject(X2, sid)
    assert np.allclose(Z1[:10], Z2[:10]), "FUGA: los sujetos se normalizaron mezclados"


def test_resolve_norm_acepta_bool_historico():
    assert sp._resolve_norm(True) == "global"
    assert sp._resolve_norm(False) is None
    assert sp._resolve_norm("per_trial") == "per_trial"
    try:
        sp._resolve_norm("inventada")
    except ValueError:
        pass
    else:
        raise AssertionError("debería rechazar una estrategia desconocida")


# --------------------------------------------------------------------------- #
# 4. Índice de cambio de dominio
# --------------------------------------------------------------------------- #

def test_dsi_basico():
    # sin degradación -> DSI 0 ; degradación a la mitad -> DSI 0.5
    assert np.allclose(sp.domain_shift_index([0.6], [0.6]), 0.0)
    assert np.allclose(sp.domain_shift_index([0.6], [0.3]), 0.5)


def test_dsi_evita_division_por_cero():
    out = sp.domain_shift_index([0.0], [0.0])
    assert np.isnan(out).all(), "DSI debería ser NaN cuando el within es 0 (kappa=azar)"


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


# --------------------------------------------------------------------------- #
# separar_sujetos: aparta SUJETOS ENTEROS (experimento 44 + 10)
# --------------------------------------------------------------------------- #

def _subs(n=54):
    return [f"sub-{i:02d}" for i in range(1, n + 1)]


def test_separar_sujetos_particion_completa():
    tr, ho = sp.separar_sujetos(_subs(), 10)
    assert len(ho) == 10 and len(tr) == 44
    assert set(tr) | set(ho) == set(_subs())
    assert not (set(tr) & set(ho)), "un sujeto no puede estar en ambos lados"


def test_separar_sujetos_es_determinista():
    assert sp.separar_sujetos(_subs(), 10)[1] == sp.separar_sujetos(_subs(), 10)[1]


def test_separar_sujetos_no_depende_del_orden_de_entrada():
    """El cache devuelve los sujetos en el orden del sistema de ficheros. Si la seleccion
    dependiera de ese orden, no seria reproducible en otra maquina."""
    a = sp.separar_sujetos(_subs(), 10)[1]
    b = sp.separar_sujetos(list(reversed(_subs())), 10)[1]
    assert a == b


def test_separar_sujetos_usa_la_semilla():
    assert sp.separar_sujetos(_subs(), 10)[1] != sp.separar_sujetos(_subs(), 10, seed=7)[1]


def test_separar_sujetos_rechaza_tamanos_invalidos():
    for n in (0, 54, 55, -1):
        try:
            sp.separar_sujetos(_subs(), n)
        except ValueError:
            continue
        raise AssertionError(f"n_holdout={n} deberia haber fallado")
