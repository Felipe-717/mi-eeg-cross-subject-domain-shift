import sys
from pathlib import Path

import numpy as np
from mne.filter import filter_data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from variantes import derivar_cache


def test_cache_fir(tmp_path):
    origen, destino = tmp_path / "origen", tmp_path / "destino"
    origen.mkdir()
    x = np.random.default_rng(2026).normal(size=(1, 2, 1000)).astype(np.float32)
    np.savez(origen / "sub-01_full.npz", X=x, ch_names=["C3", "C4"], sfreq=200., tmin=0.)
    derivar_cache(origen, destino, "banda_fir", banda=(8, 13))
    with np.load(destino / "sub-01_full.npz") as datos:
        esperado = filter_data(x.astype(np.float64), 200., 8, 13, method="fir",
                               phase="zero", fir_design="firwin", verbose=False)
        np.testing.assert_allclose(datos["X"], esperado.astype(np.float32))
    assert not derivar_cache(origen, destino, "banda_fir", banda=(8, 13))
