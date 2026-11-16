"""Compara respuestas teoricas y simetria temporal sobre un ensayo real."""
from pathlib import Path
import sys
import os
os.environ.setdefault('_MNE_FAKE_HOME_DIR', str(Path(__file__).resolve().parents[1] / 'tmp'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
from scipy.signal import butter, sosfreqz, freqz

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / 'src'))
from variantes import filtro_banda


def filtro_banda_fir(x, fs, lo, hi):
    return mne.filter.filter_data(x.astype(np.float64), fs, lo, hi, method='fir',
                                  phase='zero', fir_design='firwin', verbose=False)


def main():
    archivo = RAIZ / 'data/processed/openbmi/epochs_cache_ica/sub-01_full.npz'
    with np.load(archivo, allow_pickle=True) as d:
        x, fs = d['X'][0, 0].astype(float), float(d['sfreq'])
    filas = []
    fig, ejes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for eje, (nombre, lo, hi) in zip(ejes, [('mu', 8, 13), ('beta', 13, 30), ('mubeta', 8, 30)]):
        b = mne.filter.create_filter(None, fs, lo, hi, method='fir', phase='zero', fir_design='firwin', verbose=False)
        f, h = freqz(b, worN=32768, fs=fs)
        _, s = sosfreqz(butter(4, [lo, hi], btype='bandpass', fs=fs, output='sos'), worN=32768, fs=fs)
        for tipo, magnitud, fun in [('FIR', abs(h), filtro_banda_fir), ('IIR', abs(s)**2, filtro_banda)]:
            y = fun(x, fs, lo, hi)
            reverso = fun(x[::-1], fs, lo, hi)[::-1]
            error = np.linalg.norm(y-reverso) / np.linalg.norm(y)
            db = 20*np.log10(np.maximum(magnitud, 1e-12))
            pico = f[(f >= lo) & (f <= hi)][np.argmax(magnitud[(f >= lo) & (f <= hi)])]
            # Transicion definida entre -3 y -40 dB desde el pico hacia cada extremo.
            def cruce(lado, nivel):
                indices = np.where((f < pico) if lado == 'inferior' else (f > pico))[0]
                if lado == 'inferior':
                    indices = indices[::-1]
                encontrados = indices[db[indices] <= nivel]
                return f[encontrados[0]] if len(encontrados) else np.nan
            fila = {'banda': nombre, 'filtro': tipo, 'error_reversion_relativo': error}
            fila.update({f'atenuacion_{hz}_db': -float(np.interp(hz, f, db)) for hz in [8, 13, 30]})
            fila.update({f'transicion_{lado}_hz': abs(cruce(lado, -3)-cruce(lado, -40)) for lado in ['inferior', 'superior']})
            filas.append(fila)
            eje.plot(f, db, label=tipo)
        eje.set(title=nombre, ylabel='Magnitud (dB)', ylim=(-80, 5)); eje.grid(); eje.legend()
    assert len(filas) == 6 and all(np.isfinite(r['error_reversion_relativo']) for r in filas)
    ejes[-1].set(xlabel='Frecuencia (Hz)', xlim=(0, 45))
    out = RAIZ / 'results/informe_final'; (out / 'figuras').mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(out / 'filtros_fir_iir.csv', index=False)
    fig.tight_layout(); fig.savefig(out / 'figuras/filtros_fir_iir.pdf'); plt.close(fig)


if __name__ == '__main__':
    main()
