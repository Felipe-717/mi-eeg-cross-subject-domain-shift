#!/usr/bin/env python
"""Robustez de los pesos 44+10, perturbando antes de normalizar con origen."""
import argparse
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
os.environ['OPENBMI_PROC_ROOT'] = str(RAIZ / 'data/processed/openbmi')
os.environ['MNE_DATA'] = str(RAIZ / 'data/mne_data')
os.environ['LABRAM_FINETUNE'] = 'full'
# El checkpoint completo reemplaza todos los pesos, sin descargar la inicializacion.
os.environ['LABRAM_ALLOW_SCRATCH'] = '1'
os.environ['EEGMAMBA_HEAD'] = 'flatten'
sys.path.insert(0, str(RAIZ / 'src'))

import numpy as np
import pandas as pd
import torch
import splits as sp
import training as tr
import eval_cruzado as ev

CACHE = RAIZ / 'data/processed/openbmi/epochs_cache_ica'
SALIDA = RAIZ / 'results/informe_final'
ARQ = ('eegnet', 'conformer', 'labram', 'eegmamba')
CONDICIONES = [('ruido_db', x) for x in (20, 10, 5, 0)] + [('red', x) for x in (.1, .5)] + [('canales', x) for x in (.1, .2, .3)] + [('escala', x) for x in (.5, 2)] + [('desplazamiento_s', x) for x in (.1, .25, .5)]


def perturbar(x, tipo, valor, rng, sfreq):
    if tipo == 'ruido_db':
        potencia = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
        return x + rng.normal(size=x.shape) * np.sqrt(potencia / 10 ** (valor / 10))
    if tipo == 'red':
        fase = rng.uniform(0, 2 * np.pi, (*x.shape[:2], 1))
        return x + valor * x.std(axis=-1, keepdims=True) * np.sin(2 * np.pi * 60 * np.arange(x.shape[-1]) / sfreq + fase)
    if tipo == 'canales':
        y = x.copy()
        indices = np.argsort(rng.random(x.shape[:2]), axis=1)[:, :round(valor * x.shape[1])]
        y[np.arange(len(x))[:, None], indices, :] = 0
        return y
    if tipo == 'escala':
        return x * valor
    raise ValueError(tipo)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arq', nargs='+', choices=ARQ, default=ARQ)
    ap.add_argument('--salida', type=Path, default=SALIDA)
    args = ap.parse_args()
    assert np.all(perturbar(np.ones((2, 10, 4)), 'canales', .2, np.random.default_rng(0), 200).sum(axis=(1, 2)) == 32)
    args.salida.mkdir(parents=True, exist_ok=True)
    filas = []
    for arq in args.arq:
        try:
            ckpt, = (RAIZ / 'checkpoints_pod/punto6b' / f'{arq}_epochs_cache_ica/all').glob('*.pt')
            meta = ev._cargar_meta(ckpt)
            subs = sorted(meta['subs_apartados'])
            assert len(subs) == 10 and not set(subs) & set(meta['subs_entrenamiento'])
            publicado = pd.read_csv(RAIZ / f'results/punto6b/holdout44/holdout_openbmi_{arq}.csv')
            publicado = publicado[publicado.norm_destino == 'source'].set_index('sub').acc
            assert set(publicado.index) == set(subs) and publicado.index.is_unique
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            modelo = tr.build_model(meta['modelo'], n_chans=meta['n_chans'], n_times=meta['n_times'], n_classes=len(meta['class_order']), ch_names=meta['ch_names'])
            modelo.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model_state'])
            modelo = modelo.to(device).eval()
            def evaluar(x, y):
                return tr.evaluate(modelo, ev._normalizar(x, 'source', meta).astype(np.float32), y, device, len(meta['class_order']))['acc']
            limpios = {}
            for sub in subs:
                x, etiquetas, chs = sp.load_subject(sub, CACHE, window=tuple(meta['window']))
                x = ev._alinear_canales(x, chs, meta['ch_names'])
                y = tr.encode_labels(etiquetas, meta['class_order'])
                limpios[sub] = evaluar(x, y)
            error = max(abs(limpios[s] - publicado[s]) for s in subs)
            print(f'{arq}: error limpio maximo {error:.6f}', flush=True)
            if error > .005:
                print(f'{arq}: excluido por no reproducir la referencia', flush=True)
                continue
            for sub in subs:
                x, etiquetas, chs = sp.load_subject(sub, CACHE, window=tuple(meta['window']))
                x = ev._alinear_canales(x, chs, meta['ch_names'])
                y = tr.encode_labels(etiquetas, meta['class_order'])
                for tipo, valor in CONDICIONES:
                    acc_fija = None
                    for semilla in range(5):
                        if acc_fija is not None:
                            filas.append(dict(arq=arq, sub=sub, condicion=tipo, valor=valor, sorteo=semilla, acc=acc_fija, limpia=limpios[sub], caida=limpios[sub]-acc_fija))
                            continue
                        if tipo == 'desplazamiento_s':
                            ventana = tuple(t + valor for t in meta['window'])
                            try:
                                xp, _, chs2 = sp.load_subject(sub, CACHE, window=ventana)
                            except AssertionError:
                                continue
                            xp = ev._alinear_canales(xp, chs2, meta['ch_names'])
                        else:
                            xp = perturbar(x, tipo, valor, np.random.default_rng(semilla), meta['sfreq'])
                        acc = evaluar(xp, y)
                        if tipo in ('desplazamiento_s', 'escala'):
                            acc_fija = acc
                        filas.append(dict(arq=arq, sub=sub, condicion=tipo, valor=valor, sorteo=semilla, acc=acc, limpia=limpios[sub], caida=limpios[sub]-acc))
                print(f'{arq}: {sub} terminado', flush=True)
        except (ImportError, ModuleNotFoundError) as exc:
            print(f'{arq}: excluido por dependencia no disponible: {exc}', flush=True)
    if not filas:
        raise SystemExit('Ningun modelo produjo resultados validos')
    df = pd.DataFrame(filas)
    df.to_csv(args.salida / 'robustez.csv', index=False, encoding='utf-8')
    por_sub = df.groupby(['arq', 'condicion', 'valor', 'sub'], as_index=False)[['acc', 'limpia', 'caida']].mean()
    resumen = []
    for clave, g in por_sub.groupby(['arq', 'condicion', 'valor']):
        assert len(g) == 10
        rng = np.random.default_rng(2026)
        indices = rng.integers(0, len(g), (10000, len(g)))
        lo, hi = np.quantile(g.caida.to_numpy()[indices].mean(axis=1), [.025, .975])
        acc_lo, acc_hi = np.quantile(g.acc.to_numpy()[indices].mean(axis=1), [.025, .975])
        resumen.append(dict(zip(['arq', 'condicion', 'valor'], clave), n_sujetos=len(g), acc=g.acc.mean(), limpia=g.limpia.mean(), caida=g.caida.mean(), ic95_inf=lo, ic95_sup=hi, acc_ic95_inf=acc_lo, acc_ic95_sup=acc_hi))
    pd.DataFrame(resumen).to_csv(args.salida / 'robustez_resumen.csv', index=False, encoding='utf-8')
    print('Una semilla de entrenamiento (2026), 10 sujetos; IC de caida por bootstrap de sujetos.', flush=True)


if __name__ == '__main__':
    main()
