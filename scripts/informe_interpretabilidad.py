#!/usr/bin/env python
"""Saliencia de los modelos 44+10 sobre los sujetos apartados."""
import argparse
from pathlib import Path

import informe_robustez as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch


def indice(izquierda, derecha):
    total = derecha + izquierda
    return (derecha - izquierda) / total if total else np.nan


def mapa(valores, info, titulo, destino):
    fig, ax = plt.subplots(figsize=(5, 4))
    imagen, _ = mne.viz.plot_topomap(valores, info, axes=ax, show=False)
    ax.set_title(titulo)
    fig.colorbar(imagen, ax=ax, label='Importancia media')
    fig.savefig(destino, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arq', nargs='+', choices=R.ARQ, default=R.ARQ)
    ap.add_argument('--salida', type=Path, default=R.SALIDA)
    args = ap.parse_args()
    assert indice(1, 3) == .5 and indice(3, 1) == -.5
    figuras = args.salida / 'figuras'
    figuras.mkdir(parents=True, exist_ok=True)
    filas = []
    for arq in args.arq:
        try:
            ckpt, = (R.RAIZ / 'checkpoints_pod/punto6b' / f'{arq}_epochs_cache_ica/all').glob('*.pt')
            meta = R.ev._cargar_meta(ckpt)
            subs = sorted(meta['subs_apartados'])
            assert len(subs) == 10 and not set(subs) & set(meta['subs_entrenamiento'])
            publicado = pd.read_csv(R.RAIZ / f'results/punto6b/holdout44/holdout_openbmi_{arq}.csv')
            publicado = publicado[publicado.norm_destino == 'source'].set_index('sub').acc
            assert publicado.index.is_unique and set(publicado.index) == set(subs)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            modelo = R.tr.build_model(meta['modelo'], n_chans=meta['n_chans'], n_times=meta['n_times'], n_classes=len(meta['class_order']), ch_names=meta['ch_names'])
            modelo.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model_state'])
            modelo = modelo.to(device).eval()
            datos = []
            for sub in subs:
                x, etiquetas, chs = R.sp.load_subject(sub, R.CACHE, window=tuple(meta['window']))
                x = R.ev._normalizar(R.ev._alinear_canales(x, chs, meta['ch_names']), 'source', meta).astype(np.float32)
                y = R.tr.encode_labels(etiquetas, meta['class_order'])
                acc = R.tr.evaluate(modelo, x, y, device, len(meta['class_order']))['acc']
                datos.append((sub, x, y, acc))
            error = max(abs(acc - publicado[sub]) for sub, _, _, acc in datos)
            print(f'{arq}: error limpio maximo {error:.6f}', flush=True)
            if error > .005:
                print(f'{arq}: excluido por no reproducir la referencia', flush=True)
                continue
            info = mne.create_info(meta['ch_names'], meta['sfreq'], 'eeg')
            info.set_montage('standard_1020')
            if arq == 'eegnet':
                pesos = modelo.conv_spatial.weight.detach().cpu().numpy()[:, 0, :, 0]
                fig, axes = plt.subplots(4, 4, figsize=(10, 10))
                for i, (ax, peso) in enumerate(zip(axes.flat, pesos)):
                    mne.viz.plot_topomap(peso, info, axes=ax, show=False)
                    ax.set_title(f'Filtro {i + 1}')
                fig.suptitle('Filtros espaciales de EEGNet')
                fig.savefig(figuras / 'interp_eegnet_filtros.pdf', bbox_inches='tight')
                plt.close(fig)
            suma = np.zeros((len(meta['class_order']), len(meta['ch_names'])))
            conteo = np.zeros(len(meta['class_order']), dtype=int)
            for sub, x, y, _ in datos:
                for inicio in range(0, len(y), 16):
                    xb = torch.tensor(x[inicio:inicio+16], device=device, requires_grad=True)
                    yb = y[inicio:inicio+16]
                    logits = R.tr._forward_logits(modelo, xb)
                    objetivo = logits[torch.arange(len(yb), device=device), torch.as_tensor(yb, device=device)].sum()
                    grad = torch.autograd.grad(objetivo, xb)[0].abs().mean(dim=-1).detach().cpu().numpy()
                    for clase in range(len(conteo)):
                        suma[clase] += grad[yb == clase].sum(axis=0)
                        conteo[clase] += (yb == clase).sum()
                print(f'{arq}: {sub} terminado', flush=True)
            nombres = meta['ch_names']
            for clase, nombre in enumerate(meta['class_order']):
                assert conteo[clase] > 0
                v = suma[clase] / conteo[clase]
                izq = np.mean([v[nombres.index(c)] for c in ('FC3', 'C3', 'CP3')])
                der = np.mean([v[nombres.index(c)] for c in ('FC4', 'C4', 'CP4')])
                etiqueta = {'left_hand': 'izquierda', 'right_hand': 'derecha'}[nombre]
                filas.append(dict(arq=arq, clase=etiqueta, ensayos=conteo[clase], c3=v[nombres.index('C3')], c4=v[nombres.index('C4')], indice_c3_c4=indice(v[nombres.index('C3')], v[nombres.index('C4')]), indice_vecindad=indice(izq, der)))
                mapa(v, info, f'{arq}: mano {etiqueta}', figuras / f'interp_{arq}_{etiqueta}.pdf')
        except (ImportError, ModuleNotFoundError) as exc:
            print(f'{arq}: excluido por dependencia no disponible: {exc}', flush=True)
    if not filas:
        raise SystemExit('Ningun modelo produjo resultados validos')
    pd.DataFrame(filas).to_csv(args.salida / 'interpretabilidad_lateralizacion.csv', index=False, encoding='utf-8')
    print('Saliencia: gradiente absoluto del logit de la clase real respecto a la entrada normalizada; semilla 2026, 10 sujetos.', flush=True)


if __name__ == '__main__':
    main()
