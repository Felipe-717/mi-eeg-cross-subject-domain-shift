"""Estabilidad de exactitud por sujeto y semilla; 2026 puede tener otro regimen."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import avance7_semillas as semillas


def resumir(matriz, referencia):
    assert matriz.index.equals(referencia.index) and matriz.columns.equals(referencia.columns)
    diferencia = np.sign((matriz - referencia).round(9))
    taus = [kendalltau(matriz[a], matriz[b]).statistic
            for a, b in combinations(matriz.columns, 2)]
    return dict(sd_medias=matriz.mean().std(ddof=1),
                sd_sujeto_media=matriz.std(axis=1, ddof=1).mean(),
                kendall_tau_media=np.mean(taus),
                porcentaje_signo_consistente=100 * diferencia.nunique(axis=1).eq(1).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--salida', type=Path, default=Path('results/informe_final/estabilidad_semillas.csv'))
    args = parser.parse_args()
    prueba = pd.DataFrame({2027: [1., 2., 3.], 2028: [1., 2., 3.]})
    assert resumir(prueba, prueba)['kendall_tau_media'] == 1
    filas = []
    for protocolo in ('within', 'loso'):
        matrices = {a: semillas.mat(protocolo, a) for a in semillas.ARQ}
        for arq, matriz in matrices.items():
            for conjunto in ((2026, 2027, 2028), (2027, 2028)):
                m = matriz[list(conjunto)]
                fila = dict(protocolo=protocolo, arquitectura=arq,
                            semillas='-'.join(map(str, conjunto)), n_sujetos=len(m),
                            regimen=semillas.regimen_str(protocolo, arq, conjunto),
                            **{f'media_{s}': m[s].mean() for s in conjunto},
                            **resumir(m, matrices['eegnet'][list(conjunto)]))
                if arq == 'eegnet':
                    fila['porcentaje_signo_consistente'] = np.nan
                filas.append(fila)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(args.salida, index=False, encoding='utf-8')


if __name__ == '__main__':
    main()
