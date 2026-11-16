"""Resume diferencias del mismo modelo y bootstrap agrupado por particion."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1] / 'results/informe_final/apartado5'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--directorio', type=Path, default=RAIZ / 'completo')
    ap.add_argument('--modelo', choices=['eegnet', 'conformer', 'labram', 'eegmamba'], default='eegnet')
    a = ap.parse_args()
    a.directorio = a.directorio / a.modelo
    filas = []
    for carpeta in sorted(a.directorio.glob('p*_s*')):
        logs = list(carpeta.glob(f'baseline_{a.modelo}_all*.csv'))
        if len(logs) != 1 or not (carpeta / 'openbmi.csv').exists() or not (carpeta / 'wbcic.csv').exists():
            continue
        val = pd.read_csv(logs[0]).iloc[-1]
        op = pd.read_csv(carpeta / 'openbmi.csv').query("norm_destino == 'source'")
        wb = pd.read_csv(carpeta / 'wbcic.csv').query("norm_destino == 'source'")
        assert len(op) and len(wb)
        for _, r in op.iterrows():
            filas.append({'particion': carpeta.name.split('_')[0], 'seed': int(val['seed']), 'sujeto': r['sub'],
                          'visto': val['acc'], 'nuevo': r['acc'], 'wbcic': wb['acc'].mean(),
                          'caida_persona': val['acc'] - r['acc'], 'caida_protocolo': r['acc'] - wb['acc'].mean()})
    if not filas:
        raise SystemExit('No hay ejecuciones completas con OpenBMI y WBCIC')
    df = pd.DataFrame(filas)
    a.directorio.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.directorio / 'caidas_persona.csv', index=False)
    rng = np.random.default_rng(2026)
    resumen = []
    # Cada particion comparte pesos y evaluacion WBCIC: se remuestrean grupos completos.
    grupos = list(df.groupby('particion'))
    if len(grupos) < 5:
        raise SystemExit('Se necesitan las cinco particiones para calcular el IC')
    for columna in ['caida_persona', 'caida_protocolo']:
        muestras = []
        for _ in range(10000):
            elegidos = rng.integers(0, len(grupos), len(grupos))
            muestras.append(pd.concat([grupos[j][1] for j in elegidos])[columna].mean())
        li, ls = np.quantile(muestras, [0.025, 0.975])
        resumen.append({'medida': columna, 'media': df[columna].mean(), 'ic95_inferior': li, 'ic95_superior': ls})
    assert len(resumen) == 2
    pd.DataFrame(resumen).to_csv(a.directorio / 'resumen.csv', index=False)


if __name__ == '__main__':
    main()
