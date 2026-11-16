"""Cinco particiones de sujetos y ejecuciones aisladas del experimento apartado."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get('OPENBMI_PROC_ROOT', RAIZ / 'data/processed/openbmi')) / 'epochs_cache_ica'
WBCIC = Path(os.environ.get('WBCIC_CACHE', RAIZ / 'data/processed/wbcic/epochs_cache_ica'))
SALIDA = RAIZ / 'results/informe_final/apartado5'
os.environ.setdefault('OPENBMI_PROC_ROOT', str(CACHE.parent))
os.environ.setdefault('MNE_DATA', str(RAIZ / 'data/mne_data'))

MODELOS = {
    'eegnet': (['--lr', '2.8e-3', '--batch', '256'], {}),
    'conformer': (['--lr', '1.0e-3', '--batch', '256'], {}),
    'labram': (['--lr', '1.0e-4', '--batch', '32'], {'LABRAM_FINETUNE': 'full', 'LABRAM_UV_SCALE': '1'}),
    'eegmamba': (['--lr', '1.0e-3', '--batch', '32'], {'EEGMAMBA_HEAD': 'flatten', 'EEGMAMBA_UV_SCALE': '1', 'AMP_BF16': '1'}),
}


def particiones():
    sujetos = sorted(p.name.replace('_full.npz', '') for p in CACHE.glob('sub-*_full.npz'))
    if len(sujetos) != 54:
        raise ValueError(f'Se requieren 54 sujetos, hay {len(sujetos)}')
    grupos = np.array_split(np.random.default_rng(2026).permutation(sujetos), 5)
    assert [len(g) for g in grupos] == [11, 11, 11, 11, 10]
    assert len(set(np.concatenate(grupos))) == 54
    return [sorted(g.tolist()) for g in grupos]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modelos', nargs='+', choices=MODELOS, default=list(MODELOS))
    ap.add_argument('--particiones', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--semillas', type=int, nargs='+', default=[2026, 2027, 2028])
    ap.add_argument('--max-epochs', type=int)
    ap.add_argument('--solo-particiones', action='store_true')
    a = ap.parse_args()
    grupos = particiones()
    tabla = SALIDA.parent / 'particiones_apartado.csv'
    tabla.parent.mkdir(parents=True, exist_ok=True)
    with tabla.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(['particion', 'sujeto'])
        w.writerows((i, s) for i, grupo in enumerate(grupos, 1) for s in grupo)
    if a.solo_particiones:
        return
    for modelo in a.modelos:
      for i in a.particiones:
        if i not in range(1, 6):
            ap.error('Las particiones deben estar entre 1 y 5')
        for semilla in a.semillas:
            carpeta = SALIDA / (f'prueba_{a.max_epochs}ep' if a.max_epochs else 'completo') / modelo / f'p{i}_s{semilla}'
            carpeta.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env['TESIS_CKPT'] = str(carpeta / 'checkpoints')
            cmd = [sys.executable, __file__, '--interno', str(i), str(semilla), str(carpeta), modelo]
            if a.max_epochs:
                cmd += ['--max-epochs', str(a.max_epochs)]
            subprocess.run(cmd, cwd=RAIZ, env=env, check=True)


def interno():
    i, semilla, carpeta = int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    modelo = sys.argv[5]
    for clave in ('LABRAM_FINETUNE', 'LABRAM_UV_SCALE', 'EEGMAMBA_HEAD', 'EEGMAMBA_UV_SCALE', 'AMP_BF16'):
        os.environ.pop(clave, None)
    extra_modelo, variables = MODELOS[modelo]
    os.environ.update(variables)
    grupo = particiones()[i - 1]
    sys.path.insert(0, str(RAIZ / 'scripts'))
    import train_baseline as tb
    configurar = tb._configurar
    def aislado(*args):
        configurar(*args)
        tb.REPORTS = carpeta
    tb._configurar = aislado
    def separar(sujetos, n, seed):
        assert set(grupo) <= set(sujetos) and n == len(grupo)
        return [s for s in sujetos if s not in grupo], grupo
    tb.sp.separar_sujetos = separar
    extra = sys.argv[6:]
    sys.argv = ['train_baseline.py', '--mode', 'all', '--dataset', 'openbmi', '--model', modelo,
                '--channels', 'comun_wbcic', '--seeds', str(semilla), '--cache', str(CACHE),
                '--holdout-n', str(len(grupo)), '--holdout-seed', '2026'] + extra_modelo + extra
    tb.main()
    ckpts = list((carpeta / 'checkpoints').rglob('*.pt'))
    if len(ckpts) != 1:
        raise RuntimeError(f'Se esperaba un checkpoint, se encontraron {len(ckpts)}')
    for nombre, ds, cache, args in [('openbmi', 'openbmi', CACHE, ['--subjects', *grupo]),
                                    ('wbcic', 'wbcic_2c', WBCIC, ['--solo-reservado', '0.20'])]:
        out = carpeta / f'{nombre}.csv'
        if out.exists():
            continue
        if not list(cache.glob('sub-*_full.npz')):
            print(f'Cache {nombre} ausente: evaluacion pendiente', flush=True)
            continue
        subprocess.run([sys.executable, str(RAIZ / 'scripts/eval_cruzado.py'), '--ckpt', str(ckpts[0]),
                        '--dataset', ds, '--cache', str(cache), '--out', str(out), *args], cwd=RAIZ, check=True)


if __name__ == '__main__':
    interno() if len(sys.argv) > 1 and sys.argv[1] == '--interno' else main()
