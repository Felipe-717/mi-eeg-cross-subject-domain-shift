"""Orquesta los experimentos pendientes del laboratorio."""
import argparse
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import zipfile

R = Path(__file__).resolve().parents[1]
OUT = R / 'results/informe_final/laboratorio'
CO = Path(os.environ.get('OPENBMI_PROC_ROOT', R / 'data/processed/openbmi')) / 'epochs_cache_ica'
CW = Path(os.environ.get('WBCIC_CACHE', R / 'data/processed/wbcic/epochs_cache_ica'))
PY = sys.executable
os.environ.setdefault('OPENBMI_PROC_ROOT', str(CO.parent))
os.environ.setdefault('MNE_DATA', str(R / 'data/mne_data'))
os.environ.setdefault('WBCIC_CACHE', str(CW))
os.environ.setdefault('_MNE_FAKE_HOME_DIR', str(R / 'tmp'))

# Tiempos A40 por entrenamiento intra-sujeto: pod/avance_7/avance_7.py,
# derivados de mediciones en results; no incluyen construccion de caches.
HORAS = {'E1': 3 * 1.31 / 7.3, 'E2': None, 'E3': None,
         'E4': 10.06 / 2 + 10.99 / 1.33, 'E5': None}
CFG = {'eegnet': (['--lr', '2.8e-3', '--batch', '256'], {}),
       'labram': (['--lr', '1.0e-4', '--batch', '32'], {'LABRAM_FINETUNE': 'full', 'LABRAM_UV_SCALE': '1'}),
       'eegmamba': (['--lr', '1.0e-3', '--batch', '32'], {'EEGMAMBA_HEAD': 'flatten', 'EEGMAMBA_UV_SCALE': '1', 'AMP_BF16': '1'})}

def trabajos(humo=False):
    t = {x: [] for x in HORAS}
    def agregar(j, cmd, env=None):
        t[j].append(([str(x) for x in cmd], env or {}))
    def cache(j, nombre, tipo, lo, hi):
        destino = Path(str(CO) + '_' + nombre)
        if len(list(destino.glob('sub-*_full.npz'))) != 54:
            agregar(j, [PY, 'scripts/construir_variantes.py', '--origen', CO, '--destino', destino, '--tipo', tipo, '--banda', lo, hi])
        return destino
    def entrenar(j, modelo, destino, variante):
        extra, env = CFG[modelo]
        cmd = [PY, 'scripts/train_baseline.py', '--mode', 'within', '--dataset', 'openbmi', '--model', modelo, '--channels', 'completo', '--per-session', '--workers', '8', '--seeds', '2026', '--cache', destino] + extra
        if humo:
            cmd += ['--subjects', 'sub-01', '--max-epochs', '2', '--variante', 'smoke' + variante]
        agregar(j, cmd, env)
    for nombre, lo, hi in [('mu', 8, 13), ('beta', 13, 30), ('mubeta', 8, 30)]:
        destino = cache('E1', nombre + '_fir', 'banda_fir', lo, hi)
        entrenar('E1', 'eegnet', destino, nombre)
    # Un comando por arquitectura: sin mamba_ssm solo se omite EEGMamba, no el trabajo entero.
    for j, modelo in [('E2', 'eegnet'), ('E3', 'conformer'), ('E3', 'labram'), ('E3', 'eegmamba')]:
        cmd = [PY, 'scripts/informe_apartado_5particiones.py', '--modelos', modelo]
        if humo:
            cmd += ['--particiones', '1', '--semillas', '2026', '--max-epochs', '2']
        agregar(j, cmd)
    destino = cache('E4', 'mubeta', 'banda', 8, 30)
    for m in ('labram', 'eegmamba'):
        entrenar('E4', m, destino, 'mubeta')
    if not humo:
        for s in ('informe_robustez.py', 'informe_interpretabilidad.py'):
            agregar('E5', [PY, 'scripts/' + s, '--arq', 'eegmamba', '--salida', OUT / 'eegmamba'])
    return t

DESC = {'E1': 'FIR y EEGNet intra-sujeto', 'E2': 'Cinco particiones EEGNet', 'E3': 'Cinco particiones Conformer, LaBraM y EEGMamba', 'E4': 'mu+beta IIR LaBraM y EEGMamba', 'E5': 'Robustez e interpretabilidad EEGMamba'}

def comprobar():
    print('Python:', sys.version.split()[0])
    gpu = False
    try:
        import torch
        gpu = torch.cuda.is_available()
        print('torch:', torch.__version__, '| CUDA:', torch.version.cuda, '| disponible:', gpu)
        if gpu:
            p = torch.cuda.get_device_properties(0)
            print('GPU:', p.name, '| memoria GB:', round(p.total_memory / 1e9, 2))
    except Exception as e:
        print('torch no disponible:', e)
    try:
        import mamba_ssm
        mamba = True
    except Exception as e:
        mamba = False
        print('mamba_ssm no disponible:', e)
    print('mamba_ssm:', mamba)
    n = len(list(CO.glob('sub-*_full.npz'))); w = len(list(CW.glob('sub-*_full.npz')))
    print('OpenBMI:', n, '/ 54', CO); print('WBCIC:', w, '/ 51', CW)
    for j in DESC:
        razones = []
        if not gpu: razones.append('CUDA')
        if n != 54: razones.append('OpenBMI')
        if j in ('E2', 'E3') and w != 51: razones.append('WBCIC')
        if j == 'E5' and not mamba: razones.append('mamba_ssm')
        if j in ('E3', 'E4') and not mamba: razones.append('parcial: sin mamba_ssm se omiten los comandos de EEGMamba')
        if j == 'E5':
            if CO != R / 'data/processed/openbmi/epochs_cache_ica': razones.append('E5 exige cache OpenBMI en ruta predeterminada')
            if not list((R / 'checkpoints_pod/punto6b/eegmamba_epochs_cache_ica/all').glob('*.pt')): razones.append('checkpoint EEGMamba 44+10')
            if not (R / 'results/punto6b/holdout44/holdout_openbmi_eegmamba.csv').exists(): razones.append('CSV 44+10 EEGMamba')
        print(j + ':', 'pendiente: ' + ', '.join(razones) if razones else 'puede correr')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', action='store_true'); ap.add_argument('--comprobar', action='store_true')
    ap.add_argument('--solo', nargs='+', choices=list(DESC)); ap.add_argument('--humo', action='store_true')
    ap.add_argument('--empaquetar', action='store_true')
    a = ap.parse_args(); seleccion = a.solo or list(DESC); t = trabajos(a.humo)
    assert set(t) == set(DESC) and len(t['E2']) == 1
    if a.comprobar: comprobar()
    if a.plan:
        for j in seleccion:
            print(j, DESC[j], '| horas A40:', round(HORAS[j], 2) if HORAS[j] is not None else 'sin medida', '| requisitos:', 'OpenBMI' + (', WBCIC' if j in ('E2','E3') else '') + (', mamba_ssm' if j in ('E3','E4','E5') else ''))
            for cmd, env in t[j]: print(' ', env, shlex.join(cmd))
            if a.humo and j == 'E5': print('  omitido: no admite limitar sujetos ni epocas')
    if a.empaquetar:
        OUT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(OUT / 'resultados.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for p in (R / 'results').rglob('*.csv'):
                if 'checkpoints' not in p.parts: z.write(p, p.relative_to(R))
        print('Paquete:', OUT / 'resultados.zip')
    if a.plan or a.comprobar or a.empaquetar: return 0
    OUT.mkdir(parents=True, exist_ok=True); (R / 'tmp').mkdir(exist_ok=True)
    estado = OUT / ('estado_humo.json' if a.humo else 'estado.json')
    historial = json.loads(estado.read_text(encoding='utf-8')) if estado.exists() else []
    fallos = False
    try:
        import mamba_ssm  # noqa: F401
        mamba = True
    except Exception:
        mamba = False
    for j in seleccion:
        if not t[j]: print(j, 'omitido: no admite humo'); continue
        for cmd, variables in t[j]:
            if 'eegmamba' in cmd and not mamba:
                print(j, 'omitido (falta mamba_ssm):', shlex.join(cmd), flush=True)
                continue
            env = os.environ.copy()
            for k in ('LABRAM_FINETUNE', 'LABRAM_UV_SCALE', 'EEGMAMBA_HEAD', 'EEGMAMBA_UV_SCALE', 'AMP_BF16'): env.pop(k, None)
            env.update(variables)
            registro = {'trabajo': j, 'comando': cmd, 'inicio': dt.datetime.now().isoformat(), 'fin': None, 'codigo_salida': None}
            historial.append(registro); estado.write_text(json.dumps(historial, indent=2, ensure_ascii=False), encoding='utf-8')
            print(j, shlex.join(cmd), flush=True)
            with (OUT / (j + ('_humo' if a.humo else '') + '.log')).open('a', encoding='utf-8') as log:
                log.write('\n' + shlex.join(cmd) + '\n'); log.flush()
                rc = subprocess.call(cmd, cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT)
            registro.update(fin=dt.datetime.now().isoformat(), codigo_salida=rc)
            estado.write_text(json.dumps(historial, indent=2, ensure_ascii=False), encoding='utf-8')
            if rc: print(j, 'fallo:', rc); fallos = True; break
    return int(fallos)

if __name__ == '__main__':
    raise SystemExit(main())
