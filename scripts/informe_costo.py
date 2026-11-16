#!/usr/bin/env python
"""Costo de modelos y suma de tiempos de entrenamiento registrados."""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

RAIZ = Path(__file__).resolve().parents[1]
os.environ.setdefault('_MNE_FAKE_HOME_DIR', str(RAIZ / 'tmp'))
sys.path.insert(0, str(RAIZ / 'src'))
import training
import avance7_datos as A

INFORME = {'eegnet': 2898, 'conformer': 710146, 'labram': 5818138, 'eegmamba': 3376445}
CANALES = ('Fp1 Fp2 F7 F3 Fz F4 F8 FC5 FC1 FC2 FC6 T7 C3 Cz C4 T8 TP9 CP5 CP1 CP2 CP6 TP10 '
           'P7 P3 Pz P4 P8 PO9 O1 Oz O2 PO10 FC3 FC4 C5 C1 C2 C6 CP3 CPz CP4 P1 P2 POz '
           'FT9 FTT9h TTP7h TP7 TPP9h FT10 FTT10h TPP8h TP8 TPP10h F9 F10 AF7 AF3 AF4 AF8 PO3 PO4').split()
COMUNES = [c for c in CANALES if c not in 'TP9 TP10 PO9 PO10 CPz P1 P2 FT9 FTT9h TTP7h TPP9h FT10 FTT10h TPP8h TPP10h F9 F10'.split()]


def medir(modelo, canales, dispositivo, lote):
    modelo = modelo.to(dispositivo).eval()
    x = torch.randn(lote, canales, 801, device=dispositivo)
    if dispositivo == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(5):
            modelo(x)
        tiempos = []
        for _ in range(20):
            if dispositivo == 'cuda':
                torch.cuda.synchronize()
            inicio = time.perf_counter()
            modelo(x)
            if dispositivo == 'cuda':
                torch.cuda.synchronize()
            tiempos.append((time.perf_counter() - inicio) * 1000)
    pico = torch.cuda.max_memory_allocated() / 1e6 if dispositivo == 'cuda' and lote == 64 else np.nan
    return np.median(tiempos), np.percentile(tiempos, 95), pico


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    os.environ['LABRAM_ALLOW_SCRATCH'] = '1'
    os.environ['LABRAM_FINETUNE'] = 'full'
    torch.set_num_threads(min(4, torch.get_num_threads()))
    filas = []
    for arq in A.ARQ:
        archivos = sorted((RAIZ / 'checkpoints_pod' / 'punto6b' / f'{arq}_epochs_cache_ica' / 'all').glob('*.pt'))
        tamano = archivos[0].stat().st_size / 1e6 if archivos else np.nan
        for canales in (62, 45):
            notas = ['Checkpoint punto6b de 45 canales; MB decimales'] if archivos else ['Checkpoint no disponible']
            fila = dict(arquitectura=arq, canales=canales, muestras=801, checkpoint_mb=tamano,
                        param_informe=INFORME[arq] if canales == 62 else np.nan)
            if arq == 'eegmamba':
                try:
                    estado = torch.load(archivos[0], map_location='cpu', weights_only=False)['model_state']
                    total = sum(v.numel() for v in estado.values())
                    estimado = sum(v.numel() for k, v in estado.items() if not any(t in k for t in ('running_', 'num_batches_tracked')))
                    fila.update(parametros=np.nan, entrenables=np.nan, elementos_state_dict=total,
                                elementos_sin_buffers_conocidos=estimado)
                    notas.append('Checkpoint de 45 canales; no se puede distinguir todos los buffers de parametros ni inferir requires_grad; no extrapolado a 62 canales')
                    if canales == 62:
                        fila.update(elementos_state_dict=np.nan, elementos_sin_buffers_conocidos=np.nan)
                except Exception as e:
                    notas.append(f'No se pudo leer checkpoint: {e}')
                notas.append('FLOP, latencia y memoria no disponibles: mamba_ssm no disponible en Windows')
            else:
                try:
                    nombres = CANALES if canales == 62 else COMUNES
                    modelo = training.build_model(arq, canales, 801, 2, ch_names=nombres).eval()
                    fila['parametros'] = sum(p.numel() for p in modelo.parameters())
                    fila['entrenables'] = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
                    if arq == 'eegnet' and canales == 62:
                        assert fila['parametros'] == INFORME['eegnet']
                    try:
                        with torch.no_grad(), FlopCounterMode(display=False) as contador:
                            modelo(torch.randn(1, canales, 801))
                        fila['flop_ensayo'] = contador.get_total_flops()
                        notas.append('FLOP: solo operaciones compatibles con FlopCounterMode')
                    except Exception as e:
                        notas.append(f'FLOP no disponible: {e}')
                    for dispositivo in ('cpu', 'cuda'):
                        for lote in (1, 64):
                            try:
                                mediana, p95, pico = medir(modelo, canales, dispositivo, lote)
                                fila[f'{dispositivo}_lote{lote}_mediana_ms'] = mediana
                                fila[f'{dispositivo}_lote{lote}_p95_ms'] = p95
                                if dispositivo == 'cuda' and lote == 64:
                                    fila['cuda_lote64_pico_mb'] = pico
                            except Exception as e:
                                notas.append(f'{dispositivo} lote {lote}: {type(e).__name__}: {e}')
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                except AssertionError:
                    raise
                except Exception as e:
                    notas.append(f'Modelo no disponible: {type(e).__name__}: {e}')
            fila['coincide'] = fila.get('parametros') == INFORME[arq] if canales == 62 and pd.notna(fila.get('parametros')) else np.nan
            fila['nota'] = '; '.join(notas)
            filas.append(fila)
    salida = RAIZ / 'results' / 'informe_final'
    salida.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(salida / 'costo_modelos.csv', index=False)
    tiempos = []
    for arq in A.ARQ:
        for protocolo, cargar in [('intra', A.within), ('LOSO', A.loso)]:
            datos = cargar(arq).copy()
            datos['elapsed_s'] = pd.to_numeric(datos['elapsed_s'], errors='coerce')
            for (semilla, regimen), grupo in datos.groupby(['seed', 'regimen']):
                tiempos.append(dict(arquitectura=arq, protocolo=protocolo, semilla=semilla,
                                    gpu={'previo': 'A40', 'avance_7': 'L40S'}[regimen], regimen=regimen,
                                    horas=grupo.elapsed_s.sum(min_count=1) / 3600, ejecuciones=len(grupo),
                                    nota=f'Suma de tiempos por ejecucion, no tiempo de pared; faltantes: {grupo.elapsed_s.isna().sum()}'))
    pd.DataFrame(tiempos).to_csv(salida / 'costo_entrenamiento.csv', index=False)


if __name__ == '__main__':
    main()
