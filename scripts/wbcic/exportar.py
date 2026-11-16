#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
exportar.py — consolida el cache de WBCIC en un unico fp16 para llevarlo al pod.

Mismo patron que `pod/run.py --fase export` para OpenBMI: un solo fichero viaja mucho
mejor que 51 comprimidos, y fp16 lo deja en la mitad. Lo expande `expandir.py`.

POR QUE fp16 NO ESTROPEA NADA
    Es la misma decision ya tomada y medida para OpenBMI (ver pod/expandir_cache.py): la
    log-varianza por canal, que es la caracteristica que explotan CSP y EEGNet, se desvia
    3.5e-04, y el z-score se recalcula por particion de todos modos. Ademas el
    entrenamiento usa precision mixta en fp16.

DIFERENCIA CON EL CONSOLIDADO DE OpenBMI
    Aquel guarda una ventana amplia (-0.5 a 4.5 s) para poder desplazar la ventana de
    modelado sin reprocesar. Aqui no se puede: el .mat de los autores YA viene epocado en
    [0, 4] s y no existe senal fuera de ese tramo. El consolidado va con tmin=0.0 y 801
    muestras exactas, que es justo la ventana de modelado.

Uso:
    ~/anaconda3/envs/eeg/python.exe scripts/wbcic/exportar.py
    ~/anaconda3/envs/eeg/python.exe scripts/wbcic/exportar.py --subir Felipe717/wbcic-2c-ica
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TARJETA = """---
license: cc-by-4.0
task_categories:
  - other
tags:
  - eeg
  - motor-imagery
  - brain-computer-interface
---

# WBCIC-SHU 2C, cache derivado para experimentos de transferencia

Cache de tensores derivado del subconjunto de dos clases de **WBCIC-SHU**, preprocesado
para ser comparable con [Felipe717/openbmi-ica](https://huggingface.co/datasets/Felipe717/openbmi-ica).

Solo la **sesion 1**. Los autores documentan un efecto de aprendizaje entre dias (0.817
en la sesion 1 frente a 0.885 en la 3), asi que la primera es la condicion mas conservadora.

## Contenido

Un unico `wbcic_2c_ica_fp16.npz`:

| Campo | Forma | Descripcion |
|---|---|---|
| `X` | ({n_seg}, {n_ch}, {n_t}) float16 | Senal en microvoltios |
| `y` | ({n_seg},) | `left_hand` / `right_hand` |
| `subject` | ({n_seg},) | Sujeto de cada segmento |
| `session` | ({n_seg},) | Sesion de cada segmento |
| `ch_names` | ({n_ch},) | Canales, en orden |

{n_sub} sujetos, {sf} Hz, ventana 0 a 4 s, `cfg_hash` `{cfg_hash}`.

## Preprocesamiento

Partiendo de `derivatives/2C dataset_processeddata` del original (0.5-40 Hz, notch 49-51 Hz,
referencia a Pz, 250 Hz):

1. Pz reinsertado como fila de ceros y **referencia promedio comun (CAR)**, que se recupera
   de forma exacta porque `(x_i - x_Pz) - media(x_j - x_Pz) = x_i - media(x_j)`
2. Remuestreo a 200 Hz, 801 muestras
3. **ICA + ICLabel**: infomax extendido, 30 componentes, semilla 2026
4. **45 canales** comunes con el montaje de OpenBMI, en identico orden
5. Rechazo de segmentos con picos por encima de 5000 uV: elimina **1 de 10199**, un caso
   de saturacion del amplificador (78477 uV) que por si solo aportaba el 99.5 % de la suma
   de cuadrados de su canal

`float16` reduce el fichero a la mitad; la log-varianza por canal se desvia 3.5e-04 y el
z-score se reajusta por particion de todos modos.

## Cita

Los datos originales son de Yang et al. Si usas esto, cita el articulo:

> Yang, B., Rong, F., et al. (2025). A multi-day and high-quality EEG dataset for motor
> imagery brain-computer interface. *Scientific Data*, 12, 488.
> https://doi.org/10.1038/s41597-025-04826-y

Original: https://doi.org/10.25452/figshare.plus.22671172 (CC BY 4.0)
"""


LIMITE_FP16 = 65504.0            # mayor finito representable en float16
ORIGEN = Path("data/processed/wbcic/epochs_cache_ica")
DESTINO = Path("data/processed/wbcic/wbcic_2c_ica_fp16.npz")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origen", type=Path, default=ORIGEN)
    ap.add_argument("--destino", type=Path, default=DESTINO)
    ap.add_argument("--subir", default=None,
                    help="repo de HF donde subirlo, p. ej. Felipe717/wbcic-2c-ica. "
                         "Requiere sesion iniciada con `hf auth login`.")
    ap.add_argument("--privado", action="store_true",
                    help="crea el repo privado. Por defecto es publico, como "
                         "Felipe717/openbmi-ica, para que el pod no necesite token.")
    a = ap.parse_args()

    fs = sorted(a.origen.glob("sub-*_full.npz"))
    if not fs:
        raise SystemExit(f"x No hay sujetos en {a.origen}. Ejecuta antes build_cache.py.")

    d0 = np.load(fs[0], allow_pickle=False)
    chs, sf, tmin = d0["ch_names"], float(d0["sfreq"]), float(d0["tmin"])
    n_t = d0["X"].shape[2]

    # Los sujetos NO tienen todos el mismo numero de trials (sub-026 trae 199, no 200),
    # asi que no se puede apilar en un array regular como en OpenBMI. Se concatena a lo
    # largo del eje de trials y se guarda el sujeto de cada uno.
    X, y, ses, sid = [], [], [], []
    for f in fs:
        d = np.load(f, allow_pickle=False)
        if d["X"].shape[1:] != d0["X"].shape[1:]:
            raise SystemExit(f"x {f.name}: forma {d['X'].shape} incompatible con {d0['X'].shape}")
        if float(d["sfreq"]) != sf or float(d["tmin"]) != tmin:
            raise SystemExit(f"x {f.name}: sfreq/tmin distintos del resto")
        if not np.array_equal(d["ch_names"], chs):
            raise SystemExit(f"x {f.name}: los canales no coinciden con {fs[0].name}")
        # fp16 satura en 65504. Un canal que se dispara por encima se convierte en `inf`
        # SIN AVISO de numpy, y el consolidado viaja al pod con infinitos dentro. Paso
        # de verdad: sub-037 tiene un segmento de 78.477 uV en O1 (2026-09-06).
        bruto = d["X"]
        pico = float(np.abs(bruto).max())
        if pico > LIMITE_FP16:
            raise SystemExit(
                f"x {f.name}: pico de {pico:.0f} uV, por encima del limite de fp16 "
                f"({LIMITE_FP16:.0f}).\n"
                f"  Convertirlo escribiria `inf` en el consolidado sin ningun aviso.\n"
                f"  Ese pico no es senal EEG: revisa el sujeto y decide si se descarta el\n"
                f"  segmento (ver --umbral-uv en scripts/wbcic/build_cache.py) antes de\n"
                f"  exportar. Exportar en float32 tampoco lo arregla: un artefacto asi\n"
                f"  distorsiona el z-score global de TODO el dataset.")
        X.append(bruto.astype(np.float16))
        y.append(d["y"]); ses.append(d["session"])
        sid.append(np.array([f.name.replace("_full.npz", "")] * len(d["y"])))

    X = np.concatenate(X)
    if not np.isfinite(X).all():
        raise SystemExit("x El consolidado tiene valores no finitos pese a la comprobacion "
                         "previa. No se escribe nada.")
    y, ses, sid = np.concatenate(y), np.concatenate(ses), np.concatenate(sid)
    print(f"Consolidado: {X.shape} {X.dtype} = {X.nbytes / 1e9:.2f} GB "
          f"| {len(fs)} sujetos | {n_t} muestras | sfreq {sf} | tmin {tmin}")

    reg = a.origen / "_registro.json"
    cfg_hash = json.loads(reg.read_text(encoding="utf-8"))["cfg_hash"] if reg.exists() else ""
    a.destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.destino, X=X, y=y, session=ses, subject=sid, ch_names=chs,
             sfreq=sf, tmin=tmin, cfg_hash=cfg_hash)
    print(f"Escrito {a.destino} ({a.destino.stat().st_size / 1e9:.2f} GB)")

    if a.subir:
        from huggingface_hub import HfApi
        api = HfApi()
        # Publico, igual que Felipe717/openbmi-ica: asi `pod/preparar.sh` lo descarga sin
        # token en cada sesion. La licencia de WBCIC-SHU es CC BY 4.0, que permite
        # redistribuir con atribucion; de ahi la tarjeta con la cita a los autores.
        api.create_repo(a.subir, repo_type="dataset", private=a.privado, exist_ok=True)
        tarjeta = TARJETA.format(n_sub=len(fs), n_seg=len(y), n_ch=len(chs),
                                 n_t=n_t, sf=sf, cfg_hash=cfg_hash)
        api.upload_file(path_or_fileobj=tarjeta.encode("utf-8"), path_in_repo="README.md",
                        repo_id=a.subir, repo_type="dataset")
        api.upload_file(path_or_fileobj=str(a.destino), path_in_repo=a.destino.name,
                        repo_id=a.subir, repo_type="dataset")
        print(f"Subido a https://huggingface.co/datasets/{a.subir}")
    else:
        print("Sin --subir: el fichero se queda aqui.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
