# Componentes de terceros

## Codigo y pesos

| Componente | Uso en este repositorio | Archivo | Origen | Licencia |
| --- | --- | --- | --- | --- |
| LaBraM | Vocabulario de canales y logica de entrada adaptados; arquitectura via braindecode. Se descargan pesos Base (`model.safetensors`) desde el repositorio indicado en `src/training.py`. | `src/adapters.py`, `src/training.py` | [LaBraM](https://github.com/935963004/LaBraM); [checkpoint en Hugging Face](https://huggingface.co/braindecode/labram-pretrained) | MIT en el codigo original; BSD-3-Clause declarada en la tarjeta del checkpoint |
| EEGMamba | Backbone vendorizado, con bloques mixer adaptados del codigo del proyecto. | `src/eegmamba_vendored.py`, adaptador en `src/adapters.py` | [EEGMamba](https://github.com/wjq-learning/EEGMamba) | MIT |
| Mamba | Implementacion de bloques state-space y utilidades usadas por el vendorizado EEGMamba. | `src/eegmamba_vendored.py` | [Mamba](https://github.com/state-spaces/mamba) | Apache-2.0 |
| EEGNet y EEG Conformer | Se usan las implementaciones distribuidas por braindecode; no se encontro codigo copiado de los articulos originales. | `src/training.py` | [braindecode](https://github.com/braindecode/braindecode) | BSD-3-Clause |
| MNE-Python | API de lectura, transformacion y procesamiento EEG; no se encontro codigo copiado. | `src/`, `scripts/` | [MNE-Python](https://github.com/mne-tools/mne-python) | BSD-3-Clause |

La licencia del checkpoint se consulto en la API de Hugging Face el 16-11-2026. Los pesos se descargan en ejecucion con `huggingface_hub`; no estan incluidos en esta publicacion. LaBraM identifica el origen de su vocabulario y convenciones en comentarios de `src/adapters.py`.

## Dependencias principales

Versiones del entorno `eeg` local usado para generar `environment.yml`. Licencias contrastadas con los metadatos/repositorios oficiales.

| Paquete | Version local | Licencia |
| --- | --- | --- |
| PyTorch (`torch`) | 2.5.1+cu121 | BSD-3-Clause |
| MNE-Python (`mne`) | 1.12.1 | BSD-3-Clause |
| braindecode | 1.6.1 | BSD-3-Clause |
| SciPy | 1.17.1 | BSD-3-Clause |
| NumPy | 2.4.6 | BSD-3-Clause |
| pandas | 3.0.3 | BSD-3-Clause |
| autoreject | 0.4.4 | BSD-3-Clause |
| scikit-learn | 1.9.0 | BSD-3-Clause |
| MOABB | 1.5.0 | BSD-3-Clause |
| mne-icalabel | 0.9.0 | BSD-3-Clause |
| huggingface-hub | 1.30.0 | Apache-2.0 |
| einops | 0.8.2 | MIT |
| matplotlib | 3.11.1 | PSF-based license |
| pytest | 9.1.1 | MIT |
| PyYAML | 6.0.3 | MIT |

`safetensors`, `mamba-ssm`, `causal-conv1d` y Triton no estan instalados en el entorno local, aunque se importan en las rutas que cargan LaBraM o EEGMamba. No se les asigna una version local. EEGMamba requiere el stack compatible con Linux y CUDA del pod. Las licencias de dependencias son las de los proyectos aguas arriba; consulte sus avisos al redistribuir binarios.
