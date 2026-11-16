# Guía del laboratorio

## 1. Preparar lo que se lleva

Clona el repositorio público y entra en su raíz. Lleva también, en una memoria USB, los pesos 44+10 de EEGMamba (`checkpoints_pod/punto6b/eegmamba_epochs_cache_ica/all/`) y `results/punto6b/holdout44/holdout_openbmi_eegmamba.csv`, copiados en las mismas rutas dentro del repositorio: E5 los necesita y los puntos de control no están en GitHub. LaBraM descarga sus pesos preentrenados de Hugging Face la primera vez; si la máquina no tiene internet, lleva una copia de la caché de Hugging Face.

Descarga y comprueba los dos consolidados (los SHA-256 corresponden a los archivos publicados en Hugging Face):

```bash
hf download Felipe717/openbmi-ica --repo-type dataset --local-dir data/processed/openbmi
hf download Felipe717/wbcic-2c-ica --repo-type dataset --local-dir data/processed/wbcic
sha256sum data/processed/openbmi/openbmi_ica_fp16.npz data/processed/wbcic/wbcic_2c_ica_fp16.npz
```

| Archivo | SHA-256 |
|---|---|
| openbmi_ica_fp16.npz | `c3eed8c42dbe65b3ee1f327665162fe852bdaa5d440ec9e573cca57e3e658f4f` |
| wbcic_2c_ica_fp16.npz | `e96f4acbce2e0d18d1e566d2fd404f829d7934dbf0849edfbecc042ed5c1d2a8` |

En Windows usa `Get-FileHash -Algorithm SHA256 <archivo>`. Si la descarga se corta, `curl -L -C -` reanuda (es lo que funcionó en el PC personal). Expande los datos, sin `--forzar`:

```bash
python pod/expandir_cache.py --src data/processed/openbmi/openbmi_ica_fp16.npz --dest data/processed/openbmi/epochs_cache_ica
python scripts/wbcic/expandir.py --src data/processed/wbcic/wbcic_2c_ica_fp16.npz --dest data/processed/wbcic/epochs_cache_ica
```

Para otras ubicaciones, define `OPENBMI_PROC_ROOT` como la carpeta padre de `epochs_cache_ica`, `WBCIC_CACHE` como la carpeta de los 51 archivos y `MNE_DATA` como directorio de MNE. E5 usa la ruta predeterminada de OpenBMI, independientemente de la variable.

## 2. Entorno y verificación

```bash
conda env create -f environment.yml
conda activate eeg
```

El entorno instala PyTorch 2.5.1 con CUDA 12.1, `braindecode[hub]` (necesario para cargar los pesos de LaBraM) y `baycomp`. En Linux con CUDA instala además `pip install mamba-ssm causal-conv1d`; verifica que sean compatibles con la versión de torch y CUDA instalada. `pod/preparar.sh` es la referencia del proyecto para estas dependencias; no fija versiones.

```bash
python scripts/laboratorio.py --comprobar
python scripts/laboratorio.py --plan
python scripts/laboratorio.py --humo
```

`--comprobar` informa Python, torch, CUDA, la GPU y su memoria, si `mamba_ssm` está disponible, las cachés de OpenBMI (54) y WBCIC (51) y qué trabajos pueden correr. El humo de E1 y E4 construye las cachés completas, pero entrena solo un sujeto, semilla 2026 y dos épocas; E2 y E3 usan una partición y dos épocas; E5 se omite porque sus scripts no admiten limitar sujetos. El humo usa variantes y estado separados; revisa los logs antes de lanzar el experimento completo.

## 3. Ejecutar y reanudar

En Linux, con el entorno activado:

```bash
nohup python -u scripts/laboratorio.py > laboratorio.log 2>&1 &
```

También puedes usar `tmux`. En Windows deja una ventana dedicada abierta. Para seleccionar trabajos usa, por ejemplo, `python scripts/laboratorio.py --solo E1 E2`. Vuelve a lanzar el mismo comando después de una interrupción: las cachés y los scripts subyacentes saltan lo que ya tienen. El historial por comando está en `results/informe_final/laboratorio/estado.json`, y hay un log por trabajo en la misma carpeta. Un fallo detiene los comandos restantes de ese trabajo y continúa con los demás.

Si `mamba_ssm` no está disponible (por ejemplo, en Windows), el ejecutor omite automáticamente los comandos de EEGMamba de E3 y E4 y corre los de Conformer y LaBraM; E5 queda pendiente. Lo omitido se informa en pantalla.

## 4. Traer resultados

```bash
python scripts/laboratorio.py --empaquetar
```

Copia `results/informe_final/laboratorio/resultados.zip`. Contiene los CSV de `results/`, incluidos los previos, sin puntos de control, y conserva el árbol de rutas. Copia aparte los logs y el estado si quieres auditar la ejecución. No borres los puntos de control necesarios para reanudar antes de comprobar las salidas.

## Tiempos

| Trabajo | Estimación | Origen y límite |
|---|---:|---|
| E1 | 0.54 h en A40 | 3 × 1.31 / 7.3 h, mediciones intra-sujeto de `pod/avance_7/avance_7.py`; no incluye el filtrado FIR de las cachés. |
| E2 | Unas 20 h en RTX 3050 | 52 s por época medidos en la RTX 3050 con 43 sujetos de entrenamiento, 91 épocas como en la ejecución original, 5 particiones × 3 semillas. En una GPU de escritorio debería ser bastante menor. |
| E3 | Sin medida completa | Prueba corta de Conformer, 1 época: 2.4 min en RTX 3050. LaBraM y EEGMamba son más lentos que EEGNet y Conformer. |
| E4 | 13.29 h en A40 | 10.06 / 2 + 10.99 / 1.33 h, `pod/avance_7/avance_7.py`; no incluye la construcción de la caché. |
| E5 | Sin medida | Solo inferencia; depende de la GPU. |

Las cifras no son garantías para la GPU del laboratorio.

## Preguntas para el laboratorio

- ¿Qué sistema operativo y qué GPU tiene la máquina, y cuánta memoria de GPU?
- ¿Se puede instalar conda y las dependencias de CUDA (`mamba-ssm` necesita Linux)?
- ¿Puede quedar encendida la máquina varios días seguidos?
- ¿Hay acceso remoto, y cómo se conservan los datos entre reinicios?
- ¿Tiene acceso a internet para descargar las cachés y los pesos de LaBraM?
