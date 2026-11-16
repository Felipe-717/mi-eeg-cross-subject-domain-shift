# Evaluación del desempeño de generalización de máquinas de aprendizaje profundo frente al cambio de dominio en sistemas BCI multiusuario de imaginación motora

This master's thesis studies cross-subject domain shift in motor imagery brain-computer interfaces using OpenBMI and external validation on WBCIC-SHU. It compares EEGNet, Conformer, EEGMamba and fully fine-tuned LaBraM Base under within-subject and leave-one-subject-out protocols.

**Autor:** Felipe Ayala Valencia  
**Director:** Humberto Loaiza Correa  
**Grupo:** PSI, Escuela de Ingeniería Eléctrica y Electrónica, Universidad del Valle, 2026.

## Resumen

El cambio de dominio entre sujetos limita la generalización de las interfaces cerebro-computador basadas en imaginación motora.  
Se estudia OpenBMI, con 54 sujetos, y se emplea WBCIC-SHU para validación externa.  
EEGNet sirve como referencia; Conformer y EEGMamba se entrenan desde cero.  
LaBraM Base se evalúa mediante ajuste fino completo.  
Los protocolos comprenden evaluación intra-sujeto por sesión y validación dejando un sujeto fuera (LOSO).  
Las comparaciones utilizan Wilcoxon con corrección de Holm, TOST con margen 0.03 y bootstrap.

## Estructura

- `src/`: entrenamiento, particiones, adaptadores, aumentación y modelos.
- `src/datasets/`: carga y definición de OpenBMI, WBCIC y MIND.
- `scripts/`: entrenamiento, evaluación, análisis estadístico y verificación de cifras; `avance7_*.py` analiza la segunda tanda de experimentos (semillas, ablaciones, alineamiento, calibración y transferencia).
- `scripts/openbmi/`, `scripts/wbcic/`, `scripts/mind/`: preparación y análisis específicos de cada conjunto.
- `pod/`: preparación y ejecución en RunPod.
- `pod/avance_7/`: planificación y ejecución de la segunda tanda de experimentos; `pod/punto6/` y `pod/punto6b/`: transferencia y sujetos reservados.
- `tests/`: pruebas de particiones, ausencia de fuga y comparación de sujetos vistos y no vistos.
- `results/`: CSV de experimentos y análisis.
- `results/avance_7/`: resultados y análisis de la segunda tanda de experimentos; `results/punto6/` y `results/punto6b/`: validación externa y sujetos reservados.
- `results/pod_*/` y `results/reports/`: resultados anteriores y reportes por conjunto.
- `informe/`: fuente LaTeX y bibliografía.
- `informe/preliminares/`, `informe/capitulos/`, `informe/anexos/`, `informe/figuras/` e `informe/herramientas/`: portada y secciones preliminares, texto, anexos, imágenes y herramientas del informe.

## Instalación

El entorno se llama `eeg`. Desde la raíz del repositorio:

```bash
conda env create -f environment.yml
conda activate eeg
```

`environment.yml` refleja el entorno de desarrollo (PyTorch 2.5.1 con CUDA 12.1). Los entrenamientos en RunPod usaron PyTorch 2.8.0 con CUDA 12.8; EEGMamba requiere además `mamba-ssm` y `causal-conv1d`, que solo se instalan en Linux con GPU.

## Datos

Los datos no se redistribuyen en este repositorio. OpenBMI se obtiene de su [fuente original](http://gigadb.org/dataset/100542); véase Lee et al. (2019), *GigaScience*, [doi:10.1093/gigascience/giz002](https://doi.org/10.1093/gigascience/giz002). WBCIC-SHU se descarga con `python scripts/wbcic/descargar.py`. Las cachés públicas con ICA están en [Felipe717/openbmi-ica](https://huggingface.co/datasets/Felipe717/openbmi-ica) y [Felipe717/wbcic-2c-ica](https://huggingface.co/datasets/Felipe717/wbcic-2c-ica). Configure la ruta de la caché con `--cache` al entrenar.

## Reproducción

Desde `publicacion/`, con el entorno activado y las cachés preparadas. Los entrenamientos siguientes son ejemplos de los protocolos; seleccione la arquitectura con `--model` y la ruta local de la caché con `--cache`. Las ejecuciones existentes se reanudan según su identificador.

| Experimento | Comando | Salida en `results/` |
| --- | --- | --- |
| Intra-sujeto por sesión | `python scripts/train_baseline.py --dataset openbmi --mode within --model eegnet --channels completo --per-session --cache RUTA_CACHE` | `reports/openbmi/baseline_eegnet_within_*.csv` |
| LOSO | `python scripts/train_baseline.py --dataset openbmi --mode loso --model eegnet --channels completo --cache RUTA_CACHE` | `reports/openbmi/baseline_eegnet_loso_*.csv` |
| Plan de la segunda tanda | `python pod/avance_7/avance_7.py --plan` | Solo muestra el plan; no genera resultados. |
| Ejecución de la segunda tanda en el pod | `python pod/avance_7/avance_7.py` | `avance_7/`, incluidos estado, registros y reportes. |
| Análisis de calibración few-shot | `python scripts/avance7_fewshot.py` | `avance_7/analisis/fewshot_*.csv` |
| Análisis de transferencia | `python scripts/avance7_transferencia.py` | `avance_7/analisis/transf_*.csv` |
| Análisis de semillas | `python scripts/avance7_semillas.py` | `avance_7/analisis/semillas_*.csv` y `tres_semillas_*.csv` |

Los análisis requieren los CSV de entrada correspondientes. El planificador utiliza rutas y configuración del pod; véase `pod/avance_7/avance_7.py` para las fases, semillas y parámetros específicos de cada arquitectura.

## Verificación de cifras

Desde `publicacion/`:

```bash
python scripts/verificar_cifras.py
python scripts/verificar_cifras.py --avance7
```

Se esperan, respectivamente, 195 y 3608 cifras OK.

## RunPod

`pod/` contiene scripts para preparar y ejecutar los trabajos. Las rutas son las del contenedor, como `/root/ws` y `/workspace`; adapte las rutas de datos y de trabajo al entorno utilizado. Consulte `pod/README.md` y el plan del avance 7 antes de ejecutar.

## Informe

Desde `informe/`, con una distribución LaTeX y Biber instalados:

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

## Licencia y cita

El código se distribuye bajo la licencia MIT; véase [LICENSE](LICENSE). Los componentes de terceros se documentan en [THIRD_PARTY.md](THIRD_PARTY.md).

```bibtex
@mastersthesis{ayala2026generalizacion,
  author = {Ayala Valencia, Felipe},
  title = {Evaluación del desempeño de generalización de máquinas de aprendizaje profundo frente al cambio de dominio en sistemas BCI multiusuario de imaginación motora},
  school = {Universidad del Valle},
  year = {2026},
  type = {Tesis de maestría}
}
```
