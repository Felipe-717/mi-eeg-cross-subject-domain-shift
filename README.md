# Generalization performance of deep learning models under domain shift in multi-user motor imagery BCI systems

The thesis report (in Spanish) is in `informe/`.

This master's thesis studies cross-subject domain shift in motor imagery brain-computer interfaces using OpenBMI and external validation on WBCIC-SHU. It compares EEGNet, Conformer, EEGMamba and fully fine-tuned LaBraM Base under within-subject and leave-one-subject-out protocols.

**Author:** Felipe Ayala Valencia

**Advisor:** Humberto Loaiza Correa

**Group:** PSI, School of Electrical and Electronics Engineering, Universidad del Valle, 2026.

## Abstract

Cross-subject domain shift limits the generalization of motor imagery brain-computer interfaces. This study evaluates OpenBMI, comprising 54 subjects, and uses WBCIC-SHU for external validation. EEGNet serves as the baseline; Conformer and EEGMamba are trained from scratch. LaBraM Base is evaluated with full fine-tuning. The protocols comprise session-wise within-subject evaluation and leave-one-subject-out (LOSO) validation. Comparisons use the Wilcoxon test with Holm correction, TOST with a 0.03 margin, and bootstrap.

## Repository structure

- `src/`: training, data splits, adapters, augmentation, and models.
- `src/datasets/`: data loading and definitions for OpenBMI, WBCIC, and MIND.
- `scripts/`: training, evaluation, statistical analysis, and verification of the numbers reported in the thesis; `avance7_*.py` analyzes the second experiment batch (folder `avance_7/`), including seeds, ablations, alignment, calibration, and transfer.
- `scripts/openbmi/`, `scripts/wbcic/`, `scripts/mind/`: dataset-specific preparation and analysis.
- `pod/`: preparation and runs on RunPod.
- `pod/avance_7/`: planning and execution of the second experiment batch (folder `avance_7/`); `pod/punto6/` and `pod/punto6b/`: transfer and held-out subjects.
- `tests/`: tests for data splits, leakage prevention, and comparison of seen and unseen subjects.
- `results/`: experiment and analysis CSV files.
- `results/avance_7/`: results and analysis for the second experiment batch (folder `avance_7/`); `results/punto6/` and `results/punto6b/`: external validation and held-out subjects.
- `results/pod_*/` and `results/reports/`: earlier results and dataset reports.
- `informe/`: LaTeX source and bibliography.
- `informe/preliminares/`, `informe/capitulos/`, `informe/anexos/`, `informe/figuras/`, and `informe/herramientas/`: cover and preliminary sections, text, appendices, images, and report tools.

## Installation

The environment is named `eeg`. From the repository root:

```bash
conda env create -f environment.yml
conda activate eeg
```

`environment.yml` reflects the development environment (PyTorch 2.5.1 with CUDA 12.1). RunPod training used PyTorch 2.8.0 with CUDA 12.8. EEGMamba also requires `mamba-ssm` and `causal-conv1d`, which are installed only on Linux with a GPU.

## Data

The datasets are not redistributed in this repository. OpenBMI is available from its [original source](http://gigadb.org/dataset/100542); see Lee et al. (2019), *GigaScience*, [doi:10.1093/gigascience/giz002](https://doi.org/10.1093/gigascience/giz002). Download WBCIC-SHU with `python scripts/wbcic/descargar.py`. Public ICA caches are available at [Felipe717/openbmi-ica](https://huggingface.co/datasets/Felipe717/openbmi-ica) and [Felipe717/wbcic-2c-ica](https://huggingface.co/datasets/Felipe717/wbcic-2c-ica). Set the cache path with `--cache` when training.

## Reproduction

From the repository root, with the environment activated and caches prepared. The following training commands illustrate the protocols. Select the architecture with `--model` and the local cache path with `--cache`. Existing runs resume according to their identifier.

| Experiment | Command | Output in `results/` |
| --- | --- | --- |
| Session-wise within-subject | `python scripts/train_baseline.py --dataset openbmi --mode within --model eegnet --channels completo --per-session --cache RUTA_CACHE` | `reports/openbmi/baseline_eegnet_within_*.csv` |
| LOSO | `python scripts/train_baseline.py --dataset openbmi --mode loso --model eegnet --channels completo --cache RUTA_CACHE` | `reports/openbmi/baseline_eegnet_loso_*.csv` |
| Second experiment batch (folder `avance_7/`) plan | `python pod/avance_7/avance_7.py --plan` | Displays the plan only; produces no results. |
| Second experiment batch (folder `avance_7/`) run on the pod | `python pod/avance_7/avance_7.py` | `avance_7/`, including state, logs, and reports. |
| Few-shot calibration analysis | `python scripts/avance7_fewshot.py` | `avance_7/analisis/fewshot_*.csv` |
| Transfer analysis | `python scripts/avance7_transferencia.py` | `avance_7/analisis/transf_*.csv` |
| Seed analysis | `python scripts/avance7_semillas.py` | `avance_7/analisis/semillas_*.csv` and `tres_semillas_*.csv` |

The analyses require the corresponding input CSV files. The scheduler uses pod paths and configuration; see `pod/avance_7/avance_7.py` for the phases, seeds, and architecture-specific parameters.

## Figure verification

From the repository root:

```bash
python scripts/verificar_cifras.py
python scripts/verificar_cifras.py --avance7
```

The expected counts are 195 and 3608 verified figures, respectively.

## RunPod

`pod/` contains scripts for preparing and running jobs. Paths are container paths such as `/root/ws` and `/workspace`; adapt the data and working paths to the environment in use. See `pod/README.md` and the plan in `avance_7/` before running.

## Thesis report

From `informe/`, with a LaTeX distribution and Biber installed:

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

## License and citation

The code is distributed under the MIT License; see [LICENSE](LICENSE). Third-party components are documented in [THIRD_PARTY.md](THIRD_PARTY.md).

```bibtex
@mastersthesis{ayala2026generalizacion,
  author = {Ayala Valencia, Felipe},
  title = {Evaluación del desempeño de generalización de máquinas de aprendizaje profundo frente al cambio de dominio en sistemas BCI multiusuario de imaginación motora},
  school = {Universidad del Valle},
  year = {2026},
  type = {Master's thesis},
  note = {Original title in Spanish}
}
```
