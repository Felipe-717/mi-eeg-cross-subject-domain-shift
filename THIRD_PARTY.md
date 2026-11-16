# Third-party components

## Code and weights

| Component | Use in this repository | File | Source | License |
| --- | --- | --- | --- | --- |
| LaBraM | Adapted channel vocabulary and input logic; architecture provided through braindecode. Base weights (`model.safetensors`) are downloaded from the repository specified in `src/training.py`. | `src/adapters.py`, `src/training.py` | [LaBraM](https://github.com/935963004/LaBraM); [Hugging Face checkpoint](https://huggingface.co/braindecode/labram-pretrained) | MIT in the original code; BSD-3-Clause declared in the checkpoint card |
| EEGMamba | Vendored backbone with mixer blocks adapted from the project code. | `src/eegmamba_vendored.py`, adapter in `src/adapters.py` | [EEGMamba](https://github.com/wjq-learning/EEGMamba) | MIT |
| Mamba | Implementation of state-space blocks and utilities used by vendored EEGMamba. | `src/eegmamba_vendored.py` | [Mamba](https://github.com/state-spaces/mamba) | Apache-2.0 |
| EEGNet and EEG Conformer | Implementations distributed by braindecode are used; no code copied from the original papers was found. | `src/training.py` | [braindecode](https://github.com/braindecode/braindecode) | BSD-3-Clause |
| MNE-Python | API for reading, transforming, and processing EEG; no copied code was found. | `src/`, `scripts/` | [MNE-Python](https://github.com/mne-tools/mne-python) | BSD-3-Clause |

The checkpoint license was checked through the Hugging Face API on 2026-11-16. Weights are downloaded at run time with `huggingface_hub`; they are not included in this publication. LaBraM's channel vocabulary and conventions are attributed in comments in `src/adapters.py`.

## Main dependencies

Versions from the local `eeg` environment used to generate `environment.yml`. Licenses were checked against official metadata and repositories.

| Package | Local version | License |
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

`safetensors`, `mamba-ssm`, `causal-conv1d`, and Triton are not installed in the local environment, although they are imported in paths that load LaBraM or EEGMamba. No local version is assigned to them. EEGMamba requires the pod's Linux- and CUDA-compatible stack. Dependency licenses are those of the upstream projects; consult their notices when redistributing binaries.
