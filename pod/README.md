# `pod/`: RunPod workflows

Run on RunPod: prepare the environment, rebuild the cache from scratch, **verify
that it reproduces Phase 2**, and measure the actual training cost.

The cache is rebuilt there instead of being uploaded because **the pod downloads at
data-center speeds**. Your internet connection only transfers metric CSV files, which are kilobytes.

---

## Cost model: read once

**Charges accrue every second while the pod is RUNNING**, whether it is busy or not. An
idle pod costs the same as one at 100% utilization.

> **Commands are free. An idle pod left running is not.** Run checks as needed; remember
> to shut the pod down.

| action | GPU | disk |
|---|---|---|
| running | charged | $0.10/GB/month |
| **stopped** | no | **$0.20/GB/month, twice as much** |
| terminated | no | no, but **`/workspace` is deleted** |

**Stopping costs more than terminating**, and does not protect data: if the balance reaches $0, pods without a
network volume are terminated and their data cannot be recovered. The rule is **terminate, do not
stop**; therefore, export the cache from the pod first (the `export` phase).

**Recommended GPU: A40 48 GB (~$0.44/h).** It uses Ampere `sm_86`, the same architecture as
the local RTX 3050, so `mamba-ssm` compiles and torch cu121 works without changes.
An H100 for EEGNet (2,898 parameters) is not cost-effective.

---

## Persistent and ephemeral storage

| path | disk | after *stop* | after *terminate* |
|---|---|:--:|:--:|
| `/workspace/**` | volume | ✅ | ❌ **deleted** |
| all other paths (`/`, `~`, `/tmp`, `/raw`) | container | ❌ | ❌ |

```
/workspace/tesis/     code                      persists until terminate
/workspace/venv/      virtual environment       persists until terminate
/workspace/cache/     processed cache           persists until terminate
/workspace/results/   metric CSV files          persists until terminate
/workspace/logs/      process logs              persists until terminate

/raw/mne_data/        raw MOABB data, tens of GB   EPHEMERAL, as intended
~/.cache/             library caches              ephemeral
```

Two implications follow:

- **`pip install` does not survive a stop** (it writes to the system Python outside
  `/workspace`). Therefore, `setup.sh` creates the virtual environment **inside** `/workspace`.
- **MNE and HF default directories are ephemeral.** `env.sh` redirects them.

---

## First session

```bash
# 1. on your laptop: upload the repository (or clone it on the pod with REPO_URL)
scp -P <puerto> -r . root@<ip>:/workspace/tesis

# 2. on the pod
bash pod/setup.sh
source /workspace/env.sh

# 3. run all phases (~2 h, ~$0.90 on an A40)
python pod/run.py --fase gpu download cache verify export bench --hf-repo usuario/openbmi-ica
```

| phase | operation | time |
|---|---|---|
| `gpu` | checks GPU, architecture, and VRAM; **warns if over-provisioned** | sec |
| `download` | MOABB, 54 subjects, resumable | ~20-30 min |
| `cache` | preprocessing + ICA, **parallel by subject** | ~30 min |
| `verify` | **Does it reproduce Phase 2?** | ~5 min |
| `export` | consolidates to fp16 (2.68 GB) and uploads to HF | ~5 min |
| `bench` | measures 1 LOSO fold and 1 within-subject participant | ~30 min |

All phases can be rerun: `download` and `cache` skip completed work; `verify` writes no files.

### Subsequent sessions

```bash
bash pod/setup.sh && source /workspace/env.sh
huggingface-cli download usuario/openbmi-ica --repo-type dataset --local-dir /workspace/cache
```

Downloading 2.68 GB takes a few minutes; training can then begin.

---

## Do not skip the `verify` phase

If the cache is rebuilt on another machine, **verify that it matches** the locally
validated cache; otherwise, the Phase 2 results are not reproducible.

Check three items: integrity (54 subjects, 400×62, balanced, no NaNs), that
`cfg_hash` is **`6391bc7fe8d4`**, and that Control A reproduces known values:

| subject | EOG proxy | motor imagery | rationale |
|---|---|---|---|
| sub-29 | 0.705 | 0.943 | irreducible muscle activity: should **remain high** |
| sub-04 | 0.518 | 0.608 | ocular artifact removed by ICA |
| sub-35 | 0.543 | 0.608 | ocular artifact removed |
| sub-01 | 0.478 | 0.808 | clean: negative control, should remain stable |

Tolerance: ±0.03. **ICA is stochastic**: `random_state` is fixed, but a different version
of `mne` or `mne-icalabel` may converge to different components and produce different
values *even when the hash matches*. Therefore, `requirements.txt` pins exact versions and
`verify` measures the result, not only the configuration.

If verification fails, the script **stops intentionally**. `--seguir-si-falla-verify` is
for debugging, not for producing thesis results.

---

## What to expect from `bench`

This phase turns estimates into measurements. It measures one actual LOSO fold (53 training
subjects) and one within-subject participant, then extrapolates to the relevant combinations.

These two measurements inform **GPU, number of folds, and budget**. Until then,
estimates of duration and cost extrapolate linearly from a six-subject LOSO run and may be
off by a factor of two.

The reported estimates are **sequential**. EEGNet does not saturate a large GPU, so
parallel folds may reduce total time substantially; decide after seeing the measurements.

---

## Not yet included

The production `within` and `loso` phases are **intentionally not included**: they depend
on two decisions pending from the advisor.

- **DSI definition (H2).** The proposal is to measure *seen versus unseen subjects using the
  same data*, which requires a third evaluation set (10% reserved from each
  training subject, stratified by class and session). Implementing it before approval
  would rely on a definition that may change.
- **LOSO scope.** 18 stratified folds versus all 54 folds.

`bench` does not depend on either decision: it measures the cost of one fold under the
current design, and adding another evaluation set does not change the runtime.
