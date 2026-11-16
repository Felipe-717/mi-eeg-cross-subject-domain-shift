#!/usr/bin/env bash
# Reintento SOLO de las dos ejecuciones de EEGMamba del punto 6.
# Fallaron con ModuleNotFoundError: no module named 'mamba_ssm'. El entorno del pod se
# preparo sin la rama de eegmamba de pod/preparar.sh, que es la que compila los kernels.
# Las otras seis ya estan hechas y NO se vuelven a lanzar.
set -uo pipefail
cd /workspace/tesis
source /workspace/env.sh
export TESIS_EXIGIR_GPU=1
CO=/workspace/cache/epochs_cache_ica
CW=/workspace/cache_wbcic/epochs_cache_ica
REP=/workspace/tesis/results/reports
T0=$(date +%s)
MIN_SEG=20

echo "=== 1/3  kernels de Mamba2 (compila; varios minutos) ==="
if ! command -v nvcc >/dev/null 2>&1; then
    CUDA_DIR=$(ls -d /usr/local/cuda-* /usr/local/cuda 2>/dev/null | sort -V | tail -1)
    if [ -n "$CUDA_DIR" ] && [ -x "$CUDA_DIR/bin/nvcc" ]; then
        export CUDA_HOME="$CUDA_DIR"; export PATH="$CUDA_DIR/bin:$PATH"
        echo "  nvcc no estaba en PATH; usando $CUDA_DIR"
    fi
fi
python -m pip install -q ninja transformers
# Versiones EXACTAS: los sdist viejos vienen sin csrc/ y no hay que compilar.
# --no-deps: sin el, mamba-ssm arrastra un torch que pisa el 2.8 de la imagen.
python -m pip install --no-build-isolation --no-deps "causal-conv1d==1.7.0" "mamba-ssm==2.3.2.post1" || exit 1
python - <<'PY'
import sys
try:
    import mamba_ssm
    from mamba_ssm.modules.mamba2 import Mamba2            # noqa: F401
    from mamba_ssm.ops.triton.layer_norm import RMSNorm    # noqa: F401
    import torch
except Exception as e:
    sys.exit(f"  x mamba-ssm NO importa: {type(e).__name__}: {e}")
print(f"  ok: mamba-ssm {mamba_ssm.__version__} | torch {torch.__version__}")
if not torch.__version__.startswith("2.8"):
    sys.exit("  x la instalacion piso el torch 2.8 de la imagen. Aborta.")
PY
[ $? -eq 0 ] || exit 1

corre() {
    local et="$1" ds="$2" mo="$3"; shift 3
    echo ""
    echo "############################################################"
    echo "###  $et   (t+$(( ($(date +%s)-T0)/60 )) min)"
    echo "############################################################"
    python scripts/train_baseline.py "$@" || { echo "  !! FALLO en $et"; return 1; }
    local csv
    csv=$(find "$REP/$ds" -name "baseline_${mo}_all_*" -not -path "*curvas*" 2>/dev/null | head -1)
    if [ -z "$csv" ]; then echo "  !! GUARDIA: no aparecio el CSV de $et"; return 1; fi
    python - "$csv" "$ds" "$MIN_SEG" <<'PY'
import csv, sys
ruta, ds, minimo = sys.argv[1], sys.argv[2], float(sys.argv[3])
filas = list(csv.DictReader(open(ruta, encoding="utf-8")))
if not filas: sys.exit("  !! GUARDIA: el CSV esta vacio")
f = filas[-1]
if ds not in f["run_id"]:
    sys.exit(f"  !! GUARDIA: el run_id {f['run_id']!r} no lleva el dataset {ds!r}")
if float(f["elapsed_s"]) < minimo:
    sys.exit(f"  !! GUARDIA: solo {f['elapsed_s']} s para {f['n_epochs']} epocas: "
             f"reanudo un checkpoint que no le tocaba.")
print(f"  guardia ok: {f['run_id']}  {f['elapsed_s']}s  {f['n_epochs']} ep  "
      f"acc={float(f['acc']):.3f}")
PY
}

FALLOS=0
# Misma config que el entrenamiento base (cfg_hash eabca26d5bdf).
# AMP_BF16=1 es obligatorio: los nucleos de Mamba2 saturan fp16 y dan NaN pasado el warmup.
export EEGMAMBA_HEAD=flatten EEGMAMBA_UV_SCALE=1 AMP_BF16=1
EXTRA="--lr 1.0e-3 --batch 32"

echo ""; echo "=== 2/3  transferencia (OpenBMI) ==="
OPENBMI_PROC_ROOT=/workspace/cache \
corre "TRANSFERENCIA eegmamba (54 sujetos OpenBMI)" openbmi eegmamba \
    --mode all --dataset openbmi --model eegmamba --channels comun_wbcic \
    --seeds 2026 --cache "$CO" $EXTRA || FALLOS=$((FALLOS+1))

echo ""; echo "=== 3/3  nativo (WBCIC) ==="
corre "NATIVO eegmamba (51 sujetos WBCIC, 20% reservado)" wbcic_2c eegmamba \
    --mode all --dataset wbcic_2c --model eegmamba --channels comun_wbcic \
    --seeds 2026 --cache "$CW" --frac-test 0.20 $EXTRA || FALLOS=$((FALLOS+1))

echo ""
echo "=== TERMINADO en $(( ($(date +%s)-T0)/60 )) min | fallos: $FALLOS ==="
find /raw/checkpoints -path "*/all/*eegmamba*" -o -path "*eegmamba*/all/*.pt" 2>/dev/null | sed "s/^/  /"
