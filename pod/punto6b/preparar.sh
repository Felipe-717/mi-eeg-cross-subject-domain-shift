#!/usr/bin/env bash
# Deja un pod NUEVO listo para el experimento 44 + 10: entorno, kernels de Mamba2 y los
# DOS caches (OpenBMI y WBCIC), ambos desde repos publicos de Hugging Face, sin token.
#
# Idempotente: si algo ya esta, no lo repite.
#
# Uso, desde /workspace, con el repo ya en /workspace/tesis:
#     bash /workspace/tesis/pod/punto6b/preparar.sh
set -euo pipefail
WS="${WS:-/root/ws}"       # disco del CONTENEDOR (el volumen de red se colgo el 2026-11-01)
CO="$WS/cache/epochs_cache_ica"
CW="$WS/cache_wbcic/epochs_cache_ica"

_hf() {   # $1=repo  $2=destino
    if command -v hf >/dev/null 2>&1; then
        hf download "$1" --repo-type dataset --local-dir "$2"
    else
        huggingface-cli download "$1" --repo-type dataset --local-dir "$2"
    fi
}

echo "=== 1/4  entorno base, cache de OpenBMI y kernels de Mamba2 ==="
# preparar.sh con el modelo eegmamba instala ademas causal-conv1d + mamba-ssm. Sin ese
# argumento, las dos ejecuciones de EEGMamba fallan al construir el modelo con
# ModuleNotFoundError, que es exactamente lo que paso el 2026-09-06.
bash "$WS/tesis/pod/preparar.sh" eegmamba
# shellcheck disable=SC1091
source "$WS/env.sh"

echo ""
echo "=== 2/4  cache de WBCIC ==="
if ! ls "$CW"/sub-*_full.npz >/dev/null 2>&1; then
    echo "  ausente: descargando Felipe717/wbcic-2c-ica (dataset PUBLICO, sin login)..."
    _hf Felipe717/wbcic-2c-ica "$WS/cache_wbcic"
    echo "  expandiendo el consolidado fp16 a un .npz por sujeto..."
    python "$WS/tesis/scripts/wbcic/expandir.py" \
        --src "$WS/cache_wbcic/wbcic_2c_ica_fp16.npz" --dest "$CW"
fi
n=$(ls "$CW"/sub-*_full.npz 2>/dev/null | wc -l)
[ "$n" -eq 51 ] || { echo "  x WBCIC tiene $n sujetos, se esperaban 51"; exit 1; }
echo "  ok: 51 sujetos en $CW"

echo ""
echo "=== 3/4  comprobacion de los dos caches ==="
WS="$WS" python - <<'PY'
import os, sys
WS = os.environ["WS"]
sys.path.insert(0, f"{WS}/tesis/src")
from pathlib import Path
import splits as sp
for nombre, ruta, esperados in (("OpenBMI", f"{WS}/cache/epochs_cache_ica", 54),
                                ("WBCIC", f"{WS}/cache_wbcic/epochs_cache_ica", 51)):
    subs = sorted(p.name.replace("_full.npz", "") for p in Path(ruta).glob("sub-*_full.npz"))
    if len(subs) != esperados:
        sys.exit(f"  x {nombre}: {len(subs)} sujetos, se esperaban {esperados}")
    print(f"  ok {nombre}: {len(subs)} sujetos ({subs[0]} .. {subs[-1]})")
# Los 10 apartados se fijan AQUI y se imprimen, para que queden en el log antes de
# entrenar nada. Se recalculan igual en el entrenamiento y en la evaluacion.
subs = sorted(p.name.replace("_full.npz", "")
              for p in Path(f"{WS}/cache/epochs_cache_ica").glob("sub-*_full.npz"))
tr, ho = sp.separar_sujetos(subs, 10)
print(f"  apartados (semilla {sp.SEMILLA_HOLDOUT}): {' '.join(ho)}")
print(f"  entrenamiento: {len(tr)} sujetos")
PY

echo ""
echo "=== 4/4  GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""
echo "LISTO. Lanza:  bash $WS/tesis/pod/punto6b/correr.sh"
