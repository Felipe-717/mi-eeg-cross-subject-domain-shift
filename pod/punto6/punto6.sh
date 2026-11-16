#!/usr/bin/env bash
# Punto 6: 4 modelos de transferencia sobre OpenBMI + 4 nativos sobre WBCIC.
#
# Sin `| tail`: la salida va entera a pantalla para poder seguir el avance.
#
# TRES FALLOS QUE ESTE SCRIPT YA NO COMETE (2026-09-06)
#   1. El `run_id` no llevaba el dataset. Como la ruta del checkpoint es
#      {TESIS_CKPT}/{modelo}_{nombre_cache}/all/{run_id}.pt y los dos caches se llaman
#      `epochs_cache_ica`, el modelo nativo de WBCIC cargaba los pesos de OpenBMI y
#      devolvia en 1,7 s sin entrenar. Corregido en train_baseline.py.
#   2. EEGMamba necesita AMP_BF16=1. Los nucleos de Mamba2 saturan el rango de fp16 y
#      producen NaN pasado el warmup. No estaba puesto.
#   3. Los `export` del `case` persistian entre iteraciones. Ahora se limpian.
set -uo pipefail
cd /workspace/tesis
source /workspace/env.sh
export TESIS_EXIGIR_GPU=1        # aborta en vez de caer al pipeline de CPU en silencio
CO=/workspace/cache/epochs_cache_ica
CW=/workspace/cache_wbcic/epochs_cache_ica
REP=/workspace/tesis/results/reports
T0=$(date +%s)
MIN_SEG=20                       # un entrenamiento real nunca baja de esto

corre() {   # $1=etiqueta  $2=dataset  $3=modelo  $4..=argumentos
    local et="$1" ds="$2" mo="$3"; shift 3
    echo ""
    echo "############################################################"
    echo "###  $et   (t+$(( ($(date +%s)-T0)/60 )) min)"
    echo "############################################################"
    python scripts/train_baseline.py "$@" || { echo "  !! FALLO en $et"; return 1; }

    # GUARDIA: confirma que ENTRENO de verdad y no reanudo un checkpoint ajeno.
    local csv
    csv=$(find "$REP/$ds" -name "baseline_${mo}_all_*" -not -path "*curvas*" 2>/dev/null | head -1)
    if [ -z "$csv" ]; then
        echo "  !! GUARDIA: no aparecio el CSV de $et"; return 1
    fi
    python - "$csv" "$ds" "$MIN_SEG" <<'PY'
import csv, sys
ruta, ds, minimo = sys.argv[1], sys.argv[2], float(sys.argv[3])
filas = list(csv.DictReader(open(ruta, encoding="utf-8")))
if not filas:
    sys.exit("  !! GUARDIA: el CSV esta vacio")
f = filas[-1]
if ds not in f["run_id"]:
    sys.exit(f"  !! GUARDIA: el run_id {f['run_id']!r} no lleva el dataset {ds!r}: "
             f"puede colisionar con el otro conjunto")
if float(f["elapsed_s"]) < minimo:
    sys.exit(f"  !! GUARDIA: solo {f['elapsed_s']} s para {f['n_epochs']} epocas. "
             f"Eso no es un entrenamiento: reanudo un checkpoint que no le tocaba.")
print(f"  guardia ok: {f['run_id']}  {f['elapsed_s']}s  {f['n_epochs']} ep  "
      f"acc={float(f['acc']):.3f}")
PY
}

FALLOS=0
for M in eegnet conformer labram eegmamba; do
    # Cada arquitectura con SU configuracion, la reconstruida por cfg_hash (tab:configs).
    unset LABRAM_FINETUNE LABRAM_UV_SCALE EEGMAMBA_HEAD EEGMAMBA_UV_SCALE AMP_BF16
    case $M in
        eegnet)    EXTRA="--lr 2.8e-3 --batch 256" ;;
        conformer) EXTRA="--lr 1.0e-3 --batch 256" ;;
        labram)    EXTRA="--lr 1.0e-4 --batch 32"
                   export LABRAM_FINETUNE=full LABRAM_UV_SCALE=1 ;;
        eegmamba)  EXTRA="--lr 1.0e-3 --batch 32"
                   export EEGMAMBA_HEAD=flatten EEGMAMBA_UV_SCALE=1 AMP_BF16=1 ;;
    esac

    OPENBMI_PROC_ROOT=/workspace/cache \
    corre "TRANSFERENCIA $M (54 sujetos OpenBMI)" openbmi "$M" \
        --mode all --dataset openbmi --model "$M" --channels comun_wbcic \
        --seeds 2026 --cache "$CO" $EXTRA || FALLOS=$((FALLOS+1))

    corre "NATIVO $M (51 sujetos WBCIC, 20% reservado)" wbcic_2c "$M" \
        --mode all --dataset wbcic_2c --model "$M" --channels comun_wbcic \
        --seeds 2026 --cache "$CW" --frac-test 0.20 $EXTRA || FALLOS=$((FALLOS+1))
done

echo ""
echo "=== TERMINADO en $(( ($(date +%s)-T0)/60 )) min | fallos: $FALLOS ==="
echo -n "checkpoints: "; find /raw/checkpoints -path "*/all/*.pt" | wc -l
find /raw/checkpoints -path "*/all/*.pt" -printf "  %f\n" 2>/dev/null
