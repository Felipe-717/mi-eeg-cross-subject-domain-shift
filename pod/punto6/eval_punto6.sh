#!/usr/bin/env bash
# Punto 6, medida final: los 4 modelos entrenados en OpenBMI aplicados SIN reentrenar al
# 20% reservado de WBCIC. Es transferencia de disparo cero.
#
# El techo nativo NO se calcula aqui: `train_baseline.py --frac-test 0.20` ya evaluo cada
# modelo nativo sobre ESE MISMO 20%, y su cifra esta en la columna `reservado_acc` del CSV.
# Asi las dos ramas se miden sobre los mismos segmentos.
set -uo pipefail
cd /workspace/tesis
source /workspace/env.sh
CK=/raw/checkpoints
CW=/workspace/cache_wbcic/epochs_cache_ica
OUT=/workspace/tesis/results/reports/cruzado
mkdir -p "$OUT"
FALLOS=0

corre() {   # $1=modelo  $2=nombre del checkpoint
    local mo="$1" pt="$CK/${1}_epochs_cache_ica/all/$2"
    echo ""
    echo "############################################################"
    echo "###  CRUZADO $mo : OpenBMI -> WBCIC (20% reservado)"
    echo "############################################################"
    if [ ! -f "$pt" ]; then echo "  !! no existe $pt"; return 1; fi
    python scripts/eval_cruzado.py --ckpt "$pt" --dataset wbcic_2c --cache "$CW" \
        --solo-reservado 0.20 --out "$OUT/cruzado_${mo}.csv" || return 1
}

for M in eegnet conformer labram eegmamba; do
    case $M in
        labram)   PT=all_openbmi_seed2026_comun_wbcic_global_full_uv1_lr0.0001.pt
                  export LABRAM_FINETUNE=full LABRAM_UV_SCALE=1 ;;
        eegmamba) PT=all_openbmi_seed2026_comun_wbcic_global_flatten_uv1_lr0.001.pt
                  export EEGMAMBA_HEAD=flatten EEGMAMBA_UV_SCALE=1 ;;
        *)        PT=all_openbmi_seed2026_comun_wbcic_global.pt ;;
    esac
    corre "$M" "$PT" || FALLOS=$((FALLOS+1))
done

echo ""
echo "=== TERMINADO | fallos: $FALLOS ==="
python /workspace/resumen_punto6.py
