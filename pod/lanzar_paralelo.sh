#!/usr/bin/env bash
# Lanza el entrenamiento repartido en varios procesos, cada uno con su propio CSV.
#
# Motivo: los 54 folds LOSO (y las semillas) son independientes entre si, asi que el
# trabajo es vergonzosamente paralelo. Lo unico que impedia correr varios procesos era
# que todos escribian el MISMO csv; con --log-tag cada trabajador escribe el suyo y al
# final se fusionan con scripts/merge_shards.py.
#
# Uso:
#   pod/lanzar_paralelo.sh <modelo> <modo> "<semillas>" <n_gpus> [trozos_por_semilla] [extra...]
#
# Ejemplos:
#   # 4 semillas de LOSO de EEGMamba, una por GPU (4 GPUs)
#   pod/lanzar_paralelo.sh eegmamba loso "2027 2028 2029 2030" 4 1 --lr 1e-3
#   # 4 semillas partidas en 2 trozos de folds cada una = 8 procesos sobre 4 GPUs
#   pod/lanzar_paralelo.sh eegmamba loso "2027 2028 2029 2030" 4 2 --lr 1e-3
#   # within de EEGMamba con la cabeza flatten, 5 semillas
#   pod/lanzar_paralelo.sh eegmamba within "2026 2027 2028 2029 2030" 4 1 --lr 1e-3
#
# Cada proceso queda en su propia ventana de tmux dentro de la sesion actual, y su log
# en logs/<modelo>_<modo>_<tag>.log
set -euo pipefail

MODEL=${1:?modelo (eegnet|conformer|labram|eegmamba)}
MODE=${2:?modo (loso|within)}
SEEDS=${3:?semillas entre comillas, p. ej. "2027 2028"}
NGPU=${4:?numero de GPUs}
CHUNKS=${5:-1}
shift 5 || shift 4
EXTRA=("$@")

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
mkdir -p logs

# Argumentos comunes a todos los trabajadores. Cambiar aqui, nunca por trabajador, para
# que los shards sean comparables entre si.
COMUN=(--dataset openbmi --model "$MODEL" --channels completo --norm global
       --cache epochs_cache_ica)
if [ "$MODE" = "within" ]; then
  COMUN+=(--mode within --per-session)
else
  COMUN+=(--mode loso)
fi

N_FOLDS=54          # LOSO: un fold por sujeto de OpenBMI
i=0
for SEED in $SEEDS; do
  for ((c=0; c<CHUNKS; c++)); do
    TAG="s${SEED}_c${c}"
    GPU=$(( i % NGPU ))
    ARGS=("${COMUN[@]}" --seeds "$SEED" --log-tag "$TAG" "${EXTRA[@]}")
    if [ "$MODE" = "loso" ] && [ "$CHUNKS" -gt 1 ]; then
      # trozo c: folds c, c+CHUNKS, c+2*CHUNKS, ... (intercalado, no contiguo, para que
      # los trozos duren parecido aunque los sujetos no cuesten todos lo mismo)
      FOLDS=$(seq "$c" "$CHUNKS" $((N_FOLDS - 1)) | tr '\n' ' ')
      ARGS+=(--loso-folds $FOLDS)
    fi
    LOG="logs/${MODEL}_${MODE}_${TAG}.log"
    echo "GPU$GPU  ${TAG}  -> $LOG"
    CUDA_VISIBLE_DEVICES=$GPU nohup python scripts/train_baseline.py "${ARGS[@]}" \
        > "$LOG" 2>&1 &
    i=$((i + 1))
  done
done

echo
echo "$i procesos lanzados. Seguimiento:  tail -f logs/${MODEL}_${MODE}_*.log"
echo "Al terminar, fusionar los shards:"
echo "  python scripts/merge_shards.py results/reports/openbmi/baseline_${MODEL}_${MODE}_....csv"
wait
