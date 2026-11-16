#!/usr/bin/env bash
# Experimento 44 + 10: aisla el cambio de PROTOCOLO del cambio de PERSONA.
#
# POR QUE HACE FALTA
#   El modelo del punto 6 se entreno con los 54 sujetos de OpenBMI, asi que no queda
#   ninguna persona sin ver y no puede producir una cifra de "persona nueva, mismo
#   protocolo". Comparar su cifra cruzada contra el promedio LOSO es invalido: el LOSO son
#   54 modelos entrenados con 53 sujetos cada uno, no el mismo modelo evaluado dos veces.
#
# QUE HACE
#   Entrena UN modelo por arquitectura con 44 sujetos y lo evalua, CON LOS MISMOS PESOS,
#   sobre tres conjuntos:
#       1. su propia validacion          -> personas vistas,  protocolo conocido
#       2. los 10 sujetos apartados      -> personas nuevas,  protocolo conocido
#       3. el 20 % reservado de WBCIC    -> personas nuevas,  protocolo nuevo
#   La diferencia 1->2 es el efecto de la persona; la diferencia 2->3, el del protocolo.
#   Como es el mismo modelo en los tres, la resta si es legitima.
#
# NO TOCA NADA DE LO YA PUBLICADO: el `run_id` lleva `_hold10s2026` y el CSV tambien, asi
# que ni los checkpoints ni las filas colisionan con los modelos de 54 sujetos.
set -uo pipefail
cd /workspace/tesis
source /workspace/env.sh
export TESIS_EXIGIR_GPU=1
CO=/workspace/cache/epochs_cache_ica
CW=/workspace/cache_wbcic/epochs_cache_ica
CK=/raw/checkpoints
REP=/workspace/tesis/results/reports
OUT=$REP/holdout44
HOLD_N=10
HOLD_SEED=2026
MIN_SEG=20
mkdir -p "$OUT"
T0=$(date +%s)
FALLOS=0

titulo() {
    echo ""
    echo "############################################################"
    echo "###  $1   (t+$(( ($(date +%s)-T0)/60 )) min)"
    echo "############################################################"
}

for M in eegnet conformer labram eegmamba; do
    # Cada arquitectura con SU configuracion base, la reconstruida por cfg_hash.
    # El `unset` evita que los export de una iteracion se filtren a la siguiente.
    unset LABRAM_FINETUNE LABRAM_UV_SCALE EEGMAMBA_HEAD EEGMAMBA_UV_SCALE AMP_BF16
    case $M in
        eegnet)    EXTRA="--lr 2.8e-3 --batch 256"; TAG="" ;;
        conformer) EXTRA="--lr 1.0e-3 --batch 256"; TAG="" ;;
        labram)    EXTRA="--lr 1.0e-4 --batch 32";  TAG="_full_uv1_lr0.0001"
                   export LABRAM_FINETUNE=full LABRAM_UV_SCALE=1 ;;
        eegmamba)  EXTRA="--lr 1.0e-3 --batch 32";  TAG="_flatten_uv1_lr0.001"
                   # AMP_BF16 es obligatorio: los nucleos de Mamba2 saturan fp16 y dan NaN
                   # pasado el warmup. No entra en el cfg_hash, asi que hay que ponerlo aqui.
                   export EEGMAMBA_HEAD=flatten EEGMAMBA_UV_SCALE=1 AMP_BF16=1 ;;
    esac
    RUN="all_openbmi_seed2026_comun_wbcic_global${TAG}_hold${HOLD_N}s${HOLD_SEED}"
    PT="$CK/${M}_epochs_cache_ica/all/$RUN.pt"

    # --------------------------------------------------------------- entrenamiento
    titulo "ENTRENAMIENTO $M (44 sujetos de OpenBMI, $HOLD_N apartados)"
    OPENBMI_PROC_ROOT=/workspace/cache \
    python scripts/train_baseline.py --mode all --dataset openbmi --model "$M" \
        --channels comun_wbcic --seeds 2026 --cache "$CO" \
        --holdout-n "$HOLD_N" --holdout-seed "$HOLD_SEED" $EXTRA \
        || { echo "  !! FALLO entrenando $M"; FALLOS=$((FALLOS+1)); continue; }

    # GUARDIA: que entreno de verdad, con 44 sujetos, y no reanudo un checkpoint ajeno.
    CSV=$(find "$REP/openbmi" -name "baseline_${M}_all_*hold${HOLD_N}s${HOLD_SEED}*" \
          -not -path "*curvas*" 2>/dev/null | head -1)
    python - "$CSV" "$MIN_SEG" "$HOLD_N" <<'PY' || { FALLOS=$((FALLOS+1)); continue; }
import csv, sys
ruta, minimo, hold = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
if not ruta:
    sys.exit("  !! GUARDIA: no aparecio el CSV")
f = list(csv.DictReader(open(ruta, encoding="utf-8")))[-1]
if f"hold{hold}" not in f["run_id"]:
    sys.exit(f"  !! GUARDIA: el run_id {f['run_id']!r} no lleva el holdout")
if int(f["n_subjects"]) != 54 - hold:
    sys.exit(f"  !! GUARDIA: entreno con {f['n_subjects']} sujetos, se esperaban {54-hold}")
if float(f["elapsed_s"]) < minimo:
    sys.exit(f"  !! GUARDIA: solo {f['elapsed_s']} s: reanudo un checkpoint que no le tocaba")
print(f"  guardia ok: {f['run_id']}  {f['n_subjects']} sujetos  {f['elapsed_s']}s  "
      f"{f['n_epochs']} ep  val acc={float(f['acc']):.3f}")
print(f"  apartados: {f['subs_apartados']}")
PY

    # -------------------------------------------- 2. personas nuevas, mismo protocolo
    # Los sujetos salen del .meta.json del checkpoint, no de una lista escrita a mano:
    # asi es imposible evaluar por error sobre alguien que estuvo en el entrenamiento.
    # eval_cruzado.py aborta ademas si algun sujeto pedido figura como visto.
    APART=$(python -c "import json,sys;print(' '.join(json.load(open(sys.argv[1]))['subs_apartados']))" \
            "${PT%.pt}.meta.json") || { echo "  !! sin meta"; FALLOS=$((FALLOS+1)); continue; }
    titulo "EVAL $M sobre los $HOLD_N apartados de OpenBMI (persona nueva, mismo protocolo)"
    python scripts/eval_cruzado.py --ckpt "$PT" --dataset openbmi --cache "$CO" \
        --subjects $APART --out "$OUT/holdout_openbmi_${M}.csv" \
        || { echo "  !! FALLO en eval OpenBMI de $M"; FALLOS=$((FALLOS+1)); }

    # ------------------------------------------- 3. personas nuevas, protocolo nuevo
    titulo "EVAL $M sobre el 20% reservado de WBCIC (persona nueva, protocolo nuevo)"
    python scripts/eval_cruzado.py --ckpt "$PT" --dataset wbcic_2c --cache "$CW" \
        --solo-reservado 0.20 --out "$OUT/holdout_wbcic_${M}.csv" \
        || { echo "  !! FALLO en eval WBCIC de $M"; FALLOS=$((FALLOS+1)); }
done

echo ""
echo "=== TERMINADO en $(( ($(date +%s)-T0)/60 )) min | fallos: $FALLOS ==="
python /workspace/tesis/pod/punto6b/resumen.py
