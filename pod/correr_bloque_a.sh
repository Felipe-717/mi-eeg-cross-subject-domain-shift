#!/usr/bin/env bash
# ============================================================================
# correr_bloque_a.sh — EEGNet + Conformer, LOSO, semillas 2027 y 2028, A LA VEZ.
#
# Por que a la vez y no uno detras de otro: los dos son cuello de LANZAMIENTO de
# kernels, no de computo (un paso con lote 32 tarda ~11,8 ms y de eso casi nada es
# aritmetica). Con un solo proceso la GPU pasa la mayor parte del tiempo esperando
# ordenes. Repartiendo folds entre varios procesos se llena ese hueco: medido en la
# A100, 16 workers dieron 8x de rendimiento neto (cada corrida tarda el doble, pero
# corren 16). Mezclar los dos modelos en la misma GPU no los frena entre si por la
# misma razon.
#
# Cuanto trabajo es: 2 semillas x 54 folds x 2 modelos = 216 corridas.
#   EEGNet   ~368 s/fold  ->  11,0 GPU-h secuenciales
#   Conformer ~393 s/fold ->  11,8 GPU-h secuenciales
#   Total 22,8 GPU-h secuenciales; con ~8 workers deberian ser 5 a 8 h de reloj.
#
# Uso:
#   bash pod/correr_bloque_a.sh              # reparte los workers segun VRAM y vCPU
#   bash pod/correr_bloque_a.sh 5 4          # 5 workers a EEGNet, 4 a Conformer
#   bash pod/correr_bloque_a.sh 5 4 solo-monitor   # no lanza nada, solo mira lo que ya corre
#
# NOTA sobre las semillas: van EXPLICITAS y no por reanudacion porque `pod/subir.sh`
# excluye `results/` a proposito. El pod arranca sin CSV, asi que la reanudacion por
# run_id no puede saber que la 2026 ya se ejecuto. Lo ya hecho se combina en local.
# ============================================================================
set -euo pipefail

WS=/workspace
REPO="$WS/tesis"
CACHE=/workspace/cache/epochs_cache_ica     # ABSOLUTA: --cache se resuelve contra el cwd
SEEDS="2027 2028"
# 6 GB por worker, no 5: `train_run` EXIGE (X_train+X_val)*1.5 libres, que en LOSO son
# ~5,7 GB, y encima cada proceso gasta ~0,6 GB de contexto CUDA. Con 5 el chequeo fallaba y
# el worker se degradaba al pipeline de CPU (~9x mas lento) sin morir.
GB_POR_WORKER=6
# Que un worker que no quepa MUERA en vez de arrastrarse en CPU. Mejor verlo caer en el
# monitor a los 2 minutos que descubrir a las 6 horas que la mitad iba por CPU.
export TESIS_EXIGIR_GPU=1

MODO="${3:-completo}"

cd "$REPO"
mkdir -p logs
# shellcheck disable=SC1091
source "$WS/env.sh"

# --------------------------------------------------------------------------- #
echo "=== 1/4  entorno ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
NPROC=$(nproc)
VRAM_GB=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) / 1024 ))
echo "  vCPU: $NPROC | VRAM: ${VRAM_GB} GB"

python - <<'PY'
import sys, glob, torch
n = len(glob.glob("/workspace/cache/epochs_cache_ica/sub-*_full.npz"))
if n != 54:
    sys.exit(f"  x el cache tiene {n} sujetos, se esperaban 54. Corre antes: bash pod/preparar.sh eegnet")
if not torch.cuda.is_available():
    sys.exit("  x sin GPU")
print(f"  ok: 54 sujetos | torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
PY

# Reparto de workers. Dos topes: la VRAM (cada worker sube su train entero) y los vCPU
# (el cuello es lanzar kernels desde CPU, asi que pedir mas workers que nucleos no ayuda).
# Se deja 1 vCPU y ~6 GB de margen para el monitor y el sistema.
if [ -n "${1:-}" ] && [ -n "${2:-}" ]; then
    W_EEG="$1"; W_CONF="$2"
else
    CABEN_VRAM=$(( (VRAM_GB - 4) / GB_POR_WORKER ))
    CABEN_CPU=$(( NPROC - 1 ))
    TOTAL=$(( CABEN_VRAM < CABEN_CPU ? CABEN_VRAM : CABEN_CPU ))
    [ "$TOTAL" -lt 2 ] && TOTAL=2
    W_EEG=$(( (TOTAL + 1) / 2 ))
    W_CONF=$(( TOTAL - W_EEG ))
    echo "  caben $TOTAL workers (VRAM da para $CABEN_VRAM, CPU para $CABEN_CPU)"
fi
echo "  reparto -> EEGNet: $W_EEG workers | Conformer: $W_CONF workers"
echo ""

COMUN=(--dataset openbmi --channels completo --norm global --cache "$CACHE"
       --mode loso --seeds $SEEDS)

# --------------------------------------------------------------------------- #
if [ "$MODO" != "solo-monitor" ]; then
    echo "=== 2/4  lanzando los dos entrenamientos ==="
    nohup python scripts/train_baseline.py "${COMUN[@]}" --model eegnet \
        --workers "$W_EEG"  > logs/bloque_a_eegnet.log    2>&1 &
    PID_EEG=$!
    nohup python scripts/train_baseline.py "${COMUN[@]}" --model conformer \
        --workers "$W_CONF" > logs/bloque_a_conformer.log 2>&1 &
    PID_CONF=$!
    echo "  EEGNet   PID $PID_EEG   -> logs/bloque_a_eegnet.log"
    echo "  Conformer PID $PID_CONF -> logs/bloque_a_conformer.log"
    # Los workers arrancan ESCALONADOS (8 s entre uno y otro, ver _repartir) para no pelearse
    # por la VRAM, asi que el ultimo tarda 8*(W-1) s solo en lanzarse, mas ~1 min de torch y
    # cache. Esperar menos hace que el monitor arranque mostrando ceros y asuste sin motivo.
    ESPERA=$(( 60 + 8 * (W_EEG > W_CONF ? W_EEG : W_CONF) ))
    echo "  (arranque escalonado; esperando ${ESPERA}s a que carguen torch y el cache)"
    sleep "$ESPERA"
else
    echo "=== 2/4  OMITIDO (solo-monitor) ==="
    PID_EEG=""; PID_CONF=""
fi
echo ""

# --------------------------------------------------------------------------- #
echo "=== 3/4  monitor en vivo (Ctrl-C solo cierra el monitor, NO el entrenamiento) ==="
echo ""
set +e
python scripts/monitor.py --semillas 2 --filtro _loso_ --intervalo 30
set -e

# Si el monitor termino porque todo esta completo, aun asi hay que esperar a que los
# procesos cierren y fusionen sus shards.
if [ -n "$PID_EEG" ]; then
    echo ""
    echo "  esperando a que los procesos terminen de fusionar los CSV..."
    wait "$PID_EEG"  || echo "  [aviso] EEGNet termino con error, revisa logs/bloque_a_eegnet.log"
    wait "$PID_CONF" || echo "  [aviso] Conformer termino con error, revisa logs/bloque_a_conformer.log"
fi

# --------------------------------------------------------------------------- #
echo ""
echo "=== 4/4  verificacion ==="
python - <<'PY'
import csv, sys
from pathlib import Path
base = Path("results/reports/openbmi")
esperado = {
    "baseline_eegnet_loso_completo_global_epochs_cache_ica.csv":    ["2027", "2028"],
    "baseline_conformer_loso_completo_global_epochs_cache_ica.csv": ["2027", "2028"],
}
malo = False
for nom, semillas in esperado.items():
    p = base / nom
    if not p.exists():
        print(f"  x FALTA {nom}"); malo = True; continue
    filas = list(csv.DictReader(open(p)))
    ids = [r["run_id"] for r in filas]
    dup = len(ids) - len(set(ids))
    cuenta = {}
    for r in filas:
        cuenta[r["seed"]] = cuenta.get(r["seed"], 0) + 1
    print(f"  {nom}")
    print(f"    filas={len(filas)}  duplicados={dup}  por semilla={dict(sorted(cuenta.items()))}")
    for s in semillas:
        if cuenta.get(s, 0) != 54:
            print(f"    x semilla {s}: {cuenta.get(s,0)} de 54"); malo = True
    if dup:
        print("    x hay run_id duplicados"); malo = True
if malo:
    sys.exit("  VERIFICACION FALLIDA: relanza el mismo comando, la reanudacion salta lo hecho")
print("  ok: EEGNet y Conformer completos a 3 semillas en LOSO")
PY

echo ""
echo "================================================================"
echo " BLOQUE A TERMINADO. Baja los resultados desde tu portatil:"
echo ""
echo "   scp -P <puerto> -r root@<ip>:/workspace/tesis/results/reports/openbmi ./results/pod_bloque_a"
echo ""
echo " NO destruyas el pod hasta confirmar que los CSV llegaron completos."
echo "================================================================"
