#!/usr/bin/env bash
# ============================================================================
# correr_eegnet.sh — completa EEGNet a 5 semillas, de una sola vez.
#
# Hace, en orden:
#   1) comprueba entorno, GPU y cache
#   2) BENCHMARK de --workers (1, 2, 4, 8, 12, 16) para ver cuantos procesos aguanta la GPU
#   3) within, las 2 semillas que faltan
#   4) LOSO, las 4 semillas que faltan
#   5) verifica los CSV y deja lista la orden de descarga
#
# Uso:
#   bash pod/correr_eegnet.sh          # mide y elige el mejor numero de workers
#   bash pod/correr_eegnet.sh 8        # salta el benchmark y usa 8
#   bash pod/correr_eegnet.sh 8 solo-bench   # solo mide, no entrena
#
# POR QUE SOLO LAS SEMILLAS QUE FALTAN
#   `pod/subir.sh` excluye `results/` a proposito (un log de otra maquina hace que la
#   reanudacion salte trabajo que no esta hecho). Como el pod arranca con results vacio,
#   la reanudacion por run_id NO puede saber que la semilla 2026 ya se ejecuto: habria que
#   subir los CSV, y eso es justo lo que no se hace. Por eso las semillas van EXPLICITAS.
#   Lo ya ejecutado se combina en local al bajar los resultados.
#
# POR QUE EL BENCHMARK USA UN DIRECTORIO DE CHECKPOINTS APARTE
#   El benchmark entrena 3 epocas para medir velocidad. El `run_id` del checkpoint NO
#   incluye el `--log-tag`, asi que esas corridas truncadas escribirian un checkpoint con
#   finished=True y el entrenamiento REAL las daria por terminadas con 3 epocas. Aislar
#   `TESIS_CKPT` (y borrarlo despues) es lo que impide envenenar los folds.
# ============================================================================
set -euo pipefail

WORKERS_FIJO="${1:-}"
MODO="${2:-completo}"

WS=/workspace
REPO="$WS/tesis"
# Ruta ABSOLUTA a proposito: `--cache` se resuelve relativo al cwd, y el cache real vive
# en /workspace/cache, no en /workspace/tesis. Un nombre pelado hacia que los workers
# murieran con 'No existe el cache'. El .name sigue siendo epochs_cache_ica, asi que el
# sufijo del CSV no cambia.
CACHE_NOMBRE=/workspace/cache/epochs_cache_ica
SEEDS_WITHIN="2758 9779"                 # faltan 2 (ya estaban 2026 2027 2028)
SEEDS_LOSO="2027 2028 2758 9779"         # faltan 4 (ya estaba 2026)

cd "$REPO"
# shellcheck disable=SC1091
source "$WS/env.sh"

COMUN=(--dataset openbmi --model eegnet --channels completo --norm global
       --cache "$CACHE_NOMBRE")

# --------------------------------------------------------------------------- #
echo "=== 1/5  entorno, GPU y cache ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
NPROC=$(nproc)
VRAM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
VRAM_GB=$((VRAM_GB / 1024))
echo "  vCPU disponibles: $NPROC | VRAM: ${VRAM_GB} GB"
python - <<'PY'
import sys, glob, torch
n = len(glob.glob("/workspace/cache/epochs_cache_ica/sub-*_full.npz"))
if n != 54:
    sys.exit(f"  x el cache tiene {n} sujetos, se esperaban 54. Corre antes: bash pod/preparar.sh eegnet")
if not torch.cuda.is_available():
    sys.exit("  x sin GPU")
print(f"  ok: 54 sujetos en cache | torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
PY
# El cuello de EEGNet es el lanzamiento de kernels desde CPU, asi que cada worker necesita
# su nucleo: no tiene sentido pedir mas workers que vCPU. El otro limite es la VRAM
# (~5 GB por worker). El benchmark respeta los dos.
echo ""

# --------------------------------------------------------------------------- #
elegir_workers() {
    echo "=== 2/5  benchmark de --workers (16 folds LOSO, 3 epocas, checkpoints aislados) ==="
    local bench_ckpt="$WS/ckpt_bench"
    local mejor_n=1 mejor_t=999999
    rm -rf "$bench_ckpt"
    for N in 1 2 4 8 12 16; do
        if [ "$N" -gt "$NPROC" ]; then
            echo "  workers=$N: omitido, solo hay $NPROC vCPU"
            continue
        fi
        # Cada worker sube su train a VRAM (~4,5 GB) y ademas su propio contexto CUDA
        # (~0,5 GB). Si no cabe, train_run NO falla: degrada al pipeline de CPU, que es
        # mucho mas lento y ademas silencioso salvo por un "[aviso]" en el log. Mejor
        # no pedir mas workers de los que caben.
        if [ $((N * 5)) -gt "$VRAM_GB" ]; then
            echo "  workers=$N: omitido, no cabe en ${VRAM_GB} GB de VRAM (~5 GB por worker)"
            continue
        fi
        rm -rf "$bench_ckpt"
        rm -f results/reports/openbmi/*__bench*.csv results/reports/openbmi/*_bench.csv
        local t0 t1 dt
        t0=$(date +%s)
        TESIS_CKPT="$bench_ckpt" python scripts/train_baseline.py "${COMUN[@]}" \
            --mode loso --loso-folds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
            --seeds 2026 --max-epochs 3 \
            --log-tag bench --workers "$N" > "logs/bench_w${N}.log" 2>&1 || {
                echo "  workers=$N: FALLO, ver logs/bench_w${N}.log"; continue; }
        t1=$(date +%s); dt=$((t1 - t0))
        echo "  workers=$N -> ${dt}s"
        if [ "$dt" -lt "$mejor_t" ]; then mejor_t=$dt; mejor_n=$N; fi
    done
    rm -rf "$bench_ckpt"
    rm -f results/reports/openbmi/*__bench*.csv results/reports/openbmi/*_bench.csv
    WORKERS=$mejor_n
    echo "  -> elegido --workers $WORKERS (${mejor_t}s en el benchmark)"
    echo ""
}

mkdir -p logs
if [ -n "$WORKERS_FIJO" ]; then
    WORKERS="$WORKERS_FIJO"
    echo "=== 2/5  benchmark OMITIDO, se usa --workers $WORKERS ==="
    echo ""
else
    elegir_workers
fi

if [ "$MODO" = "solo-bench" ]; then
    echo "Modo solo-bench: no se entrena. Relanza con:  bash pod/correr_eegnet.sh $WORKERS"
    exit 0
fi

# --------------------------------------------------------------------------- #
# within primero: es lo mas barato (2,6 h secuenciales) y ejerce todo el camino
# (reparto, CSV por worker, fusion) antes de comprometer las ~27 h de LOSO.
echo "=== 3/5  within, semillas $SEEDS_WITHIN (--workers $WORKERS) ==="
time python scripts/train_baseline.py "${COMUN[@]}" \
    --mode within --per-session --seeds $SEEDS_WITHIN --workers "$WORKERS"
echo ""

echo "=== 4/5  LOSO, semillas $SEEDS_LOSO (--workers $WORKERS) ==="
time python scripts/train_baseline.py "${COMUN[@]}" \
    --mode loso --seeds $SEEDS_LOSO --workers "$WORKERS"
echo ""

# --------------------------------------------------------------------------- #
echo "=== 5/5  verificacion de los CSV ==="
python - <<'PY'
import csv, sys
from pathlib import Path
base = Path("results/reports/openbmi")
esperado = {
    "baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica.csv": (["2758","9779"], 1080),
    "baseline_eegnet_loso_completo_global_epochs_cache_ica.csv":            (["2027","2028","2758","9779"], 54),
}
malo = False
for nom, (semillas, por_semilla) in esperado.items():
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
        if cuenta.get(s, 0) != por_semilla:
            print(f"    x semilla {s}: {cuenta.get(s,0)} filas, se esperaban {por_semilla}")
            malo = True
    if dup:
        print("    x hay run_id duplicados"); malo = True
if malo:
    sys.exit("  VERIFICACION FALLIDA: revisa logs/ y relanza (la reanudacion salta lo hecho)")
print("  ok: todo completo y sin duplicados")
PY

echo ""
echo "================================================================"
echo " EEGNet TERMINADO. Baja los resultados desde tu portatil:"
echo ""
echo "   scp -P <puerto> -r root@<ip>:/workspace/tesis/results/reports/openbmi ./results/pod_eegnet_semillas"
echo ""
echo " Y NO destruyas el pod hasta confirmar que los CSV llegaron completos."
echo "================================================================"
