#!/usr/bin/env bash
# Sube el repo al pod EXCLUYENDO lo que no debe viajar.
#
# POR QUE ESTE SCRIPT EXISTE
#   El 2026-07-26 se subio el repo entero con `tar` y viajaron los CSV de resultados
#   locales. Tanto `build_cache.py` como `train_baseline.py` deciden que trabajo saltar
#   leyendo SU CSV de log, asi que en el pod creyeron que todo estaba hecho:
#     * build_cache proceso 0 sujetos de 54 -> `verify` fallo con el cache vacio
#     * el within de 16 sujetos se habria saltado en silencio (no llego a pasar por
#       casualidad: el sujeto del benchmark era el unico que no estaba en el log)
#   No es un descuido puntual: CUALQUIER script con reanudacion por log es vulnerable a
#   recibir el log de otra maquina. La regla es que los resultados no viajan.
#
# Uso:
#   bash pod/subir.sh <ip> <puerto>
#   bash pod/subir.sh 69.30.85.193 22094
#   WS_POD=/workspace bash pod/subir.sh <ip> <puerto>   # solo si se usa volumen de red
set -euo pipefail

IP="${1:?falta la IP}"
PUERTO="${2:?falta el puerto}"
D="${WS_POD:-/root/ws}"   # disco del contenedor; WS_POD=/workspace para un volumen de red
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOMBRE="$(basename "$REPO")"

echo "Subiendo $REPO -> root@$IP:$PUERTO:$D/tesis"

cd "$(dirname "$REPO")"
tar czf - \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='results' \
    --exclude='data' \
    --exclude='*.docx' \
    --exclude='*.pdf' \
    --exclude='~$*' \
    "$NOMBRE" \
| ssh ${SSH_KEY:+-i "$SSH_KEY"} "root@$IP" -p "$PUERTO" "D=$D;"'
    set -e
    rm -rf /tmp/subida && mkdir -p /tmp/subida "$D/tesis"
    # --no-same-owner: los UID de Windows no existen en Linux y tar aborta sin esto
    tar xzf - -C /tmp/subida --no-same-owner --no-same-permissions
    cp -r /tmp/subida/*/. "$D/tesis/"
    rm -rf /tmp/subida
    # Normaliza finales de linea: si un .sh viaja con CRLF desde Windows, bash lo
    # rompe en la primera linea (`set -euo pipefail\r` -> "invalid option name").
    # Barato y idempotente, asi que se hace siempre en vez de confiar en el editor.
    find "$D/tesis" -name "*.sh" -exec sed -i "s/\r$//" {} +
    mkdir -p "$D/tesis/logs"
    mkdir -p "$D/tesis/results/reports/openbmi"
    echo "--- subido ---"
    ls "$D/tesis"
    echo "--- results/ (debe estar VACIO) ---"
    ls -A "$D/tesis/results/reports/openbmi" || true
'

cat <<'EOF'

Nota: `results/` se excluye a proposito y se crea vacio en el pod.
Los resultados que genere el pod se traen DESPUES con:

    scp -P <puerto> -r root@<ip>:/root/ws/tesis/results/reports/openbmi ./results/pod
EOF
