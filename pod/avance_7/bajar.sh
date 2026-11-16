#!/usr/bin/env bash
# Copia de seguridad FUERA del pod: baja el paquete de resultados de la ejecución cada N
# minutos a avance_7/respaldo/. Se ejecuta en el PC, no en el pod.
#
# Por que existe: la ejecución vive en el disco del contenedor, que se pierde al destruir o
# reiniciar el pod. El 2026-11-01 los resultados quedaron atrapados en un volumen de red
# caido; con esto, lo peor que se pierde es lo de la ultima vuelta.
#
# Uso (Git Bash en el PC):
#     bash pod/avance_7/bajar.sh <ip> <puerto> [minutos]      # por defecto cada 30 min
set -uo pipefail          # sin -e: un fallo de red no debe cortar el bucle

IP="${1:?falta la IP}"
PUERTO="${2:?falta el puerto}"
MIN="${3:-30}"
KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
WS_POD="${WS_POD:-/root/ws}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$REPO/avance_7/respaldo"
mkdir -p "$DEST"

while true; do
    sello=$(date +%Y%m%d_%H%M)
    if scp -q -o ConnectTimeout=20 -i "$KEY" -P "$PUERTO" \
           "root@$IP:$WS_POD/avance_7_resultados.tgz" "$DEST/tmp.tgz" \
       && tar tzf "$DEST/tmp.tgz" >/dev/null 2>&1; then
        mv "$DEST/tmp.tgz" "$DEST/avance_7_resultados_$sello.tgz"
        cp "$DEST/avance_7_resultados_$sello.tgz" "$DEST/avance_7_resultados_ultimo.tgz"
        scp -q -o ConnectTimeout=20 -i "$KEY" -P "$PUERTO" \
            "root@$IP:$WS_POD/avance_7.log" "$DEST/avance_7.log" || true
        # Se conservan las 6 copias fechadas mas recientes.
        ls -t "$DEST"/avance_7_resultados_2*.tgz | tail -n +7 | xargs -r rm -f
        echo "[$sello] ok: $(du -h "$DEST/avance_7_resultados_ultimo.tgz" | cut -f1)"
    else
        rm -f "$DEST/tmp.tgz"
        echo "[$sello] sin paquete todavia o pod inaccesible; reintento en $MIN min"
    fi
    sleep $((MIN * 60))
done
