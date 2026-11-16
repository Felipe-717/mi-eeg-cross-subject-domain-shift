#!/usr/bin/env bash
# Panel de seguimiento: muestra en vivo el log mas reciente (por trabajador o de trabajo).
# Lo abre preparar.sh debajo de la ejecución. Cada 60 s vuelve a buscar el log mas nuevo,
# porque al cambiar de trabajo o de trabajador cambia el archivo.
T="${WS:-/root/ws}/tesis"
while true; do
    f=$(ls -t "$T"/logs/*.log "$T"/results/avance_7/logs/*.log 2>/dev/null | head -1)
    clear
    if [ -z "$f" ]; then
        echo "(aun no hay logs en $T/logs ni en $T/results/avance_7/logs)"; sleep 30; continue
    fi
    echo ">>> $f"; echo
    timeout 60 tail -n 25 -f "$f"
done
