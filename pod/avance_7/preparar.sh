#!/usr/bin/env bash
# Deja un pod NUEVO listo para la ejecución y la lanza en tmux, con el panel de épocas.
#
# Reutiliza lo que ya existe y está probado:
#   pod/punto6b/preparar.sh  entorno, kernels de Mamba2 y los DOS caches (OpenBMI, WBCIC)
#   pod/preparar.sh labram   pesos preentrenados de LaBraM + preflight de VRAM
#
# TODO VA AL DISCO DEL CONTENEDOR ($WS, por defecto /root/ws). El 2026-11-01 el volumen de
# red de EU-SE-1 se colgó a mitad de ejecución: procesos en E/S bloqueada, imposibles de
# matar, y los resultados atrapados. Este script se niega a trabajar sobre un FUSE de red.
# El disco del contenedor se pierde al destruir el pod: desde el PC, en paralelo,
#     bash pod/avance_7/bajar.sh <ip> <puerto>
#
# Uso, con el repo ya subido (SSH_KEY=~/.ssh/id_ed25519 bash pod/subir.sh <ip> <puerto>):
#     tmux new -s avance_7
#     bash /root/ws/tesis/pod/avance_7/preparar.sh            # prepara y lanza todo
#     bash /root/ws/tesis/pod/avance_7/preparar.sh F5 F6      # prepara y lanza esas fases
#
# Requisitos del pod: GPU de 48 GB (A40 o similar; 3 trabajadores LOSO necesitan ~19 GB),
# SIN volumen de red, disco de contenedor de al menos 150 GB (caches ~60 GB + resultados).
set -euo pipefail
export WS="${WS:-/root/ws}"
export AVANCE7_WS="$WS"
T=$WS/tesis

echo "=== 0  el destino no puede ser un volumen de red ==="
mkdir -p "$WS"
fs=$(stat -f -c %T "$WS")
if [ "$fs" = "fuseblk" ] || [ "$fs" = "fuse" ] || grep -qs " $WS fuse" /proc/mounts; then
    echo "  x $WS es un FUSE de red ($fs). Usa el disco del contenedor (WS=/root/ws)."
    [ "${PERMITIR_FUSE:-0}" = "1" ] || exit 1
fi
echo "  ok: $WS en $fs"

bash "$T/pod/punto6b/preparar.sh"
bash "$T/pod/preparar.sh" labram
# shellcheck disable=SC1091
source "$WS/env.sh"
python -m pip install -q "scipy==1.17.1"      # filtros de las subbandas (ya suele estar)

echo ""
echo "=== espacio en disco ==="
df -h "$WS"
libre=$(df -BG --output=avail "$WS" | tail -1 | tr -dc '0-9')
if [ "$libre" -lt 40 ]; then
    echo "  x solo ${libre} GB libres en $WS: los caches derivados necesitan ~30 GB"; exit 1
fi

echo ""
echo "=== plan ==="
cd "$T"
# Sin "| head": con pipefail, el SIGPIPE de head mataba el script antes de lanzar.
python pod/avance_7/avance_7.py --plan ${1:+--fases "$@"} > "$WS/avance_7_plan.txt"
head -14 "$WS/avance_7_plan.txt"

# AVANCE7_F4_GRANDES va explicito en la orden: una sesion nueva de tmux no hereda el entorno.
LANZAR="cd $T && source $WS/env.sh && AVANCE7_WS=$WS AVANCE7_F4_GRANDES='${AVANCE7_F4_GRANDES:-augcr augfft w02}' python pod/avance_7/avance_7.py ${1:+--fases $*} 2>&1 | tee -a $WS/avance_7.log"
EPOCAS="WS=$WS bash $T/pod/avance_7/ver_epocas.sh"
if [ -n "${TMUX:-}" ]; then
    # Ya estamos dentro de tmux: se lanza aqui mismo, sin abrir otra sesion. El panel de
    # epocas se abre debajo, una sola vez (si ya hay dos paneles, no se toca).
    if [ "$(tmux display -p '#{window_panes}')" -lt 2 ]; then
        tmux split-window -d -v -l 45% "$EPOCAS"
    fi
    echo ""
    echo "Lanzando en esta sesion de tmux. Log continuo: $WS/avance_7.log"
    eval "$LANZAR"
elif command -v tmux >/dev/null 2>&1; then
    tmux new-session -d -s avance_7 -x 200 -y 50 "$LANZAR"
    tmux split-window -d -v -l 45% -t avance_7:0 "$EPOCAS"
    echo ""
    echo "Lanzada en tmux (sesion 'avance_7', panel de epocas abajo):  tmux attach -t avance_7"
    echo "Log continuo: $WS/avance_7.log  |  estado: $T/results/avance_7/estado.json"
    echo "Resultados empaquetados tras cada trabajo: $WS/avance_7_resultados.tgz"
else
    echo "  x no hay tmux: lanza a mano  python pod/avance_7/avance_7.py"
fi
