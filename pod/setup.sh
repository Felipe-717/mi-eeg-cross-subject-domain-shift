#!/usr/bin/env bash
# Prepara un pod de RunPod desde cero. IDEMPOTENTE: se puede relanzar sin romper nada.
#
# DONDE VIVE TODO: en $WS, por defecto /root/ws, en el disco del CONTENEDOR.
#   Hasta oct-2026 se usaba un volumen de red en /workspace para sobrevivir a un stop.
#   El 2026-11-01 el volumen de EU-SE-1 se colgo (E/S bloqueada, procesos imposibles de
#   matar) y se perdio la ejecucion. Ahora: pod SIN volumen de red, todo en el contenedor,
#   y los resultados se copian fuera del pod con pod/avance_7/bajar.sh. Se pierde al
#   destruir o reiniciar el pod; WS=/workspace sigue funcionando si se quiere un volumen.
#
# Uso:
#   bash pod/setup.sh
set -euo pipefail

WS="${WS:-/root/ws}"       # disco del CONTENEDOR (el volumen de red se colgo el 2026-11-01)
REPO_URL="${REPO_URL:-}"          # exportar si se quiere clonar en vez de subir por scp
RAW=/raw/mne_data                 # EFIMERO a proposito: son decenas de GB de crudo

echo "=== 1/5  directorios ==="
mkdir -p "$WS/cache" "$WS/results" "$WS/logs" "$WS/checkpoints" "$RAW"
if [ -n "$REPO_URL" ] && [ ! -d "$WS/tesis/.git" ]; then
    git clone "$REPO_URL" "$WS/tesis"
fi
echo "  ok"

echo "=== 2/5  entorno virtual en $WS ==="
# --system-site-packages: las imagenes de RunPod ya traen torch compilado para SU CUDA
# (p. ej. runpod/pytorch:...cu1281-torch280 -> torch 2.8.0 + cu12.8). Sin esta opcion el
# venv no lo veria y bajariamos otro torch de ~2.5 GB, ademas con una CUDA distinta a la
# de la imagen. Nuestras dependencias fijadas se instalan en el venv y tienen prioridad.
if [ ! -d "$WS/venv" ]; then
    python3 -m venv --system-site-packages "$WS/venv"
fi
# shellcheck disable=SC1091
source "$WS/venv/bin/activate"
python -m pip install --upgrade pip -q
echo "  $(python --version) en $WS/venv"

echo "=== 3/5  torch para la GPU de este pod ==="
# Si la imagen ya trae torch (lo normal en las plantillas de RunPod), se USA EL SUYO:
# esta compilado contra la CUDA de la imagen y bajar otro solo puede empeorarlo.
# Solo se instala si no hay ninguno.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
if python -c "import torch" 2>/dev/null; then
    echo "  torch ya presente en la imagen: no se reinstala"
else
    echo "  sin torch: instalando desde $TORCH_INDEX"
    pip install torch --index-url "$TORCH_INDEX" -q
fi
python - <<'PY'
import sys, torch
print(f"  torch {torch.__version__} | CUDA disponible: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("  ERROR: torch no ve la GPU. Revisa la imagen del pod.")
p = torch.cuda.get_device_properties(0)
sm = f"sm_{p.major}{p.minor}"
print(f"  {p.name} | {p.total_memory/1e9:.1f} GB | {sm}")
# Esta rueda de torch, ¿trae kernels para ESTA tarjeta? Es el fallo que rompe en
# Blackwell (RTX 5090, sm_120) con ruedas cu121: instala bien y falla al primer kernel.
# Un cubin sm_XY corre en sm_XZ con Z >= Y (misma generacion): las ruedas oficiales no
# traen sm_89 y la L40S (Ada) usa los de sm_86. Entre generaciones distintas solo vale PTX.
arcos = torch.cuda.get_arch_list()
compat = [a for a in arcos if a.startswith("sm_") and len(a) == 5
          and int(a[3]) == p.major and int(a[4]) <= p.minor]
if sm not in arcos and not compat:
    sys.exit(f"  ERROR: la rueda de torch no soporta {sm}. Compiladas: {arcos}\n"
             f"         Relanza con:  TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash pod/setup.sh")
PY

echo "=== 4/5  dependencias fijadas ==="
pip install -r "$WS/tesis/pod/requirements.txt" -q
echo "  ok"

echo "=== 5/5  variables de entorno ==="
cat > "$WS/env.sh" <<EOF
# Cargar con:  source $WS/env.sh
source $WS/venv/bin/activate
export MNE_DATA=$RAW                      # crudo -> EFIMERO (no lo queremos guardar)
export OPENBMI_PROC_ROOT=$WS/cache        # procesado -> persistente
export TESIS_LOGS=$WS/logs
# Checkpoints a /raw (disco local, nunca a un FUSE de red). Se escribe uno por
# epoca; con el entrenamiento a ~5 s/fold esa latencia de red pasaba a dominar el tiempo
# total. Son prescindibles: lo que se conserva son los CSV de metricas.
export TESIS_CKPT=/raw/checkpoints
export PYTHONPATH=$WS/tesis/src:\${PYTHONPATH:-}
export TESIS=$WS/tesis
# HF guarda el token en ~/.cache, que es EFIMERO: se perderia al parar el pod y habria
# que volver a autenticarse. Apuntando a $WS queda junto a todo lo demas.
# OJO: eso deja el token en el volumen -> razon de mas para DESTRUIR el pod al acabar.
export HF_HOME=$WS/hf
# Pesos preentrenados de LaBraM: se cargan desde el Hub (repo braindecode/labram-pretrained)
# y quedan en la cache de HF (\$HF_HOME, persistente). Override opcional del repo:
# OJO: este heredoc es <<EOF SIN comillas, asi que bash expande las variables AL ESCRIBIR
# env.sh, comentarios incluidos. Con `set -u`, un \$VAR que no exista aun en el pod aborta
# el setup entero. Paso el 2026-09-06 con \$HF_HOME en un pod recien creado: hay que
# escapar toda variable que deba llegar literal a env.sh.
export LABRAM_HF_REPO=braindecode/labram-pretrained

# CPUs REALES del contenedor. \`nproc\` devuelve las del HOST (p. ej. 96) mientras el
# cgroup te asigna muchas menos (8.5 en el A40 medido el 2026-07-26). numpy/MNE/sklearn
# leen nproc y lanzan un hilo OpenMP por CPU del host -> sobresuscripcion que RALENTIZA.
# Hay que mirar cgroup v2 Y v1: los pods de RunPod usan v1 (cpu.max no existe alli).
_cpus() {
    if [ -r /sys/fs/cgroup/cpu.max ]; then                       # cgroup v2
        read -r q p < /sys/fs/cgroup/cpu.max
        [ "\$q" != "max" ] && echo \$(( q / p )) && return
    fi
    if [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then          # cgroup v1
        q=\$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        p=\$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
        [ "\$q" -gt 0 ] && echo \$(( q / p )) && return
    fi
    nproc
}
export TESIS_CPUS=\$(_cpus)
export OMP_NUM_THREADS=\$TESIS_CPUS
export MKL_NUM_THREADS=\$TESIS_CPUS
export OPENBLAS_NUM_THREADS=\$TESIS_CPUS
export NUMEXPR_NUM_THREADS=\$TESIS_CPUS
cd \$TESIS
EOF
mkdir -p "$WS/hf"
echo "  escrito $WS/env.sh"

cat <<EOF

Listo. En CADA sesion nueva de terminal:

    source $WS/env.sh

Y despues:

    python pod/run.py --fase verify     # comprueba que el entorno reproduce la Fase 2

RECORDATORIO DE COSTES
  * Se cobra por SEGUNDO mientras el pod este encendido, haga algo o no.
    Los comandos son gratis; el pod olvidado encendido no.
  * PARAR el pod cuesta MAS por disco (\$0.20/GB/mes) que tenerlo encendido (\$0.10).
  * Al TERMINAR se borra el disco del contenedor ($WS): bajar los resultados ANTES.
EOF
