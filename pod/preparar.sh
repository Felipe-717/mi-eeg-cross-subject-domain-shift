#!/usr/bin/env bash
# ============================================================================
# preparar.sh — deja el pod LISTO para entrenar, con preflight fail-fast.
#
# Encadena las piezas que YA existen (setup.sh + env.sh + cache) y ANTES de
# gastar GPU verifica las tres cosas que nos han hecho perder tiempo:
#   1) versiones de libreria != requirements.txt
#   2) cache ausente o incompleto
#   3) el modelo no construye / no cabe en VRAM con el batch real
#
# NO reinstala ni reinventa nada: reutiliza pod/setup.sh. Idempotente.
#
# UN SOLO COMANDO deja el pod listo: instala entorno, BAJA el cache (dataset publico
# en HF, sin login), lo expande, BAJA el checkpoint de LaBraM si hace falta, verifica
# versiones y hace el preflight. Ya no hay comandos sueltos que correr a mano.
#
# Uso:
#   bash pod/preparar.sh [modelo]        # modelo por defecto: conformer
#   bash pod/preparar.sh labram          # ademas baja labram-base.pt
# ============================================================================
set -euo pipefail

MODELO="${1:-conformer}"
WS="${WS:-/root/ws}"       # disco del CONTENEDOR (el volumen de red se colgo el 2026-11-01)
CACHE="$WS/cache/epochs_cache_ica"

# Descarga un dataset de HF con el CLI que exista (hf en versiones nuevas, huggingface-cli
# en las antiguas). El repo del cache es publico: no necesita token.
_hf_dataset() {   # $1=repo  $2=local_dir
    if command -v hf >/dev/null 2>&1; then
        hf download "$1" --repo-type dataset --local-dir "$2"
    else
        huggingface-cli download "$1" --repo-type dataset --local-dir "$2"
    fi
}

echo "=== 1/5  setup base (env + deps + torch/GPU) ==="
bash "$WS/tesis/pod/setup.sh"
# shellcheck disable=SC1091
source "$WS/env.sh"

echo "=== 2/5  versiones EXACTAS vs requirements.txt ==="
python - <<'PY'
import importlib.metadata as M, pathlib, sys
req = (pathlib.Path(__import__("os").environ["TESIS"]) / "pod/requirements.txt").read_text().splitlines()
mal = []
for ln in req:
    ln = ln.split("#")[0].strip()
    if "==" not in ln:                 # ignora vacios, comentarios y '>='
        continue
    name, ver = ln.split("==")
    name = name.split("[")[0].strip()  # 'braindecode[hub]' -> 'braindecode'
    try:
        got = M.version(name)
    except M.PackageNotFoundError:
        mal.append(f"{name}: NO INSTALADO (esperado {ver})"); continue
    if got != ver:
        mal.append(f"{name}: instalado {got}, esperado {ver}")
if mal:
    print("  x versiones que NO cuadran:")
    for x in mal: print("     -", x)
    sys.exit("  Aborta: reinstala con  pip install -r pod/requirements.txt")
print("  ok: todas las versiones coinciden con requirements.txt")
PY

echo "=== 3/5  cache ICA presente (se baja y expande solo si falta) ==="
if ! ls "$CACHE"/sub-*_full.npz >/dev/null 2>&1; then
    echo "  cache ausente: descargando Felipe717/openbmi-ica (dataset PUBLICO, sin login)..."
    _hf_dataset Felipe717/openbmi-ica "$WS/cache"
    echo "  expandiendo el consolidado fp16 a los 54 .npz por sujeto..."
    python "$WS/tesis/pod/expandir_cache.py"
fi
if ! ls "$CACHE"/sub-*_full.npz >/dev/null 2>&1; then
    echo "  x el cache sigue ausente tras la descarga: revisa la conexion a HF"
    exit 1
fi
echo "  ok: $(ls "$CACHE"/sub-*_full.npz | wc -l) sujetos en $CACHE"

echo "=== 4/5  dependencias especiales del modelo ==="
if [ "$MODELO" = "eegmamba" ]; then
    # EEGMamba usa Mamba2 -> kernels CUDA de mamba-ssm + causal-conv1d. Compilan con nvcc
    # contra el torch de la imagen, por eso van con --no-build-isolation (para que vean ese
    # torch y no bajen otro). Solo Linux+CUDA: por eso NO están en requirements.txt (que se
    # instala en local para EEGNet/Conformer). La A40 es sm_86, la misma familia que compila
    # sin tocar nada. Si ya están instalados, pip no recompila.
    # nvcc no siempre está en el PATH de las imágenes de RunPod, aunque el toolkit sí está
    # en /usr/local/cuda-*. Se exporta CUDA_HOME/PATH para que, si no hay wheel prebuilt
    # para este torch/cuda, mamba-ssm pueda COMPILAR los kernels en vez de fallar.
    if ! command -v nvcc >/dev/null 2>&1; then
        CUDA_DIR=$(ls -d /usr/local/cuda-* /usr/local/cuda 2>/dev/null | sort -V | tail -1)
        if [ -n "$CUDA_DIR" ] && [ -x "$CUDA_DIR/bin/nvcc" ]; then
            export CUDA_HOME="$CUDA_DIR"
            export PATH="$CUDA_DIR/bin:$PATH"
            echo "  nvcc no estaba en PATH; usando $CUDA_DIR"
        else
            echo "  AVISO: sin nvcc y sin toolkit CUDA. Si no hay wheel prebuilt, la compilación fallará."
        fi
    fi
    # ninja acelera la compilación; transformers lo necesita mamba_ssm al importar (utils.hf).
    # Ninguno arrastra torch. VERIFICADO en el pod 2026-08-28: con esto torch se queda en 2.8.
    pip install -q ninja transformers
    echo "  instalando mamba-ssm + causal-conv1d con --no-deps (compila kernels; minutos)..."
    # Dos cosas CLAVE, aprendidas a golpes:
    #  1) --no-deps: sin él, mamba-ssm 2.3.x arrastra transformers+tvm+torch 2.13/cu13 que
    #     PISA el torch 2.8 de la imagen y rompe el .so compilado (undefined symbol).
    #  2) Estas versiones EXACTAS (1.7.0 / 2.3.2.post1) y NO las viejas (1.4.0 / 2.2.2):
    #     los sdist viejos de PyPI vienen SIN la carpeta csrc/ (fuentes CUDA) y ninja no tiene
    #     qué compilar. Las 1.7.0/2.3.2.post1 traen csrc completo y compilan contra torch 2.8.
    pip install --no-build-isolation --no-deps "causal-conv1d==1.7.0" "mamba-ssm==2.3.2.post1"
    python - <<'PY'
import sys
try:
    import mamba_ssm
    from mamba_ssm.modules.mamba2 import Mamba2                      # noqa: F401
    from mamba_ssm.ops.triton.layer_norm import RMSNorm             # noqa: F401
    import causal_conv1d                                            # noqa: F401
except Exception as e:
    sys.exit(f"  x mamba-ssm/causal-conv1d NO importan tras instalar: {type(e).__name__}: {e}")
print(f"  ok: mamba-ssm {mamba_ssm.__version__} con Mamba2 + kernels Triton")
PY
fi
if [ "$MODELO" = "labram" ]; then
    # Se precachea el repo BUENO (position_embedding 129, casa con braindecode 1.6.1).
    # build_model lo carga desde la cache de HF e interpola el temporal_embedding.
    python - <<'PY'
import sys
try:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("braindecode/labram-pretrained", "model.safetensors")
except Exception as e:
    sys.exit(f"  x no se pudieron bajar los pesos de LaBraM: {e}")
print(f"  ok: pesos LaBraM en cache -> {p}")
PY
else
    echo "  ok: $MODELO no usa checkpoint preentrenado"
fi

echo "=== 5/5  PREFLIGHT del modelo ($MODELO): build + forward/backward + VRAM ==="
python - "$MODELO" "$CACHE" <<'PY'
import sys, glob, numpy as np, torch
sys.path.insert(0, __import__("os").environ["TESIS"])   # para 'import src.training'
import src.training as tr
modelo, cache = sys.argv[1], sys.argv[2]
# ch_names REALES del cache: LaBraM los necesita (con ch_names=None exige los 128 del
# vocabulario). Para eegnet/conformer se ignoran, pero fijan n_chans coherente.
f = sorted(glob.glob(f"{cache}/sub-*_full.npz"))[0]
ch_names = [str(c) for c in np.load(f, allow_pickle=True)["ch_names"]]
n_chans, n_times, n_classes, BATCH = len(ch_names), 801, 2, 256   # OpenBMI completo, batch del baseline
if not torch.cuda.is_available():
    sys.exit("  x sin GPU")
dev = torch.device("cuda")
torch.cuda.reset_peak_memory_stats(dev)
m = tr.build_model(modelo, n_chans=n_chans, n_times=n_times, n_classes=n_classes,
                   ch_names=ch_names).to(dev)
n = sum(p.numel() for p in m.parameters())
x = torch.randn(BATCH, n_chans, n_times, device=dev)
m.train()
out = tr._forward_logits(m, x)
assert out.shape == (BATCH, n_classes), f"salida inesperada: {tuple(out.shape)}"
out.float().pow(2).mean().backward()               # forward+backward REALES -> VRAM de verdad
torch.cuda.synchronize()
pico = torch.cuda.max_memory_allocated(dev) / 1e9
tot = torch.cuda.get_device_properties(dev).total_memory / 1e9
print(f"  ok: {modelo}: {n:,} params | out={tuple(out.shape)} | "
      f"VRAM pico ~ {pico:.1f} GB / {tot:.1f} GB @ batch={BATCH}")
PY

echo ""
echo "================================================================"
echo " LISTO. Lanza el entrenamiento en tmux (ver pod/MANANA_CONFORMER.md)."
echo "================================================================"
