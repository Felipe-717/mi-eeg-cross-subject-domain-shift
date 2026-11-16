"""Esquema de entrenamiento unificado (Actividad 3.3, plan §3B / Anexo C).

Diseñado para reutilizarse SIN CAMBIOS por las siguientes arquitecturas de la
Fase 3 (Conformer, EEGMamba, LaBraM): mismo optimizador, mismo scheduler,
mismo protocolo de early stopping / checkpointing / augmentation, para que la
comparación entre familias sea justa (Anexo C.3: "comparación justa").

Uso típico (ver scripts/train_eegnet_baseline.py):
    ds = datasets.get("openbmi")                            # o "mind"
    y = training.encode_labels(y_texto, ds.CLASS_ORDER)     # etiquetas -> 0..n-1
    model = training.build_model("eegnet", n_chans=59, n_times=501, n_classes=ds.N_CLASSES)
    hist, _ = training.train_run(model, X_train, y_train, X_val, y_val,
                                  n_classes=ds.N_CLASSES, sfreq=ds.SFREQ,
                                  cfg=training.DEFAULT_TRAIN_CFG,
                                  device=device, ckpt_path=ckpt_path)
    metrics = training.evaluate(model, X_test, y_test, device=device, n_classes=ds.N_CLASSES)

NOTA de forma de entrada (importante, difiere de `adapters.describe()`):
los modelos de `braindecode` (EEGNetv4 incluido) reciben (batch, n_canales,
n_muestras) y añaden la dimensión de "canal de imagen" INTERNAMENTE
(`Ensure4d`). Por eso aquí NO se usa `adapters.adapt_raw` (que ya trae esa
dimensión añadida) para EEGNet/Conformer: se alimenta el tensor crudo 3D tal
como sale de `splits.build_fold` / `splits.build_within_fold`.
"""
from __future__ import annotations
import json
import os
import random
import time
from pathlib import Path
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix

import augment as aug

DEFAULT_TRAIN_CFG = {
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "batch_size_physical": 32,
    "batch_size_effective": 32,     # accumulation_steps = effective // physical
    "max_epochs": 200,
    "warmup_epochs": 10,
    "patience": 20,                 # early stopping sobre val-loss
    "label_smoothing": 0.1,
    "amp": True,
    "grad_clip": 1.0,               # recorte de norma de gradiente (estabiliza SSM/Mamba)
    "ckpt_cada": 1,                 # cada N epocas; 0 = solo el final (ver train_run)
    "augment": {                    # se aplica SOLO al split de train
        "time_shift_frac": 0.05,
        "gaussian_noise_std": 0.02,
        "freq_mask_hz": 2.0,
        "mixup_alpha": 0.0,          # desactivado por defecto para el baseline EEGNet
    },
}


def encode_labels(y, class_order) -> np.ndarray:
    """Mapea las etiquetas de texto del caché a índices 0..n-1, en el orden dado.

    `class_order` viene del módulo del dataset (`datasets.<nombre>.CLASS_ORDER`), no
    está fijado aquí: así este módulo sirve para cualquier dataset. El orden es fijo
    para que el índice de una clase signifique siempre lo mismo en todas las corridas,
    folds y arquitecturas.
    """
    idx = {str(c): i for i, c in enumerate(class_order)}
    faltan = sorted({str(v) for v in y} - set(idx))
    if faltan:
        raise ValueError(f"Etiquetas desconocidas en el caché: {faltan}. "
                         f"Esperadas: {list(class_order)}")
    return np.array([idx[str(v)] for v in y], dtype=np.int64)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


LABRAM_PATCH = 200   # 1 s por parche @ 200 Hz (input_size del repo oficial)

# EEGMamba fija patch=200 y d_model=200 (la rama espectral usa Linear(101, d_model) con
# 101 = 200//2+1 de la rFFT): no son ajustables sin reescribir el modelo. n_layer=12 es
# el del repo oficial. Ver src/eegmamba_vendored.py.
EEGMAMBA_PATCH = 200
EEGMAMBA_N_LAYER = 12


class _LabramWrapper(nn.Module):
    """Adapta LaBraM (braindecode) al esquema unificado, que alimenta (B, C, T).

    Tres cosas que LaBraM necesita y el resto de modelos no:
      1. `n_times` DEBE ser múltiplo de `patch_size` (braindecode valida
         "n_times must be divisible by patch_size", sin recorte interno). Nuestra
         ventana da 801 muestras (0..4 s inclusive), así que se recorta a 800.
      2. `ch_names` es un argumento keyword-only del forward que mapea cada canal a
         los embeddings posicionales preentrenados. braindecode normaliza el casing
         internamente (`_LABRAM_CANONICAL_INDEX[n.upper()]`), así que se pasan tal cual
         (p. ej. 'FTT9h' -> 'FTT9H' coincide con la clave del vocabulario).
      3. Escala de amplitud: LaBraM se preentrenó con µV/100. Con z-score por fold
         (--norm global) NO se aplica (`uv_scale=1`); para la variante "fiel al
         preentrenamiento" (--norm none) se pone `uv_scale=100`. Lo decide el gate.
    """

    def __init__(self, labram: nn.Module, keep: int, ch_names=None, uv_scale: float = 1.0):
        super().__init__()
        self.labram = labram
        self.keep = int(keep)
        # verbatim: braindecode ya hace n.upper() al mapear al vocabulario
        self.ch_names = [str(c) for c in ch_names] if ch_names else None
        self.uv_scale = float(uv_scale)

    def forward(self, x):
        if x.dim() == 4:                 # por si llega (B, 1, C, T)
            x = x.squeeze(1)
        if x.shape[-1] != self.keep:     # 801 -> 800 (múltiplo del parche)
            x = x[..., :self.keep]
        if self.uv_scale and self.uv_scale != 1.0:
            x = x / self.uv_scale
        return self.labram(x, ch_names=self.ch_names)


class _EEGMambaWrapper(nn.Module):
    """Adapta EEGMamba al esquema unificado, que alimenta (B, C, T).

    El backbone quiere (B, C, n_parches, 200); aquí se hace ese reshape en torch (mismo
    recorte que adapters.adapt_eegmamba: se trunca T al múltiplo de 200 más cercano por
    debajo). El backbone (con proj_out=Identity) devuelve (B, C, n_parches, d_model) y la
    cabeza produce los logits.

    Dos cabezas seleccionables por `EEGMAMBA_HEAD` (el gate elige, como en LaBraM):
      * 'avgpool' (por defecto): AdaptiveAvgPool sobre canales y parches -> Linear(200, k).
        Pocos parámetros, agnóstica al nº de canales (coherente con el ST-Adaptive del
        modelo) y más robusta en cross-subject. ~400 parámetros de cabeza.
      * 'flatten' (all_patch_reps_onelayer del repo): Linear(C*parches*200, k). Es la
        cabeza de sus ejemplos, pero con 62 canales son ~50k features -> puede sobreajustar
        en within (pocos trials). Útil para contrastar en el gate.

    Escala de amplitud: EEGMamba nativo divide por 100 (µV/100). Con z-score por fold
    (--norm global) se usa uv_scale=1; con --norm none se pone uv_scale=100.
    """

    def __init__(self, backbone: nn.Module, n_chans: int, n_patches: int, d_model: int,
                 n_classes: int, head: str = "avgpool", uv_scale: float = 1.0):
        super().__init__()
        self.backbone = backbone
        self.keep = int(n_patches) * EEGMAMBA_PATCH
        self.uv_scale = float(uv_scale)
        if head == "avgpool":
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),           # (B, d, C, parches) -> (B, d, 1, 1)
                nn.Flatten(),
                nn.Linear(d_model, n_classes))
        elif head == "flatten":
            self.classifier = nn.Sequential(
                nn.Flatten(),                           # (B, C*parches*d)
                nn.Linear(n_chans * n_patches * d_model, n_classes))
        else:
            raise ValueError(f"EEGMAMBA_HEAD desconocido: {head!r} (usa 'avgpool' o 'flatten')")
        self.head = head

    def forward(self, x):
        if x.dim() == 4:                    # por si llega (B, 1, C, T)
            x = x.squeeze(1)
        if x.shape[-1] != self.keep:        # trunca al múltiplo de 200 (p. ej. 801 -> 800)
            x = x[..., :self.keep]
        if self.uv_scale and self.uv_scale != 1.0:
            x = x / self.uv_scale
        b, c, t = x.shape
        x = x.reshape(b, c, self.keep // EEGMAMBA_PATCH, EEGMAMBA_PATCH)
        feats = self.backbone(x)            # (B, C, parches, d_model)
        if self.head == "avgpool":
            feats = feats.permute(0, 3, 1, 2)   # -> (B, d_model, C, parches) para el pool
        return self.classifier(feats)


# Repo con los pesos que CASAN con braindecode 1.6.1 (position_embedding = 129, vocabulario
# de 128 canales). OJO: NO usar el mirror `braindecode/Labram-Braindecode` (es de una versión
# vieja, con 64 canales -> position_embedding 65, incompatible).
LABRAM_HF_REPO = "braindecode/labram-pretrained"
_LABRAM_STATE_CACHE: dict = {}


def _labram_pretrained_state(repo: str, tgt_temporal_len: int):
    """State_dict preentrenado de LaBraM con el `temporal_embedding` interpolado.

    braindecode.Labram fija el nº de parches en la construcción y `from_pretrained` hace
    carga ESTRICTA: revienta si nuestra ventana (4 parches -> temporal 5) no coincide con la
    del preentrenamiento (16). Aquí se replica lo que hace el finetuning oficial de LaBraM:
    interpolar el `temporal_embedding` al nº de parches objetivo conservando el primer token.
    El `position_embedding` (129) ya coincide, no se toca. Se descarta `final_layer` (otra
    tarea) para reiniciar la cabeza. Cacheado por proceso para no re-descargar en cada fold.
    """
    key = (repo, tgt_temporal_len)
    if key in _LABRAM_STATE_CACHE:
        return _LABRAM_STATE_CACHE[key]
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    sd = load_file(hf_hub_download(repo, "model.safetensors"))
    te = sd.get("temporal_embedding")
    if te is not None and te.shape[1] != tgt_temporal_len:
        cls_t, patches = te[:, :1, :], te[:, 1:, :]                 # token 0 aparte + parches
        patches = F.interpolate(patches.transpose(1, 2), size=tgt_temporal_len - 1,
                                mode="linear", align_corners=False).transpose(1, 2)
        sd["temporal_embedding"] = torch.cat([cls_t, patches], dim=1)
    sd = {k: v for k, v in sd.items() if not k.startswith("final_layer")}
    _LABRAM_STATE_CACHE[key] = sd
    return sd


def build_model(name: str, n_chans: int, n_times: int, n_classes: int,
                ch_names=None) -> nn.Module:
    """Factoría de modelos. Punto de extensión explícito para las siguientes fases.

    `ch_names` solo lo usa LaBraM (mapeo a embeddings posicionales); los demás
    modelos lo ignoran, así la firma sigue siendo común al esquema unificado.
    """
    if name == "eegnet":
        # braindecode >= 1.0 expone la clase como `EEGNet`; en versiones anteriores
        # se llamaba `EEGNetv4` (verificado con braindecode 1.6.1: la firma es
        # n_chans / n_outputs / n_times).
        try:
            from braindecode.models import EEGNet as _EEGNet
        except ImportError:
            from braindecode.models import EEGNetv4 as _EEGNet
        return _EEGNet(n_chans=n_chans, n_outputs=n_classes, n_times=n_times)
    if name == "conformer":
        # EEG Conformer (Song et al. 2023), implementación de braindecode. Consume el MISMO
        # tensor (B, C, T) que EEGNet -> entra al esquema unificado sin adaptador.
        # Verificado con braindecode 1.6.1: 631.812 parámetros con n_chans=13, n_times=801.
        from braindecode.models import EEGConformer
        return EEGConformer(n_chans=n_chans, n_outputs=n_classes, n_times=n_times)
    if name == "labram":
        # LaBraM (Jiang et al., ICLR 2024), foundation model. Se usa la implementación de
        # braindecode (>=1.0 trae la clase; >=1.3.2 el from_pretrained; >=1.4 el extra
        # [hub]). braindecode.models.Labram consume (B, C, T) con neural_tokenizer=True.
        from braindecode.models import Labram
        patch = LABRAM_PATCH
        n_patches = n_times // patch
        if n_patches < 1:
            raise ValueError(f"La ventana ({n_times} muestras) es menor que un parche ({patch})")
        n_times_lab = n_patches * patch                 # 800 para 801
        # n_chans=128 es el TAMAÑO DEL VOCABULARIO de posiciones (position_embedding=129),
        # NO el nº de canales de los datos: esos entran por `ch_names` en el forward.
        model = Labram(n_chans=128, n_times=n_times_lab, n_outputs=n_classes,
                       patch_size=patch, neural_tokenizer=True)

        # --- pesos preentrenados (sin ellos NO es un foundation model) ---
        if os.environ.get("LABRAM_ALLOW_SCRATCH") == "1":
            warn("LaBraM SIN pesos preentrenados (LABRAM_ALLOW_SCRATCH=1): es un transformer "
                 "desde cero, NO un foundation model. Úsalo solo para el preflight.")
        else:
            repo = os.environ.get("LABRAM_HF_REPO", LABRAM_HF_REPO)
            state = _labram_pretrained_state(repo, model.temporal_embedding.shape[1])
            faltan, sobran = model.load_state_dict(state, strict=False)
            faltan = [k for k in faltan if not k.startswith("final_layer")]
            if sobran:
                warn(f"LaBraM: {len(sobran)} claves del checkpoint no se usaron (p. ej. {sobran[:3]})")
            if faltan:
                warn(f"LaBraM: {len(faltan)} pesos quedaron SIN cargar (p. ej. {faltan[:3]}); "
                     f"revisa que el checkpoint case con la arquitectura")

        # --- linear probing vs full fine-tuning (lo fija el gate) ---
        modo = os.environ.get("LABRAM_FINETUNE", "full").lower()
        if modo in ("linear", "probe", "linear_probe"):
            for p in model.parameters():
                p.requires_grad = False
            for p in model.final_layer.parameters():
                p.requires_grad = True
        elif modo != "full":
            raise ValueError(f"LABRAM_FINETUNE desconocido: {modo!r} (usa 'full' o 'linear')")

        uv = float(os.environ.get("LABRAM_UV_SCALE", "1"))
        return _LabramWrapper(model, keep=n_times_lab, ch_names=ch_names, uv_scale=uv)
    if name == "eegmamba":
        # EEGMamba (Wang et al. 2024) vendorizado (NO está en braindecode). Requiere
        # Linux + CUDA con kernels mamba-ssm + causal-conv1d + Triton -> corre en el pod.
        # DESDE CERO: sin pesos preentrenados. La versión pública del repo es la
        # simplificada de una tarea (sin MoE), y sus propios números en MI no superan de
        # forma clara a EEGNet/Conformer, así que entra como arquitectura eficiente (SSM)
        # comparada en igualdad, no como foundation model (ese punto lo cubre LaBraM).
        from eegmamba_vendored import EEGMamba
        n_patches = n_times // EEGMAMBA_PATCH
        if n_patches < 1:
            raise ValueError(f"La ventana ({n_times} muestras) es menor que un parche "
                             f"({EEGMAMBA_PATCH}): EEGMamba necesita >= 1 s a 200 Hz")
        backbone = EEGMamba(in_dim=EEGMAMBA_PATCH, out_dim=EEGMAMBA_PATCH,
                            d_model=EEGMAMBA_PATCH, n_layer=EEGMAMBA_N_LAYER)
        backbone.proj_out = nn.Identity()               # el backbone devuelve (B, C, parches, d_model)
        uv = float(os.environ.get("EEGMAMBA_UV_SCALE", "1"))
        head = os.environ.get("EEGMAMBA_HEAD", "avgpool").lower()
        return _EEGMambaWrapper(backbone, n_chans=n_chans, n_patches=n_patches,
                                d_model=EEGMAMBA_PATCH, n_classes=n_classes,
                                head=head, uv_scale=uv)
    raise ValueError(f"modelo desconocido: {name}")


def _forward_logits(model, xb):
    out = model(xb)
    if out.dim() > 2:            # algunas versiones de braindecode devuelven (B, n_classes, 1)
        out = out.squeeze(-1)
    return out


def _soft_cross_entropy(logits, target_probs, label_smoothing: float = 0.0):
    """Cross-entropy con targets blandos (one-hot, o mezclados por mixup)."""
    n_classes = logits.shape[-1]
    if label_smoothing > 0:
        target_probs = target_probs * (1 - label_smoothing) + label_smoothing / n_classes
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1).mean()


def _lr_scale(epoch: int, warmup_epochs: int, max_epochs: int) -> float:
    """Warmup lineal + cosine decay. Devuelve un factor multiplicativo de `cfg['lr']`."""
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
    return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))


def _save_checkpoint(path: Path, model, optimizer, epoch, best_val_loss, epochs_no_improve,
                      history, finished: bool, best_state=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "epoch": epoch, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss, "epochs_no_improve": epochs_no_improve,
        "history": history, "finished": finished,
        # Pesos de la MEJOR epoca. Sin ellos, al reanudar `best_state` se inicializaba con
        # los pesos de la ultima epoca guardada, y si no habia mejora posterior el modelo
        # final era el de esa epoca y no el seleccionado por la parada temprana.
        "best_state": best_state,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    # Guardado con REINTENTOS. El overlayfs de algunos pods de RunPod falla de forma
    # transitoria al renombrar bajo escritura rápida y repetida (los folds within pueden durar
    # <1 s, así que se escribe un checkpoint de decenas de MB casi seguido) con un
    # FileNotFoundError espurio en el replace. El fallo es momentáneo (un `mv` a mano en el
    # mismo directorio funciona), así que se reintenta con una pausa breve; solo si falla
    # REPETIDAMENTE se propaga (ahí sí es un problema real de disco). Reintentar preserva la
    # invariante "tras esta función el checkpoint existe", de la que depende quien lo relee.
    ultimo = None
    for intento in range(5):
        try:
            torch.save(payload, tmp)
            tmp.replace(path)   # escritura atómica: evita checkpoints truncados si se interrumpe
            return
        except OSError as e:
            ultimo = e
            try:
                tmp.unlink()
            except OSError:
                pass
            time.sleep(0.5 * (intento + 1))
    raise ultimo


def train_run(model: nn.Module, X_train, y_train, X_val, y_val, n_classes: int,
              sfreq: float, cfg: dict, device, ckpt_path: Path, seed: int = 2026):
    """Entrena UNA corrida con el esquema unificado. Reanudable desde `ckpt_path`.

    Si `ckpt_path` ya existe y quedó `finished=True` (corrida completa por
    early stopping o max_epochs), NO reentrena: carga los pesos y devuelve el
    historial guardado (mismo patrón resumible que `build_tensor_cache.py`).

    Devuelve (history, finished): history es una lista de dicts por época
    (epoch, train_loss, val_loss).
    """
    ckpt_path = Path(ckpt_path)
    model = model.to(device)
    # `fused=True` colapsa la actualizacion de TODOS los parametros en UN kernel en vez de uno
    # por tensor. La aritmetica es identica (AdamW es elemento a elemento: no hay reduccion
    # cuyo orden pueda cambiar), asi que los resultados no varian; lo que baja es el numero de
    # lanzamientos, que es nuestro cuello real (en LOSO son ~574 pasos por epoca). Solo existe
    # en CUDA, de ahi el guardia, y el try cubre versiones de torch sin el argumento.
    _fused = device.type == "cuda" and os.environ.get("TESIS_SIN_FUSED") != "1"
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                      weight_decay=cfg["weight_decay"], fused=_fused)
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                      weight_decay=cfg["weight_decay"])
    use_amp = cfg["amp"] and device.type == "cuda"
    # bfloat16 en vez de fp16 para los modelos que lo pidan (EEGMamba): los kernels de Mamba2
    # (selective scan, exponenciales) saturan el rango de fp16 (~6.5e4) y producen NaN pasado
    # el warmup. bf16 tiene el rango de fp32, así que NO hace overflow. Con bf16 no se usa
    # GradScaler (es para el underflow de gradiente de fp16, que bf16 no sufre).
    amp_bf16 = use_amp and (cfg.get("amp_bf16") or os.environ.get("AMP_BF16") == "1")
    amp_dtype = torch.bfloat16 if amp_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not amp_bf16)

    start_epoch, best_val_loss, epochs_no_improve, history = 0, float("inf"), 0, []
    _best_resumido = None

    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        if ck.get("finished"):
            return ck["history"], True
        optimizer.load_state_dict(ck["optimizer_state"])
        start_epoch = ck["epoch"] + 1
        best_val_loss = ck["best_val_loss"]
        epochs_no_improve = ck["epochs_no_improve"]
        history = ck["history"]
        _best_resumido = ck.get("best_state")
        torch.set_rng_state(ck["torch_rng_state"].cpu())
        np.random.set_state(ck["numpy_rng_state"])

    accum_steps = max(1, cfg["batch_size_effective"] // cfg["batch_size_physical"])
    rng = np.random.default_rng(seed)
    n_train = X_train.shape[0]
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if start_epoch > 0 and _best_resumido is not None:
        best_state = {k: v.to(device) for k, v in _best_resumido.items()}
    elif start_epoch > 0:
        warn("checkpoint sin best_state (anterior a 2026-10-31): al reanudar, el modelo "
             "final puede no ser el de la mejor epoca")
    epoch = start_epoch

    # --- PRECARGA A VRAM -----------------------------------------------------------
    # Medido en A40 el 2026-07-27: con la augmentation en CPU (numpy, float64, bucles
    # por trial) un fold LOSO iba a ~1 min/epoca con la GPU al 9 % y un nucleo al 96 %.
    # El cuello era el pipeline de datos, no el modelo. Aqui el train entero se sube una
    # vez a la GPU (LOSO: ~19.000 x 62 x 801 x 4 B = 3.8 GB, de sobra en 45 GB) y cada
    # lote es una seleccion de indices: sin transferencias por lote y sin trabajo de CPU.
    en_gpu = device.type == "cuda" and cfg.get("gpu_pipeline", True)
    if en_gpu:
        libre = torch.cuda.mem_get_info(device)[0]
        necesita = (X_train.nbytes + X_val.nbytes) * 1.5      # margen para las copias
        if necesita > libre:
            # Degradar a CPU en silencio es peligroso cuando se corre en paralelo: el pipeline
            # de CPU va ~9x mas lento (1 min/epoca frente a 6,75 s medidos en LOSO), asi que un
            # solo worker degradado arruina la economia del pod y lo unico que lo delata es este
            # aviso perdido en su log. Con TESIS_EXIGIR_GPU=1 el worker MUERE en vez de
            # arrastrarse: se ve al instante en el monitor y se relanza con menos workers.
            msg = (f"train+val ({necesita/1e9:.1f} GB) no cabe en la VRAM libre "
                   f"({libre/1e9:.1f} GB)")
            if os.environ.get("TESIS_EXIGIR_GPU") == "1":
                raise SystemExit(f"    x {msg}: aborta (TESIS_EXIGIR_GPU=1). "
                                 f"Relanza con menos --workers.")
            print(f"    [aviso] {msg}: se sigue con el pipeline de CPU", flush=True)
            en_gpu = False
    if en_gpu:
        import augment_gpu as aug_gpu
        Xtr_d = torch.as_tensor(X_train, dtype=torch.float32, device=device)
        ytr_d = torch.as_tensor(y_train, dtype=torch.long, device=device)
        g_aug = torch.Generator(device=device).manual_seed(seed)
        # El val TAMBIEN se sube una vez. Antes se copiaba de numpy a GPU en trozos de 128 en
        # CADA epoca (y se reconstruia el one-hot con np.eye cada trozo): con ~2.000 trials de
        # val eso son ~16 transferencias host->device por epoca, puro tiempo muerto. El chequeo
        # de VRAM de arriba ya contaba X_val.nbytes, asi que el espacio estaba previsto.
        Xva_d = torch.as_tensor(X_val, dtype=torch.float32, device=device)
        yva_d = torch.eye(n_classes, dtype=torch.float32, device=device)[
            torch.as_tensor(y_val, dtype=torch.long, device=device)]

    _t_inicio = time.time()
    for epoch in range(start_epoch, cfg["max_epochs"]):
        model.train()
        perm = (torch.randperm(n_train, generator=g_aug, device=device) if en_gpu
                else rng.permutation(n_train))
        for g in optimizer.param_groups:
            g["lr"] = cfg["lr"] * _lr_scale(epoch, cfg["warmup_epochs"], cfg["max_epochs"])

        optimizer.zero_grad()
        # Suma de perdidas ACUMULADA EN GPU. Antes se hacia `train_losses.append(loss.item())`
        # dentro del bucle: cada `.item()` es una sincronizacion GPU->CPU que vacia la cola de
        # kernels, y en LOSO son ~593 pasos por epoca (19.000 trials / lote 32), o sea ~59.000
        # sincronizaciones por fold. Acumulando en un tensor solo se sincroniza UNA vez al
        # cerrar la epoca. La perdida es la misma (media de las perdidas por paso), y de todas
        # formas solo se registra: el control de parada usa val_loss.
        loss_sum = torch.zeros((), device=device)
        n_steps = 0
        for step, start in enumerate(range(0, n_train, cfg["batch_size_physical"])):
            batch_idx = perm[start:start + cfg["batch_size_physical"]]
            # Augmentation POR MINIBATCH, no sobre el train completo: augmentarlo entero
            # reserva varias copias del tensor. Por lote el pico es acotado y ademas cada
            # lote/epoca ve ruido y mascaras distintos -> mas diversidad efectiva.
            if en_gpu:
                xb, yb = aug_gpu.augment_batch(Xtr_d[batch_idx], ytr_d[batch_idx],
                                               n_classes, sfreq, cfg["augment"], g_aug)
            else:
                xb_np, yb_np = aug.augment_batch(X_train[batch_idx], y_train[batch_idx],
                                                 n_classes, sfreq, cfg["augment"], rng)
                xb = torch.as_tensor(xb_np, dtype=torch.float32, device=device)
                yb = torch.as_tensor(yb_np, dtype=torch.float32, device=device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = _forward_logits(model, xb)
                loss = _soft_cross_entropy(logits, yb, cfg["label_smoothing"]) / accum_steps
            scaler.scale(loss).backward()
            loss_sum += loss.detach() * accum_steps
            n_steps += 1
            if (step + 1) % accum_steps == 0:
                # Recorte de gradiente: EEGMamba (SSM) es inestable a lr alto y diverge a
                # NaN sin esto (verificado: lr 1e-3 sin clip -> NaN en el fold LOSO). Es
                # práctica estándar y no perjudica a los modelos que ya funcionaban. Hay que
                # deshacer la escala de AMP antes de medir la norma real del gradiente.
                clip = cfg.get("grad_clip", 0)
                if clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # val_loss POR TROZOS: forward de todo el val de golpe también agota memoria en LOSO
        # (inner_val = 2 sujetos). La media ponderada por tamaño de trozo = media global exacta.
        model.eval()
        with torch.no_grad():
            vloss_sum, vn = torch.zeros((), device=device), 0
            for vs in range(0, X_val.shape[0], 128):
                if en_gpu:
                    xv, yv = Xva_d[vs:vs + 128], yva_d[vs:vs + 128]
                else:
                    xv = torch.as_tensor(X_val[vs:vs + 128], dtype=torch.float32, device=device)
                    yv = torch.as_tensor(np.eye(n_classes, dtype=np.float32)[y_val[vs:vs + 128]],
                                         device=device)
                bl = _soft_cross_entropy(_forward_logits(model, xv), yv, 0.0)
                vloss_sum += bl * xv.shape[0]
                vn += xv.shape[0]
            val_loss = (vloss_sum / vn).item()

        train_loss = (loss_sum / max(n_steps, 1)).item()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        # Progreso por época. Sin esto, un fold LOSO son horas sin una sola línea de
        # salida y no hay forma de saber si avanza, si converge o si está atascado.
        if cfg.get("log_cada", 1) and epoch % cfg.get("log_cada", 1) == 0:
            transcurrido = time.time() - _t_inicio
            por_epoca = transcurrido / (epoch - start_epoch + 1)
            restantes = min(cfg["max_epochs"], epoch + cfg["patience"] + 1) - epoch - 1
            print(f"    ep {epoch:3d}/{cfg['max_epochs']} | train {train_loss:.4f} "
                  f"| val {val_loss:.4f}{' *' if val_loss < best_val_loss - 1e-5 else '  '} "
                  f"| {por_epoca:5.1f} s/ep | sin mejora {epochs_no_improve:2d}/{cfg['patience']} "
                  f"| ~{restantes * por_epoca / 60:.0f} min rest.", flush=True)

        improved = val_loss < best_val_loss - 1e-5
        if improved:
            best_val_loss, epochs_no_improve = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch >= cfg["warmup_epochs"]:
            # La paciencia NO corre durante el warmup: ahí el LR es deliberadamente
            # diminuto (lr * (epoch+1)/warmup) y el modelo aún no puede mejorar, así que
            # contar esas épocas gastaría la paciencia antes de empezar a entrenar de verdad.
            epochs_no_improve += 1

        finished = epochs_no_improve >= cfg["patience"] or epoch == cfg["max_epochs"] - 1
        # Frecuencia de guardado. El checkpoint intermedio solo sirve para REANUDAR tras una
        # caida; lo que hay que conservar es el final. Por eso la frecuencia se decide segun lo
        # que costaria perder el fold, no de forma uniforme:
        #   ckpt_cada = 1  cada epoca. Es lo correcto en LOSO, donde un fold dura 6 a 25 min y
        #                  escribir cuesta ~0,4 s frente a epocas de 7 a 48 s (1 a 6 %).
        #   ckpt_cada = 0  solo el final. Es lo correcto en within, donde una ejecucion entera
        #                  dura de 3 a 22 s: guardar cada epoca llegaba a ser el 83 % del tiempo
        #                  en LaBraM (epocas de 0,52 s frente a 0,4 s de escritura) y lo unico
        #                  que protegia eran segundos de trabajo.
        # El checkpoint final se escribe SIEMPRE, sea cual sea el valor, asi que la capacidad de
        # reutilizar los pesos (segundo dataset, few-shot) no depende de esta palanca.
        cada = int(cfg.get("ckpt_cada", 1))
        if finished or (cada > 0 and epoch % cada == 0):
            _save_checkpoint(ckpt_path, model, optimizer, epoch, best_val_loss,
                             epochs_no_improve, history, finished, best_state)
        if finished:
            break

    model.load_state_dict(best_state)
    _save_checkpoint(ckpt_path, model, optimizer, epoch, best_val_loss, epochs_no_improve,
                      history, True, best_state)
    return history, True


def evaluate(model: nn.Module, X, y, device, n_classes: int, batch_size: int = 128):
    """Métricas sobre un split SIN augmentation (val o test)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            logits = _forward_logits(model, xb)
            preds.append(logits.argmax(dim=-1).cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.asarray(y)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": json.dumps(confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist()),
    }
