"""EEGMamba (Wang et al., 2024) vendorizado para la Fase 3 de la tesis.

Fuente: repositorio oficial `wjq-learning/EEGMamba` (licencia MIT, (c) 2025 Jiquan Wang),
archivos `models/eegmamba.py` y `modules/mixer_seq_simple.py`. Se copia aquí porque
EEGMamba NO está en `braindecode` (a diferencia de EEGNet/Conformer/LaBraM): es la
UNICA arquitectura de la Fase 3 cuyo modelo hay que traer del repo de sus autores.

Qué se conserva y qué NO respecto al repo original:
  * Se conserva el backbone tal cual: PatchEmbedding (rama temporal por convolución +
    rama espectral por |rFFT| + codificación posicional) y el MixerModel de bloques
    Mamba2 BIDIRECCIONALES (el `flip(1)` por capa es lo que lo hace bidireccional).
  * Se DESCARTA la Mixture-of-Experts y los task-tokens del paper: la versión pública
    del repo ya es la simplificada de una sola tarea (sin MoE), justo lo que queremos
    para entrenar DESDE CERO sobre OpenBMI en igualdad con las otras arquitecturas.
  * La cabeza de clasificación NO vive aquí: la monta `training.build_model` según el
    nº de canales/parches reales del caché (ver `_EEGMambaWrapper`).

Requisitos de ejecución (solo Linux + CUDA, por eso corre en el pod, no en local):
`mamba-ssm` (con Mamba2) + `causal-conv1d` + los kernels Triton de RMSNorm. La
configuración fija `patch_size = 200` y `d_model = 200` (la rama espectral usa
`Linear(101, d_model)` con 101 = 200//2 + 1 de la rFFT), así que la ventana debe
alimentarse a 200 Hz en parches de 1 s, que es lo que produce `adapters.adapt_eegmamba`.
"""
from __future__ import annotations

import copy
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:                       # sin CUDA/Triton no hay kernels: falla explícito al usar
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# --------------------------------------------------------------------------- #
# Bloque Mamba/atención (verbatim de modules/mixer_seq_simple.py, sin lo no usado)
# --------------------------------------------------------------------------- #
def create_block(d_model, d_intermediate, ssm_cfg=None, attn_layer_idx=None,
                 attn_cfg=None, norm_epsilon=1e-5, rms_norm=False,
                 residual_in_fp32=False, fused_add_norm=False, layer_idx=None,
                 device=None, dtype=None):
    ssm_cfg = ssm_cfg or {}
    attn_layer_idx = attn_layer_idx or []
    attn_cfg = attn_cfg or {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        ssm_cfg = copy.deepcopy(ssm_cfg)
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ("Mamba1", "Mamba2"):
            raise ValueError(f"ssm_layer inválido: {ssm_layer} (solo Mamba1 o Mamba2)")
        mixer_cls = partial(Mamba2 if ssm_layer == "Mamba2" else Mamba,
                            layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm,
                       eps=norm_epsilon, **factory_kwargs)
    mlp_cls = nn.Identity if d_intermediate == 0 else partial(
        GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs)
    block = Block(d_model, mixer_cls, mlp_cls, norm_cls=norm_cls,
                  fused_add_norm=fused_add_norm, residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    """Pila de bloques Mamba2 recorrida en AMBOS sentidos (bidireccional por `flip`)."""

    def __init__(self, d_model, n_layer, d_intermediate, ssm_cfg=None,
                 attn_layer_idx=None, attn_cfg=None, norm_epsilon=1e-5,
                 rms_norm=False, fused_add_norm=False, residual_in_fp32=False,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm and (layer_norm_fn is None or rms_norm_fn is None):
            raise ImportError("Faltan los kernels Triton de LayerNorm/RMSNorm (mamba-ssm).")
        self.layers = nn.ModuleList([
            create_block(d_model, d_intermediate=d_intermediate, ssm_cfg=ssm_cfg,
                         attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg,
                         norm_epsilon=norm_epsilon, rms_norm=rms_norm,
                         residual_in_fp32=residual_in_fp32, fused_add_norm=fused_add_norm,
                         layer_idx=i, **factory_kwargs)
            for i in range(n_layer)])
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None, **mixer_kwargs):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual,
                                            inference_params=inference_params, **mixer_kwargs)
            hidden_states = hidden_states.flip(1)        # <- bidireccionalidad
            residual = residual.flip(1)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            hidden_states = layer_norm_fn(
                hidden_states, self.norm_f.weight, self.norm_f.bias, eps=self.norm_f.eps,
                residual=residual, prenorm=False, residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm))
        return hidden_states


# --------------------------------------------------------------------------- #
# EEGMamba (verbatim de models/eegmamba.py)
# --------------------------------------------------------------------------- #
class PatchEmbedding(nn.Module):
    """Cada parche de 1 s -> embedding de d_model, combinando tiempo y espectro."""

    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(7, 7), stride=(1, 1),
                      padding=(3, 3), groups=d_model, bias=False))
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        self.proj_in = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24), bias=False),
            nn.GroupNorm(5, 25), nn.GELU())
        self.spectral_proj = nn.Sequential(nn.Linear(101, d_model, bias=False), nn.Dropout(0.1))

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        mask_x = x if mask is None else x.clone()
        if mask is not None:
            mask_x[mask == 1] = self.mask_encoding
        mask_x = rearrange(mask_x, 'b c l d -> b d c l')
        time_x = rearrange(mask_x, 'b d c l -> b (c l) d').unsqueeze(1)
        time_emb = self.proj_in(time_x)
        time_emb = time_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)
        freq_x = rearrange(mask_x, 'b d c l -> b c l d')
        spectral = torch.abs(torch.fft.rfft(freq_x, dim=-1, norm='forward'))
        patch_emb = time_emb + self.spectral_proj(spectral)
        pos = self.positional_encoding(patch_emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return patch_emb + pos


class EEGMamba(nn.Module):
    """Backbone: PatchEmbedding + MixerModel bidireccional (Mamba2). Sin cabeza."""

    def __init__(self, in_dim=200, out_dim=200, d_model=200, seq_len=30, n_layer=12):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)
        self.encoder = MixerModel(
            d_model=d_model, n_layer=n_layer, d_intermediate=0,
            ssm_cfg={"layer": "Mamba2", "headdim": 50, "d_state": 64},
            attn_layer_idx=[], attn_cfg={}, rms_norm=True,
            fused_add_norm=True, residual_in_fp32=True)
        self.proj_out = nn.Sequential(nn.Linear(d_model, out_dim))
        self.apply(_weights_init)

    def forward(self, x, mask=None):
        bz, ch_num, seq_len, patch_size = x.shape
        h = self.patch_embedding(x, mask=mask)
        h = rearrange(h, 'b c l d -> b (c l) d')
        h = self.encoder(h)
        h = rearrange(h, 'b (c l) d -> b c l d', l=seq_len)
        return self.proj_out(h)


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
