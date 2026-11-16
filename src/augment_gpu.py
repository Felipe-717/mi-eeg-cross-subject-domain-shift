"""Data augmentation EEG en GPU. Equivalente funcional de `augment.py`, sin bucles.

POR QUE EXISTE (medido el 2026-07-27 en una A40)
    Con la version de numpy un fold LOSO iba a ~1 min/epoca con la **GPU al 9%** y un
    nucleo de CPU al 96%. El cuello no era el modelo (EEGNet son 2.898 parametros) sino
    la augmentation: se hacia en CPU, monohilo, con `np.fft` en float64, y ademas con
    bucles de Python por trial en `time_shift` y `freq_mask`. Con ~600 lotes por epoca
    salian ~120.000 llamadas por fold.

DIFERENCIAS CON `augment.py` (deliberadas, afectan a los numeros)
    * Se opera en float32 sobre GPU; la FFT de numpy calcula en float64.
    * Todo esta VECTORIZADO: no hay bucles por trial.
    * El generador aleatorio es `torch.Generator`, no `numpy.random.Generator`.
    Los resultados NO son bit a bit iguales a los de `augment.py`. Es aceptable porque
    los resultados de la Fase 2 estan congelados y no se re-ejecutan; lo que exige la
    comparacion de la Fase 3 es que las TRES arquitecturas reciban el mismo trato, no
    que coincida con una implementacion anterior.

Mismas garantias que la version de CPU: se aplica SOLO al split de entrenamiento, nunca
a validacion ni test, y es determinista dada la semilla del generador.
"""
from __future__ import annotations

import torch


def time_shift(X: torch.Tensor, max_shift: int, g: torch.Generator) -> torch.Tensor:
    """Desplazamiento circular por trial, distinto para cada uno.

    `torch.roll` aplica el MISMO desplazamiento a todo el tensor, asi que se construye
    un indice desplazado por trial y se hace un gather. Sin bucles.
    """
    if max_shift <= 0:
        return X
    n, _, t = X.shape
    s = torch.randint(-max_shift, max_shift + 1, (n,), generator=g, device=X.device)
    base = torch.arange(t, device=X.device).unsqueeze(0)          # (1, t)
    idx = (base - s.unsqueeze(1)) % t                             # (n, t)
    return torch.gather(X, 2, idx.unsqueeze(1).expand(-1, X.shape[1], -1))


def gaussian_noise(X: torch.Tensor, std_frac: float, g: torch.Generator) -> torch.Tensor:
    """Ruido gaussiano proporcional al std de CADA canal (no un valor absoluto)."""
    if std_frac <= 0:
        return X
    ch_std = X.std(dim=(0, 2), keepdim=True)                      # (1, n_ch, 1)
    ruido = torch.randn(X.shape, generator=g, device=X.device, dtype=X.dtype)
    return X + ruido * (ch_std * std_frac)


def freq_mask(X: torch.Tensor, sfreq: float, max_width_hz: float,
              g: torch.Generator) -> torch.Tensor:
    """Pone a cero una banda de frecuencia aleatoria por trial, via FFT real.

    Vectorizado: la mascara de los n trials se construye de una vez por difusion, en vez
    del bucle `for i in range(...)` de la version de numpy.
    """
    if max_width_hz <= 0:
        return X
    n, _, t = X.shape
    Xf = torch.fft.rfft(X, dim=-1)                                # complex64 desde float32
    freqs = torch.fft.rfftfreq(t, d=1.0 / sfreq, device=X.device)  # (n_freq,)
    ancho = torch.rand(n, 1, generator=g, device=X.device) * max_width_hz
    tope = torch.clamp(freqs[-1] - ancho, min=0.0)
    f0 = torch.rand(n, 1, generator=g, device=X.device) * tope
    fuera = ~((freqs >= f0) & (freqs <= f0 + ancho))               # (n, n_freq)
    Xf = Xf * fuera.unsqueeze(1)
    return torch.fft.irfft(Xf, n=t, dim=-1).to(X.dtype)


def mixup(X: torch.Tensor, y_onehot: torch.Tensor, alpha: float,
          g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Mixup (Zhang et al. 2018): interpola señales y etiquetas con el mismo lambda."""
    if alpha <= 0:
        return X, y_onehot
    n = X.shape[0]
    # torch no trae Beta con generator; Beta(a,a) se obtiene de dos Gamma. Para el caso
    # habitual alpha=1 (Beta(1,1) = Uniforme(0,1)) basta con rand, que si acepta generator.
    if abs(alpha - 1.0) < 1e-9:
        lam = torch.rand(n, generator=g, device=X.device)
    else:
        d = torch.distributions.Beta(alpha, alpha)
        lam = d.sample((n,)).to(X.device)
    perm = torch.randperm(n, generator=g, device=X.device)
    lx = lam.view(-1, 1, 1)
    ly = lam.view(-1, 1)
    return lx * X + (1 - lx) * X[perm], ly * y_onehot + (1 - ly) * y_onehot[perm]


def channel_reflection(X: torch.Tensor, y_idx: torch.Tensor, p: float, perm, perm_clases,
                       g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Reflexion de canales (Wang et al., 2024): intercambia los hemisferios con prob. p.

    Cada segmento elegido pasa sus canales izquierdos a la posicion de los derechos y al
    reves, y su etiqueta se intercambia (mano izquierda <-> mano derecha), porque la
    desincronizacion de la imaginacion motora es contralateral. `perm` y `perm_clases`
    vienen de `variantes.pares_reflexion` y `variantes.permutacion_clases`.
    """
    if p <= 0:
        return X, y_idx
    elegir = torch.rand(X.shape[0], generator=g, device=X.device) < p
    pc = torch.as_tensor(perm, device=X.device, dtype=torch.long)
    pk = torch.as_tensor(perm_clases, device=X.device, dtype=torch.long)
    X = torch.where(elegir.view(-1, 1, 1), X[:, pc, :], X)
    y_idx = torch.where(elegir, pk[y_idx], y_idx)
    return X, y_idx


def fft_noise(X: torch.Tensor, std_frac: float, g: torch.Generator) -> torch.Tensor:
    """Ruido gaussiano multiplicativo sobre la AMPLITUD de la transformada de Fourier.

    Se conserva la fase y se perturba el modulo de cada componente,
        |F'(f)| = |F(f)| * max(0, 1 + std_frac * N(0, 1)),
    y se vuelve al tiempo con la transformada inversa. Perturba el espectro sin desplazar
    los eventos en el tiempo.
    """
    if std_frac <= 0:
        return X
    t = X.shape[-1]
    Xf = torch.fft.rfft(X, dim=-1)
    ganancia = torch.clamp(1.0 + std_frac * torch.randn(Xf.shape, generator=g,
                                                         device=X.device), min=0.0)
    return torch.fft.irfft(Xf * ganancia, n=t, dim=-1).to(X.dtype)


def augment_batch(X: torch.Tensor, y_idx: torch.Tensor, n_classes: int, sfreq: float,
                  cfg: dict, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Aplica las augmentations activadas en `cfg`. Devuelve (X_aug, y_onehot).

    Mismas claves de configuracion que `augment.augment_batch`, para que cambiar entre
    una implementacion y otra no exija tocar la configuracion del entrenamiento.
    Las claves `channel_reflection_p` y `fft_noise_std` solo existen aqui (ejecucion de
    octubre de 2026); valen 0 en la configuracion base, que por tanto no cambia.
    """
    t = X.shape[-1]
    if cfg.get("channel_reflection_p", 0):
        X, y_idx = channel_reflection(X, y_idx, cfg["channel_reflection_p"],
                                      cfg["cr_perm"], cfg["cr_perm_clases"], g)
    if cfg.get("fft_noise_std", 0):
        X = fft_noise(X, cfg["fft_noise_std"], g)
    if cfg.get("time_shift_frac", 0):
        X = time_shift(X, int(round(cfg["time_shift_frac"] * t)), g)
    if cfg.get("gaussian_noise_std", 0):
        X = gaussian_noise(X, cfg["gaussian_noise_std"], g)
    if cfg.get("freq_mask_hz", 0):
        X = freq_mask(X, sfreq, cfg["freq_mask_hz"], g)
    y_onehot = torch.nn.functional.one_hot(y_idx, n_classes).to(X.dtype)
    return mixup(X, y_onehot, cfg.get("mixup_alpha", 0.0), g)
