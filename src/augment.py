"""Data augmentation EEG (Actividad 3.3, plan §3B).

Se aplica ÚNICAMENTE al split de entrenamiento (inner-train), nunca a
validación ni test — igual disciplina anti-fuga que la normalización
(`src/splits.py`). Todas las funciones operan sobre arrays ya normalizados
(z-score por canal) con forma (n_trials, n_canales, n_tiempo) y son
deterministas dado un `numpy.random.Generator`.
"""
from __future__ import annotations
import numpy as np


def time_shift(X, max_shift_samples: int, rng: np.random.Generator):
    """Desplaza cada trial en el tiempo un nº aleatorio de muestras (relleno por wrap)."""
    if max_shift_samples <= 0:
        return X
    out = np.empty_like(X)
    shifts = rng.integers(-max_shift_samples, max_shift_samples + 1, size=X.shape[0])
    for i, s in enumerate(shifts):
        out[i] = np.roll(X[i], s, axis=-1)
    return out


def gaussian_noise(X, std_frac: float, rng: np.random.Generator):
    """Añade ruido gaussiano proporcional al std de CADA canal (no un valor absoluto)."""
    if std_frac <= 0:
        return X
    ch_std = X.std(axis=(0, 2), keepdims=True)  # (1, n_ch, 1)
    noise = rng.normal(0.0, 1.0, size=X.shape).astype(X.dtype) * (ch_std * std_frac)
    return X + noise


def freq_mask(X, sfreq: float, max_width_hz: float, rng: np.random.Generator,
              chunk: int = 1024):
    """Enmascara (pone a cero) una banda de frecuencia aleatoria por trial, vía FFT real.

    La FFT se hace POR TROZOS de `chunk` trials. `np.fft.rfft` calcula siempre en doble
    precisión (complex128), así que hacerla sobre todo el train de una vez reserva una
    matriz enorme: en LOSO son ~20.400 trials -> 7.5 GB y revienta la RAM. Trocear acota
    el pico de memoria (~chunk x n_ch x n_freq) sin cambiar el resultado. En within-subject
    (144 trials) entra en un solo trozo, así que no hay cambio de comportamiento allí.
    """
    if max_width_hz <= 0:
        return X
    n_t = X.shape[-1]
    freqs = np.fft.rfftfreq(n_t, d=1.0 / sfreq)
    out = np.empty_like(X)
    for start in range(0, X.shape[0], chunk):
        sl = slice(start, start + chunk)
        Xf = np.fft.rfft(X[sl], axis=-1)
        for i in range(Xf.shape[0]):
            width = rng.uniform(0.0, max_width_hz)
            f0 = rng.uniform(0.0, max(freqs[-1] - width, 0.0))
            mask = ~((freqs >= f0) & (freqs <= f0 + width))
            Xf[i] *= mask[None, :]
        out[sl] = np.fft.irfft(Xf, n=n_t, axis=-1).astype(X.dtype)
    return out


def mixup(X, y, n_classes: int, alpha: float, rng: np.random.Generator):
    """Mixup (Zhang et al. 2018). Devuelve (X_mezclado, y_onehot_mezclado).

    y: etiquetas enteras (n_trials,). Se convierten a one-hot antes de mezclar
    porque mixup interpola también las etiquetas, no solo las señales.
    """
    n = X.shape[0]
    y_onehot = np.eye(n_classes, dtype=np.float32)[y]
    if alpha <= 0:
        return X, y_onehot
    lam = rng.beta(alpha, alpha, size=n).astype(np.float32)
    perm = rng.permutation(n)
    lam_x = lam.reshape(-1, *([1] * (X.ndim - 1)))
    lam_y = lam.reshape(-1, 1)
    X_mix = lam_x * X + (1 - lam_x) * X[perm]
    y_mix = lam_y * y_onehot + (1 - lam_y) * y_onehot[perm]
    return X_mix, y_mix


def augment_batch(X, y, n_classes: int, sfreq: float, cfg: dict, rng: np.random.Generator):
    """Aplica el subconjunto de augmentations activado en `cfg`.

    cfg reconocidas (todas opcionales, default = desactivada):
      time_shift_frac      : fracción de n_tiempo -> max_shift_samples
      gaussian_noise_std    : std_frac para gaussian_noise
      freq_mask_hz          : max_width_hz para freq_mask
      mixup_alpha           : alpha para mixup (0 = desactivado)

    Devuelve (X_aug, y_onehot) SIEMPRE en one-hot (n_classes), para que el
    train loop maneje un único formato de etiqueta con o sin mixup.
    """
    n_t = X.shape[-1]
    # Las aumentaciones de la avance 7 (octubre de 2026) solo estan en augment_gpu.py.
    # Ignorarlas aqui en silencio produciria una "variante" identica a la base.
    for k in ("channel_reflection_p", "fft_noise_std"):
        if cfg.get(k, 0):
            raise NotImplementedError(f"{k} solo existe en el pipeline de GPU (augment_gpu.py)")
    if cfg.get("time_shift_frac", 0):
        X = time_shift(X, int(round(cfg["time_shift_frac"] * n_t)), rng)
    if cfg.get("gaussian_noise_std", 0):
        X = gaussian_noise(X, cfg["gaussian_noise_std"], rng)
    if cfg.get("freq_mask_hz", 0):
        X = freq_mask(X, sfreq, cfg["freq_mask_hz"], rng)
    alpha = cfg.get("mixup_alpha", 0.0)
    X, y_onehot = mixup(X, y, n_classes, alpha, rng)
    return X, y_onehot
