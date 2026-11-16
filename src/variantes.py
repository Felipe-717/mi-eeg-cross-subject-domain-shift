"""Variantes del preprocesamiento y de la aumentación para la campaña de octubre de 2026.

Todo lo que aquí se define es una VARIANTE de la línea base: nada de este módulo se
ejecuta en el entrenamiento base, cuyos resultados ya están publicados y no se tocan.

Dos tipos de variante, con dos mecanismos distintos:

1. Variantes del DATO (alineamiento de covarianzas, subbandas). Se materializan como un
   caché derivado en un directorio aparte, con la misma estructura que el original. El
   resto del código no cambia: basta con pasar `--cache <derivado>` a `train_baseline.py`.
   Como el nombre del directorio entra en el nombre del CSV y en la ruta del checkpoint,
   los resultados de una variante no pueden colisionar con los de la base.

2. Variantes de la AUMENTACIÓN (reflexión de canales, ruido en el dominio de Fourier).
   Se activan con `--aug <preset>` y se aplican por minilote en GPU (`augment_gpu.py`).

Alineamiento euclidiano (He y Wu, 2019)
    Para cada sujeto se calcula la covarianza media de sus segmentos,
        R = (1/n) sum_i X_i X_i^T / T,
    y se transforma cada segmento como X_i <- R^{-1/2} X_i. Tras ello la covarianza
    media de cada sujeto es la identidad, de modo que todos quedan referidos a un mismo
    punto. No usa etiquetas. Sí usa los segmentos sin etiquetar del sujeto de prueba, lo
    que lo hace transductivo, igual que la normalización `per_subject`: hay que declararlo.

    RANGO. Los datos llevan referencia promedio común y se les retiraron componentes de
    ICA, así que R es singular (rango C - 1 - n_componentes_retiradas). Invertir sus
    autovalores nulos amplificaría ruido numérico en un factor de 1e5 o más. Se usa la
    pseudoinversa de la raíz: solo se invierten los autovalores mayores que
    `TOL_RANGO * max(autovalores)`, y las direcciones nulas se anulan. El rango usado se
    guarda en el caché para poder informarlo.

    Se calcula sobre la VENTANA DE MODELADO (0 a 4 s), que es la señal que ven los
    modelos, y se aplica como filtro espacial al segmento completo del caché.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

TOL_RANGO = 1e-6

# Pares de clases que se intercambian al reflejar los hemisferios. Las clases que no
# figuran (p. ej. pies) se reflejan sobre sí mismas.
CLASES_REFLEJADAS = {"left_hand": "right_hand", "right_hand": "left_hand"}


# --------------------------------------------------------------------------- #
# Reflexión de canales (Wang et al., 2024)
# --------------------------------------------------------------------------- #

_RE_CANAL = re.compile(r"^([A-Za-z]+?)(\d+)([A-Za-z]*)$")


def pares_reflexion(ch_names):
    """Permutación que intercambia cada canal con su simétrico del otro hemisferio.

    Regla del sistema 10-20 y sus extensiones: impar a la izquierda, par a la derecha, y
    el simétrico de `n` es `n + 1` (C3-C4, FTT9h-FTT10h). Los canales de la línea media
    (sufijo z) y los que no tienen pareja en el montaje se quedan en su sitio.

    Devuelve (perm, sin_pareja): `perm[i]` es el índice del canal que pasa a ocupar la
    posición `i` tras la reflexión.
    """
    nombres = [str(c) for c in ch_names]
    idx = {c.upper(): k for k, c in enumerate(nombres)}
    perm = list(range(len(nombres)))
    sin_pareja = []
    for k, c in enumerate(nombres):
        m = _RE_CANAL.match(c)
        if not m:
            continue                                    # línea media: Fz, Cz, Pz, Oz...
        pre, num, suf = m.group(1), int(m.group(2)), m.group(3)
        par = num + 1 if num % 2 else num - 1
        j = idx.get(f"{pre}{par}{suf}".upper())
        if j is None:
            sin_pareja.append(c)
        else:
            perm[k] = j
    assert sorted(perm) == list(range(len(nombres))), "la reflexión no es una permutación"
    return perm, sin_pareja


def permutacion_clases(class_order):
    """Índice de la clase reflejada de cada clase, en el orden de `class_order`."""
    orden = [str(c) for c in class_order]
    return [orden.index(CLASES_REFLEJADAS.get(c, c)) for c in orden]


# Presets de aumentación. "base" es la configuración publicada y NO se modifica aquí.
# Los valores de los métodos nuevos están en la tabla del seguimiento y se justifican
# en el documento.
PRESETS_AUG = {
    "base": {},                                          # sin cambios sobre DEFAULT_TRAIN_CFG
    "none": {"time_shift_frac": 0.0, "gaussian_noise_std": 0.0, "freq_mask_hz": 0.0,
             "mixup_alpha": 0.0},
    "cr": {"channel_reflection_p": 0.5},                 # base + reflexión de canales
    "fft": {"fft_noise_std": 0.1},                       # base + ruido en amplitud de Fourier
}


# --------------------------------------------------------------------------- #
# Cachés derivados
# --------------------------------------------------------------------------- #

def _ventana_idx(tmin, sfreq, ventana, n_t):
    a, b = ventana
    i0 = int(round((a - tmin) * sfreq))
    i1 = int(round((b - tmin) * sfreq)) + 1
    assert 0 <= i0 < i1 <= n_t, f"ventana {ventana} fuera del segmento cacheado"
    return i0, i1


def matriz_ea(X_ventana, tol=TOL_RANGO):
    """R^{-1/2} (pseudoinversa) de la covarianza media. X_ventana: (n, C, T)."""
    X = np.asarray(X_ventana, dtype=np.float64)
    R = np.einsum("nct,ndt->cd", X, X) / (X.shape[0] * X.shape[2])
    R = (R + R.T) / 2
    w, U = np.linalg.eigh(R)
    keep = w > tol * w.max()
    inv = np.zeros_like(w)
    inv[keep] = 1.0 / np.sqrt(w[keep])
    return (U * inv) @ U.T, int(keep.sum())


def filtro_banda(X, sfreq, lo, hi, orden=4):
    """Butterworth de fase cero (sosfiltfilt), con relleno par de 1 s."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(orden, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
    pad = min(int(sfreq), X.shape[-1] - 1)
    return sosfiltfilt(sos, X, axis=-1, padtype="even", padlen=pad).astype(X.dtype)


def derivar_cache(origen, destino, tipo, ventana=(0.0, 4.0), canales=None,
                  banda=None, forzar=False):
    """Escribe un caché derivado, sujeto a sujeto. Idempotente salvo con `forzar`.

    tipo     "ea" (alineamiento euclidiano) o "banda" (filtro paso banda).
    canales  vista de canales sobre la que se calcula el alineamiento. El alineamiento
             depende de qué canales entran en la covarianza, así que un caché alineado
             con 62 canales NO sirve para un modelo de 45: se materializa solo la vista.
    """
    origen, destino = Path(origen), Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    resumen = []
    for f in sorted(origen.glob("sub-*_full.npz")):
        out = destino / f.name
        if out.exists() and not forzar:
            continue
        d = np.load(f, allow_pickle=True)
        datos = {k: d[k] for k in d.files}
        X = datos["X"]
        chs = [str(c) for c in datos["ch_names"]]
        if canales is not None:
            up = {c.upper(): k for k, c in enumerate(chs)}
            faltan = [c for c in canales if c.upper() not in up]
            if faltan:
                raise SystemExit(f"{f.name}: faltan canales de la vista {faltan}")
            sel = [up[c.upper()] for c in canales]
            X = X[:, sel, :]
            datos["ch_names"] = np.array([chs[k] for k in sel])
        sfreq, tmin = float(datos["sfreq"]), float(datos["tmin"])
        info = {"tipo": tipo, "origen": str(origen), "n_canales": int(X.shape[1])}
        if tipo == "ea":
            i0, i1 = _ventana_idx(tmin, sfreq, ventana, X.shape[2])
            W, rango = matriz_ea(X[:, :, i0:i1])
            X = np.einsum("cd,ndt->nct", W, X.astype(np.float64)).astype(np.float32)
            info.update(rango=rango, ventana=list(ventana), tol=TOL_RANGO)
        elif tipo == "banda":
            lo, hi = banda
            X = filtro_banda(X.astype(np.float32), sfreq, lo, hi)
            info.update(banda=[lo, hi], orden=4, filtro="butter_sosfiltfilt")
        else:
            raise ValueError(tipo)
        datos["X"] = X
        datos["variante"] = np.array(json.dumps(info))
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, **datos)
        tmp.replace(out)
        resumen.append({"sub": f.name.replace("_full.npz", ""), **info})
    if resumen:
        ruta = destino / "variante.json"
        previo = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else []
        hechos = {r["sub"] for r in resumen}
        todo = [r for r in previo if r["sub"] not in hechos] + resumen
        ruta.write_text(json.dumps(sorted(todo, key=lambda r: r["sub"]), indent=1,
                                   ensure_ascii=False), encoding="utf-8")
    return resumen
