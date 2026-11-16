#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""avance7_stats.py: utilidades estadisticas comunes del analisis del avance 7.

Mismas definiciones que scripts/estadistica.py (Wilcoxon bilateral con zero_method="zsplit",
r = z/sqrt(n) con z reconstruido desde el p bilateral) y scripts/bootstrap_ic.py (percentil
al 95 %, B = 10 000, semilla 2026). Anade Holm y TOST con margen 0.03 (protocolo, reglas 7 y 8).
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from ruido_flotante import wilcoxon_pareado  # Wilcoxon sobre diferencias redondeadas a 1e-9

B = 10_000
SEMILLA_BOOT = 2026
MARGEN = 0.03


def wilcoxon_r(d):
    """Wilcoxon de rangos con signo bilateral sobre la diferencia pareada `d`."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2 or np.allclose(d, 0):
        return dict(n=n, p=np.nan, r=np.nan)
    p = wilcoxon_pareado(d, zero_method="zsplit", alternative="two-sided").pvalue
    z = abs(stats.norm.ppf(p / 2)) if p > 0 else abs(stats.norm.ppf(1e-16))
    return dict(n=n, p=float(p), r=float(z / np.sqrt(n)))


def boot_ic(x, b=B, seed=SEMILLA_BOOT):
    """IC 95 % percentil de la media de `x` remuestreando sujetos (n,)."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(b, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def holm(p):
    """Ajuste de Holm (step-down) sobre un vector de p; devuelve p ajustados."""
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    orden = np.argsort(pv)
    aj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(orden):
        run = max(run, (m - rank) * pv[i])
        aj[i] = min(1.0, run)
    out[ok] = aj
    return out


def tost(d, margen=MARGEN):
    """TOST con Wilcoxon unilateral: equivalencia si la diferencia esta en (-margen, margen).
    p = max de los dos unilaterales."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    p1 = wilcoxon_pareado(d + margen, zero_method="zsplit", alternative="greater").pvalue
    p2 = wilcoxon_pareado(d - margen, zero_method="zsplit", alternative="less").pvalue
    return float(max(p1, p2))


def magnitud(r):
    return "grande" if r >= 0.5 else "mediano" if r >= 0.3 else "pequeno" if r >= 0.1 else "trivial"
