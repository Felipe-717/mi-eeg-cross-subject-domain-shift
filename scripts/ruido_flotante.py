#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ruido_flotante.py: Wilcoxon robusto al ruido de coma flotante (funcion comun a todos los scripts).

POR QUE EXISTE
    Las exactitudes por sujeto son multiplos de 1/n ensayos. Muchas diferencias pareadas son
    por tanto IGUALES en aritmetica exacta, pero al restar medias calculadas con otro orden de
    suma difieren en ~1e-16. `scipy.stats.wilcoxon` rompe esos empates por el ruido (rangos
    distintos a empates reales) y el p cambia hasta 0.1 segun como se promedio antes. Es el mismo
    fenomeno que el recuento <= 0.70 de verificar_cifras.le_umbral.

DECISION (supervisor del avance 7, 2026-11)
    Las diferencias se redondean a DECIMALES_EMPATE = 9 antes de cualquier prueba de rangos
    (Wilcoxon, TOST): 1e-9 esta cuatro ordenes por encima del ruido (1e-16) y ocho por debajo
    de cualquier diferencia real (la menor es 1/(n ensayos) ~ 1e-3). Los empates reales pasan a
    ser empates.

USO
    from ruido_flotante import wilcoxon_pareado, redondear
    res = wilcoxon_pareado(a, b, zero_method="zsplit", alternative="two-sided")
    res = wilcoxon_pareado(d)            # d ya es la diferencia
"""
from __future__ import annotations

import numpy as np
from scipy import stats

DECIMALES_EMPATE = 9


def redondear(d):
    """Diferencias redondeadas a 1e-9 (elimina -0.0 para que los ceros sean ceros)."""
    return np.round(np.asarray(d, float), DECIMALES_EMPATE) + 0.0


def wilcoxon_pareado(x, y=None, **kw):
    """stats.wilcoxon sobre la diferencia x - y (o x si y es None) redondeada a 1e-9."""
    x = np.asarray(x, float)
    d = x if y is None else x - np.asarray(y, float)
    return stats.wilcoxon(redondear(d), **kw)
