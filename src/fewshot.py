"""Escenario intermedio entre LOSO e intra-sujeto: calibración con pocos ensayos.

Propuesta del director (tercer avance): usar entre 10 y 20 ensayos del sujeto no visto
para calibrar la última capa o los estadísticos de normalización, con el resto del
modelo congelado.

Se ejecuta sobre el modelo LOSO ya entrenado de cada fold, justo después de evaluarlo, así
que no añade ningún entrenamiento completo. Para cada `k` (ensayos de calibración) y cada
repetición se sortean `k` ensayos del sujeto de prueba, estratificados por clase, y se
mide la exactitud sobre el RESTO de sus ensayos con cuatro métodos:

    cero      el modelo LOSO tal cual (sin calibrar), medido sobre el mismo resto. Es la
              referencia pareada: sin ella, la comparación mezclaría el efecto de la
              calibración con el de evaluar sobre menos ensayos.
    stats     se sustituyen la media y la desviación por canal del entrenamiento por las
              de los `k` ensayos de calibración. No se toca ningún peso.
    cabeza    se reentrena solo la última capa con los `k` ensayos; el resto, congelado.
    stats+cabeza  las dos cosas.

Los ensayos de calibración nunca se evalúan. Las capas de normalización por lotes y el
dropout permanecen en modo evaluación durante la calibración de la cabeza: con 10 ensayos,
recalcular sus estadísticos sería otra forma de adaptación, no la que se quiere medir.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

import splits as sp
import training as tr

KS = (10, 20)
REPS = 5
EPOCAS_CABEZA = 100
LR_CABEZA = 1e-3


def cabeza(model):
    """Última capa de clasificación de cada arquitectura del estudio."""
    if hasattr(model, "labram"):                      # _LabramWrapper
        return model.labram.final_layer
    if hasattr(model, "classifier"):                  # _EEGMambaWrapper
        return model.classifier
    if hasattr(model, "final_layer"):                 # EEGNet y EEGConformer (braindecode)
        return model.final_layer
    raise ValueError(f"no se identifica la cabeza de {type(model).__name__}")


def _sorteo(y, k, rng):
    """k índices estratificados por clase (k/n_clases de cada una)."""
    clases = np.unique(y)
    por_clase = k // len(clases)
    sel = [rng.choice(np.flatnonzero(y == c), size=por_clase, replace=False) for c in clases]
    return np.sort(np.concatenate(sel))


def _acc(model, X, y, device, n_classes):
    return tr.evaluate(model, X.astype(np.float32), y, device, n_classes)["acc"]


def _calibrar_cabeza(model, X, y, device, n_classes, seed):
    m = copy.deepcopy(model).to(device)
    for p in m.parameters():
        p.requires_grad = False
    params = list(cabeza(m).parameters())
    for p in params:
        p.requires_grad = True
    m.eval()                                          # BN y dropout congelados
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(params, lr=LR_CABEZA, weight_decay=1e-2)
    xb = torch.as_tensor(X, dtype=torch.float32, device=device)
    yb = torch.as_tensor(y, dtype=torch.long, device=device)
    for _ in range(EPOCAS_CABEZA):                    # lote completo: son 10 o 20 ensayos
        opt.zero_grad()
        loss = F.cross_entropy(tr._forward_logits(m, xb).float(), yb)
        loss.backward()
        opt.step()
    return m


def evaluar(model, X_test, y_test, stats, device, n_classes, seed, ks=KS, reps=REPS):
    """Devuelve filas {k, rep, metodo, acc, n_calib, n_eval}.

    X_test llega normalizado con los estadísticos del entrenamiento (`stats`). Para el
    método `stats` hace falta la señal sin normalizar, que se recupera invirtiendo esa
    transformación. Sin `stats` (normalización distinta de `global`) el método no aplica.
    """
    y_test = np.asarray(y_test)
    crudo = None
    if stats is not None:
        crudo = X_test * stats["std"][None] + stats["mean"][None]
    filas = []
    for k in ks:
        for r in range(reps):
            rng = np.random.default_rng(seed * 1000 + k * 10 + r)
            cal = _sorteo(y_test, k, rng)
            resto = np.setdiff1d(np.arange(len(y_test)), cal)
            base = {"k": k, "rep": r, "n_calib": len(cal), "n_eval": len(resto)}
            filas.append({**base, "metodo": "cero",
                          "acc": _acc(model, X_test[resto], y_test[resto], device, n_classes)})
            Xs = None
            if crudo is not None:
                st = sp.fit_normalizer(crudo[cal])
                Xs = sp.apply_normalizer(crudo, st)
                filas.append({**base, "metodo": "stats",
                              "acc": _acc(model, Xs[resto], y_test[resto], device, n_classes)})
            m = _calibrar_cabeza(model, X_test[cal], y_test[cal], device, n_classes, seed + r)
            filas.append({**base, "metodo": "cabeza",
                          "acc": _acc(m, X_test[resto], y_test[resto], device, n_classes)})
            del m
            if Xs is not None:
                m = _calibrar_cabeza(model, Xs[cal], y_test[cal], device, n_classes, seed + r)
                filas.append({**base, "metodo": "stats+cabeza",
                              "acc": _acc(m, Xs[resto], y_test[resto], device, n_classes)})
                del m
    return filas
