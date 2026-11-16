#!/usr/bin/env python
"""Norma del gradiente previa al recorte, por paso de optimizacion (avance 7).

Pregunta: con la configuracion del avance 7 de cada arquitectura (CFG de
`pod/avance_7/avance_7.py`), que fraccion de los pasos de optimizacion supera la norma 1.0
y por tanto es recortada por `clip_grad_norm_(..., grad_clip=1.0)` de `src/training.py`?

Como funciona (no modifica `src/training.py`): sustituye `torch.nn.utils.clip_grad_norm_`
por un envoltorio que llama a la funcion original con los mismos argumentos y guarda su
valor de retorno, que es la norma TOTAL del gradiente ANTES de recortar (documentacion de
PyTorch). El entrenamiento es el mismo que en el avance 7: mismo modelo, mismo
`DEFAULT_TRAIN_CFG`, mismo lr y lote, misma aumentacion, mismo calendario de lr. Despues de
`--epocas` epocas el envoltorio lanza una excepcion y se corta la ejecucion (el calendario
de lr sigue siendo el de 200 epocas con calentamiento de 10: las primeras 10 epocas estan
en calentamiento, con lr reducido; por eso se recomienda `--epocas 15` o mas y se informa
la fraccion por epoca).

Requisito: cache OpenBMI con ICA (`epochs_cache_ica`), que NO esta en la maquina local
(solo existe el de WBCIC-SHU, otro conjunto de datos). Sin cache el script se detiene con
un mensaje; en el pod:

    python scripts/avance7_norma_gradiente.py --cache /root/ws/cache/epochs_cache_ica \\
        --arqs eegnet conformer labram eegmamba --subs sub-01 sub-02 --epocas 15

Comprobacion de la instrumentacion sin datos reales (EEGNet, datos sinteticos, ~1 min):

    ~/anaconda3/envs/eeg/python.exe scripts/avance7_norma_gradiente.py --selftest

Salidas (CSV, `results/avance_7/analisis/`):
    norma_gradiente_pasos.csv    una fila por paso: arq, sub, protocolo, fold, epoca, paso, norma
    norma_gradiente_resumen.csv  por arq y epoca: n, fraccion > 1.0, mediana, p90, p99, max,
                                 y n de pasos no finitos (NaN/inf)
En --selftest los nombres llevan el sufijo `_selftest` y NO son resultados de la tesis.

LaBraM y EEGMamba: LaBraM Base (~5.8 M parametros) probablemente cabe con lote 32 en 4 GB;
EEGMamba requiere `mamba_ssm` (solo Linux con CUDA), que no esta instalado localmente, asi
que en la maquina local solo se podrian medir EEGNet y Conformer.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402

# Misma configuracion que pod/avance_7/avance_7.py (CFG). Se duplica a proposito: ese
# archivo importa cosas solo-pod; aqui se verifica con un assert opcional (ver --verificar).
CFG = {
    "eegnet":    (2.8e-3, 256, {}),
    "conformer": (1.0e-3, 256, {}),
    "labram":    (1.0e-4, 32, {"LABRAM_FINETUNE": "full", "LABRAM_UV_SCALE": "1"}),
    "eegmamba":  (1.0e-3, 32, {"EEGMAMBA_HEAD": "flatten", "EEGMAMBA_UV_SCALE": "1",
                               "AMP_BF16": "1"}),
}
OUT = REPO / "results" / "avance_7" / "analisis"


class _Fin(Exception):
    pass


class Registro:
    """Envoltorio de clip_grad_norm_ que guarda la norma previa al recorte."""

    def __init__(self, max_pasos: int, pasos_por_epoca: int):
        self.normas: list[torch.Tensor] = []
        self.max_pasos = max_pasos
        self.ppe = pasos_por_epoca
        self.orig = torch.nn.utils.clip_grad_norm_

    def __call__(self, params, max_norm, *a, **k):
        total = self.orig(params, max_norm, *a, **k)      # devuelve la norma ANTES de recortar
        self.normas.append(total.detach().float().cpu())
        if len(self.normas) >= self.max_pasos:
            raise _Fin()
        return total

    def tabla(self) -> pd.DataFrame:
        v = torch.stack(self.normas).numpy() if self.normas else np.zeros(0)
        paso = np.arange(len(v))
        return pd.DataFrame({"paso": paso, "epoca": paso // self.ppe, "norma": v})


def _entrenar(arq, X_tr, y_tr, X_va, y_va, n_classes, sfreq, ch_names, epocas, seed, tmp):
    import training as tr
    lr, lote, env = CFG[arq]
    for k, v in env.items():
        os.environ[k] = v
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in tr.DEFAULT_TRAIN_CFG.items()}
    cfg.update(lr=lr, batch_size_physical=lote, batch_size_effective=lote)
    ppe = int(np.ceil(len(y_tr) / lote))
    reg = Registro(max_pasos=epocas * ppe, pasos_por_epoca=ppe)
    torch.nn.utils.clip_grad_norm_ = reg
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tr.set_seed(seed)
        modelo = tr.build_model(arq, n_chans=X_tr.shape[1], n_times=X_tr.shape[2],
                                n_classes=n_classes, ch_names=ch_names)
        try:
            tr.train_run(modelo, X_tr, y_tr, X_va, y_va, n_classes, sfreq, cfg, dev,
                         Path(tmp) / f"{arq}.pt", seed=seed)
        except _Fin:
            pass
    finally:
        torch.nn.utils.clip_grad_norm_ = reg.orig
    return reg.tabla(), cfg["grad_clip"]


def _datos_reales(args):
    import splits as sp
    import train_baseline as tb
    import training as tr
    tb._configurar(args.dataset, "eegnet", args.cache)   # falla con mensaje si no hay cache
    subs = args.subs or tb._list_subjects()[:2]
    for sub in subs:
        X, y_txt, chs = sp.load_subject(sub, tb.CACHE, window=tb.WINDOW,
                                        ch_names=tb.DS.CH_SETS["completo"])
        y = tr.encode_labels(y_txt, tb.DS.CLASS_ORDER)
        folds = sp.make_within_subject_folds(y, n_folds=5, seed=2026)
        b = sp.build_within_fold(folds[args.fold], X, y, normalize="global")
        yield (sub, "within", args.fold, b["X_train"], b["y_train"], b["X_val"], b["y_val"],
               tb.N_CLASSES, tb.SFREQ, list(chs) if chs is not None else None)


def _datos_sinteticos():
    rng = np.random.default_rng(0)
    n, c, t = 400, 62, 800
    y = rng.integers(0, 2, n)
    X = rng.standard_normal((n, c, t)).astype(np.float32)
    X[y == 1, :4] += 0.3
    yield ("sintetico", "within", 0, X[:320], y[:320], X[320:], y[320:], 2, 200.0, None)


def resumir(df: pd.DataFrame, umbral: float) -> pd.DataFrame:
    filas = []
    for (arq, ep), g in df.groupby(["arq", "epoca"]):
        v = g["norma"].to_numpy()
        fin = v[np.isfinite(v)]
        filas.append({"arq": arq, "epoca": int(ep), "n_pasos": len(v),
                      "n_no_finitos": int((~np.isfinite(v)).sum()),
                      "frac_mayor_umbral": float((fin > umbral).mean()) if len(fin) else np.nan,
                      "mediana": float(np.median(fin)), "p90": float(np.percentile(fin, 90)),
                      "p99": float(np.percentile(fin, 99)), "max": float(fin.max())})
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="openbmi")
    ap.add_argument("--cache", default=None, help="carpeta epochs_cache_ica de OpenBMI")
    ap.add_argument("--arqs", nargs="+", default=["eegnet", "conformer"], choices=list(CFG))
    ap.add_argument("--subs", nargs="*", default=None)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epocas", type=int, default=15)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--selftest", action="store_true",
                    help="datos sinteticos, solo EEGNet: valida la instrumentacion")
    args = ap.parse_args()

    if args.selftest:
        args.arqs, args.epocas, sufijo = ["eegnet"], min(args.epocas, 3), "_selftest"
        fuente = _datos_sinteticos()
    else:
        sufijo = ""
        fuente = _datos_reales(args)

    OUT.mkdir(parents=True, exist_ok=True)
    partes = []
    umbral = None
    with tempfile.TemporaryDirectory() as tmp:
        for (sub, prot, fold, Xt, yt, Xv, yv, nc, sf, chs) in fuente:
            for arq in args.arqs:
                print(f"{arq} {sub} {prot} fold{fold}: {args.epocas} epocas", flush=True)
                d, umbral = _entrenar(arq, Xt, yt, Xv, yv, nc, sf, chs, args.epocas,
                                      args.seed, tmp)
                d.insert(0, "fold", fold); d.insert(0, "protocolo", prot)
                d.insert(0, "sub", sub); d.insert(0, "arq", arq)
                partes.append(d)
    df = pd.concat(partes, ignore_index=True)
    df.to_csv(OUT / f"norma_gradiente_pasos{sufijo}.csv", index=False)
    res = resumir(df, umbral)
    res.to_csv(OUT / f"norma_gradiente_resumen{sufijo}.csv", index=False)
    print(res.to_string(index=False))
    print(f"\numbral de recorte (grad_clip) = {umbral}; CSV en {OUT}")


if __name__ == "__main__":
    main()
