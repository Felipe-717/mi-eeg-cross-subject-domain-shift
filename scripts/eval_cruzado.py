#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_cruzado.py — evalua un modelo entrenado en un dataset sobre OTRO dataset.

Responde al punto 6 del director: validar en un segundo conjunto de datos. El modelo se
entrena una sola vez con `train_baseline.py --mode all` sobre los 54 sujetos de OpenBMI y
aqui se aplica, SIN reentrenar y SIN ver ninguna etiqueta del destino, a cada sujeto del
segundo dataset. Es transferencia de disparo cero.

    ~/anaconda3/envs/eeg/python.exe scripts/eval_cruzado.py \
        --ckpt checkpoints/openbmi/eegnet/all/all_seed2026_completo_global.pt \
        --dataset wbcic_2c

LO QUE ESTE SCRIPT SE NIEGA A HACER
    No adapta nada en silencio. Si los canales, la frecuencia de muestreo, la ventana o
    el orden de clases no coinciden, aborta y dice cual es la discrepancia. Rellenar un
    canal ausente con ceros o reordenar clases por su cuenta produciria un numero que
    parece un resultado y no lo es.

LA DECISION QUE NO ES OBVIA: la normalizacion del destino (--norm-destino)
    source     Aplica la media y la desviacion del ENTRENAMIENTO en el origen. Es lo mas
               estricto: el modelo llega al dataset nuevo sin ninguna informacion de el.
               Tambien es lo mas duro, porque las escalas de amplitud entre montajes y
               amplificadores distintos pueden diferir mucho.
    per_trial  Cada segmento se escala por su propia media y desviacion. No mira ningun
               otro segmento, asi que NO es transductivo, y absorbe la diferencia de
               escala entre datasets. Es la opcion defendible si `source` colapsa.
    per_subject Z-score con los estadisticos de cada sujeto del destino. SI es
               transductivo: asume tener por delante todos los datos del sujeto antes de
               clasificar. Se puede usar, pero HAY QUE DECLARARLO en la memoria.

    Por defecto se ejecutan las tres y se reportan las tres, porque la diferencia entre
    ellas es en si misma un resultado sobre el cambio de dominio.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import datasets                      # noqa: E402
import splits as sp                  # noqa: E402
import training as tr                # noqa: E402

REPORTS = Path("results/reports")
POLITICAS = ("source", "per_trial", "per_subject")


def _cargar_meta(ckpt: Path) -> dict:
    meta = ckpt.with_suffix(".meta.json")
    if not meta.exists():
        raise SystemExit(
            f"x No existe {meta}.\n"
            f"  El checkpoint solo, sin el sidecar, no basta: falta el orden de los canales\n"
            f"  y los estadisticos de normalizacion. Reentrena con --mode all, que lo escribe."
        )
    return json.loads(meta.read_text(encoding="utf-8"))


def _comprobar_compatibilidad(meta: dict, ds) -> None:
    """Aborta con un mensaje concreto ante cualquier discrepancia de protocolo."""
    problemas = []
    if float(ds.SFREQ) != float(meta["sfreq"]):
        problemas.append(f"frecuencia de muestreo: origen {meta['sfreq']} Hz, "
                         f"destino {ds.SFREQ} Hz")
    if list(ds.WINDOW) != list(meta["window"]):
        problemas.append(f"ventana: origen {meta['window']} s, destino {list(ds.WINDOW)} s")
    if list(ds.CLASS_ORDER) != list(meta["class_order"]):
        problemas.append(f"orden de clases: origen {meta['class_order']}, "
                         f"destino {list(ds.CLASS_ORDER)}. Reordenar aqui cambiaria el "
                         f"significado de las predicciones.")
    if problemas:
        raise SystemExit("x El modelo y el dataset destino no son compatibles:\n  - "
                         + "\n  - ".join(problemas))


def _alinear_canales(X: np.ndarray, chs_destino: list[str],
                     chs_modelo: list[str]) -> np.ndarray:
    """Reordena y recorta los canales del destino al orden EXACTO que espera el modelo.

    La primera capa de todas estas arquitecturas es espacial: la posicion de cada canal en
    el tensor tiene significado aprendido. Pasar los mismos canales en otro orden no es un
    detalle de formato, es alimentar el modelo con basura.
    """
    idx = {c.upper(): k for k, c in enumerate(chs_destino)}
    faltan = [c for c in chs_modelo if c.upper() not in idx]
    if faltan:
        raise SystemExit(
            f"x El destino no tiene {len(faltan)} de los {len(chs_modelo)} canales que el "
            f"modelo espera:\n    {faltan}\n"
            f"  Rellenarlos con ceros daria un numero sin sentido. La salida correcta es\n"
            f"  reentrenar el modelo sobre la INTERSECCION de los dos montajes:\n"
            f"    train_baseline.py --mode all --channels <vista_comun>"
        )
    return X[:, [idx[c.upper()] for c in chs_modelo], :]


def _normalizar(X: np.ndarray, politica: str, meta: dict) -> np.ndarray:
    if politica == "source":
        if meta.get("norm_mean") is None:
            raise SystemExit("x El modelo se entreno sin normalizacion global: "
                             "la politica 'source' no aplica.")
        stats = {"mean": np.asarray(meta["norm_mean"], dtype=np.float64)[:, None],
                 "std": np.asarray(meta["norm_std"], dtype=np.float64)[:, None]}
        return sp.apply_normalizer(X, stats)
    if politica == "per_trial":
        return sp.normalize_per_trial(X)
    if politica == "per_subject":
        # Un solo sujeto por llamada, asi que el identificador es constante.
        return sp.normalize_per_subject(X, np.zeros(len(X), dtype=int))
    raise ValueError(politica)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=Path,
                    help="checkpoint de --mode all (junto a su .meta.json)")
    ap.add_argument("--dataset", required=True, choices=list(datasets.disponibles()),
                    help="dataset DESTINO")
    ap.add_argument("--cache", default=None, help="cache del destino (por defecto el suyo)")
    ap.add_argument("--norm-destino", nargs="*", default=list(POLITICAS), choices=POLITICAS,
                    help="politicas de normalizacion a reportar (por defecto, las tres)")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--solo-reservado", type=float, default=None, metavar="FRAC",
                    help="evaluar SOLO sobre la particion reservada del destino, la misma "
                         "que `train_baseline.py --mode all --frac-test FRAC` aparta al "
                         "entrenar el modelo nativo. Con esto el modelo transferido y el "
                         "nativo se miden sobre los MISMOS segmentos y su diferencia es el "
                         "costo del cambio de dominio. Pasar el mismo valor que alli.")
    ap.add_argument("--out", default=None, help="CSV de salida")
    a = ap.parse_args()

    meta = _cargar_meta(a.ckpt)
    ds = datasets.get(a.dataset)
    _comprobar_compatibilidad(meta, ds)

    # Las variantes de LaBraM y EEGMamba se eligen por variable de entorno y cambian la
    # arquitectura, no solo el nombre del fichero. Si no coinciden, los pesos no encajan.
    if meta.get("run_tag"):
        print(f"  [aviso] el modelo se entreno con run_tag='{meta['run_tag']}'. "
              f"Comprueba que LABRAM_FINETUNE / EEGMAMBA_HEAD valen lo mismo ahora.",
              flush=True)

    cache = Path(a.cache) if a.cache else ds.CACHE_DIR
    if not cache.exists():
        raise SystemExit(f"x No existe el cache del destino: {cache}")
    subs = a.subjects or sorted(p.name.replace("_full.npz", "")
                                for p in cache.glob("sub-*_full.npz"))
    if not subs:
        raise SystemExit(f"x El cache {cache} no tiene ningun sujeto.")

    # GUARDIA DE FUGA. Evaluar sobre el MISMO dataset solo es legitimo en los sujetos que
    # el modelo no vio. Sin esta comprobacion, un `--subjects` mal escrito produce una cifra
    # sobre gente que estaba en el entrenamiento y parece un resultado de generalizacion.
    if ds.NAME == meta.get("dataset"):
        vistos = set(meta.get("subs_entrenamiento") or [])
        apartados = set(meta.get("subs_apartados") or [])
        if not apartados:
            raise SystemExit(
                f"x El checkpoint se entreno sobre {ds.NAME} SIN apartar ningun sujeto "
                f"(`subs_apartados` vacio), asi que no queda ninguna persona no vista sobre "
                f"la que evaluarlo en su propio dataset. Reentrena con --holdout-n N.")
        intrusos = sorted(set(subs) & vistos)
        if intrusos:
            raise SystemExit(
                f"x FUGA: {len(intrusos)} de los sujetos pedidos estuvieron en el "
                f"entrenamiento de este modelo: {' '.join(intrusos)}. "
                f"Los apartados son: {' '.join(sorted(apartados))}")
        print(f"  guardia ok: los {len(subs)} sujetos estan fuera del entrenamiento",
              flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = tr.build_model(meta["modelo"], n_chans=meta["n_chans"],
                            n_times=meta["n_times"], n_classes=len(meta["class_order"]),
                            ch_names=meta["ch_names"])
    estado = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    modelo.load_state_dict(estado["model_state"])
    modelo = modelo.to(device).eval()

    print(f"Modelo {meta['modelo']} ({meta['dataset']}, semilla {meta['seed']}) "
          f"-> {ds.NAME}: {len(subs)} sujetos, {meta['n_chans']} canales", flush=True)

    # La particion reservada se reconstruye, no se lee de disco: `separar_visto` es
    # determinista dados los datos, la fraccion y la semilla, asi que sale identica a la
    # que aparto `--mode all --frac-test`. Se calcula sobre TODOS los sujetos a la vez
    # porque ese es el orden de concatenacion con el que se genero alli.
    reservado = None
    if a.solo_reservado:
        m = sp.mascara_reservada(subs, cache, a.solo_reservado)
        reservado, ini = {}, 0
        for sub in subs:
            n = len(sp.load_subject(sub, cache, window=tuple(meta["window"]))[1])
            reservado[sub] = m[ini:ini + n]
            ini += n
        if ini != len(m):
            raise SystemExit(f"x Descuadre al repartir la mascara: {ini} != {len(m)}")
        print(f"  solo la particion reservada: {int(m.sum())} de {len(m)} segmentos "
              f"({m.mean():.1%})", flush=True)

    filas = []
    for sub in subs:
        X, y_txt, chs = sp.load_subject(sub, cache, window=tuple(meta["window"]),
                                        ch_names=None)
        X = _alinear_canales(X, chs, meta["ch_names"])
        y = tr.encode_labels(y_txt, meta["class_order"])
        if reservado is not None:
            X, y = X[reservado[sub]], y[reservado[sub]]
        for pol in a.norm_destino:
            m = tr.evaluate(modelo, _normalizar(X, pol, meta).astype(np.float32),
                            y, device, len(meta["class_order"]))
            filas.append({"sub": sub, "norm_destino": pol, "n_trials": len(y),
                          "acc": m["acc"], "kappa": m["kappa"], "f1_macro": m["f1_macro"],
                          "confusion_matrix": m["confusion_matrix"],
                          "model": meta["modelo"], "seed": meta["seed"],
                          "origen": meta["dataset"], "destino": ds.NAME,
                          "channels_view": meta["channels_view"],
                          "cfg_hash": meta["cfg_hash"], "ckpt_path": str(a.ckpt)})
        ult = {f["norm_destino"]: f["acc"] for f in filas if f["sub"] == sub}
        print(f"  {sub}: " + "  ".join(f"{k}={v:.3f}" for k, v in ult.items()), flush=True)

    df = pd.DataFrame(filas)
    out = Path(a.out) if a.out else (
        REPORTS / ds.NAME / f"cruzado_{meta['modelo']}_{meta['dataset']}_a_{ds.NAME}"
                            f"_seed{meta['seed']}_{meta['channels_view']}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n{'politica':<12} {'acc media':>10} {'sd':>7} {'kappa':>8}  (azar {ds.CHANCE:.2f})")
    for pol, g in df.groupby("norm_destino"):
        print(f"{pol:<12} {g.acc.mean():>10.4f} {g.acc.std():>7.4f} {g.kappa.mean():>8.4f}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
