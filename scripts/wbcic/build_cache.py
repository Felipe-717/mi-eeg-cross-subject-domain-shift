#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_cache.py — cache de tensores de WBCIC-SHU 2C, replicando el pipeline de OpenBMI.

El objetivo NO es preprocesar bien WBCIC por su cuenta: es que los dos datasets lleguen
al modelo habiendo pasado por el MISMO tratamiento, para que la validacion cruzada del
punto 6 mida cambio de dominio y no diferencia de pipeline.

DE DONDE SE PARTE (derivatives/2C dataset_processeddata/*.mat, verificado el 2026-09-06)
    (58 canales, 1000 muestras, 200 trials) a 250 Hz, etiquetas 1=izquierda 2=derecha.
    Los autores lo produjeron con `code/pre-processed/preprocessed.m`:
        quitar ECG/HEOR/HEOL/VEOU/VEOL   -> 59 canales
        pop_reref(EEG, 43)               -> referencia a Pz, que queda a cero y se cae
        pasa-banda 0.5-40 Hz             -> IGUAL que el nuestro
        notch 49-51 Hz                   -> el nuestro no lo lleva, pero 50 Hz ya cae
                                            fuera del pasa-banda: es inerte
        pop_rmbase sobre la epoca entera -> el nuestro no lo hace (ver DIFERENCIAS)
        epoca [0, 4] s                   -> IGUAL que nuestra WINDOW
        remuestreo a 250 Hz

LO QUE HACE ESTE SCRIPT PARA IGUALAR EL PIPELINE
    1. Reinserta Pz como fila de ceros -> 59 canales. Es exacto: los datos son X_i - Pz,
       y Pz - Pz = 0.
    2. CAR sobre los 59. Tambien exacto y no una aproximacion:
           (X_i - Pz) - media_j(X_j - Pz) = X_i - media_j(X_j)
       que es literalmente el CAR que aplica scripts/openbmi/build_cache.py.
    3. Remuestrea 250 -> 200 Hz, ANTES del ICA, en el mismo orden que OpenBMI.
    4. ICA + ICLabel con los MISMOS parametros: infomax extendido, 30 componentes,
       semilla 2026, ajuste sobre una copia filtrada a >=1 Hz y los mismos umbrales.
    5. Selecciona los 45 canales de la vista `comun_wbcic`.

DIFERENCIAS QUE NO SE PUEDEN ELIMINAR (declararlas en la memoria)
    * `pop_rmbase` de los autores resta a cada epoca y canal su media. El cache de
      OpenBMI usa `baseline=None`. No se puede igualar sin rehacer OpenBMI, que
      invalidaria los resultados ya publicados. La politica de normalizacion
      `per_trial` de eval_cruzado.py absorbe esta diferencia; `source` no.
    * El ICA de OpenBMI se ajusta por RUN sobre la senal continua y sobre 62 canales.
      Aqui se ajusta por sujeto y sesion sobre datos ya epocados y sobre 59 canales,
      porque la senal continua no esta disponible (las etiquetas de trial no se pueden
      leer del BDF crudo sin el lector propietario de Neuracle).

Uso:
    ~/anaconda3/envs/eeg/python.exe scripts/wbcic/build_cache.py --ica
    ~/anaconda3/envs/eeg/python.exe scripts/wbcic/build_cache.py            # sin ICA
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import mne
import numpy as np
import scipy.io as sio

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import datasets                                   # noqa: E402

mne.set_log_level("ERROR")

DERIV = Path("data/raw/wbcic/WBCIC_SHU Motor Imagery dataset/derivatives/"
             "2C dataset_processeddata")
CANALES_TSV = Path("data/raw/wbcic/WBCIC_SHU Motor Imagery dataset/"
                   "task-motorimagery_channels.tsv")
REFERENCIA = "Pz"                    # canal 43 de los 59: pop_reref(EEG, 43)
CLASES = {1: "left_hand", 2: "right_hand"}

# Identicos a scripts/openbmi/build_cache.py: si divergen, el pipeline deja de ser el
# mismo y la comparacion cruzada mide preprocesado en vez de cambio de dominio.
CFG = {
    "l_freq": 0.5, "h_freq": 40.0,   # ya aplicados en origen; se dejan como constancia
    "resample_hz": 200,
    "ventana_s": 4.0,          # WINDOW de los dos datasets -> 801 muestras
    "montage": "standard_1005",
    "use_ica": False,
    "ica_n_components": 30,
    "ica_decim": 3,
    "ica_seed": 2026,
    "ica_drop_thresholds": {"eye blink": 0.8, "muscle artifact": 0.9, "channel noise": 0.5},
    # None = sin rechazo, IGUAL que `amp_reject_uv` en scripts/openbmi/build_cache.py.
    # Se deja apagado por defecto para no romper la simetria de preprocesado sin decirlo.
    "amp_reject_uv": None,
}


def cfg_hash() -> str:
    return hashlib.sha256(json.dumps(CFG, sort_keys=True, default=str).encode()).hexdigest()[:12]


def canales_59() -> list[str]:
    """Los 59 EEG en el orden del fichero de canales BIDS."""
    filas = CANALES_TSV.read_text(encoding="utf-8").splitlines()
    cab = filas[0].split("\t")
    i_n, i_t = cab.index("name"), cab.index("type")
    ch = [f.split("\t")[i_n] for f in filas[1:] if f.split("\t")[i_t] == "EEG"]
    if len(ch) != 59 or ch[42] != REFERENCIA:
        raise SystemExit(f"x Se esperaban 59 canales con {REFERENCIA} en la posicion 43, "
                         f"llegaron {len(ch)} con {ch[42] if len(ch) > 42 else '?'}")
    return ch


def reinsertar_referencia(X: np.ndarray, ch59: list[str]) -> np.ndarray:
    """(58, t, n) -> (59, t, n) metiendo la fila de ceros de la referencia en su sitio."""
    k = ch59.index(REFERENCIA)
    return np.insert(X, k, 0.0, axis=0)


def remuestrear(X: np.ndarray) -> np.ndarray:
    """250 Hz x 1000 muestras -> 200 Hz x 801 muestras, que es lo que espera el modelo.

    POR QUE 801 Y NO 800. El cache de OpenBMI epoca una ventana amplia y recorta [0, 4] s
    con `load_subject`, que incluye los DOS extremos: 4.0 s x 200 Hz + 1 = 801 muestras.
    Los cuatro modelos se construyeron con `n_times=801`, asi que 800 no encaja.

    EL DESAJUSTE, MEDIDO Y DECLARADO. Las 1000 muestras a 250 Hz cubren 999 intervalos de
    4 ms, o sea [0, 3.996] s, no [0, 4.000]. Llevarlas a 801 puntos deja un espaciado de
    4.995 ms en vez de 5.000: un estiramiento temporal del 0.10 %. A 10 Hz eso mueve el
    ritmo a 10.010 Hz. Es inmaterial frente a la variabilidad del propio ritmo mu entre
    sujetos, y la alternativa (duplicar la ultima muestra) fabricaria un dato.

    Se usa el remuestreo por Fourier, valido aqui porque la senal ya viene limitada a
    40 Hz por el pasa-banda de los autores.
    """
    from scipy.signal import resample
    n_obj = int(round((CFG["ventana_s"]) * CFG["resample_hz"])) + 1
    return resample(X, n_obj, axis=-1)


def quitar_artefactos_ica(ep: mne.EpochsArray) -> tuple[mne.EpochsArray, int, list[str]]:
    """Copia literal del criterio de OpenBMI, sobre epocas en vez de senal continua.

    El ICA es NO SUPERVISADO: no ve las etiquetas, asi que no puede introducir fuga.
    """
    from mne.preprocessing import ICA
    from mne_icalabel import label_components

    # ICLabel se entreno con referencia promedio y paso alto >=1 Hz. La referencia ya es
    # promedio en este punto; el paso alto se aplica solo a la copia de ajuste.
    ep_fit = ep.copy().filter(1.0, None, verbose="ERROR")
    ica = ICA(n_components=min(CFG["ica_n_components"], len(ep.ch_names) - 2),
              method="infomax", fit_params=dict(extended=True),
              max_iter="auto", random_state=CFG["ica_seed"])
    ica.fit(ep_fit, decim=CFG["ica_decim"], verbose="ERROR")

    res = label_components(ep_fit, ica, method="iclabel")
    etiquetas = res["labels"]
    probs = np.asarray(res["y_pred_proba"])
    thr = {k.lower(): v for k, v in CFG["ica_drop_thresholds"].items()}
    excl = [i for i, (lab, p) in enumerate(zip(etiquetas, probs))
            if lab.lower() in thr and p >= thr[lab.lower()]]
    ica.exclude = excl
    return ica.apply(ep, verbose="ERROR"), len(excl), [etiquetas[i] for i in excl]


def construir_sujeto(ruta: Path, ch59: list[str], vista: list[str]):
    m = sio.loadmat(ruta)
    X, y = m["data"].astype(np.float64), m["labels"].ravel()
    if X.shape[0] != 58 or X.shape[1] != 1000:
        raise SystemExit(f"x {ruta.name}: forma inesperada {X.shape}")
    desconocidas = set(np.unique(y).tolist()) - set(CLASES)
    if desconocidas:
        raise SystemExit(f"x {ruta.name}: etiquetas desconocidas {desconocidas}")

    X = reinsertar_referencia(X, ch59)              # (59, 1000, n)
    X = np.transpose(X, (2, 0, 1)) * 1e-6           # (n, 59, 1000), uV -> V para MNE
    X = remuestrear(X)                              # (n, 59, 801) a 200 Hz

    info = mne.create_info(ch59, sfreq=float(CFG["resample_hz"]), ch_types="eeg")
    ep = mne.EpochsArray(X, info, tmin=0.0, verbose="ERROR")
    ep.set_montage(CFG["montage"], on_missing="warn")
    ep.set_eeg_reference("average", verbose="ERROR")     # recupera EXACTAMENTE el CAR

    n_excl, labels = 0, []
    if CFG["use_ica"]:
        ep, n_excl, labels = quitar_artefactos_ica(ep)

    ep.pick(vista)                                   # los 45 compartidos, en su orden
    n_rech = 0
    if CFG["amp_reject_uv"]:
        # Rechazo grosero POR SEGMENTO, despues del ICA y de recortar la vista: lo que
        # importa es la amplitud que va a ver el modelo, no la de canales que se descartan.
        # Motivo medido (2026-09-06): sub-037 trae UN segmento con 78.477 uV en O1 sobre un
        # maximo mediano de 50 uV. El CAR reparte ese pico entre los 45 canales, y ese
        # unico segmento de 10.199 infla la desviacion tipica de O1 catorce veces y la de
        # otros 42 canales mas de un 5 %. Es saturacion del amplificador, no senal.
        pico = np.abs(ep.get_data(copy=False)).max(axis=(1, 2)) * 1e6
        conservar = pico <= CFG["amp_reject_uv"]
        n_rech = int((~conservar).sum())
        if n_rech:
            if not conservar.any():
                raise SystemExit(f"x {ruta.name}: el umbral descarta TODOS los segmentos")
            ep = ep[conservar]
            y = y[conservar]
    if ep.ch_names != vista:
        raise SystemExit(f"x {ruta.name}: el orden de canales no es el de la vista")
    Xo = ep.get_data(copy=True) * 1e6                # de vuelta a uV, como OpenBMI
    return (Xo.astype(np.float32), np.array([CLASES[v] for v in y]), ep.ch_names,
            n_excl, labels, n_rech)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ica", action="store_true", help="aplicar ICA + ICLabel (recomendado: "
                                                       "el cache de OpenBMI en uso lo lleva)")
    ap.add_argument("--vista", default="comun_wbcic")
    ap.add_argument("--salida", type=Path, default=None)
    ap.add_argument("--subs", default=None, help="restringir sujetos, p. ej. `1-3`")
    ap.add_argument("--umbral-uv", type=float, default=None,
                    help="descarta los segmentos cuyo pico supere este valor en uV. Apagado "
                         "por defecto, igual que en OpenBMI. Con 500 se elimina el unico "
                         "segmento saturado de sub-037 (78.477 uV) sin tocar nada mas.")
    a = ap.parse_args()
    CFG["use_ica"] = a.ica
    CFG["amp_reject_uv"] = a.umbral_uv

    ds = datasets.get("wbcic_2c")
    vista = ds.CH_SETS[a.vista]
    if vista is None:
        raise SystemExit(f"x La vista {a.vista!r} es None (todos): usa una vista explicita.")
    ch59 = canales_59()
    faltan = [c for c in vista if c not in ch59]
    if faltan:
        raise SystemExit(f"x La vista {a.vista!r} pide canales que WBCIC no tiene: {faltan}")

    salida = a.salida or Path("data/processed/wbcic") / (
        "epochs_cache_ica" if a.ica else "epochs_cache")
    salida.mkdir(parents=True, exist_ok=True)

    ficheros = sorted(DERIV.glob("sub-*/ses-*/**/*.mat"))
    if a.subs:
        pedidos = set()
        for p in a.subs.split(","):
            if "-" in p:
                x, y = p.split("-"); pedidos |= set(range(int(x), int(y) + 1))
            else:
                pedidos.add(int(p))
        ficheros = [f for f in ficheros
                    if int(re.search(r"sub-(\d+)", str(f)).group(1)) in pedidos]
    if not ficheros:
        raise SystemExit(f"x No hay .mat en {DERIV}")

    print(f"WBCIC-2C -> {salida}\n  {len(ficheros)} ficheros | vista {a.vista} "
          f"({len(vista)} ch) | ICA {'SI' if a.ica else 'NO'} | cfg_hash {cfg_hash()}",
          flush=True)

    t0, registro = time.time(), []
    for k, f in enumerate(ficheros, 1):
        sub = re.search(r"sub-(\d+)", str(f)).group(0)
        ses = re.search(r"ses-(\d+)", str(f)).group(0)
        X, y, chs, n_excl, labs, n_rech = construir_sujeto(f, ch59, vista)
        # Un .npz por SUJETO, como en OpenBMI. Con una sola sesion no hay que fusionar.
        destino = salida / f"{sub}_full.npz"
        np.savez_compressed(destino, X=X, y=y, ch_names=np.array(chs),
                            sfreq=float(CFG["resample_hz"]), tmin=0.0,
                            session=np.array([f"session_{int(ses[-2:])}"] * len(y)))
        registro.append({"sub": sub, "ses": ses, "n_trials": int(len(y)),
                         "n_ica_excl": n_excl, "ica_labels": ";".join(labs),
                         "n_amp_rechazados": n_rech})
        print(f"  [{k}/{len(ficheros)}] {sub} {ses}: {X.shape} | ICA excl {n_excl} "
              f"({', '.join(labs) if labs else '-'})"
              + (f" | AMP rechazados {n_rech}" if n_rech else "")
              + f" | {(time.time() - t0) / 60:.1f} min", flush=True)

    reg = salida / "_registro.json"
    reg.write_text(json.dumps({"cfg": CFG, "cfg_hash": cfg_hash(), "vista": a.vista,
                               "canales": vista, "sujetos": registro}, indent=2),
                   encoding="utf-8")
    print(f"\nListo en {(time.time() - t0) / 60:.1f} min -> {salida}\n  registro: {reg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
