"""Genera el CACHÉ DE TENSORES de OpenBMI/Lee2019 — entregable de la fase de datos.

Mismo diseño que `scripts/mind/build_cache.py` (probado sobre 30 sujetos):

  * VENTANA AMPLIA (-2..5 s): cualquier ventana de modelado es luego un simple RECORTE,
    así que cambiar de ventana no obliga a reprocesar horas de cómputo.
  * MONTAJE COMPLETO (62 canales EEG): la selección se aplica después como VISTA
    (`CH_SETS`), lo que habilita el ablation de nº de canales.
  * SIN NORMALIZAR: el z-score se ajusta por fold, solo con train (anti-fuga).
  * Escritura incremental: si se interrumpe, al relanzar salta los sujetos ya hechos.

DIFERENCIAS DELIBERADAS CON EL PIPELINE DE MIND:
  * **SIN notch.** El pasa-banda 0.5-40 Hz ya deja 60 y 120 Hz en la banda de rechazo
    (atenuación ~-50 dB en 60 Hz con firwin), así que un notch en esas frecuencias es
    redundante e inerte. Se retiró (antes se aplicaba a 60/120 Hz por la red de Corea).
  * **SIN ICA y SIN autoreject por defecto.** En MIND el ICA era imprescindible por el
    confundidor ocular del paradigma direccional; aquí el paradigma es mano izq/der y
    los baselines publicados (70.1 % / 75.88 %) NO aplican ese tratamiento. Para que la
    réplica sea comparable con sus cifras hay que parecerse a su pipeline, no al nuestro.
    Ambos quedan disponibles como flags para revisarlo más adelante.

MODO --ica (Actividad 2.3, añadido 2026-07-24)
  El Control A sobre los 54 sujetos midió que **9/54 (17 %)** tienen información ocular
  ESPECÍFICA DE CLASE significativa (binomial exacto + Holm sobre el proxy HEOG), con
  sub-29 en 0.838. El mecanismo: OpenBMI es izq/der con cue visual LATERALIZADO y la
  ventana de modelado arranca en el cue, así que la sacada hacia el estímulo cae dentro.

  La contaminación es HETEROGÉNEA, que para esta tesis es peor que si fuera uniforme:
  un sesgo uniforme se cancelaría al comparar arquitecturas, pero en LOSO cada sujeto es
  el test de SU fold, así que los 9 afectados meten varianza no neural justo en la medida
  inter-sujeto, que es el objeto de la investigación.

  `--ica` escribe a un caché SEPARADO (`epochs_cache_ica`). Es deliberado: el caché sin
  ICA debe seguir existiendo para que la Puerta 1 siga siendo reproducible y para poder
  presentar la comparación antes/después como control metodológico.

METADATOS POR TRIAL (imprescindibles, no los tenía MIND):
    session : '0' (día 1) | '1' (día 2)     -> eje de cambio de dominio TEMPORAL
    run     : '1train' (offline) | '4test' (online)  -> eje offline/online del paper
  Son ejes distintos y no hay que confundirlos.

Uso:
  python scripts/openbmi/build_cache.py                 # los 54 sujetos, SIN ICA
  python scripts/openbmi/build_cache.py --subjects 1 2  # prueba con pocos
  python scripts/openbmi/build_cache.py --ica           # caché limpio -> epochs_cache_ica
"""
import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mne

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import datasets

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

DS = datasets.get("openbmi")
LOG_DIR = REPO_ROOT / "results" / "reports" / "openbmi"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CFG = {
    "l_freq": 0.5, "h_freq": 40.0,   # el pasa-banda 0.5-40 ya suprime 60/120 Hz: sin notch
    "resample_hz": int(DS.SFREQ),    # 200 Hz, exigido por LaBraM
    "epoch_tmin": -2.0, "epoch_tmax": 5.0,   # ventana AMPLIA -> recortar después
    "montage": "standard_1005",      # hacen falta 10-05: FTT9h, TTP7h, TPP9h...
    "use_ica": False,
    "use_autoreject": False,
    "amp_reject_uv": None,           # sin rechazo grosero: los baselines no lo aplican
    # --- solo activos con --ica ---
    "ica_n_components": 30,          # de 62 canales; PCA previa, abarata mucho el ajuste
    "ica_decim": 3,                  # submuestreo SOLO para ajustar (no afecta a la salida)
    "ica_seed": 2026,
    # Umbrales por clase, alineados con los autores de MIND (docs/COMPARACION_PIPELINE_AUTORES.md).
    # OJO con el nombre: 'eye blink' es la clase **Eye** entera de ICLabel (mapea a 'eog'),
    # no solo parpadeos -> cubre las SACADAS HORIZONTALES, que son nuestro artefacto real.
    "ica_drop_thresholds": {"eye blink": 0.8, "muscle artifact": 0.9, "channel noise": 0.5},
}


def cfg_hash():
    return hashlib.sha256(json.dumps(CFG, sort_keys=True, default=str).encode()).hexdigest()[:12]


def preparar_raw(raw):
    """Canales EEG + montaje + pasa-banda + (CAR + ICA) + remuestreo. NO normaliza."""
    raw = raw.copy()
    presentes = [c for c in DS.NON_EEG_CH if c in raw.ch_names]
    if presentes:
        raw.drop_channels(presentes)
    raw.set_montage(CFG["montage"], on_missing="warn")
    raw.filter(CFG["l_freq"], CFG["h_freq"], verbose="ERROR")
    # Sin notch: 60/120 Hz ya quedan fuera del pasa-banda 0.5-40 (redundante e inerte).
    if raw.info["sfreq"] != CFG["resample_hz"]:
        # Remuestrear el RAW (no los índices de evento): MNE reajusta las anotaciones,
        # que van en segundos. Calcular índices a 1000 Hz y remuestrear después descarta
        # épocas en silencio (trampa #3 del HANDOFF, ya nos pasó en MIND).
        raw.resample(CFG["resample_hz"], verbose="ERROR")

    n_excl, labels = 0, []
    if CFG["use_ica"]:
        # Se remuestrea ANTES del ICA: a 200 Hz el ajuste cuesta 5x menos que a 1000 Hz
        # y el ICA no necesita el ancho de banda que ya quitó el paso bajo de 40 Hz.
        raw, n_excl, labels = quitar_artefactos_ica(raw)
    return raw, n_excl, labels


def quitar_artefactos_ica(raw):
    """ICA + ICLabel automático (sin inspección manual). Devuelve (raw, n_excl, etiquetas).

    El ICA es NO SUPERVISADO: no ve las etiquetas de clase, así que no puede introducir
    fuga. Se ajusta por run, que es la unidad en la que el montaje y el estado del sujeto
    son estables.
    """
    from mne.preprocessing import ICA
    from mne_icalabel import label_components

    # ICLabel FUE ENTRENADO con referencia promedio y paso alto >=1 Hz: sin ambas cosas
    # las probabilidades que devuelve no son las que dice su tabla y los umbrales
    # calibrados (0.8/0.9/0.5) dejan de significar lo que creemos.
    raw = raw.copy().set_eeg_reference("average", verbose="ERROR")
    raw_fit = raw.copy().filter(1.0, None, verbose="ERROR")

    ica = ICA(n_components=min(CFG["ica_n_components"], len(raw.ch_names) - 2),
              method="infomax", fit_params=dict(extended=True),   # ICLabel exige infomax ext.
              max_iter="auto", random_state=CFG["ica_seed"])
    ica.fit(raw_fit, decim=CFG["ica_decim"], verbose="ERROR")

    res = label_components(raw_fit, ica, method="iclabel")
    etiquetas = res["labels"]
    probs = np.asarray(res["y_pred_proba"])
    thr = {k.lower(): v for k, v in CFG["ica_drop_thresholds"].items()}
    excl = [i for i, (lab, p) in enumerate(zip(etiquetas, probs))
            if lab.lower() in thr and p >= thr[lab.lower()]]
    ica.exclude = excl
    return ica.apply(raw, verbose="ERROR"), len(excl), [etiquetas[i] for i in excl]


def epocar(raw):
    """Épocas de la ventana amplia, sin baseline (se decide en el modelado)."""
    eventos, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    quedan = {k: v for k, v in event_id.items() if k in DS.CLASS_ORDER}
    if not quedan:
        raise ValueError(f"No hay eventos de clase en {list(event_id)}")
    ep = mne.Epochs(raw, eventos, event_id=quedan,
                    tmin=CFG["epoch_tmin"], tmax=CFG["epoch_tmax"],
                    baseline=None, preload=True, reject=None,
                    on_missing="warn", verbose="ERROR")
    return ep


def construir_sujeto(ds_moabb, sub: int):
    """Procesa las 2 sesiones x 2 runs de un sujeto y devuelve el tensor concatenado."""
    datos = ds_moabb.get_data(subjects=[sub])[sub]
    Xs, ys, sess, runs = [], [], [], []
    ch_ref = None
    ica_por_run = {}      # POR RUN, no sumado: en MIND leí un total de 25 como si fuera
                          # de un solo run y frené un cacheo de 3.5 h sin motivo (trampa #7).

    for ses in sorted(datos):
        for run in sorted(datos[ses]):
            raw, n_excl, labels = preparar_raw(datos[ses][run])
            ep = epocar(raw)
            inv = {v: k for k, v in ep.event_id.items()}
            y = np.array([inv[c] for c in ep.events[:, 2]])
            X = ep.get_data() * 1e6                     # µV, SIN normalizar
            if ch_ref is None:
                ch_ref = ep.ch_names
            elif ep.ch_names != ch_ref:
                raise ValueError(f"{sub} ses{ses} run{run}: los canales no coinciden")
            Xs.append(X.astype(np.float32)); ys.append(y)
            sess += [ses] * len(y); runs += [run] * len(y)
            if CFG["use_ica"]:
                ica_por_run[f"{ses}/{run}"] = {"n": n_excl, "labels": labels}

    X = np.concatenate(Xs); y = np.concatenate(ys)
    return (X, y, np.array(sess), np.array(runs), ch_ref,
            ep.tmin, ep.info["sfreq"], ica_por_run)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--ica", action="store_true",
                    help="aplica CAR + ICA/ICLabel y escribe a un cache SEPARADO")
    ap.add_argument("--eye-thr", type=float, default=None,
                    help="umbral de ICLabel para 'eye blink' (por defecto 0.8)")
    ap.add_argument("--muscle-thr", type=float, default=None,
                    help="umbral de ICLabel para 'muscle artifact' (por defecto 0.9)")
    ap.add_argument("--ica-n-components", type=int, default=None,
                    help="componentes del ICA (por defecto 30 de 62 canales)")
    ap.add_argument("--suffix", default="",
                    help="sufijo del cache, para comparar configuraciones sin pisarlas")
    args = ap.parse_args()

    CFG["use_ica"] = args.ica
    if args.eye_thr is not None:
        CFG["ica_drop_thresholds"]["eye blink"] = args.eye_thr
    if args.muscle_thr is not None:
        CFG["ica_drop_thresholds"]["muscle artifact"] = args.muscle_thr
    if args.ica_n_components is not None:
        CFG["ica_n_components"] = args.ica_n_components
    if args.ica:
        try:
            import mne_icalabel  # noqa: F401
        except ImportError:
            sys.exit("Falta mne-icalabel:  conda activate eeg && pip install mne-icalabel")
    # Caches separados: el de sin-ICA sostiene la reproducibilidad de la Puerta 1 y sirve
    # de termino de comparacion "antes" del control metodologico.
    nombre = (("epochs_cache_ica" + args.suffix) if args.ica else DS.CACHE_DIR.name)
    cache = DS.CACHE_DIR.parent / nombre
    cache.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / (f"tensor_cache_log_ica{args.suffix}.csv" if args.ica
                     else "tensor_cache_log.csv")
    ch = cfg_hash()

    ds_moabb = DS.make_dataset(test_run=True)     # sortea los 2 bugs de MOABB 1.5.0
    subs = args.subjects or list(ds_moabb.subject_list)

    hechos = set()
    if log.exists():
        try:
            hechos = set(pd.read_csv(log)["sub"].tolist())
        except Exception:
            pass
    print(f"OpenBMI | sujetos: {len(subs)} | ya cacheados: {len(hechos)} | cfg {ch}",
          flush=True)
    print(f"ICA: {'SI (CAR + ICLabel ' + str(CFG['ica_drop_thresholds']) + ')' if args.ica else 'NO'}",
          flush=True)
    print(f"Destino: {cache}\n", flush=True)

    for i, sub in enumerate(subs, 1):
        if sub in hechos:
            continue
        t0 = time.time()
        try:
            X, y, sess, runs, chs, tmin, sfreq, ica_run = construir_sujeto(ds_moabb, sub)
        except Exception as e:
            print(f"[{i}/{len(subs)}] sub-{sub:02d}: ERROR {type(e).__name__}: {e}", flush=True)
            continue

        np.savez_compressed(
            cache / f"sub-{sub:02d}_full.npz",
            X=X, y=y, session=sess, run=runs,
            ch_names=np.array(chs), sfreq=sfreq, tmin=tmin,
            subject=f"sub-{sub:02d}", cfg_hash=ch,
        )
        clases = {k: int(np.sum(y == k)) for k in sorted(set(y.tolist()))}
        por_ses = {s: int(np.sum(sess == s)) for s in sorted(set(sess.tolist()))}
        fila = {
            "sub": sub, "n_trials": len(y), "n_canales": X.shape[1], "n_muestras": X.shape[2],
            "sfreq": sfreq, "tmin": tmin, "clases": json.dumps(clases),
            "por_sesion": json.dumps(por_ses),
            "por_run": json.dumps({r: int(np.sum(runs == r)) for r in sorted(set(runs.tolist()))}),
            "cfg_hash": ch, "segundos": round(time.time() - t0, 1),
            "ica_por_run": json.dumps({k: v["n"] for k, v in ica_run.items()}) if ica_run else "",
            "ica_labels": json.dumps(sorted({l for v in ica_run.values() for l in v["labels"]}))
                          if ica_run else "",
        }
        pd.DataFrame([fila]).to_csv(log, mode="a", header=not log.exists(), index=False)
        extra = ""
        if ica_run:
            n = [v["n"] for v in ica_run.values()]
            extra = f" | ICA excl/run {n} de {CFG['ica_n_components']}"
        print(f"[{i}/{len(subs)}] sub-{sub:02d}: {len(y)} trials {X.shape[1]}ch "
              f"{X.shape[2]}m | clases {clases} | sesiones {por_ses}{extra} "
              f"({fila['segundos']}s)", flush=True)

    print(f"\nCaché en {cache}")
    print("ESPERADO por sujeto: 400 trials · 200 por clase · 200 por sesión · 62 canales")
    if args.ica:
        print("Revisa la columna ica_por_run: ~3-8 componentes de 30 por run es lo normal.")
        print("Si sale >15 de forma sistemática, el ICA está comiéndose señal: parar y revisar.")


if __name__ == "__main__":
    main()
