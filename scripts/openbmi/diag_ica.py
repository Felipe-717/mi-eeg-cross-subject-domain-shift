"""DIAGNOSTICO: ¿por que el ICA no limpia a sub-29?

CONTEXTO (2026-07-24). Tras aplicar CAR + ICA/ICLabel con umbral 0.8 sobre 'eye blink',
el proxy ocular de sub-29 baja de 0.838 a 0.705 — sigue con p=7e-17. El tratamiento
funciona parcialmente. Hay DOS causas posibles que piden arreglos OPUESTOS:

  (1) ICLabel SI identifica el componente ocular, pero con confianza <0.8, y el umbral
      lo deja pasar.                                    -> arreglo: BAJAR el umbral.
  (2) El artefacto esta REPARTIDO entre varios componentes y ninguno llega a ser un
      "ojo" reconocible. Es plausible porque truncamos a 30 componentes de 62 canales:
      una fuente partida en trozos reparte tambien su firma y baja la confianza de cada
      trozo.                                            -> arreglo: SUBIR n_components.

Bajar el umbral cuando el problema es el (2) borra cerebro sin quitar el artefacto, asi
que hay que distinguirlas ANTES de tocar nada.

QUE MIDE
  Para cada componente: etiqueta de ICLabel, probabilidad, y — lo que de verdad decide —
  su CORRELACION con el proxy HEOG (Fp2-Fp1) y si su serie temporal por trial DECODIFICA
  la clase por si sola. Un componente que decodifica la clase y correlaciona con el HEOG
  ES el confundidor, lleve la etiqueta que lleve.

LECTURA
  * Si los componentes que decodifican estan etiquetados 'eye blink' con prob 0.5-0.8
    -> causa (1), bajar umbral.
  * Si estan etiquetados 'brain'/'other', o hay varios repartiendose el HEOG
    -> causa (2), subir n_components.

Uso:
  python scripts/openbmi/diag_ica.py                     # sub-29, primer run
  python scripts/openbmi/diag_ica.py --subject 1         # comparar con uno limpio
  python scripts/openbmi/diag_ica.py --n-components 45   # probar mas componentes
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import mne
from mne.preprocessing import ICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import datasets  # noqa: E402

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")
DS = datasets.get("openbmi")


def preparar(raw, sfreq):
    raw = raw.copy()
    fuera = [c for c in DS.NON_EEG_CH if c in raw.ch_names]
    if fuera:
        raw.drop_channels(fuera)
    raw.set_montage("standard_1005", on_missing="warn")
    raw.filter(0.5, 40.0, verbose="ERROR")
    notch = [f for f in (60.0, 120.0) if f < raw.info["sfreq"] / 2]
    if notch:
        raw.notch_filter(notch, verbose="ERROR")
    if raw.info["sfreq"] != sfreq:
        raw.resample(sfreq, verbose="ERROR")
    return raw.set_eeg_reference("average", verbose="ERROR")


def decodifica(feat, y, seed=2026):
    """Potencia log por trial de UNA serie -> LDA. Devuelve accuracy de CV 5x."""
    X = np.log(np.var(feat, axis=-1) + 1e-12).reshape(len(y), -1)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pipe = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    return cross_val_score(pipe, X, y, cv=cv, scoring="accuracy").mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", type=int, default=29)
    ap.add_argument("--n-components", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    ds = DS.make_dataset(test_run=True)
    datos = ds.get_data(subjects=[args.subject])[args.subject]
    ses = sorted(datos)[0]
    run = sorted(datos[ses])[0]
    print(f"sub-{args.subject:02d} | sesion {ses} run {run} | "
          f"n_components={args.n_components}\n", flush=True)

    raw = preparar(datos[ses][run], int(DS.SFREQ))
    ica = ICA(n_components=min(args.n_components, len(raw.ch_names) - 2),
              method="infomax", fit_params=dict(extended=True),
              max_iter="auto", random_state=args.seed)
    raw_fit = raw.copy().filter(1.0, None, verbose="ERROR")
    ica.fit(raw_fit, decim=3, verbose="ERROR")

    from mne_icalabel import label_components
    res = label_components(raw_fit, ica, method="iclabel")
    labels, probs = res["labels"], np.asarray(res["y_pred_proba"])

    # Epocas de la ventana de modelado, sobre las fuentes y sobre el proxy HEOG.
    ev, eid = mne.events_from_annotations(raw, verbose="ERROR")
    eid = {k: v for k, v in eid.items() if k in DS.CLASS_ORDER}
    ep = mne.Epochs(raw, ev, event_id=eid, tmin=DS.WINDOW[0], tmax=DS.WINDOW[1],
                    baseline=None, preload=True, verbose="ERROR")
    inv = {v: k for k, v in ep.event_id.items()}
    y = np.array([inv[c] for c in ep.events[:, 2]])
    S = ica.get_sources(ep).get_data()               # (n_trials, n_comp, n_muestras)

    ch = {c.upper(): i for i, c in enumerate(ep.ch_names)}
    heog = (ep.get_data()[:, ch["FP2"], :] - ep.get_data()[:, ch["FP1"], :])

    print(f"{'#':>3} {'etiqueta':16s} {'prob':>6} {'|r| HEOG':>9} {'decod.':>7}")
    print("-" * 48)
    filas = []
    for i in range(S.shape[1]):
        r = abs(np.corrcoef(S[:, i, :].ravel(), heog.ravel())[0, 1])
        acc = decodifica(S[:, i, :], y, args.seed)
        filas.append((i, labels[i], probs[i], r, acc))
    # Ordenar por lo que importa: cuanto decodifica la clase.
    for i, lab, p, r, acc in sorted(filas, key=lambda t: -t[4])[:12]:
        print(f"{i:>3} {lab:16s} {p:6.2f} {r:9.3f} {acc:7.3f}")

    print(f"\nHEOG completo decodifica: {decodifica(heog[:, None, :], y, args.seed):.3f}")
    quitados = [i for i, lab, p, _, _ in filas
                if lab.lower() == "eye blink" and p >= 0.8]
    print(f"Se quitarian con umbral 0.8 sobre 'eye blink': {quitados}")

    print("\n--- LECTURA ---")
    top = sorted(filas, key=lambda t: -t[4])[:5]
    ojo_bajo = [(i, p) for i, lab, p, _, _ in top if lab.lower() == "eye blink" and p < 0.8]
    no_ojo = [(i, lab, a) for i, lab, p, _, a in top if lab.lower() != "eye blink"]
    if ojo_bajo:
        print(f"CAUSA (1): hay componentes 'eye blink' con prob <0.8 entre los que mas")
        print(f"  decodifican {ojo_bajo} -> BAJAR el umbral los captura.")
    if no_ojo:
        print(f"CAUSA (2): componentes que decodifican SIN etiqueta ocular:")
        for i, lab, a in no_ojo:
            print(f"    #{i} '{lab}' decodifica {a:.3f}")
        print("  Si ademas correlacionan con el HEOG, el artefacto esta repartido")
        print("  -> SUBIR n_components para que se separe en una fuente propia.")


if __name__ == "__main__":
    main()
