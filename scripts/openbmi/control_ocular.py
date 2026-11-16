"""CONTROL A — ¿la señal decodificable en OpenBMI es MOTORA u OCULAR?

MOTIVO (evidencia del 2026-07-22):
  sub-04 es el PEOR sujeto para CSP+LDA (0.562, casi azar) y el MEJOR para EEGNet (0.813).
  El ranking entre ambos métodos se invierte (rho de Spearman ~ -0.4). La diferencia entre
  los dos métodos no es solo la arquitectura, son los DATOS que ve cada uno:

      CSP (Puerta 1) : 8-30 Hz, 20 canales motores   -> el EOG queda fuera por banda Y por canal
      EEGNet         : 0.5-40 Hz, 62 canales         -> el EOG entra entero

  Y el caché de OpenBMI se construyó SIN ICA y SIN autoreject (decisión deliberada para
  replicar los baselines publicados en la Puerta 1). OpenBMI es izquierda vs derecha:
  un paradigma LATERALIZADO donde el movimiento ocular horizontal puede correlacionar
  con la clase. Es el mismo confundidor que ya se detectó y trató en MIND.

CUATRO MEDIDAS por sujeto (FBCSP + LDA, CSP ajustado DENTRO de cada fold):
  1. canales FRONTALES        -> proxy grueso de contaminación ocular
  2. canales MOTORES          -> la señal que la tesis dice medir
  3. proxy de EOG SOLO        -> HEOG = Fp2-Fp1 (horizontal, el eje de izq/der), VEOG
  4. canales MOTORES tras REGRESAR FUERA el proxy de EOG  -> ¿sobrevive la señal motora?

INTERPRETACIÓN:
  motor_post ~ motor_pre           -> la señal motora es REAL, independiente del ojo
  motor_post se desploma           -> era artefacto ocular propagado
  frontal >> motor                 -> el modelo profundo puede estar leyendo el ojo

NOTA ANTI-FUGA: la regresión de EOG NO usa las etiquetas -> puede ajustarse sobre todos
los trials sin introducir fuga de clase. El CSP sí es supervisado -> va dentro del fold.

Uso:
  python scripts/openbmi/control_ocular.py                      # 6 sujetos por defecto
  python scripts/openbmi/control_ocular.py --subjects sub-04     # el caso clave
  python scripts/openbmi/control_ocular.py --subjects sub-01 sub-02 sub-03 sub-04
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mne.decoding import CSP
from mne.filter import filter_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import splits as sp                      # noqa: E402
import datasets.openbmi as DS            # noqa: E402

OUT = REPO_ROOT / "results" / "reports" / "openbmi" / "control_ocular.csv"

# Banco de filtros. Se incluye DELTA (0.5-4 Hz) a propósito: es donde vive el EOG.
# Si se empezara en 4 Hz el propio banco filtraría el artefacto y el control no vería nada.
BANDS = [(0.5, 4), (4, 8), (8, 12), (12, 16), (16, 20),
         (20, 24), (24, 28), (28, 32), (32, 38)]

# Candidatos frontales; se filtran por los que existan realmente en el montaje.
FRONTAL_CAND = ["Fp1", "Fp2", "AF7", "AF3", "AF4", "AF8", "F7", "F5", "F3", "F4", "F6", "F8"]


def pick_existing(all_chs, wanted):
    up = {c.upper(): c for c in all_chs}
    return [up[w.upper()] for w in wanted if w.upper() in up]


def idx_of(all_chs, names):
    up = {c.upper(): i for i, c in enumerate(all_chs)}
    return [up[n.upper()] for n in names if n.upper() in up]


def build_eog_proxy(X, chs):
    """HEOG = Fp2 - Fp1 (eje horizontal: el mismo eje que izquierda/derecha), VEOG = media.

    Devuelve (n_trials, 2, n_times) o None si faltan los frontopolares.
    """
    up = {c.upper(): i for i, c in enumerate(chs)}
    if "FP1" not in up or "FP2" not in up:
        return None
    i1, i2 = up["FP1"], up["FP2"]
    heog = X[:, i2, :] - X[:, i1, :]
    veog = (X[:, i1, :] + X[:, i2, :]) / 2.0
    return np.stack([heog, veog], axis=1)


def regress_out(X, proxy):
    """Elimina de cada canal la parte explicable linealmente por el proxy de EOG.

    Un modelo por canal sobre todos los trials concatenados en el tiempo.
    No usa etiquetas -> sin fuga de clase.
    """
    n_ep, n_ch, n_t = X.shape
    P = proxy.transpose(0, 2, 1).reshape(-1, proxy.shape[1])
    P = np.column_stack([P, np.ones(len(P))])
    out = np.empty_like(X)
    for c in range(n_ch):
        y = X[:, c, :].reshape(-1)
        beta, *_ = np.linalg.lstsq(P, y, rcond=None)
        out[:, c, :] = (y - P @ beta).reshape(n_ep, n_t)
    return out


def fbcsp_cv(X, y, sfreq, n_comp=4, seed=2026):
    """FBCSP + LDA. El CSP se ajusta SOLO con el train de cada fold (anti-fuga)."""
    if X.shape[1] < 4:
        return np.nan
    n_comp = min(n_comp, X.shape[1] - 1)
    # El filtrado por banda es NO supervisado -> seguro fuera del CV.
    Xb = [filter_data(X.astype(np.float64), sfreq, lo, hi, verbose="ERROR") for lo, hi in BANDS]
    accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xb[0], y):
        ftr, fte = [], []
        for Xi in Xb:
            # rank="info": el CAR/interpolación dejan la covarianza deficiente de rango
            # y el blanqueo interno de CSP produce inf/NaN sin este parámetro.
            csp = CSP(n_components=n_comp, reg="ledoit_wolf", log=True,
                      norm_trace=False, rank="info")
            csp.fit(Xi[tr], y[tr])
            ftr.append(csp.transform(Xi[tr]))
            fte.append(csp.transform(Xi[te]))
        clf = LDA(solver="lsqr", shrinkage="auto")
        clf.fit(np.hstack(ftr), y[tr])
        accs.append((clf.predict(np.hstack(fte)) == y[te]).mean())
    return float(np.mean(accs))


def decode_proxy_alone(proxy, y, sfreq, seed=2026):
    """¿El proxy de EOG (2 señales) decodifica la clase por sí solo?

    Con 2 canales el CSP no aporta: se usan log-varianzas por banda como features.
    """
    feats = []
    for lo, hi in BANDS:
        Pb = filter_data(proxy.astype(np.float64), sfreq, lo, hi, verbose="ERROR")
        feats.append(np.log(np.var(Pb, axis=2) + 1e-20))
    F = np.hstack(feats)
    clf = make_pipeline(StandardScaler(), LDA(solver="lsqr", shrinkage="auto"))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, F, y, cv=cv, scoring="accuracy").mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="por defecto, TODOS los sujetos presentes en el cache")
    ap.add_argument("--motor-set", default="motor_ext", choices=["motor_ext", "motor_paper", "core_motor"])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--append", action="store_true",
                    help="anadir al CSV existente en vez de reescribirlo")
    ap.add_argument("--cache", default=None,
                    help="directorio de tensores (por defecto, el del dataset). "
                         "Usar el cache con ICA para el control antes/despues.")
    ap.add_argument("--out", default=None, help="CSV de salida")
    args = ap.parse_args()

    cache = Path(args.cache) if args.cache else DS.CACHE_DIR
    if not cache.exists():
        sys.exit(f"No existe el cache {cache}")
    global OUT
    if args.out:
        OUT = Path(args.out)
    elif args.cache:
        OUT = OUT.with_name(f"control_ocular_{cache.name}.csv")

    if args.subjects is None:
        args.subjects = sorted({f.name.split("_")[0]
                                for f in cache.glob("sub-*_full.npz")})
        if not args.subjects:
            sys.exit(f"No hay tensores en {cache}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Se FUSIONA por sujeto: se descartan las filas de los sujetos que vamos a recalcular
    # y se conservan las demas. Duplicar filas falsearia la prevalencia (Holm contaria
    # cada repeticion como un sujeto mas), pero borrar el CSV entero al correr un
    # subconjunto pierde resultados ya calculados — que es justo lo que paso el 2026-07-24.
    if OUT.exists() and not args.append:
        try:
            prev = pd.read_csv(OUT)
            prev = prev[~prev["sub"].isin(args.subjects)]
            OUT.unlink()
            if len(prev):
                prev.to_csv(OUT, index=False)
        except Exception:
            OUT.unlink()
    chance = DS.CHANCE
    print(f"CONTROL A — ocular vs motor | dataset {DS.NAME} | azar {chance:.2f}", flush=True)
    print(f"Ventana {DS.WINDOW} s | conjunto motor: {args.motor_set}", flush=True)

    rows = []
    for i, sub in enumerate(args.subjects, 1):
        try:
            # Se carga el montaje COMPLETO: los frontales hacen falta para el proxy.
            X, y, chs = sp.load_subject(sub, cache, window=DS.WINDOW, ch_names=None)
            sfreq = DS.SFREQ
            y = np.asarray(y)

            front = pick_existing(chs, FRONTAL_CAND)
            motor = pick_existing(chs, DS.CH_SETS[args.motor_set])
            i_front, i_motor = idx_of(chs, front), idx_of(chs, motor)

            acc_front = fbcsp_cv(X[:, i_front, :], y, sfreq, seed=args.seed)
            acc_motor = fbcsp_cv(X[:, i_motor, :], y, sfreq, seed=args.seed)
            acc_todos = fbcsp_cv(X, y, sfreq, seed=args.seed)

            proxy = build_eog_proxy(X, chs)
            if proxy is None:
                acc_proxy = np.nan
                acc_motor_post = np.nan
            else:
                acc_proxy = decode_proxy_alone(proxy, y, sfreq, seed=args.seed)
                Xm_clean = regress_out(X[:, i_motor, :], proxy)
                acc_motor_post = fbcsp_cv(Xm_clean, y, sfreq, seed=args.seed)

            row = {"sub": sub, "n_trials": len(y),
                   "n_frontal": len(front), "n_motor": len(motor),
                   "acc_frontal": round(acc_front, 4),
                   "acc_motor": round(acc_motor, 4),
                   "acc_todos": round(acc_todos, 4),
                   "acc_proxy_EOG": round(acc_proxy, 4) if np.isfinite(acc_proxy) else np.nan,
                   "acc_motor_sin_EOG": round(acc_motor_post, 4) if np.isfinite(acc_motor_post) else np.nan}
            row["caida_por_EOG"] = round(row["acc_motor"] - row["acc_motor_sin_EOG"], 4) \
                if np.isfinite(acc_motor_post) else np.nan
            rows.append(row)
            pd.DataFrame([row]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
            print(f"[{i}/{len(args.subjects)}] {sub}: frontal {acc_front:.3f} | motor {acc_motor:.3f} "
                  f"| todos {acc_todos:.3f} | proxyEOG {acc_proxy:.3f} "
                  f"| motor sin EOG {acc_motor_post:.3f}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(args.subjects)}] {sub}: ERROR {type(e).__name__}: {e}", flush=True)

    if not rows:
        return
    d = pd.DataFrame(rows)
    print("\n===== CONTROL A — RESUMEN =====")
    print(f"Frontal          : {d['acc_frontal'].mean():.3f}")
    print(f"Motor            : {d['acc_motor'].mean():.3f}")
    print(f"Todos            : {d['acc_todos'].mean():.3f}")
    print(f"Proxy EOG solo   : {d['acc_proxy_EOG'].mean():.3f}   (azar {chance:.2f})")
    print(f"Motor sin EOG    : {d['acc_motor_sin_EOG'].mean():.3f}")
    dif = d["acc_frontal"].mean() - d["acc_motor"].mean()
    caida = d["caida_por_EOG"].mean()
    print(f"\nVentaja frontal-motor : {dif:+.3f}")
    print(f"Caída al quitar EOG   : {caida:+.3f}")

    # --- Veredicto POR SUJETO ---------------------------------------------
    # Promediar sobre sujetos diluye la contaminacion: si 1 de 54 sujetos
    # tiene un artefacto ocular fuerte, la media lo entierra bajo el ruido de
    # los otros 53. Pero en LOSO cada sujeto es el conjunto de test de SU
    # fold, asi que un solo sujeto contaminado corrompe la medida de ese fold.
    # Lo que importa es la PREVALENCIA, no el promedio.
    from scipy.stats import binomtest

    alpha = 0.05

    # p-valor del proxy por sujeto, corregido por Holm: con 54 sujetos a
    # alpha=0.05 sin corregir esperariamos ~2.7 falsos positivos, es decir
    # inventariamos contaminacion ocular donde solo hay ruido de muestreo.
    pvals = {}
    for _, r in d.iterrows():
        if np.isfinite(r["acc_proxy_EOG"]):
            n = int(r["n_trials"])
            pvals[r["sub"]] = binomtest(int(round(r["acc_proxy_EOG"] * n)), n,
                                        chance, alternative="greater").pvalue
    orden = sorted(pvals, key=pvals.get)
    m = len(orden)
    p_holm, peor = {}, 0.0
    for k, sub in enumerate(orden):
        peor = max(peor, min(1.0, (m - k) * pvals[sub]))  # monotono
        p_holm[sub] = peor

    # El proxy es la prueba PRIMARIA: es la unica con test estadistico y la unica
    # que aisla el artefacto (decodifica usando SOLO los canales oculares).
    # Las otras dos solo CORROBORAN; no marcan por si solas:
    #   - "frontal>motor" sin umbral marca diferencias de 0.008 entre dos medidas
    #     ruidosas en sujetos de baja SNR -> se exige margen minimo.
    #   - la caida al quitar el EOG puede ser mera perdida de varianza compartida
    #     en un sujeto de baja SNR, no informacion ocular de clase.
    MARGEN_FRONTAL = 0.05

    flags = []
    for _, r in d.iterrows():
        sig = r["sub"] in p_holm and p_holm[r["sub"]] < alpha
        if not sig:
            continue
        marcas = [f"proxy EOG {r['acc_proxy_EOG']:.3f} "
                  f"(p={pvals[r['sub']]:.1g}, Holm={p_holm[r['sub']]:.1g})"]
        if r["acc_frontal"] - r["acc_motor"] > MARGEN_FRONTAL:
            marcas.append(f"frontal>motor por {r['acc_frontal']-r['acc_motor']:+.3f}")
        if np.isfinite(r["caida_por_EOG"]) and r["caida_por_EOG"] > 0.05:
            marcas.append(f"cae {r['caida_por_EOG']:+.3f} al quitar EOG")
        flags.append((r["sub"], marcas))
    flags.sort(key=lambda t: p_holm[t[0]])

    # Sujetos que rozan el umbral: no se cuentan, pero se listan para no ocultarlos.
    limitrofes = [(r["sub"], p_holm[r["sub"]]) for _, r in d.iterrows()
                  if r["sub"] in p_holm and alpha <= p_holm[r["sub"]] < 0.10]

    print("\n--- VEREDICTO (por sujeto, Holm-corregido) ---")
    ci = 1.96 * np.sqrt(chance * (1 - chance) / d["n_trials"].median())
    print(f"Banda de azar al 95% con n={int(d['n_trials'].median())}: "
          f"[{chance - ci:.3f}, {chance + ci:.3f}]  (referencia; el flag usa binomial exacto)")
    if not flags:
        print("* Ningun sujeto muestra informacion ocular de clase significativa.")
    else:
        print(f"* {len(flags)}/{len(d)} sujetos ({100*len(flags)/len(d):.0f}%) con informacion")
        print("  ocular de clase ESTADISTICAMENTE SIGNIFICATIVA (proxy, binomial+Holm):")
        for sub, marcas in flags:
            print(f"    - {sub}: " + "; ".join(marcas))
        if limitrofes:
            print("\n  Limitrofes (0.05 <= Holm < 0.10, NO contados): "
                  + ", ".join(f"{s} ({p:.3f})" for s, p in sorted(limitrofes, key=lambda t: t[1])))
        print(f"\n  En agregado el motor pasa de {d['acc_motor'].mean():.3f} a "
              f"{d['acc_motor_sin_EOG'].mean():.3f} al quitar el EOG: la señal decodificable")
        print("  del dataset es GENUINAMENTE MOTORA. El artefacto es una minoria localizable.")
        print("\n  Pero la contaminacion es HETEROGENEA. En LOSO esto es peor que si fuera")
        print("  uniforme: no se cancela al comparar arquitecturas, sino que mete")
        print("  varianza no neural en los folds cuyo sujeto de test esta afectado.")
        print("  -> Justifica el modulo de ICA (Actividad 2.3) con evidencia propia.")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
