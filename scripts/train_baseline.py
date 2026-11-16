"""Entrenamiento del baseline (Actividades 3.1-3.5) — GENÉRICO por dataset y arquitectura.

Todo lo específico del dataset (clases, canales, ventana, azar, rutas) viene de
`src/datasets/<nombre>.py`; nada está fijado aquí. Eso es lo que permite que MIND y
OpenBMI se entrenen con EXACTAMENTE el mismo código, requisito de la comparación justa.

Modos:
  smoke       1 sujeto, 1 fold, 1 semilla. Verifica el pipeline end-to-end y MIDE EL
              TIEMPO de una corrida antes de lanzar un lote (regla de HANDOFF.md §3.bis).
  smoke-loso  1 fold LOSO. Diagnostica si el modelo generaliza con datos suficientes.
  sanity      Sobreajuste deliberado sobre pocos trials: separa un BUG de plomería de un
              problema real de señal. Hacerlo ANTES de tocar hiperparámetros ahorra días.
  compare-norm  Compara las 3 normalizaciones sobre varios folds LOSO.
  within      Monousuario (Anexo B.1).
  loso        Multiusuario, un fold por sujeto (Anexo B.1/B.2).

Escritura incremental y reanudable en results/reports/<dataset>/baseline_<modelo>_...csv:
si se interrumpe, al relanzar salta las corridas ya registradas (columna run_id).

Uso:
  python scripts/train_baseline.py --dataset openbmi --mode smoke
  python scripts/train_baseline.py --dataset openbmi --mode within --norm global
  python scripts/train_baseline.py --dataset mind    --mode loso   --channels completo
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import datasets
import splits as sp
import training as tr

# LAS MISMAS 5 SEMILLAS PARA LAS 4 ARQUITECTURAS. Es una constante de modulo, no un
# parametro por modelo, justamente para que no pueda divergir entre arquitecturas: si dos
# modelos se evaluaran sobre inicializaciones distintas, la comparacion entre ellos dejaria
# de ser pareada y el contraste de Wilcoxon perderia sentido.
# 2026-2028 son las tres originales (ya ejecutadas en within de EEGNet y Conformer, y en
# todos los LOSO la 2026). 2758 y 9779 se sortearon UNA vez con `secrets.randbelow` el
# 2026-09-05, para que las dos nuevas no sean consecutivas ni elegidas a mano.
SEEDS = [2026, 2027, 2028, 2758, 9779]


class Progreso:
    """Contador de progreso para los lotes: imprime [hechas/total  %  transcurrido  ETA].

    `i` avanza en CADA corrida planificada (también las que se saltan por estar ya hechas,
    para que el % refleje el avance real). El ETA se estima solo con las corridas realmente
    EJECUTADAS en esta sesión (las saltadas no consumen tiempo), así que es honesto al reanudar.
    """
    def __init__(self, total: int):
        self.total = total
        self.i = 0
        self.n_exec = 0
        self.t0 = time.time()

    def saltada(self):
        self.i += 1

    def ejecutada(self) -> str:
        self.i += 1
        self.n_exec += 1
        transcurrido = time.time() - self.t0
        por_corrida = transcurrido / self.n_exec
        eta = por_corrida * (self.total - self.i)
        return (f"[{self.i}/{self.total} {100 * self.i / self.total:3.0f}%  "
                f"transcurrido {transcurrido / 60:4.1f}min  ETA ~{eta / 60:4.0f}min]")

# --------------------------------------------------------------------------- #
# Estado activo: DATASET y MODELO. Los fija main() a partir del CLI.
# Todo lo específico del dataset (clases, canales, ventana, azar, rutas) viene del
# módulo correspondiente en src/datasets/, nunca hardcodeado aquí: así este script
# entrena cualquier dataset con el MISMO código, que es lo que exige la comparación justa.
# --------------------------------------------------------------------------- #
DS = None               # módulo del dataset (datasets.mind / datasets.openbmi)
MODEL = "eegnet"
CACHE = None
CKPT_ROOT = None
REPORTS = None
N_CLASSES = None
WINDOW = None
SFREQ = None
# Sufijo que distingue configuraciones del MISMO modelo que, si no, colisionarían en el
# CSV y el run_id (y la reanudación saltaría la segunda). Solo se usa en LaBraM, donde
# full/linear y las variantes de escalado son ejecuciones distintas que hay que separar.
# Vacío para EEGNet/Conformer -> sus nombres de fichero y run_id quedan INTACTOS.
RUN_TAG = ""
# Avance 7 (octubre de 2026). VARIANTE se anade al final de RUN_TAG ("_v<nombre>") para
# que una variante (otra ventana, otra aumentacion) no comparta run_id, CSV ni checkpoint
# con la base. Vacia por defecto: la base conserva EXACTAMENTE sus nombres.
VARIANTE = ""
# Callback opcional tras evaluar cada fold LOSO (calibracion con pocos ensayos).
FEWSHOT = False
# Ruta del CSV de metricas del barrido en curso. La fijan los barridos para que
# `_run_one` sepa donde dejar la curva de perdida, sin cambiar su firma (la llaman
# cinco sitios distintos).
LOG_PATH = None


def _configurar(dataset: str, model: str, cache: str | None = None):
    global DS, MODEL, CACHE, CKPT_ROOT, REPORTS, N_CLASSES, WINDOW, SFREQ, RUN_TAG
    DS = datasets.get(dataset)
    MODEL = model
    # LaBraM: separar full/linear, la escala de amplitud y el lr en CSV y run_id distintos
    # (el lr NO va en el run_id base, así que sin esto un re-run con otro lr se saltaría).
    RUN_TAG = ""
    if model == "labram":
        ft = os.environ.get("LABRAM_FINETUNE", "full").lower()
        uv = os.environ.get("LABRAM_UV_SCALE", "1")
        lr = tr.DEFAULT_TRAIN_CFG["lr"]          # ya incorpora el override de --lr (main lo aplica antes)
        RUN_TAG = f"_{ft}_uv{uv}_lr{lr:g}"
    if model == "eegmamba":
        # EEGMamba desde cero: la cabeza (avgpool/flatten), la escala y el lr definen la
        # variante del gate; sin esto un re-run con otra cabeza pisaría el CSV anterior.
        head = os.environ.get("EEGMAMBA_HEAD", "avgpool").lower()
        uv = os.environ.get("EEGMAMBA_UV_SCALE", "1")
        lr = tr.DEFAULT_TRAIN_CFG["lr"]
        RUN_TAG = f"_{head}_uv{uv}_lr{lr:g}"
    if VARIANTE:
        RUN_TAG += f"_v{VARIANTE}"
    CACHE = Path(cache) if cache else DS.CACHE_DIR
    if not CACHE.exists():
        raise SystemExit(f"No existe el cache {CACHE}")
    # Checkpoints separados por cache: entrenar sobre datos limpios y sucios comparte
    # nombre de modelo pero NO debe compartir pesos ni reanudarse cruzado.
    #
    # TESIS_CKPT existe porque en el pod `/workspace` es un FUSE de red: se escribe un
    # checkpoint POR EPOCA, y con ~60 epocas x 20 folds x 54 sujetos son decenas de miles
    # de escrituras pequenas con latencia de red. Medido el 2026-07-27: una vez que el
    # entrenamiento bajo a ~5 s/fold, esa escritura pasó a dominar el tiempo total (el
    # proceso se quedaba en `request_wait_answer`, con GPU y CPU al 0 %).
    # Los checkpoints son PRESCINDIBLES —solo sirven para reanudar; lo que se conserva son
    # los CSV de metricas— asi que van al disco local, que va ~10x mas rapido.
    raiz_ckpt = Path(os.environ.get("TESIS_CKPT", CACHE.parent / "checkpoints"))
    CKPT_ROOT = raiz_ckpt / f"{model}_{CACHE.name}"
    REPORTS = REPO_ROOT / "results" / "reports" / DS.NAME
    REPORTS.mkdir(parents=True, exist_ok=True)
    N_CLASSES, SFREQ = DS.N_CLASSES, DS.SFREQ
    WINDOW = WINDOW or DS.WINDOW          # --window lo fija antes; si no, la del dataset

# Claves de DEFAULT_TRAIN_CFG que NO afectan al entrenamiento y por tanto NO deben mover el
# hash: cambiarlas no cambia ni un peso, solo el ritmo de escritura a disco o de log. Sin esta
# exclusion, anadir `ckpt_cada` movio el hash de eabca26d5bdf a a59e0e313eaf y dejo las semillas
# nuevas etiquetadas como si fueran otra configuracion que las ya ejecutadas.
_CLAVES_SIN_EFECTO = {"ckpt_cada", "log_cada"}


def _calcular_cfg_hash() -> str:
    """Huella de la configuracion de entrenamiento REALMENTE usada.

    OJO con el momento en que se llama. Antes esto era una constante evaluada al importar el
    modulo, o sea ANTES de que `main()` aplicara `--lr`, `--batch` o `--max-epochs` (lineas
    ~647). El resultado es que esos overrides NUNCA entraron en el hash: dos ejecuciones con
    learning rates distintos quedaban registradas con el mismo `cfg_hash`, que es justo la
    confianza falsa que el hash pretendia dar. Ahora `main()` lo recalcula DESPUES de aplicar
    los argumentos, asi que la columna del CSV identifica de verdad la configuracion.
    """
    return hashlib.sha256(json.dumps(
        {k: v for k, v in tr.DEFAULT_TRAIN_CFG.items() if k not in _CLAVES_SIN_EFECTO},
        sort_keys=True).encode()).hexdigest()[:12]


CFG_HASH = _calcular_cfg_hash()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _argv_para_worker(quitar=("--workers", "--loso-folds", "--subjects", "--log-tag")):
    """Reconstruye los argumentos de CLI quitando los que el repartidor vuelve a fijar.

    Se consume el valor (o los valores, para los `nargs="*"`) de cada opcion retirada:
    todo lo que siga hasta el proximo token que empiece por `--`.
    """
    fuera, i = [], 1
    while i < len(sys.argv):
        a = sys.argv[i]
        base = a.split("=", 1)[0]
        if base in quitar:
            i += 1
            if "=" not in a:
                while i < len(sys.argv) and not sys.argv[i].startswith("--"):
                    i += 1
            continue
        fuera.append(a)
        i += 1
    return fuera


def _repartir(n_workers: int, log_path: Path, unidades: list, flag: str, tag_padre=None):
    """Ejecuta N copias de este script EN PARALELO sobre la misma GPU y fusiona los CSV.

    Por que sirve en una sola GPU: los modelos son pequenos y el lote es de 32, asi que un
    proceso solo deja la GPU en buena parte ociosa (en within, EEGNet tarda 3,6 s por corrida
    sobre 144 trials: casi todo es sobrecoste de lanzamiento de kernels, no calculo). Varios
    procesos intercalan su trabajo y llenan los huecos del otro. No cambia ningun resultado:
    los folds son independientes y cada `run_id` es unico.

    El reparto es INTERCALADO (fold i -> worker i % N), no por bloques, para que todos los
    trabajadores terminen a la vez aunque unos sujetos cuesten mas que otros.

    Ojo con la VRAM: cada proceso sube su train a la GPU y `train_run` EXIGE
    (X_train.nbytes + X_val.nbytes) * 1.5 libres, que en LOSO son ~5,7 GB, mas ~0,6 GB del
    contexto CUDA de cada proceso. Cuenta con ~6,3 GB por worker, no con 4. Si no cabe,
    `train_run` degrada al pipeline de CPU (~9x mas lento) y solo lo avisa en el log; usa
    TESIS_EXIGIR_GPU=1 para que muera en vez de arrastrarse.

    `tag_padre` COMPONE la etiqueta: si el usuario ya paso `--log-tag bench`, el `log_path`
    del padre acaba en `__bench` y los hijos tienen que escribir `__bench__w0`, `__bench__w1`,
    etc. Con la etiqueta suelta (`__w0`) la fusion no los encontraria (busca
    `{log_path.stem}__*.csv`) y, peor, quedarian sueltos fuera del fichero al que pertenecen.
    """
    import subprocess
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from merge_shards import fusionar

    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    comun = _argv_para_worker()
    procs = []
    for w in range(n_workers):
        trozo = unidades[w::n_workers]
        if not trozo:
            continue
        tag = f"{tag_padre}__w{w}" if tag_padre else f"w{w}"
        cmd = ([sys.executable, str(Path(sys.argv[0]).resolve())] + comun
               + [flag] + [str(u) for u in trozo] + ["--log-tag", tag])
        log = logs_dir / f"{log_path.stem}__w{w}.log"
        print(f"  worker {w}: {len(trozo)} unidades -> {log}", flush=True)
        procs.append((w, subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT)))
        # ARRANQUE ESCALONADO. `train_run` decide si cabe en la GPU leyendo `mem_get_info`, y
        # esa lectura ocurre ANTES de reservar. Lanzados a la vez, los N workers miden la VRAM
        # libre inicial, todos concluyen que caben, y luego se pelean por la memoria. El retardo
        # deja que cada uno reserve su train antes de que el siguiente mida. Tambien reparte el
        # pico de lectura del cache en disco. El costo es despreciable: N x 8 s de una vez, sobre
        # ejecuciones de horas.
        if w < n_workers - 1:
            time.sleep(float(os.environ.get("TESIS_ESCALON_S", "8")))

    fallos = [w for w, pr in procs if pr.wait() != 0]
    if fallos:
        print(f"\n[AVISO] workers con error: {fallos}. "
              f"Revisa logs/{log_path.stem}__w*.log")
        print("Se fusiona igualmente lo que si termino; relanzar reanuda lo que falte.")
    print()
    fusionar(log_path)
    # Las curvas viven en su propio directorio y tienen su propio juego de shards.
    for base in (_curvas_path(log_path), _fewshot_path(log_path)):
        if list(base.parent.glob(f"{base.stem}__*.csv")):
            fusionar(base)
    if fallos:
        sys.exit(1)


def _load_log(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame()


def _curvas_path(log_path: Path) -> Path:
    """CSV con la curva por epoca, en el subdirectorio `curvas/`.

    Va en subdirectorio y NO como hermano `<base>_curvas.csv` a proposito: `merge_shards`
    busca los shards con el patron `<base>__*.csv`, que tambien casaria con
    `<base>__w0_curvas.csv` y acabaria mezclando filas de curvas dentro del CSV de
    metricas. Separandolos por directorio el patron no puede colisionar, y ademas las
    curvas se fusionan con el mismo mecanismo aplicado a su propia carpeta.
    """
    return log_path.parent / "curvas" / f"{log_path.name}"


def _escribir_curva(log_path: Path, run_id: str, history: list):
    """Vuelca la curva de perdida por epoca, en modo APPEND.

    Por que existe: el CSV de metricas solo guarda el resultado final (acc, kappa,
    n_epochs), asi que la evolucion de train_loss y val_loss quedaba unicamente en el .log
    del pod y en el campo `history` del checkpoint, ambos volatiles. De los cuatro
    entrenamientos base solo sobreviven las curvas de EEGNet, porque se bajo su .log a
    mano; las de Conformer, LaBraM y EEGMamba se perdieron al destruir los pods. Sin la
    curva no se puede argumentar convergencia ni sobreajuste, que es justo lo que hace
    falta para explicar por que el protocolo intra-sujeto satura.

    Se escribe en APPEND y no reescribiendo el fichero entero como `_append_row`: son
    decenas de miles de filas (1080 ejecuciones x ~57 epocas en within) y reescribir en
    cada una seria cuadratico. El riesgo de esquema desalineado que motivo aquella
    reescritura no aplica aqui, porque estas columnas son fijas.
    """
    if not history:
        return
    ruta = _curvas_path(log_path)
    es_nuevo = not ruta.exists()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if es_nuevo:
            w.writerow(["run_id", "epoch", "train_loss", "val_loss"])
        for h in history:
            w.writerow([run_id, h.get("epoch"), h.get("train_loss"), h.get("val_loss")])


def _append_row(path: Path, row: dict):
    """Añade una fila al log, TOLERANTE A CAMBIOS DE ESQUEMA.

    Un `mode="a"` ciego escribe los valores en el orden del dict, sin mirar la cabecera
    ya existente: si el esquema cambió entre ejecuciones (p. ej. al añadir las columnas
    `model`/`n_params`/`best_val_loss`), las filas nuevas quedan desalineadas respecto a
    la cabecera y el CSV se vuelve ilegible. Aquí se reescribe el archivo completo con la
    UNIÓN de columnas, rellenando con NaN lo que falte en las filas antiguas.
    Los logs son de cientos de filas, así que el coste es irrelevante frente al riesgo.
    """
    nueva = pd.DataFrame([row])
    if path.exists():
        try:
            previa = pd.read_csv(path)
            nueva = pd.concat([previa, nueva], ignore_index=True)
        except Exception:
            # CSV ya corrupto de una ejecución anterior: se preserva y se empieza limpio
            path.replace(path.with_suffix(".corrupto.csv"))
    tmp = path.with_suffix(".tmp")
    nueva.to_csv(tmp, index=False)
    tmp.replace(path)       # escritura atómica


def _borrar_ckpt(ckpt_path: Path):
    """Borra el checkpoint de una ejecucion within/loso YA registrada en el CSV.

    Solo con TESIS_BORRAR_CKPT=1. En la avance 7 (octubre de 2026) son unos 900
    checkpoints LOSO y 9000 within; los de LaBraM pesan ~70 MB con el optimizador, lo que
    no cabe en el disco del pod. Una vez escrita la fila, el checkpoint ya no sirve para
    reanudar (la reanudacion salta por run_id) y la calibracion con pocos ensayos se hace
    en memoria antes de borrarlo. Los de `--mode all` NO se borran: los usa eval_cruzado.
    """
    if os.environ.get("TESIS_BORRAR_CKPT") == "1":
        Path(ckpt_path).unlink(missing_ok=True)


def _already_done(done_df: pd.DataFrame, run_id: str) -> bool:
    if done_df.empty or "run_id" not in done_df.columns:
        return False
    return bool((done_df["run_id"].astype(str) == run_id).any())


def _cfg_run(ch_names):
    """Configuracion de UNA ejecucion: la global mas lo que depende de los canales.

    La reflexion de canales necesita la permutacion de hemisferios, que depende de la
    vista de canales cargada. Se anade aqui y no en DEFAULT_TRAIN_CFG para no mover el
    cfg_hash con una lista larga: el hash ya distingue la variante por
    `channel_reflection_p`, y la permutacion se deduce de los nombres sin ambiguedad.
    """
    cfg = tr.DEFAULT_TRAIN_CFG
    if cfg["augment"].get("channel_reflection_p", 0):
        import variantes as va
        if ch_names is None:
            raise SystemExit("la reflexion de canales necesita los nombres de los canales")
        perm, sin_par = va.pares_reflexion(ch_names)
        cfg = json.loads(json.dumps(cfg))
        cfg["augment"]["cr_perm"] = perm
        cfg["augment"]["cr_perm_clases"] = va.permutacion_clases(DS.CLASS_ORDER)
        if sin_par and not getattr(_cfg_run, "_avisado", False):
            print(f"  [reflexion] canales sin pareja, se quedan en su sitio: {sin_par}",
                  flush=True)
            _cfg_run._avisado = True
    return cfg


def _run_one(run_id: str, scenario: str, meta: dict, n_chans: int, n_times: int, seed: int,
             X_train, y_train, X_val, y_val, X_test, y_test, device, ckpt_path: Path,
             X_visto=None, y_visto=None, ch_names=None, post=None):
    tr.set_seed(seed)
    # ch_names solo lo usa LaBraM (mapeo a embeddings posicionales); el resto lo ignora.
    model = tr.build_model(MODEL, n_chans=n_chans, n_times=n_times, n_classes=N_CLASSES,
                           ch_names=ch_names)
    t0 = time.time()
    history, _ = tr.train_run(model, X_train, y_train, X_val, y_val, N_CLASSES, SFREQ,
                               _cfg_run(ch_names), device, ckpt_path, seed=seed)
    metrics = tr.evaluate(model, X_test, y_test, device, N_CLASSES)
    if post is not None:
        post(model)
    if LOG_PATH is not None:
        _escribir_curva(LOG_PATH, run_id, history)
    elapsed = round(time.time() - t0, 1)
    row = {
        "run_id": run_id, "scenario": scenario, "seed": seed,
        "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test),
        "n_epochs": len(history), "elapsed_s": elapsed,
        "acc": metrics["acc"], "kappa": metrics["kappa"], "f1_macro": metrics["f1_macro"],
        "confusion_matrix": metrics["confusion_matrix"],
        "model": MODEL, "n_params": sum(p.numel() for p in model.parameters()),
        "ckpt_path": str(ckpt_path), "cfg_hash": CFG_HASH,
    }
    # Conjunto "SUJETO VISTO": trials reservados de los sujetos de ENTRENAMIENTO. Mismo
    # modelo, mismos datos de entrenamiento, misma cantidad de parámetros; lo único que
    # cambia frente a `X_test` es si la persona estaba o no en el entrenamiento. Es una
    # sola pasada hacia adelante, así que el coste es despreciable.
    if X_visto is not None and len(y_visto):
        mv = tr.evaluate(model, X_visto, y_visto, device, N_CLASSES)
        row.update({"n_visto": len(y_visto), "acc_visto": mv["acc"],
                    "kappa_visto": mv["kappa"], "f1_visto": mv["f1_macro"]})
        # DSI = degradación relativa al pasar de sujeto conocido a sujeto nuevo.
        # Se reporta también sobre kappa porque es la métrica acordada para el indicador.
        for m in ("acc", "kappa"):
            base = mv[m]
            row[f"dsi_{m}"] = round((base - metrics[m]) / base, 4) if base > 1e-9 else np.nan
    row.update(meta)
    return row


def _list_subjects():
    return sorted(p.name.replace("_full.npz", "") for p in CACHE.glob("sub-*_full.npz"))


# --------------------------------------------------------------------------- #
# Within-subject (monousuario, Anexo B.1: 5-fold CV por sujeto)
# --------------------------------------------------------------------------- #

def within_subject_all(channels: str, device, log_path: Path, subs=None, n_folds: int = 5,
                       only_folds=None, seeds=None, norm="global", by_session=False):
    """only_folds / seeds: si se pasan, restringen el barrido (lo usa el smoke test).

    by_session=True: 10-fold POR SESIÓN sin mezclar (protocolo OpenBMI, §10). Cada sujeto
    produce dos métricas (una por sesión), etiquetadas en la columna `session` del CSV.
    """
    global LOG_PATH
    LOG_PATH = log_path
    done_df = _load_log(log_path)
    subs = subs or _list_subjects()
    seeds = seeds or SEEDS
    ch_view = DS.CH_SETS[channels]
    n_f = len(only_folds) if only_folds is not None else (n_folds * 2 if by_session else n_folds)
    prog = Progreso(len(subs) * n_f * len(seeds))
    print(f"Within-subject: {len(subs)} sujetos x {n_f} folds x {len(seeds)} semillas "
          f"= {prog.total} corridas "
          f"(canales={channels}, {'POR SESION' if by_session else 'mezcla sesiones'})", flush=True)

    for sub in subs:
        X, y_txt, chs = sp.load_subject(sub, CACHE, window=WINDOW, ch_names=ch_view)
        y = tr.encode_labels(y_txt, DS.CLASS_ORDER)
        if by_session:
            sess = np.load(CACHE / f"{sub}_full.npz", allow_pickle=True)["session"]
            folds = sp.make_within_session_folds(y, sess, n_folds=n_folds, seed=2026)
        else:
            folds = sp.make_within_subject_folds(y, n_folds=n_folds, seed=2026)
        if only_folds is not None:
            folds = [f for f in folds if f.idx in only_folds]
        for fold in folds:
            built = sp.build_within_fold(fold, X, y, normalize=norm)
            ses_tag = f"_s{fold.session}" if fold.session is not None else ""
            for seed in seeds:
                run_id = f"within_{sub}{ses_tag}_fold{fold.idx}_seed{seed}_{channels}_{built['norm']}{RUN_TAG}"
                if _already_done(done_df, run_id):
                    prog.saltada()
                    continue
                ckpt_path = CKPT_ROOT / "within" / f"{run_id}.pt"
                meta = {"sub": sub, "fold": fold.idx, "channels": channels, "norm": built["norm"]}
                if fold.session is not None:
                    meta["session"] = fold.session
                row = _run_one(
                    run_id, "within", meta,
                    n_chans=X.shape[1], n_times=built["X_train"].shape[2], seed=seed,
                    X_train=built["X_train"], y_train=built["y_train"],
                    X_val=built["X_val"], y_val=built["y_val"],
                    X_test=built["X_test"], y_test=built["y_test"],
                    device=device, ckpt_path=ckpt_path, ch_names=chs,
                )
                _append_row(log_path, row)
                _borrar_ckpt(ckpt_path)
                done_df = pd.concat([done_df, pd.DataFrame([row])], ignore_index=True)
                print(f"  {prog.ejecutada()} {run_id}: acc={row['acc']:.3f} "
                      f"kappa={row['kappa']:.3f} f1={row['f1_macro']:.3f} "
                      f"({row['elapsed_s']}s, {row['n_epochs']} ep)", flush=True)


# --------------------------------------------------------------------------- #
# LOSO (multiusuario, Anexo B.1/B.2)
# --------------------------------------------------------------------------- #

def loso_all(channels: str, device, log_path: Path, only_folds=None, seeds=None, norm="global",
             frac_visto: float = 0.10):
    """only_folds / seeds: si se pasan, restringen el barrido (lo usa el smoke test).

    frac_visto: fracción de trials reservada de cada sujeto de entrenamiento para medir
    el desempeño sobre SUJETO VISTO. Con el mismo modelo y los mismos datos, la única
    diferencia frente al sujeto excluido es si la persona era conocida -> aísla el cambio
    de dominio del efecto del tamaño del conjunto de entrenamiento.
    """
    global LOG_PATH
    LOG_PATH = log_path
    done_df = _load_log(log_path)
    subs = _list_subjects()
    seeds = seeds or SEEDS
    folds = sp.make_loso_folds(subs)
    if only_folds is not None:
        folds = [f for f in folds if f.idx in only_folds]
    ch_view = DS.CH_SETS[channels]
    prog = Progreso(len(folds) * len(seeds))
    print(f"LOSO: {len(folds)} folds x {len(seeds)} semillas = {prog.total} corridas "
          f"(canales={channels})", flush=True)

    for fold in folds:
        built = sp.build_fold(fold, CACHE, window=WINDOW, ch_names=ch_view, normalize=norm,
                              frac_visto=frac_visto)
        y_train = tr.encode_labels(built["y_train"], DS.CLASS_ORDER)
        y_val = tr.encode_labels(built["y_val"], DS.CLASS_ORDER)
        y_test = tr.encode_labels(built["y_test"], DS.CLASS_ORDER)
        X_visto = built.get("X_visto")
        y_visto = (tr.encode_labels(built["y_visto"], DS.CLASS_ORDER)
                   if X_visto is not None else None)
        if X_visto is not None:
            print(f"  fold {fold.idx} ({fold.test}): train {len(y_train)} | "
                  f"visto {len(y_visto)} | no visto {len(y_test)}", flush=True)
        for seed in seeds:
            run_id = f"loso_{fold.test}_seed{seed}_{channels}_{built['norm']}{RUN_TAG}"
            if _already_done(done_df, run_id):
                prog.saltada()
                continue
            ckpt_path = CKPT_ROOT / "loso" / f"{run_id}.pt"
            post = None
            if FEWSHOT:
                post = _post_fewshot(log_path, run_id, fold, seed, built, y_test, device)
            row = _run_one(
                run_id, "loso",
                {"test_sub": fold.test, "fold": fold.idx, "channels": channels, "norm": built["norm"]},
                n_chans=built["X_train"].shape[1], n_times=built["X_train"].shape[2], seed=seed,
                X_train=built["X_train"], y_train=y_train,
                X_val=built["X_val"], y_val=y_val,
                X_test=built["X_test"], y_test=y_test,
                device=device, ckpt_path=ckpt_path,
                X_visto=X_visto, y_visto=y_visto, ch_names=built["ch_names"], post=post,
            )
            _append_row(log_path, row)
            _borrar_ckpt(ckpt_path)
            done_df = pd.concat([done_df, pd.DataFrame([row])], ignore_index=True)
            print(f"  {prog.ejecutada()} {run_id}: acc={row['acc']:.3f} "
                  f"kappa={row['kappa']:.3f} f1={row['f1_macro']:.3f} "
                  f"({row['elapsed_s']}s, {row['n_epochs']} ep)", flush=True)


def _fewshot_path(log_path: Path) -> Path:
    """CSV de la calibracion con pocos ensayos, en su subdirectorio (igual que las curvas)."""
    return log_path.parent / "fewshot" / log_path.name


def _post_fewshot(log_path, run_id, fold, seed, built, y_test, device):
    """Devuelve el callback que calibra el modelo LOSO de este fold con k ensayos.

    Escribe en APPEND, y antes de escribir comprueba que el run_id no este ya: si el
    proceso cae entre la fila de calibracion y la fila principal, al relanzar el modelo
    se recarga del checkpoint terminado y la calibracion se repetiria.
    """
    import fewshot as fs

    def post(model):
        ruta = _fewshot_path(log_path)
        if ruta.exists() and (pd.read_csv(ruta, usecols=["run_id"])["run_id"] == run_id).any():
            return
        t0 = time.time()
        filas = fs.evaluar(model, built["X_test"], y_test, built["stats"], device,
                           N_CLASSES, seed)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(filas).assign(run_id=run_id, test_sub=fold.test, seed=seed,
                                        model=MODEL, norm=built["norm"], cfg_hash=CFG_HASH)
        df.to_csv(ruta, mode="a", header=not ruta.exists(), index=False)
        res = df.groupby(["k", "metodo"])["acc"].mean().unstack()
        print(f"    few-shot ({time.time() - t0:.0f}s): " + " | ".join(
            f"k={k} " + " ".join(f"{m}={v:.3f}" for m, v in fila.items())
            for k, fila in res.iterrows()), flush=True)
    return post


# --------------------------------------------------------------------------- #
# Modelo sobre TODOS los sujetos (punto 6: validacion en un segundo dataset)
# --------------------------------------------------------------------------- #

def all_subjects(channels: str, device, log_path: Path, norm="global", seeds=None,
                 frac_val: float = 0.10, frac_test: float = 0.0,
                 holdout_n: int = 0, holdout_seed: int = sp.SEMILLA_HOLDOUT):
    """Entrena UN modelo por semilla con los 54 sujetos y guarda los pesos.

    No reporta exactitud de prueba porque aqui no hay conjunto de prueba: el modelo se
    entrena para evaluarse en OTRO dataset (ver scripts/eval_cruzado.py). Lo unico que
    se mide es la perdida de validacion, que es la que decide la parada temprana.

    Junto al checkpoint se escribe un `.meta.json` con los canales y los estadisticos de
    normalizacion del entrenamiento. Sin eso el checkpoint es inservible fuera de aqui:
    no se sabria en que orden van los canales ni con que media y desviacion se
    normalizo, y aplicar otros convertiria la transferencia en algo no comparable.
    """
    global LOG_PATH
    LOG_PATH = log_path
    done_df = _load_log(log_path)
    subs = _list_subjects()
    # Con holdout_n > 0 se apartan SUJETOS ENTEROS, que es lo unico que permite medir
    # "persona nueva, mismo protocolo" con los mismos pesos que se llevan al otro dataset.
    # La lista no se guarda: se recalcula con sp.separar_sujetos(subs, n, semilla).
    apartados = []
    if holdout_n:
        subs, apartados = sp.separar_sujetos(subs, holdout_n, holdout_seed)
        print(f"HOLDOUT: {len(apartados)} sujetos fuera del entrenamiento "
              f"(semilla {holdout_seed}): {' '.join(apartados)}", flush=True)
    seeds = seeds or SEEDS
    ch_view = DS.CH_SETS[channels]
    built = sp.build_all(subs, CACHE, window=WINDOW, ch_names=ch_view, normalize=norm,
                         frac_val=frac_val, frac_test=frac_test)
    y_train = tr.encode_labels(built["y_train"], DS.CLASS_ORDER)
    y_val = tr.encode_labels(built["y_val"], DS.CLASS_ORDER)
    # El reservado solo existe con --frac-test: es el conjunto COMPARTIDO sobre el que se
    # mide tambien el modelo transferido desde el otro dataset (ver scripts/eval_cruzado.py).
    X_res = built.get("X_test")
    y_res = tr.encode_labels(built["y_test"], DS.CLASS_ORDER) if X_res is not None else None
    prog = Progreso(len(seeds))
    print(f"ALL: {len(subs)} sujetos, train {len(y_train)} | val {len(y_val)}"
          + (f" | reservado {len(y_res)}" if X_res is not None else "")
          + f" | {len(seeds)} semillas (canales={channels}, {len(built['ch_names'])} ch)",
          flush=True)

    for seed in seeds:
        # El DATASET va en el run_id. Sin el, entrenar la misma arquitectura sobre dos
        # conjuntos distintos produce el mismo identificador y, como la ruta del
        # checkpoint es {TESIS_CKPT}/{modelo}_{nombre_cache}/all/{run_id}.pt y los dos
        # caches se llaman `epochs_cache_ica`, el segundo entrenamiento CARGA los pesos
        # del primero y devuelve sin entrenar. Paso el 2026-09-06: el modelo nativo de
        # WBCIC "termino" en 1,7 s con los pesos de OpenBMI dentro.
        # El holdout tambien va en el run_id: un modelo de 44 sujetos y uno de 54 son
        # modelos DISTINTOS y no pueden compartir checkpoint ni fila de CSV.
        hold = f"_hold{holdout_n}s{holdout_seed}" if holdout_n else ""
        run_id = f"all_{DS.NAME}_seed{seed}_{channels}_{built['norm']}{RUN_TAG}{hold}"
        if _already_done(done_df, run_id):
            prog.saltada()
            continue
        ckpt_path = CKPT_ROOT / "all" / f"{run_id}.pt"
        tr.set_seed(seed)
        model = tr.build_model(MODEL, n_chans=built["X_train"].shape[1],
                               n_times=built["X_train"].shape[2], n_classes=N_CLASSES,
                               ch_names=built["ch_names"])
        t0 = time.time()
        history, _ = tr.train_run(model, built["X_train"], y_train,
                                  built["X_val"], y_val, N_CLASSES, SFREQ,
                                  _cfg_run(built["ch_names"]), device, ckpt_path, seed=seed)
        metrics = tr.evaluate(model, built["X_val"], y_val, device, N_CLASSES)
        _escribir_curva(log_path, run_id, history)
        _escribir_meta(ckpt_path, built, channels, seed, apartados, holdout_seed)
        row = {
            "run_id": run_id, "scenario": "all", "seed": seed,
            "n_train": len(y_train), "n_val": len(y_val), "n_test": 0,
            "n_epochs": len(history), "elapsed_s": round(time.time() - t0, 1),
            "acc": metrics["acc"], "kappa": metrics["kappa"], "f1_macro": metrics["f1_macro"],
            "confusion_matrix": metrics["confusion_matrix"],
            "model": MODEL, "n_params": sum(p.numel() for p in model.parameters()),
            "ckpt_path": str(ckpt_path), "cfg_hash": CFG_HASH,
            "channels": channels, "norm": built["norm"], "n_subjects": len(subs),
            "n_holdout": len(apartados), "subs_apartados": " ".join(apartados),
        }
        if X_res is not None:
            mr = tr.evaluate(model, X_res, y_res, device, N_CLASSES)
            row.update({"n_test": len(y_res), "acc_reservado": mr["acc"],
                        "kappa_reservado": mr["kappa"], "f1_reservado": mr["f1_macro"],
                        "cm_reservado": mr["confusion_matrix"],
                        "frac_test": frac_test, "semilla_reserva": built["semilla_reserva"]})
        _append_row(log_path, row)
        done_df = pd.concat([done_df, pd.DataFrame([row])], ignore_index=True)
        extra = (f" | reservado acc={row['acc_reservado']:.3f} "
                 f"kappa={row['kappa_reservado']:.3f}") if X_res is not None else ""
        print(f"  {prog.ejecutada()} {run_id}: val acc={row['acc']:.3f} "
              f"kappa={row['kappa']:.3f}{extra} ({row['elapsed_s']}s, "
              f"{row['n_epochs']} ep)", flush=True)


def _escribir_meta(ckpt_path: Path, built: dict, channels: str, seed: int,
                   apartados=None, holdout_seed=None):
    """Sidecar del checkpoint: sin esto los pesos no se pueden usar en otro dataset."""
    st = built.get("stats")
    meta = {
        "modelo": MODEL, "dataset": DS.NAME, "cache": CACHE.name, "seed": seed,
        "channels_view": channels, "ch_names": list(built["ch_names"]),
        "norm": built["norm"], "window": list(WINDOW), "sfreq": SFREQ,
        "class_order": list(DS.CLASS_ORDER), "cfg_hash": CFG_HASH,
        "n_chans": int(built["X_train"].shape[1]), "n_times": int(built["X_train"].shape[2]),
        "run_tag": RUN_TAG, "variante": VARIANTE,
        "augment": {k: v for k, v in tr.DEFAULT_TRAIN_CFG["augment"].items()},
        # Los estadisticos van en el sidecar y no en el .pt para poder leerlos sin torch.
        "norm_mean": st["mean"].squeeze(-1).tolist() if st is not None else None,
        "norm_std": st["std"].squeeze(-1).tolist() if st is not None else None,
        "frac_test": built.get("frac_test", 0.0),
        "semilla_reserva": built.get("semilla_reserva"),
        # Sujetos que el modelo NO vio. Sin esto no se puede saber sobre cuales es
        # legitimo evaluarlo como "persona nueva".
        "subs_entrenamiento": list(built["subs"]),
        "subs_apartados": list(apartados or []),
        "semilla_holdout": holdout_seed if apartados else None,
    }
    ruta = ckpt_path.with_suffix(".meta.json")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Smoke test (Actividad 2.8 / plan §0.13): 1 sujeto, 1 fold, 1 semilla
# --------------------------------------------------------------------------- #

def smoke_test(channels: str, device):
    """1 sujeto, 1 fold, 1 semilla. Verifica que el pipeline corre y mide el tiempo."""
    log_path = REPORTS / f"baseline_eegnet_smoke_{channels}.csv"
    sub = _list_subjects()[0]
    print(f"Smoke test WITHIN: {sub}, fold 0, seed {SEEDS[0]}, canales={channels}", flush=True)
    within_subject_all(channels, device, log_path, subs=[sub], n_folds=5,
                       only_folds=[0], seeds=[SEEDS[0]])
    _report_curva(CKPT_ROOT / "within" / f"within_{sub}_fold0_seed{SEEDS[0]}_{channels}.pt")


def smoke_loso(channels: str, device):
    """1 fold LOSO, 1 semilla.

    DIAGNÓSTICO CLAVE: en LOSO el inner-train tiene ~2.795 trials frente a los ~76 del
    within-subject (37x más). Si aquí el val_loss BAJA de ln(4)=1.386, el pipeline está
    sano y el problema del within-subject es de tamaño de muestra. Si tampoco baja,
    hay algo roto aguas arriba y no tiene sentido lanzar el lote.
    También mide el tiempo por corrida LOSO, que es el que decide el costo del lote.
    """
    log_path = REPORTS / f"baseline_eegnet_smokeloso_{channels}.csv"
    print(f"Smoke test LOSO: fold 0, seed {SEEDS[0]}, canales={channels}", flush=True)
    loso_all(channels, device, log_path, only_folds=[0], seeds=[SEEDS[0]])
    folds = sp.make_loso_folds(_list_subjects())
    _report_curva(CKPT_ROOT / "loso" / f"loso_{folds[0].test}_seed{SEEDS[0]}_{channels}.pt")


def compare_norm(channels: str, device, n_folds: int = 6, seeds=None):
    """Compara las 3 estrategias de normalización sobre los MISMOS folds LOSO.

    Todo lo demás queda idéntico (mismos folds, misma semilla, mismos HPs): lo único que
    cambia es la normalización, así que la diferencia es atribuible a ella.

    Se promedia sobre VARIOS folds (default 6) a propósito: con un solo fold las diferencias
    quedan dentro del ruido y no permiten elegir (trampa #4 del HANDOFF).
    """
    seeds = seeds or [SEEDS[0]]
    log_path = REPORTS / f"baseline_{MODEL}_comparanorm_{channels}.csv"
    global LOG_PATH
    LOG_PATH = log_path
    done_df = _load_log(log_path)
    folds = sp.make_loso_folds(_list_subjects())[:n_folds]
    chance = float(np.log(N_CLASSES))
    print(f"Comparación de normalización: {len(folds)} folds x 3 estrategias x {len(seeds)} "
          f"semillas (modelo={MODEL}, canales={channels})", flush=True)

    filas = []
    for norm in ("global", "per_subject", "per_trial"):
        for fold in folds:
            built = sp.build_fold(fold, CACHE, window=WINDOW, ch_names=DS.CH_SETS[channels],
                                  normalize=norm)
            y_train = tr.encode_labels(built["y_train"], DS.CLASS_ORDER)
            y_val = tr.encode_labels(built["y_val"], DS.CLASS_ORDER)
            y_test = tr.encode_labels(built["y_test"], DS.CLASS_ORDER)
            for seed in seeds:
                run_id = f"cmpnorm_{norm}_{fold.test}_seed{seed}_{channels}"
                ckpt = CKPT_ROOT / "compare_norm" / f"{run_id}.pt"
                if _already_done(done_df, run_id):
                    prev = done_df[done_df["run_id"].astype(str) == run_id].iloc[0].to_dict()
                    filas.append(prev)
                    continue
                row = _run_one(
                    run_id, "compare_norm",
                    {"norm": norm, "test_sub": fold.test, "fold": fold.idx, "channels": channels},
                    n_chans=built["X_train"].shape[1], n_times=built["X_train"].shape[2],
                    seed=seed,
                    X_train=built["X_train"], y_train=y_train,
                    X_val=built["X_val"], y_val=y_val,
                    X_test=built["X_test"], y_test=y_test,
                    device=device, ckpt_path=ckpt, ch_names=built["ch_names"],
                )
                ck = torch.load(ckpt, map_location="cpu", weights_only=False)
                row["best_val_loss"] = min(r["val_loss"] for r in ck["history"])
                _append_row(log_path, row)
                filas.append(row)
                print(f"  [{norm}] {fold.test}: acc={row['acc']:.3f} kappa={row['kappa']:.3f} "
                      f"val_loss={row['best_val_loss']:.4f} ({row['elapsed_s']}s)", flush=True)

    d = pd.DataFrame(filas)
    print(f"\n=== NORMALIZACIÓN, promedio sobre {len(folds)} folds LOSO | "
          f"azar: acc=0.25, kappa=0, val_loss={chance:.4f} ===")
    print(f"{'estrategia':<14} {'acc':>14} {'kappa':>14} {'val_loss':>10} {'>azar':>7}")
    for norm, g in d.groupby("norm"):
        print(f"{norm:<14} {g['acc'].mean():>7.3f}±{g['acc'].std():<6.3f} "
              f"{g['kappa'].mean():>7.3f}±{g['kappa'].std():<6.3f} "
              f"{g['best_val_loss'].mean():>10.4f} "
              f"{int((g['acc'] > 0.25).sum()):>4}/{len(g)}")
    mejor = d.groupby("norm")["kappa"].mean().idxmax()
    print(f"\nMejor por kappa medio: {mejor}")
    print("Se elige por KAPPA (corregido por azar), no por accuracy: con 4 clases y "
          "desempeños de 0.25-0.35 la accuracy es inestable (HANDOFF §3.bis).")


def sanity_overfit(channels: str, device, n_trials: int = 128, epochs: int = 300):
    """TEST DE SOBREAJUSTE DELIBERADO: ¿puede el modelo memorizar un puñado de trials?

    Es el paso estándar de depuración de redes: se le da un subconjunto pequeño SIN
    augmentation y se comprueba si lleva el train_loss a ~0 evaluando sobre ESOS MISMOS
    trials. No mide generalización (es fuga a propósito) — mide si la plomería está bien.

      * Si el loss NO baja  -> hay un bug (etiquetas desalineadas, normalización que
        destruye la señal, LR mal, gradientes que no fluyen). No sirve tocar HPs.
      * Si el loss SÍ baja  -> la plomería está sana y lo de LOSO/within es un problema
        real de señal débil y cambio de dominio.
    """
    fold = sp.make_loso_folds(_list_subjects())[0]
    built = sp.build_fold(fold, CACHE, window=WINDOW, ch_names=DS.CH_SETS[channels])
    X = built["X_train"][:n_trials]
    y = tr.encode_labels(built["y_train"], DS.CLASS_ORDER)[:n_trials]
    print(f"Sanity overfit: {len(y)} trials, {X.shape[1]} canales, {X.shape[2]} muestras, "
          f"clases={np.bincount(y).tolist()}", flush=True)

    cfg = json.loads(json.dumps(tr.DEFAULT_TRAIN_CFG))
    cfg.update(max_epochs=epochs, warmup_epochs=5, patience=epochs,   # sin early stopping
               label_smoothing=0.0,                                    # el loss debe poder ir a ~0
               augment={})                                             # SIN augmentation
    tr.set_seed(SEEDS[0])
    model = tr.build_model("eegnet", n_chans=X.shape[1], n_times=X.shape[2], n_classes=N_CLASSES)
    ckpt = CKPT_ROOT / "sanity" / f"overfit_{channels}_{n_trials}.pt"
    if ckpt.exists():
        ckpt.unlink()      # siempre desde cero: es un diagnóstico, no una corrida del estudio
    t0 = time.time()
    history, _ = tr.train_run(model, X, y, X, y, N_CLASSES, SFREQ, cfg, device, ckpt, seed=SEEDS[0])
    m = tr.evaluate(model, X, y, device, N_CLASSES)

    print(f"\n--- sobreajuste deliberado ({len(history)} épocas, {time.time()-t0:.0f}s) ---")
    for r in history[::max(1, len(history) // 12)]:
        print(f"  ep {r['epoch']:3d}  train {r['train_loss']:.4f}")
    print(f"  train_loss final: {history[-1]['train_loss']:.4f}  (azar = {np.log(N_CLASSES):.4f})")
    print(f"  accuracy sobre los MISMOS trials de entrenamiento: {m['acc']:.3f}")
    if m["acc"] > 0.95:
        print("  ✓ El modelo SÍ puede memorizar -> la plomería está sana.")
    else:
        print("  ⚠ NO logra memorizar ni con fuga deliberada -> hay un BUG aguas arriba.")


def _report_curva(ckpt_path: Path):
    """Imprime la curva de pérdida: es lo que dice si el modelo GENERALIZA o solo memoriza."""
    if not ckpt_path.exists():
        return
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    h = ck["history"]
    chance = float(np.log(N_CLASSES))
    print(f"\n--- curva de pérdida ({len(h)} épocas) | azar = ln({N_CLASSES}) = {chance:.4f} ---")
    for r in h[::max(1, len(h) // 12)]:
        print(f"  ep {r['epoch']:3d}  train {r['train_loss']:.4f}   val {r['val_loss']:.4f}")
    best = min(r["val_loss"] for r in h)
    best_ep = min(h, key=lambda r: r["val_loss"])["epoch"]
    print(f"  mejor val_loss: {best:.4f} en la época {best_ep}")
    if best >= chance - 1e-3:
        print("  ⚠ El val_loss NUNCA bajó del azar -> el modelo memoriza pero NO generaliza.")
    else:
        print("  ✓ El val_loss bajó del azar -> el modelo generaliza.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["smoke", "smoke-loso", "sanity", "compare-norm",
                                       "within", "loso", "all"], required=True)
    ap.add_argument("--dataset", choices=list(datasets.disponibles()), required=True)
    ap.add_argument("--norm", choices=["global", "per_subject", "per_trial", "none"],
                    default="global",
                    help="estrategia de normalización para within/loso (ver src/splits.py). "
                         "'none' = sin z-score (µV crudos); usar con LaBraM variante B "
                         "(LABRAM_UV_SCALE=100, escalado nativo del preentrenamiento).")
    # Las vistas de canales dependen del dataset, así que no se pueden validar aquí:
    # se comprueban tras cargar el módulo correspondiente.
    ap.add_argument("--channels", default="completo")
    ap.add_argument("--model", choices=["eegnet", "conformer", "labram", "eegmamba"], default="eegnet")
    ap.add_argument("--folds", type=int, default=6, help="nº de folds LOSO en compare-norm")
    ap.add_argument("--loso-folds", type=int, nargs="*", default=None,
                    help="loso: índices de fold a ejecutar (p. ej. `0` para medir uno solo). "
                         "Por defecto, todos.")
    # Lote y lr como flags, no como constantes: cambiarlos altera los resultados, así que
    # tienen que poder medirse por separado antes de fijarlos. Ver el bench de pod/run.py.
    ap.add_argument("--batch", type=int, default=None,
                    help="tamaño de lote físico y efectivo (por defecto, el de DEFAULT_TRAIN_CFG)")
    ap.add_argument("--lr", type=float, default=None, help="tasa de aprendizaje")
    ap.add_argument("--cpu-pipeline", action="store_true",
                    help="fuerza la augmentation en CPU (numpy) en vez de en GPU")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--frac-visto", type=float, default=0.10,
                    help="loso: fracción de trials reservada de cada sujeto de "
                         "entrenamiento para medir 'sujeto visto' (DSI). 0 = desactivado")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="within: restringe a estos sujetos (p. ej. gate acotado)")
    ap.add_argument("--holdout-n", type=int, default=0, metavar="N",
                    help="all: aparta N SUJETOS ENTEROS del entrenamiento. Permite evaluar "
                         "el mismo modelo sobre personas nuevas del mismo protocolo, que es "
                         "lo que no se puede medir entrenando con los 54.")
    ap.add_argument("--holdout-seed", type=int, default=sp.SEMILLA_HOLDOUT,
                    help="semilla de la seleccion de sujetos apartados")
    ap.add_argument("--frac-val", type=float, default=0.10,
                    help="all: fraccion de trials de cada sujeto reservada para la parada "
                         "temprana. Estratificada por (sujeto, clase, sesion).")
    ap.add_argument("--frac-test", type=float, default=0.0,
                    help="all: fraccion RESERVADA que no entra en entrenamiento, ni en la "
                         "parada temprana, ni en los estadisticos de normalizacion. Es el "
                         "conjunto COMPARTIDO sobre el que se mide tambien el modelo "
                         "transferido desde el otro dataset. Usar en el dataset DESTINO "
                         "(p. ej. 0.20 en wbcic_2c) para tener un techo nativo comparable.")
    ap.add_argument("--per-session", action="store_true",
                    help="within: 10-fold POR SESION sin mezclar (protocolo OpenBMI, §10)")
    ap.add_argument("--within-folds", type=int, default=10,
                    help="within: folds por sesión/sujeto (metodología §10: 10)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="restringe las semillas (por defecto las 5 de SEEDS). "
                         "Útil para un gate rápido con 1 semilla.")
    # Sufijo del CSV para poder ejecutar VARIOS procesos en paralelo (sharding por folds).
    # `_append_row` reescribe el fichero completo en cada fila, asi que dos procesos sobre
    # el MISMO csv se pisan. Con --log-tag cada trabajador escribe el suyo y luego se
    # fusionan con scripts/merge_shards.py. No toca run_id ni resultados.
    ap.add_argument("--workers", type=int, default=1,
                    help="numero de procesos EN PARALELO sobre la misma GPU. Reparte los folds "
                         "(loso) o los sujetos (within) de forma intercalada, cada uno con su "
                         "propio CSV, y fusiona al terminar. Sube el uso de GPU sin cambiar "
                         "ningun resultado. Vigilar la VRAM: cada proceso sube su train (~4 GB).")
    ap.add_argument("--log-tag", default=None,
                    help="sufijo del CSV de salida (p. ej. `shard0`). Para ejecutar varios "
                         "procesos en paralelo sin que se pisen el log. Fusionar despues "
                         "con scripts/merge_shards.py.")
    ap.add_argument("--ckpt-cada", type=int, default=None,
                    help="guardar checkpoint cada N epocas; 0 = solo el final. Por defecto "
                         "1 en loso y 0 en within. No afecta a los resultados.")
    ap.add_argument("--cache", default=None,
                    help="directorio de tensores (por defecto el del dataset). "
                         "Usar epochs_cache_ica para entrenar sobre datos limpios.")
    # --- Avance 7 (octubre de 2026) (variantes de la base) ---
    ap.add_argument("--variante", default="",
                    help="nombre corto de la variante; se anade al run_id, al CSV y al "
                         "checkpoint como _v<nombre>. Obligatorio con --window o --aug.")
    ap.add_argument("--window", type=float, nargs=2, default=None, metavar=("TMIN", "TMAX"),
                    help="ventana de modelado en s (por defecto la del dataset, 0 a 4)")
    ap.add_argument("--aug", default="base", choices=["base", "none", "cr", "fft"],
                    help="preset de aumentacion (src/variantes.py:PRESETS_AUG)")
    ap.add_argument("--fewshot", action="store_true",
                    help="loso: tras evaluar cada fold, calibra con k ensayos del sujeto no "
                         "visto (src/fewshot.py). CSV aparte en reports/<ds>/fewshot/")
    args = ap.parse_args()
    if args.norm == "none":          # cadena de CLI -> None de splits (sin normalización)
        args.norm = None

    global VARIANTE, FEWSHOT, WINDOW
    if (args.window is not None or args.aug != "base") and not args.variante:
        ap.error("--window y --aug exigen --variante: sin ella la variante compartiria "
                 "run_id y CSV con la base, y la reanudacion la saltaria")
    VARIANTE = args.variante
    FEWSHOT = args.fewshot
    if args.window is not None:
        WINDOW = tuple(args.window)
    if args.aug != "base":
        sys.path.insert(0, str(REPO_ROOT / "src"))
        import variantes as va
        tr.DEFAULT_TRAIN_CFG["augment"].update(va.PRESETS_AUG[args.aug])

    # Se aplican ANTES de calcular CFG_HASH para que dos ejecuciones con hiperparámetros
    # distintos no compartan hash y se confundan al comparar resultados.
    if args.batch is not None:
        tr.DEFAULT_TRAIN_CFG["batch_size_physical"] = args.batch
        tr.DEFAULT_TRAIN_CFG["batch_size_effective"] = args.batch
    if args.lr is not None:
        tr.DEFAULT_TRAIN_CFG["lr"] = args.lr
    if args.max_epochs is not None:
        tr.DEFAULT_TRAIN_CFG["max_epochs"] = args.max_epochs
    if args.cpu_pipeline:
        tr.DEFAULT_TRAIN_CFG["gpu_pipeline"] = False
    # En within cada ejecucion dura segundos, asi que el checkpoint por epoca no protege
    # nada y llega a dominar el tiempo (83 % en LaBraM within). Solo se guarda el final, que
    # es el que se reutiliza. En LOSO se mantiene por epoca: ahi un fold son minutos.
    # No mueve el cfg_hash: ckpt_cada esta en _CLAVES_SIN_EFECTO.
    if args.ckpt_cada is not None:
        tr.DEFAULT_TRAIN_CFG["ckpt_cada"] = args.ckpt_cada
    elif args.mode in ("within", "smoke"):
        tr.DEFAULT_TRAIN_CFG["ckpt_cada"] = 0

    # Recalcular AHORA, ya con los overrides del CLI aplicados (ver _calcular_cfg_hash).
    global CFG_HASH
    CFG_HASH = _calcular_cfg_hash()
    _configurar(args.dataset, args.model, args.cache)
    if args.channels not in DS.CH_SETS:
        ap.error(f"--channels {args.channels!r} no existe en el dataset '{DS.NAME}'. "
                 f"Disponibles: {list(DS.CH_SETS)}")
    device = get_device()
    # cuDNN autotune: las formas de entrada son FIJAS durante todo un barrido (mismo numero de
    # canales y de muestras en cada lote), asi que cuDNN mide una vez los algoritmos de
    # convolucion y reutiliza el mejor. Sin esto aplica una heuristica generica en cada llamada.
    # APAGADO POR DEFECTO: puede elegir un algoritmo distinto al de las semillas ya ejecutadas,
    # lo que mueve los ultimos decimales. Es ruido por debajo de la variacion entre semillas
    # (0,426 pp medidos en within), pero la base no debe cambiar por una optimizacion.
    # Se activa con TESIS_CUDNN_BENCHMARK=1 si alguna vez se rehace todo de cero.
    if device.type == "cuda" and os.environ.get("TESIS_CUDNN_BENCHMARK") == "1":
        torch.backends.cudnn.benchmark = True
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device: {device} ({gpu_name}) | dataset: {DS.NAME} ({DS.N_CLASSES} clases, "
          f"azar {DS.CHANCE:.2f}) | modelo: {MODEL} | cfg_hash: {CFG_HASH} | "
          f"cudnn.benchmark={torch.backends.cudnn.benchmark}", flush=True)

    if args.mode == "smoke":
        smoke_test(args.channels, device)
    elif args.mode == "smoke-loso":
        smoke_loso(args.channels, device)
    elif args.mode == "sanity":
        sanity_overfit(args.channels, device)
    elif args.mode == "compare-norm":
        compare_norm(args.channels, device, n_folds=args.folds)
    # El CSV se separa por cache. Sin esto, dos corridas sobre datos distintos escriben
    # en el mismo fichero Y comparten run_id (que no lleva el cache), asi que la logica
    # de reanudacion SALTA runs creyendolos hechos: el 2026-07-25 se perdieron las
    # semillas 2026 de sub-02 y sub-04 sobre el cache limpio por esta via.
    suf = "" if CACHE.name == DS.CACHE_DIR.name else f"_{CACHE.name}"
    if args.log_tag:
        suf += f"__{args.log_tag}"

    if args.mode == "within":
        tag = "porsesion" if args.per_session else "mezcla"
        log_path = REPORTS / f"baseline_{MODEL}_within_{args.channels}_{args.norm}_{tag}{RUN_TAG}{suf}.csv"
        if args.workers > 1:
            subs = args.subjects or _list_subjects()
            return _repartir(args.workers, log_path, list(subs), "--subjects", args.log_tag)
        within_subject_all(args.channels, device, log_path, subs=args.subjects,
                           n_folds=args.within_folds, norm=args.norm,
                           by_session=args.per_session, seeds=args.seeds)
    elif args.mode == "all":
        hold = f"_hold{args.holdout_n}s{args.holdout_seed}" if args.holdout_n else ""
        log_path = REPORTS / f"baseline_{MODEL}_all_{args.channels}_{args.norm}{RUN_TAG}{hold}{suf}.csv"
        all_subjects(args.channels, device, log_path, norm=args.norm, seeds=args.seeds,
                     frac_val=args.frac_val, frac_test=args.frac_test,
                     holdout_n=args.holdout_n, holdout_seed=args.holdout_seed)
    elif args.mode == "loso":
        log_path = REPORTS / f"baseline_{MODEL}_loso_{args.channels}_{args.norm}{RUN_TAG}{suf}.csv"
        # only_folds/seeds existian en loso_all pero main() no los pasaba, asi que no
        # habia forma de correr un subconjunto de folds desde la linea de ordenes —
        # imprescindible para medir el coste de UN fold antes de alquilar GPU.
        if args.workers > 1:
            folds = args.loso_folds
            if folds is None:
                folds = [f.idx for f in sp.make_loso_folds(_list_subjects())]
            return _repartir(args.workers, log_path, list(folds), "--loso-folds", args.log_tag)
        loso_all(args.channels, device, log_path, norm=args.norm,
                 only_folds=args.loso_folds, seeds=args.seeds,
                 frac_visto=args.frac_visto)


if __name__ == "__main__":
    main()
