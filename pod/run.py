"""Punto de entrada UNICO para la nube. Fases seleccionables, todas reanudables.

  gpu       verifica la tarjeta y avisa si esta sobredimensionada para el modelo
  download  descarga los 54 sujetos con MOABB              (~20-30 min)
  cache     preprocesado + ICA, PARALELO por sujeto        (~30 min con 8 procesos)
  verify    ¿este entorno reproduce la Fase 2?             (~5 min)  <- NO SALTAR
  export    consolida a un unico fp16 y lo sube a HF       (~5 min)
  bench     mide 1 fold LOSO y 1 sujeto within             (~30 min)

Uso tipico de la PRIMERA sesion:

    source /workspace/env.sh
    python pod/run.py --fase gpu download cache verify export bench

Las fases se pueden repetir sin efectos adversos: `download` y `cache` saltan lo ya
hecho, y `verify` no escribe nada.

COSTES (leer una vez)
  Se cobra por SEGUNDO mientras el pod este ENCENDIDO, este ocupado o no. Los comandos
  no cuestan; el pod olvidado si. PARAR el pod cuesta MAS por disco que tenerlo
  encendido. Al TERMINAR se borra /workspace -> correr `export` ANTES de destruir.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Valores de referencia medidos en local (Fase 2). La fase `verify` los compara.
CFG_HASH_ESPERADO = "6391bc7fe8d4"
# Medidos en local sobre el cache limpio (results/reports/openbmi/control_ocular_epochs_cache_ica.csv).
CONTROL_ESPERADO = {                    # sujeto -> (proxy_EOG, motor), tolerancia +-0.03
    "sub-29": (0.7050, 0.9425),         # muscular irreducible: debe SEGUIR alto
    "sub-04": (0.5175, 0.6075),         # ocular resuelto por el ICA
    "sub-01": (0.4775, 0.8075),         # limpio: control negativo, no debe moverse
    "sub-35": (0.5425, 0.6075),         # ocular resuelto
}
TOL = 0.03

# Ventana exportada: 0.5 s de margen a cada lado de la ventana de modelado (0-4 s).
# El margen no es capricho — permite desplazar la ventana (p. ej. recortar a 0.5-4 s
# para excluir la sacada del cue) y hacer correccion de linea base, sin reprocesar.
EXPORT_WINDOW = (-0.5, 4.5)


def cpus_reales():
    """CPUs asignadas al CONTENEDOR, no las del host.

    `os.cpu_count()` y `nproc` devuelven las del host (en un pod de RunPod, 96) mientras
    el cgroup asigna muchas menos (8.5 en el A40 medido el 2026-07-26). Usar la cifra del
    host lanza decenas de procesos e hilos OpenMP sobre pocos nucleos: sobresuscripcion
    que ralentiza en vez de acelerar.
    """
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if q != "max":
            return max(1, int(int(q) / int(p)))
    except Exception:
        pass
    try:                                   # cgroup v1
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return max(1, q // p)
    except Exception:
        pass
    return os.cpu_count() or 4


def mem_limite_gb():
    """Memoria asignada al CONTENEDOR. `free` reporta la del HOST (503 GB en el pod del
    2026-07-26) mientras el cgroup asignaba 55 GB: fiarse de `free` lleva a lanzar
    demasiados procesos y que el kernel mate algunos por OOM sin dejar traceback."""
    for p in ("/sys/fs/cgroup/memory.max",                       # v2
              "/sys/fs/cgroup/memory/memory.limit_in_bytes"):    # v1
        try:
            v = Path(p).read_text().strip()
            if v != "max":
                n = int(v)
                if n < (1 << 60):        # v1 usa un numero enorme para "sin limite"
                    return n / 1e9
        except Exception:
            pass
    return None


# Pico observado por proceso de build_cache (carga el .mat de 609 MB, lo expande a
# float64 y hace copias para el ICA). Medido el 2026-07-26: 6 procesos = 55 GB.
GB_POR_PROCESO_CACHE = 9.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd, **kw):
    log(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=True, **kw)


# --------------------------------------------------------------------------- #
# gpu
# --------------------------------------------------------------------------- #
def fase_gpu():
    import torch
    if not torch.cuda.is_available():
        sys.exit("ERROR: no hay GPU visible.")
    p = torch.cuda.get_device_properties(0)
    libre, total = torch.cuda.mem_get_info()
    log(f"GPU     : {p.name}")
    log(f"VRAM    : {total/1e9:.1f} GB total | {libre/1e9:.1f} GB libre")
    log(f"Arquit. : sm_{p.major}{p.minor} | torch {torch.__version__}")
    log(f"vCPU    : {cpus_reales()} asignadas al contenedor ({os.cpu_count()} en el host)")

    sm = f"sm_{p.major}{p.minor}"
    if sm not in torch.cuda.get_arch_list():
        sys.exit(f"ERROR: torch no trae kernels para {sm}. Ver pod/setup.sh")

    # El dataset ENTERO cabe en VRAM: cada fold LOSO entrena con 53 de 54 sujetos, asi
    # que la union de todos los folds es simplemente el dataset completo. Se carga una
    # vez y cada fold es una seleccion de indices -> sin DataLoader ni transferencias.
    n_muestras = int((EXPORT_WINDOW[1] - EXPORT_WINDOW[0]) * 200) + 1
    gb = 54 * 400 * 62 * n_muestras * 2 / 1e9
    log(f"Dataset completo en fp16: {gb:.2f} GB -> {'CABE' if gb < total/1e9*0.5 else 'AJUSTADO'} en VRAM")

    # Aviso de sobredimensionamiento: el error caro es pagar una H100 para EEGNet.
    if total / 1e9 > 60:
        log("")
        log("  AVISO: esta tarjeta tiene mucha VRAM. EEGNet son 2.898 parametros y NO")
        log("  la satura: el cuello sera el lanzamiento de kernels, no el computo.")
        log("  Para modelos pequeños sale mas a cuenta una A40/A6000, o ejecutar")
        log("  varios folds en paralelo para amortizarla.")


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def fase_download(subjects=None):
    import datasets
    DS = datasets.get("openbmi")
    ds = DS.make_dataset(test_run=True)
    subs = subjects or list(ds.subject_list)
    log(f"Descargando {len(subs)} sujetos a {os.environ.get('MNE_DATA')}")
    log("(MOABB salta lo ya descargado, asi que esta fase es reanudable)")
    t0 = time.time()
    for i, s in enumerate(subs, 1):
        try:
            ds.get_data(subjects=[s])
            log(f"  [{i}/{len(subs)}] sub-{s:02d} ok")
        except Exception as e:
            log(f"  [{i}/{len(subs)}] sub-{s:02d} ERROR {type(e).__name__}: {e}")
    log(f"Descarga terminada en {(time.time()-t0)/60:.1f} min")


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def fase_cache(n_jobs):
    """Preprocesado + ICA en paralelo por sujeto.

    build_cache.py es secuencial (2.8 h en el portatil). Aqui se reparten los sujetos
    entre procesos: cada uno escribe SU .npz y SU fila de log, asi que no compiten.
    """
    import datasets
    DS = datasets.get("openbmi")
    cache = Path(os.environ["OPENBMI_PROC_ROOT"]) / "epochs_cache_ica"

    # build_cache.py decide que saltar leyendo SU log, no los ficheros. Si el repo se
    # sube desde una maquina donde ya se construyo el cache, ese log viaja con el codigo
    # y build_cache cree que esta todo hecho: procesa 0 sujetos y `verify` falla con el
    # cache vacio. Paso el 2026-07-26. Aqui se detecta y se aparta el log heredado.
    log_bc = REPO / "results" / "reports" / "openbmi" / "tensor_cache_log_ica.csv"
    hechos = {f.name.split("_")[0] for f in cache.glob("sub-*_full.npz")} if cache.exists() else set()
    if log_bc.exists():
        import pandas as pd
        try:
            n_log = len(pd.read_csv(log_bc))
        except Exception:
            n_log = 0
        if n_log > len(hechos):
            viejo = log_bc.with_suffix(".csv.heredado")
            log_bc.rename(viejo)
            log(f"  Log con {n_log} filas pero solo {len(hechos)} .npz en disco:")
            log(f"  viene de otra maquina. Apartado a {viejo.name}")

    faltan = [s for s in range(1, 55) if f"sub-{s:02d}" not in hechos]
    if not faltan:
        log(f"Cache completo ({len(hechos)} sujetos). Nada que hacer.")
        return

    # Tope por MEMORIA, no solo por CPU. El 2026-07-26, 6 procesos en un contenedor de
    # 55 GB tocaron el techo y el kernel mato dos SIN traceback (build_cache captura las
    # excepciones, asi que un log que acaba de golpe = proceso matado, no fallo del codigo).
    mem = mem_limite_gb()
    if mem:
        tope = max(1, int(mem / GB_POR_PROCESO_CACHE))
        if n_jobs > tope:
            log(f"  {n_jobs} procesos no caben en {mem:.0f} GB "
                f"(~{GB_POR_PROCESO_CACHE:.0f} GB cada uno) -> se bajan a {tope}")
            n_jobs = tope
    log(f"Faltan {len(faltan)} sujetos. {n_jobs} procesos en paralelo.")

    logs = Path(os.environ.get("TESIS_LOGS", REPO.parent / "logs"))
    logs.mkdir(parents=True, exist_ok=True)

    # UN hilo por proceso: con n_jobs procesos ya saturamos las CPUs asignadas. Si cada
    # uno abriera ademas su pool de OpenMP se pelearian por los mismos nucleos.
    entorno = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}

    procs, fhs = [], []
    t0 = time.time()
    for k, lote in enumerate(faltan[i::n_jobs] for i in range(n_jobs)):
        if not lote:
            continue
        fh = open(logs / f"cache_{k}.log", "w")
        fhs.append(fh)
        procs.append(subprocess.Popen(
            [sys.executable, "scripts/openbmi/build_cache.py", "--ica", "--subjects",
             *[str(s) for s in lote]],
            cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, env=entorno))
    log(f"  logs por proceso en {logs}/cache_*.log")
    for p in procs:
        p.wait()
    for fh in fhs:
        fh.close()
    n = len(list(cache.glob("sub-*_full.npz")))
    log(f"Cache: {n}/54 sujetos en {(time.time()-t0)/60:.1f} min")
    if n < 54:
        log("  AVISO: faltan sujetos. Relanza esta fase (es reanudable) y mira logs/.")


# --------------------------------------------------------------------------- #
# verify  — la fase que NO hay que saltarse
# --------------------------------------------------------------------------- #
def fase_verify():
    """¿El cache generado aqui es el MISMO que el validado en local?

    Si se reconstruye el cache en otra maquina hay que demostrarlo, o toda la Fase 2
    queda en el aire. Dos comprobaciones: el hash de configuracion y los numeros del
    Control A en sujetos conocidos.
    """
    import numpy as np
    import pandas as pd

    cache = Path(os.environ["OPENBMI_PROC_ROOT"]) / "epochs_cache_ica"
    fs = sorted(cache.glob("sub-*_full.npz"))
    log(f"--- 1/3  integridad del cache ({len(fs)} archivos) ---")
    if len(fs) != 54:
        log(f"  FALLO: hay {len(fs)} sujetos, se esperaban 54")
        return False
    malos = []
    for f in fs:
        d = np.load(f, allow_pickle=False)
        X, y = d["X"], d["y"]
        if X.shape[0] != 400 or X.shape[1] != 62 or not np.isfinite(X).all():
            malos.append(f.name)
        elif sorted(pd.Series(y).value_counts().tolist()) != [200, 200]:
            malos.append(f.name)
    log(f"  {'FALLO: ' + str(malos) if malos else 'ok — 54 sujetos, 400x62, balanceados, sin NaN'}")

    log("--- 2/3  hash de configuracion ---")
    hashes = set(np.load(f, allow_pickle=False)["cfg_hash"].item() for f in fs[:5])
    ok_hash = hashes == {CFG_HASH_ESPERADO}
    log(f"  {hashes} vs esperado {{{CFG_HASH_ESPERADO}}} -> {'ok' if ok_hash else 'FALLO'}")

    log("--- 3/3  Control A sobre sujetos conocidos ---")
    log("  (el ICA es estocastico: una version distinta de mne/icalabel puede converger")
    log("   a componentes distintos y dar otros numeros aunque el hash coincida)")
    subs = " ".join(CONTROL_ESPERADO)
    out = REPO / "results" / "reports" / "openbmi" / "verify_pod.csv"
    sh(f"{sys.executable} scripts/openbmi/control_ocular.py --cache {cache} "
       f"--subjects {subs} --out {out}", cwd=REPO)

    d = pd.read_csv(out).set_index("sub")
    ok_ctrl = True
    for s, (proxy_esp, motor_esp) in CONTROL_ESPERADO.items():
        if s not in d.index:
            log(f"  {s}: FALTA"); ok_ctrl = False; continue
        dp = abs(d.loc[s, "acc_proxy_EOG"] - proxy_esp)
        dm = abs(d.loc[s, "acc_motor"] - motor_esp)
        bien = dp <= TOL and dm <= TOL
        ok_ctrl &= bien
        log(f"  {s}: proxy {d.loc[s,'acc_proxy_EOG']:.3f} (esp {proxy_esp:.3f}, d={dp:.3f}) | "
            f"motor {d.loc[s,'acc_motor']:.3f} (esp {motor_esp:.3f}, d={dm:.3f})  "
            f"{'ok' if bien else 'FALLO'}")

    todo = not malos and ok_hash and ok_ctrl
    log("")
    log(f"=== VERIFICACION: {'SUPERADA' if todo else 'FALLIDA'} ===")
    if not todo:
        log("NO entrenar sobre este cache. Revisar versiones en pod/requirements.txt")
        log("y comparar con las del entorno local que produjo la Fase 2.")
    return todo


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def fase_export(repo_hf):
    """Consolida los 54 .npz en un unico array fp16 y lo sube fuera del pod.

    Hay que sacarlo del pod ANTES de destruirlo: /workspace se borra al terminar.
    fp16 verificado en local: la log-varianza (la caracteristica que usan CSP y EEGNet)
    se desvia 3.5e-04, y el z-score se recalcula por fold de todos modos.
    """
    import numpy as np
    import datasets
    DS = datasets.get("openbmi")

    cache = Path(os.environ["OPENBMI_PROC_ROOT"]) / "epochs_cache_ica"
    dest = Path(os.environ["OPENBMI_PROC_ROOT"]) / "openbmi_ica_fp16.npz"
    fs = sorted(cache.glob("sub-*_full.npz"))
    if len(fs) != 54:
        sys.exit(f"Solo hay {len(fs)} sujetos; exporta cuando esten los 54.")

    d0 = np.load(fs[0], allow_pickle=False)
    sf, tmin = float(d0["sfreq"]), float(d0["tmin"])
    i0 = int(round((EXPORT_WINDOW[0] - tmin) * sf))
    i1 = int(round((EXPORT_WINDOW[1] - tmin) * sf)) + 1
    log(f"Ventana {EXPORT_WINDOW} s -> muestras [{i0}:{i1}] ({i1-i0})")

    X, y, ses, run, subs = [], [], [], [], []
    for f in fs:
        d = np.load(f, allow_pickle=False)
        X.append(d["X"][:, :, i0:i1].astype(np.float16))
        y.append(d["y"]); ses.append(d["session"]); run.append(d["run"])
        subs.append(d["subject"].item())
    X = np.stack(X)                       # (54, 400, 62, n)
    log(f"Consolidado: {X.shape} {X.dtype} = {X.nbytes/1e9:.2f} GB")

    np.savez(dest, X=X, y=np.stack(y), session=np.stack(ses), run=np.stack(run),
             subjects=np.array(subs), ch_names=d0["ch_names"], sfreq=sf,
             tmin=EXPORT_WINDOW[0], cfg_hash=CFG_HASH_ESPERADO)
    log(f"Escrito {dest} ({dest.stat().st_size/1e9:.2f} GB)")

    if repo_hf:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_hf, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=str(dest), path_in_repo=dest.name,
                        repo_id=repo_hf, repo_type="dataset")
        log(f"Subido a https://huggingface.co/datasets/{repo_hf}")
    else:
        log("Sin --hf-repo: el archivo se queda en el pod. SUBELO antes de destruirlo.")


# --------------------------------------------------------------------------- #
# bench
# --------------------------------------------------------------------------- #
def fase_bench():
    """Convierte las extrapolaciones en numeros medidos.

    Lo que hay que saber antes de comprometer presupuesto:
      * cuanto cuesta UN fold LOSO real (53 sujetos de entrenamiento)
      * cuanto cuesta UN sujeto de within
    """
    cache = Path(os.environ["OPENBMI_PROC_ROOT"]) / "epochs_cache_ica"

    # Se mide con POCAS epocas: lo que interesa es el coste POR EPOCA, y para eso no hace
    # falta esperar a que converja. La extrapolacion usa las epocas tipicas hasta el early
    # stopping, no las 200 del tope.
    EPOCAS = 12
    EPOCAS_TIPICAS = 80          # estimacion prudente de convergencia con paciencia 20

    # Se comparan configuraciones en vez de medir una sola: sin el "antes" no se puede
    # afirmar que la optimizacion sirvio de algo.
    # (nombre, modelo, flags). Las tres primeras aislan UNA variable cada vez sobre
    # EEGNet; la ultima mide un modelo grande para saber cuanto escala el coste, porque
    # extrapolar a Conformer/Mamba/LaBraM desde un modelo de 2.898 parametros subestima.
    variantes = [
        ("eegnet_cpu_lote32",  "eegnet",    "--cpu-pipeline --batch 32"),
        ("eegnet_gpu_lote32",  "eegnet",    "--batch 32"),
        ("eegnet_gpu_lote256", "eegnet",    "--batch 256 --lr 2.8e-3"),
        ("conformer_gpu_l256", "conformer", "--batch 256 --lr 2.8e-3"),
    ]

    med = {}
    for nombre, modelo, flags in variantes:
        log(f"=== fold LOSO, {EPOCAS} epocas | {nombre} ===")
        limpiar_checkpoints(cache)
        t0 = time.time()
        try:
            sh(f"{sys.executable} scripts/train_baseline.py --mode loso --dataset openbmi "
               f"--model {modelo} --cache {cache} --loso-folds 0 --seeds 2026 "
               f"--max-epochs {EPOCAS} {flags}", cwd=REPO)
            med[nombre] = round((time.time() - t0) / EPOCAS, 2)
            log(f"  -> {med[nombre]:.2f} s/epoca")
        except subprocess.CalledProcessError as e:
            log(f"  FALLO: {e}")

    if not med:
        return
    log("")
    log("=== s/EPOCA MEDIDO (fold LOSO, 53 sujetos de entrenamiento) ===")
    base = med.get("eegnet_cpu_lote32")
    for k, v in med.items():
        mejora = f"  ({base/v:.1f}x vs CPU)" if base and v else ""
        log(f"  {k:20s} {v:7.2f} s/epoca{mejora}")

    eeg = med.get("eegnet_gpu_lote256") or min(med.values())
    conf = med.get("conformer_gpu_l256")
    log("")
    log(f"=== PRESUPUESTO (A40 a $0.47/h, ~{EPOCAS_TIPICAS} epocas/fold) ===")
    log("  Cuatro entrenamientos: EEGNet (baseline) + Conformer + Mamba + LaBraM.")
    if conf and eeg:
        log(f"  Conformer cuesta {conf/eeg:.1f}x lo que EEGNet -> Mamba y LaBraM, mas.")
        # Prudente: EEGNet al ritmo medido y las otras TRES al ritmo del Conformer.
        s_ronda = eeg + 3 * conf
        etiqueta = "EEGNet + 3 x ritmo Conformer"
    else:
        s_ronda = 4 * eeg
        etiqueta = "4 x ritmo EEGNet (OPTIMISTA: los otros 3 modelos son mayores)"
    log(f"  Base de calculo: {etiqueta}")
    log("")
    log(f"  {'folds':>6} {'semillas':>9} {'horas':>8} {'coste':>9}")
    for n_folds in (18, 54):
        for n_semillas in (1, 3):
            h = s_ronda * EPOCAS_TIPICAS * n_folds * n_semillas / 3600
            log(f"  {n_folds:6d} {n_semillas:9d} {h:8.1f} {'$' + format(h * 0.47, '.0f'):>9}")

    out = REPO / "results" / "reports" / "openbmi" / "bench_pod.json"
    out.write_text(json.dumps({"s_por_epoca": med, "epocas_tipicas": EPOCAS_TIPICAS,
                               "n_arquitecturas": 4}, indent=2))
    log(f"-> {out}")
    log("  Cifras SECUENCIALES. Si la GPU sigue infrautilizada con lote 256, ejecutar")
    log("  varios folds a la vez bajaria mas — medir antes de complicarlo.")


def limpiar_checkpoints(cache):
    """Los checkpoints se reanudan por ruta, y la ruta NO lleva los hiperparametros: sin
    borrarlos, la segunda variante cargaria los pesos de la primera y no mediria nada."""
    import shutil
    for d in (cache.parent / "checkpoints").glob("*/loso"):
        shutil.rmtree(d, ignore_errors=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fase", nargs="+", required=True,
                    choices=["gpu", "download", "cache", "verify", "export", "bench"])
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="procesos paralelos en la fase cache (por defecto, las CPUs "
                         "asignadas al contenedor menos una)")
    ap.add_argument("--hf-repo", default=None,
                    help="repo privado de HF donde subir el cache, p.ej. usuario/openbmi-ica")
    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--seguir-si-falla-verify", action="store_true",
                    help="continuar aunque verify no pase (NO usar para resultados de tesis)")
    args = ap.parse_args()

    for v in ("MNE_DATA", "OPENBMI_PROC_ROOT"):
        if not os.environ.get(v):
            sys.exit(f"Falta ${v}. Ejecuta:  source /workspace/env.sh")

    for fase in args.fase:
        log(f"########## FASE {fase.upper()} ##########")
        if fase == "gpu":
            fase_gpu()
        elif fase == "download":
            fase_download(args.subjects)
        elif fase == "cache":
            fase_cache(args.n_jobs or max(1, cpus_reales() - 1))
        elif fase == "verify":
            if not fase_verify() and not args.seguir_si_falla_verify:
                sys.exit("Verificacion fallida: se detiene aqui a proposito.")
        elif fase == "export":
            fase_export(args.hf_repo)
        elif fase == "bench":
            fase_bench()
    log("Fases completadas.")
    log("Antes de DESTRUIR el pod: comprueba que el cache esta subido (fase export).")


if __name__ == "__main__":
    main()
