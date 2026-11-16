#!/usr/bin/env python
"""Avance 7 (octubre de 2026): TODOS los entrenamientos pendientes en un solo script.

Responde a las sugerencias del director de los avances 3 a 6 que necesitan GPU. Cada
trabajo es una llamada a `scripts/train_baseline.py`, `scripts/eval_cruzado.py` o
`scripts/construir_variantes.py`, con la configuración base de cada arquitectura
(ESTADO.md §2). Nada de lo ya publicado se re-ejecuta ni se sobrescribe: las variantes van
con `--variante`, o con un caché derivado cuyo nombre entra en el CSV y en el checkpoint.

Uso (en el pod, dentro de tmux):
    bash pod/avance_7/preparar.sh                 # una vez
    python pod/avance_7/avance_7.py --plan         # lista trabajos, horas estimadas y estado
    python pod/avance_7/avance_7.py                # ejecuta todo lo pendiente, en orden
    python pod/avance_7/avance_7.py --fases F5 F6  # solo esas fases (p. ej. en un 2.o pod)

Reanudable a dos niveles: la ejecución salta los trabajos marcados como hechos en
`results/avance_7/estado.json`, y dentro de cada trabajo `train_baseline.py` salta las
ejecuciones que ya están en su CSV. Si un trabajo falla, se registra y se sigue con el
siguiente; al final se listan los fallos. Relanzar reintenta solo lo que falló.

Tras cada trabajo se empaqueta `results/` en $AVANCE7_WS/avance_7_resultados.tgz, para poder
bajarlo en cualquier momento con scp sin esperar a que termine la ejecución
(`pod/avance_7/bajar.sh` lo hace en bucle desde el PC).

Todo vive en el disco del CONTENEDOR (/root/ws), no en un volumen de red: el 2026-11-01 el
volumen de EU-SE-1 se colgó a mitad de ejecución y dejó los procesos bloqueados en E/S sin
posibilidad de matarlos. El disco del contenedor se pierde al destruir el pod, por eso la
copia fuera del pod con bajar.sh es obligatoria.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WS = Path(os.environ.get("AVANCE7_WS", "/root/ws"))
CO = WS / "cache" / "epochs_cache_ica"                   # OpenBMI, 62 canales
CW = WS / "cache_wbcic" / "epochs_cache_ica"             # WBCIC-SHU
REP = REPO / "results" / "reports"
OUT = REPO / "results" / "avance_7"
ESTADO = OUT / "estado.json"
PAQUETE = WS / "avance_7_resultados.tgz"

MODELOS = ["eegnet", "conformer", "labram", "eegmamba"]
# Configuracion base de cada arquitectura, reconstruida por cfg_hash (ESTADO.md §2).
CFG = {
    "eegnet":    (["--lr", "2.8e-3", "--batch", "256"], {}),
    "conformer": (["--lr", "1.0e-3", "--batch", "256"], {}),
    "labram":    (["--lr", "1.0e-4", "--batch", "32"],
                  {"LABRAM_FINETUNE": "full", "LABRAM_UV_SCALE": "1"}),
    "eegmamba":  (["--lr", "1.0e-3", "--batch", "32"],
                  {"EEGMAMBA_HEAD": "flatten", "EEGMAMBA_UV_SCALE": "1", "AMP_BF16": "1"}),
}
# Horas de reloj MEDIDAS por semilla (CSV de las ejecuciones base, elapsed_s sumado),
# divididas por la aceleracion medida con trabajadores en paralelo: 1.35x en LOSO con 3,
# 7.3x en intra-sujeto con 8 (ESTADO.md §5).
# Ese 7.3x solo vale para los modelos pequenos. LaBraM y EEGMamba saturan la GPU (100 %) y
# los 8 trabajadores apenas aceleran: F2.within.labram (2 semillas) tardo 10.06 h MEDIDAS en
# la A40 el 2026-11-01, es decir 5.03 h por semilla (1.33x). EEGMamba se escala con la misma
# aceleracion a falta de medida propia.
H_LOSO = {"eegnet": 6.7 / 1.35, "conformer": 6.1 / 1.35, "labram": 21.3 / 1.35,
          "eegmamba": 25.0 / 1.35}
H_WITHIN = {"eegnet": 1.31 / 7.3, "conformer": 1.37 / 7.3, "labram": 10.06 / 2,
            "eegmamba": 10.99 / 1.33}
H_ALL = {"eegnet": 0.25, "conformer": 0.25, "labram": 0.6, "eegmamba": 0.7}   # punto6b: 92 min las 4

W_LOSO, W_WITHIN = 3, 8
PY = sys.executable
TB = ["scripts/train_baseline.py"]
EV = ["scripts/eval_cruzado.py"]


def J(id_, fase, desc, cmd, env=None, horas=0.0, guardia=None):
    return {"id": id_, "fase": fase, "desc": desc, "cmd": [str(c) for c in cmd],
            "env": env or {}, "horas": round(horas, 2), "guardia": guardia}


def _modelo(m):
    extra, env = CFG[m]
    return list(extra), dict(env)


# --------------------------------------------------------------------------- #
# Definicion de la ejecución
# --------------------------------------------------------------------------- #

def trabajos():
    T = []

    # F0 ---------------------------------------------------- caches derivados (CPU)
    cv = [PY, "scripts/construir_variantes.py"]
    T += [
        J("F0.ea62", "F0", "Alineamiento euclidiano, OpenBMI 62 canales",
          cv + ["--origen", CO, "--destino", f"{CO}_ea", "--tipo", "ea"], horas=0.3),
        J("F0.ea45", "F0", "Alineamiento euclidiano, OpenBMI vista comun (45)",
          cv + ["--origen", CO, "--destino", f"{CO}_ea45", "--tipo", "ea",
                "--canales", "comun_wbcic"], horas=0.2),
        J("F0.ea45w", "F0", "Alineamiento euclidiano, WBCIC vista comun (45)",
          cv + ["--origen", CW, "--destino", f"{CW}_ea45", "--tipo", "ea",
                "--dataset", "wbcic_2c", "--canales", "comun_wbcic"], horas=0.1),
    ]
    for nom, lo, hi in (("mu", 8, 13), ("beta", 13, 30), ("mubeta", 8, 30)):
        T.append(J(f"F0.{nom}", "F0", f"Subbanda {nom} {lo}-{hi} Hz, OpenBMI 62",
                   cv + ["--origen", CO, "--destino", f"{CO}_{nom}", "--tipo", "banda",
                         "--banda", lo, hi], horas=0.3))

    # F1 ---------------------------------------------------- prueba de humo
    # Una epoca, un fold, con --variante smoke*: sus filas no pueden colisionar con
    # ninguna ejecucion real (regla de ESTADO.md §7: el humo envenena la reanudacion).
    for m in MODELOS:
        ex, env = _modelo(m)
        T.append(J(f"F1.loso.{m}", "F1", f"Humo LOSO + few-shot {m}",
                   [PY] + TB + ["--mode", "loso", "--dataset", "openbmi", "--model", m,
                                "--cache", CO, "--loso-folds", 0, "--seeds", 2026,
                                "--max-epochs", 1, "--fewshot", "--variante", "smoke"] + ex,
                   env, horas=0.1))
    smoke_w = [PY] + TB + ["--mode", "within", "--dataset", "openbmi", "--model", "eegnet",
                           "--per-session", "--subjects", "sub-01", "--seeds", 2026,
                           "--max-epochs", 1] + _modelo("eegnet")[0]
    for nom, extra in (("cr", ["--aug", "cr"]), ("fft", ["--aug", "fft"]),
                       ("w02", ["--window", 0, 2]), ("ea", ["--cache", f"{CO}_ea"]),
                       ("mu", ["--cache", f"{CO}_mu"])):
        cache = [] if "--cache" in extra else ["--cache", CO]
        T.append(J(f"F1.within.{nom}", "F1", f"Humo within EEGNet variante {nom}",
                   smoke_w + cache + extra + ["--variante", f"smoke{nom}"], horas=0.02))
    # EEGMamba y LaBraM con la ventana corta: la que cambia el numero de parches.
    for m in ("labram", "eegmamba"):
        ex, env = _modelo(m)
        T.append(J(f"F1.w02.{m}", "F1", f"Humo within {m} ventana 0-2 s",
                   [PY] + TB + ["--mode", "within", "--dataset", "openbmi", "--model", m,
                                "--per-session", "--subjects", "sub-01", "--seeds", 2026,
                                "--max-epochs", 1, "--cache", CO, "--window", 0, 2,
                                "--variante", "smokew02"] + ex, env, horas=0.05))

    # F2 ---------------------------------------------------- semillas intra-sujeto
    # Homogeneiza el numero de semillas: hoy EEGNet y Conformer tienen 3 (2026-2028) y
    # LaBraM y EEGMamba 1. Se llevan las cuatro a 3, las mismas que en LOSO. Decision de
    # Felipe (2026-10-31) para acotar tiempo y gasto; 2758 y 9779 quedan sin usar.
    for m, semillas in (("labram", [2027, 2028]), ("eegmamba", [2027, 2028])):
        ex, env = _modelo(m)
        T.append(J(f"F2.within.{m}", "F2", f"Intra-sujeto {m}, semillas {semillas}",
                   [PY] + TB + ["--mode", "within", "--dataset", "openbmi", "--model", m,
                                "--channels", "completo", "--per-session", "--cache", CO,
                                "--workers", W_WITHIN, "--seeds"] + semillas + ex,
                   env, horas=H_WITHIN[m] * len(semillas),
                   guardia={"tipo": "filas", "patron": f"openbmi/baseline_{m}_within_completo_"
                            f"global_porsesion*_epochs_cache_ica.csv", "seeds": semillas,
                            "por_semilla": 1080}))

    # F3 ---------------------------------------------------- transferencia (--mode all)
    # Semillas 2027 y 2028 del modelo de 54 sujetos, del nativo de WBCIC y del 44 + 10,
    # que hoy tienen una sola. Y la variante con alineamiento euclidiano (semilla 2026).
    T += _transferencia([2027, 2028], CO, CW, "base")
    T += _transferencia([2026], Path(f"{CO}_ea45"), Path(f"{CW}_ea45"), "ea")

    # F4 ---------------------------------------------------- ablaciones intra-sujeto
    # Semilla 2026, contra la base intra-sujeto con esa misma semilla (pareado).
    abl = [("mu", ["--cache", f"{CO}_mu"]), ("beta", ["--cache", f"{CO}_beta"]),
           ("mubeta", ["--cache", f"{CO}_mubeta"]),
           ("w02", ["--cache", CO, "--window", 0, 2, "--variante", "w02"]),
           ("w03", ["--cache", CO, "--window", 0, 3, "--variante", "w03"]),
           ("augnone", ["--cache", CO, "--aug", "none", "--variante", "augnone"]),
           ("augcr", ["--cache", CO, "--aug", "cr", "--variante", "augcr"]),
           ("augfft", ["--cache", CO, "--aug", "fft", "--variante", "augfft"])]
    # LaBraM y EEGMamba cuestan ~5 y ~8 h por ablacion (la GPU no escala con 8 trabajadores).
    # Decision de Felipe (2026-11-01): en ellos solo las dos aumentaciones que pidio el
    # director y la ventana 0-2 s (cambia el numero de parches). Bandas, w03 y augnone
    # quedan probadas solo en EEGNet y Conformer. AVANCE7_F4_GRANDES=todas restaura las 8.
    grandes = os.environ.get("AVANCE7_F4_GRANDES", "augcr augfft w02").split()
    if grandes == ["todas"]:
        grandes = []
    for nom, extra in abl:
        for m in MODELOS:
            if grandes and m in ("labram", "eegmamba") and nom not in grandes:
                continue
            ex, env = _modelo(m)
            T.append(J(f"F4.{nom}.{m}", "F4", f"Intra-sujeto {m}, ablacion {nom}",
                       [PY] + TB + ["--mode", "within", "--dataset", "openbmi", "--model", m,
                                    "--channels", "completo", "--per-session",
                                    "--workers", W_WITHIN, "--seeds", 2026] + extra + ex,
                       env, horas=H_WITHIN[m]))

    # F5-F8 ------------------------------------------------- LOSO
    # Cada fase son las 4 arquitecturas. La calibracion con pocos ensayos va en todas las
    # LOSO nuevas: no anade entrenamiento y los modelos LOSO base no se conservaron.
    def loso(fase, nom, seeds, extra, fewshot=True):
        out = []
        for m in MODELOS:
            ex, env = _modelo(m)
            out.append(J(f"{fase}.{m}", fase, f"LOSO {m} {nom}, semillas {seeds}",
                         [PY] + TB + ["--mode", "loso", "--dataset", "openbmi", "--model", m,
                                      "--channels", "completo", "--workers", W_LOSO,
                                      "--seeds"] + seeds + extra + ex
                         + (["--fewshot"] if fewshot else []),
                         env, horas=H_LOSO[m] * len(seeds)))
        return out

    T += loso("F5", "semilla 2027", [2027], ["--cache", CO])
    T += loso("F6", "alineamiento euclidiano", [2026], ["--cache", f"{CO}_ea"])
    T += loso("F7", "semilla 2028", [2028], ["--cache", CO])
    T += loso("F8", "reflexion de canales", [2026],
              ["--cache", CO, "--aug", "cr", "--variante", "augcr"], fewshot=False)
    # Opcionales: no entran en la ejecucion por defecto (ver OPCIONALES).
    T += loso("F9", "subbanda 8-30 Hz", [2026], ["--cache", f"{CO}_mubeta"], fewshot=False)
    T += loso("F10", "ruido en Fourier", [2026],
              ["--cache", CO, "--aug", "fft", "--variante", "augfft"], fewshot=False)
    T += loso("F11", "semillas 2758 y 9779", [2758, 9779], ["--cache", CO])
    return T


OPCIONALES = {"F9", "F10", "F11"}


def _transferencia(seeds, co, cw, nom):
    """Modelo de 54 sujetos + nativo de WBCIC + 44 + 10, con sus evaluaciones cruzadas."""
    T = []
    dst = OUT / "transferencia" / nom
    for m in MODELOS:
        ex, env = _modelo(m)
        tag = {"labram": "_full_uv1_lr0.0001", "eegmamba": "_flatten_uv1_lr0.001"}.get(m, "")
        # Los checkpoints de --mode all se conservan (son pocos y los usa la evaluacion
        # cruzada); los de intra-sujeto y LOSO van a /raw y se borran al terminar.
        ck = WS / "ckpt_avance_7"
        env["TESIS_CKPT"] = str(ck)
        for s in seeds:
            base = [PY] + TB + ["--mode", "all", "--model", m, "--channels", "comun_wbcic",
                                "--seeds", s] + ex
            g_all = {"tipo": "all", "min_s": 20, "ea": co.name.endswith("_ea45"),
                     "ds": "openbmi"}
            # 1. transferencia: 54 sujetos de OpenBMI -> WBCIC
            run = f"all_openbmi_seed{s}_comun_wbcic_global{tag}"
            pt = ck / f"{m}_{co.name}" / "all" / f"{run}.pt"
            T.append(J(f"F3.{nom}.{m}.s{s}.trans", "F3",
                       f"Transferencia {m} ({nom}) s{s}: entrenar con 54 de OpenBMI",
                       base + ["--dataset", "openbmi", "--cache", co], env, H_ALL[m],
                       g_all | {"run": run, "n_subjects": 54}))
            T.append(J(f"F3.{nom}.{m}.s{s}.trans_eval", "F3",
                       f"Transferencia {m} ({nom}) s{s}: evaluar en el 20 % de WBCIC",
                       [PY] + EV + ["--ckpt", pt, "--dataset", "wbcic_2c", "--cache", cw,
                                    "--solo-reservado", 0.20,
                                    "--out", dst / f"cruzado_{m}_s{s}.csv"], env, 0.05))
            # 2. nativo: 51 de WBCIC con el 20 % reservado
            T.append(J(f"F3.{nom}.{m}.s{s}.nativo", "F3",
                       f"Nativo WBCIC {m} ({nom}) s{s}",
                       base + ["--dataset", "wbcic_2c", "--cache", cw, "--frac-test", 0.20],
                       env, H_ALL[m] * 0.4,
                       g_all | {"run": f"all_wbcic_2c_seed{s}_comun_wbcic_global{tag}",
                                "ds": "wbcic_2c"}))
            # 3. 44 + 10: mismo modelo sobre vistos, 10 apartados y WBCIC
            run44 = f"{run}_hold10s2026"
            pt44 = ck / f"{m}_{co.name}" / "all" / f"{run44}.pt"
            T.append(J(f"F3.{nom}.{m}.s{s}.h44", "F3", f"44 + 10 {m} ({nom}) s{s}: entrenar",
                       base + ["--dataset", "openbmi", "--cache", co, "--holdout-n", 10,
                               "--holdout-seed", 2026], env, H_ALL[m] * 0.8,
                       g_all | {"run": run44, "n_subjects": 44}))
            T.append(J(f"F3.{nom}.{m}.s{s}.h44_eval10", "F3",
                       f"44 + 10 {m} ({nom}) s{s}: evaluar en los 10 apartados",
                       [PY, "pod/avance_7/eval_apartados.py", "--ckpt", pt44, "--cache", co,
                        "--out", dst / f"holdout_openbmi_{m}_s{s}.csv"], env, 0.05))
            T.append(J(f"F3.{nom}.{m}.s{s}.h44_evalw", "F3",
                       f"44 + 10 {m} ({nom}) s{s}: evaluar en el 20 % de WBCIC",
                       [PY] + EV + ["--ckpt", pt44, "--dataset", "wbcic_2c", "--cache", cw,
                                    "--solo-reservado", 0.20,
                                    "--out", dst / f"holdout_wbcic_{m}_s{s}.csv"], env, 0.05))
    return T


# --------------------------------------------------------------------------- #
# Ejecucion
# --------------------------------------------------------------------------- #

def _estado():
    if ESTADO.exists():
        return json.loads(ESTADO.read_text(encoding="utf-8"))
    return {}


def _guardar_estado(est):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = ESTADO.with_suffix(".tmp")
    tmp.write_text(json.dumps(est, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(ESTADO)


def _guardia(job):
    """Comprobacion posterior: que el trabajo produjo lo que debia y no reanudo algo ajeno."""
    g = job.get("guardia")
    if not g:
        return None
    import pandas as pd
    if g["tipo"] == "filas":
        rutas = [p for p in glob.glob(str(REP / g["patron"])) if "__" not in Path(p).name]
        if not rutas:
            return f"no aparecio ningun CSV con el patron {g['patron']}"
        d = pd.concat([pd.read_csv(p) for p in rutas])
        for s in g["seeds"]:
            n = int((d["seed"] == s).sum())
            if n < g["por_semilla"]:
                return f"semilla {s}: {n} filas de {g['por_semilla']}"
        return None
    if g["tipo"] == "all":
        # Mismo run_id en la base y en la variante alineada (solo cambia el cache, que va
        # en el nombre del CSV): se filtra por dataset y por sufijo del cache.
        filas = []
        for p in glob.glob(str(REP / g["ds"] / "baseline_*_all_*.csv")):
            nombre = Path(p).name
            if "__" in nombre or ("_ea45" in nombre) != g["ea"]:
                continue
            d = pd.read_csv(p)
            filas.append(d[d["run_id"].astype(str) == g["run"]])
        f = pd.concat(filas) if filas else pd.DataFrame()
        if f.empty:
            return f"no aparece la fila {g['run']}"
        f = f.iloc[-1]
        if float(f["elapsed_s"]) < g["min_s"]:
            return f"solo {f['elapsed_s']} s: reanudo un checkpoint ajeno"
        if "n_subjects" in g and int(f["n_subjects"]) != g["n_subjects"]:
            return f"{f['n_subjects']} sujetos, se esperaban {g['n_subjects']}"
    return None


def _empaquetar():
    try:
        tmp = PAQUETE.with_suffix(".tmp")
        with tarfile.open(tmp, "w:gz") as t:
            for sub in ("results", "logs"):
                if (REPO / sub).exists():
                    t.add(REPO / sub, arcname=sub)
        tmp.replace(PAQUETE)
    except Exception as e:                       # empaquetar nunca debe tumbar la ejecución
        print(f"  [aviso] no se pudo empaquetar: {e}", flush=True)


def ejecutar(job, est):
    log = OUT / "logs" / f"{job['id']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for k in ("LABRAM_FINETUNE", "LABRAM_UV_SCALE", "EEGMAMBA_HEAD", "EEGMAMBA_UV_SCALE",
              "AMP_BF16"):
        env.pop(k, None)                        # que no se filtren de un modelo al siguiente
    env.update({"TESIS_EXIGIR_GPU": "1", "TESIS_BORRAR_CKPT": "1",
                "OPENBMI_PROC_ROOT": str(WS / "cache"), "PYTHONUNBUFFERED": "1"})
    env.update(job["env"])
    t0 = time.time()
    print(f"\n### {job['id']}  {job['desc']}  (est. {job['horas']:.1f} h)", flush=True)
    print("    " + " ".join(job["cmd"]), flush=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(job['cmd'])}\n")
        fh.flush()
        rc = subprocess.call(job["cmd"], cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
    horas = (time.time() - t0) / 3600
    error = f"codigo de salida {rc}" if rc else _guardia(job)
    est[job["id"]] = {"ok": error is None, "error": error, "horas": round(horas, 3),
                      "fin": time.strftime("%Y-%m-%d %H:%M:%S")}
    _guardar_estado(est)
    print(f"    {'OK' if error is None else 'FALLO: ' + error}  ({horas:.2f} h)  log: {log}",
          flush=True)
    _empaquetar()
    return error is None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fases", nargs="*", default=None,
                    help="fases a ejecutar (por defecto todas salvo las opcionales "
                         f"{sorted(OPCIONALES)})")
    ap.add_argument("--solo", nargs="*", default=None, help="ids concretos de trabajo")
    ap.add_argument("--plan", action="store_true", help="lista y estima; no ejecuta")
    ap.add_argument("--rehacer", nargs="*", default=[],
                    help="ids que se reejecutan aunque consten como hechos")
    a = ap.parse_args()

    todos = trabajos()
    fases = set(a.fases) if a.fases else {j["fase"] for j in todos} - OPCIONALES
    sel = [j for j in todos if j["fase"] in fases and (not a.solo or j["id"] in a.solo)]
    est = _estado()
    pend = [j for j in sel if not est.get(j["id"], {}).get("ok") or j["id"] in a.rehacer]

    print(f"Ejecución: {len(sel)} trabajos seleccionados, {len(pend)} pendientes")
    por_fase = {}
    for j in sel:
        f = por_fase.setdefault(j["fase"], [0, 0.0, 0])
        f[0] += 1
        f[1] += j["horas"] if j in pend else 0
        f[2] += int(bool(est.get(j["id"], {}).get("ok")))
    for f, (n, h, ok) in sorted(por_fase.items(), key=lambda x: int(x[0][1:])):
        print(f"  {f:<4} {n:3d} trabajos  {ok:3d} hechos  pendiente ~{h:6.1f} h")
    total = sum(j["horas"] for j in pend)
    print(f"  TOTAL pendiente ~{total:.0f} h de reloj (~{total / 24:.1f} dias en una GPU)")
    if a.plan:
        for j in sel:
            e = est.get(j["id"], {})
            marca = "ok " if e.get("ok") else ("ERR" if e else "   ")
            print(f"  [{marca}] {j['id']:<34} {j['horas']:5.2f} h  {j['desc']}")
        return 0

    # Los fallos del humo (F1) detienen todo: si una variante no arranca, cada trabajo
    # de las fases largas fallaria igual tras horas de cola.
    t0 = time.time()
    fallos = []
    h_est_hechas = h_real = 0.0
    for i, j in enumerate(pend):
        # ETA: lo que queda segun la estimacion, corregido por la razon real/estimado de
        # los trabajos ya terminados en esta ejecucion (1 hasta que haya alguno).
        resto = sum(x["horas"] for x in pend[i:])
        factor = h_real / h_est_hechas if h_est_hechas > 0 else 1.0
        eta = time.time() + resto * factor * 3600
        print(f"\n[{i + 1}/{len(pend)}] {time.strftime('%m-%d %H:%M')} | "
              f"transcurrido {(time.time() - t0) / 3600:.1f} h | "
              f"quedan ~{resto * factor:.0f} h (factor {factor:.2f}) | "
              f"fin ~{time.strftime('%m-%d %H:%M', time.localtime(eta))} (hora del pod, UTC)",
              flush=True)
        t_j = time.time()
        ok = ejecutar(j, est)
        h_real += (time.time() - t_j) / 3600
        h_est_hechas += j["horas"]
        if not ok:
            fallos.append(j["id"])
            if j["fase"] in ("F0", "F1"):
                print(f"\n!! Fallo en la preparacion ({j['id']}): se detiene la ejecución. "
                      f"Revisa {OUT / 'logs' / (j['id'] + '.log')}", flush=True)
                return 1
    print(f"\n=== Ejecución terminada en {(time.time() - t0) / 3600:.1f} h | "
          f"fallos: {len(fallos)} {fallos}")
    _empaquetar()
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
