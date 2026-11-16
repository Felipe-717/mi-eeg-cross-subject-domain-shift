#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
monitor.py — vista agregada del avance de TODOS los workers, en vivo.

El problema que resuelve: `_repartir` lanza N procesos y cada uno escribe su propio
CSV y su propio log. El ETA que imprime cada worker es SOLO el suyo, asi que con 8
workers hay 8 ETA parciales y ninguna cifra global. Este script junta todo.

Fuente de verdad = los CSV por worker (una fila = una corrida terminada). Los logs se
leen solo para saber que esta haciendo cada worker AHORA (fold y epoca en curso), que
el CSV no puede decir porque la fila se escribe al terminar.

Uso:
    python scripts/monitor.py --semillas 2                    # descubre trabajos solo
    python scripts/monitor.py --semillas 2 --intervalo 30
    python scripts/monitor.py --semillas 2 --una-vez          # una foto y sale

    # limitar a ciertos trabajos (subcadena del nombre del CSV)
    python scripts/monitor.py --semillas 2 --filtro loso

Se puede dejar corriendo en una ventana de tmux aparte sin tocar el entrenamiento:
solo lee archivos, nunca escribe.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# Filas esperadas POR SEMILLA en cada escenario. Salen del diseno del barrido:
#   loso   -> 54 folds (un sujeto excluido cada vez)
#   within -> 54 sujetos x 20 (folds x sesiones)
FILAS_POR_SEMILLA = {"loso": 54, "within": 1080}

RE_EPOCA = re.compile(r"ep\s+(\d+)/(\d+).*?([\d.]+)\s*s/ep")
RE_PROG = re.compile(r"\[(\d+)/(\d+)\s+(\d+)%")
RE_FOLD = re.compile(r"fold\s+(\d+)\s+\(([^)]+)\)")
RE_RESTA = re.compile(r"~\s*(\d+)\s*min rest")


def _escenario(nombre: str) -> str:
    return "loso" if "_loso_" in nombre else "within"


def descubrir(dir_csv: Path, dir_log: Path, filtro: str | None) -> dict[str, list[int]]:
    """Agrupa los workers por su base comun, devolviendo {base: [ids de worker]}.

    Se miran los LOGS ademas de los CSV a proposito. Un shard `<base>__w<N>.csv` no existe
    hasta que ese worker TERMINA su primera corrida, y en LOSO eso son 23 min (EEGNet) o 37
    (Conformer). Descubriendo solo por CSV, el monitor pasaba media hora sin listar un solo
    worker y pareciendo que no habia nada corriendo. El log se crea al arrancar.
    """
    grupos: dict[str, set[int]] = defaultdict(set)
    for patron, carpeta in ((f"*__w*.csv", dir_csv), (f"*__w*.log", dir_log)):
        if not carpeta.exists():
            continue
        for p in carpeta.glob(patron):
            m = re.search(r"^(.*)__w(\d+)$", p.stem)
            if not m:
                continue
            base, w = m.group(1), int(m.group(2))
            if "__bench" in base:                 # shards del benchmark, no son trabajo real
                continue
            if filtro and filtro not in base:
                continue
            grupos[base].add(w)
    return {b: sorted(ws) for b, ws in grupos.items()}


def leer_shards(shards: list[Path]) -> tuple[int, dict[str, int], set[str]]:
    """Devuelve (filas unicas, filas por semilla, run_ids)."""
    vistos: set[str] = set()
    por_semilla: dict[str, int] = defaultdict(int)
    for p in shards:
        try:
            with open(p, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    rid = r.get("run_id")
                    if rid and rid not in vistos:
                        vistos.add(rid)
                        por_semilla[r.get("seed", "?")] += 1
        except (OSError, csv.Error):
            continue                              # el worker puede estar escribiendo
    return len(vistos), dict(por_semilla), vistos


def estado_worker(log: Path) -> dict:
    """Ultimo estado conocido de un worker, leyendo la cola de su log."""
    est = {"vivo": False, "edad": None, "epoca": "", "fold": "", "prog": "", "s_ep": None,
           "resta": ""}
    if not log.exists():
        return est
    est["edad"] = time.time() - log.stat().st_mtime
    try:
        with open(log, "rb") as fh:                # solo la cola: los logs crecen mucho
            fh.seek(0, os.SEEK_END)
            tam = fh.tell()
            fh.seek(max(0, tam - 8192))
            cola = fh.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return est
    for ln in reversed(cola):
        if not est["epoca"]:
            m = RE_EPOCA.search(ln)
            if m:
                est["epoca"] = f"ep {m.group(1)}/{m.group(2)}"
                est["s_ep"] = float(m.group(3))
        if not est["fold"]:
            m = RE_FOLD.search(ln)
            if m:
                est["fold"] = f"fold {m.group(1)} ({m.group(2)})"
        if not est["resta"]:
            m = RE_RESTA.search(ln)
            if m:
                est["resta"] = f"~{m.group(1)}min"
        if not est["prog"]:
            m = RE_PROG.search(ln)
            if m:
                est["prog"] = f"{m.group(1)}/{m.group(2)}"
        if est["epoca"] and est["fold"] and est["prog"] and est["resta"]:
            break
    return est


def gpu() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8)
        if out.returncode != 0:
            return []
        filas = []
        for ln in out.stdout.strip().splitlines():
            i, u, mu, mt, t = [x.strip() for x in ln.split(",")]
            filas.append(f"GPU{i}  {u:>3}% util   {int(mu)/1024:5.1f}/{int(mt)/1024:.0f} GB   {t}C")
        return filas
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def n_procesos(patron: str = "train_baseline") -> int:
    try:
        out = subprocess.run(["pgrep", "-fc", patron], capture_output=True, text=True, timeout=8)
        return int(out.stdout.strip() or 0)
    except (OSError, subprocess.SubprocessError, ValueError):
        return -1


def dur(seg: float) -> str:
    if seg is None or seg < 0 or seg != seg:
        return "  ?  "
    seg = int(seg)
    if seg < 3600:
        return f"{seg // 60:>2}min"
    return f"{seg // 3600:>2}h{(seg % 3600) // 60:02d}"


def pintar(dir_csv: Path, dir_log: Path, n_semillas: int, filtro: str | None,
           hist: dict, ancho: int) -> bool:
    """Dibuja una foto. Devuelve True si queda trabajo pendiente."""
    grupos = descubrir(dir_csv, dir_log, filtro)
    ahora = time.time()
    lineas = []
    total_hechas = total_esperadas = 0
    en_curso = 0
    pendiente = False

    if not grupos:
        lineas.append("  (todavia no hay shards; los workers aun no han terminado su primera corrida)")
        # Sin shards no se puede concluir NADA. Devolver 'pendiente' aqui es lo que impide que
        # el monitor anuncie "TODO COMPLETO" en su segundo refresco, cuando lo unico que pasa
        # es que la primera corrida (6 min en LOSO) todavia no ha terminado de escribirse.
        pendiente = True

    for base, workers in sorted(grupos.items()):
        esc = _escenario(base)
        shards = [dir_csv / f"{base}__w{w}.csv" for w in workers]
        shards = [p for p in shards if p.exists()]
        hechas, por_semilla, _ = leer_shards(shards)
        esperadas = FILAS_POR_SEMILLA[esc] * n_semillas
        total_hechas += hechas
        total_esperadas += esperadas
        if hechas < esperadas:
            pendiente = True

        pct = 100 * hechas / esperadas if esperadas else 0
        barra = "#" * int(pct / 100 * 28) + "." * (28 - int(pct / 100 * 28))
        corto = base.replace("baseline_", "").replace("_completo_global", "")
        corto = corto.replace("_epochs_cache_ica", "").replace("_porsesion", "/sesion")
        lineas.append(f"  {corto[:46]:<46} [{barra}] {hechas:>5}/{esperadas:<5} {pct:5.1f}%")

        det = "  ".join(f"{s}:{n}" for s, n in sorted(por_semilla.items()))
        lineas.append(f"      por semilla: {det if det else '(ninguna aun)'}")

        for w in workers:
            est = estado_worker(dir_log / f"{base}__w{w}.log")
            edad = est["edad"]
            if edad is None:
                marca, nota = "?", "sin log todavia"
            elif edad > 900:
                marca, nota = "X", f"SIN AVANCE hace {dur(edad)}"
            elif edad > 300:
                marca, nota = "!", f"quieto {dur(edad)}"
            else:
                marca, nota = "o", ""
                if est["epoca"]:
                    en_curso += 1
            sep = f"{est['s_ep']:.1f}s/ep" if est["s_ep"] else ""
            lineas.append(f"      {marca} w{str(w):<3} {est['prog']:>7}  {est['fold'][:26]:<26} "
                          f"{est['epoca']:<11} {sep:>9}  {est['resta']:>9}  {nota}")
        lineas.append("")

    # --- rendimiento global, medido entre refrescos --------------------------- #
    hist.setdefault("muestras", []).append((ahora, total_hechas))
    hist["muestras"] = [m for m in hist["muestras"] if ahora - m[0] <= 3600][-400:]
    cab = []
    if len(hist["muestras"]) >= 2:
        t0, h0 = hist["muestras"][0]
        dt, dh = ahora - t0, total_hechas - h0
        if dt > 30 and dh > 0:
            ritmo = dh / (dt / 60)
            faltan = total_esperadas - total_hechas
            eta = faltan / ritmo * 60 if ritmo > 0 else None
            cab.append(f"  ritmo {ritmo:5.2f} corridas/min  (ventana {dur(dt)})")
            cab.append(f"  ETA GLOBAL ~{dur(eta)}   (fin aprox. "
                       f"{time.strftime('%H:%M', time.localtime(ahora + eta))})")
        else:
            cab.append("  ritmo: midiendo, hace falta que termine alguna corrida...")
    else:
        cab.append("  ritmo: midiendo...")

    pct_g = 100 * total_hechas / total_esperadas if total_esperadas else 0
    barra_g = "#" * int(pct_g / 100 * 44) + "." * (44 - int(pct_g / 100 * 44))

    print("\033[2J\033[H", end="")
    print("=" * min(ancho, 100))
    print(f" MONITOR  {time.strftime('%Y-%m-%d %H:%M:%S')}   "
          f"procesos train_baseline: {n_procesos()}")
    print("=" * min(ancho, 100))
    print(f"  TOTAL [{barra_g}] {total_hechas}/{total_esperadas}  {pct_g:.1f}%"
          f"   +{en_curso} en curso")
    if total_hechas == 0 and en_curso:
        print("  (0% es normal al arrancar: una corrida LOSO tarda 20-40 min en cerrarse;")
        print("   mira la columna s/ep de cada worker, eso si avanza en vivo)")
    for c in cab:
        print(c)
    print()
    for g in gpu():
        print("  " + g)
    print()
    print("  columnas: corridas  fold  epoca  s/ep  restante | o activo  ! quieto>5min  X muerto>15min")
    print("-" * min(ancho, 100))
    for ln in lineas:
        print(ln)
    sys.stdout.flush()
    return pendiente


def main() -> int:
    ap = argparse.ArgumentParser(description="Monitor agregado de los workers de entrenamiento.")
    ap.add_argument("--csv-dir", default="results/reports/openbmi")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--semillas", type=int, required=True,
                    help="cuantas semillas se estan ejecutando (para calcular el total)")
    ap.add_argument("--filtro", default=None, help="subcadena para limitar los trabajos mostrados")
    ap.add_argument("--intervalo", type=int, default=30)
    ap.add_argument("--una-vez", action="store_true")
    a = ap.parse_args()

    dir_csv, dir_log = Path(a.csv_dir), Path(a.log_dir)
    if not dir_csv.exists():
        print(f"x no existe {dir_csv}", file=sys.stderr)
        return 2
    ancho = shutil.get_terminal_size((100, 40)).columns
    hist: dict = {}

    if a.una_vez:
        pintar(dir_csv, dir_log, a.semillas, a.filtro, hist, ancho)
        return 0
    try:
        while True:
            queda = pintar(dir_csv, dir_log, a.semillas, a.filtro, hist, ancho)
            # Solo se declara el fin si de verdad hay trabajo contado y esta todo hecho. La
            # comprobacion del total (>0) es la que evita salir por "no encontre nada".
            ultimo = hist["muestras"][-1][1] if hist.get("muestras") else 0
            if not queda and ultimo > 0:
                print("\n  TODO COMPLETO.")
                return 0
            time.sleep(a.intervalo)
    except KeyboardInterrupt:
        print("\n  (monitor detenido; el entrenamiento sigue)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
