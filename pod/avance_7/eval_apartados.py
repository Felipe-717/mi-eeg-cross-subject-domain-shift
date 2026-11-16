"""Evalúa un modelo 44 + 10 sobre SUS 10 sujetos apartados de OpenBMI.

Los sujetos salen del `.meta.json` del checkpoint, no de una lista escrita a mano, igual
que en pod/punto6b/correr.sh: así no se puede evaluar por error sobre alguien que estuvo
en el entrenamiento (eval_cruzado.py lo comprueba otra vez).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, type=Path)
ap.add_argument("--cache", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
meta = json.loads(a.ckpt.with_suffix(".meta.json").read_text(encoding="utf-8"))
apartados = meta["subs_apartados"]
if len(apartados) != 10:
    sys.exit(f"x el checkpoint tiene {len(apartados)} apartados, se esperaban 10")
print("apartados:", " ".join(apartados), flush=True)
raise SystemExit(subprocess.call(
    [sys.executable, "scripts/eval_cruzado.py", "--ckpt", str(a.ckpt), "--dataset", "openbmi",
     "--cache", a.cache, "--subjects", *apartados, "--out", a.out]))
