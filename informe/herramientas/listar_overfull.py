"""Lista los Overfull hbox/vbox de main.log mayores a un umbral, con el ultimo .tex abierto antes de cada uno."""
import re
import sys
from pathlib import Path

UMBRAL = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
log = (Path(__file__).resolve().parents[1] / "main.log").read_text(encoding="utf-8", errors="replace")
log = log.replace("\n", "")  # el log corta las rutas largas a 79 columnas

archivos = [(m.start(), m.group(1)) for m in re.finditer(r"\((\.?/?[\w./-]+\.tex)", log)]
salida = []
for m in re.finditer(r"Overfull \\([hv])box \((\d+\.\d+)pt too \w+\)( in paragraph at lines (\d+)--(\d+))?", log):
    if float(m.group(2)) <= UMBRAL:
        continue
    previo = [f for pos, f in archivos if pos < m.start()]
    salida.append(f"{previo[-1] if previo else '?'}  {m.group(1)}box {m.group(2)}pt  lineas {m.group(4)}-{m.group(5)}")
print("\n".join(salida) if salida else "ninguno")
