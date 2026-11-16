"""Lista los .tex del informe con tildes perdidas (patrón letra, '?', letra) o que no son UTF-8."""
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PATRON = re.compile(r"[A-Za-z]\?[A-Za-z]")

malos = 0
for f in sorted(RAIZ.rglob("*.tex")):
    try:
        texto = f.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"NO UTF-8: {f.relative_to(RAIZ)}")
        malos += 1
        continue
    n = len(PATRON.findall(texto))
    if n:
        print(f"{f.relative_to(RAIZ)}: {n}")
        malos += 1
print("OK" if not malos else f"{malos} archivos con problemas")
sys.exit(1 if malos else 0)
