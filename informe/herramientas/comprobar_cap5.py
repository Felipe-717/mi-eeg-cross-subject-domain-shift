from pathlib import Path
from collections import Counter
import re, sys
sys.path.insert(0, str(Path(__file__).parent))
from comprobar_estructura import expand, without_headings
ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'tmp/cap5_previo'
NEW = ROOT / 'informe/capitulos/05_resultados'

def chapter(root):
    def read(name):
        text = (root / name).read_text(encoding='utf-8')
        text = re.sub(r'(?<!\\)%[^\n]*', '', text)
        return re.sub(r'\\input\{capitulos/05_resultados/([^{}]+)\}', lambda m: read(m[1]+'.tex'), text)
    return read('05_resultados.tex')

def words(text):
    text = without_headings(text)
    text = re.sub(r'\b(?:subsección|sección)(?=~\\ref\{)', 'sección', text, flags=re.I)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    return Counter(re.findall(r'[^\W_]+', text.casefold()))

def main():
    before, after = chapter(OLD), chapter(NEW)
    missing, extra = words(before)-words(after), words(after)-words(before)
    oldlabels = Counter(re.findall(r'\\label\{([^{}]+)\}', before))
    alltext = expand(ROOT/'informe')
    labels = Counter(re.findall(r'\\label\{([^{}]+)\}', alltext))
    refs = set(re.findall(r'\\(?:ref|eqref|autoref|pageref)\{([^{}]+)\}', alltext))
    bad = {k: labels[k] for k in oldlabels if labels[k] != 1}
    dup = {k:v for k,v in labels.items() if v != 1}
    unresolved = sorted(refs-labels.keys())
    print('Palabras faltantes:', dict(missing))
    print('Palabras adicionales:', dict(extra))
    print('Etiquetas originales ausentes o repetidas:', bad)
    print('Todas las etiquetas repetidas:', dup)
    print('Referencias sin etiqueta:', unresolved)
    ok = not (missing or extra or bad or dup or unresolved)
    print('RESULTADO:', 'OK' if ok else 'ERROR')
    return 0 if ok else 1
if __name__ == '__main__':
    sys.exit(main())
