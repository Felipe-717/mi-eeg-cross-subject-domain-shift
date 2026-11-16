from pathlib import Path
from collections import Counter
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'tmp/informe_previo'
NEW = ROOT / 'informe'

def expand(root, name='main.tex'):
    text = (root / name).read_text(encoding='utf-8')
    text = re.sub(r'(?<!\\)%[^\n]*', '', text)
    def include(match):
        return expand(root, match.group(1) + '.tex')
    return re.sub(r'\\(?:input|include)\{([^{}]+)\}', include, text)

def without_headings(text):
    pattern = re.compile(r'\\(?:chapter|section|subsection|subsubsection|paragraph)\*?(?:\[[^]]*\])?\s*\{')
    while match := pattern.search(text):
        i, depth = match.end(), 1
        while depth and i < len(text):
            if text[i] == '{' and (i == 0 or text[i-1] != '\\'):
                depth += 1
            elif text[i] == '}' and (i == 0 or text[i-1] != '\\'):
                depth -= 1
            i += 1
        if depth:
            raise ValueError('Título sin cerrar')
        text = text[:match.start()] + ' ' + text[i:]
    return text

def words(text):
    text = without_headings(text)
    text = text.replace('de los capítulos~\\ref{cap:resultados} y~\\ref{cap:validacion-externa}', 'del capítulo~\\ref{cap:resultados} y de la sección~\\ref{cap:validacion-externa}')
    text = re.sub(r'(?i)\b(?:el capítulo|la sección)(?=~\\ref\{cap:validacion-externa\})', 'sección', text)
    text = re.sub(r'\\(?:label|ref|eqref)\{[^{}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'(?i)\b(?:capítulo|sección)\b', '', text)
    return Counter(re.findall(r'[^\W_]+', text.casefold(), re.UNICODE))

def main():
    before, after = expand(OLD), expand(NEW)
    missing, extra = words(before) - words(after), words(after) - words(before)
    original = Counter(re.findall(r'\\label\{([^{}]+)\}', before))
    labels = Counter(re.findall(r'\\label\{([^{}]+)\}', after))
    refs = set(re.findall(r'\\(?:ref|eqref|autoref|pageref)\{([^{}]+)\}', after))
    bad = {k: labels[k] for k in original if labels[k] != 1}
    dup = {k: v for k, v in labels.items() if v != 1}
    unresolved = sorted(refs - labels.keys())
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
