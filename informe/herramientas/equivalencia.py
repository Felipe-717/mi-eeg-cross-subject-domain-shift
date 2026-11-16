from difflib import SequenceMatcher
from pathlib import Path
import re
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
REFERENCIA = ROOT.parent / 'Avance_Metodologia_20261108.pdf'

def palabras(path):
    pages = PdfReader(path).pages
    text = '\n'.join(page.extract_text() or '' for page in pages)
    text = re.sub(r'-\s*\n\s*', '', text.lower())
    text = re.sub(r'(?m)^\s*(?:\d+|[ivxlcdm]+)\s*$', '', text)
    return re.findall(r'[^\W_]+(?:[.,][0-9]+)*', text, flags=re.UNICODE), len(pages)

def comparar(a, b):
    x, nx = palabras(a)
    y, ny = palabras(b)
    ops = SequenceMatcher(None, x, y, autojunk=False).get_opcodes()
    cambios = [op for op in ops if op[0] != 'equal']
    lines = [f'{a}: {nx} páginas; {b}: {ny} páginas', f'Cambios sin clasificar: {len(cambios)}']
    for tag, i, j, k, l in cambios:
        lines.append(f'{tag}: {" ".join(x[i:j])} -> {" ".join(y[k:l])}')
    return '\n'.join(lines)

def main():
    informes = []
    for pdf in (ROOT / 'main.pdf', ROOT.parent / 'Overleaf/main.pdf'):
        if pdf.exists() and REFERENCIA.exists():
            informes.append(comparar(REFERENCIA, pdf))
        else:
            informes.append(f'No se puede comparar: falta {pdf if not pdf.exists() else REFERENCIA}')
    result = '\n\n'.join(informes) + '\n'
    (ROOT / 'herramientas/equivalencia.txt').write_text(result, encoding='utf-8')
    print(result)

if __name__ == '__main__':
    main()
