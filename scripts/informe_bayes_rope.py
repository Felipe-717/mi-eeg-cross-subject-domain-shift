#!/usr/bin/env python
"""Rangos con signo bayesianos sobre medias por sujeto de tres semillas."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from baycomp import SignedRankTest

import avance7_semillas as X

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / 'results/informe_final'
ROPE = (0.01, 0.02, 0.03)


def probabilidades(a, b, margen):
    # baycomp devuelve primero la probabilidad de x > y.
    return SignedRankTest.probs(a, b, rope=margen, nsamples=50000, random_state=2026)


def comprobar():
    a = np.linspace(0.5, 0.8, 54)
    p = probabilidades(a, a, 0.03)
    assert p[1] > 0.99, p
    p = probabilidades(a, a + 0.10, 0.03)
    assert p[0] < 0.01 and p[2] > 0.99, p


def filas(vectores, protocolo, degradacion=False):
    salida = []
    for a, b in X.PARES:
        va, vb = vectores[a], vectores[b]
        assert va.index.equals(vb.index) and len(va) == 54
        for margen in ROPE:
            # Menor degradacion es mejor; se invierte el signo para la prueba.
            pa, pe, pb = probabilidades(-va if degradacion else va, -vb if degradacion else vb, margen)
            decision = next((nombre for p, nombre in ((pa, 'A mejor'), (pe, 'equivalentes'),
                                                        (pb, 'B mejor')) if p >= 0.95), 'indeciso')
            salida.append(dict(protocolo=protocolo, par=f'{a}-{b}', A=a, B=b, n=54,
                               rope=margen, diferencia_media=float((va-vb).mean()),
                               p_A_mejor=pa, p_equivalentes=pe, p_B_mejor=pb, decision=decision))
    return salida


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    comprobar()
    exactitud = []
    for proto in ('within', 'loso'):
        vectores = {a: X.mat(proto, a).mean(axis=1) for a in X.ARQ}
        exactitud.extend(filas(vectores, proto))
    degradacion = {a: X.mat('loso', a, 'acc_visto').mean(axis=1)
                      - X.mat('loso', a, 'acc').mean(axis=1) for a in X.ARQ}
    deg = pd.DataFrame(filas(degradacion, 'loso_visto_menos_no_visto', True))
    ex = pd.DataFrame(exactitud)
    publicado = pd.read_csv(RAIZ / 'results/avance_7/analisis/semillas_contrastes.csv')
    publicado = publicado.query("metrica == 'acc' and variante == 'media3'")
    columnas = ['protocolo', 'par', 'p_holm', 'p_tost_wilcoxon', 'p_tost_wilcoxon_holm', 'equiv_ic90']
    concordancia = ex.merge(publicado[columnas], on=['protocolo', 'par'], validate='many_to_one')
    assert len(concordancia) == 36
    principal = deg[deg.rope == 0.03].set_index('par').p_equivalentes
    loso = ex[(ex.protocolo == 'loso') & (ex.rope == 0.03)].set_index('par').p_equivalentes
    print('Equivalencia conjunta LOSO y degradacion (ROPE 0.03):',
          list(loso.index[(loso >= 0.95) & (principal.reindex(loso.index) >= 0.95)]))
    SALIDA.mkdir(parents=True, exist_ok=True)
    ex.to_csv(SALIDA / 'bayes_rope_exactitud.csv', index=False, encoding='utf-8')
    deg.to_csv(SALIDA / 'bayes_rope_degradacion.csv', index=False, encoding='utf-8')
    concordancia.to_csv(SALIDA / 'bayes_rope_concordancia.csv', index=False, encoding='utf-8')


if __name__ == '__main__':
    main()
