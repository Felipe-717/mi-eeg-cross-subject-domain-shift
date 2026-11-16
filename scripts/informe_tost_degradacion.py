#!/usr/bin/env python
"""TOST de degradación pareada por sujeto, con ajuste de Holm."""
import numpy as np
import pandas as pd
import avance7_semillas as X
import avance7_stats as S
from avance7_semillas_contrastes import ic_pareado


def main():
    assert S.tost(np.linspace(-0.005, 0.005, 54)) < 0.05
    vectores = {a: X.mat('loso', a, 'acc_visto').mean(axis=1)
                   - X.mat('loso', a, 'acc').mean(axis=1) for a in X.ARQ}
    filas = []
    for a, b in X.PARES:
        assert vectores[a].index.equals(vectores[b].index) and len(vectores[a]) == 54
        d = vectores[a] - vectores[b]
        lo, hi = ic_pareado(d, .90)
        filas.append(dict(par=f'{a}-{b}', n=len(d), diferencia_media=d.mean(),
                          ic90_inf=lo, ic90_sup=hi, p_tost=S.tost(d)))
    df = pd.DataFrame(filas)
    df['p_tost_holm'] = S.holm(df.p_tost)
    df.to_csv(X.RAIZ / 'results/informe_final/tost_degradacion.csv', index=False)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
