#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""avance7_semillas_contrastes.py: semillas y contrastes entre arquitecturas (puntos 3.2, 5.1, 5.2, 5.15).

Se ejecuta desde scripts/avance7_semillas.py (--etapa contrastes); usa sus utilidades.
Salida en results/avance_7/analisis/:
    semillas_por_semilla.csv         exactitud y kappa por protocolo, arquitectura y semilla
    semillas_resumen.csv             media entre semillas y desviaciones (cada una con nombre propio)
    semillas_contrastes.csv          los 6 contrastes por protocolo y metrica, variantes
                                     s2026 (publicada), s2027, s2028, media3 y media2728
    semillas_variabilidad_dif.csv    variabilidad entre semillas de la DIFERENCIA (punto 5.1)
    semillas_bootstrap_jerarquico.csv bootstrap jerarquico (sujetos y, dentro, semillas), exactitud
    semillas_cambios_conclusion.csv  conclusion con la semilla 2026 frente a la de 3 semillas

FAMILIAS DE HOLM (declaradas antes de mirar los resultados, iguales a la publicada): una familia
por protocolo, de los 6 contrastes entre pares de arquitecturas; la exactitud y el kappa forman
cada uno su propia familia, y cada variante (s2026, media3, ...) se corrige por separado dentro de
su misma familia de 6. No se mezclan protocolos ni variantes en una familia.

BOOTSTRAP JERARQUICO (exactitud, variantes con >= 2 semillas): para cada una de B = 10 000
repeticiones (numpy.random.default_rng(2026), generador nuevo por contraste) (1) se remuestrean con
reemplazo los 54 sujetos; (2) dentro de cada sujeto elegido y para cada arquitectura se remuestrean
con reemplazo sus k semillas (por defecto de forma independiente entre las dos arquitecturas,
porque la inicializacion es independiente; la variante "comun" usa los mismos indices de semilla
para las dos); (3) se promedian las semillas por sujeto y arquitectura, se restan y se promedian
los sujetos. Los IC son percentiles. Remuestrear en los dos niveles cuenta dos veces el ruido de
semilla (que ya esta dentro de la dispersion entre sujetos), de modo que es conservador.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import avance7_semillas as X
from avance7_semillas import (ARQ, B, MARGEN, OUT, PARES, SEEDS, mat, p_t, p_w, regimen_str,
                              sin_recorte, tost_t)
import avance7_stats as S

VARIANTES = {"s2026": [2026], "s2027": [2027], "s2028": [2028],
             "media3": [2026, 2027, 2028], "media2728": [2027, 2028]}


def ic_pareado(d, nivel):
    """Percentil sobre sujetos; mismo algoritmo que verificar_cifras._ic90_dif (generador 2026)."""
    d = np.asarray(d, float)
    g = np.random.default_rng(X.SEMILLA_GEN)
    m = d[g.integers(0, len(d), (B, len(d)))].mean(axis=1)
    a = (1 - nivel) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


def hboot(XA, XB, comun=False):
    """Bootstrap jerarquico de la media (sobre sujetos) de la diferencia XA - XB (54 x k)."""
    n, k = XA.shape
    g = np.random.default_rng(X.SEMILLA_GEN)
    si = g.integers(0, n, (B, n))
    ia = g.integers(0, k, (B, n, k))
    ib = ia if comun else g.integers(0, k, (B, n, k))
    a = XA[si[:, :, None], ia].mean(axis=2)
    b = XB[si[:, :, None], ib].mean(axis=2)
    return (a - b).mean(axis=1)


def cota_uso():
    """Cota del confusor por (protocolo, arquitectura): max(cota bruta, cota neta). EEGMamba: 0."""
    c = pd.read_csv(OUT / "confusor_cota.csv")
    out = {}
    for _, r in c.iterrows():
        out[(r.protocolo, r.arq)] = float(np.nanmax([r.cota_bruta, r.get("cota_neta", np.nan)]))
    for proto in ("loso", "within"):
        out[(proto, "eegmamba")] = 0.0
    # EEGNet y Conformer intra-sujeto nunca tuvieron recorte: no hay forma de estimar su delta ahi.
    # Se usa la de LOSO como SUPUESTO (declarado en el informe), no como medida.
    for a in ("eegnet", "conformer"):
        out[("within", a)] = out[("loso", a)]
    return out


def holm_familia(ps):
    return S.holm(np.asarray(ps, float))


def por_semilla():
    filas, res = [], []
    for proto in ("loso", "within"):
        for arq in ARQ:
            for metrica in ("acc", "kappa"):
                M = mat(proto, arq, metrica)
                for s in SEEDS:
                    filas.append(dict(protocolo=proto, arq=arq, metrica=metrica, seed=s,
                                      sin_recorte=sin_recorte(proto, arq, s),
                                      media_sujetos=M[s].mean(), sd_entre_sujetos=M[s].std(ddof=1),
                                      mediana_sujetos=M[s].median()))
                por_s = M.mean(axis=0)
                m3 = M.mean(axis=1)
                res.append(dict(protocolo=proto, arq=arq, metrica=metrica,
                                regimen_2026_2027_2028=regimen_str(proto, arq),
                                media_semillas=por_s.mean(),
                                sd_entre_semillas=por_s.std(ddof=1),
                                rango_entre_semillas=por_s.max() - por_s.min(),
                                sd_entre_sujetos_media3=m3.std(ddof=1),
                                sd_semilla_dentro_sujeto=float(M.std(axis=1, ddof=1).mean())))
    pd.DataFrame(filas).to_csv(OUT / "semillas_por_semilla.csv", index=False)
    pd.DataFrame(res).to_csv(OUT / "semillas_resumen.csv", index=False)


def contrastes():
    cu = cota_uso()
    filas, var = [], []
    for proto in ("loso", "within"):
        for metrica in ("acc", "kappa"):
            MM = {a: mat(proto, a, metrica) for a in ARQ}
            bloque = {}
            for vname, seeds in VARIANTES.items():
                reg = {a: regimen_str(proto, a, seeds) for a in ARQ}
                f_sin = {a: np.mean([sin_recorte(proto, a, s) for s in seeds]) for a in ARQ}
                fam = []
                for (a, b) in PARES:
                    xa = MM[a][seeds].mean(axis=1).to_numpy()
                    xb = MM[b][seeds].mean(axis=1).to_numpy()
                    d = xa - xb
                    p, r = p_w(d)
                    fila = dict(protocolo=proto, metrica=metrica, variante=vname,
                                par=f"{a}-{b}", regimen_A=reg[a], regimen_B=reg[b],
                                mismo_regimen=reg[a] == reg[b],
                                un_solo_regimen=len(set(reg[a] + reg[b])) == 1,
                                dif_media=d.mean(), dif_mediana=float(np.median(d)),
                                n_A_mejor=int((d > 0).sum()), n_B_mejor=int((d < 0).sum()),
                                p_wilcoxon=p, r=r, magnitud=S.magnitud(r),
                                ic95_inf=ic_pareado(d, .95)[0], ic95_sup=ic_pareado(d, .95)[1])
                    if metrica == "acc":
                        lo90, hi90 = ic_pareado(d, .90)
                        bsg = f_sin[a] * cu[(proto, a)] + f_sin[b] * cu[(proto, b)]
                        # sensibilidad al confusor: desplazar toda la diferencia hacia cero por el
                        # sesgo maximo (peor caso para una conclusion de diferencia) y reevaluar
                        sg = np.sign(d.mean()) or 1.0
                        p_cero, _ = p_w(d - sg * min(bsg, abs(d.mean())))     # sin cruzar el cero
                        fila.update(ic90_inf=lo90, ic90_sup=hi90,
                                    margen_minimo_equiv=max(abs(lo90), abs(hi90)),
                                    holgura_equiv=MARGEN - max(abs(lo90), abs(hi90)),
                                    equiv_ic90=bool(lo90 > -MARGEN and hi90 < MARGEN),
                                    p_tost_wilcoxon=S.tost(d), p_tost_t=tost_t(d),
                                    sesgo_max_confusor=bsg,
                                    p_wilcoxon_con_sesgo_hacia_cero=p_cero,
                                    margen_minimo_con_sesgo=max(abs(lo90), abs(hi90)) + bsg)
                    fam.append(fila)
                ph = holm_familia([f["p_wilcoxon"] for f in fam])
                for f, h in zip(fam, ph):
                    f["p_holm"] = float(h)
                    f["sig_sin_corregir"] = bool(f["p_wilcoxon"] < .05)
                    f["sig_holm"] = bool(h < .05)
                if metrica == "acc":
                    pt = S.holm([f["p_tost_wilcoxon"] for f in fam])
                    for f, h in zip(fam, pt):
                        f["p_tost_wilcoxon_holm"] = float(h)
                filas += fam
                bloque[vname] = fam
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "semillas_contrastes.csv", index=False)

    # ---- variabilidad entre semillas de la diferencia (punto 5.1)
    for proto in ("loso", "within"):
        for metrica in ("acc", "kappa"):
            MM = {a: mat(proto, a, metrica) for a in ARQ}
            for (a, b) in PARES:
                D = (MM[a] - MM[b])                                # 54 x 3, semillas emparejadas por indice
                por_s = D.mean(axis=0)
                sub = df[(df.protocolo == proto) & (df.metrica == metrica) & (df.par == f"{a}-{b}")]
                sub = sub.set_index("variante")
                v_ruido = float(D.var(axis=1, ddof=1).mean())      # varianza entre semillas, dentro de sujeto
                v_m3 = float(D.mean(axis=1).var(ddof=1))           # varianza entre sujetos de la media de 3
                v_1 = float(D[2026].var(ddof=1))                   # varianza entre sujetos con una semilla
                ps = [sub.loc[f"s{s}", "p_wilcoxon"] for s in SEEDS]
                rs = [sub.loc[f"s{s}", "r"] for s in SEEDS]
                var.append(dict(
                    protocolo=proto, metrica=metrica, par=f"{a}-{b}",
                    dif_s2026=por_s[2026], dif_s2027=por_s[2027], dif_s2028=por_s[2028],
                    dif_media3=por_s.mean(),
                    sd_dif_entre_semillas=float(por_s.std(ddof=1)),
                    rango_dif_entre_semillas=float(por_s.max() - por_s.min()),
                    sd_dif_semilla_dentro_sujeto=float(D.std(axis=1, ddof=1).mean()),
                    sd_dif_entre_sujetos_s2026=float(np.sqrt(v_1)),
                    sd_dif_entre_sujetos_media3=float(np.sqrt(v_m3)),
                    fraccion_var_media3_por_ruido_de_semilla=float(v_ruido / 3 / v_m3),
                    p_s2026=ps[0], p_s2027=ps[1], p_s2028=ps[2],
                    r_s2026=rs[0], r_s2027=rs[1], r_s2028=rs[2],
                    n_semillas_con_p_menor_05=int(sum(p < .05 for p in ps)),
                    p_media3=sub.loc["media3", "p_wilcoxon"], r_media3=sub.loc["media3", "r"]))
    pd.DataFrame(var).to_csv(OUT / "semillas_variabilidad_dif.csv", index=False)
    return df


def dif_2026_vs_2728():
    """Para cada contraste: dif(s2026) - dif(media 2027-2028) por sujeto = efecto del cambio de regimen
    sobre el contraste mas ruido de semilla (cuando A y B estan en el mismo regimen es solo ruido)."""
    filas = []
    for proto in ("loso", "within"):
        MM = {a: mat(proto, a, "acc") for a in ARQ}
        for (a, b) in PARES:
            dd = ((MM[a][2026] - MM[b][2026]) - ((MM[a][[2027, 2028]].mean(axis=1)
                                                  - MM[b][[2027, 2028]].mean(axis=1)))).to_numpy()
            lo, hi = X.boot_media_ic(dd)
            pw, _ = p_w(dd)
            filas.append(dict(protocolo=proto, par=f"{a}-{b}", media=dd.mean(), sd=dd.std(ddof=1),
                              ic95_inf=lo, ic95_sup=hi, p_wilcoxon=pw, p_t=p_t(dd),
                              regimen_A_2026_vs_2728=f"{regimen_str(proto, a, [2026])}/{regimen_str(proto, a, [2027, 2028])}",
                              regimen_B_2026_vs_2728=f"{regimen_str(proto, b, [2026])}/{regimen_str(proto, b, [2027, 2028])}"))
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "semillas_dif_2026_vs_2728.csv", index=False)
    return df


def jerarquico():
    filas = []
    for proto in ("loso", "within"):
        MM = {a: mat(proto, a, "acc") for a in ARQ}
        for vname in ("media3", "media2728"):
            seeds = VARIANTES[vname]
            for (a, b) in PARES:
                XA, XB = MM[a][seeds].to_numpy(), MM[b][seeds].to_numpy()
                fila = dict(protocolo=proto, variante=vname, par=f"{a}-{b}",
                            dif_media=float((XA - XB).mean()))
                for etiqueta, comun in (("indep", False), ("comun", True)):
                    m = hboot(XA, XB, comun)
                    fila.update({f"jer_{etiqueta}_ic95_inf": float(np.quantile(m, .025)),
                                 f"jer_{etiqueta}_ic95_sup": float(np.quantile(m, .975)),
                                 f"jer_{etiqueta}_ic90_inf": float(np.quantile(m, .05)),
                                 f"jer_{etiqueta}_ic90_sup": float(np.quantile(m, .95))})
                    fila[f"jer_{etiqueta}_equiv_ic90"] = bool(
                        fila[f"jer_{etiqueta}_ic90_inf"] > -MARGEN and fila[f"jer_{etiqueta}_ic90_sup"] < MARGEN)
                    fila[f"jer_{etiqueta}_excluye_0"] = bool(
                        fila[f"jer_{etiqueta}_ic95_inf"] > 0 or fila[f"jer_{etiqueta}_ic95_sup"] < 0)
                lo, hi = ic_pareado((XA - XB).mean(axis=1), .95)
                fila.update(subj_ic95_inf=lo, subj_ic95_sup=hi,
                            subj_excluye_0=bool(lo > 0 or hi < 0))
                lo, hi = ic_pareado((XA - XB).mean(axis=1), .90)
                fila.update(subj_ic90_inf=lo, subj_ic90_sup=hi,
                            subj_equiv_ic90=bool(lo > -MARGEN and hi < MARGEN))
                filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv(OUT / "semillas_bootstrap_jerarquico.csv", index=False)
    return df


def cambios(df, jer, d26):
    """Conclusion con la semilla 2026 (publicada) frente a la media de 3 semillas."""
    filas = []
    cat = lambda r: "grande" if r >= .5 else "mediano" if r >= .3 else "pequeno" if r >= .1 else "trivial"
    for proto in ("loso", "within"):
        regs_cambio = {a: X.regimen_str(proto, a) not in ("SSS", "CCC") for a in ARQ}
        for (a, b) in PARES:
            par = f"{a}-{b}"
            q = df[(df.protocolo == proto) & (df.metrica == "acc") & (df.par == par)].set_index("variante")
            j = jer[(jer.protocolo == proto) & (jer.par == par) & (jer.variante == "media3")].iloc[0]
            pub, m3, m2 = q.loc["s2026"], q.loc["media3"], q.loc["media2728"]
            e = d26[(d26.protocolo == proto) & (d26.par == par)].iloc[0]
            fila = dict(protocolo=proto, par=par,
                        regimen_s2026=f"{pub.regimen_A}|{pub.regimen_B}",
                        regimen_media3=f"{m3.regimen_A}|{m3.regimen_B}",
                        involucra_arq_con_cambio_de_regimen=bool(regs_cambio[a] or regs_cambio[b]),
                        dif_mediana_pub=pub.dif_mediana, dif_mediana_media3=m3.dif_mediana,
                        cambio_dif_2026_vs_2728=e.media, cambio_ic95_inf=e.ic95_inf,
                        cambio_ic95_sup=e.ic95_sup, cambio_p_wilcoxon=e.p_wilcoxon,
                        holgura_equiv_pub=pub.holgura_equiv, holgura_equiv_media3=m3.holgura_equiv,
                        dif_pub=pub.dif_media, dif_media3=m3.dif_media, dif_media2728=m2.dif_media,
                        p_pub=pub.p_wilcoxon, p_media3=m3.p_wilcoxon, p_media2728=m2.p_wilcoxon,
                        holm_pub=pub.p_holm, holm_media3=m3.p_holm, holm_media2728=m2.p_holm,
                        r_pub=pub.r, r_media3=m3.r, r_media2728=m2.r,
                        sig_pub=pub.sig_sin_corregir, sig_media3=m3.sig_sin_corregir,
                        sig_media2728=m2.sig_sin_corregir,
                        sigholm_pub=pub.sig_holm, sigholm_media3=m3.sig_holm,
                        sigholm_media2728=m2.sig_holm,
                        magnitud_pub=cat(pub.r), magnitud_media3=cat(m3.r), magnitud_media2728=cat(m2.r),
                        equiv_pub=pub.equiv_ic90, equiv_media3=m3.equiv_ic90, equiv_media2728=m2.equiv_ic90,
                        equiv_jer_media3=j.jer_indep_equiv_ic90,
                        margen_min_pub=pub.margen_minimo_equiv, margen_min_media3=m3.margen_minimo_equiv,
                        tost_w_pub=pub.p_tost_wilcoxon, tost_w_media3=m3.p_tost_wilcoxon,
                        tost_t_pub=pub.p_tost_t, tost_t_media3=m3.p_tost_t,
                        sesgo_max_pub=pub.sesgo_max_confusor, sesgo_max_media3=m3.sesgo_max_confusor,
                        p_pub_con_sesgo=pub.p_wilcoxon_con_sesgo_hacia_cero,
                        p_media3_con_sesgo=m3.p_wilcoxon_con_sesgo_hacia_cero,
                        margen_min_media3_con_sesgo=m3.margen_minimo_con_sesgo)
            fila["cambia_sig"] = fila["sig_pub"] != fila["sig_media3"]
            fila["cambia_sig_holm"] = fila["sigholm_pub"] != fila["sigholm_media3"]
            fila["cambia_magnitud"] = fila["magnitud_pub"] != fila["magnitud_media3"]
            fila["cambia_equiv"] = fila["equiv_pub"] != fila["equiv_media3"]
            fila["cambia_alguna"] = bool(fila["cambia_sig"] or fila["cambia_sig_holm"]
                                         or fila["cambia_magnitud"] or fila["cambia_equiv"])
            filas.append(fila)
    out = pd.DataFrame(filas)
    out.to_csv(OUT / "semillas_cambios_conclusion.csv", index=False)
    return out


def main():
    por_semilla()
    df = contrastes()
    jer = jerarquico()
    d26 = dif_2026_vs_2728()
    out = cambios(df, jer, d26)
    print(f"[contrastes] {len(df)} filas de contrastes; {int(out.cambia_alguna.sum())} de {len(out)} "
          f"contrastes (exactitud) cambian alguna conclusion con 3 semillas")
