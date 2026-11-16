#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""avance7_semillas_tablas.py: formatea en Markdown los CSV de avance7_semillas*.py para el informe
avance_7/analisis/04_SEMILLAS_Y_CONFUSOR.md. No calcula nada: solo lee results/avance_7/analisis/*.csv.
Salida: results/avance_7/analisis/semillas_tablas.md
Comando: ~/anaconda3/envs/eeg/python.exe scripts/avance7_semillas_tablas.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).resolve().parents[1] / "results" / "avance_7" / "analisis"
N = {"eegnet": "EEGNet", "conformer": "Conformer", "labram": "LaBraM", "eegmamba": "EEGMamba"}
out = []


def f(x, d=3, signo=False):
    if pd.isna(x):
        return ""
    return f"{x:+.{d}f}" if signo else f"{x:.{d}f}"


def pf(p):
    return "<0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def md(titulo, cols, filas):
    out.append(f"\n**{titulo}**\n")
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "|".join("---" for _ in cols) + "|")
    for r in filas:
        out.append("| " + " | ".join(str(x) for x in r) + " |")


# T1 confusor
t = pd.read_csv(D / "confusor_tabla.csv")
filas = []
for _, r in t.iterrows():
    filas.append([r.protocolo, N[r.arq], r.regimen_2026_2027_2028,
                  f"{f(r.cruce_media, 4, True)} [{f(r.cruce_ic95_boot_inf, 4, True)}, {f(r.cruce_ic95_boot_sup, 4, True)}]",
                  f(r.cruce_mediana, 4, True), f(r.cruce_sd, 4), pf(r.cruce_p_wilcoxon), pf(r.cruce_p_t),
                  f"{f(r.intra_media, 4, True)}", f(r.intra_sd, 4), pf(r.intra_p_wilcoxon)])
md("T1. d_cruce = acc(2026) - media(acc 2027, 2028) y d_intra = acc(2027) - acc(2028), por sujeto (n = 54). IC 95 % bootstrap sobre sujetos (B = 10 000, generador 2026).",
   ["Prot.", "Arq.", "Reg. 26/27/28", "d_cruce media [IC95]", "mediana", "sd", "p Wilcoxon", "p t", "d_intra media", "sd", "p Wilcoxon"], filas)

filas = []
for _, r in t.iterrows():
    filas.append([r.protocolo, N[r.arq], f(r.cruce_sd, 4), f(r.sd_cruce_esperada_por_ruido, 4),
                  f"{r.razon_var_cruce_sobre_esperada:.2f}", f"[{r.razon_ic95_boot_inf:.2f}, {r.razon_ic95_boot_sup:.2f}]",
                  pf(r.p_F_bilateral), f(r.sd_heterogeneidad_efecto, 4), f(r.sd_heterogeneidad_ic95_sup, 4),
                  f"{f(r.par2026_2027_sd, 3)} / {f(r.par2026_2028_sd, 3)} / {f(r.par2027_2028_sd, 3)}"])
md("T2. Varianza del cruce frente a la esperada por ruido de semilla (H0: Var(d_cruce) = 0.75 Var(d_intra)). Heterogeneidad = sqrt(max(Var(d_cruce) - 0.75 Var(d_intra), 0)).",
   ["Prot.", "Arq.", "sd d_cruce", "sd esperada", "razon de varianzas", "IC95 boot", "p F", "sd heterog.", "IC95 sup", "sd de (26-27) / (26-28) / (27-28)"], filas)

c = pd.read_csv(D / "confusor_vs_control.csv")
filas = [[r.protocolo, N[r.arq], "si" if r.hay_cambio_de_regimen else "no (placebo)",
          f"{f(r.media_dd, 4, True)} [{f(r.ic95_boot_inf, 4, True)}, {f(r.ic95_boot_sup, 4, True)}]",
          pf(r.p_wilcoxon), pf(r.p_t)] for _, r in c.iterrows()]
md("T3. Diferencia de diferencias: d_cruce(arq) - d_cruce(EEGMamba), por sujeto.",
   ["Prot.", "Arq.", "Cambio de regimen", "media [IC95 boot]", "p Wilcoxon", "p t"], filas)

k = pd.read_csv(D / "confusor_cota.csv")
filas = [[r.protocolo, N[r.arq], f"{f(r.delta_bruto, 4, True)} [{f(r.bruto_ic95_inf, 4, True)}, {f(r.bruto_ic95_sup, 4, True)}]",
          f(r.cota_bruta, 4), f"{f(r.delta_neto, 4, True)} [{f(r.neto_ic95_inf, 4, True)}, {f(r.neto_ic95_sup, 4, True)}]",
          f(r.cota_neta, 4), f(max(r.cota_bruta, r.cota_neta), 4), f(max(r.cota_bruta, r.cota_neta) / 3, 4)] for _, r in k.iterrows()]
md("T4. Cota del efecto del recorte sobre la exactitud media por sujeto (delta = sin recorte menos con recorte).",
   ["Prot.", "Arq.", "delta bruto [IC95]", "cota bruta", "delta neto (vs EEGMamba) [IC95]", "cota neta", "cota de uso", "cota en media de 3 semillas"], filas)

# T5 por semilla
s = pd.read_csv(D / "semillas_por_semilla.csv")
r = pd.read_csv(D / "semillas_resumen.csv")
filas = []
for proto in ("loso", "within"):
    for a in N:
        x = s[(s.protocolo == proto) & (s.arq == a)]
        xa = x[x.metrica == "acc"].set_index("seed").media_sujetos
        xk = x[x.metrica == "kappa"].set_index("seed").media_sujetos
        ra = r[(r.protocolo == proto) & (r.arq == a) & (r.metrica == "acc")].iloc[0]
        rk = r[(r.protocolo == proto) & (r.arq == a) & (r.metrica == "kappa")].iloc[0]
        filas.append([proto, N[a], ra.regimen_2026_2027_2028,
                      " / ".join(f(xa[q]) for q in (2026, 2027, 2028)), f(ra.media_semillas), f(ra.sd_entre_semillas, 4),
                      f(ra.sd_entre_sujetos_media3), f(ra.sd_semilla_dentro_sujeto),
                      " / ".join(f(xk[q]) for q in (2026, 2027, 2028)), f(rk.media_semillas), f(rk.sd_entre_semillas, 4)])
md("T5. Exactitud y kappa por semilla (media sobre 54 sujetos). sd entre semillas = sd (ddof = 1) de los 3 valores por semilla; sd entre sujetos = sd de los 54 valores de la media de las 3 semillas; sd semilla dentro de sujeto = media sobre sujetos de la sd (ddof = 1) de sus 3 semillas.",
   ["Prot.", "Arq.", "Reg.", "acc 2026 / 2027 / 2028", "acc media", "acc sd entre semillas", "acc sd entre sujetos (media 3)", "acc sd semilla dentro de sujeto",
    "kappa 2026 / 2027 / 2028", "kappa media", "kappa sd entre semillas"], filas)

# T6 contrastes
c = pd.read_csv(D / "semillas_contrastes.csv")
v = pd.read_csv(D / "semillas_variabilidad_dif.csv")
j = pd.read_csv(D / "semillas_bootstrap_jerarquico.csv")
for proto in ("loso", "within"):
    filas = []
    for par in c[(c.protocolo == proto)].par.unique():
        q = c[(c.protocolo == proto) & (c.metrica == "acc") & (c.par == par)].set_index("variante")
        a, b = par.split("-")
        vv = v[(v.protocolo == proto) & (v.metrica == "acc") & (v.par == par)].iloc[0]
        jj = j[(j.protocolo == proto) & (j.par == par) & (j.variante == "media3")].iloc[0]
        p0, p3 = q.loc["s2026"], q.loc["media3"]
        filas.append([f"{N[a]} vs {N[b]}", p3.regimen_A + "/" + p3.regimen_B,
                      f"{f(p0.dif_mediana, 3, True)} / {f(p0.dif_media, 3, True)}", pf(p0.p_wilcoxon), f"{p0.r:.2f}", pf(p0.p_holm),
                      f"{f(p3.dif_media, 3, True)} / {f(p3.dif_mediana, 3, True)}", pf(p3.p_wilcoxon), f"{p3.r:.2f}", pf(p3.p_holm),
                      f"{f(vv.dif_s2026, 3, True)} {f(vv.dif_s2027, 3, True)} {f(vv.dif_s2028, 3, True)}", f(vv.sd_dif_entre_semillas, 4),
                      f"{pf(vv.p_s2026)} {pf(vv.p_s2027)} {pf(vv.p_s2028)}",
                      f"[{f(jj.jer_indep_ic95_inf, 3, True)}, {f(jj.jer_indep_ic95_sup, 3, True)}]"])
    md(f"T6-{proto}. Contrastes de exactitud ({proto}). 2026 = semilla publicada (dif mediana / media); media3 = media de las 3 semillas por sujeto (dif media / mediana). Holm: familia de 6 de ese protocolo y variante.",
       ["Contraste", "Reg. 3 semillas A/B", "2026 dif med/media", "p", "r", "p Holm", "media3 dif media/med", "p", "r", "p Holm",
        "dif media por semilla 26 27 28", "sd entre semillas de la dif", "p por semilla", "IC95 jerarquico media3"], filas)

# T7 equivalencia
filas = []
for proto in ("loso", "within"):
    for par in c[(c.protocolo == proto)].par.unique():
        q = c[(c.protocolo == proto) & (c.metrica == "acc") & (c.par == par)].set_index("variante")
        a, b = par.split("-")
        jj = j[(j.protocolo == proto) & (j.par == par) & (j.variante == "media3")].iloc[0]
        filas.append([proto, f"{N[a]} vs {N[b]}"] + [
            f"[{f(q.loc[x].ic90_inf, 3, True)}, {f(q.loc[x].ic90_sup, 3, True)}] {'SI' if q.loc[x].equiv_ic90 else 'no'}"
            for x in ("s2026", "media3", "media2728")] +
            [f"[{f(jj.jer_indep_ic90_inf, 3, True)}, {f(jj.jer_indep_ic90_sup, 3, True)}] {'SI' if jj.jer_indep_equiv_ic90 else 'no'}",
             pf(q.loc["s2026"].p_tost_wilcoxon_holm), pf(q.loc["media3"].p_tost_wilcoxon_holm), pf(q.loc["media3"].p_tost_t)])
md("T7. Equivalencia practica con margen +-0.03 (IC 90 % bootstrap de la diferencia media dentro del margen; TOST con Wilcoxon, corregido por Holm en la familia de 6; TOST con t sin corregir).",
   ["Prot.", "Contraste", "IC90 2026", "IC90 media3", "IC90 2027-28", "IC90 jerarquico media3", "TOST-W Holm 2026", "TOST-W Holm media3", "TOST-t media3"], filas)

# T8 cambios
cc = pd.read_csv(D / "semillas_cambios_conclusion.csv")
filas = [[r.protocolo, r.par, r.regimen_s2026.replace("|", "/") + " -> " + r.regimen_media3.replace("|", "/"), "si" if r.involucra_arq_con_cambio_de_regimen else "no",
          "si" if r.cambia_sig else "no", "si" if r.cambia_sig_holm else "no", f"{r.magnitud_pub} -> {r.magnitud_media3}",
          "si" if r.cambia_equiv else "no",
          f"{f(r.cambio_dif_2026_vs_2728, 4, True)} [{f(r.cambio_ic95_inf, 4, True)}, {f(r.cambio_ic95_sup, 4, True)}]",
          f(r.holgura_equiv_pub, 3, True), f(r.holgura_equiv_media3, 3, True)] for _, r in cc.iterrows()]
md("T8. Cambios de conclusion de la semilla 2026 a la media de 3 semillas (exactitud). Holgura = 0.03 - mayor |extremo| del IC90.",
   ["Prot.", "Contraste", "Reg. 2026 -> 3 semillas", "Arq. con cambio de reg.", "cambia p<0.05", "cambia Holm", "magnitud r", "cambia equiv.",
    "dif(2026) - dif(2027-28) [IC95]", "holgura 2026", "holgura media3"], filas)

# T9 caida
cd = pd.read_csv(D / "tres_semillas_caida_within_loso.csv")
filas = [[N[r.arq], r.semilla, r.regimen_intra_loso, f(r.intra), f(r.loso), f(r.caida_loso_menos_intra, 3, True), f(r.mediana, 3, True),
          f"[{f(r.ic95_inf, 3)}, {f(r.ic95_sup, 3)}]", f"{r.n_mejoran_con_loso}/54", pf(r.p_wilcoxon), f"{r.r:.2f}",
          (f"[{f(r.jer_ic95_inf, 3)}, {f(r.jer_ic95_sup, 3)}]" if not pd.isna(r.jer_ic95_inf) else "")] for _, r in cd.iterrows()]
md("T9. Diferencia LOSO - intra-sujeto por sujeto. 'publicada_mixta' reproduce tab:within-loso (intra con las semillas que tenia cada arquitectura, LOSO 2026).",
   ["Arq.", "Semilla", "Reg. intra / LOSO", "intra", "LOSO", "dif media", "mediana", "IC95 sujetos", "mejoran", "p", "r", "IC95 jerarquico"], filas)

# T10 ids
ii = pd.read_csv(D / "tres_semillas_ids.csv")
filas = [[N[r.arq], r.semilla, r.regimen_loso, f(r.acc_visto), f(r.acc_no_visto), f(r.ids_acc, 3, True), f(r.ids_kappa, 3, True),
          f"{r.folds_mayor_0}/54", pf(r.p_wilcoxon), pf(r.p_holm), f"{r.r:.2f}", "SI" if r.declara_degradacion else "no",
          (f"[{f(r.ids_acc_jer_ic95_inf, 3, True)}, {f(r.ids_acc_jer_ic95_sup, 3, True)}]" if not pd.isna(r.ids_acc_jer_ic95_inf) else ""),
          (f(r.ids_acc_sd_entre_semillas, 4) if not pd.isna(r.ids_acc_sd_entre_semillas) else "")] for _, r in ii.iterrows()]
md("T10. IDS por semilla y con 3 semillas (formula sobre medias; Holm sobre los 4 contrastes de cada fila-semilla).",
   ["Arq.", "Semilla", "Reg.", "vistos", "no vistos", "IDS acc", "IDS kappa", "folds>0", "p", "p Holm", "r", "degradacion", "IC95 jerarquico IDS acc", "sd IDS entre semillas"], filas)

# T11 idp
pp = pd.read_csv(D / "tres_semillas_idp.csv")
filas = [[N[r.arq], r.semilla, f(r.hold_persona), f(r.hold_wbcic), f(r.idp, 3), (f(r.idp_sd_entre_semillas, 4) if not pd.isna(r.idp_sd_entre_semillas) else ""),
          (f"[{f(r.jer_ic95_inf, 3)}, {f(r.jer_ic95_sup, 3)}]" if not pd.isna(r.jer_ic95_inf) else "")] for _, r in pp.iterrows()]
md("T11. IDP por semilla (44 sujetos de entrenamiento, 10 apartados y WBCIC, normalizacion origen; todas con recorte).",
   ["Arq.", "Semilla", "persona nueva", "WBCIC", "IDP", "sd entre semillas", "IC95 jerarquico"], filas)

# T12 bajo umbral
bu = pd.read_csv(D / "tres_semillas_bajo_umbral.csv")
filas = []
for sem in bu.semilla.unique():
    x = bu[bu.semilla == sem].set_index("arq")
    filas.append([sem] + [f"{x.loc[a].bajo_intra} / {x.loc[a].bajo_loso}" for a in N])
md("T12. Sujetos con exactitud <= 0.70, intra-sujeto / LOSO.", ["Semilla"] + [N[a] for a in N], filas)

so = pd.read_csv(D / "tres_semillas_solapamiento.csv")
filas = [[r.semilla] + [f"{r[f'obs_k{i}']} ({r[f'esp_k{i}']:.1f})" for i in range(5)] for _, r in so.iterrows()]
md("T13. Solapamiento intra-sujeto: sujetos bajo el umbral en exactamente k arquitecturas, observado (esperado bajo independencia).",
   ["Semilla", "k=0", "k=1", "k=2", "k=3", "k=4"], filas)

es = pd.read_csv(D / "tres_semillas_estabilidad_bajo.csv")
filas = [[r.protocolo, N.get(r.arq, r.arq), f"{r.n_2026}/{r.n_2027}/{r.n_2028}", r.n_en_las_3, r.n_en_alguna, f(r.jaccard_medio_pares, 2),
          f"{r.jaccard_min:.2f}-{r.jaccard_max:.2f}", r.n_media3, f(r.jaccard_media3_vs_2026, 2), r.sujetos_inestables] for _, r in es.iterrows()]
md("T14. Estabilidad entre semillas del conjunto de bajo rendimiento (umbral <= 0.70 por semilla).",
   ["Prot.", "Arq.", "n por semilla", "en las 3", "en alguna", "Jaccard medio entre pares", "rango", "n (media 3)", "Jaccard media3 vs 2026", "sujetos que no estan en las 3"], filas)

co = pd.read_csv(D / "tres_semillas_correlacion.csv")
filas = []
for sem in co.semilla.unique():
    x = co[co.semilla == sem].set_index("arq")
    filas.append([sem] + [f"{x.loc[a].r_pearson_intra_loso:.3f} [{x.loc[a].ic95_inf:.2f}, {x.loc[a].ic95_sup:.2f}]" for a in N])
md("T15. Correlacion de Pearson por sujeto entre exactitud intra-sujeto y LOSO.", ["Semilla"] + [N[a] for a in N], filas)

(D / "semillas_tablas.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("escrito", D / "semillas_tablas.md")
