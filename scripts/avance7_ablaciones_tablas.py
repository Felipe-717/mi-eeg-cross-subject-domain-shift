# -*- coding: utf-8 -*-
"""Tablas en Markdown de `avance7_ablaciones.py` (se importan desde ahi; no se ejecuta solo)."""
from __future__ import annotations

import pandas as pd

ARQ = ["eegnet", "conformer", "labram", "eegmamba"]
CONFUNDIDO = {"eegnet": True, "conformer": True, "labram": True, "eegmamba": False}


def _p(x):
    return "n/d" if pd.isna(x) else (f"{x:.1e}" if x < 0.001 else f"{x:.3f}")


def _f(x, k=3):
    return "n/d" if pd.isna(x) else f"{x:+.{k}f}"


def _md(cab, filas):
    out = ["| " + " | ".join(cab) + " |", "|" + "|".join(["---"] * len(cab)) + "|"]
    out += ["| " + " | ".join(map(str, f)) + " |" for f in filas]
    return "\n".join(out)


def _fila(r, con_arq=True, holm2=None):
    eq = "si" if r["equiv_0.03"] else "no"
    f = ([r.arq] if con_arq else []) + [
        r.variante, f"{r.media_ref:.4f}", f"{r.media_var:.4f}", _f(r.dif_media, 4),
        _f(r.dif_mediana, 4), f"[{r.ic95_inf:+.4f}, {r.ic95_sup:+.4f}]",
        f"{r.n_mejoran}/{r.n_empeoran}", _p(r.p_wilcoxon),
        _p(r.p_holm) + (" *" if r.p_holm < 0.05 else ""), f"{r.r:.2f}",
        f"{r.margen_equiv_minimo:.3f} ({eq})"]
    if holm2 is not None:
        f.append(_p(r[holm2]))
    return f


CAB = ["variante", "ref.", "var.", "dif. media", "dif. mediana", "IC 95 %", "mejoran/empeoran",
       "p Wilcoxon", "p Holm", "r", "margen min. equiv. (equiv. 0.03)"]


def tablas_md(f4df, f68df, conc, conc4, ea, br, rs, mc, cc, pa=None) -> str:
    L = ["<!-- generado por scripts/avance7_ablaciones.py; no editar a mano -->", ""]
    L.append("### T1. F4 principal: variante menos base intra-sujeto 2026 (n = 54 sujetos)\n")
    L.append("Fuente: `results/avance_7/analisis/ablaciones_f4_contrastes.csv`. `*`: p Holm < 0.05 "
             "dentro de la familia de la arquitectura. Dif. = variante menos base. 'ref.' y 'var.' "
             "son exactitudes medias entre sujetos.\n")
    for m in ARQ:
        d = f4df[(f4df.tipo == "principal") & (f4df.arq == m)]
        L.append(f"**{m}** (familia de {len(d)}; "
                 f"{'confundido con el recorte' if CONFUNDIDO[m] else 'contraste limpio'})\n")
        L.append(_md(CAB + ["p Holm (22)"], [_fila(r, False, "p_holm_22") for _, r in d.iterrows()]))
        L.append("")
    L.append("### T2. F4 sensibilidad: variante (2026) menos media por sujeto de la base 2027-2028\n")
    L.append("Mismo regimen (con recorte) pero semilla distinta.\n")
    d = f4df[f4df.tipo == "sensibilidad"]
    L.append(_md(["arq"] + CAB, [_fila(r) for _, r in d.iterrows()]))
    L.append("")
    L.append("### T2b. F4: concordancia entre la lectura principal y la de sensibilidad\n")
    L.append(_md(["arq", "variante", "dif. principal", "dif. sensib.", "p Holm princ.",
                  "p Holm sensib.", "coinciden"],
                 [[r.arq, r.variante, _f(r.dif_principal, 4), _f(r.dif_sensibilidad, 4),
                   _p(r.p_holm_principal), _p(r.p_holm_sensibilidad),
                   "si" if r.coinciden else "NO"] for _, r in conc4.iterrows()]))
    L.append("")
    L.append("### T3. F6 (EA) y F8 (reflexion de canales): LOSO semilla 2026 menos base\n")
    L.append("Fuente: `loso_f6_f8_contrastes.csv`. Principal = base LOSO 2026; sensibilidad = media "
             "por sujeto de la base 2027-2028. Familia de 4 por tipo; 'p Holm (8)' es la secundaria "
             "sobre F6 + F8.\n")
    for fam in ("F6", "F8"):
        for tipo in ("principal", "sensibilidad"):
            d = f68df[(f68df.familia == fam) & (f68df.tipo == tipo)]
            L.append(f"**{fam}, {tipo}**\n")
            L.append(_md(["arq"] + CAB + (["p Holm (8)"] if tipo == "principal" else []),
                         [_fila(r, True, "p_holm_8" if tipo == "principal" else None)
                          for _, r in d.iterrows()]))
            L.append("")
    L.append("### T3b. F6 y F8: concordancia principal / sensibilidad\n")
    L.append(_md(["familia", "arq", "dif. princ.", "dif. sensib.", "p Holm princ.", "p Holm sensib.",
                  "mismo signo", "misma signif.", "coinciden"],
                 [[r.familia, r.arq, _f(r.dif_principal, 4), _f(r.dif_sensibilidad, 4),
                   _p(r.p_holm_principal), _p(r.p_holm_sensibilidad),
                   "si" if r.mismo_signo else "NO", "si" if r.misma_significacion else "NO",
                   "si" if r.coinciden else "NO"] for _, r in conc.iterrows()]))
    L.append("")
    L.append("### T4. EA y la caida por persona nueva (vistos menos no vistos), exploratorio\n")
    L.append(_md(["arq", "acc no visto base", "acc no visto EA", "acc visto base", "acc visto EA",
                  "caida base", "caida EA", "IDS base", "IDS EA", "dif. caida [IC 95 %]", "p", "r",
                  "DSI clasico base", "DSI clasico EA"],
                 [[r.arq, f"{r.acc_no_visto_base:.4f}", f"{r.acc_no_visto_ea:.4f}",
                   f"{r.acc_visto_base:.4f}", f"{r.acc_visto_ea:.4f}", f"{r.caida_persona_base:.4f}",
                   f"{r.caida_persona_ea:.4f}", f"{r.IDS_base:.4f}", f"{r.IDS_ea:.4f}",
                   f"{r.dif_caida_media:+.4f} [{r.dif_caida_ic95_inf:+.4f}, {r.dif_caida_ic95_sup:+.4f}]",
                   _p(r.p_wilcoxon_caida), f"{r.r_caida:.2f}",
                   f"{r.DSI_clasico_base:+.3f}", f"{r.DSI_clasico_ea:+.3f}"]
                  for _, r in ea.iterrows()]))
    L.append("")
    L.append("### T5. Ganancia segun el rendimiento de referencia del sujeto, exploratorio\n")
    L.append("Referencia = media por sujeto de la base LOSO 2027-2028. Ganancia = variante 2026 menos "
             "base 2026. Spearman con Holm exploratorio dentro de la familia (4).\n")
    L.append(_md(["familia", "arq", "rho", "p", "p Holm", "gan. tercil bajo", "gan. tercil alto",
                  "p Mann-Whitney", "n ref <= 0.70", "gan. ref <= 0.70", "gan. ref > 0.70"],
                 [[r.familia, r.arq, f"{r.spearman_ganancia_vs_ref:+.2f}", _p(r.p_spearman),
                   _p(r.p_spearman_holm), _f(r.ganancia_tercil_bajo, 4),
                   _f(r.ganancia_tercil_alto, 4), _p(r.p_mannwhitney), r["n_ref_<=0.70"],
                   _f(r["ganancia_ref_<=0.70"], 4), _f(r["ganancia_ref_>0.70"], 4)]
                  for _, r in br.iterrows()]))
    L.append("")
    L.append("### T6. Ruido entre semillas de la base (diferencia pareada por sujeto), exploratorio\n")
    L.append("`mismo regimen = NO` indica que una de las dos semillas se entreno sin recorte: la "
             "diferencia contiene el cambio de regimen ademas de la semilla.\n")
    L.append(_md(["protocolo", "arq", "semillas (b - a)", "mismo regimen", "dif. media", "IC 95 %", "p"],
                 [[r.protocolo, r.arq, f"{r.semilla_b} - {r.semilla_a}",
                   "si" if r.mismo_regimen else "NO", _f(r.dif_media_b_menos_a, 4),
                   f"[{r.ic95_inf:+.4f}, {r.ic95_sup:+.4f}]", _p(r.p_wilcoxon)]
                  for _, r in rs.iterrows()]))
    L.append("")
    L.append("### T7. Medias por condicion (exactitud media entre sujetos)\n")
    piv = mc.pivot_table(index=["protocolo", "arq", "condicion"], columns="semilla", values="media")
    filas = []
    for k in piv.index:
        v = [("" if pd.isna(piv.loc[k, s]) else f"{piv.loc[k, s]:.4f}") if s in piv.columns else ""
             for s in (2026, 2027, 2028)]
        filas.append([*k, *v])
    L.append(_md(["protocolo", "arq", "condicion", "2026", "2027", "2028"], filas))
    L.append("")
    L.append("### T8. Contraste frente a la cota del confusor\n")
    L.append("Cota = mayor de `cota_bruta` y `cota_neta` de `confusor_cota.csv` (otro agente; el "
             "archivo 04 en Markdown aun no existia al cerrar este informe, verificar al consolidar). "
             "Regla fijada de antemano: |dif. media| menor o igual que la cota implica 'no interpretable'.\n")
    L.append(_md(["familia", "arq", "variante", "protocolo", "confundido", "dif. media", "p Holm",
                  "cota bruta", "cota neta", "cota usada", "lectura"],
                 [[r.familia, r.arq, r.variante, r.protocolo, "si" if r.confundido else "no",
                   _f(r.dif_media, 4), _p(r.p_holm),
                   "n/d" if pd.isna(r.cota_bruta) else f"{r.cota_bruta:.4f}",
                   "n/d" if pd.isna(r.cota_neta) else f"{r.cota_neta:.4f}",
                   "n/d" if pd.isna(r.cota_confusor) else f"{r.cota_confusor:.4f}", r.lectura]
                  for _, r in cc.iterrows()]))
    if pa is not None:
        L.append(tabla_parada(pa))
    return "\n".join(L) + "\n"


def tabla_parada(pa) -> str:
    L = ["### T9. Parada temprana en las ablaciones intra-sujeto (exploratorio)\n",
         "Porcentaje de las 1080 particiones cuyo mejor modelo es el de la epoca 30 (suelo: 10 de "
         "calentamiento + 20 de paciencia) y mediana de epocas. Ver 01_INVENTARIO_Y_QA.md, A2.\n",
         _md(["arq", "condicion", "% en el suelo (30 ep.)", "mediana de epocas", "% en el tope (200)"],
             [[r.arq, r.condicion, f"{r.pct_suelo:.1f}", f"{r.mediana_epocas:.0f}", f"{r.pct_tope:.1f}"]
              for _, r in pa.iterrows()]), ""]
    return "\n".join(L)
