#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""avance7_informe03_tablas.py: arma avance_7/analisis/03_FEWSHOT_TRANSFERENCIA.md a partir de
una plantilla con marcadores {{NOMBRE}} y de los CSV de results/avance_7/analisis/.

Solo da formato: ninguna cifra se calcula aqui (salen de avance7_fewshot.py y
avance7_transferencia.py). Uso:
    ~/anaconda3/envs/eeg/python.exe scripts/avance7_informe03_tablas.py PLANTILLA.md
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CSV = RAIZ / "results" / "avance_7" / "analisis"
N = {"eegnet": "EEGNet", "conformer": "Conformer", "labram": "LaBraM", "eegmamba": "EEGMamba"}
SINO = {True: "si", False: "no"}


def md(filas, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    out += ["| " + " | ".join(str(x) for x in f) + " |" for f in filas]
    return "\n".join(out)


def f(x, n=3):
    return "NA" if pd.isna(x) else f"{x:.{n}f}"


def fs_tabla(v):
    c = pd.read_csv(CSV / f"fewshot_contrastes_{v}.csv")
    filas = [[N[r.arq], r.k, r.metodo, f(r.cero), f(r.gan_media, 4), f"[{r.ic_inf:.4f}, {r.ic_sup:.4f}]",
              f"{r.p:.4f}", f"{r.p_holm:.4f}", f(r.r, 2), f"{r.sujetos_mejoran}/{r.sujetos_empeoran}",
              SINO[bool(r.equiv_003)]] for r in c.itertuples()]
    return md(filas, ["Arq", "k", "Método", "cero", "Ganancia media", "IC 95 %", "p", "p Holm", "r",
                      "mejoran/empeoran", "Equiv. ±0.03"])


def fs_cons():
    c = pd.read_csv(CSV / "fewshot_consistencia_semillas.csv")
    filas = [[N[r.arq], r.k, r.metodo, f(r.gan_2027, 4), f(r.gan_2028, 4), f(r.dif_entre_semillas, 4),
              f"{r.p_2027:.4f}", f"{r.p_2028:.4f}", f(r.rho_sujeto_entre_semillas, 2),
              f(r.sd_sorteos_media, 3)] for r in c.itertuples()]
    return md(filas, ["Arq", "k", "Método", "Gan. 2027", "Gan. 2028", "Dif. semillas", "p 2027", "p 2028",
                      "rho entre semillas (sujeto)", "DT entre sorteos"])


def fs_nivel():
    c = pd.read_csv(CSV / "fewshot_nivel_inicial_resumen_exploratorio.csv")
    filas = [[r.conjunto, r.referencia, r.metodo, r.filas, r.p_mw_menor_005, r.p_rho_menor_005,
              r.rho_positivo, f(r.dif_bajo_menos_alto_media, 4)] for r in c.itertuples()]
    return md(filas, ["Conjunto", "Referencia de nivel", "Método", "Filas (4 arq x 2 k)", "p MW < 0.05",
                      "p rho < 0.05", "rho > 0", "Dif. media bajo menos alto"])


def fs_nivel_k20():
    c = pd.read_csv(CSV / "fewshot_nivel_inicial_base_exploratorio.csv")
    c = c[(c.k == 20) & (c.metodo == "cabeza")]
    filas = [[N[r.arq], r.referencia, f"{r.n_bajo}/{r.n_alto}", f(r.gan_bajo, 4), f(r.gan_alto, 4),
              f(r.p_mannwhitney, 4), f(r.rho_spearman, 2), f(r.p_rho, 4)] for r in c.itertuples()]
    return md(filas, ["Arq", "Gan. de la semilla / nivel de la otra", "n bajo/alto", "Gan. bajo", "Gan. alto",
                      "p MW", "rho", "p rho"])


def tr_tabla(qs):
    r = pd.read_csv(CSV / "transf_resumen_3semillas.csv")
    filas = []
    for q in qs:
        for a in N:
            x = r[(r.arq == a) & (r.cantidad == q)].iloc[0]
            filas.append([q, N[a], f(x.v2026), f(x.v2027), f(x.v2028), f(x.media_3sem),
                          f(x.sd_entre_semillas), f"[{x['min']:.3f}, {x['max']:.3f}]"])
    return md(filas, ["Cantidad", "Arq", "2026", "2027", "2028", "Media", "DT entre semillas", "Rango"])


def tr_repro():
    r = pd.read_csv(CSV / "transf_reproduccion_2026.csv")
    ok = int(r.coincide_3dec.sum())
    return (f"{ok} de {len(r)} cifras coinciden a tres decimales; diferencia absoluta máxima "
            f"{r.diferencia.abs().max():.4f}.")


def tr_repro_ic():
    r = pd.read_csv(CSV / "transf_reproduccion_ic_2026.csv")
    filas = [[N[x.arq], x.cantidad, f"{x.pub_est:.3f} [{x.pub_inf:.3f}, {x.pub_sup:.3f}]",
              f"{x.rec_est:.3f} [{x.rec_inf:.3f}, {x.rec_sup:.3f}]", f(x.max_dif_limite)] for x in r.itertuples()]
    return md(filas, ["Arq", "Cantidad", "Publicado", "Recalculado", "Máx. dif. de límite"])


def tr_ic():
    c = pd.read_csv(CSV / "transf_ic_caidas.csv")
    nom = {"2026_reproduccion": "2026", "3sem_sujetos": "3 sem., sujetos", "3sem_jerarquico": "3 sem., jerárquico",
           "solo_2027": "solo 2027", "solo_2028": "solo 2028"}
    filas = []
    for q in ("caida_persona", "caida_protocolo", "protocolo_menos_persona", "idp"):
        for a in N:
            x = c[(c.arq == a) & (c.cantidad == q)].set_index("metodo")
            celdas = []
            for m in nom:
                r = x.loc[m]
                celdas.append(f"{r.estimacion:+.3f} [{r.ic_inf:+.3f}, {r.ic_sup:+.3f}]" + (" *" if r.excluye_cero else ""))
            filas.append([q, N[a]] + celdas)
    return md(filas, ["Cantidad", "Arq"] + list(nom.values()))


def tr_pol():
    c = pd.read_csv(CSV / "transf_politicas_contrastes.csv")
    filas = [[N[r.arq], r.contraste, f(r.media_source), f(r.media_otra), f(r.dif_media, 4),
              f"[{r.ic_inf:.4f}, {r.ic_sup:.4f}]", f"{r.p:.4f}", f"{r.p_holm:.4f}", f(r.r, 2)] for r in c.itertuples()]
    return md(filas, ["Arq", "Contraste", "Media origen", "Media otra", "Dif.", "IC 95 %", "p", "p Holm", "r"])


def tr_pol_medias():
    r = pd.read_csv(CSV / "transf_resumen_3semillas.csv")
    filas = []
    for a in N:
        x = r[r.arq == a].set_index("cantidad")
        v = [x.loc[q, "media_3sem"] for q in ("transferido", "transf_per_trial", "transf_per_subject")]
        filas.append([N[a], f(v[0]), f(v[1]), f(v[2]), f(max(v) - min(v))])
    return md(filas, ["Arq", "Origen", "Por segmento", "Por sujeto", "Amplitud"])


def ea_comp():
    c = pd.read_csv(CSV / "transf_ea_comparacion.csv")
    qs = ["transferido", "nativo", "dif_nativo_menos_transferido", "vistos44", "persona", "wbcic44",
          "caida_persona", "caida_protocolo", "protocolo_menos_persona", "idp"]
    filas = []
    for q in qs:
        for a in N:
            r = c[(c.arq == a) & (c.cantidad == q)].iloc[0]
            filas.append([q, N[a], f(r.base_2026), f(r.ea_2026), f"{r.ea_menos_base:+.3f}", f(r.base_sd_3sem),
                          f"{r.cociente_cambio_sobre_sd:+.1f}", f"[{r.base_min:.3f}, {r.base_max:.3f}]",
                          SINO[bool(r.ea_fuera_rango_base)]])
    return md(filas, ["Cantidad", "Arq", "Base 2026", "EA 2026", "EA - base", "DT base (3 sem.)",
                      "(EA - base)/DT", "Rango base", "EA fuera del rango"])


def ea_cont():
    c = pd.read_csv(CSV / "transf_ea_contrastes.csv")
    filas = [[N[r.arq], r.evaluacion, r.n, f(r.base), f(r.ea), f(r.dif_media, 4),
              f"[{r.ic_inf:.4f}, {r.ic_sup:.4f}]", f"{r.p:.4f}", f"{r.p_holm:.4f}", f(r.r, 2),
              f"{r.mejoran}/{r.empeoran}"] for r in c.itertuples()]
    return md(filas, ["Arq", "Evaluación", "n", "Base", "EA", "EA - base", "IC 95 %", "p", "p Holm", "r",
                      "mejoran/empeoran"])


def ea_ic():
    c = pd.read_csv(CSV / "transf_ea_caidas_ic.csv")
    c = c[c.cantidad.str.startswith("ea_menos_base")]
    filas = [[N[r.arq], r.cantidad.replace("ea_menos_base_", ""), f"{r.estimacion:+.3f}",
              f"[{r.ic_inf:+.3f}, {r.ic_sup:+.3f}]", SINO[bool(r.excluye_cero)]] for r in c.itertuples()]
    return md(filas, ["Arq", "Cantidad (EA - base, 2026)", "Estimación", "IC 95 %", "Excluye 0"])


def ea_frac():
    c = pd.read_csv(CSV / "transf_ea_caidas_ic.csv")
    c = c[c.cantidad.str.startswith("fraccion")]
    filas = [[N[r.arq], r.cantidad.replace("fraccion_recuperada_", ""), f"{r.estimacion:+.3f}"] for r in c.itertuples()]
    return md(filas, ["Arq", "Fracción recuperada", "Valor"])


def curvas():
    c = pd.read_csv(CSV / "transf_curvas_epocas.csv")
    c = c[c.arq == "eegnet"]
    filas = [[r.tabla, r.variante, r.semilla, r.n_epocas, r.epoca_minima, f(r.val_loss_min, 4),
              f(r.mejora_ultimas_50, 4), SINO[bool(r.en_tope_200)]] for r in c.itertuples()]
    return md(filas, ["Modelo", "Variante", "Semilla", "Épocas", "Época del mínimo", "Pérdida de validación mínima",
                      "Mejora lograda en las últimas 50 épocas", "En el tope de 200"])


TABLAS = {
    "FS_BASE": lambda: fs_tabla("base"), "FS_EA": lambda: fs_tabla("ea"), "FS_CONS": fs_cons,
    "FS_NIVEL": fs_nivel, "FS_NIVEL_K20": fs_nivel_k20,
    "TR_A": lambda: tr_tabla(["transferido", "nativo", "dif_nativo_menos_transferido"]),
    "TR_B": lambda: tr_tabla(["vistos44", "persona", "wbcic44"]),
    "TR_C": lambda: tr_tabla(["caida_persona", "caida_protocolo", "protocolo_menos_persona", "idp"]),
    "TR_REPRO": tr_repro, "TR_REPRO_IC": tr_repro_ic, "TR_IC": tr_ic, "TR_POL": tr_pol,
    "TR_POL_MEDIAS": tr_pol_medias, "EA_COMP": ea_comp, "EA_CONT": ea_cont, "EA_IC": ea_ic,
    "EA_FRAC": ea_frac, "CURVAS": curvas,
}


def main():
    plantilla = Path(sys.argv[1]).read_text(encoding="utf-8")
    salida = RAIZ / "avance_7" / "analisis" / "03_FEWSHOT_TRANSFERENCIA.md"
    texto = re.sub(r"\{\{(\w+)\}\}", lambda m: TABLAS[m.group(1)](), plantilla)
    salida.write_text(texto, encoding="utf-8")
    print("escrito", salida, "em-dashes:", texto.count("—"))


if __name__ == "__main__":
    main()
