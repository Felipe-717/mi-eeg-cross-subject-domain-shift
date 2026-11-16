"""Genera las figuras de resultados de la tesis a partir de los CSV.

Salida: tesis/figuras/resultados_baseline.pdf (vectorial, para \\includegraphics)

Por que un script y no una imagen suelta: si manana cambian los CSV, la figura se
regenera identica con un comando. Una imagen exportada a mano de un notebook no es
reproducible y acaba desincronizada de las cifras del texto.

SEMILLAS (avance 7)
    --semillas 2026   (por defecto, modo anterior) LOSO de EEGNet con la semilla 2026; el eje intra-sujeto es
                      la media de las tres semillas del archivo previo (la figura publicada hasta el avance 6).
    --semillas 3      LOSO e intra-sujeto con la media de las tres semillas (2026, 2027, 2028) por sujeto,
                      leidos con scripts/avance7_datos.py. Es la figura del avance 7.

Uso:
    python scripts/figuras_resultados.py
    python scripts/figuras_resultados.py --semillas 3 --salida Overleaf/resultados_baseline.pdf --sin-png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[1]
LOSO = REPO / "results/pod_baseline/baseline_eegnet_loso_completo_global_epochs_cache_ica.csv"
WITHIN = REPO / "results/pod_baseline/baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica.csv"
CONTROL = REPO / "results/reports/openbmi/control_ocular_epochs_cache_ica.csv"
SALIDA = REPO / "tesis/figuras/resultados_baseline.pdf"

# --- paleta validada (modo claro, superficie papel blanco) ---
# Comprobada con el validador: CVD dE 24.7, vision normal 33.6, contraste >= 3:1.
AZUL     = "#2a78d6"   # slot categorico 1
NARANJA  = "#eb6834"   # slot categorico 2
TINTA    = "#0b0b0b"
MUTED    = "#898781"   # ejes y etiquetas: recesivos
REJILLA  = "#e1e0d9"
EJE      = "#c3c2b7"


def holm(p):
    """Correccion de Holm. Devuelve los p ajustados."""
    idx = np.argsort(p)
    m, adj, run = len(p), np.empty(len(p)), 0.0
    for k, i in enumerate(idx):
        run = max(run, (m - k) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def residuales(n=400, alpha=0.05):
    """Sujetos que siguen mostrando informacion de clase en canales oculares
    despues de la limpieza (binomial exacta + Holm)."""
    c = pd.read_csv(CONTROL)
    p = np.array([binomtest(int(round(a * n)), n, 0.5, alternative="greater").pvalue
                  for a in c["acc_proxy_EOG"]])
    return set(c.loc[holm(p) < alpha, "sub"])


def estilo(ax):
    """Rejilla y ejes recesivos: el dato manda, el andamiaje no compite."""
    ax.set_axisbelow(True)
    ax.grid(True, color=REJILLA, linewidth=0.6)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(EJE)
        ax.spines[lado].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color(TINTA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", choices=["2026", "3"], default="2026",
                    help="2026 = figura anterior; 3 = media de tres semillas en LOSO e intra-sujeto")
    ap.add_argument("--salida", default=str(SALIDA), help="ruta del PDF de salida")
    ap.add_argument("--sin-png", action="store_true", help="no escribir el PNG")
    args = ap.parse_args()
    salida = Path(args.salida)
    if args.semillas == "3":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import avance7_datos as A
        dl, dw = A.loso("eegnet"), A.within("eegnet")
        assert sorted(dl.seed.unique()) == [2026, 2027, 2028] and sorted(dw.seed.unique()) == [2026, 2027, 2028]
        # Redondeo a 1e-9 (como verificar_cifras.le_umbral): las medias de 3 semillas pueden valer
        # exactamente un borde de intervalo (0.55) o un empate con el eje intra-sujeto, y el ruido de
        # coma flotante (1e-16) los cambiaria de recuadro o de lado de la diagonal.
        lo = (dl.groupby("test_sub")["acc"].mean().round(9).rename("acc").rename_axis("test_sub").reset_index())
        wi = dw.groupby("sub")["acc"].mean().round(9)
    else:
        lo = pd.read_csv(LOSO)
        wi = pd.read_csv(WITHIN).groupby("sub")["acc"].mean()
    resid = residuales()

    lo = lo.assign(within=lo["test_sub"].map(wi),
                   resid=lo["test_sub"].isin(resid))
    print(f"{len(lo)} sujetos | residuales: {sorted(resid)}")

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                         "axes.labelsize": 9, "pdf.fonttype": 42})
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.7))

    # ---------- (a) distribucion: una sola serie, sin leyenda ----------
    bins = np.arange(0.50, 1.001, 0.05)
    a.hist(lo["acc"], bins=bins, color=AZUL, edgecolor="white", linewidth=1.2)
    # Techo holgado: las anotaciones van ARRIBA de las barras, nunca encima.
    a.set_ylim(0, 14.5)
    med = lo["acc"].median()
    a.axvline(med, color=TINTA, linestyle="--", linewidth=1.0)
    a.text(med + 0.012, 13.6, f"mediana {med:.3f}", fontsize=7.5, color=TINTA,
           va="top")
    a.axvline(0.5, color=MUTED, linewidth=0.8)
    a.text(0.508, 13.6, "azar", fontsize=7.5, color=MUTED, va="top")
    a.set_xlabel("Exactitud LOSO")
    a.set_ylabel("Número de sujetos")
    a.set_title("(a) Distribución entre sujetos", fontsize=9, color=TINTA, loc="left")
    estilo(a)

    # ---------- (b) intra-sujeto vs LOSO: dos categorias, con leyenda ----------
    # Mismos limites en ambos ejes: si difirieran, la diagonal dejaria de estar
    # a 45 grados y enganaria sobre cual protocolo gana.
    lim = (0.42, 1.10)          # margen arriba para que quepa la anotacion
    b.plot(lim, lim, color=MUTED, linestyle="--", linewidth=1.0, zorder=1)
    b.text(0.965, 0.915, "y = x", fontsize=7.5, color=MUTED, rotation=34)
    for mask, color, etiqueta, marca in (
            (~lo["resid"], AZUL, "resto (n=50)", "o"),
            (lo["resid"], NARANJA, "contaminación residual (n=4)", "^")):
        d = lo[mask]
        b.scatter(d["within"], d["acc"], s=34, c=color, marker=marca,
                  edgecolors="white", linewidths=0.9, zorder=3, label=etiqueta)
    b.set_xlim(*lim)
    b.set_ylim(*lim)
    b.set_xlabel("Exactitud intra-sujeto")
    b.set_ylabel("Exactitud LOSO")
    b.set_title("(b) Intra-sujeto frente a LOSO", fontsize=9, color=TINTA, loc="left")
    leg = b.legend(fontsize=7, frameon=False, loc="lower right", handletextpad=0.3)
    for t in leg.get_texts():
        t.set_color(TINTA)
    estilo(b)

    # Va en la franja libre por encima del punto mas alto (max acc = 0.993).
    n_arriba = int((lo["acc"] > lo["within"]).sum())
    b.text(0.43, 1.085, f"{n_arriba} de 54 sujetos por encima de la diagonal",
           fontsize=7.5, color=TINTA, va="top")

    fig.tight_layout(pad=0.6)
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, bbox_inches="tight")
    if not args.sin_png:
        fig.savefig(salida.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"escrito {salida}")


if __name__ == "__main__":
    main()
