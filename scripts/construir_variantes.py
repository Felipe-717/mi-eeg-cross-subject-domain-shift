"""Materializa los cachés derivados de la campaña de octubre de 2026 (src/variantes.py).

    python scripts/construir_variantes.py --origen /workspace/cache/epochs_cache_ica \
        --destino /workspace/cache/epochs_cache_ica_ea --tipo ea

    python scripts/construir_variantes.py --origen ... --destino ..._mu --tipo banda --banda 8 13

Con `--canales <vista>` (p. ej. comun_wbcic) se guarda solo esa vista, y el alineamiento
se calcula sobre ella. Idempotente: salta los sujetos ya escritos.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import datasets            # noqa: E402
import variantes as va     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origen", required=True)
    ap.add_argument("--destino", required=True)
    ap.add_argument("--tipo", required=True, choices=["ea", "banda", "banda_fir"])
    ap.add_argument("--banda", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--dataset", default="openbmi", choices=list(datasets.disponibles()),
                    help="solo para resolver la vista de --canales")
    ap.add_argument("--canales", default=None, help="vista de canales del dataset")
    ap.add_argument("--forzar", action="store_true")
    a = ap.parse_args()
    if a.tipo in ("banda", "banda_fir") and not a.banda:
        ap.error("--tipo banda o banda_fir exige --banda LO HI")
    canales = None
    if a.canales:
        canales = datasets.get(a.dataset).CH_SETS[a.canales]
    res = va.derivar_cache(a.origen, a.destino, a.tipo, canales=canales, banda=a.banda,
                           forzar=a.forzar)
    n = len(list(Path(a.destino).glob("sub-*_full.npz")))
    print(f"{a.destino}: {len(res)} sujetos escritos ahora, {n} en total")
    if a.tipo == "ea" and res:
        rangos = sorted({r["rango"] for r in res})
        print(f"  rango de la covarianza media por sujeto: {rangos} "
              f"(de {res[0]['n_canales']} canales)")


if __name__ == "__main__":
    main()
