"""Fusiona los CSV de los trabajadores paralelos en el CSV canonico.

`train_baseline.py --log-tag shardN` hace que cada proceso escriba su propio
`..._<base>__shardN.csv`. Este script los une en `..._<base>.csv`, deduplicando por
`run_id` (una corrida repetida en dos shards se queda con la primera aparicion) y
ordenando por scenario/seed/fold para que el fichero quede estable.

Uso:
  python scripts/merge_shards.py results/reports/openbmi/baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica.csv
  python scripts/merge_shards.py --borrar-shards <ruta_base.csv>

Se puede pasar el CSV base aunque todavia no exista: se crea a partir de los shards.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

ORDEN = ["scenario", "seed", "sub", "test_sub", "session", "fold"]


def fusionar(base: Path, borrar: bool = False) -> Path:
    shards = sorted(base.parent.glob(f"{base.stem}__*.csv"))
    if not shards:
        sys.exit(f"No hay shards para {base.name} (patron: {base.stem}__*.csv)")
    partes = []
    if base.exists():
        partes.append(pd.read_csv(base))
    for s in shards:
        partes.append(pd.read_csv(s))
    todo = pd.concat(partes, ignore_index=True)
    antes = len(todo)
    if "run_id" in todo.columns:
        # Los CSV de curvas tienen VARIAS filas por run_id (una por epoca), asi
        # que deduplicar solo por run_id se cargaria la curva entera menos su
        # primera epoca.
        # Lo mismo con la calibracion de pocos ensayos: una fila por (k, rep, metodo).
        if "epoch" in todo.columns:
            clave = ["run_id", "epoch"]
        elif {"k", "rep", "metodo"} <= set(todo.columns):
            clave = ["run_id", "k", "rep", "metodo"]
        else:
            clave = "run_id"
        todo = todo.drop_duplicates(subset=clave, keep="first")
    cols = [c for c in ORDEN if c in todo.columns]
    if cols:
        todo = todo.sort_values(cols, kind="stable").reset_index(drop=True)

    tmp = base.with_suffix(".tmp")
    todo.to_csv(tmp, index=False)
    tmp.replace(base)
    print(f"{len(shards)} shards + base -> {base.name}: {antes} filas leidas, "
          f"{len(todo)} tras deduplicar por run_id")
    if "seed" in todo.columns:
        print("  filas por semilla:", todo.seed.value_counts().sort_index().to_dict())
    if borrar:
        for s in shards:
            s.unlink()
        print(f"  {len(shards)} shards borrados")
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_csv", help="ruta del CSV canonico (sin el sufijo __shardN)")
    ap.add_argument("--borrar-shards", action="store_true",
                    help="elimina los CSV de shard tras fusionar (hacerlo solo tras verificar)")
    a = ap.parse_args()
    fusionar(Path(a.base_csv), a.borrar_shards)


if __name__ == "__main__":
    main()
