#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
descargar.py — descarga SELECTIVA de WBCIC-SHU desde Figshare.

El dataset se publica como UN solo ZIP de 61,1 GiB (DOI 10.25452/figshare.plus.22671172).
Bajarlo entero y descomprimirlo exige ~127 GiB de disco y no cabe en la mayoria de los
pods. No hace falta: el ZIP es ZIP64 con cada fichero comprimido por separado, y el
servidor de Figshare admite peticiones por rango (HTTP 206). Asi que se lee el indice
central (0,3 MiB) y se descarga solo lo que se necesita.

Del contenido real (1944 entradas, inspeccionado el 2026-09-06):
    derivatives/2C dataset_processeddata/   153 .mat, 6,14 GiB -> LO QUE USAMOS
        Ya segmentados y etiquetados: (58 ch, 1000 muestras, 200 trials) a 250 Hz,
        etiquetas 1=izquierda y 2=derecha, 100 de cada una.
    sourcedata/2C dataset/                  306 .bdf, 41,0 GiB -> NO SE PUEDE USAR
        Es la grabacion continua a 1000 Hz, pero LAS ETIQUETAS DE TRIAL NO ESTAN AHI:
        `evt.bdf` devuelve 2 anotaciones (codigos 7 y 8) en vez de los 200 eventos, y
        `data.bdf` no trae ninguna. Sin descifrar el formato de eventos de Neuracle no
        hay nada que segmentar. Se deja el modo --subs por si alguna vez se resuelve.
    code/                                   codigo de los autores (FBCNet, etc.)

DIFERENCIA DE PREPROCESADO QUE HAY QUE DECLARAR
    El espectro de `derivatives` llega hasta ~40 Hz con el notch de 50 Hz ya aplicado.
    Es MAS ANCHO que nuestros 8-30 Hz, asi que nuestro Butterworth se aplica encima y la
    banda queda igual; el remuestreo de 250 a 200 Hz tambien se recupera. Lo que NO se
    recupera es la limpieza por ICA: nuestro cache de OpenBMI la lleva y este no.

Uso tipico:

    # 1) indice (unos KB) y metadatos del montaje
    python scripts/wbcic/descargar.py --indice
    python scripts/wbcic/descargar.py --meta

    # 2) los 51 sujetos, sesion 1 (2,05 GiB)
    python scripts/wbcic/descargar.py --deriv 1-51 --sesiones 1

Es reanudable: un fichero ya presente con el tamano correcto no se vuelve a bajar.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import struct
import threading
import time
import urllib.request
import zlib
from pathlib import Path

ARTICLE = 22671172
FILE_ID = 51001884
URL = f"https://ndownloader.figshare.com/files/{FILE_ID}"
TOTAL = 65630556299                      # tamano del ZIP, verificado contra la API
RAIZ = "WBCIC_SHU Motor Imagery dataset"
DESTINO = Path("data/raw/wbcic")
INDICE = DESTINO / "_indice_zip.json"

META = [f"{RAIZ}/{n}" for n in (
    "dataset_description.json", "participants.json", "participants_2C.tsv",
    "participants_3C.tsv", "README.md", "CHANGES",
    "task-motorimagery_channels.tsv", "task-motorimagery_electrodes.tsv",
    "task-motorimagery_coordsystem.json", "task-motorimagery_eeg.json",
    "task-motorimagery_events.json")]


def _get(inicio: int, fin: int, reintentos: int = 4) -> bytes:
    """Descarga [inicio, fin] inclusive. La URL firmada de Figshare caduca en 10 s, asi
    que se pide siempre a `ndownloader`, que redirige y firma de nuevo en cada intento."""
    ultimo = None
    for k in range(reintentos):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={inicio}-{fin}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                if r.status not in (200, 206):
                    raise OSError(f"HTTP {r.status}")
                return r.read()
        except Exception as e:                       # red inestable, firma caducada, 403
            ultimo = e
            time.sleep(2 ** k)
    raise SystemExit(f"x Fallo al pedir el rango {inicio}-{fin}: {ultimo}")


def leer_indice(forzar: bool = False) -> list[dict]:
    """Indice central del ZIP: nombre, tamanos y desplazamiento de cada fichero."""
    if INDICE.exists() and not forzar:
        return json.loads(INDICE.read_text(encoding="utf-8"))

    cola = _get(TOTAL - 70000, TOTAL - 1)
    k = cola.rfind(b"PK\x06\x06")                    # EOCD64: el ZIP pasa de 4 GiB
    if k < 0:
        raise SystemExit("x No se encontro el EOCD64. El fichero no es un ZIP64.")
    n_ent, cd_size, cd_off = (struct.unpack_from("<Q", cola, k + o)[0]
                              for o in (32, 40, 48))
    d = _get(cd_off, cd_off + cd_size - 1)

    ent, off = [], 0
    while off < len(d) and d[off:off + 4] == b"PK\x01\x02":
        (_, _, _, mth, _, _, _, csz, usz, nl, el, cl,
         _, _, _, lho) = struct.unpack_from("<HHHHHHIIIHHHHHII", d, off + 4)
        nombre = d[off + 46:off + 46 + nl].decode("utf-8", "replace")
        extra = d[off + 46 + nl:off + 46 + nl + el]
        if 0xFFFFFFFF in (usz, csz, lho):            # los reales van en el extra 0x0001
            p = 0
            while p + 4 <= len(extra):
                hid, hsz = struct.unpack_from("<HH", extra, p)
                cuerpo, q = extra[p + 4:p + 4 + hsz], 0
                if hid == 1:
                    if usz == 0xFFFFFFFF:
                        usz, = struct.unpack_from("<Q", cuerpo, q); q += 8
                    if csz == 0xFFFFFFFF:
                        csz, = struct.unpack_from("<Q", cuerpo, q); q += 8
                    if lho == 0xFFFFFFFF:
                        lho, = struct.unpack_from("<Q", cuerpo, q); q += 8
                p += 4 + hsz
        ent.append({"n": nombre, "usz": usz, "csz": csz, "lho": lho, "mth": mth})
        off += 46 + nl + el + cl
    if len(ent) != n_ent:
        raise SystemExit(f"x El indice trae {len(ent)} entradas, el ZIP declara {n_ent}.")
    INDICE.parent.mkdir(parents=True, exist_ok=True)
    INDICE.write_text(json.dumps(ent), encoding="utf-8")
    return ent


def extraer(e: dict, destino: Path) -> bool:
    """Descarga y descomprime UNA entrada. Devuelve True si la bajo, False si ya estaba."""
    salida = destino / e["n"]
    if salida.exists() and salida.stat().st_size == e["usz"]:
        return False
    # La cabecera local repite el nombre y trae su propio campo extra, cuyo tamano no
    # esta en el indice central: hay que leerla para saber donde empiezan los datos.
    cab = _get(e["lho"], e["lho"] + 29)
    if cab[:4] != b"PK\x03\x04":
        raise SystemExit(f"x Cabecera local invalida en {e['n']}")
    nl, el = struct.unpack_from("<HH", cab, 26)
    ini = e["lho"] + 30 + nl + el
    crudo = _get(ini, ini + e["csz"] - 1)
    if e["mth"] == 0:
        datos = crudo
    elif e["mth"] == 8:
        datos = zlib.decompressobj(-zlib.MAX_WBITS).decompress(crudo)
    else:
        raise SystemExit(f"x Metodo de compresion {e['mth']} no soportado en {e['n']}")
    if len(datos) != e["usz"]:
        raise SystemExit(f"x {e['n']}: se esperaban {e['usz']} B y salieron {len(datos)}")
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_bytes(datos)
    return True


def _rango(txt: str) -> list[int]:
    out = []
    for parte in txt.split(","):
        if "-" in parte:
            a, b = parte.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(parte))
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indice", action="store_true", help="solo leer y resumir el indice")
    ap.add_argument("--forzar-indice", action="store_true")
    ap.add_argument("--meta", action="store_true",
                    help="descargar los ficheros de metadatos BIDS (unos KB)")
    ap.add_argument("--subs", default=None,
                    help="sujetos 2C a descargar en BDF CRUDO: `1`, `1-5`, `1,3,7`, `1-51`. "
                         "Ojo: las etiquetas de trial no se han localizado en el crudo "
                         "(evt.bdf trae 2 anotaciones, no 200). Usa --deriv salvo que "
                         "eso se resuelva.")
    ap.add_argument("--deriv", default=None,
                    help="sujetos 2C a descargar de `derivatives`: ya segmentados y "
                         "etiquetados, (58 ch, 1000 muestras, 200 trials) a 250 Hz. "
                         "Mismo formato de rango que --subs.")
    ap.add_argument("--sesiones", default="1,2,3")
    ap.add_argument("--hilos", type=int, default=4,
                    help="descargas simultaneas. Medido el 2026-09-06: 1 conexion da "
                         "0,56 MiB/s y 4 dan 1,24 MiB/s, asi que el limite es por "
                         "conexion pero no escala en linea recta.")
    ap.add_argument("--destino", type=Path, default=DESTINO)
    a = ap.parse_args()

    ent = leer_indice(a.forzar_indice)
    bdf2c = [e for e in ent
             if e["n"].startswith(f"{RAIZ}/sourcedata/2C dataset/") and e["n"].endswith(".bdf")]
    print(f"Indice: {len(ent)} entradas | 2C: {len(bdf2c)} .bdf, "
          f"{sum(e['usz'] for e in bdf2c) / 2**30:.1f} GiB sin comprimir, "
          f"{sum(e['csz'] for e in bdf2c) / 2**30:.1f} GiB de descarga", flush=True)
    if a.indice:
        return 0

    pedidos = []
    if a.meta:
        idx = {e["n"]: e for e in ent}
        faltan = [n for n in META if n not in idx]
        if faltan:
            print(f"  [aviso] no estan en el ZIP: {faltan}", flush=True)
        pedidos += [idx[n] for n in META if n in idx]
    ses = {f"ses-{int(s):02d}" for s in a.sesiones.split(",")}
    if a.subs:
        subs = {f"sub-{i:03d}" for i in _rango(a.subs)}
        pedidos += [e for e in bdf2c
                    if e["n"].split("/")[3] in subs and e["n"].split("/")[4] in ses]
    if a.deriv:
        subs = {f"sub-{i:03d}" for i in _rango(a.deriv)}
        pedidos += [e for e in ent
                    if e["n"].startswith(f"{RAIZ}/derivatives/2C dataset_processeddata/")
                    and e["n"].endswith(".mat")
                    and any(f"/{s}/" in e["n"] for s in subs)
                    and any(f"/{s}/" in e["n"] for s in ses)]
    if not pedidos:
        ap.error("nada que descargar: usa --meta, --subs, --deriv o --indice")

    total = sum(e["csz"] for e in pedidos)
    print(f"A descargar: {len(pedidos)} ficheros, {total / 2**30:.2f} GiB "
          f"con {a.hilos} hilos\n", flush=True)

    t0, estado = time.time(), {"hecho": 0, "n": 0, "saltados": 0}
    cerrojo = threading.Lock()

    def trabajo(e):
        nuevo = extraer(e, a.destino)
        with cerrojo:
            estado["n"] += 1
            k, corto = estado["n"], e["n"].split("/", 2)[-1]
            if not nuevo:
                estado["saltados"] += 1
                print(f"  [{k}/{len(pedidos)}] ya estaba: {corto}", flush=True)
                return
            estado["hecho"] += e["csz"]
            tr = time.time() - t0
            vel = estado["hecho"] / tr / 2**20
            resta = (total - estado["hecho"]) / (estado["hecho"] / tr) / 60
            print(f"  [{k}/{len(pedidos)}] {corto}  {e['usz'] / 2**20:.1f} MiB  |  "
                  f"{vel:.2f} MiB/s  |  ~{resta:.0f} min rest.", flush=True)

    with cf.ThreadPoolExecutor(max_workers=a.hilos) as ex:
        for fut in cf.as_completed([ex.submit(trabajo, e) for e in pedidos]):
            fut.result()          # propaga cualquier SystemExit de un hilo
    print(f"\nListo en {(time.time() - t0) / 60:.1f} min "
          f"({estado['saltados']} ya estaban) -> {a.destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
