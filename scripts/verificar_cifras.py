#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""verificar_cifras.py — recalcula toda cifra del informe y la contrasta con lo declarado.

POR QUE EXISTE
    En este trabajo ya se cometieron dos errores numericos que no detecto ninguna prueba:
    restar dos cifras procedentes de regimenes de entrenamiento distintos, y publicar en
    una misma fila de tabla dos definiciones de desviacion tipica. Los dos eran invisibles
    leyendo el documento, porque una cifra sola nunca parece mal.

    Este script cierra ese hueco. Cada cifra del informe queda declarada aqui junto a la
    funcion que la produce a partir de los CSV, de modo que reejecutarlo dice si el
    documento sigue siendo coherente con los datos. `BITACORA_NUMERICA.md` documenta cada
    entrada en prosa; este fichero es su parte ejecutable.

COMO SE USA
    ~/anaconda3/envs/eeg/python.exe scripts/verificar_cifras.py
    ~/anaconda3/envs/eeg/python.exe scripts/verificar_cifras.py --verboso

    Salida: una linea por cifra. `ok` si coincide con lo declarado dentro de la tolerancia
    de redondeo, `DISCREPA` si no. Devuelve codigo 1 si alguna discrepa.

COMO SE ANADE UNA CIFRA
    Se agrega una entrada a CIFRAS con: identificador, valor declarado en el documento,
    donde aparece, y una funcion sin argumentos que lo recalcula desde los CSV. La
    tolerancia por defecto es media unidad de la ultima cifra declarada.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from ruido_flotante import wilcoxon_pareado  # Wilcoxon sobre diferencias redondeadas a 1e-9

# ---------------------------------------------------------------- rutas de datos

LOSO = {
    "eegnet": "results/pod_baseline/baseline_eegnet_loso_completo_global_epochs_cache_ica.csv",
    "conformer": "results/pod_conformer/openbmi/baseline_conformer_loso_completo_global_epochs_cache_ica.csv",
    "labram": "results/pod_labram/baseline_labram_loso_completo_global_full_uv1_lr0.0001_epochs_cache_ica.csv",
    "eegmamba": "results/pod_eegmamba/baseline_eegmamba_loso_completo_global_flatten_uv1_lr0.001_epochs_cache_ica.csv",
}

WITHIN = {
    "eegnet": "results/pod_baseline/baseline_eegnet_within_completo_global_porsesion_epochs_cache_ica.csv",
    "conformer": "results/pod_conformer/openbmi/baseline_conformer_within_completo_global_porsesion_epochs_cache_ica.csv",
    "labram": "results/pod_labram/baseline_labram_within_completo_global_porsesion_full_uv1_lr0.0001_epochs_cache_ica.csv",
    "eegmamba": "results/pod_eegmamba_flatten/baseline_eegmamba_within_completo_global_porsesion_flatten_uv1_lr0.001_epochs_cache_ica.csv",
}

CRUZADO = "results/punto6/reports/cruzado/cruzado_{m}.csv"
NATIVO = {
    "eegnet": "results/punto6/reports/wbcic_2c/baseline_eegnet_all_comun_wbcic_global_epochs_cache_ica.csv",
    "conformer": "results/punto6/reports/wbcic_2c/baseline_conformer_all_comun_wbcic_global_epochs_cache_ica.csv",
    "labram": "results/punto6/reports/wbcic_2c/baseline_labram_all_comun_wbcic_global_full_uv1_lr0.0001_epochs_cache_ica.csv",
    "eegmamba": "results/punto6/reports/wbcic_2c/baseline_eegmamba_all_comun_wbcic_global_flatten_uv1_lr0.001_epochs_cache_ica.csv",
}
HOLD_OPENBMI = "results/punto6b/holdout44/holdout_openbmi_{m}.csv"
HOLD_WBCIC = "results/punto6b/holdout44/holdout_wbcic_{m}.csv"

# Se considera de bajo rendimiento a quien NO SUPERA el 70 %, de modo que la comparacion
# es `<=` y no `<`. La diferencia no es cosmetica: en el protocolo intra-sujeto hay
# participantes con exactitud exactamente 0.700 y el criterio los incluye.
UMBRAL_BAJO_RENDIMIENTO = 0.70  # vidaurre2010

# Tolerancia numerica explicita de la comparacion con el umbral. Una media de exactitudes
# que vale exactamente 0.700 en papel puede salir como 0.7000000000000001 segun el orden de
# la suma (sub-31 con la media de tres semillas de EEGNet), y entonces `<= 0.70` da un
# recuento distinto. Se redondea a 9 decimales (1e-9, tres ordenes por debajo de cualquier
# diferencia real: las exactitudes son multiplos de 1/20 o 1/400 promediados) antes de
# comparar. Comprobado: con este redondeo ninguna cifra publicada cambia (ver
# avance_7/analisis/07_VERIFICACION.md).
DECIMALES_UMBRAL = 9


def le_umbral(serie):
    """Indicador `exactitud <= 0.70` robusto a la coma flotante (redondeo a 1e-9)."""
    return pd.Series(serie).round(DECIMALES_UMBRAL) <= UMBRAL_BAJO_RENDIMIENTO


# ------------------------------------------------------------------ utilidades

def _loso(m: str) -> pd.DataFrame:
    return pd.read_csv(LOSO[m]).sort_values("test_sub")


def acc_loso(m: str) -> float:
    """Exactitud LOSO: media de los 54 folds, uno por sujeto."""
    return _loso(m).groupby("test_sub").acc.mean().mean()


def acc_visto(m: str) -> float:
    return _loso(m).groupby("test_sub").acc_visto.mean().mean()


def ids(m: str, metrica: str = "acc") -> float:
    """IDS con la formula sobre las MEDIAS de los folds, que es como se reporta.

    Promediar el indicador de cada fold da otra cifra (difiere en la tercera decimal) y
    estima otra cantidad. Ver BITACORA_NUMERICA.md, entrada IDS.
    """
    d = _loso(m)
    v, n = d[f"{metrica}_visto"].mean(), d[metrica].mean()
    return (v - n) / v


def acc_within(m: str) -> float:
    """Exactitud intra-sujeto promediada ANTES por sujeto.

    Cada persona pesa lo mismo con independencia de cuantas semillas se le ejecutaron:
    EEGNet y Conformer tienen 3 y LaBraM y EEGMamba 1.
    """
    return pd.read_csv(WITHIN[m]).groupby("sub").acc.mean().mean()


def sd_within(m: str) -> float:
    """Desviacion ENTRE SUJETOS, no entre filas.

    La desviacion de todas las filas incluye la variacion entre particiones y semillas y
    es mayor. Mezclar ambas en una misma fila de tabla fue uno de los errores corregidos.
    """
    return pd.read_csv(WITHIN[m]).groupby("sub").acc.mean().std()


def sd_loso(m: str) -> float:
    return _loso(m).groupby("test_sub").acc.mean().std()


def p_wilcoxon_ids(m: str) -> float:
    """Wilcoxon pareado entre visto y no visto, sobre los 54 folds."""
    d = _loso(m)
    return wilcoxon_pareado(d.acc_visto, d.acc).pvalue


def p_holm_ids(m: str) -> float:
    """Valor de Holm del contraste de `m` entre los cuatro."""
    ps = sorted(((p_wilcoxon_ids(k), k) for k in LOSO))
    ajustados, previo = {}, 0.0
    for i, (p, k) in enumerate(ps):
        previo = max(previo, p * (len(ps) - i))
        ajustados[k] = min(previo, 1.0)
    return ajustados[m]


def bajo_umbral_within(m: str) -> int:
    d = pd.read_csv(WITHIN[m]).groupby("sub").acc.mean()
    return int(le_umbral(d).sum())


def bajo_umbral_loso(m: str) -> int:
    return int(le_umbral(_loso(m).groupby("test_sub").acc.mean()).sum())


def _matriz_bajo() -> pd.DataFrame:
    """Matriz 54 x 4 de indicadores de bajo rendimiento, protocolo intra-sujeto."""
    acc = pd.DataFrame({m: pd.read_csv(WITHIN[m]).groupby("sub").acc.mean()
                        for m in WITHIN})
    return acc.round(DECIMALES_UMBRAL) <= UMBRAL_BAJO_RENDIMIENTO


def reparto_k(i: int) -> int:
    """Participantes que quedan bajo umbral en exactamente `i` arquitecturas."""
    return int((_matriz_bajo().sum(axis=1) == i).sum())


def esperado_k(i: int) -> float:
    """Lo mismo, esperado si las cuatro fallaran de forma independiente.

    Se convoluciona la Bernoulli de cada arquitectura con su propia tasa, de modo que la
    referencia conserva las marginales y solo elimina la asociacion entre fallos.
    """
    bajo = _matriz_bajo()
    dist = np.array([1.0])
    for prob in bajo.mean():
        dist = np.convolve(dist, [1 - prob, prob])
    return float(dist[i] * len(bajo))


def interseccion_bajo_umbral() -> int:
    """Sujetos por debajo del umbral con LAS CUATRO arquitecturas a la vez, intra-sujeto."""
    conjuntos = []
    for m in WITHIN:
        d = pd.read_csv(WITHIN[m]).groupby("sub").acc.mean()
        conjuntos.append(set(d[le_umbral(d)].index))
    return len(set.intersection(*conjuntos))


def correlacion_within_loso(m: str) -> float:
    """Correlacion por participante entre la exactitud intra-sujeto y la LOSO.

    Se reporta POR ARQUITECTURA y no promediada. El promedio de las cuatro oculta que el
    valor va de 0.34 a 0.69 segun el modelo, que es justamente lo que interesa: los dos
    protocolos ordenan a las personas de forma distinta, y cuanto lo hacen depende de la
    arquitectura.
    """
    w = pd.read_csv(WITHIN[m]).groupby("sub").acc.mean()
    l = _loso(m).groupby("test_sub").acc.mean()
    comun = sorted(set(w.index) & set(l.index))
    return float(np.corrcoef(w.loc[comun], l.loc[comun])[0, 1])


def transferencia(m: str, politica: str = "source") -> float:
    """Transferencia de disparo cero: modelo `all` de OpenBMI sobre el 20 % de WBCIC."""
    d = pd.read_csv(CRUZADO.format(m=m))
    return d[d.norm_destino == politica].acc.mean()


def nativo(m: str) -> float:
    """Techo nativo: modelo `all` de WBCIC sobre su propio conjunto reservado."""
    return pd.read_csv(NATIVO[m]).acc_reservado.iloc[0]


def hold_persona(m: str, politica: str = "source") -> float:
    """Modelo de 44 sujetos evaluado sobre los 10 apartados de OpenBMI."""
    d = pd.read_csv(HOLD_OPENBMI.format(m=m))
    return d[d.norm_destino == politica].acc.mean()


def hold_wbcic(m: str, politica: str = "source") -> float:
    """El MISMO modelo de 44 sujetos evaluado sobre WBCIC."""
    d = pd.read_csv(HOLD_WBCIC.format(m=m))
    return d[d.norm_destino == politica].acc.mean()


def idp(m: str) -> float:
    """IDP: (participante nuevo - protocolo nuevo) / participante nuevo.

    Ambos terminos sobre gente NO VISTA y con un mismo conjunto de pesos, el de 44
    sujetos, de modo que no vuelve a contar el cambio de persona que IDS ya recoge.
    """
    pn = hold_persona(m)
    return (pn - hold_wbcic(m)) / pn


def n_params(m: str) -> int:
    return int(_loso(m).n_params.iloc[0])


# ----------------------------------------------------------------- declaracion

Cifra = tuple[str, float, str, Callable[[], float]]

CIFRAS: list[Cifra] = []


def declarar(ident: str, valor: float, donde: str, fn: Callable[[], float]) -> None:
    CIFRAS.append((ident, valor, donde, fn))


ARQ = {"eegnet": "EEGNet", "conformer": "Conformer",
       "labram": "LaBraM", "eegmamba": "EEGMamba"}

# tab:conformer y tab:bootstrap -- exactitud LOSO
for k, v in {"eegnet": 0.763, "conformer": 0.752, "labram": 0.754, "eegmamba": 0.717}.items():
    declarar(f"acc_loso.{k}", v, "tab:conformer, tab:bootstrap, tab:robustez",
             lambda k=k: acc_loso(k))

# tab:conformer -- exactitud LOSO sobre sujetos vistos
for k, v in {"eegnet": 0.783, "conformer": 0.786, "labram": 0.798, "eegmamba": 0.742}.items():
    declarar(f"acc_visto.{k}", v, "tab:conformer, tab:dsi", lambda k=k: acc_visto(k))

# tab:conformer -- exactitud intra-sujeto y su desviacion ENTRE SUJETOS
for k, v in {"eegnet": 0.634, "conformer": 0.602, "labram": 0.665, "eegmamba": 0.577}.items():
    declarar(f"acc_within.{k}", v, "tab:conformer, tab:bootstrap",
             lambda k=k: acc_within(k))
for k, v in {"eegnet": 0.107, "conformer": 0.117, "labram": 0.152, "eegmamba": 0.101}.items():
    declarar(f"sd_within.{k}", v, "tab:conformer", lambda k=k: sd_within(k))

# tab:robustez -- desviacion LOSO entre participantes
for k, v in {"eegnet": 0.120, "conformer": 0.125, "labram": 0.108, "eegmamba": 0.102}.items():
    declarar(f"sd_loso.{k}", v, "tab:robustez", lambda k=k: sd_loso(k))

# tab:dsi -- IDS en exactitud y en kappa
for k, v in {"eegnet": 0.026, "conformer": 0.044, "labram": 0.055, "eegmamba": 0.034}.items():
    declarar(f"ids_acc.{k}", v, "tab:dsi, tab:conformer, tab:robustez", lambda k=k: ids(k))
for k, v in {"eegnet": 0.071, "conformer": 0.121, "labram": 0.148, "eegmamba": 0.105}.items():
    declarar(f"ids_kappa.{k}", v, "tab:dsi, tab:conformer",
             lambda k=k: ids(k, "kappa"))

# tab:dsi -- contrastes
for k, v in {"eegnet": 0.226, "conformer": 0.053, "labram": 0.006, "eegmamba": 0.038}.items():
    declarar(f"p_ids.{k}", v, "tab:dsi", lambda k=k: p_wilcoxon_ids(k))
for k, v in {"eegnet": 0.226, "conformer": 0.113, "labram": 0.026, "eegmamba": 0.113}.items():
    declarar(f"p_holm_ids.{k}", v, "tab:dsi", lambda k=k: p_holm_ids(k))

# tab:robustez -- recuento bajo umbral en LOSO
for k, v in {"eegnet": 17, "conformer": 18, "labram": 15, "eegmamba": 28}.items():
    declarar(f"bajo_umbral_loso.{k}", v, "tab:robustez", lambda k=k: bajo_umbral_loso(k))

# discusion -- participantes de bajo rendimiento, protocolo intra-sujeto
for k, v in {"eegnet": 41, "conformer": 45, "labram": 37, "eegmamba": 48}.items():
    declarar(f"bajo_umbral_within.{k}", v, "discusion, segundo limite",
             lambda k=k: bajo_umbral_within(k))
declarar("interseccion_bajo_umbral", 34, "discusion, segundo limite",
         interseccion_bajo_umbral)
for i, v in {0: 5, 1: 3, 2: 4, 3: 8, 4: 34}.items():
    declarar(f"reparto_k.{i}", v, "fig:solapamiento, panel (b)", lambda i=i: reparto_k(i))
for i, v in {0: 0.1, 3: 22.9, 4: 20.8}.items():
    declarar(f"esperado_k.{i}", v, "fig:solapamiento, panel (b)",
             lambda i=i: esperado_k(i))
for k, v in {"eegnet": 0.342, "conformer": 0.387, "labram": 0.686,
             "eegmamba": 0.509}.items():
    declarar(f"correlacion_within_loso.{k}", v, "discusion, contraste con LOSO",
             lambda k=k: correlacion_within_loso(k))

# tab:transferencia -- punto 6
for k, v in {"eegnet": 0.626, "conformer": 0.611, "labram": 0.613}.items():
    declarar(f"transferencia.{k}", v, "tab:transferencia", lambda k=k: transferencia(k))
for k, v in {"eegnet": 0.808, "conformer": 0.800, "labram": 0.814, "eegmamba": 0.762}.items():
    declarar(f"nativo.{k}", v, "tab:transferencia", lambda k=k: nativo(k))

# tab:descomposicion -- experimento 44 + 10
for k, v in {"eegnet": 0.742, "conformer": 0.718, "labram": 0.745, "eegmamba": 0.700}.items():
    declarar(f"hold_persona.{k}", v, "tab:descomposicion", lambda k=k: hold_persona(k))
for k, v in {"eegnet": 0.621, "conformer": 0.618, "labram": 0.628, "eegmamba": 0.598}.items():
    declarar(f"hold_wbcic.{k}", v, "tab:descomposicion", lambda k=k: hold_wbcic(k))

# tab:robustez -- IDP
for k, v in {"eegnet": 0.163, "conformer": 0.139, "labram": 0.157, "eegmamba": 0.146}.items():
    declarar(f"idp.{k}", v, "tab:robustez", lambda k=k: idp(k))

# sec:representaciones -- numero de parametros
for k, v in {"eegnet": 2898, "conformer": 710146, "labram": 5818138,
             "eegmamba": 3376445}.items():
    declarar(f"n_params.{k}", v, "sec:representaciones", lambda k=k: n_params(k))


# ---------------------------------------------- tamanos del diseno experimental
# El texto describio durante meses que LOSO entrenaba con 53 sujetos cuando el codigo
# siempre uso 51 (mas 2 de validacion y 1 de prueba). Las cifras estaban bien y la
# descripcion no; por eso los tamanos del diseno se verifican igual que las cifras.

def _col_unica(ruta: str, col: str) -> float:
    vals = pd.read_csv(ruta)[col].unique()
    assert len(vals) == 1, f"{col} no es constante en {ruta}: {vals[:5]}"
    return float(vals[0])


for k in LOSO:
    declarar(f"loso_n_train.{k}", 18360, "sec LOSO: 51 sujetos x 360",
             lambda k=k: _col_unica(LOSO[k], "n_train"))
    declarar(f"loso_n_val.{k}", 800, "sec LOSO: 2 sujetos x 400, parada temprana",
             lambda k=k: _col_unica(LOSO[k], "n_val"))
    declarar(f"loso_n_visto.{k}", 2040, "sec:dsi: 10 % de 51 sujetos",
             lambda k=k: _col_unica(LOSO[k], "n_visto"))
    declarar(f"within_n_train.{k}", 144, "fig:within",
             lambda k=k: _col_unica(WITHIN[k], "n_train"))
    declarar(f"within_n_val.{k}", 36, "fig:within, sec:seleccion-intra",
             lambda k=k: _col_unica(WITHIN[k], "n_val"))
    declarar(f"within_n_test.{k}", 20, "fig:within",
             lambda k=k: _col_unica(WITHIN[k], "n_test"))


# ------------------------------- seleccion del modelo durante el calentamiento
# n_epochs == 30 equivale a que la mejor epoca este en el calentamiento (0 a 9): la
# paciencia no corre durante el, asi que la parada mas temprana posible es la epoca 30.

def _calentamiento(ruta: str, que: str) -> float:
    d = pd.read_csv(ruta)
    m = d.n_epochs == 30
    if que == "prop":
        return 100 * m.mean()
    return d.acc[m].mean() if que == "esas" else d.acc[~m].mean()


_SEL = {
    "within": {"eegnet": (15.2, 0.511, 0.656), "conformer": (57.6, 0.521, 0.714),
               "labram": (21.1, 0.529, 0.701), "eegmamba": (85.6, 0.543, 0.777)},
    "loso": {"eegnet": (7.4, 0.828, 0.757), "conformer": (13.0, 0.685, 0.762),
             "labram": (25.9, 0.708, 0.770), "eegmamba": (46.3, 0.712, 0.721)},
}
for prot, tabla in _SEL.items():
    rutas = WITHIN if prot == "within" else LOSO
    for k, (prop, esas, resto) in tabla.items():
        for que, v in (("prop", prop), ("esas", esas), ("resto", resto)):
            declarar(f"calentamiento_{prot}_{que}.{k}", v, "tab:seleccion-intra",
                     lambda k=k, que=que, rutas=rutas: _calentamiento(rutas[k], que))


# ------------------------- cociente protocolo / persona en el protocolo 44 + 10
# "El protocolo duplica la caida por persona" se presento como general y no lo es: en el
# Conformer las dos caidas son practicamente iguales. Se verifica cada cociente.

def _vistos_hold(m: str) -> float:
    import glob
    f = glob.glob(f"results/punto6b/openbmi/baseline_{m}_all_*hold10s2026*.csv")
    assert len(f) == 1, f"se esperaba un CSV de 44 + 10 para {m}: {f}"
    return pd.read_csv(f[0]).acc.iloc[0]


def cociente_protocolo_persona(m: str) -> float:
    persona = _vistos_hold(m) - hold_persona(m)
    return (hold_persona(m) - hold_wbcic(m)) / persona


for k, v in {"eegnet": 2.0, "conformer": 1.0, "labram": 1.7, "eegmamba": 1.9}.items():
    declarar(f"cociente_protocolo_persona.{k}", v, "sec:descomposicion, discusion",
             lambda k=k: cociente_protocolo_persona(k))
declarar("idp_sobre_ids.min", 2.8, "sec:aplicacion-robustez",
         lambda: min(idp(m) / ids(m) for m in LOSO))
declarar("idp_sobre_ids.max", 6.4, "sec:aplicacion-robustez",
         lambda: max(idp(m) / ids(m) for m in LOSO))


# ------------------------------------------------------------ fig:curvas
# El texto que comenta las curvas tenia el rango de minimos mal (0.41 a 0.47, real 0.39 a
# 0.50) y generalizaba a las cuatro un adelanto del minimo que solo ocurre en dos.

def _curva(ds: str, m: str) -> pd.DataFrame:
    import glob
    f = [x for x in glob.glob(f"results/punto6/reports/{ds}/curvas/baseline_{m}_all_*.csv")
         if "hold" not in x]
    assert len(f) == 1, f"curva de {m} en {ds}: {f}"
    return pd.read_csv(f[0])


def _ep_min(ds: str, m: str) -> int:
    d = _curva(ds, m)
    return int(d.epoch[d.val_loss.idxmin()])


declarar("curvas_min_val_a.min", 0.39, "fig:curvas, panel (a)",
         lambda: min(_curva("openbmi", m).val_loss.min() for m in LOSO))
declarar("curvas_min_val_a.max", 0.50, "fig:curvas, panel (a)",
         lambda: max(_curva("openbmi", m).val_loss.min() for m in LOSO))
# 1 si el minimo llega antes en (b) que en (a); el texto dice que ocurre en 2 de 4.
declarar("curvas_minimo_se_adelanta.n", 2, "fig:curvas: Conformer y EEGMamba",
         lambda: sum(_ep_min("wbcic_2c", m) < _ep_min("openbmi", m) for m in LOSO))
declarar("curvas_oscilacion_aumenta.n", 3, "fig:curvas: tres de cuatro",
         lambda: sum(_curva("wbcic_2c", m).val_loss.diff().abs().mean()
                     > _curva("openbmi", m).val_loss.diff().abs().mean() for m in LOSO))


# ------------------------------------------- observaciones del director (2026-10)

declarar("transferencia.eegmamba", 0.589, "tab:transferencia, politica origen",
         lambda: transferencia("eegmamba"))

for k, v in {"eegnet": 0.16, "conformer": 0.26, "labram": 0.37, "eegmamba": 0.28}.items():
    def _r_ids(m=k):
        from scipy.stats import norm
        d = _loso(m)
        p = wilcoxon_pareado(d.acc_visto, d.acc).pvalue
        return float(norm.isf(p / 2) / np.sqrt(len(d)))
    declarar(f"r_ids.{k}", v, "tab:dsi, columna r", _r_ids)

# Equivalencia: IC al 90 % por remuestreo pareado, B = 10 000, semilla propia por par
# para que cada cifra no dependa del orden en que se calculan las demas. La primera
# version del texto uso una semilla compartida entre pares y dos cotas diferian en la
# tercera cifra: el verificador lo detecto.
def _ic90_dif(a: str, b: str, extremo: int) -> float:
    xa = _loso(a).groupby("test_sub").acc.mean()
    xb = _loso(b).groupby("test_sub").acc.mean()
    d = (xa - xb).to_numpy()
    rng = np.random.default_rng(2026)
    m = d[rng.integers(0, len(d), (10_000, len(d)))].mean(axis=1)
    return float(np.quantile(m, [0.05, 0.95])[extremo])


for (a, b), (lo, hi) in {("eegnet", "conformer"): (0.002, 0.020),
                         ("eegnet", "labram"): (-0.002, 0.020),
                         ("conformer", "labram"): (-0.013, 0.009),
                         ("eegnet", "eegmamba"): (0.035, 0.057),
                         ("conformer", "eegmamba"): (0.023, 0.046),
                         ("labram", "eegmamba"): (0.028, 0.046)}.items():
    declarar(f"equiv_ic90_inf.{a}-{b}", lo, "sec:wilcoxon, equivalencia",
             lambda a=a, b=b: _ic90_dif(a, b, 0))
    declarar(f"equiv_ic90_sup.{a}-{b}", hi, "sec:wilcoxon, equivalencia",
             lambda a=a, b=b: _ic90_dif(a, b, 1))

for k, v in {"eegnet": 0.799, "conformer": 0.818, "labram": 0.821, "eegmamba": 0.756}.items():
    def _val_global(m=k):
        import glob
        f = [x for x in glob.glob(f"results/punto6/reports/openbmi/baseline_{m}_all_*.csv")
             if "hold" not in x]
        return float(pd.read_csv(f[0]).acc.iloc[0])
    declarar(f"global_val.{k}", v, "cap. 4: referencia del mismo regimen", _val_global)


def _sensibilidad_resto(que: str) -> float:
    d = _loso("eegnet").set_index("test_sub").drop(["sub-06", "sub-29", "sub-32", "sub-37"])
    if que == "acc":
        return d.acc.mean()
    return (d.kappa_visto.mean() - d.kappa.mean()) / d.kappa_visto.mean()


declarar("sensibilidad_resto.acc", 0.756, "tab:sensibilidad", lambda: _sensibilidad_resto("acc"))
declarar("sensibilidad_resto.ids_kappa", 0.092, "tab:sensibilidad, limitaciones",
         lambda: _sensibilidad_resto("kappa"))


# ----------------------------- tab:descomposicion-ic: intervalos sobre sujetos
# Se remuestrean por separado los 10 apartados y los 51 de WBCIC; "vistos" queda fijo.
# Mismo generador recorriendo las cuatro arquitecturas en orden, como en el calculo
# original, y despues el promedio con una extraccion comun a las cuatro.

def _descomp_ic() -> dict:
    import glob
    rng = np.random.default_rng(2026)
    B = 10_000
    out, P, W, V = {}, [], [], []
    for m in LOSO:
        f = glob.glob(f"results/punto6b/openbmi/baseline_{m}_all_*hold10s2026*.csv")[0]
        vis = pd.read_csv(f).acc.iloc[0]
        po = pd.read_csv(HOLD_OPENBMI.format(m=m))
        po = po[po.norm_destino == "source"].groupby("sub").acc.mean().sort_index().to_numpy()
        pw = pd.read_csv(HOLD_WBCIC.format(m=m))
        pw = pw[pw.norm_destino == "source"].groupby("sub").acc.mean().sort_index().to_numpy()
        mo = po[rng.integers(0, len(po), (B, len(po)))].mean(1)
        mw = pw[rng.integers(0, len(pw), (B, len(pw)))].mean(1)
        for nombre, x in (("persona", vis - mo), ("protocolo", mo - mw),
                          ("dif", (mo - mw) - (vis - mo))):
            lo, hi = np.quantile(x, [0.025, 0.975])
            out[f"{nombre}.{m}"] = (lo, hi)
        P.append(po); W.append(pw); V.append(vis)
    # Promedio de las cuatro: una misma extraccion de sujetos para las cuatro, porque son
    # los mismos 10 apartados y los mismos 51 de WBCIC. Generador propio.
    rng = np.random.default_rng(2026)
    Pa, Wa, Va = np.stack(P), np.stack(W), np.array(V)[:, None]
    io = rng.integers(0, Pa.shape[1], (B, Pa.shape[1]))
    iw = rng.integers(0, Wa.shape[1], (B, Wa.shape[1]))
    mo, mw = Pa[:, io].mean(2), Wa[:, iw].mean(2)            # (4, B)
    for nombre, x in (("persona", (Va - mo).mean(0)), ("protocolo", (mo - mw).mean(0)),
                      ("dif", ((mo - mw) - (Va - mo)).mean(0))):
        lo, hi = np.quantile(x, [0.025, 0.975])
        out[f"{nombre}.promedio"] = (lo, hi)
    return out


_DESC = None


def descomp(clave: str, extremo: int) -> float:
    global _DESC
    if _DESC is None:
        _DESC = _descomp_ic()
    return float(_DESC[clave][extremo])


for k, v in {
    "persona.eegnet": (-0.013, 0.131), "protocolo.eegnet": (0.042, 0.201),
    "dif.eegnet": (-0.084, 0.210),
    "persona.conformer": (0.012, 0.173), "protocolo.conformer": (0.015, 0.190),
    "dif.conformer": (-0.154, 0.174),
    "persona.labram": (0.002, 0.135), "protocolo.labram": (0.039, 0.192),
    "dif.labram": (-0.092, 0.185),
    "persona.eegmamba": (-0.006, 0.109), "protocolo.eegmamba": (0.040, 0.167),
    "dif.eegmamba": (-0.065, 0.170),
    "persona.promedio": (0.000, 0.135), "protocolo.promedio": (0.037, 0.185),
    "dif.promedio": (-0.094, 0.182),
}.items():
    declarar(f"descomp_ic_inf.{k}", v[0], "tab:descomposicion-ic",
             lambda k=k: descomp(k, 0))
    declarar(f"descomp_ic_sup.{k}", v[1], "tab:descomposicion-ic",
             lambda k=k: descomp(k, 1))


def _persona_loso_ic_sup(m: str) -> float:
    d = _loso(m)
    x = (d.acc_visto - d.acc).to_numpy()
    rng = np.random.default_rng(2026)
    return float(np.quantile(x[rng.integers(0, len(x), (10_000, len(x)))].mean(1), 0.975))


declarar("persona_loso_ic_sup.min", 0.053, "sec:descomposicion, indicio de LOSO",
         lambda: min(_persona_loso_ic_sup(m) for m in LOSO))
declarar("persona_loso_ic_sup.max", 0.074, "sec:descomposicion, indicio de LOSO",
         lambda: max(_persona_loso_ic_sup(m) for m in LOSO))


# --------------------------------- EEGMamba intra-sujeto: cabezas y azar
EEGMAMBA_AVGPOOL = ("results/pod_eegmamba/baseline_eegmamba_within_completo_global_"
                    "porsesion_avgpool_uv1_lr0.001_epochs_cache_ica.csv")


def _por_sujeto(ruta: str) -> pd.Series:
    return pd.read_csv(ruta).groupby("sub").acc.mean().sort_index()


declarar("eegmamba_avgpool_acc", 0.519, "sec:eegmamba-adapt",
         lambda: _por_sujeto(EEGMAMBA_AVGPOOL).mean())
declarar("eegmamba_avgpool_p_azar", 0.017, "sec:eegmamba-adapt, Wilcoxon frente a 0.50",
         lambda: wilcoxon_pareado(_por_sujeto(EEGMAMBA_AVGPOOL) - 0.5).pvalue)


# ------------------------------------------------------------------- ejecucion

def tolerancia(valor: float) -> float:
    """Media unidad de la ultima cifra declarada."""
    if float(valor).is_integer() and abs(valor) >= 1:
        return 0.5
    texto = f"{valor!r}"
    decimales = len(texto.split(".")[1]) if "." in texto else 0
    return 0.5 * 10 ** (-decimales) + 1e-12


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verboso", action="store_true", help="listar tambien las que coinciden")
    ap.add_argument("--avance7", action="store_true",
                    help="ejecutar ademas las cifras del avance 7 (verificar_cifras_avance7.py)")
    ap.add_argument("--informe-final", action="store_true", help="verificar cifras nuevas del informe final")
    args = ap.parse_args()

    fallos, faltan = [], []
    for ident, declarado, donde, fn in CIFRAS:
        try:
            obtenido = float(fn())
        except FileNotFoundError as e:
            faltan.append((ident, Path(str(e.filename)).name))
            continue
        tol = tolerancia(declarado)
        ok = abs(obtenido - declarado) <= tol
        if not ok:
            fallos.append((ident, declarado, obtenido, donde))
        if args.verboso or not ok:
            marca = "ok      " if ok else "DISCREPA"
            print(f"{marca} {ident:30s} declarado {declarado:>10} "
                  f"obtenido {obtenido:>12.4f}   {donde}")

    print(f"\n{len(CIFRAS)} cifras declaradas, {len(fallos)} discrepan, "
          f"{len(faltan)} sin fichero de origen")
    for ident, fichero in faltan:
        print(f"  falta  {ident}: {fichero}")
    if fallos:
        print("\nRevisar antes de compilar el informe:")
        for ident, d, o, donde in fallos:
            print(f"  {ident}: documento {d}, datos {o:.4f}  ({donde})")
    codigo = 1 if fallos else 0
    if args.avance7:
        import verificar_cifras_avance7 as v7
        print()
        print("--- avance 7 ---")
        codigo = max(codigo, v7.ejecutar(args.verboso))
    if args.informe_final:
        import verificar_cifras_informe_final as vf
        codigo = max(codigo, vf.ejecutar(args.verboso))
    return codigo


if __name__ == "__main__":
    raise SystemExit(main())
