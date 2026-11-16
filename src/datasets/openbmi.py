"""Dataset OpenBMI / Lee2019 — DATASET PRINCIPAL de la tesis (desde 2026-07-20).

Referencia: Lee et al., *EEG dataset and OpenBMI toolbox for three BCI paradigms:
an investigation into BCI illiteracy*, GigaScience 8(5), 2019. DOI 10.1093/gigascience/giz002

POR QUÉ ESTE DATASET (ver docs/PLAN_MIGRACION_WBCIC.md):
  * **La transferencia inter-sujeto está DEMOSTRADA por DOS fuentes independientes**
    (~74-76 %). Es el único candidato donde no hay que suponerla — el fallo de MIND
    fue construir todo el pipeline antes de verificar ese punto.
    OJO: el paper del dataset (Lee 2019) NO reporta cross-subject; estas cifras son
    de terceros y con protocolos que NO son exactamente el nuestro (ver CROSS_SUBJECT_REF).
  * **54 sujetos** -> potencia estadística real para Friedman/Nemenyi (CD = 0.66).
  * **Disponible en MOABB** -> sin loader manual.
  * **2 sesiones en días distintos** -> eje de cambio de dominio temporal.
  * **Limpio de fuga**: no aparece en el Apéndice D de LaBraM (que sí lista
    BCI Competition IV-1 y EEG Motor Movement/Imagery) ni en el `pretrain_main.py`
    de EEGMamba (TUEG, PhysioNet 2018, Raw EEG Data, Siena, SZBD).
    Formulación defendible: "no se encontró en las fuentes declaradas". NUNCA "garantizado".

ESTRUCTURA (por sujeto):
    2 sesiones × (100 trials offline + 100 trials online) = 400 trials, 200 por clase

REFERENCIAS PUBLICADAS a replicar (todas WITHIN-SUJETO, CSP+LDA):
    CSP-cv   sesión 1: 70.1 %   sesión 2: 72.2 %   <- el 71.1 % del abstract
    CSP      sesión 1: 67.2 %   sesión 2: 68.5 %   <- entrena offline, prueba online
  Su configuración: 8-30 Hz (Butterworth orden 5), 100 Hz, 20 canales motores,
  ventana 1000-3500 ms. **Para replicar hay que usar SU configuración, no la nuestra.**

⚠️ Analfabetismo BCI: **53.7 % de los sujetos no alcanza el 70 %** en MI (umbral del
paper), frente al 11 % en ERP. Para esta tesis eso NO es un defecto: es el fenómeno
de variabilidad inter-sujeto que se quiere medir, ya cuantificado por los autores.
"""
from __future__ import annotations
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Dónde descarga MOABB los datos crudos
# --------------------------------------------------------------------------- #
# Los datos pesados van a D: junto al resto del corpus, NUNCA al disco del sistema
# (Lee2019 son 54 sujetos x 2 sesiones x 62 canales a 1000 Hz).
# Se fija por variable de entorno del PROCESO: MNE la consulta antes que su archivo
# de configuración global, así que esto es reproducible en cualquier máquina y no
# depende de que alguien haya corrido `mne.set_config` a mano.
RAW_ROOT = Path(os.environ.get("MNE_DATA", r"D:/Tesis/data/mne_data"))
RAW_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["MNE_DATA"] = str(RAW_ROOT)


def _parchear_sanitize_path_moabb():
    """WORKAROUND de un BUG de MOABB 1.5.0 que rompe cualquier ruta absoluta en Windows.

    `moabb.datasets.download._sanitize_path` sustituye los caracteres ilegales en
    nombres de archivo de Windows (`: * ? " < > |`) por '-', pero lo aplica a la RUTA
    ENTERA, incluida la letra de unidad:

        D:\\Tesis\\data\\mne_data   ->   D-\\Tesis\\data\\mne_data

    El resultado deja de ser absoluto y se resuelve desde el directorio de trabajo,
    así que los datos acaban dentro del repo en una carpeta llamada 'D-'.
    (Verificado con MOABB 1.5.0 el 2026-07-20: se descargaron 1,2 GB ahí.)

    El parche preserva el ancla de la unidad y sanea solo el resto.
    """
    import moabb.datasets.download as _dl

    if getattr(_dl._sanitize_path, "_parcheado", False):
        return

    _TABLA = {ord(c): "-" for c in ':*?"<>|'}

    def _sanitize_path_seguro(path):
        p = Path(path)
        ancla = p.anchor                       # 'D:\\'  (o '' si es relativa)
        resto = str(p)[len(ancla):]
        return Path(ancla) / resto.translate(_TABLA) if ancla else Path(resto.translate(_TABLA))

    _sanitize_path_seguro._parcheado = True
    _dl._sanitize_path = _sanitize_path_seguro

# --------------------------------------------------------------------------- #
# Interfaz del contrato (ver src/datasets/__init__.py)
# --------------------------------------------------------------------------- #
NAME = "openbmi"
CLASS_ORDER = ["left_hand", "right_hand"]      # orden fijo -> índices 0 y 1
N_CLASSES = len(CLASS_ORDER)
CHANCE = 1.0 / N_CLASSES                        # 0.50
# 200 Hz igual que MIND: lo exige LaBraM (verificado en src/adapters.py) y mantiene
# comparables los dos datasets. Los autores usan 100 Hz solo para su análisis CSP.
SFREQ = 200.0
WINDOW = (0.0, 4.0)                             # ventana de modelado (recorte del caché)
SESSIONS = ["session_1", "session_2"]           # 2 días distintos -> eje temporal
CACHE_DIR = Path(os.environ.get("OPENBMI_PROC_ROOT",
                                r"D:/Tesis/data/processed/openbmi")) / "epochs_cache"

# --------------------------------------------------------------------------- #
# Configuración de referencia de los AUTORES (solo para la puerta de validación)
# --------------------------------------------------------------------------- #
# No es la configuración del estudio: es la que hay que usar para reproducir sus
# números y confirmar que el cargador y los splits son correctos.
AUTHORS_CFG = {
    "l_freq": 8.0, "h_freq": 30.0,      # Butterworth orden 5 en el paper
    "resample_hz": 100,
    "window": (1.0, 3.5),                # 1000-3500 ms tras el estímulo
    "channels": [                        # 20 electrodos motores
        "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    ],
}
AUTHORS_REF = {          # objetivo de la puerta de validación (CSP+LDA, 10x10-fold CV)
    "session_1": 0.701,
    "session_2": 0.722,
    "promedio": 0.711,
}

# --------------------------------------------------------------------------- #
# Referencias CROSS-SUBJECT (de terceros: el paper del dataset NO reporta ninguna)
# --------------------------------------------------------------------------- #
# ⚠️ Sus protocolos NO son exactamente el nuestro -> son orientativas, no objetivos.
#    Con LOSO entrenamos con 53 sujetos (más que los ~32 de ASPEN), así que si acaso
#    deberíamos quedar algo por encima.
#
# Las DESVIACIONES no son comparables entre sí y esa diferencia es informativa:
#   ±13.10 de IMH-Net  -> dispersión ENTRE SUJETOS (refleja el 53.7 % de analfabetismo)
#   ±1.98  de ASPEN    -> entre corridas sobre una cohorte fija
# Es la distinción inter-sujeto (el fenómeno) vs inter-semilla (ruido) del HANDOFF §3.bis.
CROSS_SUBJECT_REF = {
    # ASPEN, arXiv 2602.16147, Tabla 2. Split 60/20/20 de SUJETOS (~32/11/11).
    # VERIFICADO leyendo el paper. Da EEGNet y EEGConformer, nuestras dos primeras
    # arquitecturas, y muestra rango real entre modelos (71.40 - 76.27) -> hay
    # estructura que discriminar, a diferencia de MIND.
    "aspen_2602.16147": {
        "protocolo": "split 60/20/20 de sujetos",
        "ASPEN": 0.7627, "EEGNet": 0.7588, "CTNet": 0.7498,
        "MultiDiffNet": 0.7467, "EEGConformer": 0.7455,
        "TSformer-SA": 0.7140, "SPEN": 0.5350,
    },
    # IMH-Net, Comput Methods Biomech Biomed Engin 2023, DOI 10.1080/10255842.2023.2275244
    # LOSO estricto (53 entrenan, 1 prueba). NO VERIFICADO directamente: el artículo
    # está de pago (403); la cifra viene de un resumen de búsqueda. Tratar con cautela.
    "imhnet_2023": {"protocolo": "LOSO (53 train / 1 test)", "IMH-Net": 0.7390,
                    "std_entre_sujetos": 0.1310, "verificado": False},
}
BCI_ILLITERACY_THRESHOLD = 0.70    # umbral del paper para "analfabetismo BCI"

# --------------------------------------------------------------------------- #
# Vistas de canales
# --------------------------------------------------------------------------- #
# ✅ VERIFICADO contra los datos reales (sujeto 1, 2026-07-20):
# 67 canales = 62 EEG + 4 EMG (EMG1-4) + 1 stim (STI 014), a 1000 Hz.
# Los 62 EEG son:
#   Fp1 Fp2 F7 F3 Fz F4 F8 FC5 FC1 FC2 FC6 T7 C3 Cz C4 T8 TP9 CP5 CP1 CP2 CP6 TP10
#   P7 P3 Pz P4 P8 PO9 O1 Oz O2 PO10 FC3 FC4 C5 C1 C2 C6 CP3 CPz CP4 P1 P2 POz
#   FT9 FTT9h TTP7h TP7 TPP9h FT10 FTT10h TPP8h TP8 TPP10h F9 F10 AF7 AF3 AF4 AF8 PO3 PO4
# ⚠️ NO existe FCz en este montaje (sí CPz y Fz). Por eso `motor_ext` NO puede copiarse
# tal cual del de MIND: allí incluía FCz.
CH_SETS = {
    "core_motor": ["C3", "Cz", "C4"],
    # Los 20 del paper (VERIFICADO: los 20 existen) -> comparación directa con sus cifras.
    "motor_paper": list(AUTHORS_CFG["channels"]),
    # Equivalente anatómico al motor_ext de MIND, sin FCz (no existe aquí) -> 12 canales.
    "motor_ext": ["C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                  "FC3", "FC4", "CP3", "CPz", "CP4"],
    # Vista COMPARTIDA con el otro dataset (punto 6: validacion cruzada).
    # Interseccion real de los dos montajes, calculada el 2026-09-06: 45 de los
    # 62 de OpenBMI y de los 59 de WBCIC-SHU. El ORDEN es el de OpenBMI y tiene
    # que ser identico en ambos ficheros: la primera capa de las cuatro
    # arquitecturas es espacial, asi que la posicion de cada canal es parte del
    # modelo. Fuera quedan los 17 de OpenBMI sin equivalente (TP9/TP10, PO9/PO10,
    # CPz, P1/P2, la serie FTT/TPP/TTP de alta densidad y F9/F10) y los 14 de
    # WBCIC sin equivalente (Fpz, F1/F2/F5/F6, FCz, FT7/FT8, P5/P6, PO5-PO8).
    "comun_wbcic": [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3",
        "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "O1",
        "Oz", "O2", "FC3", "FC4", "C5", "C1", "C2", "C6", "CP3", "CP4", "POz", "TP7",
        "TP8", "AF7", "AF3", "AF4", "AF8", "PO3", "PO4"
    ],
    # Subconjunto motor de la anterior. Es `motor_ext` de OpenBMI SIN CPz, que
    # no existe en WBCIC (alli hay FCz, que a su vez no existe en OpenBMI).
    "motor_comun_wbcic": [
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "FC3", "FC4", "CP3", "CP4"
    ],
    "completo": None,      # los 62 EEG (EMG y stim se excluyen al cachear)
}
NON_EEG_CH = ["EMG1", "EMG2", "EMG3", "EMG4", "STI 014"]

# --------------------------------------------------------------------------- #
# Carga vía MOABB
# --------------------------------------------------------------------------- #
MOABB_DATASET = "Lee2019_MI"

# Estructura real (VERIFICADA en el sujeto 1, 2026-07-20):
#   sesión '0' (= día 1) -> runs '1train' (100 trials) + '4test' (100 trials)
#   sesión '1' (= día 2) -> runs '1train' (100 trials) + '4test' (100 trials)
#   => 400 trials por sujeto, 200 por clase, balanceados 50/50 en cada run.
# Los runs '1train'/'4test' son las fases OFFLINE (sin feedback) y ONLINE (con feedback)
# del paper. No confundir ese eje con el de sesiones: son cosas distintas.
SESSION_KEYS = {"0": "session_1", "1": "session_2"}
RUN_KEYS = {"1train": "offline", "4test": "online"}


def make_dataset(test_run: bool = True):
    """Construye el objeto Lee2019_MI de MOABB con DOS BUGS DE LA 1.5.0 ya sorteados.

    BUG 1 — `test_run=None` por defecto: solo devuelve la fase offline (100 trials
    por sesión en vez de 200). Se fuerza `test_run=True` para tener las 400 del sujeto.

    BUG 2 — OFF-BY-ONE QUE PIERDE LA MITAD DEL DATASET EN SILENCIO:
        `_get_single_subject_data` nombra las sesiones `str(session - 1)` -> '0', '1'
        pero `get_data` las filtra con `{str(s) for s in self._selected_sessions}` -> {'1','2'}
        Como '0' no está en el filtro, **la sesión 1 se descarta sin ningún aviso** y solo
        sobrevive la 2. Verificado: `_get_single_subject_data` devuelve {'0','1'} y
        `get_data` devuelve {'1'}.
        Se desactiva el filtro (`_selected_sessions = None`), que es lo único que hace
        y está mal. No se puede arreglar pasando `sessions=(0,1)`: Lee2019 valida que
        estén en [1, 2] y lanza ValueError.
    """
    try:
        from moabb.datasets import Lee2019_MI
    except ImportError as e:
        raise ImportError("MOABB no está instalado. Ejecutar:  pip install moabb") from e
    _parchear_sanitize_path_moabb()
    ds = Lee2019_MI(test_run=test_run)
    ds._selected_sessions = None
    return ds


def load_raw(subject: int, session=None):
    """Devuelve los datos crudos de un sujeto vía MOABB: {sesión: {run: Raw}}.

    Se mantiene deliberadamente fino: el preprocesamiento vive en
    `scripts/openbmi/build_cache.py`, igual que en MIND, para que el caché sea el
    único punto de entrada del resto del pipeline.
    """
    datos = make_dataset().get_data(subjects=[subject])[subject]
    return datos if session is None else datos[session]
