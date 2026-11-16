"""Splits LOSO con normalización por fold y disciplina anti-fuga.

NÚCLEO METODOLÓGICO DE LA TESIS. Si aquí hay fuga, todos los resultados quedan invalidados.

Esquema (plan §B.2, validación anidada):

    Para cada sujeto S_i (fold externo):
        TEST      = S_i                      # intocable, se evalúa UNA vez
        TRAINVAL  = los otros 29
            INNER_VAL   = n_inner_val sujetos de TRAINVAL   # early stopping / selección
            INNER_TRAIN = el resto
        - z-score: media y desviación calculadas SOLO con INNER_TRAIN
        - data augmentation: SOLO en INNER_TRAIN

Reglas que este módulo garantiza:
  1. El sujeto de test NUNCA aparece en train ni en validación.
  2. Los estadísticos de normalización se calculan SOLO con inner-train.
  3. Todo es determinista dada una semilla.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

def chance_level(n_classes: int) -> float:
    """Azar para n clases balanceadas. Se calcula, no se fija: cambia con el dataset
    (MIND 4 clases -> 0.25; OpenBMI/WBCIC 2 clases -> 0.50)."""
    return 1.0 / n_classes


@dataclass
class Fold:
    """Un fold LOSO: qué sujetos van a cada partición."""
    idx: int
    test: str
    inner_val: list[str]
    inner_train: list[str]

    @property
    def trainval(self) -> list[str]:
        return list(self.inner_train) + list(self.inner_val)

    def assert_disjoint(self):
        """Comprueba que las tres particiones son disjuntas por sujeto."""
        s_te, s_va, s_tr = {self.test}, set(self.inner_val), set(self.inner_train)
        assert not (s_te & s_va), f"fold {self.idx}: test también en inner_val"
        assert not (s_te & s_tr), f"fold {self.idx}: test también en inner_train"
        assert not (s_va & s_tr), f"fold {self.idx}: inner_val solapa con inner_train"
        return True


def make_loso_folds(subjects, n_inner_val: int = 2, seed: int = 2026) -> list[Fold]:
    """Construye los N folds LOSO (uno por sujeto), con validación interna anidada.

    Los sujetos de inner_val se eligen de forma determinista (barajado con la semilla)
    y rotando, para que ningún sujeto quede siempre como validación.
    """
    subjects = sorted(subjects)
    n = len(subjects)
    assert n >= n_inner_val + 2, "hacen falta más sujetos que inner_val + test + train"
    rng = np.random.default_rng(seed)
    orden = list(rng.permutation(subjects))

    folds = []
    for i, test in enumerate(subjects):
        resto = [s for s in orden if s != test]
        # rotación determinista: cada fold toma un bloque distinto como validación
        start = (i * n_inner_val) % len(resto)
        sel = [resto[(start + k) % len(resto)] for k in range(n_inner_val)]
        inner_train = [s for s in resto if s not in sel]
        f = Fold(idx=i, test=test, inner_val=sel, inner_train=inner_train)
        f.assert_disjoint()
        folds.append(f)
    return folds


# --------------------------------------------------------------------------- #
# Carga desde el caché de tensores
# --------------------------------------------------------------------------- #

def load_subject(sub: str, cache_dir, window=None, ch_names=None):
    """Carga el tensor cacheado de un sujeto.

    window : (tmin, tmax) en segundos -> recorta la ventana de modelado.
    ch_names : lista de canales a conservar (vista); None = todos.
    Devuelve (X, y, ch_names_final).
    """
    d = np.load(Path(cache_dir) / f"{sub}_full.npz", allow_pickle=True)
    X, y = d["X"], d["y"]
    chs = [str(c) for c in d["ch_names"]]
    sfreq, tmin = float(d["sfreq"]), float(d["tmin"])

    if window is not None:
        a, b = window
        i0 = int(round((a - tmin) * sfreq))
        i1 = int(round((b - tmin) * sfreq)) + 1
        assert 0 <= i0 < i1 <= X.shape[2], f"ventana {window} fuera del rango cacheado"
        X = X[:, :, i0:i1]
    if ch_names is not None:
        up = {c.upper(): k for k, c in enumerate(chs)}
        keep = [up[c.upper()] for c in ch_names if c.upper() in up]
        X = X[:, keep, :]
        chs = [chs[k] for k in keep]
    return X, y, chs


def load_group(subs, cache_dir, window=None, ch_names=None):
    """Concatena varios sujetos. Devuelve (X, y, subject_ids, ch_names)."""
    Xs, ys, sid = [], [], []
    chs_ref = None
    for s in subs:
        X, y, chs = load_subject(s, cache_dir, window, ch_names)
        if chs_ref is None:
            chs_ref = chs
        else:
            assert chs == chs_ref, f"{s}: los canales no coinciden con los del resto"
        Xs.append(X); ys.append(y); sid += [s] * len(y)
    return np.concatenate(Xs), np.concatenate(ys), np.array(sid), chs_ref


# --------------------------------------------------------------------------- #
# Normalización por fold (ANTI-FUGA)
# --------------------------------------------------------------------------- #

def fit_normalizer(X_train):
    """Calcula media y desviación POR CANAL usando SOLO datos de entrenamiento.

    X_train: (n_trials, n_canales, n_tiempo). Devuelve dict con arrays (n_canales, 1).
    """
    mean = X_train.mean(axis=(0, 2), keepdims=True)[0]     # (n_ch, 1)
    std = X_train.std(axis=(0, 2), keepdims=True)[0]
    std = np.where(std < 1e-12, 1.0, std)                   # evita división por cero
    return {"mean": mean, "std": std}


def apply_normalizer(X, stats):
    """Aplica z-score por canal con estadísticos ya ajustados (de train)."""
    return (X - stats["mean"][None]) / stats["std"][None]


# --- Estrategias de normalización (se comparan empíricamente; ver Actividad 3.3) ---
#
# "global"      z-score por canal con estadísticos SOLO del inner-train (anti-fuga estricto).
#               Deja intactas las diferencias de escala ENTRE sujetos: el sujeto de test
#               puede llegar con std 0.51 frente al 1.0 del train (= el cambio de dominio).
# "per_subject" z-score por canal con los estadísticos de CADA sujeto sobre sus propios
#               datos. Es el estándar del campo en inter-sujeto. NO es fuga de etiquetas
#               (no mira las clases), pero SÍ es TRANSDUCTIVO: asume disponer de los datos
#               del sujeto objetivo antes de clasificar. DEBE declararse en la memoria.
# "per_trial"   cada trial se escala por su propia media/desviación GLOBAL (agregando todos
#               los canales). No es transductivo en absoluto: ningún trial mira a otro.
#               Preserva el patrón espacial relativo entre canales, que es justo la
#               información que explotan CSP/EEGNet — por eso NO se normaliza canal a canal.
NORM_STRATEGIES = ("global", "per_subject", "per_trial", None)


def normalize_per_subject(X, subject_ids):
    """Z-score por canal, ajustado independientemente para cada sujeto."""
    X = np.asarray(X)
    subject_ids = np.asarray(subject_ids)
    out = np.empty_like(X)
    for s in np.unique(subject_ids):
        m = subject_ids == s
        out[m] = apply_normalizer(X[m], fit_normalizer(X[m]))
    return out


def normalize_per_trial(X):
    """Escala cada trial por su propia media/desviación global (todos los canales juntos)."""
    X = np.asarray(X)
    mean = X.mean(axis=(1, 2), keepdims=True)
    std = X.std(axis=(1, 2), keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (X - mean) / std


def _resolve_norm(normalize):
    """Acepta el bool histórico (True == 'global') además del nombre de la estrategia."""
    if normalize is True:
        return "global"
    if normalize is False:
        return None
    if normalize not in NORM_STRATEGIES:
        raise ValueError(f"normalize debe ser uno de {NORM_STRATEGIES}, llegó {normalize!r}")
    return normalize


def cargar_sesiones(subs, cache_dir):
    """Etiqueta de sesión de cada trial, concatenada en el mismo orden que `load_group`."""
    out = []
    for s in subs:
        d = np.load(Path(cache_dir) / f"{s}_full.npz", allow_pickle=True)
        out.append(np.asarray(d["session"]).astype(str))
    return np.concatenate(out) if out else np.array([])


def separar_visto(y, sub_ids, sesiones, frac: float, seed: int = 2026):
    """Reserva una fracción de los trials de CADA sujeto de entrenamiento.

    Es el conjunto "SUJETO VISTO" del indicador de cambio de dominio: se evalúa el mismo
    modelo sobre gente que estaba en el entrenamiento (sin cambio de dominio) y sobre el
    sujeto excluido (con cambio de dominio). Al ser el MISMO modelo entrenado con los
    MISMOS datos, la única diferencia entre ambas medidas es si la persona era conocida
    — que es justo lo que se quiere aislar. La formulación clásica (within vs LOSO)
    confunde ese efecto con el del tamaño del conjunto de entrenamiento, porque el
    modelo LOSO ve ~200 veces más datos que el within.

    Estratificado por (sujeto, clase, sesión): el conjunto reservado debe estar tan
    balanceado como el del sujeto no visto, o se estaría mezclando cambio TEMPORAL
    (día 1 vs día 2) con cambio INTER-SUJETO.

    Devuelve una máscara booleana: True = va al conjunto "visto", False = se entrena.
    """
    rng = np.random.default_rng(seed)
    visto = np.zeros(len(y), dtype=bool)
    if frac <= 0:
        return visto
    for s in np.unique(sub_ids):
        for c in np.unique(y):
            for ses in np.unique(sesiones):
                idx = np.flatnonzero((sub_ids == s) & (y == c) & (sesiones == ses))
                if len(idx) == 0:
                    continue
                n = max(1, int(round(frac * len(idx))))
                visto[rng.choice(idx, size=min(n, len(idx)), replace=False)] = True
    return visto


def build_fold(fold: Fold, cache_dir, window=None, ch_names=None, normalize="global",
               frac_visto: float = 0.0, seed: int = 2026):
    """Materializa un fold completo, normalizado sin fuga.

    normalize: ver NORM_STRATEGIES. True equivale a "global" (compatibilidad histórica).
    frac_visto: fracción de trials de cada sujeto de ENTRENAMIENTO que se reserva como
        conjunto "sujeto visto" (ver `separar_visto`). 0.0 = comportamiento clásico.
        Esos trials NO entrenan y NO entran en los estadísticos de normalización.
    Devuelve dict con X/y de inner_train, inner_val, test y —si frac_visto>0— visto.
    """
    fold.assert_disjoint()
    norm = _resolve_norm(normalize)
    Xtr, ytr, str_, chs = load_group(fold.inner_train, cache_dir, window, ch_names)

    Xvi = yvi = svi = None
    if frac_visto > 0:
        ses_tr = cargar_sesiones(fold.inner_train, cache_dir)
        m = separar_visto(ytr, str_, ses_tr, frac_visto, seed)
        Xvi, yvi, svi = Xtr[m], ytr[m], str_[m]
        # Se sacan del entrenamiento ANTES de ajustar la normalización: si contribuyeran
        # a la media y la desviación serían datos de test filtrados en el preprocesado.
        Xtr, ytr, str_ = Xtr[~m], ytr[~m], str_[~m]
    Xva, yva, sva, _ = load_group(fold.inner_val, cache_dir, window, ch_names)
    Xte, yte, ste, _ = load_group([fold.test], cache_dir, window, ch_names)

    stats = None
    if norm == "global":
        stats = fit_normalizer(Xtr)          # <- SOLO inner_train (ya sin el "visto")
        Xtr = apply_normalizer(Xtr, stats)
        Xva = apply_normalizer(Xva, stats)   # se aplican los MISMOS estadísticos
        Xte = apply_normalizer(Xte, stats)
        if Xvi is not None:
            Xvi = apply_normalizer(Xvi, stats)
    elif norm == "per_subject":
        Xtr = normalize_per_subject(Xtr, str_)
        Xva = normalize_per_subject(Xva, sva)
        Xte = normalize_per_subject(Xte, ste)
        if Xvi is not None:
            Xvi = normalize_per_subject(Xvi, svi)
    elif norm == "per_trial":
        Xtr, Xva, Xte = normalize_per_trial(Xtr), normalize_per_trial(Xva), normalize_per_trial(Xte)
        if Xvi is not None:
            Xvi = normalize_per_trial(Xvi)

    out = {
        "fold": fold, "ch_names": chs, "stats": stats, "norm": norm,
        "X_train": Xtr, "y_train": ytr, "sub_train": str_,
        "X_val": Xva, "y_val": yva, "sub_val": sva,
        "X_test": Xte, "y_test": yte, "sub_test": ste,
    }
    if Xvi is not None:
        out.update({"X_visto": Xvi, "y_visto": yvi, "sub_visto": svi})
    return out


# --------------------------------------------------------------------------- #
# Métricas de cambio de dominio
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Folds within-subject (Anexo B.1: 5-fold CV por sujeto, anidado como LOSO)
# --------------------------------------------------------------------------- #

@dataclass
class WithinFold:
    """Un fold within-subject: qué ÍNDICES DE TRIAL (de un único sujeto) van a cada partición.

    `session` es None cuando el fold mezcla todas las sesiones (paradigma de sesión única
    tipo MIND) y lleva la etiqueta de la sesión cuando el fold es INTRA-SESIÓN (OpenBMI).
    """
    idx: int
    test_idx: np.ndarray
    inner_val_idx: np.ndarray
    inner_train_idx: np.ndarray
    session: str | None = None

    def assert_disjoint(self):
        s_te, s_va, s_tr = set(self.test_idx.tolist()), set(self.inner_val_idx.tolist()), set(self.inner_train_idx.tolist())
        assert not (s_te & s_va), f"within-fold {self.idx}: test también en inner_val"
        assert not (s_te & s_tr), f"within-fold {self.idx}: test también en inner_train"
        assert not (s_va & s_tr), f"within-fold {self.idx}: inner_val solapa con inner_train"
        return True


def make_within_subject_folds(y, n_folds: int = 5, n_inner_val_frac: float = 0.2,
                               seed: int = 2026) -> list[WithinFold]:
    """5-fold CV estratificado sobre los trials de UN sujeto (plan Anexo B.1).

    Por cada uno de los `n_folds` folds externos (test), el resto de trials se
    reparte de forma estratificada y determinista en inner_train/inner_val,
    para que el early stopping nunca vea el fold de test — misma disciplina
    anidada que `make_loso_folds`, aplicada aquí a nivel de trial.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split

    y = np.asarray(y)
    idx_all = np.arange(len(y))
    outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for i, (trainval_idx, test_idx) in enumerate(outer.split(idx_all, y)):
        inner_train_idx, inner_val_idx = train_test_split(
            trainval_idx, test_size=n_inner_val_frac, random_state=seed + i,
            stratify=y[trainval_idx],
        )
        f = WithinFold(idx=i, test_idx=test_idx, inner_val_idx=inner_val_idx,
                        inner_train_idx=inner_train_idx)
        f.assert_disjoint()
        folds.append(f)
    return folds


def make_within_session_folds(y, session, n_folds: int = 10, n_inner_val_frac: float = 0.2,
                               seed: int = 2026) -> list[WithinFold]:
    """CV within-subject PERO SIN MEZCLAR SESIONES (protocolo OpenBMI, §10 del HANDOFF).

    En OpenBMI la sesión es un EJE DE CAMBIO DE DOMINIO (día 1 vs día 2): mezclar sus
    trials en un mismo fold metería señal de un día en el train del otro e inflaría el
    techo within-subject. Aquí se hace `n_folds`-fold estratificado DENTRO de cada sesión
    por separado, así que ningún fold cruza el eje temporal.

    `session`: array paralelo a `y` con la etiqueta de sesión de cada trial (p. ej. '0'/'1').
    Devuelve una lista de WithinFold con ÍNDICES ABSOLUTOS al tensor del sujeto y el campo
    `.session` relleno. Resultado: `n_folds` folds por cada sesión -> dos métricas por
    sujeto (una por sesión), que se reportan por separado.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split

    y = np.asarray(y)
    session = np.asarray(session)
    folds, fidx = [], 0
    for ses in sorted(np.unique(session).tolist()):
        abs_idx = np.where(session == ses)[0]        # posiciones absolutas de esta sesión
        y_ses = y[abs_idx]
        outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for j, (tv, te) in enumerate(outer.split(abs_idx, y_ses)):   # tv/te indexan abs_idx
            inner_tr, inner_va = train_test_split(
                tv, test_size=n_inner_val_frac, random_state=seed + j, stratify=y_ses[tv])
            f = WithinFold(idx=fidx, session=str(ses),
                           test_idx=abs_idx[te],
                           inner_val_idx=abs_idx[inner_va],
                           inner_train_idx=abs_idx[inner_tr])
            f.assert_disjoint()
            folds.append(f)
            fidx += 1
    return folds


def build_within_fold(fold: WithinFold, X, y, normalize="global"):
    """Materializa un fold within-subject a partir del tensor YA CARGADO de un sujeto.

    X: (n_trials, n_canales, n_tiempo) de UN sujeto (p. ej. de `load_subject`).
    Normaliza igual que `build_fold`: estadísticos SOLO de inner_train.

    NOTA: con un único sujeto, "per_subject" usaría los estadísticos de ese mismo sujeto
    también para el test, lo que aquí SÍ sería fuga entre particiones. Por eso se trata
    como "global" (estadísticos solo de inner-train), que es el equivalente correcto.
    """
    fold.assert_disjoint()
    norm = _resolve_norm(normalize)
    if norm == "per_subject":
        norm = "global"
    Xtr, ytr = X[fold.inner_train_idx], y[fold.inner_train_idx]
    Xva, yva = X[fold.inner_val_idx], y[fold.inner_val_idx]
    Xte, yte = X[fold.test_idx], y[fold.test_idx]

    stats = None
    if norm == "global":
        stats = fit_normalizer(Xtr)
        Xtr = apply_normalizer(Xtr, stats)
        Xva = apply_normalizer(Xva, stats)
        Xte = apply_normalizer(Xte, stats)
    elif norm == "per_trial":
        Xtr, Xva, Xte = normalize_per_trial(Xtr), normalize_per_trial(Xva), normalize_per_trial(Xte)

    return {
        "fold": fold, "stats": stats, "norm": norm,
        "X_train": Xtr, "y_train": ytr,
        "X_val": Xva, "y_val": yva,
        "X_test": Xte, "y_test": yte,
    }


def domain_shift_index(within, loso):
    """DSI = (within - loso) / within. Menor = más robusto al cambio de dominio.

    IMPORTANTE: usar KAPPA de Cohen, no accuracy. Con 4 clases y desempeños de 0.30-0.37
    sobre un azar de 0.25, el DSI sobre accuracy es inestable (diferencias absolutas
    minúsculas producen DSI grandes). Kappa ya está corregido por azar (0 = azar).
    """
    within = np.asarray(within, dtype=float)
    loso = np.asarray(loso, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dsi = (within - loso) / within
    return np.where(np.abs(within) < 1e-9, np.nan, dsi)


# --------------------------------------------------------------------------- #
# Entrenamiento sobre TODOS los sujetos (modelo de transferencia)
# --------------------------------------------------------------------------- #

# Semilla del reparto reservado. Es una CONSTANTE y no la semilla de entrenamiento:
# el conjunto reservado tiene que ser el mismo para el modelo transferido y para el
# nativo, y para todas las arquitecturas y semillas, o dejan de compararse sobre los
# mismos datos.
SEMILLA_RESERVA = 2026


def mascara_reservada(subs, cache_dir, frac_test: float, seed: int = SEMILLA_RESERVA):
    """Máscara booleana del conjunto RESERVADO, reconstruible desde fuera.

    Existe para que `scripts/eval_cruzado.py` evalúe el modelo transferido exactamente
    sobre los segmentos que el modelo nativo del dataset destino NO usó para entrenar.
    No se guarda en disco porque no hace falta: `separar_visto` es determinista dados
    los datos, la fracción y la semilla, así que recalcularla da el mismo resultado.

    El orden de los segmentos es el de `load_group(subs, ...)`, así que la máscara solo
    es válida si se aplica sobre esa misma concatenación y con la misma lista `subs`.
    """
    _, y, sid, _ = load_group(subs, cache_dir)
    ses = cargar_sesiones(subs, cache_dir)
    return separar_visto(y, sid, ses, frac_test, seed)


def build_all(subs, cache_dir, window=None, ch_names=None, normalize="global",
              frac_val: float = 0.10, frac_test: float = 0.0, seed: int = 2026):
    """Materializa un único split con TODOS los sujetos.

    Dos usos, según `frac_test`:

    frac_test = 0
        Modelo de TRANSFERENCIA: entrena con todo salvo la validación, y su conjunto de
        prueba es otro dataset. No se reserva nada más a propósito, porque cualquier
        partición interna mediría generalización dentro del mismo dataset, que es lo que
        ya miden `within` y `loso`.

    frac_test > 0
        Modelo NATIVO del dataset destino, que sirve de techo de referencia. Reserva una
        porción que no entra ni en el entrenamiento ni en la parada temprana ni en los
        estadísticos de normalización, y que es la MISMA sobre la que se evalúa el modelo
        transferido. Así ambos se miden sobre los mismos segmentos y su diferencia es el
        costo del cambio de dominio, no un artefacto de evaluar cosas distintas.

    El reparto reutiliza `separar_visto`, que estratifica por (sujeto, clase, sesión): las
    particiones quedan balanceadas en las tres dimensiones y ningún sujeto ni ninguna
    sesión queda sin representación. El reservado se separa PRIMERO y con
    `SEMILLA_RESERVA`, no con `seed`, para que no dependa de la semilla de entrenamiento.

    Los estadísticos de normalización se ajustan SOLO sobre el conjunto de entrenamiento
    y se devuelven en `stats`, porque hay que aplicarlos tal cual al segundo dataset:
    reajustarlos allí convertiría la evaluación en transductiva sin declararlo.
    """
    norm = _resolve_norm(normalize)
    X, y, sid, chs = load_group(subs, cache_dir, window, ch_names)
    ses = cargar_sesiones(subs, cache_dir)

    Xte = yte = ste = None
    if frac_test > 0:
        mt = separar_visto(y, sid, ses, frac_test, SEMILLA_RESERVA)
        if not mt.any():
            raise ValueError(f"frac_test={frac_test} no reservó ningún trial")
        Xte, yte, ste = X[mt], y[mt], sid[mt]
        # Se saca ANTES de repartir train/val y antes de ajustar la normalización: si
        # contribuyera a la media y la desviación seria fuga en el preprocesado.
        X, y, sid, ses = X[~mt], y[~mt], sid[~mt], ses[~mt]

    m = separar_visto(y, sid, ses, frac_val, seed)
    if not m.any():
        raise ValueError(f"frac_val={frac_val} no reservó ningún trial para validación")
    Xva, yva, sva = X[m], y[m], sid[m]
    Xtr, ytr, str_ = X[~m], y[~m], sid[~m]

    stats = None
    if norm == "global":
        stats = fit_normalizer(Xtr)
        Xtr, Xva = apply_normalizer(Xtr, stats), apply_normalizer(Xva, stats)
        if Xte is not None:
            Xte = apply_normalizer(Xte, stats)
    elif norm == "per_subject":
        Xtr, Xva = normalize_per_subject(Xtr, str_), normalize_per_subject(Xva, sva)
        if Xte is not None:
            Xte = normalize_per_subject(Xte, ste)
    elif norm == "per_trial":
        Xtr, Xva = normalize_per_trial(Xtr), normalize_per_trial(Xva)
        if Xte is not None:
            Xte = normalize_per_trial(Xte)

    out = {
        "ch_names": chs, "stats": stats, "norm": norm, "subs": list(subs),
        "frac_test": frac_test, "semilla_reserva": SEMILLA_RESERVA,
        "X_train": Xtr, "y_train": ytr, "sub_train": str_,
        "X_val": Xva, "y_val": yva, "sub_val": sva,
    }
    if Xte is not None:
        out.update({"X_test": Xte, "y_test": yte, "sub_test": ste})
    return out


SEMILLA_HOLDOUT = 2026


def separar_sujetos(subs, n_holdout: int, seed: int = SEMILLA_HOLDOUT):
    """Aparta `n_holdout` SUJETOS enteros del entrenamiento, de forma determinista.

    Es una particion distinta de `separar_visto`, que reparte trials DENTRO de cada sujeto.
    Aqui el sujeto entero queda fuera, asi que el modelo nunca ve un solo segmento suyo.

    PARA QUE: entrenando con los 54 sujetos de OpenBMI no queda ninguno sin ver, de modo que
    ese modelo no puede producir una cifra de "persona nueva, mismo protocolo". Al apartar 10
    sujetos, UN MISMO modelo se evalua sobre tres conjuntos y la comparacion ya no mezcla
    regimenes de entrenamiento:
        1. su propia validacion            -> sujetos vistos, mismo protocolo
        2. los `n_holdout` apartados       -> persona nueva, mismo protocolo
        3. el otro dataset                 -> persona nueva Y protocolo nuevo

    La seleccion NO se guarda en disco: se recalcula con esta funcion dados `subs`, la
    cantidad y la semilla. Por eso `subs` se ordena antes de permutar, para no depender del
    orden en que el sistema de ficheros devuelva los sujetos.

    Devuelve (entrenamiento, apartados), ambos ordenados.
    """
    subs = sorted(subs)
    if not 0 < n_holdout < len(subs):
        raise ValueError(f"n_holdout={n_holdout} fuera de rango para {len(subs)} sujetos")
    orden = np.random.default_rng(seed).permutation(len(subs))
    fuera = {subs[i] for i in orden[:n_holdout]}
    return [s for s in subs if s not in fuera], sorted(fuera)
