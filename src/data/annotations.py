import hashlib
import logging
from pathlib import Path

import pandas as pd

from core.config import CLEANED_DIR, RAW_DIR, P
from data.raven import BOX_COLUMNS, CLEANED_BOX_COLUMNS
from data.species import CALL_TYPES, VALID_PAIRS

logger = logging.getLogger(__name__)

NOISE = "noise"
# Por encima del tope del mel la anotación no se ve: se recorta ahí
MAX_FREQ_HZ = P.f_max
# Duración mínima de una anotación; el manifest y el jitter derivan de acá su caja mínima
MIN_DURATION_S = 0.01
# Bytes que identifican un wav junto con su tamaño: alcanza para reconocer copias del mismo
SIGNATURE_BYTES = 1 << 20

DROP_COLUMNS = ["selection", "view", "channel", "reference", "begin_file", "file_offset_s"]
MANUAL_SYNONYMS = {
    "noises": NOISE,
    "avevoc": "voc",  # PteroSet: `ID` es siempre AVEVOC
    "cs_a": "cs",
    "whinnie": "whc",
    "tca": "ta",
    "tac": "ta",
    "contact": "cc",
    "chcj": "chc",
    "php": "phc",
    "sqr": "sqc",
    "tc": "tr",
}
MANUAL_FIXES: dict[tuple[str, str], tuple[str, str]] = {("aa", "hc"): ("aa", "hm")}

# Las tablas mezclan el código y el nombre legible del tipo de llamada.
CALL_SYNONYMS: dict[str, dict[str, str]] = {
    species.name.lower(): {name: code for code, name in codes.items()}
    for species, codes in CALL_TYPES.items()
}


def cleaned(column: str) -> str:
    # Nombre de la columna en `cleaned/`: el de Raven en snake_case.
    from slugify import slugify  # grupo `train`: sólo al preparar datos

    return slugify(column, separator="_")


def clean_annotations(df: pd.DataFrame, species: str) -> pd.DataFrame:
    from slugify import slugify

    assert [cleaned(c) for c in BOX_COLUMNS] == CLEANED_BOX_COLUMNS
    df = df.copy()
    df.columns = [cleaned(col) for col in df.columns]
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    if "call_type" not in df and "id" in df:
        # PteroSet: `Tipo`/`ID` en vez de `Species`/`Call type`
        df["call_type"] = df["id"]

    df["call_type"] = (
        df["call_type"]
        .map(lambda v: slugify(v, separator="_") or None if isinstance(v, str) else None)
        .replace(MANUAL_SYNONYMS | CALL_SYNONYMS.get(species, {}))
    )
    df["species"] = species
    df = df.loc[df["call_type"].notna() & df["call_type"].ne(NOISE)]

    for (bad_sp, bad_ct), (sp, ct) in MANUAL_FIXES.items():
        wrong = df["species"].eq(bad_sp) & df["call_type"].eq(bad_ct)
        df.loc[wrong, ["species", "call_type"]] = [sp, ct]

    pairs = pd.Series(list(zip(df["species"], df["call_type"], strict=True)), index=df.index)
    df["requires_review"] = ~pairs.isin(VALID_PAIRS)

    df["duration_s"] = df["end_time_s"] - df["begin_time_s"]
    df["high_freq_hz"] = df["high_freq_hz"].clip(upper=MAX_FREQ_HZ)
    df["bandwidth_hz"] = df["high_freq_hz"] - df["low_freq_hz"]
    df = df.loc[(df["duration_s"] >= MIN_DURATION_S) & (df["bandwidth_hz"] > 0)]
    return df.sort_values("begin_time_s").reset_index(drop=True)


def species_of(wav_path: Path) -> str:
    # Las carpetas se llaman `<nombre_comun>__<CODIGO>`; vale el código.
    for parent in wav_path.parents:
        if "__" in parent.name:
            return parent.name.split("__")[-1].lower()
    return wav_path.parent.name.lower()


def audio_signature(path: Path) -> tuple[int, str]:
    with open(path, "rb") as audio:
        return path.stat().st_size, hashlib.md5(audio.read(SIGNATURE_BYTES)).hexdigest()


def unify_copies(annotations: pd.DataFrame) -> pd.DataFrame:
    # El mismo wav está copiado en varias carpetas de especie y cada copia anotada sólo para la
    # suya. Todas pasan a ser una grabación con la unión de sus anotaciones: si no, el resto de
    # especies queda como fondo en cada copia y el mismo audio puede caer en dos splits.
    canonical: dict[str, str] = {}
    by_signature: dict[tuple[int, str], str] = {}
    for path in sorted(annotations["audio_path"].unique()):
        canonical[path] = by_signature.setdefault(audio_signature(Path(path)), path)
    n_copies = len(canonical) - len(by_signature)
    if n_copies:
        logger.info("%d copias de grabaciones unidas con su original", n_copies)
    unified = annotations.assign(audio_path=annotations["audio_path"].map(canonical))
    # Dos copias anotadas para la misma especie repetirían sus cajas.
    return unified.drop_duplicates().reset_index(drop=True)


def load_annotations(root: Path = CLEANED_DIR, audio_root: Path = RAW_DIR) -> pd.DataFrame:
    # Cada `.txt` de cleaned/ se llama como su `.wav` en raw/.
    frames: list[pd.DataFrame] = []
    without_audio: list[str] = []
    for annotation_path in sorted(root.rglob("*.txt")):
        relative = annotation_path.relative_to(root)
        audio_path = (audio_root / relative).with_suffix(".wav")
        if not audio_path.is_file():
            without_audio.append(str(relative))
            continue
        frame = pd.read_csv(annotation_path, sep="\t")
        if frame.empty:  # una grabación sin anotaciones; en el concat volvería object las columnas
            continue
        frame["audio_path"] = str(audio_path)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"no hay anotaciones en {root}; corré `python src/prepare_annotations.py`"
        )
    if without_audio:
        logger.warning(
            "%d grabaciones sin .wav en %s, quedan fuera:%s",
            len(without_audio),
            audio_root,
            "".join(f"\n  {line}" for line in without_audio),
        )

    annotations_df = unify_copies(pd.concat(frames, ignore_index=True))
    annotations_df[CLEANED_BOX_COLUMNS] = annotations_df[CLEANED_BOX_COLUMNS].astype(float)
    annotations_df["duration_s"] = annotations_df["end_time_s"] - annotations_df["begin_time_s"]
    annotations_df["bandwidth_hz"] = annotations_df["high_freq_hz"] - annotations_df["low_freq_hz"]
    return annotations_df
