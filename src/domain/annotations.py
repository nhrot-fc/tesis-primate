import logging
from pathlib import Path

import pandas as pd
from slugify import slugify

from domain.species import CALL_TYPES, VALID_PAIRS, Species

logger = logging.getLogger(__name__)

NOISE = "noise"
MAX_FREQ_HZ = 22050.0
MIN_DURATION_S = 0.01

DROP_COLUMNS = ["selection", "view", "channel", "reference", "begin_file", "file_offset_s"]
MANUAL_SYNONYMS = {
    "noises": NOISE,
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


def clean_annotations(df: pd.DataFrame, species_dir: str) -> pd.DataFrame:
    """Normaliza columnas, códigos de llamada y rangos de un `.txt` de Raven."""
    df = df.copy()
    df.columns = [slugify(col, separator="_") for col in df.columns]
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    species = species_dir.split("__")[-1].lower()
    species_enum = next((s for s in Species if s.name.lower() == species), None)
    label_synonyms = (
        {label: code for code, label in CALL_TYPES[species_enum].items()} if species_enum else {}
    )
    df["call_type"] = (
        df["call_type"]
        .map(lambda v: slugify(v, separator="_") or None if isinstance(v, str) else None)
        .replace(MANUAL_SYNONYMS | label_synonyms)
    )
    df["species"] = species
    df = df[df["call_type"].notna() & df["call_type"].ne(NOISE)]

    for (bad_sp, bad_ct), (sp, ct) in MANUAL_FIXES.items():
        wrong = df["species"].eq(bad_sp) & df["call_type"].eq(bad_ct)
        df.loc[wrong, ["species", "call_type"]] = [sp, ct]

    pairs = pd.Series(list(zip(df["species"], df["call_type"], strict=True)), index=df.index)
    df["requires_review"] = ~pairs.isin(VALID_PAIRS)

    df["duration_s"] = df["end_time_s"] - df["begin_time_s"]
    df["high_freq_hz"] = df["high_freq_hz"].clip(upper=MAX_FREQ_HZ)
    df["bandwidth_hz"] = df["high_freq_hz"] - df["low_freq_hz"]

    df = df[(df["duration_s"] >= MIN_DURATION_S) & (df["bandwidth_hz"] > 0)]
    return df.sort_values("begin_time_s").reset_index(drop=True)


def load_annotations(root: Path) -> pd.DataFrame:
    frames = []
    for annotation_path in sorted(Path(root).rglob("*.txt")):
        wav_path = annotation_path.with_suffix(".wav")
        if not wav_path.exists():
            continue
        try:
            species_dir = wav_path.parent.name
            for parent in wav_path.parents:
                if "__" in parent.name:
                    species_dir = parent.name
                    break

            frame = pd.read_csv(annotation_path, sep="\t")
            frame["audio_path"] = str(wav_path)
            frames.append(clean_annotations(frame, species_dir))
        except Exception as exc:
            logger.warning("%s: %s", annotation_path.name, exc)
    return pd.concat(frames, ignore_index=True)
