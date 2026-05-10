from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pandas as pd

from domain.pipelines.types import Annotation
from domain.utils.text_normalization import normalize_headers, normalize_value

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = ["species", "call_type", "begin_time", "end_time", "low_freq", "high_freq"]
NUMERIC_COLUMNS = ["begin_time", "end_time", "low_freq", "high_freq"]

CALL_TYPE_CORRECTIONS = {"csa": "cs", "contactcall": "cc", "contactsyllable": "cs"}
HEADER_RENAMES = {
    "begin_time_s": "begin_time",
    "end_time_s": "end_time",
    "low_freq_hz": "low_freq",
    "high_freq_hz": "high_freq",
    "inband_power_db_fs": "inband_power",
    "species": "species",
}
ANNOTATION_EXTENSIONS = (".csv", ".txt", ".tsv")

# ---------------------------------------------------------------------------
# IO & Error Handling
# ---------------------------------------------------------------------------


def get_annotation_file(annotations_dir: Path, record_file: Path) -> Path | None:
    return next(
        (
            candidate
            for ext in ANNOTATION_EXTENSIONS
            if (candidate := annotations_dir / (record_file.stem + ext)).is_file()
        ),
        None,
    )


def load_annotation_file(annotation_file: Path | str, sep: str = "\t") -> pd.DataFrame:
    path = Path(annotation_file)
    if path.suffix.lower() == ".csv":
        sep = ","

    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found or is not a file: {path}")

    try:
        return pd.read_csv(path, sep=sep)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Annotation file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Invalid annotation file format: {path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Could not load annotation file: {path}") from exc


# ---------------------------------------------------------------------------
# DataFrame Utils
# ---------------------------------------------------------------------------


def normalize_header(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_headers(df.copy(deep=True), rename_mapping=HEADER_RENAMES)
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    return df


def normalize_species_and_calls(df: pd.DataFrame) -> pd.DataFrame:
    df["species"] = df["species"].map(normalize_value)
    df["call_type"] = df["call_type"].map(normalize_value).replace(CALL_TYPE_CORRECTIONS)
    return df


def parse_numerics(df: pd.DataFrame) -> pd.DataFrame:
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce").round(3)
    return df.dropna(subset=NUMERIC_COLUMNS).reset_index(drop=True)


def append_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        duration_s=(df["end_time"] - df["begin_time"]).round(3),
        bandwidth_hz=(df["high_freq"] - df["low_freq"]).round(3),
    )


def filter_noise(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        ~((df["species"].str.contains("noise")) | (df["call_type"].str.contains("noise")))
    ].reset_index(drop=True)


def df_to_annotations(df: pd.DataFrame) -> list[Annotation]:
    return [
        Annotation(
            species=row["species"],
            call_type=row["call_type"],
            begin_time=float(row["begin_time"]),
            end_time=float(row["end_time"]),
            low_freq=float(row["low_freq"]),
            high_freq=float(row["high_freq"]),
        )
        for _, row in df.iterrows()
    ]


def annotations_to_df(annotations: Sequence[Annotation]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "species": [ann.species for ann in annotations],
            "call_type": [ann.call_type for ann in annotations],
            "begin_time": [ann.begin_time for ann in annotations],
            "end_time": [ann.end_time for ann in annotations],
            "low_freq": [ann.low_freq for ann in annotations],
            "high_freq": [ann.high_freq for ann in annotations],
        }
    )


# ---------------------------------------------------------------------------
# Filters & Transforms
# ---------------------------------------------------------------------------


def stretch_annotations(
    annotations: Sequence[Annotation], stretch_factor: float
) -> list[Annotation]:
    assert stretch_factor > 0, "stretch_factor must be positive"
    return [
        replace(
            ann, begin_time=ann.begin_time / stretch_factor, end_time=ann.end_time / stretch_factor
        )
        for ann in annotations
    ]


def pitch_shift_annotations(
    annotations: Sequence[Annotation], semitones: float
) -> list[Annotation]:
    if semitones == 0:
        return list(annotations)
    ratio: float = 2.0 ** (semitones / 12.0)
    return [
        replace(
            ann, low_freq=max(0.0, ann.low_freq * ratio), high_freq=max(0.0, ann.high_freq * ratio)
        )
        for ann in annotations
    ]


def filter_annotations_by_window(
    annotations: Sequence[Annotation], start_sec: float, duration_sec: float
) -> list[Annotation]:
    end_sec: float = start_sec + duration_sec
    return [
        replace(ann, begin_time=ann.begin_time - start_sec, end_time=ann.end_time - start_sec)
        for ann in annotations
        if start_sec <= ann.begin_time and ann.end_time <= end_sec
    ]


def crop_overlap_to_window(
    annotations: Sequence[Annotation], start_sec: float, duration_sec: float
) -> list[Annotation]:
    end_sec: float = start_sec + duration_sec
    result: list[Annotation] = []
    for ann in annotations:
        overlap_start = max(ann.begin_time, start_sec)
        overlap_end = min(ann.end_time, end_sec)
        if overlap_end <= overlap_start:
            continue
        result.append(
            replace(ann, begin_time=overlap_start - start_sec, end_time=overlap_end - start_sec)
        )
    return result
