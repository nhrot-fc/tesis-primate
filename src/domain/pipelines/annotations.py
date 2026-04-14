from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pandas as pd

from core.logging import log_error, log_info
from domain.pipelines.types import AnnotationBox
from domain.utils.text_normalization import normalize_headers, normalize_value

REQUIRED_COLUMNS = [
    "specie",
    "call_type",
    "begin_time",
    "end_time",
    "low_freq",
    "high_freq",
]
NUMERIC_COLUMNS = ["begin_time", "end_time", "low_freq", "high_freq"]

CALL_TYPE_CORRECTIONS = {
    "csa": "cs",
    "contactcall": "cc",
    "contactsyllable": "cs",
}
HEADER_RENAMES = {
    "begin_time_s": "begin_time",
    "end_time_s": "end_time",
    "low_freq_hz": "low_freq",
    "high_freq_hz": "high_freq",
    "inband_power_db_fs": "inband_power",
    "species": "specie",
}


def get_annotation_file(annotations_dir: Path, record_file: Path) -> Path | None:
    allowed_extensions = [".csv", ".txt", ".tsv"]
    for ext in allowed_extensions:
        candidate = annotations_dir / (record_file.stem + ext)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def load_annotation_file(annotation_file: Path | str, sep: str = "\t") -> pd.DataFrame:
    if Path(annotation_file).suffix.lower() == ".csv":
        sep = ","

    path = Path(annotation_file)
    log_info("annotation.load.start", path=str(path), sep=sep)

    if not path.exists():
        log_error("annotation.load.error", path=str(path), error="file_not_found")
        raise FileNotFoundError(f"Annotation file not found: {path}")

    if not path.is_file():
        log_error("annotation.load.error", path=str(path), error="not_a_file")
        raise FileNotFoundError(f"Annotation path is not a file: {path}")

    try:
        df = pd.read_csv(path, sep=sep)
    except pd.errors.EmptyDataError as exc:
        log_error("annotation.load.error", path=str(path), error="empty_file")
        raise ValueError(f"Annotation file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        log_error("annotation.load.error", path=str(path), error="parser_error")
        raise ValueError(f"Invalid annotation file format: {path}") from exc
    except Exception as exc:
        log_error("annotation.load.error", path=str(path), error=str(exc))
        raise RuntimeError(f"Could not load annotation file: {path}") from exc

    log_info("annotation.load.success", path=str(path), rows=len(df), columns=len(df.columns))
    return df


def clean_annotation_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(deep=True)
    df = normalize_headers(df, rename_mapping=HEADER_RENAMES)
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing_text}")

    df["specie"] = df["specie"].map(normalize_value)
    df["call_type"] = df["call_type"].map(normalize_value).replace(CALL_TYPE_CORRECTIONS)

    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce").round(3)

    has_labels = df["specie"].ne("") & df["call_type"].ne("")
    has_noise = df["specie"].str.contains("noise", na=False) | df["call_type"].str.contains(
        "noise", na=False
    )

    df = df[has_labels & ~has_noise]
    df = df.dropna(subset=NUMERIC_COLUMNS).reset_index(drop=True)

    df = df.assign(
        duration_s=(df["end_time"] - df["begin_time"]).round(3),
        bandwidth_hz=(df["high_freq"] - df["low_freq"]).round(3),
    )

    return df[REQUIRED_COLUMNS + ["duration_s", "bandwidth_hz"]]


def df_to_annotations(df: pd.DataFrame) -> list[AnnotationBox]:
    boxes = []
    for _, row in df.iterrows():
        box = AnnotationBox(
            specie=row["specie"],
            call_type=row["call_type"],
            begin_time=float(row["begin_time"]),
            end_time=float(row["end_time"]),
            low_freq=float(row["low_freq"]),
            high_freq=float(row["high_freq"]),
        )
        boxes.append(box)
    return boxes


def clip_annotations_to_window(
    annotations: Sequence[AnnotationBox],
    start_sec: float,
    duration_sec: float,
) -> list[AnnotationBox]:
    end_sec = start_sec + duration_sec
    clipped_annotations: list[AnnotationBox] = []

    for ann in annotations:
        if ann.begin_time < start_sec:
            continue
        if ann.end_time > end_sec:
            continue

        overlap_start = max(ann.begin_time, start_sec)
        overlap_end = min(ann.end_time, end_sec)
        if overlap_end <= overlap_start:
            continue

        clipped_annotations.append(
            replace(
                ann,
                begin_time=overlap_start - start_sec,
                end_time=overlap_end - start_sec,
            )
        )

    return clipped_annotations
