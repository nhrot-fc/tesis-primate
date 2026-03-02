import pandas as pd
from slugify import slugify


def normalize_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return slugify(text).replace("-", "").lower()


def normalize_headers(
    df: pd.DataFrame, rename_mapping: dict[str, str] | None = None
) -> pd.DataFrame:
    df = df.copy()
    normalized_columns = []
    rename_mapping = rename_mapping or {}

    for column in df.columns:
        normalized = slugify(str(column)).replace("-", "_").lower()
        normalized_columns.append(rename_mapping.get(normalized, normalized))

    df.columns = normalized_columns
    return df
