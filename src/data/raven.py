from collections.abc import Iterable

import pandas as pd

SELECTION, VIEW, CHANNEL = "Selection", "View", "Channel"
BEGIN, END, LOW, HIGH = "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)"
SPECIES, CALL = "Species", "Call type"
SCORE = "Score"
BOX_COLUMNS = [BEGIN, END, LOW, HIGH]
DEFAULT_VIEW, DEFAULT_CHANNEL = "Spectrogram 1", 1


CLEANED_BOX_COLUMNS = ["begin_time_s", "end_time_s", "low_freq_hz", "high_freq_hz"]


def renumber(table: pd.DataFrame) -> pd.DataFrame:
    table[SELECTION] = range(1, len(table) + 1)
    return table


def selection_table(table: pd.DataFrame) -> pd.DataFrame:
    table = renumber(table)
    absent = pd.Series(index=table.index, dtype=object)
    table[VIEW] = table.get(VIEW, absent).fillna(DEFAULT_VIEW)
    channel = pd.to_numeric(table.get(CHANNEL, absent), errors="coerce")
    table[CHANNEL] = channel.fillna(DEFAULT_CHANNEL).astype(int)
    first = [SELECTION, VIEW, CHANNEL]
    return table[[*first, *(c for c in table.columns if c not in first)]]


def raven_table(
    begin: Iterable[float],
    end: Iterable[float],
    low: Iterable[float],
    high: Iterable[float],
    species: Iterable[str],
    call_type: Iterable[str],
    score: Iterable[float],
) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            VIEW: DEFAULT_VIEW,
            CHANNEL: DEFAULT_CHANNEL,
            BEGIN: begin,
            END: end,
            LOW: low,
            HIGH: high,
            SPECIES: species,
            CALL: call_type,
            SCORE: score,
        }
    )
    table = renumber(table)
    return table[[SELECTION, *table.columns[:-1]]]
