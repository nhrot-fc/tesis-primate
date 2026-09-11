from collections.abc import Iterable

import pandas as pd

# Columnas de una tabla de selección de Raven: las lee el visor y las escribe el predictor.
SELECTION, VIEW, CHANNEL = "Selection", "View", "Channel"
BEGIN, END, LOW, HIGH = "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)"
SPECIES, CALL = "Species", "Call type"
# Sólo en las tablas del modelo
SCORE = "Score"
BOX_COLUMNS = [BEGIN, END, LOW, HIGH]
DEFAULT_VIEW, DEFAULT_CHANNEL = "Spectrogram 1", 1


# Las mismas cajas en `cleaned/`: el nombre de Raven en snake_case. `data.annotations.cleaned`
# lo deriva con slugify y comprueba que coincida; acá van literales para que el visor no lo cargue.
CLEANED_BOX_COLUMNS = ["begin_time_s", "end_time_s", "low_freq_hz", "high_freq_hz"]


def renumber(table: pd.DataFrame) -> pd.DataFrame:
    # Raven exige `Selection` correlativo desde 1.
    table[SELECTION] = range(1, len(table) + 1)
    return table


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
