"""Lo que comparten los cuadernos `RE_*.ipynb`: rutas del proyecto y cómo dejan figuras, tablas
y cifras en `figures/RE_x-y/` para que el `RE_*.tex` homónimo las incluya con `\\input`."""

import sys
from collections.abc import Iterable
from pathlib import Path

import matplotlib as mpl
import pandas as pd
from matplotlib.figure import Figure

PROJECT_DIR = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pyproject.toml").exists())
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
if str(PROJECT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "src"))

# Ancho del texto del documento (A4 con márgenes 3 + 2,5 cm); una figura de este ancho se
# imprime a escala 1 y sus letras salen del tamaño que se ve en el cuaderno.
TEXT_WIDTH_IN = 6.1
DPI = 200
mpl.rcParams.update(
    {
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def out_dir(report: str) -> Path:
    path = FIGURES_DIR / report
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: Figure, path: Path) -> None:
    # PNG a 200 ppp: se ve igual en el PDF y no arrastra miles de puntos vectoriales.
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=DPI)


# Cifras a la española: miles con espacio fino y coma decimal, como el resto del documento.
def number(value: float | int, decimals: int = 0) -> str:
    if decimals == 0:
        return f"{round(value):,}".replace(",", "\\,")
    whole, fraction = f"{value:,.{decimals}f}".split(".")
    return whole.replace(",", "\\,") + "," + fraction


def percent(value: float, decimals: int = 1) -> str:
    return number(100 * value, decimals) + "\\,\\%"


def macro_name(key: str) -> str:
    # `n_boxes_train` -> `nBoxesTrain`; LaTeX no admite dígitos ni guiones en un comando.
    words = key.replace("-", "_").split("_")
    name = words[0] + "".join(w.capitalize() for w in words[1:])
    return "".join(c if c.isalpha() else "" for c in name)


def write_macros(path: Path, values: dict[str, str]) -> None:
    lines = [f"\\newcommand{{\\{macro_name(k)}}}{{{v}}}" for k, v in values.items()]
    path.write_text("% generado por el cuaderno homónimo; no editar\n" + "\n".join(lines) + "\n")


def write_rows(path: Path, rows: Iterable[Iterable[object]]) -> None:
    # Sólo las filas, sin `\\` al final de la última: un `\input` que termina en `\\` rompe el
    # `\bottomrule` que le sigue, así que el `.tex` escribe `\input{...} \\`.
    body = " \\\\\n".join(" & ".join(str(cell) for cell in row) for row in rows)
    path.write_text("% generado por el cuaderno homónimo; no editar\n" + body + "\n")


def write_data(path: Path, frame: pd.DataFrame) -> None:
    # Para pgfplots: columnas separadas por espacios, sin índice.
    frame.to_csv(path.with_suffix(".dat"), sep=" ", index=False)


def scalar(frame: pd.DataFrame, row: object, column: str) -> float:
    # Un valor numérico de una celda; `.loc` y `.at` devuelven una unión enorme en los stubs.
    return float(frame[column].to_numpy()[frame.index.get_loc(row)])
