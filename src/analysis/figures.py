"""Ventanas del caché dibujadas: el mel en magma con el grave abajo, las cajas con borde negro
(se ven igual sobre el fondo oscuro y sobre una llamada brillante) y la clase de interés en verde."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib_inline.backend_inline import set_matplotlib_formats

from analysis.windows import Split
from core.config import P
from utils.audio import hz_to_y, mel_to_unit

CLASS_COLOR, OTHER_COLOR, HIGHLIGHT_COLOR = "#3ddc84", "#bdbdbd", "#00e5ff"
OUTLINE = [patheffects.withStroke(linewidth=4, foreground="black")]
TICKS_HZ = np.array([500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0])
# Ancho de las figuras de paneles y alto de cada fila, en pulgadas
GRID_WIDTH, ROW_HEIGHT = 16.0, 3.1


def jpeg_figures(quality: int = 88) -> None:
    # Figuras de espectrogramas: en jpeg pesan diez veces menos que en png
    set_matplotlib_formats("jpeg", pil_kwargs={"quality": quality})


def panel_grid(n_rows: int, n_cols: int) -> tuple[Figure, np.ndarray]:
    # Paneles apagados; `draw_window` enciende cada uno que usa
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(GRID_WIDTH, ROW_HEIGHT * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    for ax in axes.flat:
        ax.axis("off")
    return fig, axes


def draw_box(
    ax: Axes,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    color: str,
    text: str,
    dashed: bool = False,
) -> None:
    # Caja en segundos × fila del mel en [0, 1], con su etiqueta arriba a la izquierda
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            ec=color,
            lw=2.2 if dashed else 1.6,
            ls="--" if dashed else "-",
            path_effects=OUTLINE,
        )
    )
    ax.text(x0, y1, text, color=color, fontsize=7, va="bottom", path_effects=OUTLINE)


def draw_window(
    ax: Axes, split: Split, image_id: int, class_id: int, title: str | None = None
) -> None:
    # La ventana con sus anotaciones: las de `class_id` en verde, el resto en gris
    ax.axis("on")
    ax.imshow(
        mel_to_unit(split.images[image_id][0], *split.db_range).numpy(),
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=(0, P.clip_len_s, 0, 1),
    )
    for box, label in zip(
        split.boxes[image_id].tolist(), split.labels[image_id].tolist(), strict=True
    ):
        cx, cy, w, h = box
        color = CLASS_COLOR if label == class_id else OTHER_COLOR
        x0, x1 = (cx - w / 2) * P.clip_len_s, (cx + w / 2) * P.clip_len_s
        draw_box(ax, x0, x1, cy - h / 2, cy + h / 2, color, split.label_set.name(label))
    ax.set_yticks(hz_to_y(TICKS_HZ, P).tolist(), [f"{f / 1000:g}k" for f in TICKS_HZ])
    ax.set_xlabel("s", labelpad=1)
    if title is None:
        title = f"{split.recording_of(image_id)} · {split.start_of(image_id):.0f} s"
    ax.set_title(title, fontsize=8)
    ax.grid(False)

