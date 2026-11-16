import json
import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile
import torch
import torchaudio
from cycler import cycler
from matplotlib.image import BboxImage
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import Bbox, TransformedBbox

from core.config import PROJECT_DIR, P
from utils.audio import hz_to_y, mel_spectrogram, mel_to_db

CAPITULOS = PROJECT_DIR / "research" / "capitulos"
SERVIDOR = os.environ.get("TESIS_SERVIDOR") == "1"

AZUL, CELESTE, VERDE, ROSA, AMBAR, GRIS = (
    "#1E40AF", "#00B4D8", "#22C55E", "#FF5FA2", "#F59E0B", "#6B7280",
)
MORADO, ROJO, TURQUESA = "#7C3AED", "#DC2626", "#0D9488"
ESPECIE = {
    "aa": MORADO, "ac": AMBAR, "as": ROJO, "cc": GRIS,
    "lw": AZUL, "pt": TURQUESA, "sb": CELESTE, "sm": VERDE,
}
ANOTACION, MODELO = VERDE, CELESTE
DISCONTINUA = (0, (3, 2))
ANCHO = 6.27  # pulgadas

mpl.rcParams.update(
    {
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "image.cmap": "magma",
        "axes.prop_cycle": cycler(color=[AZUL, AMBAR, VERDE, ROSA, CELESTE, GRIS]),
    }
)


def number(value: float, decimals: int = 0) -> str:
    if decimals == 0:
        return f"{round(value):,}".replace(",", "\\,")
    whole, fraction = f"{value:,.{decimals}f}".split(".")
    return whole.replace(",", "\\,") + "," + fraction


def percent(value: float, decimals: int = 1) -> str:
    return number(100 * value, decimals) + "\\,\\%"


def macro(key: str) -> str:
    # n_boxes_train -> nBoxesTrain: un comando de LaTeX sólo admite letras
    words = key.replace("-", "_").split("_")
    return "".join(c for c in words[0] + "".join(w.capitalize() for w in words[1:]) if c.isalpha())


MACRO = re.compile(r"^\\newcommand\{\\(\w+)\}\{(.*)\}$", re.MULTILINE)


class Salidas:
    def __init__(self, capitulo: str):
        self.dir = Path(os.environ.get("TESIS_SALIDAS", CAPITULOS)) / capitulo / "figures"
        self.ruta_tex = f"capitulos/{capitulo}/figures"
        self.capitulo = capitulo
        self.valores: dict[str, str] = {}

    def _escribir(self, nombre: str, texto: str) -> None:
        ruta = self.dir / nombre
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text("% generado por el cuaderno; no editar\n" + texto + "\n")

    def tabla(self, nombre: str, filas: list) -> None:
        # Una cadena sin & que empieza con \ es una regla (\midrule)
        lineas = [f if isinstance(f, str) else " & ".join(map(str, f)) for f in filas]
        lineas = [f if "&" not in f and f.startswith("\\") else f + " \\\\" for f in lineas]
        # Sin \\ final: el .tex pone `\input{...} \\` antes de \bottomrule
        self._escribir(f"tablas/{nombre}.tex", "\n".join(lineas).removesuffix(" \\\\"))

    def serie(self, nombre: str, frame: pd.DataFrame) -> None:
        (self.dir / "data").mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.dir / "data" / f"{nombre}.dat", sep=" ", index=False, float_format="%.6g")

    def figura(self, fig: plt.Figure, nombre: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.dir / f"{nombre}.png")

    def macros(self, conservar: bool = not SERVIDOR) -> pd.Series:
        archivo = self.dir / "valores.tex"
        valores = {macro(k): v for k, v in self.valores.items()}
        if conservar and archivo.exists():
            valores = dict(MACRO.findall(archivo.read_text())) | valores
        otras = {
            nombre
            for ruta in CAPITULOS.glob("*/figures/valores.tex")
            if ruta.parent.parent.name != self.capitulo
            for nombre, _ in MACRO.findall(ruta.read_text())
        }
        assert not otras & set(valores), f"macros repetidas: {sorted(otras & set(valores))}"
        lineas = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(valores.items())]
        self._escribir("valores.tex", "\n".join(lineas))
        return pd.Series(valores)


def anotaciones() -> pd.DataFrame:
    from analysis.review import load_annotations

    meta = json.loads((CAPITULOS / "04_oe1_curacion/figures/meta.json").read_text())
    tabla = load_annotations(meta["joined_labels"])
    return tabla.assign(duration_s=tabla.end_time_s - tabla.begin_time_s,
                        bandwidth_hz=tabla.high_freq_hz - tabla.low_freq_hz)


def cargar(ruta: Path | str, inicio: float = 0.0, duracion: float = P.clip_len_s) -> torch.Tensor:
    with soundfile.SoundFile(str(ruta)) as audio:
        audio.seek(int(inicio * audio.samplerate))
        cuadros = audio.read(int(duracion * audio.samplerate), dtype="float32", always_2d=True)
        onda = torch.from_numpy(cuadros.mean(axis=1))
        if audio.samplerate != P.target_sr:
            onda = torchaudio.functional.resample(onda, audio.samplerate, P.target_sr)
    return onda


MEL = mel_spectrogram(P)


def imagen(ax: plt.Axes, matriz: np.ndarray, inicio: float, duracion: float) -> None:
    bajo, alto = np.percentile(matriz, (5, 99.5))
    ax.imshow(matriz, origin="lower", aspect="auto", extent=(inicio, inicio + duracion, 0, 1),
              vmin=bajo, vmax=alto, interpolation="nearest")
    ax.grid(False)
    khz = np.array([1, 5, 15])
    ax.set_yticks(hz_to_y(khz * 1000.0, P), [f"{k}" for k in khz])
    ax.set(xlabel="s", ylabel="kHz")


def espectrograma(ax: plt.Axes, onda: torch.Tensor, inicio: float = 0.0) -> None:
    imagen(ax, mel_to_db(MEL(onda)).numpy(), inicio, onda.numel() / P.target_sr)


def caja(ax: plt.Axes, inicio: float, fin: float, baja: float, alta: float,
         modelo: bool = False, texto: str | None = None, estilo=None) -> None:
    y0, y1 = np.clip(hz_to_y(np.array([max(baja, P.f_min), alta]), P), 0, 1)
    color = MODELO if modelo else ANOTACION
    ax.add_patch(Rectangle((inicio, y0), fin - inicio, y1 - y0, fill=False, ec=color, lw=1.2,
                           ls=estilo or (DISCONTINUA if modelo else "-")))
    if texto:
        ax.text(inicio, min(y1 + 0.02, 0.95), texto, color=color, fontsize=6.5, va="bottom")


def caja_ventana(ax: plt.Axes, cxcywh, inicio: float, modelo: bool = False) -> None:
    # cxcywh en [0, 1] de la ventana que empieza en inicio; y ya está sobre el eje mel
    cx, cy, w, h = map(float, cxcywh)
    ax.add_patch(Rectangle((inicio + (cx - w / 2) * P.clip_len_s, cy - h / 2), w * P.clip_len_s, h,
                           fill=False, ec=MODELO if modelo else ANOTACION, lw=1.2,
                           ls=DISCONTINUA if modelo else "-"))


def anotada(texto: str = "anotación") -> Patch:
    return Patch(fill=False, ec=ANOTACION, lw=1.2, label=texto)


def predicha(texto: str = "detección del modelo") -> Patch:
    return Patch(fill=False, ec=MODELO, lw=1.2, ls=DISCONTINUA, label=texto)


class Energia:
    def get_label(self) -> str:
        return "energía: baja → alta"


class _Degradado(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        caja = TransformedBbox(Bbox.from_bounds(xdescent, ydescent, width, height), trans)
        degradado = BboxImage(caja, cmap=mpl.rcParams["image.cmap"], interpolation="bilinear")
        degradado.set_data(np.linspace(0, 1, 64)[None, :])
        return [degradado]


def leyenda(destino: plt.Figure | plt.Axes, *entradas, energia: bool = False, **opciones):
    # En una figura va en una fila al pie; energia: la escala de color de los espectrogramas
    entradas = (*entradas, Energia()) if energia else entradas
    base: dict[str, object] = {"handles": entradas, "labels": [e.get_label() for e in entradas],
                               "handlelength": 2.5, "handler_map": {Energia: _Degradado()}}
    if isinstance(destino, plt.Figure):
        abajo = isinstance(destino.get_layout_engine(), ConstrainedLayoutEngine)
        base["ncols"] = len(entradas)
        base["loc"] = "outside lower center" if abajo else "upper center"
        if not abajo:
            base["bbox_to_anchor"] = (0.5, 0.0)
    return destino.legend(**(base | opciones))
