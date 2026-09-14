"""Reúne lo que `avance.tex` no toma directamente de `research/reportes/figures/`: los esquemas
TikZ de los reportes (por su `\\label`), dos figuras del prototipo de 2024
(`notebooks/TF_Experimental_Training.ipynb`), las curvas de validación de la sexta configuración y
las cifras de las dos comparaciones (con y sin postprocesado de los DETR)."""

import base64
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[2]
REPORTES = PROJECT_DIR / "research" / "reportes"
FIGS = HERE / "figs"

# (reporte, label de la figura) -> nombre del fragmento TikZ
TIKZ = {
    ("RE_1-1_protocolo-curacion", "fig:pipeline"): "protocolo",
    ("RE_1-2_conjunto-particionado", "fig:flujo"): "conjunto",
    ("RE_2-1_diseno-arquitectura", "fig:arquitectura"): "arquitectura",
    ("RE_2-1_diseno-arquitectura", "fig:protocolo"): "evaluacion",
    ("RE_2-2_pipeline", "fig:pipeline"): "pipeline",
}
# (celda, índice de imagen en la celda) -> nombre; celda 42: clips de una misma anotación;
# celda 52: salida del clasificador sobre una grabación de prueba.
PROTOTIPO = {(42, 1): "prototipo_clips", (52, 0): "prototipo_prediccion"}
# Sexta configuración y su referencia sin preentrenar la cabeza: curvas de mAP@0,3 en validación.
RUNS = {"birds": "detr_t10_logmel_birds", "logmel": "detr_t10_logmel_v2"}
EPOCH_LINE = re.compile(r"\[(?P<epoch>\d+)/(?P<total>\d+)\].*mAP30=(?P<map>[0-9.]+)")
# Épocas de la meseta: la mediana de las últimas es la estimación de val que no elige el máximo.
PLATEAU_EPOCHS = 10
# Comparaciones de `compare_models.py`: con el postprocesado de los DETR (la vigente) y sin él
# (copia guardada antes del cambio); modelo -> prefijo de macro.
COMPARISONS = {"con": "comparacion_modelos", "sin": "comparacion_modelos_detr_sin_nms"}
DETR_RUNS = {
    "detr_t10_logmel_v2": "Logmel",
    "detr_t10_logmel_birds": "Birds",
    "detr_resnet": "Resnet",
}
# Referencia para leer el tamaño de la brecha con YOLO.
BEST_RUN = ("yolo26s_v2", "Yolo")


def decimal(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def val_curve(run: str) -> list[float]:
    log = PROJECT_DIR / "runs" / run / "train.log"
    return [
        float(m["map"]) for line in log.read_text().splitlines() if (m := EPOCH_LINE.search(line))
    ]


def curves_figure(curves: dict[str, list[float]]) -> str:
    # Las dos curvas de validación, con el máximo de cada una marcado.
    plots = []
    for prefix, values in curves.items():
        color = "azul" if prefix == "birds" else "gris"
        coords = " ".join(f"({i + 1},{v:.3f})" for i, v in enumerate(values))
        best = max(range(len(values)), key=lambda i: values[i])
        plots.append(f"\\addplot[{color}, thick, mark=none] coordinates {{{coords}}};")
        plots.append(
            f"\\addplot[{color}, only marks, mark=*, mark size=2pt] coordinates {{({best + 1},{values[best]:.3f})}};"
        )
    return (
        "\\begin{tikzpicture}\n"
        "\\begin{axis}[width=\\linewidth, height=4.2cm, xlabel={época}, ylabel={mAP@0,3 en val},"
        " ymin=0.1, ymax=0.7, xmin=1, grid=major, tick label style={font=\\scriptsize},"
        " label style={font=\\scriptsize}, legend style={font=\\scriptsize, at={(0.97,0.05)}, anchor=south east},"
        " legend cell align=left]\n"
        + "\n".join(plots)
        + "\n\\legend{preentrenado con aves,, sin preentrenar,}\n"
        "\\end{axis}\n\\end{tikzpicture}\n"
    )


def tikz_block(source: str, label: str) -> str:
    # La figura que lleva el label: su único entorno tikzpicture.
    figures = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", source, flags=re.S)
    figure = next(f for f in figures if f"\\label{{{label}}}" in f)
    match = re.search(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", figure, flags=re.S)
    assert match is not None, label
    return match.group(0)


def main() -> None:
    FIGS.mkdir(exist_ok=True)
    for (report, label), name in TIKZ.items():
        source = (REPORTES / f"{report}.tex").read_text()
        header = f"% extraído de research/reportes/{report}.tex ({label}) por armar_figuras.py\n"
        (FIGS / f"{name}.tex").write_text(header + tikz_block(source, label) + "\n")

    # Primera fila de la galería de clases de RE1.3 (tres subfiguras), para una diapositiva.
    galeria = (REPORTES / "figures" / "RE_1-3" / "galeria.tex").read_text()
    fila = galeria.split("\\\\[1.5ex]\n")[0]
    (FIGS / "galeria_fila.tex").write_text(
        "% primera fila de research/reportes/figures/RE_1-3/galeria.tex, por armar_figuras.py\n"
        + fila.replace("% generado por el cuaderno homónimo; no editar\n", "")
        + "\n"
    )

    notebook = json.loads(
        (PROJECT_DIR / "notebooks" / "TF_Experimental_Training.ipynb").read_text()
    )
    for (cell, position), name in PROTOTIPO.items():
        images = [
            output["data"]["image/png"]
            for output in notebook["cells"][cell].get("outputs", [])
            if "image/png" in output.get("data", {})
        ]
        (FIGS / f"{name}.png").write_bytes(base64.b64decode(images[position]))

    values: dict[str, str] = {}
    curves = {prefix: val_curve(run) for prefix, run in RUNS.items()}
    for prefix, series in curves.items():
        best = max(range(len(series)), key=lambda i: series[i])
        values[f"{prefix}Epochs"] = str(len(series))
        values[f"{prefix}BestMap"] = decimal(series[best])
        values[f"{prefix}BestEpoch"] = str(best + 1)
        values[f"{prefix}PlateauMap"] = decimal(
            sorted(series[-PLATEAU_EPOCHS:])[PLATEAU_EPOCHS // 2]
        )
        values[f"{prefix}EpochFive"] = decimal(series[4], 2)
    (FIGS / "curvas_aves.tex").write_text(
        "% curvas de mAP@0,3 en validación de runs/*/train.log, por armar_figuras.py\n"
        + curves_figure(curves)
    )

    # mAP@0,3 y punto de operación principal en test, con y sin postprocesado de los DETR.
    for variant, name in COMPARISONS.items():
        report = json.loads((PROJECT_DIR / "runs" / "comparacion" / f"{name}.json").read_text())
        by_model = {m["model"]: m for m in report["models"]}
        for run, suffix in {**DETR_RUNS, BEST_RUN[0]: BEST_RUN[1]}.items():
            model = by_model[run]
            main = model["paired"][0]
            values[f"{variant}Nms{suffix}Map"] = decimal(model["map_30"])
            values[f"{variant}Nms{suffix}Threshold"] = decimal(main["threshold"], 2)
            values[f"{variant}Nms{suffix}Recall"] = decimal(main["test"]["recall"])
            values[f"{variant}Nms{suffix}RecallLow"] = decimal(main["interval"]["recall"][0])
            values[f"{variant}Nms{suffix}RecallHigh"] = decimal(main["interval"]["recall"][1])
            values[f"{variant}Nms{suffix}Precision"] = decimal(main["test"]["precision"])
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in values.items()]
    (FIGS / "valores.tex").write_text(
        "% generado por armar_figuras.py; no editar\n" + "\n".join(lines) + "\n"
    )
    print("\n".join(sorted(p.name for p in FIGS.iterdir())))


if __name__ == "__main__":
    main()
