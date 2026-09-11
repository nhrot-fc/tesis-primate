from collections.abc import Sequence

from evaluation.metrics import MATCH_IOU, DetectionMetrics
from evaluation.protocol import Comparison, ModelComparison, Protocol

NAME_WIDTH = 24


def fmt(value: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


def fmt_interval(interval: dict[str, list[float]], key: str) -> str:
    bounds = interval.get(key)
    return f"[{bounds[0]:.3f}, {bounds[1]:.3f}]" if bounds else "n/a"


def format_line(metrics: DetectionMetrics) -> str:
    return (
        f"recall={fmt(metrics.recall)} precision={fmt(metrics.precision)} "
        f"mAP{MATCH_IOU * 100:.0f}={fmt(metrics.map_30)}"
    )


def cell(value: object, width: int) -> str:
    # Sin separador entre celdas: el relleno separa. Ancho negativo alinea a la izquierda.
    text = str(value)
    limit = abs(width) - 1 if width < 0 else abs(width)
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return f"{text:{'<' if width < 0 else '>'}{abs(width)}}"


def table(header: dict[str, int], rows: list[Sequence]) -> list[str]:
    widths = list(header.values())
    return ["  " + "".join(cell(title, width) for title, width in header.items())] + [
        "  " + "".join(cell(value, width) for value, width in zip(row, widths, strict=True))
        for row in rows
    ]


def numbered(blocks: list[list[str]]) -> list[str]:
    lines: list[str] = []
    for number, (heading, *body) in enumerate(blocks, start=1):
        lines += ["", f"{number}. {heading}", *body]
    return lines


def map_block(models: list[ModelComparison], protocol: Protocol) -> list[str]:
    return [
        "LIBRE DE UMBRAL: qué modelo detecta mejor, sin depender de su calibración",
        *table(
            {"modelo": -NAME_WIDTH, f"mAP@{protocol.iou:g}": 9},
            [(m.model, fmt(m.map_30)) for m in models],
        ),
    ]


def paired_block(models: list[ModelComparison], position: int) -> list[str]:
    min_precision = models[0].paired[position].min_precision
    rows: list[Sequence] = []
    for m in models:
        p = m.paired[position]
        if p.test is None or p.threshold is None:
            rows.append((m.model, *["n/a"] * 5))
        else:
            rows.append(
                (
                    m.model,
                    f"{p.threshold:.2f}",
                    fmt(p.test.recall),
                    fmt_interval(p.interval, "recall"),
                    fmt(p.test.precision),
                    fmt_interval(p.interval, "precision"),
                )
            )
    lines = [
        f"PUNTO PAREADO [precision >= {min_precision:.2f} en val]: umbral de VAL, medido en TEST",
        *table(
            {
                "modelo": -NAME_WIDTH,
                "umbral": 7,
                "recall": 8,
                "IC95 recall": 18,
                "prec": 7,
                "IC95 prec": 18,
            },
            rows,
        ),
    ]
    if any(m.paired[position].threshold is None for m in models):
        lines.append("  n/a: ningún umbral de val llega a esa precisión")
    return lines


def wide_table(
    first: dict[str, int],
    models: list[ModelComparison],
    subcolumns: dict[str, int],
    rows: list[list],
) -> list[str]:
    # Una fila por clase
    # Cada modelo agrega un grupo de subcolumnas a la derecha.
    group = sum(subcolumns.values())
    first_width = sum(abs(w) for w in first.values())
    lines = [
        " " * (first_width + 2) + "".join(cell(m.model, group) for m in models),
        "  "
        + "".join(cell(title, width) for title, width in first.items())
        + "".join(cell(title, width) for _ in models for title, width in subcolumns.items()),
    ]
    widths = [*first.values(), *(list(subcolumns.values()) * len(models))]
    return lines + [
        "  " + "".join(cell(value, width) for value, width in zip(row, widths, strict=True))
        for row in rows
    ]


def per_class_block(models: list[ModelComparison], protocol: Protocol) -> list[str]:
    by_model = [{row.name: row for row in m.per_class} for m in models]
    rows = []
    for name, row in by_model[0].items():
        rows.append([name, row.n_gt])
        for per_class in by_model:
            rows[-1] += [fmt(per_class[name].recall), fmt(per_class[name].ap)]
    return [
        f"POR CLASE: recall al primer punto de operación, AP@{protocol.iou:g} sobre toda la curva",
        *wide_table(
            {"clase": -12, "cajas GT": 9}, models, {"recall": 9, f"AP@{protocol.iou:g}": 9}, rows
        ),
    ]


def window_block(models: list[ModelComparison]) -> list[str]:
    by_model = [{row.name: row for row in m.window_level} for m in models]
    rows = []
    for name, row in by_model[0].items():
        rows.append([name, row.windows_gt])
        for window_level in by_model:
            r = window_level[name]
            rows[-1] += [r.windows_predicted, fmt(r.recall), fmt(r.precision)]
    return [
        "CLASES DE VENTANA: la llamada dura más que el clip; es clasificación de ventana",
        *wide_table(
            {"clase": -12, "ventanas GT": 12},
            models,
            {"predichas": 11, "recall": 9, "prec": 9},
            rows,
        ),
    ]


def format_comparison(comparison: Comparison) -> str:
    protocol, val, test, models = (
        comparison.protocol,
        comparison.val,
        comparison.test,
        comparison.models,
    )
    header = [
        "=" * 100,
        f"COMPARACIÓN | IoU >= {protocol.iou:g}, tope {protocol.max_det} cajas por ventana, "
        f"piso de score {protocol.score_floor:g}",
        f"val  {val.windows} ventanas, {val.boxes} cajas, {val.recordings} grabaciones (elige el umbral)",
        f"test {test.windows} ventanas, {test.boxes} cajas, {test.recordings} grabaciones (lo mide)",
        f"{len(comparison.detection_classes)} clases de detección"
        + (
            f" | {len(comparison.window_classes)} de ventana ({', '.join(comparison.window_classes)})"
            if comparison.window_classes
            else ""
        ),
    ]
    blocks = [
        map_block(models, protocol),
        *(paired_block(models, i) for i in range(len(models[0].paired))),
    ]
    if any(m.per_class for m in models):
        blocks.append(per_class_block(models, protocol))
    if comparison.window_classes:
        blocks.append(window_block(models))
    footer = [
        "",
        f"IC 95% por bootstrap de {protocol.n_bootstrap} remuestreos de grabaciones, semilla {protocol.seed}.",
        "Si dos IC se solapan, este experimento no separa a esos modelos.",
        "=" * 100,
    ]
    return "\n".join([*header, *numbered(blocks), *footer]) + "\n"
