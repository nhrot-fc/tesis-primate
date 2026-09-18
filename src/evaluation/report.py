from collections.abc import Sequence

from evaluation.metrics import MATCH_IOU, DetectionMetrics
from evaluation.protocol import Comparison, ModelComparison, Protocol

NAME_WIDTH = 26
WINDOW_MARK = "*"


def fmt(value: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


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


def global_block(models: list[ModelComparison], protocol: Protocol) -> list[str]:
    rows: list[Sequence] = []
    for m in models:
        point = m.test
        rows.append(
            (
                m.model,
                fmt(point.recall if point else None),
                fmt(point.precision if point else None),
                fmt(m.map_30),
                fmt(m.map_50),
            )
        )
    return [
        "GLOBAL",
        *table(
            {
                "modelo": -NAME_WIDTH,
                "recall": 9,
                "precisión": 11,
                f"mAP@{protocol.iou:g}": 9,
                f"mAP@{protocol.strict_iou:g}": 9,
            },
            rows,
        ),
    ]


def axes_block(models: list[ModelComparison], protocol: Protocol) -> list[str]:
    rows: list[Sequence] = []
    for m in models:
        a = m.axes
        rows.append(
            (
                m.model,
                fmt(a.recall if a else None),
                fmt(a.precision if a else None),
                fmt(a.median_iou if a else None),
                fmt(a.recall_strict if a else None),
                fmt(a.class_correct if a else None),
                fmt(a.species_correct if a else None),
            )
        )
    return [
        "DETECCIÓN, ENCUADRE Y CLASIFICACIÓN",
        *table(
            {
                "modelo": -NAME_WIDTH,
                "recall": 9,
                "precisión": 11,
                "IoU med.": 10,
                f"recall@{protocol.strict_iou:g}": 12,
                "clase ok": 10,
                "especie ok": 12,
            },
            rows,
        ),
        "  detección: emparejando sin mirar la clase | encuadre: IoU mediano de esos pares y "
        f"recall con clase a IoU {protocol.strict_iou:g} | clasificación: de lo encontrado, "
        "qué fracción lleva la clase y la especie anotadas",
    ]


def wide_table(
    first: dict[str, int],
    models: list[ModelComparison],
    subcolumns: dict[str, int],
    rows: list[list],
) -> list[str]:
    # Una fila por clase; cada modelo agrega un grupo de subcolumnas a la derecha.
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
        rows.append([name + (WINDOW_MARK if row.window_class else ""), row.n_gt])
        for per_class in by_model:
            r = per_class[name]
            rows[-1] += [fmt(r.recall), fmt(r.precision), fmt(r.ap), fmt(r.ap_strict)]
    lines = [
        "POR CLASE (especie/llamada)",
        *wide_table(
            {"clase": -12, "cajas": 7},
            models,
            {"recall": 8, "prec": 8, f"AP@{protocol.iou:g}": 8, f"AP@{protocol.strict_iou:g}": 8},
            rows,
        ),
    ]
    if any(row.window_class for row in models[0].per_class):
        lines.append(
            f"  {WINDOW_MARK} clase de ventana: la caja ocupa la ventana; no entra en la mAP"
        )
    return lines


def format_comparison(comparison: Comparison) -> str:
    protocol, val, test, models = (
        comparison.protocol,
        comparison.val,
        comparison.test,
        comparison.models,
    )
    header = [
        "=" * 100,
        f"COMPARACIÓN | acierto a IoU >= {protocol.iou:g} | tope {protocol.max_det} cajas por "
        "ventana | umbral de cada modelo elegido en val "
        f"(precisión >= {protocol.min_precision:.2f}), métricas en test",
        f"val  {val.windows} ventanas, {val.boxes} cajas, {val.recordings} grabaciones",
        f"test {test.windows} ventanas, {test.boxes} cajas, {test.recordings} grabaciones",
    ]
    blocks = [global_block(models, protocol), axes_block(models, protocol)]
    if any(m.per_class for m in models):
        blocks.append(per_class_block(models, protocol))
    if any(m.threshold is None for m in models):
        blocks[0].append("  n/a: ningún umbral de val llega a la precisión mínima")
    return "\n".join([*header, *numbered(blocks), "=" * 100]) + "\n"
