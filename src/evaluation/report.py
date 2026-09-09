from collections.abc import Sequence

from evaluation.metrics import BETA, MATCH_IOU, DetectionMetrics
from evaluation.protocol import (
    TARGET_PRECISION,
    Comparison,
    ModelComparison,
    Protocol,
)

NAME_WIDTH = 24


def format_metric(value: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


def format_interval(interval: dict[str, list[float]], key: str) -> str:
    bounds = interval.get(key)
    return f"[{bounds[0]:.3f}, {bounds[1]:.3f}]" if bounds else "n/a"


def format_line(metrics: DetectionMetrics) -> str:
    return (
        f"recall={format_metric(metrics.recall)} "
        f"precision={format_metric(metrics.precision)} "
        f"F{BETA:g}={format_metric(metrics.f_beta)} "
        f"FP/h={format_metric(metrics.fp_per_hour, 0)} "
        f"mAP{MATCH_IOU * 100:.0f}={format_metric(metrics.map_30)} "
        f"mAP50={format_metric(metrics.map_50)} "
        f"mAP50-95={format_metric(metrics.map_50_95)}"
    )


def cell(value: object, width: int) -> str:
    text = str(value)
    # Entre celdas no hay separador: lo que las separa es el relleno. En las columnas
    # alineadas a la izquierda un texto que llena el ancho justo se pega con la siguiente,
    # así que ahí el límite es uno menos.
    limit = abs(width) - 1 if width < 0 else abs(width)
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    alignment = "<" if width < 0 else ">"
    return f"{text:{alignment}{abs(width)}}"


def table(header: dict[str, int], rows: list[Sequence]) -> list[str]:
    widths = list(header.values())
    return ["  " + "".join(cell(title, width) for title, width in header.items())] + [
        "  " + "".join(cell(value, width) for value, width in zip(row, widths, strict=True))
        for row in rows
    ]


def numbered(blocks: list[list[str]]) -> list[str]:
    lines: list[str] = []
    for number, block in enumerate(blocks, start=1):
        heading, *body = block
        lines += ["", f"{number}. {heading}", *body]
    return lines


def threshold_free_block(models: list[ModelComparison], protocol: Protocol) -> list[str]:
    return [
        "LIBRE DE UMBRAL: qué modelo detecta mejor, sin depender de su calibración",
        *table(
            {
                "modelo": -NAME_WIDTH,
                "det/vent": 9,
                f"mAP@{protocol.iou:g}": 9,
                "mAP@0.5": 9,
                "mAP@.5:.95": 11,
                f"R max si P>={TARGET_PRECISION:g}": 18,
                "mAP todas": 10,
            },
            [
                (
                    model.model,
                    f"{model.detections_per_window:.1f}",
                    format_metric(model.map_30),
                    format_metric(model.map_50),
                    format_metric(model.map_50_95),
                    format_metric(model.max_recall),
                    format_metric(model.map_all),
                )
                for model in models
            ],
        ),
        "  det/vent es lo que emitía antes de igualar; R max se elige sobre la curva que mide.",
    ]


def paired_row(model: ModelComparison, position: int) -> tuple:
    row = model.paired[position]
    if row.test is None or row.threshold is None:
        return (model.model, *["n/a"] * 7)
    return (
        model.model,
        f"{row.threshold:.2f}",
        format_metric(row.test.recall),
        format_interval(row.interval, "recall"),
        format_metric(row.test.precision),
        format_metric(row.test.fp_per_hour, 0),
        format_metric(row.test.boxes_per_tp),
        format_metric(row.agnostic_recall),
    )


def paired_block(models: list[ModelComparison], position: int) -> list[str]:
    criterion = models[0].paired[position].criterion
    lines = [
        f"PUNTO PAREADO [{criterion}]: umbral de VAL, medido en TEST",
        *table(
            {
                "modelo": -NAME_WIDTH,
                "umbral": 7,
                "recall": 8,
                "IC95 recall": 18,
                "prec": 7,
                "FP/h": 7,
                "cajas/TP": 9,
                "R sin clase": 12,
            },
            [paired_row(model, position) for model in models],
        ),
    ]
    if any(model.paired[position].threshold is None for model in models):
        lines.append("  n/a: ningún umbral de val cumple el criterio")
    return lines


def decomposition_block(models: list[ModelComparison]) -> list[str]:
    return [
        "DESCOMPOSICIÓN (test, sin umbral): dónde se pierde",
        *table(
            {
                "modelo": -NAME_WIDTH,
                "AP agn": 8,
                "IoU t": 7,
                "IoU f": 7,
                "top1": 7,
                "especie": 9,
                "+etiquetas": 12,
                "+cajas": 9,
                "n pares": 9,
            },
            [
                (
                    model.model,
                    format_metric(model.decomposition.ap_agnostic),
                    format_metric(model.decomposition.iou_time),
                    format_metric(model.decomposition.iou_freq),
                    format_metric(model.decomposition.top1),
                    format_metric(model.decomposition.species_top1),
                    format_metric(model.decomposition.label_oracle_gain),
                    format_metric(model.decomposition.box_oracle_gain),
                    model.decomposition.n_matched,
                )
                for model in models
            ],
        ),
        "  +etiquetas y +cajas: cuánto mAP recupera un oráculo que corrige esa pieza.",
    ]


def per_class_block(models: list[ModelComparison], protocol: Protocol) -> list[str]:
    return [
        f"POR CLASE [{protocol.reference_criterion}]: recall al punto de operación, "
        f"AP@{protocol.iou:g} sobre toda la curva",
        *table(
            {
                "modelo": -NAME_WIDTH,
                "clase": -12,
                "cajas GT": 9,
                "recall": 8,
                f"AP@{protocol.iou:g}": 9,
            },
            [
                (model.model, row.name, row.n_gt, format_metric(row.recall), format_metric(row.ap))
                for model in models
                for row in model.per_class
            ],
        ),
    ]


def window_block(models: list[ModelComparison], protocol: Protocol) -> list[str]:
    return [
        f"CLASES DE VENTANA [{protocol.reference_criterion}]: la llamada dura más que el clip, "
        "así que",
        "   la caja ocupa la ventana entera y esto es clasificación de ventana, no detección",
        *table(
            {
                "modelo": -NAME_WIDTH,
                "clase": -10,
                "ventanas GT": 12,
                "predichas": 11,
                "recall": 8,
                "prec": 7,
            },
            [
                (
                    model.model,
                    row.name,
                    row.windows_gt,
                    row.windows_predicted,
                    format_metric(row.recall),
                    format_metric(row.precision),
                )
                for model in models
                for row in model.window_level
            ],
        ),
    ]


def format_comparison(comparison: Comparison) -> str:
    protocol, val, test = comparison.protocol, comparison.val, comparison.test
    models = comparison.models

    header = [
        "=" * 100,
        f"COMPARACIÓN | IoU >= {protocol.iou:g}, tope {protocol.max_det} cajas por ventana, "
        f"piso de score {protocol.score_floor:g}",
        f"val  {val.windows} ventanas, {val.boxes} cajas / {val.recordings} grabaciones "
        "(elige el umbral)",
        f"test {test.windows} ventanas, {test.boxes} cajas / {test.recordings} grabaciones "
        "(lo mide)",
        f"{len(comparison.detection_classes)} clases de detección"
        + (
            f" | {len(comparison.window_classes)} de ventana "
            f"({', '.join(comparison.window_classes)}), en su propio bloque"
            if comparison.window_classes
            else " | ninguna satura la ventana en este split"
        ),
    ]

    blocks = [
        threshold_free_block(models, protocol),
        *(paired_block(models, position) for position in range(len(models[0].paired))),
        decomposition_block(models),
    ]
    if any(model.per_class for model in models):
        blocks.append(per_class_block(models, protocol))
    if comparison.window_classes:
        blocks.append(window_block(models, protocol))

    footer = [
        "",
        f"IC 95% por bootstrap de {protocol.n_bootstrap} remuestreos de grabaciones "
        f"(semilla {protocol.seed}).",
        "Si dos IC se solapan, este experimento no separa a esos modelos.",
        "=" * 100,
    ]
    return "\n".join([*header, *numbered(blocks), *footer]) + "\n"
