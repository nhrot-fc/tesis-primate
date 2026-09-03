from evaluation.metrics import BETA, MATCH_IOU, DetectionMetrics


def format_metric(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def format_rate(value: float | None) -> str:
    return f"{value:.0f}" if value is not None else "n/a"


def format_line(metrics: DetectionMetrics) -> str:
    return (
        f"recall={format_metric(metrics.recall)} "
        f"precision={format_metric(metrics.precision)} "
        f"F{BETA:g}={format_metric(metrics.f_beta)} "
        f"FP/h={format_rate(metrics.fp_per_hour)} "
        f"mAP{MATCH_IOU * 100:.0f}={format_metric(metrics.map_30)} "
        f"mAP50={format_metric(metrics.map_50)} "
        f"mAP50-95={format_metric(metrics.map_50_95)}"
    )


def format_report(
    title: str, names: list[str], metrics: DetectionMetrics, score_threshold: float, iou: float
) -> str:
    lines = [
        title,
        f"cajas anotadas: {metrics.n_gt} | detecciones: {metrics.n_predictions} "
        f"(con score >= {score_threshold}: {metrics.n_above_threshold})",
        "",
        f"Punto de operación (score >= {score_threshold}, IoU >= {iou}): {format_line(metrics)}",
        "",
        "Por clase (recall al punto de operación, AP sobre toda la curva):",
        f"{'clase':<20}{'recall':>10}{f'AP@{MATCH_IOU:g}':>10}",
    ]
    lines += [
        f"{name:<20}{format_metric(metrics.recall_per_class.get(class_id)):>10}"
        f"{format_metric(metrics.ap_per_class.get(class_id)):>10}"
        for class_id, name in enumerate(names)
    ]
    return "\n".join(lines) + "\n"
