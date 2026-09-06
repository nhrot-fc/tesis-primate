import argparse
import json
import logging
from pathlib import Path

from core.runtime import setup_logging
from evaluation.evaluator import load_pairs
from evaluation.metrics import BETA
from evaluation.protocol import EQUALIZED_MAX_DET, N_BOOTSTRAP, Criterion, as_json, compare
from evaluation.report import format_comparison

logger = logging.getLogger("compare_models")

OUTPUT = "comparacion_modelos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reporte de evaluación a partir de los volcados de `dump_predictions.py`: mismo "
            "presupuesto de detecciones para todos, umbral elegido en val, test medido una "
            "vez. Con un solo modelo sale el mismo reporte, sin nadie con quien compararlo."
        )
    )
    parser.add_argument("dumps", nargs="+", type=Path, help="los *_predictions.pt, val y test")
    parser.add_argument("--output", type=Path, help="sin extensión; se escriben .txt y .json")
    parser.add_argument("--max-det", type=int, default=EQUALIZED_MAX_DET, help="tope común")
    parser.add_argument("--beta", type=float, default=BETA, help="la F-beta del último criterio")
    parser.add_argument(
        "--precision", nargs="*", type=float, default=[0.70, 0.50], help="puntos a precisión fija"
    )
    parser.add_argument(
        "--fp-per-hour", nargs="*", type=float, default=[100.0], help="puntos a ruido fijo"
    )
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP, help="remuestreos del IC")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    models = load_pairs(args.dumps)
    comparison = compare(
        models,
        criteria=(
            *(Criterion("precision", value) for value in args.precision),
            *(Criterion("fp_per_hour", value) for value in args.fp_per_hour),
            Criterion("f_beta", args.beta),
        ),
        max_det=args.max_det,
        beta=args.beta,
        n_boot=args.bootstrap,
    )

    report = format_comparison(comparison)
    output = args.output or args.dumps[0].parent / OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".txt").write_text(report)
    output.with_suffix(".json").write_text(
        json.dumps(as_json(comparison), indent=2, ensure_ascii=False)
    )
    logger.info("\n%s", report)
    logger.info("reporte -> %s", output.with_suffix(".txt"))


if __name__ == "__main__":
    main()
