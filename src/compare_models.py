import argparse
import json
import logging
from pathlib import Path

from core.runtime import setup_logging
from evaluation.evaluator import load_pairs
from evaluation.protocol import MAX_DETECTIONS, MIN_PRECISIONS, N_BOOTSTRAP, as_json, compare
from evaluation.report import format_comparison

logger = logging.getLogger("compare_models")

OUTPUT = "comparacion_modelos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara volcados de `dump_predictions.py`: mismo tope de detecciones para "
        "todos, umbral elegido en val, test medido una vez."
    )
    parser.add_argument("dumps", nargs="+", type=Path, help="los *_predictions.pt, val y test")
    parser.add_argument("--output", type=Path, help="sin extensión; se escriben .txt y .json")
    parser.add_argument("--max-det", type=int, default=MAX_DETECTIONS, help="tope por ventana")
    parser.add_argument(
        "--precision",
        nargs="*",
        type=float,
        default=list(MIN_PRECISIONS),
        help="precisión mínima en val",
    )
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP, help="remuestreos del IC")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    comparison = compare(
        load_pairs(args.dumps), tuple(args.precision), max_det=args.max_det, n_boot=args.bootstrap
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
