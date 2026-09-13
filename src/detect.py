import argparse
import logging
import sys
from pathlib import Path

from core.config import SCORE_THRESHOLD
from core.runtime import resolve_device, setup_logging
from inference.catalog import (
    DETECTIONS_SUFFIX,
    MODELS_DIR,
    available_models,
    checkpoint_in,
    collect_audio,
    is_checkpoint,
)

logger = logging.getLogger("detect")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a model over audio files or folders and write a Raven selection "
        "table next to each recording.",
        epilog="e.g. detect.py --model models/frcnn recordings/ other.wav",
    )
    parser.add_argument("paths", nargs="+", type=Path, metavar="PATH", help="audio or folder")
    parser.add_argument(
        "--model", type=Path, help=f"checkpoint or its folder; by default chosen from {MODELS_DIR}"
    )
    parser.add_argument(
        "--score",
        type=float,
        help="minimum score; by default the model's operating point, or "
        f"{SCORE_THRESHOLD} if it has none",
    )
    parser.add_argument("--device", help="'cuda', 'cuda:1', 'cpu'; by default cuda if available")
    parser.add_argument("--overwrite", action="store_true", help="redo tables that already exist")
    return parser.parse_args()


def choose_model(requested: Path | None) -> Path:
    if requested is not None:
        chosen = checkpoint_in(requested) if requested.is_dir() else requested
        if chosen is None or not is_checkpoint(chosen):
            raise SystemExit(f"No checkpoint found in {requested}")
        return chosen

    models = available_models()
    if not models:
        raise SystemExit(f"No models in {MODELS_DIR}; pass one with --model")
    if len(models) == 1:
        return models[0]
    print("Available models:")
    for i, path in enumerate(models, 1):
        print(f"  {i}) {path.parent.name}  ({path.name})")
    if not sys.stdin.isatty():
        raise SystemExit("Several models available: choose one with --model")
    while True:
        answer = input("Model [1]: ").strip() or "1"
        if answer.isdigit() and 1 <= int(answer) <= len(models):
            return models[int(answer) - 1]


def main() -> None:
    args = parse_args()
    setup_logging()
    checkpoint = choose_model(args.model)
    for path in args.paths:
        if not path.is_dir() and not path.is_file():
            logger.warning("Skipping %s: not a file or folder", path)
    audio = collect_audio(args.paths)
    if not audio:
        raise SystemExit("No audio files to process.")

    # torch y el resto se importan acá para que un error de argumentos no espere a torch.
    print("Loading detection engine...", flush=True)
    from inference.batch import DONE, FAILED, SKIPPED, STOPPED, run_batch
    from models.registry import load_checkpoint

    device = resolve_device(args.device)
    loaded = load_checkpoint(checkpoint, device)
    threshold = args.score
    if threshold is None:
        threshold = SCORE_THRESHOLD if loaded.operating_point is None else loaded.operating_point
    logger.info(
        "%s on %s | score >= %.2f | %d recordings",
        checkpoint.parent.name,
        device,
        threshold,
        len(audio),
    )

    failed = 0
    total = len(audio)
    for outcome in run_batch(loaded, audio, device, threshold, overwrite=args.overwrite):
        where = f"[{outcome.index + 1}/{total}] {outcome.path.name}"
        if outcome.status == SKIPPED:
            logger.info("%s already has a table; --overwrite redoes it", where)
        elif outcome.status == FAILED:
            failed += 1
            logger.error("%s: %s", where, outcome.message)
        elif outcome.status == DONE:
            logger.info(
                "%s: %d detections in %.0f s -> %s",
                where,
                outcome.detections,
                outcome.seconds,
                outcome.path.stem + DETECTIONS_SUFFIX,
            )
        elif outcome.status == STOPPED:
            break

    if failed:
        raise SystemExit(f"{failed} of {total} recordings failed.")
    logger.info("Done. Open each audio in Raven with its %s table", DETECTIONS_SUFFIX)


if __name__ == "__main__":
    main()
