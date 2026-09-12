import argparse
import logging
import sys
import time
from pathlib import Path

from core.config import PROJECT_DIR, SCORE_THRESHOLD
from core.runtime import resolve_device, setup_logging

logger = logging.getLogger("detect")

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3"}
CHECKPOINT_SUFFIXES = {".pt", ".pth"}
# Donde el paquete portable deja los checkpoints: `models/<nombre>/best.pt` (+ operating_point.json)
MODELS_DIR = PROJECT_DIR / "models"
# La tabla queda junto al audio con el mismo nombre que exporta el visor; `<audio>.txt` a
# secas es la anotación de Raven, que el visor carga como capa aparte.
DETECTIONS_SUFFIX = ".detections.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pasa un modelo por archivos o carpetas de audio y deja una tabla de "
        "selección de Raven junto a cada grabación.",
        epilog="p. ej. detect.py --model models/frcnn grabaciones/ otra.wav",
    )
    parser.add_argument("paths", nargs="+", type=Path, metavar="RUTA", help="audio o carpeta")
    parser.add_argument(
        "--model", type=Path, help=f"checkpoint o su carpeta; por defecto se elige en {MODELS_DIR}"
    )
    parser.add_argument(
        "--score",
        type=float,
        help="score mínimo; por defecto el punto de operación del modelo o "
        f"{SCORE_THRESHOLD} si no lo tiene",
    )
    parser.add_argument("--device", help="'cuda', 'cuda:1', 'cpu'; por defecto cuda si hay")
    parser.add_argument("--overwrite", action="store_true", help="rehace tablas que ya existen")
    return parser.parse_args()


def is_checkpoint(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in CHECKPOINT_SUFFIXES
        and not path.name.endswith("_predictions.pt")
    )


# Un checkpoint por carpeta: `best.pt` si está, si no el único que haya.
def checkpoint_in(directory: Path) -> Path | None:
    best = directory / "best.pt"
    if best.is_file():
        return best
    found = [p for p in directory.iterdir() if is_checkpoint(p)]
    return found[0] if len(found) == 1 else None


def available_models() -> list[Path]:
    if not MODELS_DIR.is_dir():
        return []
    found = (checkpoint_in(d) for d in sorted(MODELS_DIR.iterdir()) if d.is_dir())
    return [p for p in found if p is not None]


def choose_model(requested: Path | None) -> Path:
    if requested is not None:
        chosen = checkpoint_in(requested) if requested.is_dir() else requested
        if chosen is None or not is_checkpoint(chosen):
            raise SystemExit(f"No encuentro un checkpoint en {requested}")
        return chosen

    models = available_models()
    if not models:
        raise SystemExit(f"No hay modelos en {MODELS_DIR}; indica uno con --model")
    if len(models) == 1:
        return models[0]
    print("Modelos disponibles:")
    for i, path in enumerate(models, 1):
        print(f"  {i}) {path.parent.name}  ({path.name})")
    if not sys.stdin.isatty():
        raise SystemExit("Hay varios modelos: elige uno con --model")
    while True:
        answer = input("Modelo [1]: ").strip() or "1"
        if answer.isdigit() and 1 <= int(answer) <= len(models):
            return models[int(answer) - 1]


def collect_audio(paths: list[Path]) -> list[Path]:
    audio: list[Path] = []
    for path in paths:
        if path.is_dir():
            audio.extend(
                p
                for p in sorted(path.rglob("*"))
                if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
            audio.append(path)
        else:
            logger.warning("Ignoro %s: no es audio ni carpeta", path)
    return list(dict.fromkeys(audio))


def main() -> None:
    args = parse_args()
    setup_logging()
    checkpoint = choose_model(args.model)
    audio = collect_audio(args.paths)
    if not audio:
        raise SystemExit("No hay audios que procesar.")

    # torch y el resto se importan acá para que un error de argumentos no espere a torch.
    from inference.predictor import predict
    from models.registry import load_checkpoint

    device = resolve_device(args.device)
    loaded = load_checkpoint(checkpoint, device)
    threshold = args.score
    if threshold is None:
        threshold = SCORE_THRESHOLD if loaded.operating_point is None else loaded.operating_point
    logger.info(
        "%s en %s | score >= %.2f | %d grabaciones",
        checkpoint.parent.name,
        device,
        threshold,
        len(audio),
    )

    failed = 0
    for i, path in enumerate(audio, 1):
        output = path.with_name(f"{path.stem}{DETECTIONS_SUFFIX}")
        if output.is_file() and not args.overwrite:
            logger.info(
                "[%d/%d] %s ya tiene tabla; --overwrite la rehace", i, len(audio), path.name
            )
            continue
        started = time.perf_counter()
        try:
            table = predict(loaded, path, device, score_threshold=threshold)
            table.to_csv(output, sep="\t", index=False)
        except Exception as exc:
            failed += 1
            logger.error("[%d/%d] %s: %s: %s", i, len(audio), path.name, type(exc).__name__, exc)
            continue
        logger.info(
            "[%d/%d] %s: %d detecciones en %.0f s -> %s",
            i,
            len(audio),
            path.name,
            len(table),
            time.perf_counter() - started,
            output.name,
        )

    if failed:
        raise SystemExit(f"{failed} de {len(audio)} grabaciones fallaron.")
    logger.info("Listo. Abre cada audio en Raven con su tabla %s", DETECTIONS_SUFFIX)


if __name__ == "__main__":
    main()
