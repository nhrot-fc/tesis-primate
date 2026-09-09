"""Copia un volcado crudo aplicándole supresión, sin volver a pasar por la GPU.

`dump_predictions.py` escribe las cajas de DETR y DINO tal como salen del modelo, porque
`models.registry` los tiene con `nms_iou=None`. Esto deriva de ese volcado uno suprimido,
para que `compare_models.py` mida las dos variantes en la misma tabla:

    python src/apply_nms.py --run detr_unfreeze_ts5 --nms-iou 0.3 --name ts5_nms03

Deja `<destino>/<nombre>_{val,test}_predictions.pt` con el mismo mapa ventana -> grabación,
que es lo que `protocol.check_comparable` exige para poder compararlos.
"""

import argparse
import logging
from pathlib import Path

from core.config import settings
from core.runtime import setup_logging
from evaluation.evaluator import TEST, VAL, RawPredictions, path_for
from sweep_nms import by_window, suppress
from utils.boxes import IOMIN_THRESHOLD

logger = logging.getLogger("apply_nms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="corrida en runs/ con los volcados crudos")
    parser.add_argument("--nms-iou", type=float, required=True, help="1.0 no suprime nada")
    parser.add_argument("--iomin", type=float, default=IOMIN_THRESHOLD)
    parser.add_argument("--name", required=True, help="con el que entra a la tabla")
    parser.add_argument("--output", type=Path, help="destino; por defecto, el de la corrida")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    source = settings.runs_dir / args.run
    destination = args.output or source

    for split in (VAL, TEST):
        dump = RawPredictions.load(path_for(source, args.run, split))
        if dump.nms_iou is not None:
            raise ValueError(f"{args.run} {split} ya trae NMS a {dump.nms_iou}")

        boxes = suppress(by_window(dump.predictions), args.nms_iou, args.iomin)
        derived = dump._replace(model=args.name, predictions=boxes, nms_iou=args.nms_iou)
        path = derived.save(path_for(destination, args.name, split))
        logger.info(
            "%s %s: %d -> %d cajas (%.1f por ventana) -> %s",
            args.name,
            split,
            len(dump.predictions.boxes),
            len(boxes.boxes),
            derived.detections_per_window,
            path,
        )


if __name__ == "__main__":
    main()
