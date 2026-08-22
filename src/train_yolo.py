"""Entrena un detector YOLO (Ultralytics) sobre los espectrogramas exportados.

El dataset lo produce `src/create_yolo_dataset.py` desde el mismo caché de ventanas que
usa el Deformable-DETR, así que los dos modelos se comparan directamente.
"""

import argparse
import json
import logging
from pathlib import Path

from ultralytics import YOLO

from architectures.registry import save_checkpoint
from architectures.yolo import SpectrogramYOLO, load_ultralytics_weights
from core.config import settings
from core.setup import setup_logging, setup_project_path
from domain.species import LabelSet
from train import CACHE_DIR, SEED

logger = logging.getLogger("train_yolo")

PROJECT_DIR = Path.cwd()
YOLO_DIR = PROJECT_DIR / "data" / "yolo"
RUNS_DIR = PROJECT_DIR / "runs" / "yolo"

MODEL = "yolo26s"  # n < s < m < l < x
EPOCHS, BATCH_SIZE, IMAGE_SIZE, WORKERS = 100, 32, 512, 8
PATIENCE = 30
CLS_POWER_WEIGHT = 0.0
# Mismo punto de operación que `src/train.py`, para que los informes se comparen.
OPERATING_SCORE_THRESHOLD = 0.5
NMS_IOU = 0.3  # 1.0 pondera las clases por frecuencia inversa

# Aumentaciones: espectrograma, no foto.
AUGMENTATION = {
    "hsv_h": 0.0,  # el gris no tiene tono ni saturación
    "hsv_s": 0.0,
    "hsv_v": 0.2,  # brillo == ganancia de grabación: la única que sí varía en el campo
    "degrees": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "translate": 0.05,  # corrimiento chico en tiempo y frecuencia
    "scale": 0.2,
    "fliplr": 0.0,  # invertir el tiempo cambia la llamada
    "flipud": 0.0,  # invertir la frecuencia también
    "bgr": 0.0,
    "mosaic": 0.0,  # pega 4 clips: crea bordes y solapes que no existen en el audio
    "close_mosaic": 0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena YOLO sobre los espectrogramas.")
    parser.add_argument("--model", default=MODEL, help="p. ej. yolo26n, yolo26s, yolo26m")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--device", default=None, help="'0', '0,1', 'cpu'; por defecto, automático")
    parser.add_argument(
        "--scratch", action="store_true", help="entrena desde cero, sin los pesos de COCO"
    )
    parser.add_argument(
        "--fraction", type=float, default=1.0, help="fracción de train (para pruebas rápidas)"
    )
    parser.add_argument("--name", default=None, help="nombre de la corrida dentro de runs/yolo")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def export_checkpoint(save_dir: Path, args: argparse.Namespace, meta: dict, config: dict) -> Path:
    """Reempaqueta el `best.pt` de Ultralytics al formato del registro.

    Ultralytics guarda el modelo pickleado; el registro guarda arquitectura,
    hiperparámetros y `state_dict`, que es lo que `src/viewer/` sabe abrir.
    """
    labels = LabelSet(meta["names_original"])
    hparams = {
        "model": Path(args.model).stem,
        "imgsz": meta["image_size"],
        "db_low": meta["db_range"]["low"],
        "db_high": meta["db_range"]["high"],
    }
    model = SpectrogramYOLO(n_classes=len(labels), **hparams)
    load_ultralytics_weights(model, save_dir / "weights" / "best.pt")

    path = save_dir / "weights" / "best_spectrogram.pth"
    save_checkpoint(
        path,
        architecture="yolo",
        model=model,
        hparams=hparams,
        labels=labels,
        config=config,
    )
    return path


def train(args: argparse.Namespace) -> Path:
    data_yaml = YOLO_DIR / "dataset.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"no hay dataset YOLO en {YOLO_DIR}. Corré `python src/create_yolo_dataset.py` primero."
        )

    # El dataset de YOLO es una reexportación del caché: si el caché se regeneró (otras
    # clases, otro split), entrenar contra el export viejo invalida la comparación.
    meta = json.loads((YOLO_DIR / "meta.json").read_text())
    if meta["dataset"] != json.loads((CACHE_DIR / "meta.json").read_text()):
        raise ValueError(
            f"{YOLO_DIR} se exportó de otra versión de {CACHE_DIR}. Volvé a correr "
            "`python src/create_yolo_dataset.py`."
        )

    stem = Path(args.model).stem
    model = YOLO(f"{stem}.yaml" if args.scratch else f"{stem}.pt")
    run_name = args.name or f"{Path(args.model).stem}_{'scratch' if args.scratch else 'coco'}"
    logger.info(
        "%s | %s | %d épocas | batch %d | imgsz %d",
        args.model,
        "desde cero" if args.scratch else "preentrenado en COCO",
        args.epochs,
        args.batch,
        args.imgsz,
    )

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        fraction=args.fraction,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,
        resume=args.resume,
        pretrained=not args.scratch,
        seed=SEED,
        deterministic=True,
        patience=PATIENCE,
        optimizer="auto",
        cos_lr=True,
        cls_pw=CLS_POWER_WEIGHT,
        val=True,
        plots=True,
        **AUGMENTATION,
    )

    save_dir = Path(results.save_dir)
    config = {
        "architecture": "yolo",
        "model": args.model,
        "pretrained": not args.scratch,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "seed": SEED,
        "cls_pw": CLS_POWER_WEIGHT,
        "augmentation": AUGMENTATION,
        # YOLO26 es end-to-end (sin NMS), pero el pos-proceso de cajas anidadas de
        # `pipelines.detection_pipeline` se aplica igual que a los demás detectores.
        "operating_score_threshold": OPERATING_SCORE_THRESHOLD,
        "nms_iou": NMS_IOU,
        "dataset": meta,
    }
    (save_dir / "spectrogram_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False)
    )
    logger.info("pesos de Ultralytics -> %s", save_dir / "weights" / "best.pt")
    logger.info("checkpoint del registro -> %s", export_checkpoint(save_dir, args, meta, config))
    return save_dir


def main() -> None:
    setup_logging(settings.LOG_LEVEL)
    setup_project_path(PROJECT_DIR)
    save_dir = train(parse_args())
    logger.info(
        "evaluá con: python src/eval_detector.py --checkpoint %s --split test",
        save_dir / "weights" / "best_spectrogram.pth",
    )


if __name__ == "__main__":
    main()
