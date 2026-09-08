import argparse
import json
import logging
from pathlib import Path

from ultralytics import YOLO

from core.config import SEED, WORKERS, settings
from core.runtime import setup_logging
from data import cache
from models.registry import architecture
from models.yolo import SpectrogramYOLO, load_ultralytics_weights
from training.checkpoint import BEST, save
from training.trainer import TrainConfig

logger = logging.getLogger("train_yolo")

MODEL = "yolo26s"  # n < s < m < l < x
EPOCHS, BATCH_SIZE, IMAGE_SIZE = 50, 32, 512
PATIENCE = 30
CLS_POWER_WEIGHT = 0.0  # 1.0 pondera las clases por frecuencia inversa

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
    parser = argparse.ArgumentParser(
        description="Entrena un YOLO de Ultralytics sobre los espectrogramas exportados."
    )
    parser.add_argument("--model", default=MODEL, help="p. ej. yolo26n, yolo26s, yolo26m")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--device", default=None, help="'0', '0,1', 'cpu'; por defecto, automático")
    parser.add_argument("--scratch", action="store_true", help="sin los pesos de COCO")
    parser.add_argument("--fraction", type=float, default=1.0, help="fracción de train (pruebas)")
    parser.add_argument("--name", default=None, help="nombre de la corrida dentro de runs/")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def check_export() -> dict:
    # El dataset de YOLO es una reexportación del caché: si el caché se regeneró, entrenar
    # contra el export viejo invalida la comparación.
    data_yaml = settings.yolo_dir / "dataset.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"no hay dataset YOLO en {settings.yolo_dir}. Corré `python src/export_yolo.py`."
        )
    meta = json.loads((settings.yolo_dir / "meta.json").read_text())
    if meta["dataset"] != cache.meta():
        raise ValueError(
            f"{settings.yolo_dir} se exportó de otra versión del caché. Volvé a correr "
            "`python src/export_yolo.py`."
        )
    return meta


def export_checkpoint(run_dir: Path, weights: Path, meta: dict, config: dict) -> Path:
    # Ultralytics guarda el modelo pickleado; el registro guarda arquitectura,
    # hiperparámetros y `state_dict`, que es lo que `dump_predictions.py` y el viewer abren.
    labels = cache.labels()
    hparams = {
        "model": Path(config["model"]).stem,
        "imgsz": meta["image_size"],
        "db_low": meta["db_range"]["low"],
        "db_high": meta["db_range"]["high"],
    }
    model = SpectrogramYOLO(n_classes=len(labels), **hparams)
    load_ultralytics_weights(model, weights)

    path = run_dir / BEST
    save(
        path,
        architecture="yolo",
        model=model,
        hparams=hparams,
        labels=labels,
        config=config,
        epoch=config["epochs"] - 1,
        metrics={},  # las de Ultralytics están en results.csv; las comparables, de compare_models.py
    )
    return path


def main() -> None:
    args = parse_args()
    name = args.name or f"{Path(args.model).stem}_{'scratch' if args.scratch else 'coco'}"
    setup_logging(log_file=settings.runs_dir / name / "train.log")
    meta = check_export()

    stem = Path(args.model).stem
    results = YOLO(f"{stem}.yaml" if args.scratch else f"{stem}.pt").train(
        data=str(settings.yolo_dir / "dataset.yaml"),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        fraction=args.fraction,
        project=str(settings.runs_dir),
        name=name,
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
    if results is None:
        raise RuntimeError("Ultralytics no devolvió resultados de entrenamiento.")

    run_dir = Path(results.save_dir)  # pyright: ignore[reportAttributeAccessIssue]
    # YOLO26 es end-to-end (sin NMS), pero el pos-proceso de cajas anidadas se le aplica
    # igual que a los demás para que el punto de operación sea el mismo.
    defaults = TrainConfig()
    config = {
        "architecture": "yolo",
        "model": args.model,
        "pretrained": not args.scratch,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "imgsz": args.imgsz,
        "seed": SEED,
        "cls_pw": CLS_POWER_WEIGHT,
        "augmentation": AUGMENTATION,
        "score_threshold": defaults.score_threshold,
        "nms_iou": architecture("yolo").nms_iou,
        "dataset": meta["dataset"],
    }
    checkpoint = export_checkpoint(run_dir, run_dir / "weights" / "best.pt", meta, config)
    logger.info("checkpoint -> %s", checkpoint)
    logger.info("volcá predicciones con: python src/dump_predictions.py --run %s", name)


if __name__ == "__main__":
    main()
