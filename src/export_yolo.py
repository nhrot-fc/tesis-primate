import argparse
import json
import logging

import cv2
from tqdm import tqdm

from core.config import settings
from core.runtime import setup_logging
from data import cache
from data.datasets import YOLODataset

logger = logging.getLogger("export_yolo")

# Ultralytics sólo admite `imgsz` cuadrado en train/val: exportar ya cuadrado evita
# que el letterbox rellene ~60% de la imagen con gris.
IMAGE_SIZE = 512
PNG_COMPRESSION = 1  # 0-9; con espectrogramas suaves comprimir más casi no baja el tamaño
NAME_SEPARATOR = "-"  # "aa/gc" -> "aa-gc": las barras rompen rutas derivadas del nombre


def export_split(split: str, db_low: float, db_high: float) -> dict[str, int]:
    dataset = YOLODataset(cache.split_path(split), db_low, db_high, IMAGE_SIZE)
    image_dir = settings.yolo_dir / "images" / split
    label_dir = settings.yolo_dir / "labels" / split
    for directory in (image_dir, label_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.glob("*"):
            stale.unlink()

    n_boxes = n_empty = 0
    for index in tqdm(range(len(dataset)), desc=f"exportando {split}", disable=None):
        image, lines = dataset[index]
        stem = f"{split}_{index:06d}"
        cv2.imwrite(
            str(image_dir / f"{stem}.png"), image, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]
        )
        (label_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n" if lines else "")
        n_boxes += len(lines)
        n_empty += not lines
    return {"windows": len(dataset), "boxes": n_boxes, "empty_windows": n_empty}


def write_dataset_yaml(names: list[str]) -> None:
    lines = [
        "# Generado por src/export_yolo.py -- no editar a mano.",
        f"path: {settings.yolo_dir}",
        *(f"{split}: images/{split}" for split in cache.SPLITS),
        "names:",
        *(f"  {class_id}: {name}" for class_id, name in enumerate(names)),
    ]
    (settings.yolo_dir / "dataset.yaml").write_text("\n".join(lines) + "\n")


def main() -> None:
    argparse.ArgumentParser(
        description="Exporta el caché de ventanas al layout de dataset de Ultralytics."
    ).parse_args()
    setup_logging()
    original_names = cache.labels().names
    db_low, db_high = cache.db_range()

    settings.yolo_dir.mkdir(parents=True, exist_ok=True)
    (settings.yolo_dir / "meta.json").unlink(missing_ok=True)
    counts = {split: export_split(split, db_low, db_high) for split in cache.SPLITS}
    logger.info("ventanas exportadas -> %s", counts)

    names = [name.replace("/", NAME_SEPARATOR) for name in original_names]
    write_dataset_yaml(names)
    (settings.yolo_dir / "meta.json").write_text(
        json.dumps(
            {
                "image_size": IMAGE_SIZE,
                "db_range": {"low": db_low, "high": db_high},
                "names": names,
                "names_original": original_names,
                "counts": counts,
                "dataset": cache.meta(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info("dataset YOLO -> %s", settings.yolo_dir / "dataset.yaml")


if __name__ == "__main__":
    main()
