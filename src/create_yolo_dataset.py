"""Exporta el caché de ventanas al layout de dataset de Ultralytics.

Lee `data/processed/{train,val,test}.pt` --las mismas ventanas y el mismo split que
consume el Deformable-DETR-- y escribe PNG de 8 bits con etiquetas `class cx cy w h`.
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from core.config import settings
from core.setup import setup_logging, setup_project_path
from utils.audio import DB_PERCENTILES, mel_db_range, mel_to_unit

logger = logging.getLogger("create_yolo_dataset")

PROJECT_DIR = Path.cwd()
CACHE_DIR = PROJECT_DIR / "data" / "processed"
YOLO_DIR = PROJECT_DIR / "data" / "yolo"

SPLITS = ("train", "val", "test")
# Ultralytics sólo admite `imgsz` cuadrado en train/val: exportar ya cuadrado evita
# que el letterbox rellene ~60% de la imagen con gris.
IMAGE_SIZE = 512
PNG_COMPRESSION = 1  # 0-9; con espectrogramas suaves comprimir más casi no baja el tamaño
NAME_SEPARATOR = "-"  # "aa/gc" -> "aa-gc": las barras rompen rutas derivadas del nombre


def to_image(mel: torch.Tensor, low: float, high: float) -> np.ndarray:
    """Mel de potencia (1, n_mels, T) -> PNG en gris, grave abajo y agudo arriba.

    `architectures.yolo` rehace esta misma conversión en memoria al inferir, así que
    las dos tienen que salir de `utils.audio`.
    """
    gray = (mel_to_unit(mel[0], low, high) * 255).to(torch.uint8).numpy()
    image = np.flipud(gray)  # la fila 0 de un PNG es la de arriba
    if IMAGE_SIZE:
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return image


def to_label_lines(boxes: torch.Tensor, labels: torch.Tensor) -> list[str]:
    """`cx cy w h` del dataset -> renglones YOLO, con la `y` ya invertida como la imagen."""
    lines = []
    for (cx, cy, w, h), class_id in zip(boxes.tolist(), labels.tolist(), strict=True):
        cy = 1.0 - cy
        values = [min(max(v, 0.0), 1.0) for v in (cx, cy, w, h)]
        if values[2] <= 0.0 or values[3] <= 0.0:
            continue
        lines.append(f"{int(class_id)} " + " ".join(f"{v:.6f}" for v in values))
    return lines


def export_split(name: str, cache: dict, low: float, high: float) -> dict[str, int]:
    image_dir = YOLO_DIR / "images" / name
    label_dir = YOLO_DIR / "labels" / name
    for directory in (image_dir, label_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.glob("*"):
            stale.unlink()

    images: torch.Tensor = cache["images"]
    n_boxes = n_empty = 0
    for index in tqdm(range(len(images)), desc=f"exportando {name}"):
        stem = f"{name}_{index:06d}"
        cv2.imwrite(
            str(image_dir / f"{stem}.png"),
            to_image(images[index], low, high),
            [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
        )
        lines = to_label_lines(cache["boxes"][index], cache["labels"][index])
        (label_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n" if lines else "")
        n_boxes += len(lines)
        n_empty += not lines
    return {"windows": len(images), "boxes": n_boxes, "empty_windows": n_empty}


def write_dataset_yaml(names: list[str]) -> Path:
    path = YOLO_DIR / "dataset.yaml"
    lines = [
        "# Generado por src/create_yolo_dataset.py -- no editar a mano.",
        f"path: {YOLO_DIR}",
        *(f"{split}: images/{split}" for split in SPLITS),
        "names:",
        *(f"  {class_id}: {name}" for class_id, name in enumerate(names)),
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    setup_logging(settings.LOG_LEVEL)
    setup_project_path(PROJECT_DIR)

    if not (CACHE_DIR / "meta.json").exists():
        raise FileNotFoundError(
            f"no hay dataset cacheado en {CACHE_DIR}. Corré `python src/create_dataset.py` primero."
        )
    dataset_meta = json.loads((CACHE_DIR / "meta.json").read_text())
    original_names = list(json.loads((CACHE_DIR / "labels.json").read_text()).values())
    names = [name.replace("/", NAME_SEPARATOR) for name in original_names]

    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / "meta.json").unlink(missing_ok=True)

    low = high = 0.0
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        cache = torch.load(CACHE_DIR / f"{split}.pt", weights_only=False)
        if split == "train":
            low, high = mel_db_range(cache["images"][:, 0])
            logger.info(
                "rango de dB (percentiles %s de train) -> [%.2f, %.2f]",
                DB_PERCENTILES,
                low,
                high,
            )
        counts[split] = export_split(split, cache, low, high)
        logger.info("%s -> %s", split, counts[split])
        del cache

    yaml_path = write_dataset_yaml(names)
    (YOLO_DIR / "meta.json").write_text(
        json.dumps(
            {
                "image_size": IMAGE_SIZE,
                "db_range": {"low": low, "high": high, "percentiles": DB_PERCENTILES},
                "names": names,
                "names_original": original_names,
                "counts": counts,
                "source_cache": str(CACHE_DIR),
                "dataset": dataset_meta,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info("dataset YOLO -> %s", yaml_path)


if __name__ == "__main__":
    main()
