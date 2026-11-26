import logging
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn

from core.errors import CheckpointError, DependencyError, DeviceError
from core.runtime import resolve_device
from data.species import LabelSet
from inference.catalog import read_operating_point
from models.base import Detector
from models.coco_deformable_detr import ResNetDeformableDETR
from models.deformable_detr import ASTDeformableDETR
from models.faster_rcnn import SpectrogramFasterRCNN
from models.yolo import SpectrogramYOLO

logger = logging.getLogger(__name__)

ARCHITECTURES: dict[str, type[Detector]] = {
    "ast_deformable_detr": ASTDeformableDETR,
    "resnet_deformable_detr": ResNetDeformableDETR,
    "faster_rcnn": SpectrogramFasterRCNN,
    "yolo": SpectrogramYOLO,
}
EXTRAS = {"ast_deformable_detr": "detr", "resnet_deformable_detr": "detr", "yolo": "yolo"}
REQUIRED_KEYS = ("architecture", "hparams", "labels", "state_dict")  # training.checkpoint.save


def build_model(name: str, n_classes: int, hparams: dict[str, Any]) -> Detector:
    if name not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {name!r}; known: {sorted(ARCHITECTURES)}")
    try:
        return ARCHITECTURES[name](n_classes=n_classes, **hparams)
    except ModuleNotFoundError as exc:
        if name not in EXTRAS:
            raise
        raise DependencyError(
            f"{name!r} needs the `{EXTRAS[name]}` extra ({exc.name} is missing): "
            f"uv sync --extra {EXTRAS[name]}"
        ) from exc


def cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


class LoadedModel(NamedTuple):
    model: Detector
    architecture: str
    labels: LabelSet
    config: dict[str, Any]
    operating_point: float | None  # None: corrida sin comparar
    path: Path
    device: str  # donde vive el modelo; sus entradas van ahí


def not_a_model(path: Path) -> CheckpointError:
    return CheckpointError(
        f"'{path.name}' in {path.parent} is not a model of this program, or the file is "
        "damaged. Unzip the model again (detector-*-model-*.zip)."
    )


def read_checkpoint(path: Path | str) -> dict[str, Any]:
    # mmap: valida sin leer los pesos
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"'{path}' is not there.")
    try:
        stored = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception as exc:
        raise not_a_model(path) from exc
    if not isinstance(stored, dict) or any(key not in stored for key in REQUIRED_KEYS):
        raise not_a_model(path)
    if stored["architecture"] not in ARCHITECTURES:
        raise CheckpointError(
            f"'{path.name}' is a {stored['architecture']!r} model, which this version of the "
            "program does not know: use a newer version."
        )
    return stored


def load_checkpoint(path: Path | str, device: str | None = None) -> LoadedModel:
    # device None: la GPU si sirve, si no la CPU
    path = Path(path)
    stored = read_checkpoint(path)
    device = resolve_device(device)
    name, labels = stored["architecture"], LabelSet(stored["labels"])
    hparams = dict(stored["hparams"])
    if "pretrained" in hparams:
        hparams["pretrained"] = False  # los pesos salen del checkpoint: nada que bajar
    try:
        model = build_model(name, len(labels), hparams)
    except OSError as exc:  # falta hf/ y no hay internet para bajarlo
        raise DependencyError(
            f"{path.parent.name} needs files that are missing next to the program (hf\\): "
            "unzip its model zip again into the program folder."
        ) from exc
    try:
        model.load_state_dict(stored["state_dict"])
    except RuntimeError as exc:  # claves o formas de otra versión
        raise CheckpointError(
            f"'{path.name}' does not fit the {name} architecture of this version of the program."
        ) from exc
    try:
        model.to(device).eval()
    except torch.OutOfMemoryError as exc:
        raise DeviceError(
            f"The graphics card has no room for {path.parent.name}: close other programs "
            "that use it, or run on the cpu."
        ) from exc

    operating_point = read_operating_point(path.parent)
    logger.info(
        "%s | %d classes | epoch %s | operating point %s | %s | %s",
        name,
        len(labels),
        stored.get("epoch"),
        "none" if operating_point is None else f"{operating_point:.2f}",
        device,
        path,
    )
    return LoadedModel(model, name, labels, stored.get("config", {}), operating_point, path, device)
