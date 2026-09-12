"""Arma el paquete portable de Windows: Python embebido + librerías + código + modelos.

    dist/detector-primates-<versión>-win64-<cpu|cuda>/
      Visor.bat, Detectar.bat, LEEME.txt
      python/    intérprete embebido de python.org, Lib/site-packages, DLLs de MSVC
      src/       este repo, tal cual (core.config resuelve hf/ y models/ desde acá)
      hf/        backbones de Hugging Face que reconstruyen los modelos incluidos
      models/    <nombre>/<checkpoint> + operating_point.json

Corre en Linux: uv resuelve e instala los wheels de Windows sin ejecutarlos. El usuario
descomprime y hace doble clic; no instala nada.

    uv run python deploy/build_windows.py runs/frcnn runs/yolo26s_coco
    uv run python deploy/build_windows.py --variant cuda runs/frcnn
"""

import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tomllib
import urllib.request
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
TEMPLATES = Path(__file__).resolve().parent / "windows"
BUILD_DIR = PROJECT_DIR / "build"  # descargas reutilizables
DIST_DIR = PROJECT_DIR / "dist"

APP = "detector-primates"
# Último 3.12 con binarios en python.org (los siguientes son sólo código fuente)
PYTHON_VERSION = "3.12.10"
EMBED_URL = "https://www.python.org/ftp/python/{v}/python-{v}-embed-amd64.zip"
PLATFORM = "x86_64-pc-windows-msvc"
# Índice de torch por variante: cu128 corre con drivers NVIDIA desde la serie 527
TORCH_BACKENDS = {"cpu": "cpu", "cuda": "cu128"}
UV = os.environ.get("UV", "uv")
# GitHub admite hasta 2 GiB por archivo en un release
MAX_PART_MB = 1900
# Dentro de site-packages: cabeceras, librerías de enlace y pruebas que la inferencia no usa
PRUNE = ["torch/include", "torch/lib/*.lib", "torch/bin", "torch/share", "pandas/tests"]

sys.path.insert(0, str(PROJECT_DIR / "src"))
from models.backbone import local_ast_dir  # noqa: E402
from models.coco_deformable_detr import local_detr_dir  # noqa: E402

# Qué copia de `hf/` reconstruye cada arquitectura al cargar su checkpoint
HF_BY_ARCHITECTURE = {
    "ast_deformable_detr": [local_ast_dir()],
    "coco_deformable_detr": [local_ast_dir(), local_detr_dir()],
}

logger = logging.getLogger("build")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "models", nargs="+", type=Path, metavar="MODELO", help="checkpoint o carpeta de runs/"
    )
    parser.add_argument("--variant", choices=tuple(TORCH_BACKENDS), default="cpu")
    parser.add_argument("--version", default=project_version(), help="va en el nombre del zip")
    parser.add_argument("--no-zip", action="store_true", help="deja sólo la carpeta en dist/")
    return parser.parse_args()


def project_version() -> str:
    with open(PROJECT_DIR / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def run(*command: str | Path) -> None:
    logger.info("$ %s", " ".join(str(c) for c in command))
    subprocess.run([str(c) for c in command], check=True)


# --- Modelos ----------------------------------------------------------------------


def checkpoint_of(path: Path) -> Path:
    if path.is_file():
        return path
    best = path / "best.pt"
    if best.is_file():
        return best
    found = [
        p
        for p in path.iterdir()
        if p.suffix in (".pt", ".pth") and not p.name.endswith("_predictions.pt")
    ]
    if len(found) != 1:
        raise SystemExit(f"{path}: esperaba un checkpoint, hay {len(found)}")
    return found[0]


def architecture_of(checkpoint: Path) -> str:
    import torch

    # mmap: sólo se leen los metadatos, no los pesos
    stored = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(stored, dict) or "architecture" not in stored:
        raise SystemExit(f"{checkpoint} no es un checkpoint del proyecto")
    return stored["architecture"]


def copy_models(sources: list[Path], stage: Path) -> set[Path]:
    hf_dirs: set[Path] = set()
    for source in sources:
        checkpoint = checkpoint_of(source)
        architecture = architecture_of(checkpoint)
        target = stage / "models" / checkpoint.parent.name
        target.mkdir(parents=True)
        shutil.copy2(checkpoint, target / checkpoint.name)
        operating_point = checkpoint.parent / "operating_point.json"
        if operating_point.is_file():
            shutil.copy2(operating_point, target / operating_point.name)
        logger.info("modelo %s: %s (%s)", target.name, architecture, checkpoint.name)
        hf_dirs.update(HF_BY_ARCHITECTURE.get(architecture, []))
    return hf_dirs


def copy_hf(hf_dirs: set[Path], stage: Path) -> None:
    for source in sorted(hf_dirs):
        if not source.is_dir():
            raise SystemExit(f"Falta {source}: cargá el modelo una vez para que se descargue")
        shutil.copytree(source, stage / "hf" / source.name)
        logger.info("hf/%s", source.name)


# --- Python y librerías ----------------------------------------------------------------


def download(url: str, target: Path) -> Path:
    if not target.is_file():
        logger.info("descargando %s", url)
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, target)
    return target


def install_python(stage: Path) -> Path:
    python_dir = stage / "python"
    archive = download(
        EMBED_URL.format(v=PYTHON_VERSION), BUILD_DIR / f"python-{PYTHON_VERSION}.zip"
    )
    with zipfile.ZipFile(archive) as z:
        z.extractall(python_dir)
    # El ._pth fija el sys.path relativo a esta carpeta e ignora PYTHONPATH y el registro.
    major_minor = "".join(PYTHON_VERSION.split(".")[:2])
    (python_dir / f"python{major_minor}._pth").write_text(
        f"python{major_minor}.zip\n.\nLib\\site-packages\n..\\src\nimport site\n"
    )
    return python_dir


def install_packages(python_dir: Path, variant: str) -> None:
    site = python_dir / "Lib" / "site-packages"
    requirements = python_dir / "requirements.txt"
    common = [
        f"--python-platform={PLATFORM}",
        f"--python-version={'.'.join(PYTHON_VERSION.split('.')[:2])}",
        f"--torch-backend={TORCH_BACKENDS[variant]}",
    ]
    # Se deja resuelto en el paquete: dice qué versiones lleva exactamente.
    run(
        UV, "pip", "compile", "--quiet", "--no-header",
        PROJECT_DIR / "pyproject.toml", TEMPLATES / "extra-requirements.in",
        "--extra", "detr", "--extra", "yolo", *common, "-o", requirements,
    )  # fmt: skip
    run(
        UV, "pip", "install", "--quiet", "--target", site, "--compile-bytecode",
        "--python", sys.executable, *common, "-r", requirements,
    )  # fmt: skip
    # Las DLLs de MSVC (msvc-runtime) van junto a python.exe, donde Windows las busca.
    for dll in site.glob("*.dll"):
        shutil.move(dll, python_dir / dll.name)
    for pattern in PRUNE:
        for path in site.glob(pattern):
            shutil.rmtree(path) if path.is_dir() else path.unlink()


# --- Código y lanzadores -----------------------------------------------------------


def copy_source(stage: Path) -> None:
    shutil.copytree(
        PROJECT_DIR / "src",
        stage / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def copy_launchers(stage: Path, version: str, variant: str) -> None:
    for template in TEMPLATES.iterdir():
        if template.suffix == ".in":
            continue
        text = template.read_text(encoding="utf-8").replace("{version}", version)
        text = text.replace("{variant}", variant)
        target = stage / ("python" if template.name == "entorno.bat" else "") / template.name
        # cmd.exe quiere CRLF; el repo guarda LF
        target.write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))


# --- Zip, partes y sumas -----------------------------------------------------------


def make_zip(stage: Path) -> Path:
    archive = stage.with_name(f"{stage.name}.zip")
    logger.info("comprimiendo %s", archive.name)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(stage.parent))
    return archive


def split(archive: Path, max_mb: int = MAX_PART_MB) -> list[Path]:
    size = archive.stat().st_size
    if size <= max_mb * 2**20:
        return [archive]
    parts = []
    with open(archive, "rb") as source:
        while chunk := source.read(max_mb * 2**20):
            part = archive.with_name(f"{archive.name}.{len(parts) + 1:03d}")
            part.write_bytes(chunk)
            parts.append(part)
    archive.unlink()
    # Sin 7-Zip, `copy /b` vuelve a juntar las partes
    joiner = archive.with_suffix(".unir.bat")
    joined = "+".join(f'"{p.name}"' for p in parts)
    joiner.write_bytes(
        f'@echo off\r\ncopy /b {joined} "{archive.name}"\r\n'
        f"echo Listo: extrae {archive.name}\r\npause\r\n".encode()
    )
    logger.info("%s supera %d MB: %d partes + %s", archive.name, max_mb, len(parts), joiner.name)
    return [*parts, joiner]


def write_checksums(files: list[Path]) -> Path:
    sums = DIST_DIR / "SHA256SUMS.txt"
    lines = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}" for f in files]
    sums.write_text("\n".join(lines) + "\n")
    return sums


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    name = f"{APP}-{args.version}-win64-{args.variant}"
    stage = DIST_DIR / name
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    hf_dirs = copy_models(args.models, stage)
    copy_hf(hf_dirs, stage)
    copy_source(stage)
    python_dir = install_python(stage)
    install_packages(python_dir, args.variant)
    copy_launchers(stage, args.version, args.variant)

    total = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
    logger.info("%s: %.2f GB descomprimido", stage.name, total / 1e9)
    if args.no_zip:
        return
    files = split(make_zip(stage))
    files.append(write_checksums(files))
    for f in files:
        logger.info("  %s  %.0f MB", f.name, f.stat().st_size / 2**20)
    logger.info("Publicar: gh release create v%s %s", args.version, " ".join(str(f) for f in files))


if __name__ == "__main__":
    main()
