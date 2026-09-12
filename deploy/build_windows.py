"""Arma los zips del release de Windows: el runtime por un lado y cada modelo por otro.

    runtime  detector-<versión>-win64-<cpu|cuda>.zip   (carpeta raíz del mismo nombre)
               Visor.bat, Detectar.bat, Agregar-modelo.bat, LEEME.txt
               python/   intérprete embebido de python.org, Lib/site-packages, DLLs de MSVC
               src/      este repo, tal cual (core.config resuelve hf/ y models/ desde acá)
               models/   vacía: acá van los modelos
    models   detector-<versión>-model-<corrida>.zip   (sin carpeta raíz)
               models/<corrida>/<checkpoint> + operating_point.json
               hf/…      los backbones de Hugging Face que esa arquitectura reconstruye

El zip de un modelo se descomprime dentro de la carpeta del runtime (`Agregar-modelo.bat` lo
hace). Corre en Linux: uv resuelve e instala los wheels de Windows sin ejecutarlos.

    uv run python deploy/build_windows.py runtime --variant cpu
    uv run python deploy/build_windows.py models runs/frcnn runs/yolo26s_coco
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

APP = "detector"
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
    parser.add_argument("--version", default=project_version(), help="va en el nombre del zip")
    parser.add_argument("--no-zip", action="store_true", help="deja sólo la carpeta en dist/")
    what = parser.add_subparsers(dest="what", required=True)
    runtime = what.add_parser("runtime", help="Python, librerías, código y lanzadores")
    runtime.add_argument("--variant", choices=tuple(TORCH_BACKENDS), default="cpu")
    models = what.add_parser("models", help="un zip por modelo, con sus backbones")
    models.add_argument(
        "models", nargs="+", type=Path, metavar="MODELO", help="checkpoint o carpeta de runs/"
    )
    return parser.parse_args()


def project_version() -> str:
    with open(PROJECT_DIR / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["version"]


def run(*command: str | Path) -> None:
    logger.info("$ %s", " ".join(str(c) for c in command))
    subprocess.run([str(c) for c in command], check=True)


def fresh(name: str) -> Path:
    stage = DIST_DIR / name
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    return stage


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


def stage_model(source: Path, version: str) -> Path:
    checkpoint = checkpoint_of(source)
    architecture = architecture_of(checkpoint)
    name = checkpoint.parent.name
    stage = fresh(f"{APP}-{version}-model-{name}")

    target = stage / "models" / name
    target.mkdir(parents=True)
    shutil.copy2(checkpoint, target / checkpoint.name)
    operating_point = checkpoint.parent / "operating_point.json"
    if operating_point.is_file():
        shutil.copy2(operating_point, target / operating_point.name)
    for hf_dir in HF_BY_ARCHITECTURE.get(architecture, []):
        if not hf_dir.is_dir():
            raise SystemExit(f"Falta {hf_dir}: cargá el modelo una vez para que se descargue")
        shutil.copytree(hf_dir, stage / "hf" / hf_dir.name)
    logger.info("modelo %s: %s (%s)", name, architecture, checkpoint.name)
    return stage


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
    # `deploy/windows/` se calca sobre el paquete: python/entorno.bat, models/LEEME.txt…
    for template in TEMPLATES.rglob("*"):
        if not template.is_file() or template.suffix == ".in":
            continue
        text = template.read_text(encoding="utf-8").replace("{version}", version)
        text = text.replace("{variant}", variant)
        target = stage / template.relative_to(TEMPLATES)
        target.parent.mkdir(parents=True, exist_ok=True)
        # cmd.exe quiere CRLF; el repo guarda LF
        target.write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))


def stage_runtime(version: str, variant: str) -> Path:
    stage = fresh(f"{APP}-{version}-win64-{variant}")
    copy_source(stage)
    python_dir = install_python(stage)
    install_packages(python_dir, variant)
    copy_launchers(stage, version, variant)
    return stage


# --- Zip, partes y sumas -----------------------------------------------------------


def make_zip(stage: Path, with_root: bool) -> Path:
    # El runtime lleva su carpeta raíz; el de un modelo no, para que se vuelque dentro de ella.
    archive = stage.with_name(f"{stage.name}.zip")
    base = stage.parent if with_root else stage
    logger.info("comprimiendo %s", archive.name)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(base))
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
    # Una entrada por archivo; las corridas sucesivas (runtime, modelos) se acumulan.
    sums = DIST_DIR / "SHA256SUMS.txt"
    entries: dict[str, str] = {}
    if sums.is_file():
        for line in sums.read_text().splitlines():
            digest, _, name = line.partition("  ")
            entries[name] = digest
    for f in files:
        entries[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    sums.write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(entries.items())))
    return sums


def publish(stage: Path, with_root: bool, no_zip: bool) -> None:
    total = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
    logger.info("%s: %.2f GB descomprimido", stage.name, total / 1e9)
    if no_zip:
        return
    files = split(make_zip(stage, with_root))
    write_checksums(files)
    for f in files:
        logger.info("  %s  %.0f MB", f.name, f.stat().st_size / 2**20)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    if args.what == "runtime":
        publish(stage_runtime(args.version, args.variant), True, args.no_zip)
    else:
        for source in args.models:
            publish(stage_model(source, args.version), False, args.no_zip)
    logger.info(
        "Publicar: gh release create v%s dist/*%s*.zip* dist/SHA256SUMS.txt",
        args.version,
        args.version,
    )


if __name__ == "__main__":
    main()
