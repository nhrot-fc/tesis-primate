import argparse
import logging
import os
import re
from pathlib import Path

import pandas as pd

from core.config import CLEANED_DIR, RAW_DIR
from core.runtime import setup_logging
from data.annotations import clean_annotations, cleaned, species_of
from data.raven import CALL, CLEANED_BOX_COLUMNS, SPECIES

logger = logging.getLogger("prepare_annotations")

# Columnas de la tabla de anotaciones que se conservan en cleaned/
COLUMNS = [cleaned(SPECIES), cleaned(CALL), *CLEANED_BOX_COLUMNS]


def list_annotation_files(root: Path) -> list[Path]:
    # os.walk() para enlaces simbólicos
    return sorted(
        Path(directory) / name
        for directory, _, names in os.walk(root, followlinks=True)
        for name in names
        if name.endswith(".txt")
    )


def find_recording(annotation: Path) -> Path | None:
    # Tolera un espacio de más y el `.Table.1.selections` de Raven.
    stem = re.sub(r"(\.Table\.\d+)?\.selections$", "", annotation.name.removesuffix(".txt").strip())
    recording = annotation.with_name(f"{stem}.wav")
    return recording if recording.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normaliza las anotaciones de raw/ y las deja en cleaned/, sin copiar audio."
    )
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=CLEANED_DIR)
    parser.add_argument("--force", action="store_true", help="regenera sobre las que ya estén")
    args = parser.parse_args()

    setup_logging()
    if any(args.out.rglob("*.txt")) and not args.force:
        raise SystemExit(f"ya hay anotaciones en {args.out}; pasá --force para regenerarlas.")

    rows = 0
    review = 0
    written = 0
    empty = 0
    without_audio: list[str] = []
    failed: list[str] = []
    for annotation in list_annotation_files(args.raw):
        relative = annotation.relative_to(args.raw)
        audio = find_recording(annotation)
        if audio is None:
            without_audio.append(str(relative))
            continue
        try:
            frame = clean_annotations(pd.read_csv(annotation, sep="\t"), species_of(audio))
        except Exception as exc:
            failed.append(f"{relative}: {exc}")
            continue

        # Se llama como su `.wav`.
        out = args.out / relative.with_name(audio.stem + ".txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        frame[COLUMNS].to_csv(out, sep="\t", index=False)

        written += 1
        rows += len(frame)
        review += frame["requires_review"].astype(bool).sum()
        empty += int(frame.empty)

    if not written:
        raise SystemExit(f"no salió ninguna anotación de {args.raw}")

    logger.info("%d grabaciones | %d anotaciones -> %s", written, rows, args.out)
    if review:
        logger.warning("%d anotaciones con un par especie/llamada fuera de `VALID_PAIRS`", review)
    if empty:
        logger.info("%d grabaciones quedaron sin ninguna anotación utilizable", empty)
    if failed or without_audio:
        logger.warning(
            "%d ilegibles | %d sin .wav al lado%s",
            len(failed),
            len(without_audio),
            "".join(f"\n  {line}" for line in failed + without_audio),
        )


if __name__ == "__main__":
    main()
