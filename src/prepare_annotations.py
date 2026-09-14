import argparse
import logging
from pathlib import Path

import pandas as pd

from core.config import CLEANED_DIR, RAW_DIR
from core.runtime import setup_logging
from data.annotations import (
    MANUAL_DROP,
    MANUAL_IGNORE,
    clean_annotations,
    cleaned,
    list_files,
    recording_stem,
    species_of,
)
from data.raven import CALL, CLEANED_BOX_COLUMNS, SPECIES

logger = logging.getLogger("prepare_annotations")

# Columnas de la tabla de anotaciones que se conservan en cleaned/
COLUMNS = [cleaned(SPECIES), cleaned(CALL), *CLEANED_BOX_COLUMNS, "ignore"]


def find_recording(annotation: Path) -> Path | None:
    recording = annotation.with_name(f"{recording_stem(annotation)}.wav")
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
    ignored = 0
    written = 0
    empty = 0
    without_audio: list[str] = []
    failed: list[str] = []
    manual = set(MANUAL_IGNORE) | set(MANUAL_DROP)
    for annotation in list_files(args.raw, ".txt"):
        relative = annotation.relative_to(args.raw)
        audio = find_recording(annotation)
        if audio is None:
            without_audio.append(str(relative))
            continue
        try:
            frame = clean_annotations(
                pd.read_csv(annotation, sep="\t"), species_of(audio), table=str(relative)
            )
        except Exception as exc:
            failed.append(f"{relative}: {exc}")
            continue
        manual.discard(str(relative))

        # Se llama como su `.wav`.
        out = args.out / relative.with_name(audio.stem + ".txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        frame[COLUMNS].to_csv(out, sep="\t", index=False)

        written += 1
        rows += len(frame)
        review += frame["requires_review"].astype(bool).sum()
        ignored += int(frame["ignore"].sum())
        empty += int(frame.empty)

    if not written:
        raise SystemExit(f"no salió ninguna anotación de {args.raw}")

    logger.info("%d grabaciones | %d anotaciones -> %s", written, rows, args.out)
    if review:
        logger.warning("%d anotaciones con un par especie/llamada fuera de `VALID_PAIRS`", review)
    if ignored:
        logger.info("%d anotaciones con `ignore` (MANUAL_IGNORE)", ignored)
    if manual:
        logger.warning(
            "tablas de MANUAL_IGNORE/MANUAL_DROP que no se leyeron de %s:%s",
            args.raw,
            "".join(f"\n  {line}" for line in sorted(manual)),
        )
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
