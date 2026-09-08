import argparse
import logging
from pathlib import Path

import pandas as pd

from core.config import settings
from core.runtime import setup_logging
from data.annotations import clean_annotations, species_of

logger = logging.getLogger("prepare_annotations")

# La duración y el ancho de banda no se guardan: salen de estas cifras.
COLUMNS = ["species", "call_type", "begin_time_s", "end_time_s", "low_freq_hz", "high_freq_hz"]


def audio_for(annotation: Path) -> Path | None:
    # El `.txt` a veces trae un espacio de más, o el `.selections` que le agrega Raven.
    stem = annotation.name.removesuffix(".txt").strip().removesuffix(".selections")
    audio = annotation.with_name(f"{stem}.wav")
    return audio if audio.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normaliza las anotaciones de raw/ y las deja en cleaned/, sin copiar audio."
    )
    parser.add_argument("--raw", type=Path, default=settings.raw_dir)
    parser.add_argument("--out", type=Path, default=settings.cleaned_dir)
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
    for annotation in sorted(args.raw.rglob("*.txt")):
        relative = annotation.relative_to(args.raw)
        audio = audio_for(annotation)
        if audio is None:
            without_audio.append(str(relative))
            continue
        try:
            frame = clean_annotations(pd.read_csv(annotation, sep="\t"), species_of(audio))
        except Exception as exc:
            failed.append(f"{relative}: {exc}")
            continue

        # Se llama como su `.wav`: de acá en más la grabación es esta misma ruta contra raw/.
        out = args.out / relative.with_name(audio.stem + ".txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        frame[COLUMNS].to_csv(out, sep="\t", index=False)

        written += 1
        rows += len(frame)
        review += int(frame["requires_review"].sum())
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
