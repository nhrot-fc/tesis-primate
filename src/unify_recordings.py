import argparse
import logging
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

from core.config import RAW_DIR, UNIFIED_DIR
from core.runtime import setup_logging
from data.annotations import audio_signature, list_files, recording_stem, species_of
from data.raven import BEGIN, SELECTION, renumber

logger = logging.getLogger("unify_recordings")

# Carpeta de origen de cada fila: el mismo wav está anotado en varias carpetas de especie
# y la columna `Species` de la tabla no siempre está cargada.
FOLDER_SPECIES = "Folder Species"
# Mapa de cada archivo de unified/ a los de raw/ que lo originaron
SOURCES = "sources.tsv"
# PteroSet: symlink a /data, 95 GB y sin copias; no aporta nada a la unificación
SKIP_FOLDERS = ["birds__AV"]


def read_table(annotation: Path) -> pd.DataFrame | None:
    # Tal cual está: todo como texto para no reformatear nada al volver a escribir.
    try:
        return pd.read_csv(annotation, sep="\t", dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return None


def merge_tables(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    # Unión de las tablas de las copias: columnas en el orden en que aparecen, sin las filas
    # repetidas entre copias, ordenada por tiempo y con `Selection` correlativo para Raven.
    table = pd.concat(frames, ignore_index=True, sort=False).fillna("")
    annotation_columns = [c for c in table.columns if c not in (SELECTION, FOLDER_SPECIES)]
    n_before = len(table)
    table = table.drop_duplicates(subset=annotation_columns)
    if BEGIN in table:
        order = pd.to_numeric(table[BEGIN], errors="coerce")
        table = table.loc[order.sort_values(kind="stable").index]
    table = renumber(table.reset_index(drop=True))
    table = table[[SELECTION, *[c for c in table.columns if c != SELECTION]]]
    return table, n_before - len(table)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deja en una sola carpeta cada grabación de raw/ una única vez, con la unión "
        "de las tablas de Raven de todas sus copias, sin limpiarlas."
    )
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=UNIFIED_DIR)
    parser.add_argument("--skip", nargs="*", default=SKIP_FOLDERS, help="carpetas de raw/ a omitir")
    parser.add_argument("--force", action="store_true", help="borra y regenera la salida")
    args = parser.parse_args()

    setup_logging()
    if args.out.exists() and any(args.out.iterdir()):
        if not args.force:
            raise SystemExit(f"ya hay archivos en {args.out}; pasá --force para regenerarlos.")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    def kept(path: Path) -> bool:
        return path.relative_to(args.raw).parts[0] not in args.skip

    wavs = [w for w in list_files(args.raw, ".wav") if kept(w)]
    tables = {(t.parent, recording_stem(t)): t for t in list_files(args.raw, ".txt") if kept(t)}
    table_of = {w: tables.pop((w.parent, w.stem), None) for w in wavs}
    without_audio = sorted(str(t.relative_to(args.raw)) for t in tables.values())

    # Copias byte-idénticas, en el orden de raw/: la primera da el nombre.
    groups: dict[tuple[int, str], list[Path]] = defaultdict(list)
    for wav in wavs:
        groups[audio_signature(wav)].append(wav)

    names: dict[str, int] = defaultdict(int)
    sources: list[dict[str, object]] = []
    written = 0
    unannotated = 0
    dropped = 0
    renamed: list[str] = []
    empty: list[str] = []
    failed: list[str] = []
    for signature, copies in groups.items():
        # Mismo nombre con otro audio (raíz vs. subcarpeta): se numera la segunda en adelante.
        stem = copies[0].stem
        names[stem] += 1
        if names[stem] > 1:
            stem = f"{stem}_{names[stem]}"
            renamed.append(f"{copies[0].relative_to(args.raw)} -> {stem}.wav")
        shutil.copy2(copies[0], args.out / f"{stem}.wav")

        frames: list[pd.DataFrame] = []
        for wav in copies:
            annotation = table_of[wav]
            sources.append(
                {
                    "recording": stem,
                    "raw_wav": str(wav.relative_to(args.raw)),
                    "raw_txt": "" if annotation is None else str(annotation.relative_to(args.raw)),
                    "size_bytes": signature[0],
                    "md5_1mb": signature[1],
                }
            )
            if annotation is None:
                continue
            try:
                frame = read_table(annotation)
            except Exception as exc:
                failed.append(f"{annotation.relative_to(args.raw)}: {exc}")
                continue
            if frame is None:
                empty.append(str(annotation.relative_to(args.raw)))
                continue
            frame[FOLDER_SPECIES] = species_of(wav).upper()
            frames.append(frame)

        if not frames:
            unannotated += 1
            continue
        table, n_dropped = merge_tables(frames)
        table.to_csv(args.out / f"{stem}.txt", sep="\t", index=False)
        written += 1
        dropped += n_dropped

    pd.DataFrame(sources).to_csv(args.out / SOURCES, sep="\t", index=False)

    logger.info(
        "%d wav en %s -> %d grabaciones únicas en %s (%d copias unidas, %d con anotaciones, "
        "%d sin ninguna)",
        len(wavs),
        args.raw,
        len(groups),
        args.out,
        len(wavs) - len(groups),
        written,
        unannotated,
    )
    if dropped:
        logger.info(
            "%d filas repetidas entre copias de la misma grabación, quedan una vez", dropped
        )
    if renamed:
        logger.warning(
            "%d nombres repetidos con audio distinto, renombrados:%s",
            len(renamed),
            "".join(f"\n  {line}" for line in renamed),
        )
    if failed or empty or without_audio:
        logger.warning(
            "%d tablas ilegibles | %d vacías | %d sin .wav al lado, quedan fuera:%s",
            len(failed),
            len(empty),
            len(without_audio),
            "".join(f"\n  {line}" for line in failed + empty + without_audio),
        )


if __name__ == "__main__":
    main()
