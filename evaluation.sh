#!/usr/bin/env bash
# Compara las corridas de runs/ con un mismo protocolo: vuelca las predicciones crudas
# del best.pt de cada una sobre val y test, y deja que compare_models.py elija el umbral
# en val y mida una sola vez en test.
#
#   ./evaluation.sh                          todas las corridas de runs/
#   ./evaluation.sh dino_pcen_ts10_ft detr…  sólo las nombradas
#   FORCE=1 ./evaluation.sh                  revuelca aunque el volcado esté al día
#   DEVICE=cuda:1 ./evaluation.sh            si la GPU 0 está ocupada
set -euo pipefail

cd "$(dirname "$0")"

# uv no está en el PATH de esta máquina; se puede fijar con UV=/ruta/a/uv.
UV=${UV:-$(command -v uv || true)}
UV=${UV:-/home/fcandia/venv/bin/uv}

RUNS_DIR=${RUNS_DIR:-runs}
PROCESSED_DIR=${PROCESSED_DIR:-data/processed}
# Espejo de nombres que fijan los scripts de Python: `training.checkpoint.BEST`,
# `evaluation.evaluator.SUFFIX`, `compare_models.OUTPUT` y `data.cache.SPLITS`.
BEST=best.pt
DUMP_SUFFIX=_predictions.pt
OUTPUT=${OUTPUT:-$RUNS_DIR/comparacion/comparacion_modelos}
SPLITS=(val test)

extra=()
if [ -n "${DEVICE:-}" ]; then extra+=(--device "$DEVICE"); fi
if [ -n "${BATCH_SIZE:-}" ]; then extra+=(--batch-size "$BATCH_SIZE"); fi

names=("$@")
if [ ${#names[@]} -eq 0 ]; then
  for dir in "$RUNS_DIR"/*/; do names+=("$(basename "$dir")"); done
fi

# El IC por grabación necesita el mapa ventana -> grabación, que no vive en los .pt.
for split in "${SPLITS[@]}"; do
  if [ ! -f "$PROCESSED_DIR/${split}_sources.json" ]; then
    printf '\n== reconstruyendo el mapa ventana -> grabación ==\n'
    "$UV" run python src/prepare_data.py --sources-only
    break
  fi
done

dumps=()
for name in "${names[@]}"; do
  checkpoint="$RUNS_DIR/$name/$BEST"
  if [ ! -f "$checkpoint" ]; then
    printf 'salto %s: no tiene %s\n' "$name" "$BEST" >&2
    continue
  fi

  # Un volcado más viejo que su checkpoint quedó de un entrenamiento anterior.
  pending=()
  paths=()
  for split in "${SPLITS[@]}"; do
    dump="$RUNS_DIR/$name/${name}_${split}${DUMP_SUFFIX}"
    paths+=("$dump")
    if [ -n "${FORCE:-}" ] || [ ! -f "$dump" ] || [ "$checkpoint" -nt "$dump" ]; then
      pending+=("$split")
    fi
  done

  if [ ${#pending[@]} -eq 0 ]; then
    printf 'volcados al día: %s\n' "$name"
  else
    printf '\n== volcando %s (%s) ==\n' "$name" "${pending[*]}"
    # Que una corrida se caiga no puede dejar sin comparación a las que sí volcaron.
    if ! "$UV" run python src/dump_predictions.py --run "$name" --splits "${pending[@]}" "${extra[@]}"; then
      printf 'salto %s: el volcado falló\n' "$name" >&2
      continue
    fi
  fi
  dumps+=("${paths[@]}")
done

if [ ${#dumps[@]} -eq 0 ]; then
  printf 'no hay corridas con %s en %s/\n' "$BEST" "$RUNS_DIR" >&2
  exit 1
fi

printf '\n== comparando %d modelos ==\n' $((${#dumps[@]} / ${#SPLITS[@]}))
"$UV" run python src/compare_models.py "${dumps[@]}" --output "$OUTPUT"
