#!/usr/bin/env bash
# Barrido de la supresión de duplicados para las corridas de DETR/DINO, que hoy evalúan sin
# NMS (`nms_iou=None` en `models.registry`). Vuelca las cajas crudas una sola vez y barre la
# rejilla en CPU; los gráficos salen de notebooks/nms_sweep.ipynb.
#
#   ./nms_sweep.sh detr_unfreeze_ts10             una corrida
#   ./nms_sweep.sh detr_unfreeze_ts5 detr_unfreeze_ts10
#   FORCE=1 ./nms_sweep.sh detr_unfreeze_ts10     revuelca aunque el volcado esté al día
#   DEVICE=cuda:1 ./nms_sweep.sh detr…            si la GPU 0 está ocupada
set -euo pipefail

cd "$(dirname "$0")"

# uv no está en el PATH de esta máquina; se puede fijar con UV=/ruta/a/uv.
UV=${UV:-$(command -v uv || true)}
UV=${UV:-/home/fcandia/venv/bin/uv}

RUNS_DIR=${RUNS_DIR:-runs}
PROCESSED_DIR=${PROCESSED_DIR:-data/processed}
SPLITS=(val test)

if [ $# -eq 0 ]; then
  printf 'uso: %s <corrida> [corrida...]\n' "$0" >&2
  exit 1
fi

extra=()
if [ -n "${DEVICE:-}" ]; then extra+=(--device "$DEVICE"); fi
if [ -n "${BATCH_SIZE:-}" ]; then extra+=(--batch-size "$BATCH_SIZE"); fi

# El volcado no guarda el mapa ventana -> grabación y `RawPredictions` lo necesita.
for split in "${SPLITS[@]}"; do
  if [ ! -f "$PROCESSED_DIR/${split}_sources.json" ]; then
    printf '\n== reconstruyendo el mapa ventana -> grabación ==\n'
    "$UV" run python src/prepare_data.py --sources-only
    break
  fi
done

for name in "$@"; do
  checkpoint="$RUNS_DIR/$name/best.pt"
  if [ ! -f "$checkpoint" ]; then
    printf 'salto %s: no tiene best.pt\n' "$name" >&2
    continue
  fi

  # Un volcado más viejo que su checkpoint quedó de un entrenamiento anterior.
  pending=()
  for split in "${SPLITS[@]}"; do
    dump="$RUNS_DIR/$name/${name}_${split}_predictions.pt"
    if [ -n "${FORCE:-}" ] || [ ! -f "$dump" ] || [ "$checkpoint" -nt "$dump" ]; then
      pending+=("$split")
    fi
  done

  if [ ${#pending[@]} -gt 0 ]; then
    printf '\n== volcando %s (%s) ==\n' "$name" "${pending[*]}"
    "$UV" run python src/dump_predictions.py --run "$name" --splits "${pending[@]}" "${extra[@]}"
  fi

  printf '\n== barriendo %s ==\n' "$name"
  "$UV" run python src/sweep_nms.py --run "$name"
done

printf '\nlisto. Los gráficos:\n'
printf '  NMS_SWEEP_RUN=%s uv run jupyter lab notebooks/nms_sweep.ipynb\n' "$1"
