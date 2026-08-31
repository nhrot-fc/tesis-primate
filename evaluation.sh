!#/usr/bin/bash

for split in val test; do
  for ckpt in checkpoints/*.pth; do
    uv run python src/evaluate.py --checkpoint "$ckpt" --split "$split" --score-threshold 0.5
  done
done; for split in val test; do
  printf '\n== %s ==\n' "$split"
  { printf 'checkpoint\tarch\tthr\trecall\tprecision\tF\tmAP50\tmAP50-95\n'
    jq -r '[(.checkpoint|split("/")|last), .architecture, .score_threshold,
            .metrics.recall, .metrics.precision, .metrics.f_beta,
            .metrics.map_50, .metrics.map_50_95] | @tsv' checkpoints/*_"$split"_metrics.json
  } | column -t
done
