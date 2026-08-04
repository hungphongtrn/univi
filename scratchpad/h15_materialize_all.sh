#!/bin/zsh
set -e
for P in 00 15 50; do
  case $P in 00) R=0.0;; 15) R=0.15;; 50) R=0.50;; esac
  echo "STARTING dose p$P (rate $R)"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" uv run python -m data.preprocessing.poisoned_text \
      --out data/materialized/h15-poisoned-p$P --subset-name poisoned-text \
      --poison-rate $R --max-target-tokens 64 --max-pages 1 \
      --train-rows 46000 --val-rows 500 --num-proc 4
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" uv run python -m data.preprocessing.poisoned_text \
      --out data/materialized/h15-poisoned-p$P --subset-name poisoned-text-long \
      --poison-rate $R --max-target-tokens 200 --max-pages 1 \
      --train-rows 4000 --val-rows 500 --num-proc 4
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" uv run python -m data.preprocessing.poisoned_text \
      --out data/materialized/h15-deepval-p$P --subset-name poisoned-text \
      --poison-rate $R --max-target-tokens 1400 --max-pages 4 \
      --train-rows 1 --val-rows 500 --num-proc 4
  echo "DONE dose p$P"
done
echo "H15_MATERIALIZE_ALL_DONE"
