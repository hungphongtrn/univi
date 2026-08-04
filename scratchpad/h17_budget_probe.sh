#!/bin/zsh
set -e
CKPT=data/checkpoints/hybrid-pretrained-randstr-v0/final
for B in 560 1120; do
  echo "STARTING budget $B"
  HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python scratchpad/h13_analyze.py --checkpoint "$CKPT" -n 150 \
      --max-soft-tokens "$B" --max-length 4096 \
      --output data/eval/h17-budget-probe-${B}.json \
      > data/eval/h17-budget-probe-${B}.log 2>&1
  echo "DONE budget $B"
done
echo "ALL_BUDGET_PROBES_DONE"
