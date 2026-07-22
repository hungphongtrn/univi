#!/usr/bin/env bash
set -uo pipefail
cd /workspace/univi
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
declare -A CKPTS=(
  [base]="unsloth/gemma-4-E2B-it"
  [200]="data/checkpoints/audio-only-v0/checkpoint-200"
  [400]="data/checkpoints/audio-only-v0/checkpoint-400"
  [600]="data/checkpoints/audio-only-v0/checkpoint-600"
)
for K in base 200 400 600; do
  echo "SWEEP_RUN $K ${CKPTS[$K]}"
  uv run python data/eval/attn_probe/temporal_align.py \
    --checkpoint "${CKPTS[$K]}" --n-clips 24 \
    --out "data/eval/attn_probe/ta_${K}_6x41.json" 2>&1 \
    | grep -E "deep band|best Δ|clips;" | sed "s/^/[$K] /"
done
echo "SWEEP_DONE"
