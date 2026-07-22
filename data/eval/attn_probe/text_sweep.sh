#!/usr/bin/env bash
set -uo pipefail
cd /workspace/univi
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
declare -A CK=( [base]="unsloth/gemma-4-E2B-it" [2800]="data/checkpoints/full-v0/checkpoint-2800" )
for K in base 2800; do
  echo "TEXTSWEEP $K ${CK[$K]}"
  uv run python data/eval/attn_probe/text_align.py --checkpoint "${CK[$K]}" --n-rows 24 \
    --out "data/eval/attn_probe/text_${K}.json" 2>&1 \
    | grep -E "deep band|all layers|best Δ|rows \(" | sed "s/^/[$K] /"
done
echo "TEXTSWEEP_DONE"
