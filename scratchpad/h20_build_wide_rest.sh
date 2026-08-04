#!/usr/bin/env bash
set -euo pipefail
cd /workspace/univi
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
ROOT=data/materialized/h20-audio-wide-v0

# wait for the first build script to exit
while pgrep -f "widen_audio_pages --source data/materialized/spoken-digits-v0" >/dev/null; do sleep 15; done

echo STEP2b spoken-digits FULL train "(30,000 rows)" - the alternative anchor
uv run python -m data.preprocessing.widen_audio_pages \
  --source data/materialized/spoken-digits-v0/spoken-digits \
  --out "$ROOT/spoken-digits" \
  --splits train \
  --output-width 2000 --output-height 160 --num-proc 4

echo STEP4 manifest
uv run python scratchpad/h20_write_manifest.py --root "$ROOT"
echo STEP5 done
du -sh "$ROOT"
