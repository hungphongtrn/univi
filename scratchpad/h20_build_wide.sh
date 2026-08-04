#!/usr/bin/env bash
# H20 — build data/materialized/h20-audio-wide-v0 (both audio lanes at 2000x160).
# CPU ONLY. A GPU training queue is live; nothing here touches the card.
set -euo pipefail
cd /workspace/univi

export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

ROOT=data/materialized/h20-audio-wide-v0
W=2000
H=160
NP=4

echo "STEP0 self-test"
uv run python -m data.preprocessing.widen_audio_pages --self-test

echo "STEP1 librispeech train+validation"
uv run python -m data.preprocessing.widen_audio_pages \
  --source data/materialized/univi-3M-v0-split/librispeech \
  --out "$ROOT/librispeech" \
  --splits train validation \
  --output-width $W --output-height $H --num-proc $NP

echo "STEP2 spoken-digits train -> 15pct anchor (deterministic shuffle seed 42, n=18355)"
uv run python -m data.preprocessing.widen_audio_pages \
  --source data/materialized/spoken-digits-v0/spoken-digits \
  --out "$ROOT/spoken-digits-15pct" \
  --splits train --select-n 18355 --select-seed 42 \
  --output-width $W --output-height $H --num-proc $NP

echo "STEP3 spoken-digits validation"
uv run python -m data.preprocessing.widen_audio_pages \
  --source data/materialized/spoken-digits-v0/spoken-digits \
  --out "$ROOT/spoken-digits" \
  --splits validation \
  --output-width $W --output-height $H --num-proc $NP

echo "STEP4 manifest"
uv run python scratchpad/h20_write_manifest.py --root "$ROOT"

echo "STEP5 done"
du -sh "$ROOT"
