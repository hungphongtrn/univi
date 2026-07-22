#!/usr/bin/env bash
# E3: audio-only bounded fine-tune -> re-ablate every checkpoint.
# Self-contained log: prints checkpoint paths, trainer-state tail, and each
# ablation JSON inline, so the log file alone carries every result.
set -uo pipefail
cd /workspace/univi

# Hub blobs are all cached; force classic HTTPS + offline to dodge the Xet stall.
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

CKPT_DIR="data/checkpoints/audio-only-v0"
echo "E3_START $(date -u +%FT%TZ)"

echo "==== STEP 0: PREFLIGHT (dataset validation + manifest for this config) ===="
uv run python -m univi.train --config configs/audio_only_v0.yaml --preflight-only
PF_EXIT=$?
echo "PREFLIGHT_EXIT=$PF_EXIT"
if [ "$PF_EXIT" -ne 0 ]; then echo "PREFLIGHT FAILED — aborting"; echo "E3_DONE $(date -u +%FT%TZ)"; exit 1; fi

echo "==== STEP 1: TRAIN (audio-only, bounded) ===="
uv run python -m univi.train --config configs/audio_only_v0.yaml
echo "TRAIN_EXIT=$?"

echo "==== STEP 2: CHECKPOINTS ===="
mapfile -t CKPTS < <(ls -d ${CKPT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n)
printf '%s\n' "${CKPTS[@]}"
if [ "${#CKPTS[@]}" -eq 0 ]; then echo "NO_CHECKPOINTS — aborting ablation"; echo "E3_DONE $(date -u +%FT%TZ)"; exit 1; fi
LAST="${CKPTS[-1]}"
echo "LAST_CKPT=$LAST"
echo "---- trainer_state tail (last evals) ----"
uv run python -c "import json,sys; d=json.load(open('$LAST/trainer_state.json')); print('global_step',d.get('global_step')); [print(json.dumps(x)) for x in d.get('log_history',[]) if 'eval_librispeech_loss' in x or 'loss' in x][-12:]" 2>/dev/null

echo "==== STEP 3: ABLATE EACH CHECKPOINT (audio grounding trajectory) ===="
for CKPT in "${CKPTS[@]}"; do
  STEP=$(basename "$CKPT" | sed 's/checkpoint-//')
  OUT="data/eval/val-ablation/ablation-audioonly-${STEP}.json"
  echo "---- ABLATE step=$STEP ckpt=$CKPT out=$OUT ----"
  uv run python eval_ablation.py --config configs/val_ablation.yaml \
    --checkpoint "$CKPT" --max-samples 120 --no-base --output "$OUT"
  echo "ABLATION_JSON_BEGIN $OUT"
  cat "$OUT" 2>/dev/null || echo "MISSING_JSON $OUT"
  echo ""
  echo "ABLATION_JSON_END $OUT"
done

echo "E3_DONE $(date -u +%FT%TZ)"
