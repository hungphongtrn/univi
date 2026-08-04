#!/usr/bin/env bash
# Issue #6 E4: bounded training run of the random-init Gemma-vision + Qwen3-1.7B
# hybrid model on Univi's own mixed data (replicates the HuggingFaceM4
# encoder-free-vlm "bigger decoder + different data" experiment: Qwen 3 1.7B
# decoder, our mixture in place of FineVision subsets).
set -uo pipefail
cd /workspace/univi

export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "HYBRID_E4_START $(date -u +%FT%TZ)"
uv run python -m univi.hybrid.train --config configs/hybrid_bounded.yaml
echo "TRAIN_EXIT=$?"

CKPT_DIR="data/checkpoints/hybrid-bounded-v0"
echo "==== CHECKPOINTS ===="
mapfile -t CKPTS < <(ls -d ${CKPT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n)
printf '%s\n' "${CKPTS[@]}"
if [ "${#CKPTS[@]}" -gt 0 ]; then
  LAST="${CKPTS[-1]}"
  echo "LAST_CKPT=$LAST"
  echo "---- trainer_state loss tail ----"
  uv run python -c "
import json
d = json.load(open('$LAST/trainer_state.json'))
print('global_step', d.get('global_step'))
for x in d.get('log_history', [])[-20:]:
    print(json.dumps(x))
" 2>/dev/null
fi

echo "HYBRID_E4_DONE $(date -u +%FT%TZ)"
