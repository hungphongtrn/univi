#!/bin/zsh
while [ -d /proc/1388249 ]; do sleep 30; done
echo "H14_TRAINING_EXITED"; sleep 40
CKPT=data/checkpoints/h14-masked-short-v0/final
[ -d "$CKPT" ] || CKPT=$(ls -d data/checkpoints/h14-masked-short-v0/checkpoint-* | sort -t- -k2 -n | tail -1)
echo "USING $CKPT"
HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   uv run python scratchpad/h14_masked_eval.py --checkpoint "$CKPT" -n 150     --val-path data/materialized/h14-masked-short-v0/masked-randstr/validation     --floor data/materialized/h14-masked-short-v0/floor.json     --out data/eval/h14-masked-short.json     > data/eval/h14-masked-short.log 2>&1
echo "H14_EVAL_DONE"
tail -60 data/eval/h14-masked-short.log
