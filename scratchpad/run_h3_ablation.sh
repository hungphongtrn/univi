set -e
cd /workspace/univi
echo "==== H3-trained (best = step 500) ===="
HF_HUB_OFFLINE=1 uv run python scratchpad/h3_ablation.py \
  --checkpoint data/checkpoints/h3-randstr-v0/best --tag h3 --max-samples 200
echo "==== H1 warm-start baseline (pre-H3-training) ===="
HF_HUB_OFFLINE=1 uv run python scratchpad/h3_ablation.py \
  --checkpoint data/checkpoints/encoder-free-v0/best --tag h1-baseline --max-samples 200
echo "ALL_ABLATIONS_DONE"
