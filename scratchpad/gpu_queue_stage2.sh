#!/bin/zsh
# ---------------------------------------------------------------------------
# GPU queue, stage 2. Runs after stage 1 (H17 legs + H15 doses) signals
# GPU_QUEUE_ALL_DONE in data/eval/gpu-queue.log.
#
# Order: H16 (config exists) -> H21 leg 1 -> H20 -> H19 (integration, last,
# because it depends on everything upstream).
#
# Each step SKIPS itself if its config is missing, so this can be launched
# before the prep agents have finished writing configs — it simply runs what
# exists when it gets there, and logs what it skipped. Nothing is silently
# dropped: every skip is written to the queue log.
# ---------------------------------------------------------------------------
set -u
Q=data/eval/gpu-queue.log
say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [stage2] $*" | tee -a $Q; }

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

say "stage2 armed; waiting for stage 1"
while ! grep -q "GPU_QUEUE_ALL_DONE" $Q 2>/dev/null; do sleep 120; done
say "stage 1 complete, starting stage 2"
sleep 45

train() {  # $1 = tag, $2 = config
  local tag=$1 cfg=$2
  if [[ ! -f $cfg ]]; then say "SKIP $tag — config $cfg does not exist"; return 0; fi
  if [[ -d data/checkpoints/${tag}/final ]]; then say "SKIP $tag (final exists)"; return 0; fi
  say "TRAIN $tag ($cfg)"
  uv run python -m univi.hybrid.train_pretrained --config $cfg \
      > data/checkpoints/${tag}-run.log 2>&1
  say "TRAIN $tag done rc=$?"
}

train h16-gap-weighted-v0 configs/h16_gap_weighted.yaml
train h21-fontjitter-v0   configs/h21_fontjitter.yaml
# H20 is TWO legs on the wide 2000x160 render. h20_librispeech_1120.yaml is
# SUPERSEDED (narrow render). Leg 1 = 560 (120.5 ms/col, matches today's time
# resolution at 498 tok); leg 2 = 1120 (84.7 ms/col at 1062 tok).
train h20-wide-560-v0   configs/h20_leg1_wide_560.yaml
train h20-wide-1120-v0  configs/h20_leg2_wide_1120.yaml
# H19 DELIBERATELY NOT QUEUED: 6 <<<PENDING>>> fields (init_from, max_soft_tokens,
# max_steps, gap_weighted_loss) cannot be filled until H17 and H16 report, AND its
# DATA build is downstream of H17 (at budget 1120 the per-image cost goes 282->1122,
# forcing max_length >= 2048, new retention, and a rebuilt mixture). Launch by hand.

say "GPU_QUEUE_STAGE2_DONE"
