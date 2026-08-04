#!/bin/zsh
# ---------------------------------------------------------------------------
# Sequential GPU queue. ONE job on the card at a time (repo SOLO rule).
#
# Built 2026-07-29 after discovering that "the remaining legs follow
# automatically" was NOT true — only H17 leg 280 had been launched and nothing
# chained behind it.
#
# Gating uses PIDs / file existence, never `pgrep -f <pattern>`: a pattern that
# appears in the watcher's own command line matches the watcher itself, and that
# deadlocked the H13 discriminator for ~7 h.
# ---------------------------------------------------------------------------
set -u
Q=data/eval/gpu-queue.log
say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a $Q; }

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 0. wait for the already-running leg 280 -------------------------------
WAIT_PID=${1:-}
if [[ -n "$WAIT_PID" ]]; then
  say "waiting for in-flight PID $WAIT_PID (H17 leg 280)"
  while [ -d /proc/$WAIT_PID ]; do sleep 60; done
  say "PID $WAIT_PID exited"
  sleep 45
fi

train() {  # $1 = tag, $2 = config
  local tag=$1 cfg=$2
  if [[ -d data/checkpoints/${tag}/final ]]; then say "SKIP train $tag (final exists)"; return 0; fi
  say "TRAIN $tag  ($cfg)"
  uv run python -m univi.hybrid.train_pretrained --config $cfg \
      > data/checkpoints/${tag}-run.log 2>&1
  say "TRAIN $tag done rc=$?"
}

probe_d3() {  # $1 = tag, $2 = budget
  local tag=$1 b=$2
  local ck=data/checkpoints/${tag}/final
  [[ -d $ck ]] || ck=$(ls -d data/checkpoints/${tag}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  if [[ -z "$ck" || ! -d "$ck" ]]; then say "PROBE $tag SKIPPED (no checkpoint)"; return 1; fi
  say "PROBE $tag budget=$b ckpt=$ck"
  # --max-length 4096: at 1120 a d4 row is ~2066 tok and the 2048 default would
  # truncate exactly the deep positions K is measured at.
  uv run python scratchpad/h13_analyze.py --checkpoint "$ck" -n 150 \
      --rungs randstr-d3 --max-soft-tokens $b --max-length 4096 \
      --output data/eval/h17-K-${tag}.json > data/eval/h17-K-${tag}.log 2>&1
  say "PROBE $tag done rc=$?"
}

# --- 1. H17: three legs, probe each ----------------------------------------
probe_d3 h17-d3-280-v0 280

train    h17-d3-560-v0  configs/h17_leg_560.yaml
probe_d3 h17-d3-560-v0  560

train    h17-d3-1120-v0 configs/h17_leg_1120.yaml
probe_d3 h17-d3-1120-v0 1120

say "H17 THREE LEGS COMPLETE"

# --- 2. H15: three doses (gate on H16) -------------------------------------
train h15-poisoned-p00-v0 configs/h15_poisoned_p00.yaml
train h15-poisoned-p15-v0 configs/h15_poisoned_p15.yaml
train h15-poisoned-p50-v0 configs/h15_poisoned_p50.yaml
say "H15 THREE DOSES COMPLETE"

say "GPU_QUEUE_ALL_DONE"
