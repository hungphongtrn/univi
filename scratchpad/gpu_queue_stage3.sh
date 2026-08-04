#!/bin/zsh
# ---------------------------------------------------------------------------
# GPU queue, stage 3 — the READ-OUT queue.
#
# Stage 1 (scratchpad/gpu_queue.sh) and stage 2 (scratchpad/gpu_queue_stage2.sh)
# queued TRAINING. Stage 2 queued no grounding probes at all, so H15 (3 arms),
# H16, H21 and H20 (2 legs) trained overnight and are UNMEASURED. Aggregate eval
# CE is explicitly NOT a grounding monitor in this project (H13 §1: a fully dead
# run with Δperm = 0 still carried Δblank +1.0%…+8.7%), so nothing is known about
# any of them until the probes below run.
#
# Every command here is the read-out its hypothesis doc pre-registers. Where the
# doc names a probe and a flag, the flag is passed explicitly even when it equals
# the default — the probes' `--max-length` default of 2048 silently TRUNCATES the
# deep positions these measurements exist to read (CLAUDE.md; H20's retention
# table), so it is never left implicit.
#
# Gating rules obeyed here, each of which has cost this project time before:
#   * BLOCKS on GPU_QUEUE_STAGE2_DONE in data/eval/gpu-queue.log — the same idiom
#     stage 2 uses to block on GPU_QUEUE_ALL_DONE. Training runs SOLO.
#   * Any wait on a running job gates on a PID via `[ -d /proc/$PID ]`, NEVER on
#     `pgrep -f <pattern>`: the pattern matches the watcher's own command line and
#     that self-match deadlocked the H13 discriminator for ~7 h.
#   * `unsetopt nomatch`: in zsh an unmatched glob is a FATAL error that aborts
#     the enclosing command, so every glob here is either qualified with (N) or
#     covered by this.
#   * Each step SKIPS itself, with a logged reason, when its checkpoint is missing
#     or its output JSON already exists. Nothing is dropped silently.
#   * Every probe writes a JSON under data/eval/ and tees a log to
#     data/eval/stage3-<step>.log.
#
# VRAM: every step is a forward-only, batch-size-1 probe. The largest sequence in
# the queue is H20 leg 2's 4-page librispeech row (4 × 1062 + 151 + 21 = 4420
# tokens); prefix_dperm_probe chunks its logits at 256 positions so no full
# [S, 151936] fp32 tensor is ever materialised. Free VRAM is logged before every
# step so a squeeze is visible in the queue log rather than surfacing as an OOM.
# NOTE: another container holds ~15.7 GB of the 40 GB card, so ~24 GB is the real
# ceiling — inside the 26 GB budget, but not by much.
#
#   Launch (stage-2 shell PID as $1 so a stage-2 crash cannot deadlock this):
#     nohup zsh scratchpad/gpu_queue_stage3.sh 1508434 \
#         > scratchpad/gpu_queue_stage3.out 2>&1 &
# ---------------------------------------------------------------------------
set -u
unsetopt nomatch          # an unmatched glob must not abort the queue

Q=data/eval/gpu-queue.log
say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [stage3] $*" | tee -a $Q; }

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MARKER=GPU_QUEUE_STAGE2_DONE
STAGE2_PID=${1:-}
FREE_MIB_REQUIRED=20000   # our training holds ~9.4 GB; the neighbour container
                          # ~15.7 GB. >20 GB free ⇒ our card is released.

free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1
}

wait_free_vram() {  # only used on the degraded path (stage 2 died unsignalled)
  local f
  local i=0
  while true; do
    f=$(free_mib)
    [[ -n "$f" ]] || f=0
    if (( f >= FREE_MIB_REQUIRED )); then
      say "GPU free = ${f} MiB (>= ${FREE_MIB_REQUIRED}); card is ours"
      return 0
    fi
    (( i % 10 == 0 )) && say "waiting for the card: free ${f} MiB < ${FREE_MIB_REQUIRED} MiB"
    i=$(( i + 1 ))
    sleep 60
  done
}

# --- 0. block until stage 2 says it is done --------------------------------
# NEVER write $MARKER's literal text into $Q from here: this script's own log
# lines land in the same file, so an arming message that NAMED the marker made
# the wait loop match ITSELF and clear the gate instantly on the first launch
# (2026-07-30, caught in 6 s, no GPU work started). That is the grep form of the
# `pgrep -f` self-match that cost ~7 h on H13. Two independent defences: the
# message below does not contain the marker, and the grep ignores every line this
# stage wrote.
say "stage3 armed (read-out queue); waiting for stage 2 to signal completion"
while ! grep -v '\[stage3\]' $Q 2>/dev/null | grep -q "$MARKER"; do
  if [[ -n "$STAGE2_PID" ]] && [ ! -d /proc/$STAGE2_PID ]; then
    say "!! stage-2 shell (PID $STAGE2_PID) is GONE and it never signalled completion."
    say "!! Falling back to a VRAM gate — the SOLO rule still holds, the marker does not."
    wait_free_vram
    break
  fi
  sleep 120
done
say "gate cleared; free VRAM $(free_mib) MiB. Settling 45 s."
sleep 45

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
latest_ckpt() {  # $1 = checkpoint dir -> prints the highest-numbered checkpoint-N
  local d=$1 best="" n b=-1 x
  local -a c
  c=( $d/checkpoint-*(/N) )          # (/) dirs only, (N) empty instead of fatal
  (( ${#c} )) || return 1
  for x in $c; do
    n=${${x:t}#checkpoint-}
    [[ $n == <-> ]] || continue
    if (( n > b )); then b=$n; best=$x; fi
  done
  [[ -n $best ]] || return 1
  print -r -- $best
}

resolve_ckpt() {  # $1 = run tag -> prints final/ if present, else the latest step
  local d=data/checkpoints/$1
  if [[ -d $d/final ]]; then
    print -r -- $d/final
    return 0
  fi
  latest_ckpt $d
}

# probe <step-name> <output-json> <cmd...>
probe() {
  local name=$1 out=$2
  shift 2
  if [[ -f $out ]]; then
    say "SKIP $name — output $out already exists"
    return 0
  fi
  say "RUN  $name -> $out   (free VRAM $(free_mib) MiB)"
  "$@" > data/eval/stage3-${name}.log 2>&1
  local rc=$?
  if [[ -f $out ]]; then
    say "DONE $name rc=$rc (wrote $out)"
  else
    say "FAIL $name rc=$rc — NO OUTPUT; see data/eval/stage3-${name}.log"
  fi
  return 0
}

# ---------------------------------------------------------------------------
# 0. CPU-only: the blind-branch guards, straight out of the finished run logs.
#    `log_blank_ce: true` was on for H15 (3 arms), H16 and both H20 legs.
#    H16 pre-registers blank_ce_inflation > +0.10 as VOID; H15 pre-registers
#    prior_gap_ce < 0.05 nats for three consecutive evals as a collapse alarm.
#    Neither is a grounding metric (H13 §1) — they are VOID conditions only.
#
#    NOTE the two naming schemes: stage 2 writes each run log under its own TAG
#    (`h21-fontjitter-v0-run.log`, `h20-wide-560-v0-run.log`) while the checkpoints
#    land in the CONFIG's output_dir (`h21-fontmix-v0/`, `h20-leg1-wide-560-v0/`).
#    Both spellings are listed; a file that does not exist is reported MISSING
#    rather than skipped silently.
# ---------------------------------------------------------------------------
probe blank-ce-guard data/eval/stage3-blank-ce-guard.json \
  uv run python scratchpad/blank_ce_guard.py \
    --logs data/checkpoints/h15-poisoned-p00-v0-run.log \
           data/checkpoints/h15-poisoned-p15-v0-run.log \
           data/checkpoints/h15-poisoned-p50-v0-run.log \
           data/checkpoints/h16-gap-weighted-v0-run.log \
           data/checkpoints/h21-fontjitter-v0-run.log \
           data/checkpoints/h21-fontmix-v0-run.log \
           data/checkpoints/h20-wide-560-v0-run.log \
           data/checkpoints/h20-leg1-wide-560-v0-run.log \
           data/checkpoints/h20-wide-1120-v0-run.log \
           data/checkpoints/h20-leg2-wide-1120-v0-run.log \
    --out data/eval/stage3-blank-ce-guard.json

# ---------------------------------------------------------------------------
# 1. H15 — prior-poisoned text, THREE arms (p00 control / p15 / p50).
#    docs/hypothesis/todo/H15-prior-poisoned-text.md:197-199 names the primary
#    instrument verbatim:
#        scratchpad/hybrid_4lane_position_decay.py \
#            --val-path data/materialized/h15-deepval-p<rate>/poisoned-text/validation -n 150
#    reported PER POSITION BIN — "Not aggregate CE, not floor.json, not aggregate
#    Δperm". Line 213 adds a VOID guard that is evaluated FIRST: prefix-matched
#    Δperm over the first 48 answer tokens, < +10% ⇒ that arm did not read and
#    none of its position numbers may be interpreted. Lines 205-211 add the
#    pure-poisoned-token gain (tokens lying ENTIRELY inside poisoned characters,
#    mapped from the poison_mask CHARACTER indices).
#
#    The VOID guard and the pure-token read-out are produced by
#    scratchpad/prefix_dperm_probe.py: nothing else in the repo computes a
#    prefix-matched Δperm, and hybrid_pretrained_ablation.py CANNOT be used on
#    this lane at all — it forces one image per row while HybridCollator indexes
#    soft_counts per image content part, and the p00 arm has 112/500 two-page
#    validation rows (p15 9/500, p50 0/500) ⇒ IndexError.
#
#    --max-length 4096, not the 2048 default: a 2-page p00 row is 1400 target +
#    2 trailer + 564 image + scaffold ≈ 2010 tokens, i.e. inside a hair of the
#    default, and truncation there would clip the CONTROL arm's deepest bins and
#    inflate the poisoned-vs-control difference this experiment is built on.
#    Both probes use seed 42 (position_decay's default) so they score the SAME
#    150 rows.
# ---------------------------------------------------------------------------
for rate in p00 p15 p50; do
  ck=$(resolve_ckpt h15-poisoned-${rate}-v0) || ck=""
  val=data/materialized/h15-deepval-${rate}/poisoned-text/validation
  if [[ -z "$ck" ]]; then
    say "SKIP h15-$rate — no checkpoint under data/checkpoints/h15-poisoned-${rate}-v0"
    continue
  fi
  if [[ ! -d $val ]]; then
    say "SKIP h15-$rate — deep-val split missing: $val"
    continue
  fi
  say "H15 arm $rate: checkpoint $ck"

  # (a) VOID guard first, per the doc's ordering, + the pure-poisoned read-out
  probe h15-${rate}-void-guard data/eval/h15-void-guard-${rate}.json \
    uv run python scratchpad/prefix_dperm_probe.py \
      --checkpoint "$ck" --val-path "$val" -n 150 --seed 42 \
      --max-length 4096 --prefix-tokens 48 \
      --poison-mask-column poison_mask --poison-charset-size 26 \
      --tag h15-${rate} --out data/eval/h15-void-guard-${rate}.json

  # (b) the PRIMARY instrument, exactly as pre-registered
  probe h15-${rate}-position-decay \
    data/eval/hybrid-4lane-position-decay-h15-${rate}-deepval.json \
    uv run python scratchpad/hybrid_4lane_position_decay.py \
      --checkpoint "$ck" --val-path "$val" --lanes poisoned-text \
      -n 150 --seed 42 --max-length 4096 --tag h15-${rate}-deepval
done

# ---------------------------------------------------------------------------
# 2. H16 — prior-gap-weighted loss, fineweb-only.
#    docs/hypothesis/todo/H16-prior-gap-weighted-loss.md ("Pre-registered
#    criterion") and configs/h16_gap_weighted.yaml:21-24 name both probes:
#      Confirms: positional reading gain at answer tokens 10-100 >= +10 pts
#                (scratchpad/hybrid_4lane_position_decay.py) AND Δperm >= +30%
#                (the ablation).
#      GUARD:    eval blank-CE inflation <= +10% -> step 0 above.
#
#    The ablation is hybrid_4lane_ablation.py, NOT hybrid_pretrained_ablation.py:
#    fineweb rows render to several pages (22% of the validation split has 2+,
#    and the lane reaches 108) and only the 4-lane version supplies one donor and
#    one blank PER image, keeping the soft-token count identical across the three
#    conditions. hybrid_pretrained_ablation.py would raise IndexError.
#
#    --max-length 4096: the 2048 default keeps only 79.84% of fineweb-edu
#    validation through the trainer's own row filter, and the 20% it drops are
#    exactly the long/multi-page rows (measured on the real split; 4096 keeps
#    95.17%). NOTE this is a wider row pool than the run TRAINED on (max_length:
#    2048), so these numbers are not row-comparable with H10's 2048 measurements;
#    H16's criterion is absolute, so that is a comparability caveat, not a
#    validity one.
# ---------------------------------------------------------------------------
ck=$(resolve_ckpt h16-gap-weighted-v0) || ck=""
if [[ -z "$ck" ]]; then
  say "SKIP h16 — no checkpoint under data/checkpoints/h16-gap-weighted-v0"
else
  say "H16: checkpoint $ck"
  probe h16-ablation data/eval/h16-ablation-gap-weighted.json \
    uv run python scratchpad/hybrid_4lane_ablation.py \
      --checkpoint "$ck" --lanes fineweb-edu --max-samples 150 --seed 42 \
      --max-images 4 --max-length 4096 --tag h16-gap-weighted \
      --output data/eval/h16-ablation-gap-weighted.json

  probe h16-position-decay \
    data/eval/hybrid-4lane-position-decay-h16-gap-weighted.json \
    uv run python scratchpad/hybrid_4lane_position_decay.py \
      --checkpoint "$ck" --lanes fineweb-edu -n 150 --seed 42 \
      --max-length 4096 --tag h16-gap-weighted
fi

# ---------------------------------------------------------------------------
# 3. H21 — glyph-scale (font) sweep.
#    docs/hypothesis/todo/H21-vision-path-not-scale-invariant.md "Run order"
#    gives both invocations verbatim. The SANITY GATE runs inside each of them
#    and is not optional: it re-runs h13_analyze's own collect_rung/analyse_rung
#    on the EXISTING randstr-d3/randstr-d5 splits and must reproduce H13 §6
#    (pos-0 +28.7 / +0.0 pts, Δperm +3.05% / +0.01%) from the RECORDED
#    data/eval/h13-poscontrol-h07.json before any font rung is believable.
#
#    Leg 0 is a zero-training probe of H07's checkpoint and has never been run
#    (no data/eval/h21-leg0.json) — it is the baseline curve leg 1 is read
#    against, and it is the invocation whose gate genuinely replicates, since the
#    anchors were measured on that exact checkpoint.
#
#    !! EXPECTED CONFLICT ON LEG 1, recorded before the run: the gate treats
#    "randstr-d5 reads" as a REPLICATION FAILURE (evaluate_gate requires d5's
#    pos-0 gain to be NON-significant), while the doc's own "Secondary read-out"
#    section HOPES the font-trained checkpoint reads d5 — that is how it would
#    revive H13's line-demux branch. So a leg-1 rc=2 with `gate_failed: true` may
#    be the substantive result rather than a defect. The JSON is written BEFORE
#    the non-zero exit, so nothing is lost; --force is deliberately NOT passed,
#    so the gate signal survives in the queue log as rc=2.
#
#    --max-length 2048 is safe and explicit here: the h21 lane's real assembled
#    sequence is mean 332.9 / max 336 tokens (H21 "Measured, not assumed") and
#    the d3/d5 gate rungs are ~600, so nothing can truncate at any budget; the
#    soft-token budget is pinned at 280 in every leg.
# ---------------------------------------------------------------------------
H21_LEG0_CK=data/checkpoints/hybrid-pretrained-randstr-v0/final
if [[ ! -d $H21_LEG0_CK ]]; then
  say "SKIP h21-leg0 — H07 checkpoint missing: $H21_LEG0_CK"
elif [[ ! -d data/materialized/h21-fonts-v0/randstr-f14/validation ]]; then
  say "SKIP h21-leg0 — font rungs missing under data/materialized/h21-fonts-v0"
else
  probe h21-leg0 data/eval/h21-leg0.json \
    uv run python scratchpad/h21_font_sweep.py \
      --checkpoint $H21_LEG0_CK --tag h21-leg0 -n 150 --max-length 2048
fi

# The run tag in gpu_queue_stage2.sh is `h21-fontjitter-v0` but the config writes
# to `h21-fontmix-v0` (configs/h21_fontjitter.yaml:188). Resolve the REAL dir.
ck=$(resolve_ckpt h21-fontmix-v0) || ck=""
if [[ -z "$ck" ]]; then
  say "SKIP h21-leg1 — no checkpoint under data/checkpoints/h21-fontmix-v0"
else
  say "H21 leg 1: checkpoint $ck"
  probe h21-leg1 data/eval/h21-leg1.json \
    uv run python scratchpad/h21_font_sweep.py \
      --checkpoint "$ck" --tag h21-leg1 -n 150 --max-length 2048 \
      --trained-fonts 14 24 40 --heldout-fonts 18 31
fi

# ---------------------------------------------------------------------------
# 4. H20 — audio time resolution, TWO legs on the wide 2000x160 render.
#    docs/hypothesis/todo/H20-audio-phoneme-resolution.md "Pre-registered
#    criterion" (line 362ff) names three probes and forbids one metric:
#      Confirms: position-0 reading gain > +10 pts (currently -12.7),
#                Δblank >= +30%, and greedy decodes with NONZERO word-level
#                overlap. Probes: hybrid_4lane_position_decay.py,
#                hybrid_pretrained_ablation.py, hybrid_4lane_generate.py —
#                "not aggregate eval CE".
#    plus the gates added 2026-07-29:
#      * Anchor guard: ablate the spoken-digits lane at step 200 and at the end.
#        H08 left it at Δperm +134.8%; below +30% the run is destroying the audio
#        reading it started with and a librispeech null cannot be attributed to
#        time resolution.
#      * Progress check: librispeech Δperm at step 200 must exceed its own
#        step-100 value (a direction check, not a bar).
#
#    Δperm on LIBRISPEECH comes from prefix_dperm_probe.py, not
#    hybrid_pretrained_ablation.py: 567/2703 wide librispeech validation rows are
#    multi-page (2:506, 3:52, 4:9) and the single-image ablation raises
#    IndexError on them. The spoken-digits anchor IS run with the doc-named
#    hybrid_pretrained_ablation.py — that lane is 1000/1000 single-page, so the
#    doc's probe applies and the number stays comparable with H08's.
#
#    --max-length 8192 everywhere: leg 2's 4-page validation row is
#    4 x 1062 + 151 + 21 = 4420 tokens, and the probes' 2048 default retains only
#    20,548 of 104,014 librispeech rows at budget 1120 — dropping 80.24% of the
#    lane, keeping exactly the shortest single-page utterances. Soft-token budget
#    is NOT overridden: each checkpoint records its own (560 / 1120) and
#    resolve_max_soft_tokens reads it back.
#
#    ANCHOR CAVEAT (recorded, not resolvable here): H08's +134.8% was measured on
#    the NARROW 1000x160 spoken-digits val at budget 280. These legs train and
#    eval on the WIDE split at 560/1120, which is what the runs' own
#    validation_splits point at, so that is what is probed. Cross-render, the
#    +30% floor is a coarse "did we destroy audio reading" check, not a matched
#    comparison.
# ---------------------------------------------------------------------------
LIBRI_WIDE=data/materialized/h20-audio-wide-v0/librispeech/validation
SPDIGIT_WIDE=data/materialized/h20-audio-wide-v0/spoken-digits/validation
SPDIGIT_FLOOR=data/materialized/spoken-digits-v0/floor.json

for leg in h20-leg1-wide-560-v0 h20-leg2-wide-1120-v0; do
  ck=$(resolve_ckpt $leg) || ck=""
  if [[ -z "$ck" ]]; then
    say "SKIP $leg — no checkpoint under data/checkpoints/$leg (leg not trained)"
    continue
  fi
  if [[ ! -d $LIBRI_WIDE ]]; then
    say "SKIP $leg — wide librispeech validation missing: $LIBRI_WIDE"
    continue
  fi
  say "H20 $leg: checkpoint $ck"

  # (a) the decisive final read-outs first
  probe ${leg}-libri-dperm data/eval/${leg}-libri-dperm-final.json \
    uv run python scratchpad/prefix_dperm_probe.py \
      --checkpoint "$ck" --val-path $LIBRI_WIDE -n 150 --seed 42 \
      --max-length 8192 --prefix-tokens 48 \
      --tag ${leg}-libri-final --out data/eval/${leg}-libri-dperm-final.json

  probe ${leg}-libri-position-decay \
    data/eval/hybrid-4lane-position-decay-${leg}-libri-final.json \
    uv run python scratchpad/hybrid_4lane_position_decay.py \
      --checkpoint "$ck" --val-path $LIBRI_WIDE --lanes librispeech \
      -n 150 --seed 42 --max-length 8192 --tag ${leg}-libri-final

  # (b) the spoken-digits ANCHOR guard, at the end and at step 200
  if [[ -d $SPDIGIT_WIDE ]]; then
    probe ${leg}-anchor-final data/eval/${leg}-anchor-spdigit-final.json \
      uv run python scratchpad/hybrid_pretrained_ablation.py \
        --checkpoint "$ck" --val-dataset $SPDIGIT_WIDE --floor $SPDIGIT_FLOOR \
        --max-samples 150 --seed 42 --max-length 8192 \
        --tag ${leg}-spdigit-final \
        --output data/eval/${leg}-anchor-spdigit-final.json
  else
    say "SKIP ${leg}-anchor — wide spoken-digits validation missing: $SPDIGIT_WIDE"
  fi

  # (c) greedy decodes — H20's third CONFIRMS component (word-level overlap)
  probe ${leg}-decodes data/eval/${leg}-decodes.json \
    uv run python scratchpad/hybrid_4lane_generate.py \
      --checkpoint "$ck" --lanes librispeech --val-path $LIBRI_WIDE \
      -n 8 --max-new 48 --seed 42 --max-length 8192 \
      --out-json data/eval/${leg}-decodes.json

  # (d) trajectory gates: anchor at step 200, librispeech Δperm at 100 vs 200
  for step in 100 200; do
    sck=data/checkpoints/${leg}/checkpoint-${step}
    if [[ ! -d $sck ]]; then
      say "SKIP ${leg}-step${step} — $sck does not exist"
      continue
    fi
    probe ${leg}-libri-dperm-${step} data/eval/${leg}-libri-dperm-${step}.json \
      uv run python scratchpad/prefix_dperm_probe.py \
        --checkpoint "$sck" --val-path $LIBRI_WIDE -n 150 --seed 42 \
        --max-length 8192 --prefix-tokens 48 \
        --tag ${leg}-libri-${step} --out data/eval/${leg}-libri-dperm-${step}.json
    if [[ $step == 200 && -d $SPDIGIT_WIDE ]]; then
      probe ${leg}-anchor-200 data/eval/${leg}-anchor-spdigit-200.json \
        uv run python scratchpad/hybrid_pretrained_ablation.py \
          --checkpoint "$sck" --val-dataset $SPDIGIT_WIDE --floor $SPDIGIT_FLOOR \
          --max-samples 150 --seed 42 --max-length 8192 \
          --tag ${leg}-spdigit-200 \
          --output data/eval/${leg}-anchor-spdigit-200.json
    fi
  done
done

# H19 is deliberately NOT read out: it was never queued for TRAINING (6 PENDING
# config fields downstream of H17/H16 and a mixture rebuild). Nothing to probe.
say "H19 not probed — never trained (see gpu_queue_stage2.sh)."

say "GPU_QUEUE_STAGE3_DONE  (free VRAM $(free_mib) MiB)"
