# H09 — Balanced lane shares stop audio drowning in the real mixture

**Status:** DONE · **Verdict: PARTIAL** (2026-07-28) · Related: [H01](H01-gemma-e2b-mixture-audio-drowns.md), [H10](H10-reading-concentrated-at-start.md), [H19](../todo/H19-four-lane-rematch.md)

## Claim

Audio drowned in [H01](H01-gemma-e2b-mixture-audio-drowns.md) because librispeech was 3.5% of the
mixture. Capping every lane to librispeech's size (equal 25% share), with the
[H07](H07-pretrained-vision-adapter-qwen.md) architecture and trainable vision, will make the hybrid
read all four lanes at once.

## Setup

`configs/hybrid_4lane.yaml`, new `dataset.max_train_rows_per_subset: 104014` knob. Vision trainable,
3-group LR (adapter 3e-4 / vision 1e-4 / decoder 2e-5). 4000 steps, eff batch 64, 27h20m, 0.6153
epoch, ran clean. W&B `17b5omgn` — **note the W&B upload died fatally at ~step 1927; the log is
ground truth.**

## Result — the drowning fix worked, the reading did not

| lane | aligned CE | Δperm | Δblank | reading gain |
|---|---|---|---|---|
| fineweb-edu | 2.506 | +2.51% | +3.48% | +1.25 pts |
| densefusion | 1.557 | +0.70% | +1.93% | +0.42 pts |
| smoltalk | 1.305 | +10.07% | +6.87% | +1.45 pts |
| librispeech | 2.885 | **+12.91%** | +7.21% | +1.98 pts |

**Win:** librispeech went from the ignore-floor (+0.16% in [H01](H01-gemma-e2b-mixture-audio-drowns.md),
negative Δblank) to the **best** lane. Permuted worse than blank on smoltalk/librispeech = genuine
content-conditioning. This is **not** the [H05](H05-isolated-ocr-from-scratch.md) basin.

**Failure:** all reading is shallow and **saturated** — ablation at checkpoint-2500 is nearly
identical (Δperm 2.26 / 0.64 / 9.87 / 12.46), i.e. ~+0.3 pp of growth over 1500 steps, against
+27% → +115% in 400 steps on the prior-proof lane.

## Learning

- **"Train it longer" is falsified.** Corroborating: train loss flat ~1.95–2.0 from step ~130;
  per-lane eval frozen to 3 decimals by step ~2800–3200; **grad_norm dead flat 0.42–0.58 from step
  150** — the [H07](H07-pretrained-vision-adapter-qwen.md) dip-then-rise recruitment signature never
  appeared.
- **Eval CE is orthogonal to grounding.** Our fineweb 2.470 ≈ encoder-free
  [H04](H04-frozen-decoder-warmup.md)'s 2.482, and H04 read *zero* pixels. The config's per-lane
  "drowning monitor" measured decoder-LR × prior-fitting for 27 GPU-hours. **Every future run needs a
  live Δblank monitor.**
- **Plumbing is validated** (~95%): `pretrained.py:178-183` hard-raises if placeholder count ≠
  projected soft tokens, so 4000 steps + 40 eval passes without a crash proves no image token was
  ever truncated away.
- Greedy decodes tell the real story: blank decodes are **byte-identical constants** per lane; aligned
  decodes capture layout/genre/opening words then fabricate fluently
  (`Summary of The Haftorah:` → `Summary of the Book`). librispeech aligned decodes have **zero
  phonetic correspondence** but their length tracks the target's → **duration conditioning**, not
  transcription.
- Led directly to [H10](H10-reading-concentrated-at-start.md).

## Artifacts

`data/eval/hybrid-4lane-ablation-{final,2500}.json`, `scratchpad/hybrid_4lane_ablation.py`,
`scratchpad/hybrid_4lane_generate.py`, `data/checkpoints/hybrid-4lane-v0/`
