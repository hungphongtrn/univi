# H11 — The position decay is caused by the language prior

**Status:** DONE · **Verdict: REFUTED** (2026-07-28) · Related: [H10](H10-reading-concentrated-at-start.md), [H18](../todo/H18-no-learned-scan.md), [H13](H13-density-ladder-long-targets.md)

## Claim

[H10](H10-reading-concentrated-at-start.md)'s decay is a gradient-allocation artifact: reading is
only profitable where the prior fails, and the prior is weakest at position 0. Remove the prior and
the decay should vanish.

## Setup

Same probe (`scratchpad/hybrid_4lane_position_decay.py --val-path …`) on the two **prior-proof**
checkpoints, where blank accuracy is ~0–5% at every position so there is no prior to fall back on:
`hybrid-pretrained-randstr-v0/final` and `hybrid-pretrained-spokendigits-trainvis-v0/final`.

## Result — reading gain (pts) by position

| position | randstr | spoken-digits |
|---|---|---|
| 0 | **+56.00** (56% vs 0%) | **+82.67** (91% vs 9%) |
| 1 | +53.33 | 0.00 (format token, both 100%) |
| 2–4 | +47.11 | +28.89 |
| 5–9 | +3.47 | +8.40 |
| 10–19 | +13.85 | +10.57 |

## Learning

- **The decay is not purely prior-induced.** It persists with the prior removed entirely, and it
  appears at the *same token positions* despite ~**100× lower character density** (25 chars/page vs
  fineweb's 2,832). Whatever causes it is not chars-per-soft-token alone.
- This opens a third mechanism: **[H18](../todo/H18-no-learned-scan.md) — no learned scan.** The model may
  have no way to track *where* in the image it is as it generates.
- **But prior competition is also large.** At positions 10–19 the prior-proof lanes sustain **+13.9
  and +10.6 pts**; the real lanes manage **+0.13 (fineweb)** and **+1.32 (smoltalk)** — roughly 10×
  less at the same depth.
- Verdict: the constraints are **coupled**. Fixing the objective alone would leave a positional
  ceiling in place; fixing bandwidth alone would leave the prior suppressing depth.
- **Limit of this evidence:** randstr targets are only ~14.5 tokens, so genuinely deep positions are
  unobservable, and the non-monotonicity (+3.47 at 5–9 recovering to +13.85 at 10–19) is likely
  tokenization/format structure rather than signal. This test separates *prior-driven* from
  *not-prior-driven*; it cannot yet separate bandwidth from scan failure. That is
  [H13](H13-density-ladder-long-targets.md)'s job.

## Artifacts

`data/eval/posdecay-randstr.log`, `data/eval/posdecay-spdigit.log`
