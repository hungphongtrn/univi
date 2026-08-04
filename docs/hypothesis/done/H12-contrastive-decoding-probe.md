# H12 — Mid-page content is present but out-voted by the prior

**Status:** DONE · **Verdict: REFUTED** (2026-07-28) · Cost: 15 min GPU (no training) · Related: [H10](H10-reading-concentrated-at-start.md), [H11](H11-position-decay-is-prior-induced.md), [H13](H13-density-ladder-long-targets.md), [H17](../done/H17-raise-soft-token-budget.md)

## Claim

The soft-token stream *does* carry mid-page content, but the language prior outvotes it at argmax.
If so, PMI-style contrastive decoding should recover text that greedy decoding loses.

## Setup

Zero training. Greedy-decode `hybrid-4lane-v0/final` with

```
logits = (1 + α)·logits(aligned) − α·logits(blank)
```

for α ∈ {0.5, 1, 2} plus the α = 0 baseline, 50 rows each of fineweb-edu and librispeech (seed 42,
same rows across settings), `max_new 110` so the decode covers answer tokens 10–100.
`scratchpad/h12_contrastive_decode.py` (cloned from `hybrid_4lane_generate.py`), analysis in
`scratchpad/h12_analyze.py`.

**Numerics note:** all 7 branches (greedy + 3 α × aligned/blank) are batched into **one forward per
step**, so every setting shares bit-identical numerics. This was not cosmetic — cached-vs-uncached
greedy decodes diverged on 6/8 settings, traced to bf16 prefill-vs-decode kernel rounding (40/40
argmax agreement under teacher forcing, max |logit| diff 0.25 on a scale of 26.5) cascading once a
near-tie flips. Batching removes the confound.

## Result — target-word overlap at answer tokens 10–100

| lane | α=0 | α=0.5 | α=1 | α=2 | best ratio | criterion |
|---|---|---|---|---|---|---|
| fineweb-edu | 0.1573 | 0.1678 | 0.1695 | **0.1728** | **1.10×** | ≥ 2.00× |
| librispeech | 0.0984 | **0.1167** | 0.1079 | 0.1003 | **1.19×** | ≥ 2.00× |

Absolute deltas: **+0.0155** (fineweb), **+0.0183** (librispeech). No position-specific recovery —
`ov@0-10` moves as little as `ov@10-100`.

**Shuffled-target control** (score each decode against *another* row's target; grounded excess =
own − shuffled):

| lane | excess α=0 | excess best-α |
|---|---|---|
| fineweb-edu | 0.0037 | 0.0154 |
| librispeech | 0.0209 | 0.0405 |

~95% of the raw ~16% overlap is generic function-word overlap that a *random* target earns equally.
Content-word overlap (stopwords stripped) rises from 1.4% → 2.2% (fineweb) and 0.0% → 2.0%
(librispeech): roughly one target content word recovered every few rows.

## Learning

- **The refute branch fires.** No α doubles overlap; the information is not sitting in the
  soft-token stream waiting to be un-suppressed at this density. Per the pre-registration this
  **promotes [H13](H13-density-ladder-long-targets.md) and
  [H17](../done/H17-raise-soft-token-budget.md)** (bandwidth) over the objective-side
  [H15](../done/H15-prior-poisoned-text.md)/[H16](../done/H16-prior-gap-weighted-loss.md).
- **Scope of the claim.** A negative contrastive result bounds what a *decoding-time* intervention
  can recover. It is strong evidence against "present but out-voted", not a proof that the pixels
  carry nothing — a fix that changes *training* could still put content into the stream that is not
  there now. H15/H16 are demoted, not refuted.
- **The degeneracy confound ran backwards, and it inflates every gain above.** High α did *not*
  degenerate; the **α=0 baseline is the degenerate one**. Greedy loops phrase-level
  (`'The History of the World'` ×16; `'AND THE FATHER OF THE FATHER OF…'`), and contrastive merely
  breaks the loop into fluent, **wholly hallucinated** text — target "benefits of cinnamon" → α=2
  wrote "The Benefits of a Healthy Diet". The measured lift is loop-breaking, not reading.
- **Adjacent-token repetition is a broken degeneracy metric** — it reads ~0.00 on
  `'The History of the World'` ×16 because the loops are phrase-level. The **distinct-token ratio**
  caught it (0.41→0.53 fineweb, 0.27→0.80 librispeech, with librispeech length collapsing 54→31 as
  loops stop). Use distinct-token ratio in future decode probes.
- **Consistent with [H10](H10-reading-concentrated-at-start.md):** the first few tokens *are* read
  ("The benefits of cinnamon…" → "The Benefits of…"), then the prior takes over. Contrastive
  decoding does not extend that reach.
- **Caveat — the librispeech blank was dark red, not gray.** Found 2026-07-28 (see the
  methodological caveat in [`research.md`](../../../research.md)): `Image.new("RGB", size, 128)`
  fills band 0 only. librispeech is stored RGB, so its contrastive `logits(blank)` term used a
  dark-red image; fineweb-edu is mode `L` and was genuinely gray. This does not overturn the
  verdict — the *more* out-of-distribution blank makes the contrastive correction stronger if
  anything, and the unaffected fineweb lane shows the same null — but the librispeech magnitudes
  should not be quoted quantitatively. The probes have since been fixed to emit true gray.
- **Caveat — librispeech window is thin.** Median target is 26 tokens, so the 10–100 window is short
  and 4/50 rows are excluded outright; its `ov@full` column is the more trustworthy one there, and
  it tells the same story.

## Artifacts

`data/eval/h12-contrastive-decode.json` (per-lane/per-α metrics + all 400 raw decodes),
`data/eval/h12-contrastive-decode.log`, `scratchpad/h12_contrastive_decode.py`,
`scratchpad/h12_analyze.py`. Peak VRAM 4.3 GiB (budget 26 GB).
