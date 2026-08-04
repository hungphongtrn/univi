# H18 — The model has no mechanism to track position in the image

**Status:** TODO — **PARTIALLY SUPPORTED, not confirmed.** [H13](../done/H13-density-ladder-long-targets.md)
showed the collapse point is invariant to page density, which is consistent with this claim but does
**not** exclude the rival **(A′) fixed absolute capacity**.
**⚠️ This no longer resolves from [H17](../done/H17-raise-soft-token-budget.md)** (updated 2026-07-30).
H17 came back **VOID** — it failed to optimize at the raised budgets rather than answering — and,
independently, **this doc's own verdict table was rigged**: it branched everything on K, which is
structurally floored and can only ever report "flat", so the table could only ever fire *H18
CONFIRMED*. Both are repaired in
[The one remaining discriminator](#the-one-remaining-discriminator), which now states the
(A′)-vs-(C) prediction in `char_bins` / `pos0_gain`. **Do not close this as CONFIRMED off any
K-based bar.** · Constraint **(C)** · Related:
[H11](../done/H11-position-decay-is-prior-induced.md),
[H13](../done/H13-density-ladder-long-targets.md), [H14](../done/H14-masked-region-targets.md),
[H17](../done/H17-raise-soft-token-budget.md)

## Claim

The model can read *the start* of an image and never acquires a **scan**: it has no learned way to
track where in the image it currently is as it generates. Reading therefore decays with generated
token index regardless of how much information the soft-token stream carries.

## Where this came from

[H11](../done/H11-position-decay-is-prior-induced.md). The positional decay persists on prior-proof
lanes where there is no prior to blame, **and it appears at the same token positions despite ~100×
lower character density**. Neither (A) bandwidth nor (B) prior competition predicts that
conjunction; a missing cursor does.

Supporting texture from [H09](../done/H09-balanced-mixture-prevents-drowning.md)'s decodes: the model
reproduces a page's opening (`Summary of The Haftorah:` → `Summary of the Book`) and then generates
fluent unrelated text — the behaviour of something that read one region and then lost its place.

## Status: conditional on H17 (2026-07-29)

[H13](../done/H13-density-ladder-long-targets.md) was pre-registered as the (A)-vs-(C) discriminator
and this file was parked until it reported. It has reported. It moved this hypothesis forward, but
**not to a verdict**, and the distinction matters enough to state line by line.

### What H13 DID establish for H18

- **The collapse point is invariant to page density.** Running the pre-registered discriminator
  against the [H07](../done/H07-pretrained-vision-adapter-qwen.md) checkpoint on the ladder's own
  validation data gave **the same readable depth across d2 → d4** — a **16× span of page density**
  (119 → 1919 chars/page). The collapse point does not move.
  ⚠️ **This originally read "K = 48.8 / 48.5 / 48.5 tokens" and that is retired** (2026-07-30): all
  three numbers are the K estimator's structural floor (`k_chars ≥ 100` by construction), not
  measurements. The surviving, position-resolved form of the finding is `char_bins[0-10]` gain of
  **+27.5 / +18.6 / +15.6 pts** at d2 / d3 / d4 with ≈ 0 beyond char 25 on every rung — same shape,
  same invariance, **readable depth ~5–12 answer tokens, not 48**.
- **Perception at the start of the page does not degrade with density at all.** Position-0 reading
  gain is statistically identical at **d3 (+28.7 ± 3.7)** and **d4 (+28.0 ± 3.7)**, despite d4
  carrying ~24 chars per inked soft token against d3's ~15. Aggregate Δperm collapses 14.31% → 0.63%
  across those rungs *purely because a fixed readable prefix is a shrinking fraction of a longer
  target*.
- **Therefore (A) density-proportional bandwidth is EXCLUDED.** Its prediction — position-0 should
  degrade as characters-per-token rises — is directly contradicted.
- This is the qualitative shape H18 predicts: read the start, then lose the place, at a depth that
  has nothing to do with how much is on the page.
- H11's original motivating conjunction is reproduced on a *deep* target for the first time. H11
  could only observe ~14.5-token randstr targets; H13 observed the same collapse depth at 60, 234
  and 932 target tokens.

### What H13 did NOT establish for H18

- **(C) is not isolated.** H13 §5.1 states it explicitly: **chars/token ≈ 2.06 on every rung**, so
  character position and token index are proportional *by construction*. A readable depth constant
  across rungs (originally phrased as "a K constant across rungs", before K was retired) is
  **equally consistent** with **(C) fixed token-index scan** and with **(A′) fixed absolute
  capacity** — "the model extracts a fixed quantity of content per page regardless of how much is on
  it". What is strictly established is *the collapse point is invariant to page density*, and
  nothing more.
- **The measurement is on a model trained at ONE density.** H07 saw only d1-geometry font-14
  randstr. Its K may reflect its training distribution rather than an architectural limit. H13
  establishes invariance to **test-time** density; it does **not** establish invariance to
  **training** density.
- **d1's K is censored** (its target ends inside the 0–100-char reference window), so the trend rests
  on d2/d3/d4 — three points, not five.
- **The H13 *training* run is VOID** and contributes nothing here: `grad_norm` 0.061 by step 110,
  Δperm ≈ 0 on all five rungs, the content-blind signature. Its root cause (85.8% of supervised
  tokens beyond the scan depth, so the from-scratch adapter never bootstrapped) is itself *weak
  circumstantial support* for a shallow depth — but it is a failed run, not a measurement, and must
  not be cited as evidence for (C).
- **Nothing here is a mechanism.** "The collapse depth is fixed" is a description. Whether the model
  lacks a cursor, or has one that saturates, or is bottlenecked somewhere else entirely, is untested.

### The one remaining discriminator

**(A′) and (C) differ in exactly one testable way: raising the soft-token budget relieves a capacity
limit but not a scan limit.** That was [H17](../done/H17-raise-soft-token-budget.md), which H13 §8
promoted to the critical path for precisely this reason.

> ### ⚠️ The table that used to sit here was rigged, and H17 came back VOID (repaired 2026-07-30)
>
> It read *"Its pre-registered bar is unchanged: **K must scale ≥ 3× for 4.25× the tokens** (≥ 1.73× per
> leg)"* and branched **the entire hypothesis** on K across four rows. **K cannot scale.**
> `k_from_buckets` requires the first 25-char bucket *past* a 100-char baseline window, so
> `k_chars ≥ 100` is structurally guaranteed on any rung that reads at all and whose target runs past
> 100 chars. Three of the four rows were unreachable and the table could only ever fire
> **"K flat on both legs ⇒ H18 CONFIRMED."**
>
> H17 demonstrated this on real data: K returned **48.4913 tokens at budget 280 and 48.4913 at 560** —
> identical to four decimals, ratio 1.0000 — for legs whose actual reading differed **2.8×**
> (`char_bins[0-10]` gain +55.7 vs +19.7). **This doc would have declared itself CONFIRMED off an
> estimator artifact, on the programme's most consequential open question.** Do not restore any
> criterion stated in K. See
> [research.md's caveat](../../../research.md#methodological-caveat--k-was-an-estimator-floor-not-a-measurement)
> and [H17's verdict](../done/H17-raise-soft-token-budget.md#verdict-2026-07-30).

**H17 reported 2026-07-30: VOID.** All three legs trained (280 / 560 / 1120, identical except
`max_soft_tokens`, all warm-started from the same H07 checkpoint) and reading *fell* monotonically —
`char_bins[0-10]` gain **+55.7 → +19.7 → −0.1 pts** — while `grad_norm` fell **0.264 → 0.079 → 0.038**
and the 1120 leg's eval CE never reached the no-reading floor. "More tokens don't help" and "a
280-native adapter cannot be re-addressed onto a 33×33 grid in 500 steps" predict the same data. **So
H18 gets no verdict from H17: (A′) is not refuted, (C) is not confirmed, and this file is neither
promoted nor closed.**

### The replacement discriminator, stated position-resolved

Any successor budget ladder must be read off `pos0_gain` / `pos0_4_gain` / `char_bins` — never K,
never D50, never aggregate Δperm, never eval CE (see
[the criterion-defect table](../README.md#criterion-defects-found-by-running-the-criteria--audit-any-new-bar-against-these)).
The two accounts make *different* position-resolved predictions:

| | **(A′) fixed absolute capacity** predicts | **(C) fixed scan depth** predicts |
|---|---|---|
| `pos0_gain` | roughly unchanged (position 0 was never capacity-limited) | roughly unchanged |
| `pos0_4_gain`, `char_bins[0-10]` | **grows** with the budget | flat |
| `char_bins[25-50]` (zero at 280) | **lifts off zero** | **stays at zero** |

**Bar: (A′) requires `char_bins[25-50]` gain ≥ +5 pts with a row-clustered CI excluding 0 at some
raised budget.** Anything less is (C)-compatible. Report the **full `char_bins` curve per leg**, not a
scalar. A rise confined to `char_bins[0-10]` with 25–50 still at zero is *amplitude*, not depth, and
licenses neither account — [H21](../done/H21-vision-path-not-scale-invariant.md) already showed glyph
scale buys amplitude with the cliff unmoved.

**Two preconditions, both of which H17 failed.** The leg must be shown to have *optimized*
(`grad_norm` not sustained below 0.10 — all four of H13, H16, H17-560 and H17-1120 breached this) and
its eval CE must at least reach the lane's no-reading floor. **A leg that did not optimize measures its
own optimization, not the budget**, and H17's data cannot distinguish the two. That is why
[the grad_norm collapse](../README.md#todo) now sits ahead of this hypothesis in the queue.

Until a budget ladder is run that satisfies both preconditions, this file **stays in `todo/`** and its
claim stays at *partially supported* — on the strength of
[H13](../done/H13-density-ladder-long-targets.md)'s density invariance and
[H21](../done/H21-vision-path-not-scale-invariant.md)'s per-scale memorisation, not on H17.

**One weak hint from H17, recorded as a hint only:** at budget 560 `pos0_gain` was *unchanged*
(+68.0 vs +64.7, CIs overlap) while `pos0_4_gain` collapsed **58.3 → 21.3**. First-fixation legibility
survived the budget change and place-keeping did not, which is (C)-shaped — but it comes from a leg
that failed its own hard bar, so it is not evidence.

## Candidate interventions, if H17 confirms (C)

Unchanged in substance; the trigger has moved from H13 to H17.

- **Chunked curriculum** — train on short spans whose image is a *crop*, so image position and answer
  position are aligned and the mapping is learnable; then widen. Precedent exists in line-level OCR
  pretraining, see [synthetic prior-proof pretraining](../../literature/synthetic-pretraining-bootstraps-reading.md).
- **Localization supervision** — [H14](../done/H14-masked-region-targets.md) is the cheapest existing probe
  that *requires* localization; a positive H14 with a flat H17 would be strong evidence for (C).
  **Note H13 §8's warning before launching it:** H14 as configured measured 209.7 target tokens/row,
  i.e. **22.9% readable** — squarely in H13's failure regime (14.2% died; H07's 100% succeeded). It was
  redesigned before launch for that reason, and it CONFIRMED.
  ⚠️ **Every readable-fraction figure in this programme was computed against depth 48 and is therefore
  overstated ~4–10×** (2026-07-30). Recompute at the measured ~5–12 tokens before using any of them as
  a gate, and recompute the *reference points* (14.2% / 75.2% / 100%) at the same depth or the
  comparison is depth-inconsistent on both sides. Useful calibration:
  [H21](../done/H21-vision-path-not-scale-invariant.md)'s leg 1 trained **healthily**
  (`grad_norm` 0.65–0.86) on a lane whose corrected readable fraction is ~20%, so ~20% is empirically
  survivable.
- **Architectural** — cross-attention to soft tokens with explicit 2D position, rather than a flat
  prefix consumed by causal self-attention. `univi/hybrid/vision.py` has **no cross-patch mixing
  anywhere** (LN → Linear → LN → +factorized position → LN → RMSNorm → Linear, all per-patch), so
  every cross-cell integration is currently the decoder's job. This is the expensive option and needs
  H17 + H14 evidence first. Note it is also H17's refute-branch recommendation, arrived at
  independently — a ViT/SigLIP front-end would supply exactly the cross-patch attention this bullet
  asks for.

## Pre-registered discriminator

Not a run of its own. The discriminator **was** [H13](../done/H13-density-ladder-long-targets.md)
(*collapse at a fixed token index across densities ⇒ (C); collapse that moves with density ⇒ (A) and
this hypothesis is dropped*). H13 fired the first branch, which eliminated
density-proportional (A) but left (A′) standing, so the discriminator has **moved to
[H17](../done/H17-raise-soft-token-budget.md)** and is stated in the table above.

Any read-out for this hypothesis is in **position-resolved reading gain — `pos0_gain`,
`pos0_4_gain`, `char_bins`** — and never in K, D50, **aggregate Δperm/Δblank**, or eval CE.
⚠️ **The wording here used to be "Δperm / Δblank / per-position reading gain — never aggregate eval
CE", which banned eval CE while explicitly licensing aggregate Δperm** — and that is the loophole
every one of the four mis-specified criteria came through (2026-07-30). Aggregate Δperm is a function
of target length: H13 measured it collapsing **14.31% → 0.63% for identical reading** as targets
lengthened, and the empirical maximum ever recorded at ~60 supervised tokens is 25.80%, by a model
reading at +57.3 pts. If a Δ statistic is unavoidable, **prefix-match it to the first 5–10 answer
tokens and state the window and the target length.** H13 §7 is the standing reason: its `floor.json` computed the no-reading floor over
*target* tokens while response-only masking also supervises the deterministic `<|im_end|>\n` trailer,
so every published floor was too high by ~`T/(T−2)` and every "CE below floor ⇒ reading" mark in
H13's trajectory table was an artifact. The ablation-based verdict was untouched by the bug, which is
the whole argument for the convention.
