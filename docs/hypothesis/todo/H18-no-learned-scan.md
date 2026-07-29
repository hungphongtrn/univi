# H18 — The model has no mechanism to track position in the image

**Status:** TODO — **PARTIALLY SUPPORTED, not confirmed.** [H13](../done/H13-density-ladder-long-targets.md)
showed the collapse point is invariant to page density, which is consistent with this claim but does
**not** exclude the rival **(A′) fixed absolute capacity**. **Do not close this as CONFIRMED until
[H17](H17-raise-soft-token-budget.md) reports** — see
[Status: conditional on H17](#status-conditional-on-h17-2026-07-29). · Constraint **(C)** · Related:
[H11](../done/H11-position-decay-is-prior-induced.md),
[H13](../done/H13-density-ladder-long-targets.md), [H14](H14-masked-region-targets.md),
[H17](H17-raise-soft-token-budget.md)

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
  validation data gave **K = 48.8 / 48.5 / 48.5 tokens across d2 → d4** — a **16× span of page
  density** (119 → 1919 chars/page). The collapse point does not move.
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
  character position and token index are proportional *by construction*. A K constant across rungs is
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
limit but not a scan limit.** That is [H17](H17-raise-soft-token-budget.md), which H13 §8 promoted to
the critical path for precisely this reason. Its pre-registered bar is unchanged: **K must scale
≥ 3× for 4.25× the tokens** (≥ 1.73× per leg).

| H17 outcome | consequence for H18 |
|---|---|
| **K scales on both legs** | **(A′) confirmed, H18 REFUTED as the primary account.** More tokens buy more depth ⇒ the limit was capacity, not a missing cursor. This file closes as `done/` with verdict REFUTED. |
| **K flat on both legs** | **H18 CONFIRMED**, conditional on H17's sanity gate having passed and on every leg having *trained* at its own budget. 4.25× the tokens with optics controlled bought < 3× the depth ⇒ the limit is the scan. H18 becomes the main line and the interventions below get designed. |
| **K scales on leg 1 only** | **Bounded, no verdict.** Partial capacity relief that saturates at 560, with (C) still binding above it. H18 stays open. |
| **partial rise (1×–1.73× per leg)** | **No verdict, and specifically not a confirmation.** H17's own caveat: raising the budget also refines the *spatial addressing grid* (16×16 → 33×33), which is plausibly partial relief for a scan limit too. Report as partial. |

Until one of those lands, this file **stays in `todo/`** and its claim stays at *partially supported*.

## Candidate interventions, if H17 confirms (C)

Unchanged in substance; the trigger has moved from H13 to H17.

- **Chunked curriculum** — train on short spans whose image is a *crop*, so image position and answer
  position are aligned and the mapping is learnable; then widen. Precedent exists in line-level OCR
  pretraining, see [synthetic prior-proof pretraining](../../literature/synthetic-pretraining-bootstraps-reading.md).
- **Localization supervision** — [H14](H14-masked-region-targets.md) is the cheapest existing probe
  that *requires* localization; a positive H14 with a flat H17 would be strong evidence for (C).
  **Note H13 §8's warning before launching it:** H14 as configured measures 209.7 target tokens/row,
  i.e. **22.9% readable** within depth ≈ 48 — squarely in H13's failure regime (14.2% died; H07's
  100% succeeded). It was redesigned before launch for that reason.
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
[H17](H17-raise-soft-token-budget.md)** and is stated in the table above.

Any read-out for this hypothesis is in **Δperm / Δblank / per-position reading gain** — never
aggregate eval CE. H13 §7 is the standing reason: its `floor.json` computed the no-reading floor over
*target* tokens while response-only masking also supervises the deterministic `<|im_end|>\n` trailer,
so every published floor was too high by ~`T/(T−2)` and every "CE below floor ⇒ reading" mark in
H13's trajectory table was an artifact. The ablation-based verdict was untouched by the bug, which is
the whole argument for the convention.
