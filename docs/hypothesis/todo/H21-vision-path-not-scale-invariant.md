# H21 — Reading survives a glyph-scale change it was not trained on

**Status:** TODO · **Cost:** ~0.5 h CPU (materialize) + ~0.5 h GPU (zero-training font sweep) + ~2.6 h
GPU (one training run) ≈ **3 h** · Separates **render geometry** from the token budget · Origin:
[H13](../done/H13-density-ladder-long-targets.md) §6, incidental finding · Related:
[H17](H17-raise-soft-token-budget.md), [H20](H20-audio-phoneme-resolution.md),
[H07](../done/H07-pretrained-vision-adapter-qwen.md)

## Claim

Reading is **scale-invariant**: a model that transcribes a rendered page at font 14 also transcribes
the *same characters* rendered at a font it never trained on.

**The evidence so far REFUTES this**, which is why the claim is stated in the direction that the data
already contradicts — the open question is not whether scale-generalisation happens today (it does
not) but whether it is **learnable at all**, and that has never been tried.

## Where this came from — H13 §6

[H13](../done/H13-density-ladder-long-targets.md)'s ladder contained two rungs that draw from the
same RNG stream and therefore carry **byte-identical targets on 30/30 paired rows**, differing only
in font. Under the [H07](../done/H07-pretrained-vision-adapter-qwen.md) checkpoint, probed on the
same rows:

| rung | font | chars/page | inked soft tokens | chars / inked token | pos-0 reading gain | Δperm |
|---|---|---|---|---|---|---|
| **d3** | 14 | 479 | 32.0 (12.5% of grid) | 14.97 | **+28.7 ± 3.7** | **+3.05 %** |
| **d5** | 40 | 479 | 148.4 (58.0% of grid) | 3.23 | **+0.0** | **+0.01 %** |

A **2.9× glyph-scale change destroys reading completely** on text the same model transcribes at the
smaller size (only 32% of d5 rows beat blank — chance). This is not a rung that failed to vary what
it was built to vary: H13 measured the manipulation directly and found d5 spreads the identical 479
characters over **4.638× more inked soft tokens [95% CI 4.618, 4.658], on 30/30 rows**.

**The honest framing, which H13 §5.4 states and this hypothesis inherits.** H07 trained on exactly
one font (14, d1 geometry). So d5's null is a **scale-generalisation failure, not a demonstrated
architectural limit** — the model was never asked to be scale-invariant and never given a reason to
become so. H13 also records the consequence: its own pre-registered *"D5 ≫ D3 ⇒ the mechanism is
line-demux, fix by render geometry"* branch is **untested, not refuted**, because the model could not
read d5 at all.

## Why this is worth 3 hours

Three live proposals in the programme assume glyph scale is a free lever:

- [H17](H17-raise-soft-token-budget.md)'s *"zero-code complement: re-render at font 28–52"*, which its
  fourth correction calls **"a cheap preview of the main intervention"** — font 40 moves
  chars-per-inked-token 4.64× against the 3.73× a 4.25× budget raise buys, at zero token cost.
- H13's untested line-demux branch (fix render geometry *before* token budget, because it is free).
- [H20](H20-audio-phoneme-resolution.md)'s wide-render leg, which is the same "just re-render it
  bigger" move on the time axis.

If glyph scale is not learnable, the free lever is dead and [H17](H17-raise-soft-token-budget.md)'s
costly budget knob is the only remaining bandwidth lever. If it *is* learnable, render geometry
becomes a serious alternative to ~26 h of GPU. Either way it is the cheapest question left.

## Design

### Leg 0 — zero-training font sweep of the H07 checkpoint (~0.5 h GPU)

Turn H13's two points into a curve. Materialize d2-geometry rungs (119 chars) at fonts
**10 / 14 / 20 / 28 / 40 / 52**, all from the **same seed** so targets are byte-identical across
rungs and every comparison is **paired by row** (H13 §5.5: unpaired cross-rung comparison was a real
power loss on that ladder). Probe `hybrid-pretrained-randstr-v0/final` on all six.

This measures the *shape* of the failure — a cliff at some scale, or a gradual decay away from font
14 — which decides nothing on its own but tells leg 1 where to put its training fonts. It is also the
sanity gate.

> **Sanity gate.** The same probe invocation must reproduce H13 §6's anchors on the **existing**
> `randstr-d3` / `randstr-d5` validation splits: pos-0 **≈ +28.7** at font 14 and **≈ +0.0** at font
> 40, Δperm ≈ +3.05% / +0.01%. Same checkpoint, same data, same script — if this does not replicate,
> the probe or the checkpoint has drifted and nothing below can be believed. Failing the gate costs
> 25 minutes and kills the experiment before any materialization is trusted.

### Leg 1 — train with font as a lane variable, hold out interior fonts (~2.6 h GPU)

One run. Lane = d2 geometry (119 chars, **~60 target tokens**) at **fonts 10, 20 and 40**, ~20k train
rows each. **Held out: fonts 14 and 28** — never trained, interior to the trained range.

Recipe = [H07](../done/H07-pretrained-vision-adapter-qwen.md)'s verbatim (3-group LR
3e-4 / 5e-5 / 2e-5, no freeze, `max_soft_tokens: 280`, effective batch 64 as 4 × 16), warm-started
via `model.init_from: data/checkpoints/hybrid-pretrained-randstr-v0/final`, ~500 steps. Font is the
only variable.

Five design choices, each with its trade-off:

1. **Discrete font lanes, not per-row font jitter.** `data/preprocessing/random_strings.py` already
   takes `--font-size / --group-len / --n-groups / --canvas / --subset-name / --seed` and *merges*
   its manifest so several lanes share one root, so the mixture needs **no code change**. Per-row
   jitter would need a per-row font draw in the materializer. *Trade-off:* three discrete fonts is a
   coarser augmentation than continuous jitter, so a refute here does **not** license "the
   architecture cannot learn scale invariance" — continuous jitter is the obvious next rung before
   that conclusion.
2. **Hold out fonts 14 and 28, not the range edges.** Interpolation inside a trained range is the
   *weakest* form of scale generalisation, so a failure there is much more informative than a
   failure to extrapolate. *Trade-off:* the production font (14) is then never trained, so this run
   is a diagnostic, not a deployable recipe. Font **52** is materialized and probed as an
   extrapolation rung but is **not** part of any pre-registered branch.
3. **d2 geometry (~60 target tokens, ~80% readable within depth ≈48), not d3 or d4.** H13 died
   because 85.8% of its supervised tokens lay past the scan depth and the from-scratch adapter never
   bootstrapped. d2 sits at 80% readable — comfortably outside that regime — so a null here is about
   *scale*, not about the run collapsing. *Trade-off:* d2 is a sparser page than d3, so the font
   sweep varies absolute ink coverage over a smaller range; the dense case is covered instead by the
   free secondary read-out on d3/d5 below.
4. **Gradient share needs no rebalancing, by construction.** Every row on every font lane carries the
   same ~60 supervised tokens, so token-weighted share is uniform. H13's lesson (*balancing rows does
   not balance gradient*) is satisfied trivially, and no weighting scheme — a second changed
   variable — is introduced.
5. **Different train seeds per font, identical validation seed.** If all three train lanes shared a
   seed the run would see each 119-char string three times, i.e. ~3 epochs of target content on a
   lane whose whole premise is that targets are unguessable. Train splits therefore use distinct
   seeds; **validation splits share one seed** so every font's val rows are byte-identical and the
   held-out-vs-trained comparison is paired.

### Secondary read-out — free, and the direct repair of H13 §6

Probe the leg-1 checkpoint on H13's **existing** `randstr-d3` (font 14) and `randstr-d5` (font 40)
validation splits. Byte-identical targets, paired by row, zero training cost, and directly comparable
to the table at the top of this file. If a scale-trained model reads d5 where H07 read nothing, that
both repairs H13 §6 and **revives H13's untested "D5 ≫ D3 ⇒ line-demux" branch**, which is currently
unanswerable only because no model could read font 40.

## Pre-registered criterion

All read-outs are **Δperm / Δblank / per-position reading gain**
(`scratchpad/hybrid_4lane_position_decay.py --val-path …`,
`scratchpad/hybrid_pretrained_ablation.py --val-dataset … --floor …`). **Never aggregate eval CE** —
and on this lane in particular, because H13 §7 found `random_strings.py`'s `floor.json` divides
string entropy by *target* tokens while response-only masking also supervises the deterministic
`<|im_end|>\n` trailer, so every published floor is too high by ~`T/(T−2)`. At d2's ~60 target tokens
that is a ~3.4% inflation and it is worse on anything shorter. Regenerate `floor.json` before quoting
it; do not build a criterion on it.

The comparison is **within-run and paired**: held-out fonts against the mean of the three *trained*
fonts in the same checkpoint. That avoids having to predict absolute levels, which no one can do
before the run.

| branch | condition | verdict |
|---|---|---|
| **CONFIRMS** | Trained fonts read (**pos-0 ≥ +20 pts** and **Δperm ≥ +30%** on all three), **and** each held-out font (14, 28) reaches **≥ 50%** of the trained-font mean on **both** pos-0 gain and Δperm | **Scale generalisation is learnable.** H13 §6's null was a training-distribution artifact, not an architectural limit. Render geometry becomes a live lever and H13's line-demux branch is re-openable. |
| **REFUTES** | Trained fonts read (same bar) **while** both held-out fonts sit at the ignore floor: **pos-0 ≤ +3 pts** and **Δperm ≤ +1%** (H13's measured ignore-floor reference is ≈ +0.1%; d5 measured +0.0 pts / +0.01%) | **Reading is memorised per glyph scale** and does not transfer even by *interpolation* inside a trained range. The free render-geometry lever is dead; [H17](H17-raise-soft-token-budget.md)'s budget knob is the only bandwidth lever left, and H13's line-demux branch stays permanently untestable by re-rendering. |
| **PARTIAL** | Held-out between 15% and 50% of the trained-font mean | **No branch is named.** Report the decay curve against |font − nearest trained font| and stop. |
| **VOID** | Trained fonts themselves fail to read (pos-0 < +20 pts at every trained font) | The run collapsed — check `grad_norm` shape and the token-weighted readable fraction first. **Says nothing about scale.** This is the [H13](../done/H13-density-ladder-long-targets.md) failure mode and the reason for the warm-start and the d2 lane. |

Font 52 (extrapolation) and the d3/d5 secondary read-out are **reported in every branch** and named
in **none** — they are context, not criteria, registered as such before the run.

## What no branch licenses

1. **Font is not a single-factor dial.** H13 §5.3 is explicit: changing the font changes
   chars-per-inked-token *and* stroke width, lines per cell (**3.76 → 1.33**), grid rows occupied
   (**2 → 11**) and the fraction of the token grid in use (**12.5% → 58%**). H21 tests that whole
   bundle. A refute does not say *which* component failed to transfer, and a confirm does not say
   which one the model learned to absorb.
2. **Nothing about the soft-token budget.** It is pinned at 280 in every leg. Capacity vs scan is
   [H17](H17-raise-soft-token-budget.md)'s question and this experiment cannot touch it.
3. **Nothing about real text.** randstr is prior-proof *by construction*; fineweb-edu at ~2,832
   chars/page cannot change font without also changing the page count, so neither branch transfers to
   the real lanes without a separate test.
4. **Nothing about audio.** Spectrogram pages have no glyph structure and are upsampled at every
   budget (measured, [H20](H20-audio-phoneme-resolution.md)). H20's wide-render leg is *motivated* by
   the same intuition but is not tested here.
5. **A confirm is about a trained model, not about the vision tower.** Whether the invariance lives
   in Gemma's pretrained features, in the from-scratch adapter, or in the decoder is unmeasured; the
   follow-up is an H07-style weight-delta analysis, which is cheap and is not in this 3 h.
6. **A confirm licenses interpolation over 10–40 and nothing outside it.** The font-52 rung is
   reported, never used to claim extrapolation.
7. **Neither branch says anything about reading *deeper*.** Reading *at* an unfamiliar scale and
   reading *further down* a page are different claims; the depth limit is
   [H17](H17-raise-soft-token-budget.md) / [H18](H18-no-learned-scan.md).
8. **This does not measure the cost of augmentation at the production font.** A single-font control
   (same recipe, same warm-start, font 14 only) would, at ~2.6 h more GPU. It is a follow-up, not
   part of this design, and the trained-font numbers here must not be compared against
   [H07](../done/H07-pretrained-vision-adapter-qwen.md)'s as if they were — H07 ran a different lane
   (d1 geometry, 29 chars).

## Prep status

Nothing built. No code change is required for the design as specified —
`data/preprocessing/random_strings.py` already exposes every knob the lanes need and merges its
manifest across lanes, and `lane_specs[<subset>]["canvas"]` is recorded per lane. The training config
(`configs/h21_fontmix.yaml`) and the materialization script do not exist yet.

Two things to check before launch, both CPU-only:

- **One-page guard.** H13's deviation note records that 400 letters at font 52 overflows a 1024px
  canvas to **two** pages, which would confound a font rung with *image count*; `random_strings.py`
  gained a one-page render guard for exactly this. At d2's 119 characters every font up to 52 should
  fit one page — verify it rather than assume it, on all six rungs.
- **Retention.** `_filter_training_tokens` charges `otl + image_token_budget(280)=282 + 256`, so a
  d2 row is ~598 against the config's `max_length` — a wide margin at 280, but confirm 20000/20000
  retention per rung rather than discovering a silently dropped font lane afterwards.
