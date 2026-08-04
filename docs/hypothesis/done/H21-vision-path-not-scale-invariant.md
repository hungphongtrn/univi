# H21 — Reading survives a glyph-scale change it was not trained on

**Status:** DONE (2026-07-30) — **REFUTED, decisively.** Reading is **memorised per glyph scale** and
does not transfer *even by interpolation inside the trained range*. Trained fonts {14, 24, 40} read at
position-0 **+57.3 / +52.7 / +64.0 pts**; held-out fonts {18, 31} sit at **+0.0 / +0.7 pts**, and the
interpolated 20/28 and extrapolated 52 are at the floor too — an ~80× dissociation with no
significance anywhere off a trained font. Font **18 lies between trained 14 and 24 and reads nothing.**
The leg-0 sanity gate replicated H13 §6's anchors **exactly** (diff 0.000 against an external anchor
file from a different checkpoint), so the probe is validated. **The probe script self-reported
`branch: VOID` and that call is an artifact** of a length-diluted Δperm conjunct — see
[Verdict](#verdict-2026-07-30). Two things fall out: H13 §6's null is confirmed as a
*scale-generalisation* failure, and H13's untestable **"D5 ≫ D3 ⇒ line-demux" branch is now live**
(d5 +66.0 > d3 +55.3). · **Cost:** ~1.0 h GPU (500 steps) + ~0.5 h probes · Separates **render
geometry** from the token budget · Origin:
[H13](H13-density-ladder-long-targets.md) §6, incidental finding · Related:
[H17](H17-raise-soft-token-budget.md), [H20](../todo/H20-audio-phoneme-resolution.md),
[H07](H07-pretrained-vision-adapter-qwen.md)

## Verdict (2026-07-30)

Leg 1 = `configs/h21_fontjitter.yaml`, 500 steps, d2 geometry (119 chars, ~60 supervised tokens) at
fonts **{14, 24, 40}**, `grad_norm` 0.65–0.86 throughout — the only healthy training curve in this
batch. Probed with `scratchpad/h21_font_sweep.py` → `data/eval/h21-leg{0,1}.json`, 150 rows.

**Leg 0's sanity gate passed exactly**, which is what makes the rest believable: pos-0 **+28.6667** vs
anchor **+28.6667** (tolerance ±2.0), Δperm **3.0525%** vs **3.0525%** (tolerance ±0.5 pp), and d5
non-significant as expected. Anchors came from `data/eval/h13-poscontrol-h07.json` — a different file,
a different checkpoint — so the zero difference is a genuine external replication, not circularity.

| font | status | pos-0 gain | pos-0…4 gain | chars 0–9 | Δperm | significant |
|---|---|---|---|---|---|---|
| **14** | **trained** | **+57.3** | +47.6 | **+45.5** | 25.80% | yes |
| 18 | **held out** | +0.0 | −1.1 | −1.1 | 0.012% | no |
| 20 | interpolated | +2.0 | −1.5 | −1.5 | 0.021% | no |
| **24** | **trained** | **+52.7** | +22.7 | **+20.7** | 8.85% | yes |
| 28 | interpolated | +1.3 | +0.1 | +0.0 | 0.384% | no |
| 31 | **held out** | +0.7 | −0.3 | −0.5 | 0.086% | no |
| **40** | **trained** | **+64.0** | +29.9 | **+27.5** | 11.15% | yes |
| 52 | extrapolated | +0.0 | −0.3 | −0.5 | −0.120% | no |

This is the **REFUTES** row of the criterion table below, met on its own terms: *"Trained fonts read …
while both held-out fonts sit at the ignore floor: pos-0 ≤ +3 pts and Δperm ≤ +1%"* — font 18 gives
+0.0 pts / 0.012%, font 31 gives +0.7 pts / 0.086%. Both are inside the ignore floor, and the
interpolated rungs are too. **The free render-geometry lever is dead**, and it is dead in a stronger
sense than the hypothesis anticipated: not "fails to extrapolate" but "fails to interpolate."

### The script's `branch: VOID` is wrong, and it is the same defect that voided H17

`h21_font_sweep.py`'s `trained_read` test ANDs **pos-0 ≥ 20 pts** with **Δperm ≥ 30%**. The trained
fonts clear the first conjunct by 2.6–3.2× and miss the second (25.8 / 8.9 / 11.1%), so the script
declares *"the trained fonts themselves fail the reading bar … The run collapsed … SAYS NOTHING ABOUT
SCALE."* That conclusion is false, and the reason is measurable:

**The ≥ 30% bar was imported from a lane of a different target length.** It traces to
[H07](H07-pretrained-vision-adapter-qwen.md)'s +114.7%, measured on a **~25-char** target.
This lane is **119 chars / ~60 supervised tokens**. [H13 §4](H13-density-ladder-long-targets.md)
measured aggregate Δperm falling **14.31% → 0.63% for identical reading** as targets lengthen, and
H13's anchor for *established* reading on this exact d2 geometry is **Δperm +14.31% at pos-0 +35.3**.
Font 14 here reads at **pos-0 +57.3** — substantially *more* than H13's d2 anchor — and scores 25.8%,
i.e. **1.8× the Δperm of the established reference.** A 30% bar on a 60-token lane demands more than
this architecture has ever produced at that target length. The conjunct is unsatisfiable by
construction, so `trained_read` can only ever return false here.

The position-resolved half of the criterion is sound and is what the verdict rests on. **Repair for any
successor: drop the aggregate-Δperm conjunct from `trained_read` entirely**, or state it against the
same-geometry H13 anchor (≥ +10%, not ≥ +30%). Keep pos-0 ≥ 20 pts and add `char_bins[0-10] ≥ +10 pts`.
This is the fourth criterion in the programme found mis-specified this way — see
[H17's verdict](H17-raise-soft-token-budget.md#verdict-2026-07-30) and
[H15's](H15-prior-poisoned-text.md#verdict-2026-07-30).

### Secondary read-out — H13 §6 repaired, and the line-demux branch revived

On H13's own `randstr-d3` (font 14) and `randstr-d5` (font 40) splits — byte-identical targets
differing *only* in font — the leg-1 checkpoint reads **d3 at +55.3** and **d5 at +66.0**, where
[H07](H07-pretrained-vision-adapter-qwen.md) read **+0.0** on d5.

1. **H13 §6's null is confirmed as a scale-generalisation failure**, exactly as that section suspected:
   font 40 was unreadable because H07 never trained it, not because spreading 479 chars over a bigger
   glyph destroys information.
2. **H13's "D5 ≫ D3 ⇒ line-demux" branch is testable for the first time**, and the sign is positive:
   d5 (+66.0) > d3 (+55.3) on identical targets. Fewer characters per patch reads *better* once the
   scale is trained. That is a (A′)-flavoured signal and it is the first positive lever on reading
   *amplitude* since [H17](H17-raise-soft-token-budget.md) went VOID.

**Caveat, and it is the whole caveat:** font 40 **was a trained font**, so d5's reading is not evidence
of generalisation — this section measures what a *scale-trained* model can do, which is precisely why
it repairs H13 §6 rather than contradicting the REFUTED verdict above. And the +66.0 vs +55.3 gap is
position-0 *amplitude*; nothing here shows d5 reads **deeper**. Depth is ≈ 0 past chars 10–24 on every
rung in the sweep table (+3.2 pts at best, on font 14). **The depth ceiling is untouched by scale.**

### What this does NOT license

- **Not "render bigger and it will read more."** Trained-scale reading is high-amplitude and still
  shallow: font 14's `char_bins` go +45.5 (0–9) → +3.2 (10–24) → +0.3 (25–49). Scale buys amplitude at
  the start of the page, not depth.
- **Not a claim about *why* scale fails to transfer.** A per-scale template story and a
  "position-embedding grid never sees the intermediate glyph pitch" story both predict this table.
- **Not a verdict on H13's line-demux branch** — only that it is now *askable*. Testing it needs a
  matched pair where d5's advantage can be separated from d5's lower chars-per-patch.
- **Not transferable to audio.** [H20](../todo/H20-audio-phoneme-resolution.md) varies a spectrogram's time
  axis, not a glyph's size, and its pages are upsampled at every budget.

## Claim

Reading is **scale-invariant**: a model that transcribes a rendered page at font 14 also transcribes
the *same characters* rendered at a font it never trained on.

**The evidence so far REFUTES this**, which is why the claim is stated in the direction that the data
already contradicts — the open question is not whether scale-generalisation happens today (it does
not) but whether it is **learnable at all**, and that has never been tried.

## Where this came from — H13 §6

[H13](H13-density-ladder-long-targets.md)'s ladder contained two rungs that draw from the
same RNG stream and therefore carry **byte-identical targets on 30/30 paired rows**, differing only
in font. Under the [H07](H07-pretrained-vision-adapter-qwen.md) checkpoint, probed on the
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

## Why this is worth the GPU time

Three live proposals in the programme assume glyph scale is a free lever:

- [H17](H17-raise-soft-token-budget.md)'s *"zero-code complement: re-render at font 28–52"*, which its
  fourth correction calls **"a cheap preview of the main intervention"** — font 40 moves
  chars-per-inked-token 4.64× against the 3.73× a 4.25× budget raise buys, at zero token cost.
- H13's untested line-demux branch (fix render geometry *before* token budget, because it is free).
- [H20](../todo/H20-audio-phoneme-resolution.md)'s wide-render leg, which is the same "just re-render it
  bigger" move on the time axis.

If glyph scale is not learnable, the free lever is dead and [H17](H17-raise-soft-token-budget.md)'s
costly budget knob is the only remaining bandwidth lever. If it *is* learnable, render geometry
becomes a serious alternative to ~26 h of GPU. Either way it is the cheapest question left.

## Design

### Leg 0 — zero-training font sweep of the H07 checkpoint (~0.5 h GPU)

Turn H13's two points into a curve. Materialize d2-geometry rungs (119 chars) at fonts
**14 / 18 / 20 / 24 / 28 / 31 / 40 / 52**, all from the **same seed** so targets are byte-identical
across rungs and every comparison is **paired by row** (H13 §5.5: unpaired cross-rung comparison was
a real power loss on that ladder). Probe `hybrid-pretrained-randstr-v0/final` on all eight.

> **Font 10 does not exist.** `render_utils.py` clamps `font_size = max(font_size, 14)` in every
> render entry point, so fonts 8, 10, 12 and 14 produce the **byte-identical** page (verified by md5;
> `scratchpad/h21_materialize_fonts.py --self-test`). The originally-specified `10` rung is dropped
> and 14 is the floor of the ladder; 18, 24 and 31 are added so the sweep covers leg 1's fonts. See
> [Prep status](#prep-status-built-2026-07-29) for what this cost the design.

This measures the *shape* of the failure — a cliff at some scale, or a gradual decay away from font
14 — which decides nothing on its own but tells leg 1 where to put its training fonts. It is also the
sanity gate.

> **Sanity gate.** The same probe invocation must reproduce H13 §6's anchors on the **existing**
> `randstr-d3` / `randstr-d5` validation splits: pos-0 **≈ +28.7** at font 14 and **≈ +0.0** at font
> 40, Δperm ≈ +3.05% / +0.01%. Same checkpoint, same data, same script — if this does not replicate,
> the probe or the checkpoint has drifted and nothing below can be believed. Failing the gate costs
> 25 minutes and kills the experiment before any materialization is trusted.

> ⚠️ **The gate is only valid on the H07 checkpoint (leg 0). On the font-trained checkpoint (leg 1)
> a `gate_failed` is expected and may BE the result** (found 2026-07-30, before leg 1 was probed).
> The gate requires `randstr-d5`'s pos-0 gain to be **non-significant** — it replicates H07's *null*
> at font 40. But this hypothesis' own
> [secondary read-out](#secondary-read-out--free-and-the-direct-repair-of-h13-6) *hopes* the
> font-trained checkpoint **reads** d5, because that is what repairs H13 §6 and revives H13's
> line-demux branch. So on leg 1 the two are in direct opposition: **the experiment succeeding makes
> the gate fail.** `scratchpad/h21_font_sweep.py` writes its JSON *before* exiting non-zero, so a
> leg-1 `gate_failed: true` with `rc=2` must be read as a candidate **positive**, not as
> "nothing below can be believed". Run leg 0's gate first, on the H07 checkpoint, where a failure
> genuinely does mean drift; after that the gate is a *change detector*, not a validity check.
>
> Do **not** pass `--force` to suppress it — the non-zero exit is the signal. Interpret it by reading
> `randstr-d5`'s pos-0 gain and CI directly.

### Leg 1 — train with font as a lane variable, hold out interior fonts (~1.0 h GPU measured-forward, 2.6 h budgeted)

One run. Lane = d2 geometry (119 chars, **~60 supervised tokens**) at **fonts 14, 24 and 40**, 20k
train rows each. **Held out: fonts 18 and 31** — never trained, strictly interior to the trained
range (they are the geometric midpoints of 14–24 and 24–40).

Recipe = [H07](H07-pretrained-vision-adapter-qwen.md)'s verbatim (3-group LR
3e-4 / 5e-5 / 2e-5, no freeze, `max_soft_tokens: 280`, effective batch 64 as 16 × 4), warm-started
via `model.init_from: data/checkpoints/hybrid-pretrained-randstr-v0/final`, 500 steps. Font is the
only variable. Config: `configs/h21_fontjitter.yaml`.

Five design choices, each with its trade-off:

1. **One lane whose font varies per ROW.** The three 20k font blocks are concatenated into the single
   already-valid subset `random-strings` (`univi.trainer.VALID_SUBSETS` is a closed set with no name
   meaning "font F", and adding one is a code change). The font stays recoverable per row from
   `source_dataset_id` (`synthetic/randstr-f14|-f24|-f40`) and `render_config.font_size`. This is
   *closer* to the ideal than the originally-planned discrete lanes — it is the per-row font draw the
   original design said "would need a per-row font draw in the materializer", done by concatenating
   blocks outside the materializer. *Trade-off:* three discrete fonts is still a coarser augmentation
   than continuous jitter, so a refute here does **not** license "the architecture cannot learn scale
   invariance" — continuous jitter is the obvious next rung before that conclusion.
2. **Hold out fonts interior to the trained range, not the edges.** Interpolation inside a trained
   range is the *weakest* form of scale generalisation, so a failure there is much more informative
   than a failure to extrapolate. *Trade-off, and a real loss:* the design originally held out the
   **production** font 14 as well, which the renderer's `max(font_size, 14)` clamp makes impossible —
   14 is the smallest page this renderer can draw, so "held out **and** interior" is unsatisfiable
   for it. Font 14 is therefore **trained**, and the trained-font mean contains a lane the warm-start
   checkpoint already reads. The criterion is a held-out-vs-trained-mean ratio and **both** held-out
   fonts are new to the run *and* to the warm start, so the comparison is unaffected in direction;
   but font 14's absolute number must be reported separately from 24's and 40's. Font **52** is
   materialized and probed as an extrapolation rung but is **not** part of any pre-registered branch.
3. **d2 geometry (~60 supervised tokens, 79.9% readable within depth ≈48), not d3 or d4.** H13 died
   because 85.8% of its supervised tokens lay past the scan depth and the from-scratch adapter never
   bootstrapped. The built lane measures **79.9%** readable — above H14's repaired 75.2%, which
   CONFIRMED — so a null here is about *scale*, not about the run collapsing. *Trade-off:* d2 is a
   sparser page than d3, so the font sweep varies absolute ink coverage over a smaller range; the
   dense case is covered instead by the free secondary read-out on d3/d5 below.
4. **Gradient share needs no rebalancing, by construction.** Every row on every font carries the same
   ~60 supervised tokens (measured: 60.05 mean over the built 60k split), so token-weighted share is
   uniform. H13's lesson (*balancing rows does not balance gradient*) is satisfied trivially, and no
   weighting scheme — a second changed variable — is introduced.
5. **Different train seeds per font; the leg-0 sweep shares one seed.** If all three train blocks
   shared a seed the run would see each 119-char string three times, i.e. ~3 epochs of target content
   on a lane whose whole premise is that targets are unguessable. Train blocks therefore use distinct
   seeds (`3407 + 7919·k`). The **leg-0 validation rungs** all share one seed, so every font's rows
   are byte-identical and every cross-font comparison — including trained-vs-held-out — is paired.

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
| **CONFIRMS** | Trained fonts read (**pos-0 ≥ +20 pts** and **Δperm ≥ +30%** on all three), **and** each held-out font (18, 31) reaches **≥ 50%** of the trained-font mean on **both** pos-0 gain and Δperm | **Scale generalisation is learnable.** H13 §6's null was a training-distribution artifact, not an architectural limit. Render geometry becomes a live lever and H13's line-demux branch is re-openable. |
| **REFUTES** | Trained fonts read (same bar) **while** both held-out fonts sit at the ignore floor: **pos-0 ≤ +3 pts** and **Δperm ≤ +1%** (H13's measured ignore-floor reference is ≈ +0.1%; d5 measured +0.0 pts / +0.01%) | **Reading is memorised per glyph scale** and does not transfer even by *interpolation* inside a trained range. The free render-geometry lever is dead; [H17](H17-raise-soft-token-budget.md)'s budget knob is the only bandwidth lever left, and H13's line-demux branch stays permanently untestable by re-rendering. |
| **PARTIAL** | Held-out between 15% and 50% of the trained-font mean | **No branch is named.** Report the decay curve against |font − nearest trained font| and stop. |
| **VOID** | Trained fonts themselves fail to read (pos-0 < +20 pts at every trained font) | The run collapsed — check `grad_norm` shape and the token-weighted readable fraction first. **Says nothing about scale.** This is the [H13](H13-density-ladder-long-targets.md) failure mode and the reason for the warm-start and the d2 lane. |

Fonts **20 and 28** (interior, but probed by leg 0 rather than reserved as criterion hold-outs),
font **52** (extrapolation) and the d3/d5 secondary read-out are **reported in every branch** and
named in **none** — they are context, not criteria, registered as such before the run.

The branch table is evaluated mechanically by `scratchpad/h21_font_sweep.py --trained-fonts 14 24 40
--heldout-fonts 18 31`, which returns `UNDECIDED` rather than forcing a branch when the numbers match
none of the four rows.

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
   budget (measured, [H20](../todo/H20-audio-phoneme-resolution.md)). H20's wide-render leg is *motivated* by
   the same intuition but is not tested here.
5. **A confirm is about a trained model, not about the vision tower.** Whether the invariance lives
   in Gemma's pretrained features, in the from-scratch adapter, or in the decoder is unmeasured; the
   follow-up is an H07-style weight-delta analysis, which is cheap and is not in this 3 h.
6. **A confirm licenses interpolation over 14–40 and nothing outside it.** The font-52 rung is
   reported, never used to claim extrapolation. Nothing below 14 is testable with this renderer at
   all (see 9).
7. **Neither branch says anything about reading *deeper*.** Reading *at* an unfamiliar scale and
   reading *further down* a page are different claims; the depth limit is
   [H17](H17-raise-soft-token-budget.md) / [H18](../todo/H18-no-learned-scan.md).
8. **This does not measure the cost of augmentation at the production font.** A single-font control
   (same recipe, same warm-start, font 14 only) would, at ~2.6 h more GPU. It is a follow-up, not
   part of this design, and the trained-font numbers here must not be compared against
   [H07](H07-pretrained-vision-adapter-qwen.md)'s as if they were — H07 ran a different lane
   (d1 geometry, 29 chars).
9. **Nothing about glyph scales below 14.** The renderer floors at 14 px, so the whole "smaller
   glyphs" half of the scale axis is untested and untestable without either a code change or a
   larger canvas (which changes the resampling path as well as the glyph size, i.e. a different
   manipulation). Every statement here is about scaling **up** from the production font.
10. **A confirm at font 14 is partly inherited.** The warm start already reads font-14 randstr at
    Δperm +114.7%, and font 14 is a trained lane in this run. Only fonts 24 and 40 test whether new
    scales can be *acquired*; only 18 and 31 test whether they *transfer*.

## Prep status (BUILT 2026-07-29)

Everything below was produced CPU-only (`CUDA_VISIBLE_DEVICES=""`), with a live GPU queue untouched.
**No `univi/` source file was modified**; the one design change that *would* have needed a code change
(a new `VALID_SUBSETS` name per font) was avoided by mixing fonts inside the existing
`random-strings` subset.

### The defect that would have voided the experiment

`data/preprocessing/render_utils.py` clamps `font_size = max(font_size, 14)` in all four render entry
points. Rendering the same 119-char string at fonts 8, 10, 12 and 14 gives **one md5**. So the design
as written trains on `{10, 20, 40}` — which is really `{14, 20, 40}` — and then "holds out" font 14,
**which it has just trained 20,000 times**. The CONFIRMS branch would have fired on a memorised font
and nothing in any log would have shown it. `scratchpad/h21_materialize_fonts.py` now *refuses*
`--fonts` below 14 and asserts pairwise pixel-distinctness across every rung it writes.

### Artifacts

| what | where | verified |
|---|---|---|
| leg-0 sweep, 8 rungs × 300 val rows | `data/materialized/h21-fonts-v0/randstr-f{14,18,20,24,28,31,40,52}/validation` | targets byte-identical across all 8 rungs on 64/64 checked rows; all 28 rung pairs pixel-distinct on 64/64 rows; 1 page/row everywhere |
| leg-1 training lane, 60k rows | `data/materialized/h21-fontmix-v0/random-strings/train` (20k each at fonts 14/24/40, seeds 3407+7919·k) | `source_dataset_id` counts 20000/20000/20000 |
| leg-1 val, 1.5k rows | `…/random-strings/validation` | |
| materializer | `scratchpad/h21_materialize_fonts.py` (`--self-test`) | 5 checks incl. the clamp |
| probe (both legs) | `scratchpad/h21_font_sweep.py` (`--self-test`) | 12 checks: gate replication, pairing, cliff shape, all four branches, cluster SE |
| training config | `configs/h21_fontjitter.yaml` | |

### Measured, not assumed

- **Token retention 100.00%** — 60,000/60,000 train and 1,500/1,500 val through the *real*
  `univi.trainer.load_dataset` / `_load_eval_datasets` at `max_length: 1024`. The worst-case filter
  estimate (`otl + image_token_budget(280)=282 + 256`) over the real split is **605**. The margin is
  structural: 100 letters cannot tokenise past ~119 tokens.
- **Readable fraction 79.9%** at the ~48-token scan depth (60.05 supervised tokens/row). H13 died at
  14.2%; H14 was repaired to 75.2% and CONFIRMED; H07 succeeded at 100%.
- **Real assembled sequence: mean 332.9, max 336 tokens** through `HybridCollator` — *identical* at
  fonts 14/24/40/52, because the font changes pixels, not tokens. `max_length: 1024` can never
  truncate.
- **One page per row on 60,000/60,000 rows**, at every font up to 52 (checked to 72). H13's
  two-page overflow at font 52 was at d3's 479 chars; d2's 119 chars never come close.
- **VRAM 21.57 GB (23.15 GB pessimistic) at bs16**, seq 336 — inside the 26 GB budget, and matching
  H07's exact 16 × 4 = effective-64 geometry. **Total-card caveat:** the 40 GB A100 currently shows
  ~16 GB held by another container (~24 GB free), which leaves under 1 GB of headroom at bs16; drop
  to 8 × 8 (16.99 GB pessimistic) if that container is still resident at launch. The effective batch
  is 64 either way.
- **Floor regenerated in both conventions.** `floor.json` on every rung carries the legacy
  per-*target*-token number and `no_reading_floor_nats_per_supervised_token`, which is lower by
  exactly `T/(T−2) = 1.0345` at these ~60-token targets (H13 §7). Neither is used by any branch.

### Run order

```bash
# LEG 0 — zero training, ~0.5 h. Gate first: it must reproduce H13 §6 on the
# EXISTING d3/d5 splits (pos-0 +28.7 / +0.0, Δperm +3.05% / +0.01%), read from
# data/eval/h13-poscontrol-h07.json. Exits non-zero if it does not.
HF_HUB_OFFLINE=1 uv run python scratchpad/h21_font_sweep.py \
    --checkpoint data/checkpoints/hybrid-pretrained-randstr-v0/final \
    --tag h21-leg0 -n 150

# LEG 1 — ~1.0 h train (500 steps × 64 = 0.53 epochs of the 60k lane)
HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup uv run python -m univi.hybrid.train_pretrained \
    --config configs/h21_fontjitter.yaml \
    > data/checkpoints/h21-fontmix-v0-run.log 2>&1 &

# LEG 1 read-out — same probe, plus the pre-registered branch table
HF_HUB_OFFLINE=1 uv run python scratchpad/h21_font_sweep.py \
    --checkpoint data/checkpoints/h21-fontmix-v0/final \
    --tag h21-leg1 -n 150 --trained-fonts 14 24 40 --heldout-fonts 18 31
```

The leg-0 gate is not decoration: it is the only thing standing between a font curve and an
uncontrolled comparison against numbers retyped from a doc. It re-runs `h13_analyze`'s own
`collect_rung` / `analyse_rung` on the same rows, the same seed and the same checkpoint, and compares
against the **recorded** H13 §6 JSON rather than against the table at the top of this file.
