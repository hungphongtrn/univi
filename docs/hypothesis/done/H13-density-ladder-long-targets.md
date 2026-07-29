# H13 — The readable ceiling is set by character density, not token position

**Status:** DONE (2026-07-29) — **REFUTED.** The readable ceiling is **invariant to character
density** over a 16× range. Density-proportional bandwidth is excluded; the surviving accounts are
**(C) fixed scan depth** and a **fixed-absolute-capacity** flavour of (A), which
[H17](../todo/H17-raise-soft-token-budget.md) now discriminates.
**The H13 training run itself is VOID** (collapsed to the no-reading floor); the verdict comes from
running the pre-registered discriminator against the
[H07](../done/H07-pretrained-vision-adapter-qwen.md) checkpoint instead. See
[Verdict](#verdict-2026-07-29).
· **Cost:** ~6 h GPU (wasted) + ~25 min probe · Separates **(A) bandwidth** from **(C) scan** · Related: [H11](../done/H11-position-decay-is-prior-induced.md), [H12](../done/H12-contrastive-decoding-probe.md), [H17](../todo/H17-raise-soft-token-budget.md), [H18](../todo/H18-no-learned-scan.md)

**Promoted to first place** by [H12](../done/H12-contrastive-decoding-probe.md): contrastive decoding
recovered no mid-page content, which is H12's pre-registered "refute" branch and explicitly promotes
this hypothesis and [H17](../todo/H17-raise-soft-token-budget.md) over the objective-side line.

## Claim

Reading degrades because the soft-token stream runs out of capacity as characters-per-page rises —
not because the model loses track of position as it generates. If true, the collapse point should
move with **density** and be invariant to **token index**.

## Why this is now the first substantive run

[H11](../done/H11-position-decay-is-prior-induced.md) found the decay appears at the *same token
positions* on randstr (25 chars/page) and fineweb (2,832 chars/page) — a ~100× density gap. That is
the signature of (C), not (A). But randstr targets are only ~14.5 tokens long, so the comparison is
confounded: we have never observed a deep position on a prior-proof lane.

## Design

Extend `data/preprocessing/random_strings.py` (make `N_GROUPS`, `GROUP_LEN`, `FONT_SIZE` CLI args)
to materialize a ladder that varies density and target length **independently**:

| rung | chars/page | answer tokens | purpose |
|---|---|---|---|
| D1 | 25 | ~15 | replicates [H07](../done/H07-pretrained-vision-adapter-qwen.md) |
| D2 | 100 | ~60 | |
| D3 | 400 | ~230 | |
| D4 | 1600 | ~900 | approaches fineweb density |
| D5 | 400, **font 52** | ~230 | ~1.0 line/patch — isolates line-demux from chars/token |

~20k rows each, one mixed run, recipe = `configs/hybrid_pretrained_randstr.yaml` (the recipe that
read at +115%). 1000 steps. **This is not [H05](../done/H05-isolated-ocr-from-scratch.md) redux** —
that zero came from the from-scratch embedder, since falsified by
[H07](../done/H07-pretrained-vision-adapter-qwen.md).

Instrument every rung with **per-position** reading gain
(`scratchpad/hybrid_4lane_position_decay.py --val-path …`), not just aggregate Δperm.

## Pre-registered criterion

Define **K** = the character position beyond which positional reading gain drops below 50% of the
0–100-char gain.

- **K scales with density** (collapse point moves) ⇒ **(A) bandwidth** ⇒ [H17](../todo/H17-raise-soft-token-budget.md) is the fix.
- **K fixed at the same token index across all rungs** ⇒ **(C) scan failure** ⇒ [H18](../todo/H18-no-learned-scan.md); the fix is architectural or curricular, and more tokens will not help.
- **D5 ≫ D3** ⇒ the mechanism is lines-per-patch demultiplexing, not chars-per-token ⇒ fix by render
  geometry (free) before token budget (costly).
- K ≥ 400 ⇒ capacity is not binding at mixture-relevant density and the
  [H09](../done/H09-balanced-mixture-prevents-drowning.md) failure is purely objective-side.

## Context

Current density: fineweb is **10.7 chars/soft-token** and **3.8 text lines per 48px patch**; randstr
is 0.1 chars/token. See [full-page OCR token budgets](../../literature/full-page-ocr-token-budget.md).

## Deviations from the pre-registered design (recorded at launch, before any result)

1. **D5 renders at font 40, not font 52.** 400 letters at font 52 overflows a 1024px canvas to **2
   pages**, which would confound D5 with *image count* on top of font size. Font 40 fits one page
   and hits this rung's stated intent — ~1.0 text line per 48px patch — exactly (measured line
   height 48px; d1–d4 sit at ~2.8 lines/patch). Verified empirically across 30 samples/rung.
2. **Batch 16 → 4, accum 4 → 16.** d4 targets are ~900 answer tokens against d1's ~15, so the
   per-device batch is cut to hold peak VRAM inside the 26 GB budget (an unrelated process holds
   13.6 GB of the 40 GB card). Effective batch stays **64**, matching H07 exposure — the recipe
   variable is unchanged.
3. **500 validation rows/rung** (design said only "~20k rows each" for train); eval samples 128/rung.

Everything else is H07's recipe verbatim. `random_strings.py` gained `--group-len/--n-groups/
--font-size/--subset-name` and now *merges* its manifest so all five rungs share one root; a
one-page render guard was added because `render_text_pages` pages out and the previous
`pages[0]` would have silently truncated the image while keeping the full target.

## Observation log

**Eval trajectory** (nats/tok; per-rung floors all ≈5.60; `*` = below floor ⇒ reading):

| step | d1 (25ch) | d2 (100ch) | d3 (400ch) | d4 (1600ch) | d5 (400ch, font40) |
|---|---|---|---|---|---|
| 100 | 5.134* | 5.565* | 5.665 | 5.698 | 5.666 |
| 200 | 5.023* | 5.539* | 5.649 | 5.672 | 5.649 |
| 300 | 5.007* | 5.529* | 5.643 | 5.666 | 5.643 |

- **Only d1 clearly reads**, d2 marginally; d3/d4/d5 sit above their floors and are improving ~0.006
  per 100 steps. All rungs are flattening.
- **d5 ≡ d3 to three decimals at every eval so far.** The font-40 re-render (~1.0 text line per 48px
  patch vs d3's ~2.8) is so far *indistinguishable* from d3. If this holds, the pre-registered
  "D5 ≫ D3 ⇒ line-demux" branch is a **null**, which removes the free render-geometry fix and leaves
  the costly token-budget one.
- `grad_norm` runs 0.08–0.11 and is flagged DEAD-GRAD by the watcher. Read it as a *within-run*
  trend only — see the caveats below.

### Two interpretive caveats, recorded before the verdict

1. **Each rung gets 1/5 the exposure of a single-lane run.** [H07](../done/H07-pretrained-vision-adapter-qwen.md)
   drove the equivalent of d1 from 5.06 → 3.735 in 600 single-lane steps; here d1 is at 5.007 by step
   300 because only ~20% of gradient steps see d1 data. **Absolute reading depth is therefore not
   comparable to H07** — only the *across-rung* comparison inside this run is valid, which is exactly
   what the discriminator uses.
2. **`grad_norm` is not comparable to H07 either.** H07's ~4–5 was measured on 14-token targets;
   this run averages loss over targets up to 931 tokens, and a mean-over-tokens reduction shrinks the
   norm as sequences lengthen. The low value is also partly *expected*: on a prior-proof lane a model
   that ignores pixels sits at the uniform-letter optimum, where the gradient genuinely vanishes.
3. **Aggregate CE cannot separate (A) from (C)** — both predict improvement scaling inversely with
   density. Worse, chars/token is ~2.06 on *every* rung, so within a target "fixed character
   position" and "fixed token index" are the same thing. The discriminator is therefore strictly the
   **across-rung** comparison the design specifies: does K *shrink as density rises* (⇒ (A)), or stay
   *constant across rungs* (⇒ (C))? This must come from the per-position probe, not from these
   numbers.

## Design flaws in this hypothesis, found while building the analysis (2026-07-28, pre-verdict)

Recorded **before** results, and all printed alongside the verdict by `scratchpad/h13_analyze.py`.

1. **The ladder does NOT vary density and target length independently, despite the Design section
   claiming it does.** d1→d4 vary both in lockstep (25ch/14tok → 1600ch/931tok, perfectly
   correlated). Since "denser page" vs "longer target" *is* the (A)-vs-(C) question, the d1→d4 trend
   **cannot on its own** separate them. Only **d3-vs-d5** varies a single factor (font at fixed
   density and target).
2. **This is why position-0 gain became the primary discriminator.** At answer position 0 the target
   length has not yet come into play, so position-0 gain across rungs isolates the *density* effect
   cleanly: **(C)** predicts it survives at every density (read the start, then lose the place);
   **(A)** predicts it degrades monotonically as chars/page rises. That comparison is unaffected by
   flaw 1.
3. **The pre-registered 0–100-char reference window is wider than d1's entire target (29 chars)** and
   spans ~84% of d2's (119 chars). K is *structurally unmeasurable* on those rungs — a censoring
   status, not a null — so only d3/d4/d5 can carry a K trend. A clearly-labelled non-pre-registered
   `K_alt_baseline25` (0–25-char window) is reported so d2 can contribute something.
4. **chars/token ≈ 2.06 on every rung**, so a *constant* K is equally consistent with (C) fixed-token
   scan and with a fixed-absolute-capacity flavour of (A). Only a K that **moves with density** is
   diagnostic on its own.
5. **Cross-density rungs do not share targets.** The materializer draws from one RNG stream, so d1 is
   not a prefix of d3/d4 — cross-density comparisons are unpaired. **d3 and d5 do have byte-identical
   targets** (they differ only in font), so that comparison is run **paired by row**, a large power
   gain for precisely the "d5 ≡ d3 to three decimals" question.

The analysis refuses to force a branch: it prints **VOID** if nothing is significant anywhere,
**AMBIGUOUS/CONFLICT** if position-0 and K disagree, and
**UNDECIDABLE/UNDERPOWERED** for the K trend if fewer than 2 rungs yield a measurable K.

## Launch record

- Data: `data/materialized/h13-density-v0` (5 rungs × 20k train / 500 val), built by
  `scratchpad/h13_materialize_ladder.sh`, log `data/materialized/h13-density-build.log`.
  Row retention through the 2048 length filter is **100% on every rung** (checked, because
  dropping long d4 rows would have silently gutted the deep-position arm of this experiment).
- Config: `configs/h13_density.yaml` · Output: `data/checkpoints/h13-density-v0` ·
  Log: `data/checkpoints/h13-density-v0-run.log` · Peak VRAM ~21 GB (budget 26 GB).
- **W&B: project `huggingface`, run `driven-frog-59` (id `a6kv5ysx`)** — *not* the
  `univi-encoder-free` / `h13-density-ladder` names in the config. `train_pretrained.py` never read
  the config's `wandb:` block (it only set `report_to`), so every hybrid run so far was labelled by
  exporting `WANDB_*` by hand. Fixed after this launch, so H13 keeps the auto-generated name and
  later runs will honour their config.
- Instrumentation: `scratchpad/hybrid_4lane_position_decay.py --val-path <rung>/validation`,
  reporting reading gain by **both** answer-token position and character position (K is defined in
  characters; randstr runs ~1.72 letters/token).

## Observation — ink occupancy of the soft-token grid (2026-07-28, pre-verdict, CPU-only)

Measured because **the interpretation of the d5 ≡ d3 null depends on a fact nobody had checked**:
does d5 actually spread its ink over more soft tokens than d3? If yes, the null is evidence against
a chars-per-token bandwidth constraint; if no, the rung never varied the quantity it was designed to
vary and the null says nothing about bandwidth. These are opposite conclusions, so it was measured
rather than assumed.

Script `scratchpad/h13_ink_occupancy.py` (CPU, no model weights, `CUDA_VISIBLE_DEVICES=""` — H13 is
still training and this repo's rule is that training runs SOLO). Output
`data/eval/h13-ink-occupancy.json`, plus `-560.json` / `-1120.json` for the budget sweep used by
[H17](../todo/H17-raise-soft-token-budget.md). 30 validation rows per rung, taken by index (not
shuffled) so d3 and d5 pair.

**Method.** Ink is counted on the *processor's own output*: the merged patch vector of soft token
`m` reshaped to 48×48 **is** the pixel block it covers, so no patchifier is reimplemented. A pixel
is ink when darkness (`1 - value`) > 8/255. Because a threshold can silently do all the work, inked
cells are reported at five per-cell cutoffs and the pooled per-cell histogram is printed: the ≥1 and
≥1%-of-cell counts differ by at most 3.2 cells on any rung (d5: 148.4 vs 145.2; d1–d4: 0 to 0.03),
so **the threshold is not doing hidden work** — cells are either blank paper or carry hundreds of
ink pixels (median non-zero cell: 198/215/448/693/437 px of 2304 for d1–d5).
The pixel→patch mapping is self-tested (`--self-test`) at every budget: exact raster identity for
all tokens against a synthetic image the processor does *not* resize, index order `m == y·W + x`,
asymmetric probe cells, a **mutation test** that a mapping shifted one cell must FAIL, and a centroid
check on the resized path. An off-by-one here would have fabricated the headline number.
Re-rendering each stored target reproduces the stored PNG byte-for-byte on **150/150 rows**.

**The resize is identical for every rung — verified, not assumed.** Every rung renders on the same
1024×1024 canvas, and the processor maps 1024² → **768×768**, emitting a **16×16 = 256** grid of
48px cells (max `image_position_ids` = 15 on both axes) on 100% of rows of all five rungs. So the
budget is 280 but only **256** tokens are ever produced, and one soft token covers **64 original
page pixels**, not 48.

| rung | font | chars | soft tok | inked ≥1 | inked frac | ch/soft | **ch/inked** | lines/cell | grid rows w/ink | words/inked |
|---|---|---|---|---|---|---|---|---|---|---|
| d1 | 14 | 29 | 256 | 5.0 | 2.0% | 0.11 | **5.80** | 3.76 | 1 | 1.00 |
| d2 | 14 | 119 | 256 | 16.0 | 6.2% | 0.46 | **7.44** | 3.76 | 1 | 1.25 |
| d3 | 14 | 479 | 256 | 32.0 | 12.5% | 1.87 | **14.97** | 3.76 | 2 | 2.50 |
| d4 | 14 | 1919 | 256 | 80.0 | 31.2% | 7.50 | **23.99** | 3.76 | 5 | 4.00 |
| **d5** | **40** | 479 | 256 | **148.4** | **58.0%** | 1.87 | **3.23** | **1.33** | **11** | 0.54 |
| *masked-randstr* | 14 | 476 | 256 | 32.0 | 12.5% | 1.86 | 14.90 | 3.76 | 2 | 2.50 |
| *fineweb-edu (page 0)* | 14 | 2954 | 256 | 148.5 | 58.0% | 11.54 | 19.16 | 3.76 | 10 | 3.08 |

`ch/inked` is the **mean of the per-row ratio**, which equals (mean chars)/(mean inked) only when the
inked count is constant across rows — it is on d1–d4, near enough on d5, but *not* on the fineweb
cross-reference (inked varies 40–255, sd 66.8; its ratio-of-means would read 19.89 rather than 19.16).

Row-to-row variance is ~0 for d1–d4 (the layout is deterministic at a fixed letter count); d5 varies
145–152 inked cells (sd 1.85) because font-40 wrapping depends on letter widths. `masked-randstr`
(rendered in memory, nothing materialized) reproduces the independently-reported 32/256 = 12.5%
figure exactly and confirms that lane's geometry **is** d3's.

### d3 vs d5, paired by row — the rung DID vary what it was meant to vary

Targets are byte-identical on **30/30** paired rows (as expected: same RNG stream, font is the only
difference).

| paired quantity | d3 | d5 | ratio d5/d3 [95% CI] | rows d5 > d3 |
|---|---|---|---|---|
| inked soft tokens (≥1 ink px) | 32.00 | 148.40 | **4.638 [4.618, 4.658]** | 30/30 |
| inked soft tokens (≥1% of cell) | 32.00 | 145.20 | 4.538 [4.533, 4.544] | 30/30 |
| page ink pixels | 13,339 | 66,332 | 4.973 [4.966, 4.980] | 30/30 |
| **characters per inked soft token** | **14.97** | **3.23** | **0.216** | — |

So **d5 spreads the identical 479 characters over 4.64× more inked soft tokens than d3**, cutting
characters-per-inked-token by 4.64×, on every single row, with a CI that excludes 1 by ~200 SE. Of
the two candidate readings of the null, the data supports the **first**: the rung is a real,
large manipulation of chars-per-inked-token; it is *not* a case of the two rungs never having
differed in the quantity that matters.

### What this does and does not license

**Licensed.** The "d5 failed to test what it was designed to test" escape is closed on the
inked-token axis. If the discriminator confirms d5 ≈ d3 on *reading gain*, then a 4.64× reduction in
characters-per-inked-soft-token — holding content, target, canvas, budget and recipe fixed — bought
nothing, which is a direct hit on the crude chars-per-soft-token form of **(A)**.

**Not licensed — five ways this stops short of a verdict.**

1. **The discriminator has not run.** The d5 ≡ d3 equality is currently only in aggregate eval CE,
   and this project has already recorded that eval CE is not a grounding monitor. Nothing here is a
   verdict; `scratchpad/h13_analyze.py` and the per-position probe still decide it.
2. **Both rungs are at the no-reading floor, so the null is censored.** At step 300 d3 and d5 both
   sit at 5.643 against a 5.6086 floor — *neither* reads measurably. A null between two non-reading
   rungs is "a 4.64× change was insufficient to cross zero", not "no effect measured against a
   non-zero baseline". The outcome instrument is at its floor; the effect size is bounded, not
   estimated.
3. **d5 is not a single-factor manipulation.** Changing the font changes chars/inked-token *and*
   stroke width, lines/cell (3.76 → 1.33), grid rows occupied (2 → 11) and the fraction of the token
   grid in use (12.5% → 58%). Chars-per-inked-token is one consequence of a render-geometry change,
   not an isolated dial. It remains the cleanest single-variable pair in this ladder (design flaw 1),
   but "cleanest" is not "clean".
4. **This is geometry, not information.** It counts ink on the page. It does not show that the
   ~88% blank cells of d3 carry nothing: a blank cell still emits a soft token, and that token is a
   function of (white patch, its position embedding). Because the embedder has no attention between
   patches, a blank cell's token is *structurally* content-free, but whether blank tokens are
   harmful, neutral, or a useful positional scaffold is **not measured here** and would need the
   model, i.e. the GPU.
5. **n = 30 rows/rung**, first-by-index. Adequate given near-zero within-rung variance, but the
   fineweb cross-reference (sd 66.8 inked cells, range 40–255) is genuinely noisy and its mean should
   not be quoted as if it were a per-page constant.

### Defect found in this doc's own deviation note

Deviation 1 states d5 hits "~1.0 text line per 48px patch … exactly (measured line height 48px;
d1–d4 sit at ~2.8 lines/patch)". Those figures divide 48 by the line height **on the original 1024
canvas**. But the patch grid lives on the **resized 768** image, where one cell is 48 resized px =
**64 original px**. Corrected: **d1–d4 = 3.76 lines/cell, d5 = 1.33** — every figure is 4/3 too low,
and d5's stated intent of ~1.0 line/patch was overshot to 1.33. The *ratio* between the arms (2.82×)
and hence the rung's design logic are unaffected. Note that
[H17](../todo/H17-raise-soft-token-budget.md)'s table quotes 3.8 lines per 48px patch for fineweb,
which matches the resized measurement (3.76) — the two docs disagree by exactly 4/3 and **H17 is the
correct one**.

### Surprise worth recording

d5 inks **58.0%** of the token grid — the same fraction as an average fineweb-edu page (58.0% at
n=30; 53.6% on a matched 20-row sample) — and runs at **3.23 chars per inked token, below d1's 5.80**,
i.e. below the density of the only rung that clearly reads. Read this cautiously: d5's target is ~232
tokens against d1's ~14.5, so per design flaw 1 the d1-vs-d5 comparison confounds density with target
length and cannot on its own separate (A) from (C). It is a hint, not a result.

---

## Verdict (2026-07-29)

### 1. The H13 training run is VOID — it never learned to read anything

`data/checkpoints/h13-density-v0/final` (1000 steps, finished clean) is a **dead checkpoint**:

| step | 10 | 110 | 310 | 510 | 710 | 910 |
|---|---|---|---|---|---|---|
| train loss | 7.291 | 5.674 | 5.640 | 5.636 | 5.641 | 5.641 |
| `grad_norm` | 1.023 | **0.061** | 0.067 | 0.088 | 0.069 | 0.062 |

The ablation (`data/eval/h13-density-ladder.json`, 150 rows/rung) is unambiguous about *why* the
discriminator could not run: **Δperm ≈ 0 on all five rungs** (+0.01, −0.00, −0.00, +0.00, +0.01 %)
while **Δblank is positive** (+1.0 % … +8.7 %). A *donor* randstr page costs the model nothing versus
the aligned page; only a *blank* page hurts. That is the
[H04](../done/H04-frozen-decoder-warmup.md) content-blind signature — the image is used as a generic
"there is text here" prompt and none of its content is read.

`scratchpad/h13_analyze.py` printed **VOID / AMBIGUOUS** by design, refusing to name a branch.
It was right to.

> **Trap for the reader of the trajectory table above.** Section 1's per-rung "READS" lines and the
> `*` marks in the Observation log both measure gain against a **blank** image, which confounds
> ink-presence with content. **Δperm is the content-isolating measure and it is flat.** Nothing in
> this run read anything.

### 2. Two candidate explanations were tested; both came back clean, so the failure is real

**Mis-pairing — FALSIFIED.** If stored images did not correspond to their targets, "aligned" would
*be* "permuted" by construction and every number above would follow trivially. Audited by
re-rendering each row's own target through the materializer's own path
(`data/eval/h13-data-integrity.json`, `scratchpad/h13_data_integrity.py`): **2000/2000
byte-identical** across 5 rungs × 2 splits, max abs pixel diff 0. Negative control (row *i*'s image
vs row *i+1*'s target) **0/2000** exact, ink-IoU ≤ 0.418, minimum separation margin **0.582**; a
**one-letter** change is already detected. Zero blank images; response-only mask covers exactly
`target + <|im_end|>\n`; 0 rows dropped by the length filter; the ablation's donor is a genuinely
different image.

**Probe defect — FALSIFIED,** by the positive control that also produced the verdict (§4).

### 3. Root cause: the ladder trained the model out of reading

Rows are balanced 20k/rung, but loss is normalised **per supervised token**, so gradient share
follows target length — and 86% of it lands on tokens beyond the model's reach:

| rung | tgt tok/row | gradient share | readable within depth ≈48 tok |
|---|---|---|---|
| d1 | 17 | **1.13 %** | 100 % |
| d2 | 60 | 4.11 % | 80 % |
| d3 | 234 | 15.86 % | 20.5 % |
| d4 | 932 | **63.04 %** | 5.2 % |
| d5 | 234 | 15.86 % | 20.5 % |

**85.8 % of supervised tokens lie beyond the scan depth**, and every one of them has a single
loss-minimising behaviour — emit the uniform-letter marginal. The adapter is the only from-scratch
component and reads nothing at step 0, so it must *bootstrap* from the 14 % of tokens where reading
is achievable, against a 86 % majority teaching the opposite. It lost by step 110.
[H07](../done/H07-pretrained-vision-adapter-qwen.md) succeeded on a lane that was **100 % readable**
and gave that lane **88× more gradient share** than d1 gets here.

**The rungs built to *measure* the depth limit instead *taught* the model to ignore pixels.** This is
[H09](../done/H09-balanced-mixture-prevents-drowning.md)'s lane-drowning in token-weight rather than
row-count terms, and it is the generalisable lesson: **balancing rows does not balance gradient.**

### 4. The hypothesis was answered anyway — by running the discriminator on H07's checkpoint

The pre-registered discriminator needs *a model that reads* plus *the ladder's val data*. H13 failed
to supply the first; [H07](../done/H07-pretrained-vision-adapter-qwen.md)'s checkpoint already is
one. Running `scratchpad/h13_analyze.py` unchanged against
`data/checkpoints/hybrid-pretrained-randstr-v0/final` (`data/eval/h13-poscontrol-h07.json`):

| rung | chars/page | Δperm | pos-0 gain | **K** |
|---|---|---|---|---|
| d1 | 29 | **+113.84 %** | +62.7 ± 4.0 | ≥29 (censored) |
| d2 | 119 | +14.31 % | +35.3 ± 3.9 | 100 ch / **tok 48.8** |
| d3 | 479 | +3.05 % | +28.7 ± 3.7 | 100 ch / **tok 48.5** |
| d4 | 1919 | +0.63 % | +28.0 ± 3.7 | 100 ch / **tok 48.5** |
| d5 | 479, font 40 | +0.01 % | +0.0 | no reading |

d1's **+113.84 %** replicates H07's independently documented **+114.7 %** — the probe is calibrated,
and the ladder data is readable by a model that can read.

**K = 48.8 / 48.5 / 48.5 tokens across d2 → d4**, a **16× span of page density**. The collapse point
does not move. Per the pre-registered criterion this is the **(C)** branch, and the
**"K scales with density ⇒ (A)"** branch is **refuted**.

The sharper form, which does not depend on K's definition: **position-0 gain is statistically
identical at d3 (+28.7 ± 3.7) and d4 (+28.0 ± 3.7)** even though d4's page is 4× denser and carries
~24 chars per inked soft token against d3's ~15. Perception *at the start of the page* does not
degrade with density at all — only depth does. Aggregate Δperm collapses 14.31 % → 0.63 % purely
because a fixed readable prefix is a shrinking fraction of a longer target. Density-proportional
bandwidth predicts position-0 should degrade too. It does not.

### 5. What this does NOT license

1. **(C) is not isolated.** chars/token ≈ 2.06 on every rung (design flaw 4), so character position
   and token index are proportional *by construction*. A K constant across rungs is equally
   consistent with **(C) fixed token-index scan** and with **fixed-absolute-capacity (A′)**. What is
   strictly established is: *the collapse point is invariant to page density.* That kills the
   density-proportional account and nothing more.
2. **The two survivors differ in one testable way** — raising the soft-token budget relieves a
   capacity limit but not a scan limit. So [H17](../todo/H17-raise-soft-token-budget.md) is **not**
   undermined by "more tokens will not help"; it becomes the precise discriminator between (C) and
   (A′). Its pre-registered "K must scale ≥3× for 4× the tokens" is exactly the right test.
3. **Measured on a model trained at ONE density.** H07 saw only d1-geometry font-14 randstr. Its K
   may reflect its training distribution rather than an architectural limit. This run establishes
   that K is invariant to **test-time** density; it does **not** establish invariance to **training**
   density. A repaired H13 (§7) is what would test that.
4. **d5 tells us nothing about line-demux here.** H07 never saw font 40, so d5's null is a
   *scale-generalisation* failure, not a bandwidth result — see §6. The pre-registered
   "D5 ≫ D3 ⇒ line-demux" branch remains **untested**, not refuted.
5. **d1's K is censored** (target ends inside the reference window), so the trend rests on d2/d3/d4.

### 6. Incidental finding: the vision path is not scale-invariant

d3 and d5 draw **byte-identical targets** and differ only in font (14 vs 40). Under the H07
checkpoint, paired on the same rows: d3 reads (pos-0 **+28.7**, Δperm +3.05 %), d5 reads **nothing**
(pos-0 **+0.0**, Δperm +0.01 %, and only 32 % of rows beat blank). A 2.9× glyph-scale change
destroys reading completely on text the same model transcribes at the smaller size. Worth its own
hypothesis; it bears on any "just re-render bigger" proposal, including
[H20](../todo/H20-audio-phoneme-resolution.md)'s wide-render leg.

### 7. Defect found in the lane's instrumentation — `floor.json` is wrong

The no-reading floor divides string entropy by *target* tokens, but response-only masking also
supervises the deterministic `<|im_end|>\n` trailer (2 tokens, ~0 nats), so the published floor is
too high by roughly a factor `T/(T−2)` — an error that grows as targets shorten.

| rung | published floor | **corrected floor** | H13 final CE | vs corrected |
|---|---|---|---|---|
| d1 | 5.5973 | **4.8902** | 4.994 | **+0.104** |
| d2 | 5.6050 | **5.3742** | 5.513 | **+0.139** |
| d3 | 5.6086 | **5.5701** | 5.634 | +0.064 |
| d4 | 5.6120 | **5.6050** | 5.658 | +0.053 |
| d5 | 5.6086 | **5.5701** | 5.634 | +0.064 |

Every rung is **above** its true floor. d1's headline "0.586 nats below floor ⇒ reading" in the
Observation log is an **artifact**, and so is every `*` in that trajectory table. This independently
corroborates the Δperm ≈ 0 result: nothing read.

**Generalisable lesson:** *"CE below the analytic floor ⇒ reading"* is unsafe unless the floor is
computed over **exactly the masked token set**, including deterministic format tokens. It is
wrong-by-most on short targets. The repo convention of stating criteria in Δperm / Δblank / reading
gain and **never** eval CE is what preserved this result — §4 is ablation-based and untouched by the
bug. `floor.json` in `h13-density-v0` (and any lane built by `random_strings.py`) should be
regenerated before being quoted.

### 8. Consequences for the remaining program

- **[H18](../todo/H18-no-learned-scan.md)** — resolved to the extent H13 can resolve it: the
  collapse point is density-invariant, consistent with no learned scan. But (A′) is not excluded, so
  H18 should not be closed as CONFIRMED until H17 reports.
- **[H17](../todo/H17-raise-soft-token-budget.md)** — **promoted to the critical path.** It is now
  the discriminator between (C) and (A′), not merely "the fix if (A)".
- **[H14](../in-progress/H14-masked-region-targets.md)** — as configured it measured **209.7 target tokens/row,
  i.e. 22.9 % readable**, squarely in H13's failure regime (14.2 % died; H07's 100 % succeeded).
  Launching it unchanged would very likely buy another VOID. Redesigned before launch — see its doc.
- **Every future run on this architecture** — check the *token-weighted* readable fraction before
  launch, and prefer warm-starting from a checkpoint that already reads. `init_from` was added to
  `univi/hybrid/train_pretrained.py` for exactly this (weights only, fresh optimizer/schedule).

### 9. Artifacts

- `data/eval/h13-density-ladder.json` / `.log` — discriminator on the H13 checkpoint (VOID).
- `data/eval/h13-poscontrol-h07.json` / `.log` — **the verdict**: same probe, H07 checkpoint.
- `data/eval/h13-data-integrity.json` — pairing audit (`scratchpad/h13_data_integrity.py --self-test`).
- `data/eval/h13-ink-occupancy{,-560,-1120}.json` — ink geometry (pre-verdict).
- `data/checkpoints/h13-density-v0/` (dead checkpoint, kept as the negative example),
  `data/checkpoints/h13-density-v0-run.log`, W&B `driven-frog-59` (`a6kv5ysx`).

### 10. Process note

The discriminator was armed as a background job gated on `until ! pgrep -f
"univi.hybrid.train_pretrained"`. The watcher's **own** shell command line contains that string, so
`pgrep -f` matched the watcher itself and the loop never exited — training had finished ~7 h earlier
and the probe never launched. Gate on a **PID** (`while [ -d /proc/$PID ]`), never on a pattern that
appears in the gating command.
