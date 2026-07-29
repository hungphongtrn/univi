# H20 — Audio needs ≤120 ms/token-column to transcribe

**Status:** TODO · **Cost:** ~1 h CPU prerequisite (wide re-render) + ~5 h GPU leg 1 + ~9 h GPU leg 2
— all GPU figures estimated, never measured · Attacks **(A)** for audio · Related:
[H08](../done/H08-audio-needs-trainable-vision.md), [H17](H17-raise-soft-token-budget.md)

## Claim

Audio needs the bandwidth fix unconditionally. At 280 soft tokens a spectrogram page is **244 ms per
token-column**, against phonemes of 50–150 ms. Spoken *digits* survive that coarseness; continuous
speech cannot.

## The evidence this rests on

- [H08](../done/H08-audio-needs-trainable-vision.md): prior-proof spoken digits read at **Δperm
  +134.8%** — roughly 1–2 token-columns per digit-word.
- [H09](../done/H09-balanced-mixture-prevents-drowning.md): continuous librispeech reads at +12.9%
  but with **zero phonetic correspondence** in decodes; the signal is utterance duration.
- [H10](../done/H10-reading-concentrated-at-start.md): librispeech reading gain at position 0 is
  **−12.7 pts** — no phonetic onset reading at all.
- Working ASR encodes at 20 ms/token (Whisper); patch-based audio models at ~160 ms patches do
  classification only, and no published vision-path patch model transcribes. See
  [spectrogram patch models](../../literature/spectrogram-patch-models-do-not-transcribe.md).

## Design

**Revised 2026-07-28 by direct measurement of the processor's audio geometry — see
[the render-geometry measurement](#render-geometry-measured-2026-07-28--the-budget-is-the-weaker-of-two-levers)
at the bottom, which is the authority where this section is ambiguous.** The short version: raising
the budget alone buys **2.02×** finer time resolution (not 4.25×), because time columns grow as
**√budget**; the render's *aspect ratio* moves the same quantity at no token cost, and reaches
finer than the budget ladder can.

| leg | render | budget | tokens/page | **ms/column** | mel bins/cell | what changes vs today |
|---|---|---|---|---|---|---|
| today ([H09](../done/H09-balanced-mixture-prevents-drowning.md)/[H10](../done/H10-reading-concentrated-at-start.md)) | 1000×160 | 280 | 246 | **243.9** | 13.3 | — |
| **leg 1** (cheap, single-variable) | **2000×160** | **560** | **498** | **120.5** | **13.3 — identical** | time resolution ONLY |
| **leg 2** (the main run) | **2000×160** | **1120** | **1062** | **84.7** | 8.9 | time 2.9×, frequency 1.5× |
| ~~as originally designed~~ | 1000×160 | 1120 | 1079 | 120.5 | 6.2 | **dominated by leg 2**: same cost, coarser in time |

Leg 1 is the attributable one: the render's 2× horizontal stretch and the 2× budget cancel exactly on
the frequency axis (6 grid rows over 80 mels, before and after), so the *only* thing that changes is
the time extent of one soft token. Leg 2 costs what the original design already costed and delivers
84.7 ms/column instead of 120.5.

**Prerequisite (CPU, no GPU, no audio corpus):** materialize `data/materialized/h20-audio-wide-v0` —
both lanes at 2000×160. Measured: a LANCZOS stretch of the *stored* 1000×160 PNG differs from a
from-audio re-render at `output_width 2000` by **0.72% of the page's own sd**, because at a fixed hop
the extra width is interpolation either way. So this is an image-space transform of the existing
artifact (~187k librispeech pages + 30k digit pages), not a re-materialization from LibriSpeech.
(The source audio *is* cached locally — 169 GB under `openslr___librispeech_asr` — so the from-audio
route is available too; it is just not necessary.)

Librispeech-only, 600–800 steps, **trainable vision** (established: frozen vision cannot read
spectrograms, [H08](../done/H08-audio-needs-trainable-vision.md)), gap-weighted loss from
[H16](H16-prior-gap-weighted-loss.md) — librispeech's English transcripts are exactly the
prior-shortcut lane — and spoken-digits as a 15% anchor.

**Not adopted, and why.** A 560 rung in [H17](H17-raise-soft-token-budget.md)'s sense (recovering
resolution the 280 downsample destroyed) is *meaningless for audio*: the spectrogram page is
upsampled at every budget, so there is nothing to recover and H17's reason for its leg 1 does not
transfer. Leg 1 here exists for a different reason — to hold the frequency axis fixed.

## Prep status (2026-07-28) — config written and validated, not launched

`configs/h20_librispeech_1120.yaml`: librispeech + spoken-digits, `max_soft_tokens: 1120`,
`freeze_vision: false` (3-group LR 3e-4 / 1e-4 / 2e-5), `gap_weighted_loss: true` + `log_blank_ce:
true`, `max_length: 8192`, per-device 4 × accum 16 (**effective batch 64**), 800 steps.

**Measured, not assumed:** both lanes render 1000×160 spectrogram pages → **1079 soft tokens/page at
1120** (246 at 280), exactly matching [H17](H17-raise-soft-token-budget.md). Assembled sequences:
librispeech 1128 / 2063 / 2262 (min/mean/max, ~3400 for 3-page rows), spoken-digits 1127. The
per-image filter budget resolves to **1122** from `model.max_soft_tokens`, confirming the
`image_token_budget` fix is live. End-to-end retention through `load_dataset`: **122,369 train rows
(104,014 librispeech + 18,355 spoken-digits), nothing silently dropped**; at `max_length 8192` every
split retains 100%. The two lanes share an identical 10-column schema, so `concatenate_datasets`
works with **no** `remove_columns` (unlike `poisoned-text` / `masked-randstr`).

### Three things this doc gets wrong or cannot express

1. ~~**"≤120 ms/token-column" is not a measured property of this render.**~~ **WRONG IN BOTH
   DIRECTIONS — superseded by
   [the render-geometry measurement](#render-geometry-measured-2026-07-28--the-budget-is-the-weaker-of-two-levers).**
   The original text read: *"The page is 10 s wide at 1000 px (10 ms per mel frame), and 1079 tokens
   do not tile the time axis — the Gemma unified embedder builds a 2-D patch grid, so 'token-column'
   is a convenience, and 120 ms is simply 244 × 280/1120. The defensible claim is the 4.25× budget
   increase."* Measured: the grid is **41×6 at 280 and 83×13 at 1120**, so a token *column* is a real
   object (41 and 83 of them), and 243.9 / 120.5 ms are exact measured cell time-extents — this
   doc's headline numbers were right and the correction was wrong to retreat from them. But the
   retreat position was wrong too: 244 × 280/1120 is **61**, not 120. Columns scale as **√budget**,
   so the 4.25× budget increase is a **2.02×** time-resolution increase. The defensible claim is
   neither "4.25×" nor "120 ms is unmeasurable" — it is "**2.02× finer time resolution, 120.5 ms per
   column, and no new optical information at all**".
2. **"spoken-digits as a 15% anchor" is inexpressible in the current config schema.**
   `dataset.max_train_rows_per_subset` is a *single global* cap and cannot set per-lane shares; the
   natural mixture would be 30,000/134,014 = **22.4%** (which is still inside the 5–25% replay window
   the design cites). To hit the specified 15% exactly, a deterministic 18,355-row subsample was
   materialized at `data/materialized/h20-audio-v0/spoken-digits-15pct/train` (seed 42, 546 MB).
   Reverting to the full lane for 22.4% is equally defensible and avoids the data copy.
3. **The two lanes live in different roots** and `_load_local` takes a single `dataset.path`, so
   `data/materialized/h20-audio-v0/` is a new root of **symlinks** plus a merged manifest — no image
   data is copied and the originals are untouched.

**VRAM is unconfirmed** (no GPU was available during prep): the batch mirrors H16's, but the 4-lane
run at per-device 16 already sat near 29 GB, so the launcher must verify **< 26 GB** empirically and
halve to 2 × 32 if needed. The doc's ~9 h estimate also looks optimistic at 4× sequence length plus
the blind branch's ≈ +35% wall-clock.

## Pre-registered criterion

**Revised 2026-07-28.** The three *confirms* conditions are unchanged; the *refute* branch is
re-scoped, because as written it claimed something the geometry does not support (see below).

- **Confirms** (read on leg 2, and on leg 1 if leg 1 runs first): position-0 reading gain
  **> +10 pts** (currently −12.7), **Δblank ≥ +30%**, and greedy decodes show **nonzero word-level
  overlap** with targets (currently zero — length-only). Probes:
  `scratchpad/hybrid_4lane_position_decay.py`, `scratchpad/hybrid_pretrained_ablation.py`,
  `scratchpad/hybrid_4lane_generate.py` — **not** aggregate eval CE.
- **Leg 1 (120.5 ms/column, frequency granularity held at today's 13.3 mel bins/cell).** If it
  confirms, time resolution *is* the binding constraint and it is buyable at 2.02× sequence cost with
  the frequency axis untouched — leg 2 then measures how far it goes. If leg 1 is flat and leg 2
  confirms, the gain needed either the extra tokens or the finer frequency axis, which this design
  cannot separate; say so rather than attributing it to time.
- **Refutes** — **narrowed:** decodes still show no phonetic correspondence at **84.7 ms/column with
  the frequency axis 1.5× finer than today** ⇒ time resolution in the 85–120 ms band is not the
  binding constraint. This does **not** license "the vision path cannot represent speech at any
  budget we can afford", which is what this doc used to say. The budget ladder is not the frontier:
  **4000×160 at 1120 gives 59.9 ms/column at 1002 tokens/page — cheaper than the run this doc
  originally proposed** — and only a null *there*, inside the 50–150 ms phoneme band with margin,
  licenses the "different front-end" conclusion.

**What no branch licenses.** Nothing here tests the **frequency** axis on its own; no rung resolves
an individual mel bin (13.3 bins per cell today, 6.2 at the finest rung measured, and 1 bin/cell
would need a ~40,000-token budget), so if formant structure is what the model needs, every rung in
this design fails for a reason the design cannot see. Nothing here measures whether the **31% of
librispeech soft tokens that are right-padded silence** (44% on validation) are harmful, neutral, or
a useful positional scaffold — trimming pages to the utterance is an orthogonal ~1.45× effective
budget increase that no leg tests. And the whole ladder assumes the pretrained Gemma patch embedder
*preserves* within-cell structure; that is a property of learned weights and is unmeasured (it needs
the GPU).

## Render geometry, measured (2026-07-28) — the budget is the weaker of two levers

[H17](H17-raise-soft-token-budget.md)'s fifth correction reported, from one agent's measurement,
that the 1000×160 spectrogram page is **upsampled at every budget (1.97× even at 280)**, and told
this doc to derive its own criterion. That claim decides whether a ~9 h run is worth launching, so it
was re-derived independently, on **real materialized pages** (46 librispeech pages from 24 rows, 24
spoken-digits pages, plus whole-split column scans of 104,014 + 2,703 rows), by a different code
path. `scratchpad/h20_audio_resolution.py` → `data/eval/h20-audio-resolution.json`. CPU only, no
weights, no GPU (a training run holds the card). Nine self-tests gate the numbers, including an exact
raster round-trip of the pixel→cell mapping on a **non-square** page at a genuine fixed point of the
resize, and a mutation test that a one-cell-shifted mapping fails.

**The 1.97× replicates exactly, and sharpens: it is 1.968× on the time axis and 1.800× on the
frequency axis.** Both are upsamples, at every budget, on every page of both lanes.

### The resize rule for NON-SQUARE pages

H17 states the rule for square pages ("the resized side is a function of the budget alone"). That is
a special case. The general rule, re-derived from `get_aspect_ratio_preserving_size` and verified
against it over a 5 budgets × 8 heights × 8 widths sweep (including the degenerate extreme-aspect
branch, 0 mismatches):

> `s = sqrt(B · 48² / (H · W))`, then `target_h = floor(s·H / 48)·48`, `target_w = floor(s·W / 48)·48`.

The processor targets a constant **area** of ≈ B cells of 48 px, preserving aspect ratio — not a
constant side. H17's rule is the `H = W` case falling out of the algebra: `s·H` becomes
`48·√B`, the canvas cancels, and the target side is `floor(√B)·48` = 768 / 1104 / 1584 for
280 / 560 / 1120 whatever the render canvas. For a non-square page the *sides* depend on the aspect
ratio while the *token count* does not, and the up/down-sample direction is decided by source area
versus `B · 2304`. The audio page is 160,000 px against 645,120 px at budget 280 — which is why it is
upsampled ~4× in area at the *lowest* rung.

Corollary that matters more than the rule: **tokens = (page_ms / ms_per_column) × (n_mels /
mel_bins_per_row)**. Token count and granularity are the same quantity for audio; they cannot be
varied independently. The only genuinely free variable is *how the tokens are split between the two
axes*, and that is set by the render's aspect ratio.

### Per-budget table — identical for both audio lanes

Both lanes store **1000×160** pages, exactly as this doc assumed. Multi-page rows do **not** differ:
the renderer right-pads the samples to `n_pages × page_frames` before slicing, so every page of every
row is 1000×160 (46/46 librispeech pages checked, including 2- and 3-page rows; 24/24 digit pages).
Geometry is therefore identical for librispeech and spoken-digits:

| budget | resized | scale time / freq | grid (time × freq) | tokens | max pos id | **ms/column** | mel frames/cell | mel bins/cell |
|---|---|---|---|---|---|---|---|---|
| 280 | 1968×288 | **1.968× / 1.800× UP** | 41 × 6 | 246 | 40 / 5 | **243.9** | 24.4 | 13.3 |
| 560 | 2832×432 | 2.832× / 2.700× UP | 59 × 9 | 531 | 58 / 8 | 169.5 | 16.9 | 8.9 |
| 1120 | 3984×624 | 3.984× / 3.900× UP | 83 × 13 | 1079 | 82 / 12 | **120.5** | 12.0 | 6.2 |

Token counts and max position ids reproduce H17's prep table exactly by a third code path. Every
`image_position_id` stays far inside the pretrained table's 1120 rows.

Three readings:

1. **This doc's headline numbers are right and its own correction #1 was wrong to retreat from
   them.** 244 ms and 120 ms are *measured cell time-extents*: the grid really has 41 and 83 columns.
2. **The gain is 2.02×, not 4.25×.** Columns go 41 → 59 → 83, i.e. ×√2 per rung, because the
   processor holds *area* constant: `columns = sqrt(B · W/H)`. A 4.25× budget buys a 2.02× finer time
   axis. The doc's own arithmetic for this ("244 × 280/1120") gives 61 ms and is simply wrong; the
   right scaling is 244/√4.
3. **1120 lands at 120.5 ms — the coarse edge of the 50–150 ms phoneme band, not inside it.** At that
   rung a token column still spans roughly one whole phoneme.

### Is any optical detail waiting to be recovered? No — measured, not just argued

The upsample claim is geometric; here it is a measurement. The processor's own output was reassembled
from `pixel_values`, resampled back to 1000×160 and compared to the stored page (6 real pages ×
3 budgets):

| budget | round-trip RMSE / page sd |
|---|---|
| 280 | **3.288%** |
| 560 | 3.273% |
| 1120 | 3.268% |

**Flat — a 0.020 pp spread over a 4.25× budget range.** If 280 were destroying detail, its residual
would stand out; it does not. The model-visible page is a near-invertible view of the stored page at *every* rung, and the
0.02 pp spread across a 4.25× budget range is the whole optical benefit on offer. (Upper bound, not
an estimate: the round trip includes the return resize's own error, and there is no downsampling
audio rung to serve as a negative control.)

### Occupancy — and it is budget-invariant

A spectrogram has no blank paper, so H13's "inked" is vacuous here. Three definitions, none
privileged: `structured` = cells whose pixel sd exceeds 1/255 (a *constant* cell carries nothing
beyond its mean); `active` = cell mean darkness above the page's own silence floor (1st percentile)
by a margin, reported at three margins so the threshold can be seen not to be doing the work; and
`over-audio` = the exact, threshold-free fraction of columns overlapping the un-padded audio.

| lane | budget | structured | active +0.02 / +0.05 / +0.10 | over-audio |
|---|---|---|---|---|
| librispeech | 280 | 69.0% | 66.7 / 64.3 / 59.3% | 70.0% |
| librispeech | 560 | 68.2% | 65.5 / 62.8 / 57.8% | 69.8% |
| librispeech | 1120 | 67.1% | 64.4 / 61.4 / 56.6% | 69.6% |
| spoken-digits | 280 | 35.8% | 35.1 / 34.4 / 33.2% | *n/a* |
| spoken-digits | 1120 | 35.5% | 33.2 / 31.3 / 29.3% | *n/a* |

Occupancy is essentially **budget-invariant** (the same result H17 found for text ink): the extra
cells land on signal in about the same proportion, drifting down ~2 pp because a finer grid resolves
the silent gaps slightly better. *n/a*: spoken-digits' `render_config` records no
`source_duration_ms`, so its over-audio figure is 100% by construction and is reported as vacuous in
the JSON rather than quoted here; its honest occupancy is the ~36% structured figure — two thirds of
a digit page's soft tokens are silence between digits.

**A larger and previously unrecorded waste, whole-split and exact** (from `render_config`, no
thresholds): librispeech train is 104,014 rows / **187,497 pages**, mean utterance 12.6 s, and audio
covers only **69.0%** of the rendered time axis — **31.0% of the lane's soft tokens are right-padded
silence**. Validation is worse: 3,340 pages, mean 7.2 s, **44.4% padding**. The pages-per-row
histograms reproduce the prep-status numbers exactly (train `{1: 20548, 2: 83449, 3: 17}`,
validation `{1: 2136, 2: 506, 3: 52, 4: 9}`). Trimming pages to the utterance would recover ~1.45× of
the effective budget for free — orthogonal to everything below, and untested.

### What this does to the rationale: "optical detail" vs "representational capacity"

The two must be separated, and H20 depends on **neither** of the obvious ones:

- **Optical detail available.** Fully present at 280 already — every axis is upsampled and the
  round-trip is flat. Raising the budget cannot deliver information the render did not carry.
  *H20 never claimed otherwise; but H17's ladder does, and H20 must not inherit that reasoning.*
- **Linear capacity to encode it.** One 48 px cell at 280 carries 24.4 mel frames × 13.3 mel bins ≈
  **325 real mel values** into a **2048-dim** soft token. Page-wide: 1000 frames × 80 mels = 80,000
  real values into 246 × 2048 = 503,808 dims, i.e. **6.30 embedding dimensions per real mel value**
  at 280, rising to 27.6 at 1120. The channel is over-provisioned by a factor of six at the *lowest*
  rung, so "the tokens cannot hold the page" is not the mechanism either.
- **Granularity of an addressable unit — this is the one H20 actually depends on, and it is real.**
  `univi/hybrid/vision.py` has **no cross-patch mixing anywhere**: LN → Linear → LN → +factorized
  position → LN → RMSNorm → Linear, all per-patch, no attention. A soft token is an affine function
  of exactly one 48×48 cell, so the token grid *is* the model's spatial addressing resolution, and
  every cross-cell integration is the decoder's attention. Halving the time extent of a cell halves
  the span the decoder must resolve *inside* an unaddressable unit. That mechanism survives the
  measurement intact — what does not survive is the claim that the budget is the way to buy it.

### The decisive practical question: does a wider render deliver more, or get resized away?

**It is not resized away.** The critical render width — the largest at which the processor still
keeps every source pixel column on the time axis — is:

| page height | budget 280 | 560 | 1120 |
|---|---|---|---|
| 160 px | **3,984** | 8,016 | 16,080 |
| 80 px | 8,016 | 16,080 | 32,208 |

A 2000 px page is nowhere near any of them. And because `columns = sqrt(B · W/H)`, the render's
aspect ratio moves ms/column exactly as hard as the budget does, **at no token cost** — at a fixed
budget the token count stays ≈ B whatever the aspect (246 / 236 / 249 for 1000×160, 2000×160 and
2000×80 at budget 280):

| render | hop | budget | grid | tokens/page | **ms/column** | mel bins/cell | real frames/cell |
|---|---|---|---|---|---|---|---|
| 1000×160 (production) | 10 ms | 280 | 41×6 | 246 | 243.9 | 13.3 | 24.4 |
| 1000×160 | 10 ms | 560 | 59×9 | 531 | 169.5 | 8.9 | 16.9 |
| 1000×160 | 10 ms | 1120 | 83×13 | 1079 | 120.5 | 6.2 | 12.0 |
| **2000×160** | 10 ms | 280 | 59×4 | **236** | **169.5** | 20.0 | 16.9 |
| **2000×160** | 10 ms | **560** | 83×6 | **498** | **120.5** | **13.3** | 12.0 |
| **2000×160** | 10 ms | **1120** | 118×9 | **1062** | **84.7** | 8.9 | 8.5 |
| 1000×80 (no vertical stretch) | 10 ms | 280 | 59×4 | 236 | 169.5 | 20.0 | 16.9 |
| 2000×80 | 5 ms | 280 | 83×3 | **249** | **120.5** | 26.7 | 24.1 |
| 4000×160 | 2.5 ms | 1120 | 167×6 | 1002 | **59.9** | 13.3 | 24.0 |

The grid depends only on `(render W, render H, budget)`. The hop column changes **only** `real
frames/cell` — every ms/column figure above holds at the production 10 ms hop too.

Three consequences, in order of how much they change the design:

1. **The doc's "optionally re-render at `output_width 2000` → ~85 ms/column" is correct, and it is
   not optional — it is the better lever.** At budget 1120 it delivers **84.7 ms/column for 1062
   tokens**, i.e. *cheaper* than the 1079 tokens the current render costs at the same budget, while
   being 1.42× finer in time. The originally designed run (1000×160 @ 1120) is **dominated**.
2. **The 2× vertical stretch is spending time resolution.** 160 px for 80 mel bins is a pure LANCZOS
   upsample that carries no information, and because tokens are allocated by area, it costs a √2
   factor on the time axis: 1000×**80** at 280 gives 59 columns where 1000×160 gives 41 — 169.5 ms
   instead of 243.9, for *fewer* tokens (236 vs 246), for free.
3. **H20's target time resolution is reachable at budget 280.** A 2000×80 render at 280 gives
   **120.5 ms/column in 249 tokens** — the number this doc is named after, at ~1/4 of the sequence
   cost it proposes to pay. The cost is on the other axis: 26.7 mel bins per cell instead of 13.3.
   The token budget is a *fixed pot* that the aspect ratio divides between time and frequency.

**Does a wider render carry more real information, or only more grid?** Mostly only more grid —
measured, on 8 real LibriSpeech clips. Halving the hop to 5 ms produces frames that a linear
interpolation of the 10 ms frames already predicts to within **9.1% of the signal's own sd** (42% of
a typical adjacent-frame step): the 25 ms `n_fft` analysis window bounds how much a finer hop *can*
add. So "re-render wider" should be understood as **reallocating the token grid**, not as supplying
new detail — which is exactly what is needed, since there is no missing detail to supply. It also
means the wide render needs no audio corpus: rendering the mel directly at 2000×160 versus
LANCZOS-stretching the *stored* 1000×160 page differs by **0.72% of the page sd**.

### The frequency axis

The page is 160 px for 80 mel bins, so one soft-token row spans **13.3 mel bins at 280, 8.9 at 560,
6.2 at 1120** — **no budget on the supported ladder resolves an individual mel bin**, and none comes
close: 1 bin/row at this aspect ratio needs a grid of 80 rows, i.e. a ~40,000-token budget. This is
*not* the same as saying the bins are lost — at 2.0 source px per bin (3.6 model-visible px after the
1.800× frequency upsample at 280) they are fully present *inside* the cell, and the patch embedder
sees them as distinct pixel rows. Whether the pretrained linear patch map preserves that fine vertical structure is a
property of the learned weights and is **not measurable without loading them** (the GPU is reserved).
So the frequency axis is best described as: fully resolved optically, never resolved *addressably*,
and untested either way.

### Retention at `max_length: 8192` — verified, 100% everywhere

The [H17](H17-raise-soft-token-budget.md) trap (a filter tuned for 280 silently dropping every row of
the deepest rung) does **not** bite H20. Measured through the real
`univi.trainer._filter_training_tokens` with `per_image_tokens = univi.trainer.image_token_budget(b)`
— `image_token_budget(1120) = 1122`, `image_token_budget(280) = 282` = the legacy constant:

| split | rows | retained @280 | retained @1120 |
|---|---|---|---|
| `librispeech/train` | 104,014 | 104,014 (100.00%) | **104,014 (100.00%)** |
| `librispeech/validation` | 2,703 | 2,703 (100.00%) | **2,703 (100.00%)** |
| `spoken-digits-15pct/train` | 18,355 | 18,355 (100.00%) | **18,355 (100.00%)** |
| `spoken-digits/validation` | 1,000 | 1,000 (100.00%) | **1,000 (100.00%)** |

The revised legs are safer still, not riskier: leg 1 costs 498 tokens/page and leg 2 costs 1062, both
below the 1079 this was validated at, and `max_train_images: 4` still covers the 4-page validation
rows (4 × 1062 + 50 + 256 = 4554 ≪ 8192).

### What remains unverified without a GPU

- Whether the pretrained Gemma patch embedder actually preserves within-cell time/frequency structure
  — the entire "granularity is the binding constraint" mechanism assumes it does something useful
  with 325 mel values in one 2048-dim slot, and that is a claim about learned weights.
- Whether blank/padded soft tokens (31% of this lane) are harmful, neutral, or a positional scaffold.
- VRAM at leg 2's 1062 tokens/page; the prep status' "must confirm < 26 GB empirically" stands
  unchanged, and leg 1 at 498 tokens/page should sit near half of it.
- All wall-clock estimates. Leg 1 is 2.02× today's tokens/page and leg 2 4.32×, so the "~9 h"
  headline remains optimistic for leg 2 and roughly right for leg 1.
- The wide re-render itself is *arithmetically* validated (grid, tokens, position ids) but no
  2000×160 page has been materialized or fed through a model; the 0.72% stretch-equivalence figure is
  8 clips, not the corpus.
