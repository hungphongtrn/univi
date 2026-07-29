# H14 — Mask-dependent targets force reading *and* localization

**Status:** RUNNING (launched 2026-07-29, readable-density variant) · **Cost:** ~1.6 h GPU · Attacks **(B) objective** and **(C) scan** together · Origin: user proposal, 2026-07-28 · Related: [H13](../done/H13-density-ladder-long-targets.md), [H15](H15-prior-poisoned-text.md), [H18](H18-no-learned-scan.md)

## Design change forced by H13 (2026-07-29, pre-launch)

[H13](../done/H13-density-ladder-long-targets.md) established that the model reads only to a depth of
**~48 answer tokens**, and that a run whose supervised tokens fall mostly *beyond* that depth
collapses into the uniform-marginal basin — every unreachable token teaches "ignore the image", and
a from-scratch adapter loses the bootstrap race. H13 died at **14.2% readable**;
[H07](../done/H07-pretrained-vision-adapter-qwen.md) succeeded at **100%**.

H14 as originally configured (`configs/h14_masked_randstr.yaml`, 80 word-groups) materialized at
**209.7 target tokens/row ⇒ 22.9% readable** — nearer H13's failure regime than H07's success. It
would most likely have returned another VOID and said nothing about masks.

**Two changes, both recorded before launch:**

1. **Readable density.** `configs/h14_masked_short.yaml` renders **24 groups** ⇒ measured **63.8
   target tokens/row = 75.2% readable**. Masking is undisturbed: 22.4% mask rate, 3.08 masked runs
   and 18.6 visible words per row, ~1540 masked runs across 500 validation rows.
2. **Warm-start.** `model.init_from` (new seam in `univi/hybrid/train_pretrained.py`) loads
   `hybrid-pretrained-randstr-v0/final` — a checkpoint that already reads at +115% Δperm — so the
   adapter does not re-run the bootstrap lottery at all. Weights only; fresh optimizer and schedule.

**The 80-group set is kept, not discarded** (`data/materialized/h14-masked-v0`, 50k rows). It is the
harder variant, to be run once the mask mechanism is shown to work where the model can actually read.

**What this costs.** The shortened page is sparser, so H14 now tests localization within a
~24-word field rather than an ~80-word one. That is a weaker test of the cursor than the original
design intended, and a positive result licenses "the model can localize masks at d2 density", not
"at full-page density". The 80-group run is what would extend it.

**Not affected:** the pre-registered criteria (span-boundary accuracy, hallucinated-content rate,
mask-permutation sensitivity, Δperm) are all density-independent, and none of them touches eval CE
or `floor.json` — which matters, because H13 §7 found this lane's floor is computed over target
tokens while response-only masking also supervises the deterministic `<|im_end|>\n` trailer,
inflating it by ~`T/(T−2)`. Do **not** quote `floor.json` (4.7658) as a reading threshold.

**Launch:** `configs/h14_masked_short.yaml`, 400 steps, eff batch 64, 14.5 s/step ⇒ ~1.6 h,
21.9 GB peak VRAM (26 GB budget). Log `data/checkpoints/h14-masked-short-v0-run.log`.

## Claim

Occlude random regions of the rendered image and make the **target depend on the mask** — the model
must transcribe only what is visible and mark what is hidden. Because which spans are masked is
random per example, no language prior can predict the correct output. A prior-driven model emits
fluent continuous text and is immediately wrong.

## Why this is attractive

It is the only proposal that attacks **two** constraints at once:

- **(B) prior** — the target is a function of a random mask, so the prior cannot supply it. Unlike
  [H15](H15-prior-poisoned-text.md), it does this without corrupting the source text.
- **(C) scan** — to know that a span is hidden, the model must **localize**: it has to know where it
  is on the page. This directly trains the cursor mechanism
  [H11](../done/H11-position-decay-is-prior-induced.md) suggests is missing.

## Design

**Render.** Draw the page, record per-word bounding boxes (`data/preprocessing/render_utils.py`
needs to emit them), then occlude a random subset: contiguous runs of 1–3 words, ~15–30% of words
total, solid fill.

**Target format** — recommend (b):

| variant | target for "the cat sat on the mat" with "sat" masked | note |
|---|---|---|
| (a) omit | `the cat on the mat` | ambiguous — cannot tell something was dropped |
| (b) **sentinel** | `the cat <mask> on the mat` | **preferred** — proves localization |
| (c) sentinel + length | `the cat <mask:1> on the mat` | harder; hold in reserve |

**Metrics** (all uncontaminated by CE):
- **span-boundary accuracy** — accuracy at sentinel positions and the tokens immediately after.
- **hallucinated-content rate** — fraction of masked words that appear in the output anyway. A
  prior-driven model fills them in; a reading model cannot know them. This is a *pure* reading
  metric.
- **mask-permutation control** (novel, and the sharpest): same image, **different mask**. If the
  model reads, changing only the mask must change the output. This isolates mask perception from
  everything else.
- Standard aligned / permuted / blank ablation alongside.

**Staging.** Instantiate first on the **randstr** lane (already prior-proof, already reads at +115%)
with long targets, so the mask mechanism is tested without the density confound. Port to real text
only after it works there.

## Pre-registered criterion

- **Confirms:** span-boundary accuracy ≥ 60%, hallucinated-content rate ≤ 15%, and mask-permutation
  changes the output on ≥ 80% of rows.
- **Refutes:** hallucinated-content rate ≈ the unmasked-baseline fabrication rate ⇒ the model is
  ignoring masks and generating from priors.

## Prep status (2026-07-28) — materializer BUILT and verified, not yet run

`data/preprocessing/masked_regions.py` (lane `masked-randstr`; `masked-text` reserved for the real-text
port, which raises `NotImplementedError` for now). `render_utils.py` gained `WordBox`,
`word_boxes_for_pages`, `render_text_pages_with_boxes`, `fill_boxes`; the only change to existing
functions was hoisting the `margin = 20` literal to `TEXT_MARGIN`. Proven **pixel-identical** to
pre-edit `HEAD` over 99 cases / 316 pages (0 mismatches) — this covers the earlier `wrap_text_pages`
refactor too. 4 regression tests added in `tests/test_render_utils.py` pin the box↔pixel invariant.

**Occlusion needed a 1px horizontal dilation.** `WordBox` is the *advance* box and glyph antialiasing
bleeds past it: filling the bare boxes left **41 stray subpixels over 100 pages**, all on the `x1`
column — a genuine leak of text the model must not see. Defaults are now `mask_pad_x=1, mask_pad_y=0`
(→ 0 residual ink over 100 pages), asymmetric so a rect never crosses into the neighbouring text line.
The fineweb port will need `mask_pad_y >= 2` (accented capitals overshoot the ascent) and a re-measure.

Verified on 32 rows / 200+ aggregate: 2560 word boxes all containing ink, 0 phantom boxes;
**33.1M outside-box pixels identical** to the clean re-render, every masked word's ink fully covered,
2019 visible-word boxes byte-identical; sentinel count == run count with no masked word leaking;
mask rate **0.2205 ± 0.042**, 100% inside the 15–30% band over 20k draws; 9.4 runs/row, mean length
1.88; targets 209 tokens, 1 image/row. Storage is deterministic — the stored PNG rebuilds exactly from
the stored columns, so an eval can construct any variant without re-deriving geometry.

Re-verified after the jitter change on 200 rows: 16,000 word boxes with 0 lacking ink, **206.8M
outside-box pixels compared with 0 mismatches**, 0 residual subpixels inside masked boxes
(`mask_pad_x=1` still suffices at 3–7 letters), 12,435 visible boxes byte-identical, sentinel count ==
run count 200/200, 0 masked-word leaks, mask rate 0.2228 ± 0.0417 fully in band, and `render_utils.py`
still pixel-identical to `HEAD` (140/140 pages).

**Doc correction:** the floor quoted above as *4.8966* is an **N=32** figure. The same pre-jitter
generator gives 4.8380 at 500 rows (4.8561 at N=100, 4.8419 at N=200) — sampling variance in the
mask-rate draw, not a derivation change. Quote the 500-row numbers.

**Metric support (the degenerate-shortcut guard the risk section below demands):** `visible_words`
scores reading independently of sentinel placement; `sentinel_char_starts` / `sentinel_visible_index`
score localization; `words[mask_word_indices]` gives ground truth for hallucination rate (floor
26⁻⁵ = 8.4e-08); `mask_b_*` + `mask_b_target` give the mask-permutation control by re-rendering the
same words under a second mask. That reading is written into `floor.json["note"]` so it travels with
the data. Floor: 4.8966 nats/tok (strict lower bound — visible-letter entropy only).

### Judgement calls and one open design issue

- **Runs may cross a line break** (one sentinel, one rect per line touched), and **≥1 visible word is
  enforced between runs** — without that, two adjacent runs render as a single block while the target
  carries two sentinels, an unanswerable row.
- **Block width leaks run length — DONE, but my stated reason for caring was wrong.** Same-line words
  of a run merge into one solid block, so with fixed 5-letter groups the block width revealed whether
  1, 2 or 3 words were hidden. I claimed this "inflates span-boundary accuracy"; **it does not.**
  Under variant (b) the target carries one sentinel per run and never encodes *k*, and both the
  number of blocks (= number of sentinels) and each block's right edge (= where to resume) are
  visible from the page regardless of jitter. Knowing *k* supplies nothing the target asks for. The
  leak *would* matter for a variant-(c) port.

  **The fix is still worth having, for a different and better reason:** with a fixed word length a
  non-reading box-detector got the word-length structure for free. Jittering (`--group-len-min/max`,
  default **3–7**) raises the analytic floor by **+0.477 nats/tok (+9.9%)** — a non-reader must now
  also predict where each word ends. Measured over 500 val rows: letters-only floor 4.8266 (≈
  unchanged, mean length still 5.0), and a new `no_reading_floor_with_length_nats_per_token` =
  **5.3038**. Word lengths are uniform (3:7988, 4:8102, 5:7927, 6:8107, 7:7876).

  **The leak is only partly closed, and jitter cannot close it.** Measured on rendered pixels over
  200 rows: fixed-5 was a *lossless* readout (Pearson r = 1.0000, 100% of the mutual information,
  100% width-only MAP accuracy); jitter 3–7 gives r = 0.9326, 87% MI, 94% MAP (majority baseline
  43%). An analytic sweep shows even **4–6 stays perfectly separable** and 1–15 still leaves 71% MAP
  — because a rectangle covering *k* words has width ≈ the sum of *k* word widths, so decoupling
  needs per-word variance comparable to the mean. 2–8/2–9 is the cheap next step (66%/62% MI) if this
  ever matters.

  Side effects: `masked_word_guess_accuracy` now means `E_L[26^-L]` = **1.18e-05** (was 26⁻⁵ =
  8.42e-08), still far below any real fabrication rate; `--group-len` was removed (set
  `min == max` for the old distribution).
- Per-row mask rate is drawn `U(0.15, 0.30)` rather than fixed, so a constant sentinel count cannot be
  learned. Fill is solid black on white. Sentinel offsets are **character**-indexed (a token-level
  probe must map via `return_offsets_mapping`).
- The prompt now states the convention (*"…Write `<mask>` in place of each blacked-out region."*) —
  the sentinel string is otherwise unguessable.
- 12 extra columns are stored; like `poisoned-text` they break `concatenate_datasets` against other
  lanes, so [H19](H19-four-lane-rematch.md) must `remove_columns` first.
- **Geometry caveat feeding the secondary risk below:** at `--n-groups 80` the word pitch is ~50px
  against a 48px patch — roughly **one word per vision patch**, ~2.8 lines/patch. Word-level
  localization sits right at the patch limit, so if [H13](../done/H13-density-ladder-long-targets.md)
  reports that d3 already strains the scan, H14 inherits that limit.
- ~~Not built: a training config and the eval script consuming these columns.~~ **Both built
  2026-07-28 — see "Config + eval built" below.**

## Known risk

**Degenerate shortcut:** the model may learn to detect occlusion boxes visually and emit sentinels
*without reading the surrounding text*. Guard by scoring accuracy on the **visible** text separately
— the task still requires transcribing what is shown. If visible-text accuracy stays at the
[H09](../done/H09-balanced-mixture-prevents-drowning.md) floor while span-boundary accuracy is high,
the model has learned box-detection only, and the result does not support the claim.

Secondary risk: at 10.7 chars/soft-token the model may not resolve individual words at all, making
the task unlearnable for reasons unrelated to masking — hence the prior-proof staging above, and
why [H13](../done/H13-density-ladder-long-targets.md) should report first.

## Config + eval built (2026-07-28) — STILL TODO: needs materialization + a free GPU

`configs/h14_masked_randstr.yaml` and `scratchpad/h14_masked_eval.py`. **Nothing here touched the
GPU** (H13 was holding the card), so every claim below is a CPU measurement or an extrapolation, and
the extrapolations are labelled. Materialization has **not** been run either — the config points at
`data/materialized/h14-masked-v0`, which does not exist yet.

### Config — the numbers behind each choice

Recipe is H07 verbatim (3-group LR 3e-4 / 5e-5 / 2e-5, **no freeze**, vision trainable), effective
batch 64 as `4 x 16`, `max_soft_tokens: 280`. Only the lane changes.

- **`max_length: 1024`, measured not guessed.** The target-token distribution was computed *exactly*
  for the planned split — same seeded RNG stems as `_make_row`, tokenizer
  `encoder-free-v0/best`, no rendering — over the real 20k train draw: **mean 209.3, p50 209, p99 231,
  MAX 247** (val 500 rows: mean 209.7, max 232). This **confirms the doc's "targets 209 tokens" at
  scale**; it was an N=32 figure. `_filter_training_tokens` charges `otl + image_token_budget(280)=282
  + 256` template, so worst case 785 ⇒ retention is **1024 → 20000/20000 (100.00%)**, 768 → 98.85%
  (it would silently *drop* 230 real rows, not truncate them), 2048 → no extra retention.
- **The real assembled sequence is 493 tokens, not ~2048.** Twenty materialized rows pushed through
  the real `HybridCollator`: 256 image soft tokens + 209 target + **exactly 32** tokens of chat
  scaffold ⇒ mean 493, max 535. So 1024 leaves 489 tokens of headroom and can never truncate.
- **A 1024² page is 256 soft tokens, not 280** (verified: `num_soft_tokens_per_image = 256`,
  `image_position_ids` max 15 ⇒ a 16×16 grid, `pixel_values` `(1, 280, 6912)` = 3×48² per slot).
  The processor first resizes 1024 → **768** (max_patches 2520 ⇒ 768 is the largest multiple of
  `pooling*patch = 48`). Only the dataset length *filter* charges the conservative 282.
- **`max_steps: 700` ≈ 5.1 h — extrapolated, not measured.** H13 is running the identical `4 x 16`
  geometry at a measured **31.1 s/step** (step 671 at 5:47:58) with mean sequence ~577 tokens; H14's
  493 is 0.85× that ⇒ ~26 s/step. 700 steps also exceeds H07's 600-step exposure (which reached
  +115% Δperm), and a kill at 600 replicates H07 exactly.
- **VRAM is an argument, not a measurement.** H14's worst row (535 tokens) is ~0.43× H13's d4 row
  (~1230), at the same per-device batch of 4, so peak activation is bounded above by a configuration
  already observed to fit. `8 x 8` (3.9k tokens/microbatch vs H13's safe 4.9k) would very likely fit
  and roughly halve wall-clock; it was not taken because it changes the micro-batch geometry at the
  same time as the lane.
- **Materialize 50k train rows, not the module docstring's 20k.** At effective batch 64, 700 steps
  sees 44,800 rows. Against a 20k split that is **2.24 epochs** — every page shown twice, on a lane
  whose entire premise is that the target is unguessable, which reintroduces memorization as a weak
  confound. H07 saw 0.38 epochs. 50k keeps H14 under one epoch (~40 min of CPU rendering).

### Eval — `scratchpad/h14_masked_eval.py`

All four pre-registered metrics plus the aligned/permuted/blank ablation, `--self-test` (**90
assertions**, no torch, no GPU, no checkpoint), `--skip-decode` for a fast teacher-forced-only pass,
`--max-soft-tokens` defaulting to the *checkpoint's* recorded budget via `resolve_max_soft_tokens`,
blank control from `univi.hybrid.data.blank_like` (neutral gray L/128 — verified, not the
`(128,0,0)` dark red the old hand-rolled blanks produced on RGB lanes). Every rate carries a
**row-level cluster bootstrap CI**, and each criterion returns PASS / FAIL / **UNDECIDABLE** from the
CI rather than from the point estimate; the final line can be VOID, DEGENERATE-SHORTCUT, AMBIGUOUS or
UNDECIDABLE. The accumulation is factored as `collect(ds, tok, score, greedy, …)` so it runs against
fake scorers on a CPU box — which is how it was tested.

Verified on 8 real materialized rows, CPU-only, no model:

- **The char→token map is right.** `sentinel_char_starts` → `return_offsets_mapping` spans →
  token indices; the selected tokens decode to a string containing `<mask>` in 65/65 cases, the
  spans tile the target exactly, and — the decisive check — the **collator's supervised label ids
  equal `tokenizer(target) + <|im_end|>\n` on 8/8 rows**, so the indices really do address the
  scored positions. A deliberately corrupted `sentinel_char_starts` **or** `sentinel_visible_index`
  is caught by `check_sentinel_columns`, which voids the row instead of averaging it in.
- **The doc's determinism claim holds.** `render_masked_pages(words, mask_a_runs)` is
  **pixel-identical to the stored PNG on 8/8 rows**, and the mask-B render differs on 8/8. The
  permutation control therefore feeds the *rebuilt* page on both sides, so an A-vs-B difference
  cannot be a rendering artefact; the stored-vs-rebuilt check is reported separately and a mismatch
  VOIDs the criterion.
- End-to-end dry run of `collect → verdict → print_report → JSON` on the real rows with a synthetic
  perfect reader (→ CONFIRMS) and a synthetic blind model (→ VOID).

### What this work says the doc gets wrong, or cannot express

1. **The span-boundary criterion has a measured 50% degenerate floor, and the 60% bar sits 10 points
   over it.** Qwen3 tokenizes `<mask>` as **three** tokens (`" <"`, `"mask"`, `">"`). With the doc's
   "sentinel positions **and the tokens immediately after**", one sentinel contributes 4 scored
   tokens of which 2 are near-deterministic continuations. Measured on real rows: **130 of 260 span
   tokens (exactly 50.00%) are continuations**. A fluent model that never decides *where* a mask goes
   still scores ~50%. The script therefore also reports **first-sentinel-token accuracy** (the actual
   localization decision) and the measured floor, and downgrades a combined PASS that does not clear
   the floor to AMBIGUOUS. Quote the first-token number, not the pre-registered combined one.
2. **The pre-registered REFUTES branch is close to unfalsifiable on this lane.** "hallucinated-content
   rate ≈ the unmasked-baseline fabrication rate" assumes a prior that could supply the hidden words.
   On `masked-randstr` an occluded word is 3–7 uniform letters, so a non-reader's chance of emitting
   it is `E_L[26^-L]` = 1.18e-05 **and a reader cannot emit it either** — both hypotheses predict ~0.
   A *high* rate here means the occlusion **leaked ink**, not that the model used a prior. The metric
   only becomes a prior test on the real-text `masked-text` port. The script implements it faithfully,
   labels it a leak detector on this lane, and adds a **donor null** (another row's hidden words
   scored against the same decode) as the empirical chance baseline.
3. **The geometry caveat is computed on the wrong canvas and is ~34% optimistic.** The doc says
   "the word pitch is ~50px against a 48px patch — roughly one word per vision patch, ~2.8
   lines/patch". The 50px pitch is right (measured **50.5 px**; word advance 43.2 px, line pitch
   **17.0 px**, 4–5 lines/page). But the 48px cell lives on the **resized 768px** image, so on the
   original page one soft token covers **64 px** ⇒ **1.27 words and 3.76 text lines per soft token**,
   not 1 and 2.8. Word-level localization sits *past* the patch limit the doc names, not at it.
4. **A stronger version of the same caveat, which the doc does not state at all: 87.5% of the
   soft-token budget is blank paper.** 80 words at font 14 occupy only 4–5 lines ≈ 85 px of a 1024 px
   page. Measured over 8 real pages: **exactly 32 of the 256 soft tokens (12.5%) contain any ink, in
   2 of the 16 grid rows, at 2.50 words per inked cell** — identical on every row. So H14's real
   bandwidth question is not "280 vs 1120 tokens" but "2.5 words crammed into each of 32 cells".
   This is a cheap lever the doc never considers: raising `--font-size` (or lowering `--n-groups`)
   spreads the same words over more cells without touching the soft-token budget, and is a better
   first response to a null than H17's budget knob. It also means an H17-style budget increase would
   help far less here than the doc's secondary risk implies.
5. **`dataset.train_splits` is inert for these runs.** It is read only by `univi.trainer._load_hub`;
   with `source: local` the training splits come from the manifest's `is_training_split` flag and
   `_load_local` never looks at it. The `train_splits:` blocks in `configs/h13_density.yaml` and
   `configs/hybrid_pretrained_randstr.yaml` do nothing. The H14 config omits it and says why. Every
   one of the 36 keys in the new config was mechanically checked against the keys the code actually
   reads: **none ignored**.
6. **`univi/__init__.py` top-level-imports unsloth**, so *any* `univi.*` import needs a GPU — which
   contradicts CLAUDE.md's "modules import CPU-only" claim for the package (it is true only of the
   probe scripts, which defer their imports). CPU verification of anything touching
   `univi.hybrid.data` requires stubbing the package. Not fixed here; flagged.
7. Minor, and to the doc's credit: the tiny 8-row val split reproduces its own N-variance warning —
   floor 4.9625 at N=8 vs the 4.8380 it tells you to quote at N=500, and mask rate 0.1906 ± 0.0240 at
   N=8 vs 0.2228 ± 0.0417 at N=200, both in the 15–30% band.

### Not verified (no GPU was touched)

The model forward, the KV-cached greedy-decode path (it falls back to the uncached full-forward loop
used by `scratchpad/hybrid_4lane_generate.py` if the cache API raises, and says so once), the
wall-clock estimate, and the VRAM headroom. Decode-based metrics cost ~2 greedy decodes of ~210
tokens per row; with the cache that is cheap, without it the fallback re-runs the vision tower every
step and `-n` should be cut hard.
