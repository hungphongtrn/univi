# H17 — 280 soft tokens is the page ceiling; 1120 lifts it

**Status:** DONE (2026-07-30) — **VOID.** All three legs trained and probed; **the discrimination
failed.** Raising the budget did not relieve a capacity limit and it did not leave reading unchanged
— it **destroyed** reading, monotonically, and `grad_norm` collapsed monotonically with it
(0.264 → 0.079 → 0.038). At 1120 the run never reached the no-reading floor. So the optimization
failed *before* the capacity question could be asked: **(A′) is NOT refuted and (C) is NOT
confirmed**, and [H18](../todo/H18-no-learned-scan.md) is **not** promoted by this run. Two of this doc's own
criteria were also found to be mis-specified after the fact (a floored K, a length-confounded Δperm
gate) — see [Verdict](#verdict-2026-07-30). · **Cost:** ~11 h GPU (3 legs) + ~8 min probes ·
**Intended to discriminate (C) fixed scan depth vs (A′) fixed absolute capacity — did not** ·
Promoted by [H13](H13-density-ladder-long-targets.md) · Related:
[H18](../todo/H18-no-learned-scan.md), [H20](../todo/H20-audio-phoneme-resolution.md),
[H21](../done/H21-vision-path-not-scale-invariant.md)

## Verdict (2026-07-30)

**VOID — the ladder measured its own training failure, not the soft-token budget.**

Three legs, trained identically except `max_soft_tokens`, each 500 steps at effective batch 64, each
warm-started from the **same** `data/checkpoints/hybrid-pretrained-randstr-v0/final` (they do **not**
chain — verified in `configs/h17_leg_{280,560,1120}.yaml`, all three carry the identical `init_from`).
Probed with `scratchpad/hybrid_4lane_position_decay.py` → `data/eval/h17-K-h17-d3-{280,560,1120}-v0.json`,
150 rows, `max_length 4096`, `n_truncated_rows: 0` on all three.

| leg | budget | emitted tok | `grad_norm` (end) | Δperm rel | pos-0 gain (pts) | chars 0–10 gain | eval CE vs floor 5.6086 |
|---|---|---|---|---|---|---|---|
| baseline | 280 | 256 | **0.264** | +7.36% | **+64.67 ± 3.92** | **+55.68** | 5.531 (−0.079, below) |
| leg 1 | 560 | 529 | **0.079** | +2.29% | **+68.00 ± 3.82** | **+19.66** | 5.606 (−0.003, below) |
| leg 2 | 1120 | 1089 | **0.038** | +0.006% | **−1.33 ± 0.94 (ns)** | **−0.12** | 5.643 (**+0.034, NOT below**) |

**Why VOID and not a result.**

1. **Both legs fail this doc's own step-400 hard bar.** It required Δperm on d3 to reach **≥ +3.05%**,
   "the reading this same init already achieves". Measured **+2.29%** and **+0.006%**.
   The [sanity gate](#pre-registered-criterion) says a failure at 560 "kills the whole hypothesis
   cheaply" — and 560 is the leg that recovers the source resolution exactly, where the pretrained
   embedder is being asked for almost nothing.
2. **Leg 2 is a training collapse, not a capacity measurement.** `grad_norm` 0.038 is *below* the
   [H13 collapse signature](H13-density-ladder-long-targets.md) (0.061 at step 110, 0.06–0.09
   after), `baseline_gain_significant: false`, `K.status: "no_reading_ns"`, and
   `row_sanity.frac_rows_aligned_gt_blank` 0.353 at `sign_test_p` 0.573 — aligned is *not*
   distinguishable from blank, and if anything blank wins. Most decisively, **eval CE 5.643 sits
   ABOVE the 5.609 no-reading floor** (`below_floor: false`): the model never got as far as a
   non-reader would by predicting the letter marginal. There is nothing here to attribute to 1089
   soft tokens. Clause 2 of [What no branch licenses](#pre-registered-criterion) — "a null at a
   budget the model never trained at licenses nothing at all" — is *meant* for the untrained precursor
   probe, but the reasoning applies with full force to a budget the model was trained at and **failed
   to converge at**.
3. **The failure is monotone in the treatment.** `grad_norm` 0.264 → 0.079 → 0.038 and 0–10-char gain
   +55.7 → +19.7 → −0.1 both track the budget. "More tokens don't help" and "a 280-native init cannot
   be optimized onto a 1089-token 33×33 positional grid in 500 steps" predict *the same data*. This
   design cannot separate them, so it cannot answer the question it was built for.

**What this run does NOT license.**

- **Not "(C) fixed scan depth CONFIRMED · (A′) REFUTED."** That cell in the criterion table is
  explicitly *"conditional on the sanity gate passing and on every leg having trained at its own
  budget"*. Neither condition holds. (A′) survives untested; (C) survives untested.
- **Not the promotion of [H18](../todo/H18-no-learned-scan.md) to the main line**, and not the
  ViT/SigLIP front-end bet, both of which the flat|flat cell would have triggered.
- **Not a verdict on the budget plumbing.** `max_soft_tokens` threading is sound — the probes read the
  budget back off each checkpoint, token counts came out at the predicted 256/529/1089, and no row
  truncated. The plumbing worked; the optimization did not.
- **Do NOT quote the probe JSON's `verdict.final` field.** At 280 and 560 it reads *"(C) SCAN
  FAILURE — the start of the page is read at every density…"*. That is
  `hybrid_4lane_position_decay.py`'s **density-ladder** decision logic (written for H13's five
  simultaneous rungs) firing on a **single** trained rung, where "at every density" is vacuous. The
  1120 file self-reports correctly: *"AMBIGUOUS / VOID … (Check the training run actually converged
  and that the probe's soft-token budget matches it.)"*

**One sub-finding does survive, and it is worth keeping.** At 560, position-0 gain is **+68.0** vs
280's **+64.7** — CIs overlap, statistically indistinguishable — while `pos0_4_gain` collapses
**58.3 → 21.3** and the 0–10-char bin collapses **55.7 → 19.7**. The first glyph is read just as well
at the larger budget; the *continuation* is lost immediately. Whatever the extra tokens cost, they
cost it in place-keeping and not in first-fixation legibility. This is a hint toward (C) but it is a
hint from a leg that failed its own bar, not evidence.

### Two of this doc's criteria were mis-specified — found by running them

Both are the same defect class as [H13 §7](H13-density-ladder-long-targets.md)'s floor bug:
a criterion stated in a statistic that is not comparable across the configurations it is applied to.
Recorded here because they will recur in any successor.

1. **K is structurally floored at 100 chars, so the "≥ 1.73× per leg" bar is blind to leg 1.**
   `K.k_chars` is **100 at 280 and 100 at 560** — the baseline window's own right edge — so
   `k_tokens_from_ratio` is **48.4913 at both**, identical to four decimals, a constant of
   `100 / 2.0622` chars-per-token rather than a measurement. The bootstrap agrees with itself
   (`ci_lo = ci_hi = 100.0`) because it is resolving to a floor, not converging on a value. Meanwhile
   the *actual* reading between those two rungs changed by **2.8×** (0–10-char gain 55.7 → 19.7). **A
   criterion phrased in K would have reported "flat, K factor 1.00×" for a leg that destroyed most of
   the reading.** `K_alt_baseline25` is floored too (25 chars at both). The load-bearing readouts are
   `char_bins` and `pos0_gain`, which are not floored; use those.
2. **The "≥ +100% Δperm" sanity gate measures target length, not grounding.** It compares against
   [H07](H07-pretrained-vision-adapter-qwen.md)'s +114.7%, measured on a **~25-char** target
   lane. These probes run `randstr-d3`: `target_chars_mean` **479**, `target_tokens_mean` **234.3**, of
   which only the first ~5–12 tokens are ever readable. Aggregate Δperm averages the reading over all
   234 tokens, so it is diluted by construction — the exact effect
   [H13 §4](H13-density-ladder-long-targets.md) measured as 14.31% → 0.63% for *identical*
   reading. The **280 baseline reads unmistakably** (+64.7 pts at position 0, `sign_test_p` 1.7e-34)
   and still scores only **+7.36%**, i.e. it "fails" a ≥ +100% gate too. [H15](../done/H15-prior-poisoned-text.md)
   already repaired this for itself by replacing its `Δperm ≥ +30%` bar with a **prefix-matched** Δperm
   VOID guard over the first 48 answer tokens; H17's gate never got the same repair. **Any successor
   must state its Δperm bar prefix-matched, or state it in position-resolved gain.**

### What to do instead

The blocking problem is now upstream of the (A′)/(C) question: **`grad_norm` collapse below ~0.1 with
Δperm ≈ 0 has now voided or degraded four independent runs** — [H13](H13-density-ladder-long-targets.md),
H16, and both H17 legs — and it scales with the budget. Until a run at 560/1120 can be shown to
*optimize*, no budget ladder can be interpreted. Concretely, before re-running any leg:

- **A budget leg needs its own recruitment schedule, not H07's.** A from-scratch `Linear(3840→2048)`
  adapter inherited at 280 is being asked to re-address a 23×23 or 33×33 grid it has never seen;
  500 steps of cosine at `lr_adapter 3e-4` was evidently not it. Longer warmup on the adapter, more
  steps, or a budget *curriculum* (280 → 560 → 1120 chained, which these legs deliberately did not do)
  are the candidates.
- **`grad_norm` is the live gate, not eval CE.** All three legs' eval CE descended smoothly
  (5.636 → 5.531, 5.684 → 5.606, 5.718 → 5.644) while reading went to zero. Any successor should abort
  on `grad_norm < 0.10` sustained, and should probe Δperm at step 200 rather than after 500.
- **The conditional third arm (`--canvas 1584`) is not licensed** — it fires only "if leg 2 is flat
  while leg 1 scales", and leg 1 did not scale.

## Claim

The current 280-soft-token budget physically cannot carry a page of text or phoneme-rate audio.
Raising it to 1120 should move the readable ceiling proportionally.

**Operational form after [H13](H13-density-ladder-long-targets.md):** the model extracts a
**fixed absolute quantity** of content per page, set by the number of soft tokens. Multiply the
tokens and the collapse point **K** multiplies with them.

## This hypothesis' job changed on 2026-07-29

It was written as *"the fix, if the constraint turns out to be (A) bandwidth"*.
[H13](H13-density-ladder-long-targets.md) reported and made it **the discriminator instead**,
which is a stronger role and a different obligation.

H13 measured **K = 48.8 / 48.5 / 48.5 tokens across d2 → d4** — a **16× span of page density** —
under the [H07](H07-pretrained-vision-adapter-qwen.md) checkpoint, and found position-0
reading gain statistically identical at d3 (**+28.7 ± 3.7**) and d4 (**+28.0 ± 3.7**) despite d4
carrying ~24 chars per inked soft token against d3's ~15. So:

- **(A) density-proportional bandwidth — EXCLUDED.** More characters per token does not degrade
  reading at the start of the page.
- **(A′) fixed absolute capacity — alive.**
- **(C) fixed scan depth — alive**, and *indistinguishable from (A′) in H13's data*, because
  chars/token ≈ 2.06 on every rung makes character position and token index proportional by
  construction.

**(A′) and (C) differ in exactly one testable way: raising the soft-token budget relieves a capacity
limit but not a scan limit.** That is this experiment. H13 §8 states it directly — H17 is *"not
undermined by 'more tokens will not help'; it becomes the precise discriminator between (C) and
(A′). Its pre-registered 'K must scale ≥3× for 4× the tokens' is exactly the right test."*

Consequently [H18](../todo/H18-no-learned-scan.md) **cannot be closed** until this reports, and the criterion
below now carries an explicit (C)/(A′) verdict in every cell.

## The numbers

| | current (280) | at 1120 |
|---|---|---|
| fineweb chars / soft token | **10.7** | 2.5 |
| text lines per 48px patch | **3.8** | 1.8 |
| audio ms / token-column | **244** | 120 |

**Both the first and the third row are corrected below.** Row 1 divides by the budget rather than by
*inked* tokens — see the [fourth correction](#fourth-correction-2026-07-28-chars--soft-token-divides-by-the-budget-not-by-inked-tokens).
Row 3 is a statement about the *grid* only: the 1000×160 spectrogram page is **upsampled at every
budget** (1.97× even at 280), so halving ms-per-token-column buys smaller cells over the same
pixels, not finer time resolution — see the
[fifth correction](#fifth-correction-2026-07-28--the-resolution-confound-quantified-and-the-design-fixed).

Reference: the prior-proof lane that reads at +115% runs at **0.1 chars/token** — a ~100× gap.
Phonemes are 50–150 ms; Whisper encodes at 20 ms/token. See
[full-page OCR token budgets](../../literature/full-page-ocr-token-budget.md) and
[spectrogram patch models](../../literature/spectrogram-patch-models-do-not-transcribe.md).

## The headroom is already there

- `univi/hybrid/vision.py:211` — `_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)`.
- The **pretrained** checkpoint's factorized position table is `pos_embedding (1120, 2, 3840)`.
- Running the processor CPU-side at 560/1120 yields 529/1089 tokens for a text page and 531/1079 for
  an audio page, max position id 82 — well inside the table.

**The only gap is plumbing:** `build_pretrained_hybrid` (`univi/hybrid/pretrained.py:225`) and the
eval/probe scripts hard-instantiate the processor with defaults. Add a `model.max_soft_tokens`
config knob.

## Design

**Two legs, not one.** The [resolution confound](#fifth-correction-2026-07-28--the-resolution-confound-quantified-and-the-design-fixed)
below shows that a single 280 → 1120 step bundles two different interventions, and that a null on it
would not say which one failed. The budget is therefore raised at **560 and at 1120**:

| leg | budget | tokens/page | resize of the 1024 render | what changes |
|---|---|---|---|---|
| **baseline** | 280 | 256 | **0.75× down** | — (must be **re-run**, not inherited — see [the baseline problem](#the-baseline-must-be-re-run-too)) |
| **leg 1** | **560** | **529** | 1.08× up — source resolution exactly recovered | 2.07× tokens **+** the lost resolution |
| **leg 2** | **1120** | **1089** | 1.55× up — of the same 1024 source | 2.06× tokens, **optical content identical to leg 1** |

Leg 2 is **optically controlled by construction**: both rungs are upsamples of the same render, so
they carry the same optical information and differ only in token count. Whatever K does across leg 2
is attributable to the number of soft tokens and to nothing else.

### The training design is REPAIRED (2026-07-29) — "rerun the H13 ladder at 1120" would reproduce H13's collapse

This doc previously said *"the ladder is rerun at 560 and at 1120"*, i.e. run
`configs/h17_density_{560,1120}.yaml`, which are `configs/h13_density.yaml` with `max_soft_tokens`
changed. **Those configs encode the design that killed H13 and must not be launched as written.**

[H13](H13-density-ladder-long-targets.md) §3: rows were balanced 20k/rung but loss is
normalised **per supervised token**, so gradient share follows target length —
`d1 1.13% · d2 4.11% · d3 15.86% · d4 63.04% · d5 15.86%` — and **85.8% of supervised tokens lay
beyond the model's ≈48-token scan depth**, where the single loss-minimising behaviour is to emit the
uniform-letter marginal. The from-scratch adapter is the only component that starts at zero, so it
had to *bootstrap* reading from the 14% of tokens where reading is achievable against an 86% majority
teaching the opposite. It lost by step 110 (`grad_norm` 0.061, and 0.06–0.09 for the remaining 890
steps); the final ablation reads **Δperm ≈ 0 on all five rungs**. **Balancing rows does not balance
gradient.** Raising the budget does nothing about this: at 1120 the *same* 85.8% of tokens are still
past the depth, so a rerun buys a second VOID at 2–4× the wall-clock.

Four changes, each with its trade-off recorded.

#### 1. Warm-start from a checkpoint that already reads (`init_from`)

`model.init_from: data/checkpoints/hybrid-pretrained-randstr-v0/final` — the
[H07](H07-pretrained-vision-adapter-qwen.md) checkpoint, which reads at **Δperm +114.7%** on
randstr and which H13 §4 independently re-measured at **+113.84%** on this very ladder's d1 rung.
The seam is `univi/hybrid/train_pretrained.py` (added for exactly this reason by H13 §8): it loads
**weights only** — fresh optimizer, fresh LR schedule, so this is warm-*start*, not
`resume_from_checkpoint` — and refuses any missing/unexpected tensor except the tied
`language_model.lm_head.weight`, which Qwen3 never serializes.

- **Why:** it removes the bootstrap lottery. H13's failure was not "the budget is wrong", it was "the
  adapter never got off zero"; a run that has to re-answer that question cannot answer this one.
- **Cost 1 — it does not make the budget change free.** H07 trained at **280**. The adapter is
  `nn.Linear(3840 → 2048)` applied *per soft token* (`univi/hybrid/pretrained.py:118`), so it is
  budget-agnostic in shape; but the vision tower's factorized position table is indexed by
  `image_position_ids`, whose max on a square page goes **15 → 22 → 32** at 280 / 560 / 1120. Those
  rows exist (the table is `(1120, 2, 3840)`) but H07 never trained them. The warm-start therefore
  begins each leg slightly out of distribution, and *more* so at 1120 than at 560. This biases the
  design **against** the (A′) branch, which is the safe direction: a K rise measured despite the
  handicap is real.
- **Cost 2 — it locks in a 280-shaped solution.** If K stays flat, one cannot fully exclude "the
  warm-start had already committed to reading 280-token pages". The mitigations are (a) the
  **baseline leg is warm-started identically** (below), so the comparison is paired, and (b) 500
  steps at effective batch 64 is 1.6× H07's own exposure, i.e. ample room to move off the
  initialization.

#### 2. Train on ONE density, not the five-rung ladder

Train and measure on **`randstr-d3` only** (479 chars/page, **~234 target tokens**, K ≈ 48.5 under
H07). The ladder existed to vary *density*; H13 already answered that question and carrying the
ladder into H17 imports its failure mode for **zero** information gain.

d3 is not an arbitrary pick — it is the only rung on which the pre-registered criterion is both
**measurable** and **survivable**:

| rung | tgt tok | readable within depth ≈48 | can K rise 3× (to ~146 tok) be *observed*? |
|---|---|---|---|
| d1 | 17 | 100% | no — K censored by target end (H13 §5.5) |
| d2 | 60 | 80% | no — 3× K exceeds the target |
| **d3** | **234** | **20.5%** | **yes — 146 tok is 62% of the target** |
| d4 | 932 | 5.2% | yes, but this is the rung that supplied 63% of H13's dead gradient |

d2 and d4 are still **probed** at every budget (zero training cost, same checkpoints) as a density
cross-check against H13's K = 48.8 / 48.5 / 48.5. They are not trained on.

#### 3. Gradient share needs no explicit rebalancing — but only because the lane is single-length

Every d3 row carries the same ~234 supervised tokens, so **token-weighted share is uniform across
rows by construction** and H13's lesson is satisfied trivially. Nothing has to be reweighted, and
nothing *should* be: a weighting scheme is a second changed variable.

**The residual risk is real and is inside the row, not between rows.** 79.5% of d3's supervised
tokens still lie past depth ≈48 and still teach the uniform-letter marginal — H13's failure regime
was 14.2% readable, H07's success was 100%, and d3-only sits at 20.5%, closer to the former. The
warm-start is what is expected to carry it; that expectation is **pre-registered as a kill gate**,
not assumed:

> **Kill gate (two-stage).** Ablate on `randstr-d3/validation`. H07 reached Δperm **+27.4% by step
> 200** from a cold adapter at its native budget.
>
> - **Step 200 — progress check.** Δperm on d3 must be **> +1.0%** and above the same run's step-100
>   value (reading is being established, not destroyed — the
>   [H05](H05-isolated-ocr-from-scratch.md) mode).
> - **Step 400 — hard bar.** Δperm on d3 must reach **≥ +3.05%**, the reading this same init already
>   achieves at its native 280 budget (measured, see below). A leg that cannot recover, within 400
>   steps, the reading the architecture demonstrably has at 280 cannot yield a measurable K, and
>   continuing spends GPU on an unanswerable question.
>
> **Why the gate is absolute rather than relative to step 0 — a defect found and fixed 2026-07-29.**
> This gate originally read *"kill if step-200 Δperm is below the warm-start's own step-0 Δperm on
> d3."* The step-0 value has since been **measured at all three budgets** (precursor below) and it is
> not budget-independent — it is **+3.05% at 280, +0.03% at 560, +0.01% at 1120**. Anchoring to it
> would have set the 560/1120 thresholds at ~zero, making the gate **vacuous at exactly the two legs
> it exists to protect**, while leaving it strict at the 280 baseline. An absolute bar is the same
> bar for every leg, which is what the K-vs-budget comparison requires.
>
> **Pre-registered fallback if the gate fires:** add a **20% d1 anchor** (17-token, 100%-readable
> rows) to keep the adapter alive. This changes the lane, so the fallback run is *not* comparable to
> the primary — but K-vs-budget stays clean **provided all three budgets use the identical lane and
> identical init**. That is the invariant of this whole experiment; violating it makes every number
> below uninterpretable.

#### 3b. Precursor RAN (2026-07-29) — the zero-training budget probe is a NULL, and it is fully explained by OOD

The cheap precursor described below (probe the H07 checkpoint at 560/1120 without retraining, since
the adapter is `Linear(3840→2048)` applied per-token and therefore budget-agnostic) **has been run**:
`scratchpad/h17_budget_probe.sh` → `data/eval/h17-budget-probe-{560,1120}.json`, 150 rows/rung,
`--max-length 4096`.

| budget | d1 Δperm | d1 CE | d1 Δblank | d3 Δperm | K anywhere |
|---|---|---|---|---|---|
| **280** (native) | **+113.84%** | 3.722 | +90.65% | +3.05% | 48.8 / 48.5 / 48.5 tok |
| 560 | +0.46% | 7.203 | **−11.48%** | +0.03% | none — `no_reading` on every rung |
| 1120 | +0.13% | 6.713 | **−13.13%** | +0.01% | none — `no_reading` on every rung |

**Reading collapses completely at both raised budgets, and Δblank goes negative** — a blank page
*beats* the real one, i.e. the image has become an active distractor rather than merely uninformative.
Mechanism: 560 emits a 23×23 = 529-token grid and 1120 a 33×33, so `image_position_id` runs to 22 and
32 where H07 only ever saw **0–15**. Those position embeddings were never exercised.

**What this licenses.** Per the asymmetry pre-registered below, a null here licenses **nothing** about
capacity, and is **not** a kill signal for H17. What it does establish, now by measurement rather than
by argument, is §4: **the budget cannot be probed at inference; it must be a training variable.** The
OOD handicap is not the "slight" one assumed when the warm-start was proposed — at step 0 a raised-
budget leg starts *worse than blind*. That makes the warm-start's benefit smaller than hoped (the
decoder and vision weights still transfer; the reading behaviour does not) and it is why the step-400
hard bar above exists.

**Cost note:** ~25 min for both probes. Cheap, and it converted two hypotheticals — "step-0 Δperm is
budget-dependent" and "OOD could confound a null" — into measured facts before ~26 h of training was
committed against them.

#### 3c. Decision: the 1.6-epoch exposure is ACCEPTED, not fixed (2026-07-29)

The configs run 500 × 64 = 32,000 rows against d3's existing 20,000-row train split = **1.6 epochs**,
where this doc originally asked for d3 to be re-materialized at 40k rows to stay sub-epoch. **Not
re-materializing, deliberately:**

- Re-drawing d3 changes the **validation** split as well as train (one RNG stream), and the existing
  val split is the one on which **K = 48.5 was measured** for the H07 reference. This doc requires
  the 280 leg's K to land within **±20% of 48.5** for the series to be clean. A changed val set makes
  that check indirect for no gain.
- The confound re-materializing would remove is weak and *directionally harmless*: exposure is
  **identical across all three legs**, and the pre-registered criterion is a **ratio** (K must scale
  ≥3× for 4× the tokens). Equal inflation of all three K values cancels in the ratio.

**What this does cost.** Absolute K values may be mildly inflated by ~1.6× page re-exposure, so the
±20%-of-48.5 baseline check is the looser of the two tests and should be read as a sanity gate rather
than a precise replication. The **trend** is what the hypothesis turns on and it is unaffected.
If a future run needs absolute K, re-materialize at 40k and re-run **all three** legs together.

#### 4. The budget must be a TRAINING variable, not an inference-time one

This is the constraint that fixes the run count. A model that has never trained at 1120 has never
seen a 33×33 soft-token grid or position ids above 15, so **a null at a budget it never trained at is
confounded with out-of-distribution-ness** and licenses nothing about capacity. Every leg therefore
trains at its own budget.

##### The baseline must be re-run too

H13 §4's K = 48.5 was measured on the H07 checkpoint, which trained on **d1 geometry only** at 280.
If leg 1 were compared against *that*, it would differ in two ways at once — budget **and** having
trained on d3. So the 280 rung is a **third training run**, identical to legs 1 and 2 in lane, init,
recipe and step count. It is the honest cost of making leg 1 interpretable, and it doubles as a
replication of H13 §4's K on a model that *did* train on d3.

##### The cheap precursor — and the asymmetry that makes it worth running anyway

Because the adapter is per-token `Linear(3840 → 2048)` and the whole vision path is per-patch
(`univi/hybrid/vision.py` — LN → Linear → LN → +factorized position → LN → RMSNorm → Linear, **no
cross-patch mixing anywhere**), the H07 checkpoint **can be probed at 560 and 1120 with no retraining
at all**. ~1 h GPU, no training, three budgets on `randstr-d{2,3,4}/validation`.

**The two outcomes are not symmetric, and this must not be forgotten when reading it:**

| precursor result | what it licenses |
|---|---|
| **K scales with budget** | **Strong early evidence for (A′).** OOD-ness can only *hurt*; a K rise measured *despite* an untrained grid geometry is a lower bound on the trained effect. Worth acting on immediately — it would justify going straight to the full three-leg sequence. |
| **K flat** | **Nothing.** Fully confounded with the model never having trained at that budget. It is *not* a kill signal, and it must not be reported as one. Proceed to the training legs. |

Run it first anyway: it costs ~1 h, it can only shorten the path, and it exercises the whole probe
stack at 560/1120 before ~26 h of GPU is committed to it.

#### Configs

`configs/h17_randstr_560.yaml` and `configs/h17_randstr_1120.yaml` (the sanity gates) are **written,
correct, and unchanged by this repair** — they train the single-density, 100%-readable H07 lane,
which is not the failure regime.

`configs/h17_density_560.yaml` and `configs/h17_density_1120.yaml` are the **collapsing five-rung
design and are superseded.** They must be replaced by `configs/h17_d3_{280,560,1120}.yaml` — d3-only,
`init_from: data/checkpoints/hybrid-pretrained-randstr-v0/final`, `max_soft_tokens` the only
difference between the three — before anything is launched. *(Not written at the time of this doc
edit; this is a docs-only change.)*

**Run order, each SOLO:** precursor probe → gate 560 → **d3 @ 280** → **d3 @ 560** → gate 1120 →
**d3 @ 1120**. The 560 gate is first among the training runs because it is the cheapest kill — 560 is
a 1.08× resize, very nearly the identity, so if Δperm does not survive *there* the embedder is
hard-tied to 280 and no leg should be launched.

**Data:** at effective batch 64, 500 steps sees 32,000 rows against `randstr-d3`'s existing 20k train
split — **1.6 epochs**. d3 targets are 479 uniform random letters, so memorization is a weak confound
rather than a fatal one, but it is avoidable: re-materialize d3 at **40k train rows** (~0.4 h CPU at
this doc's measured 0.19 h per 20.5k rows) and keep the run under one epoch, as
[H14](H14-masked-region-targets.md) does for the same reason. The 500-row validation split is
unchanged and is what every probe reads.

#### Cost, staged

**Every wall-clock figure below is an estimate scaled from H13's measured 31.1 s/step** (per-device
4 × accum 16, mean sequence ~577 tokens) — no GPU was available while this design was written. The
scaling is **linear in sequence length, which is a lower bound**: attention is quadratic, and
gradient checkpointing changes the constant. Treat the 1120 leg in particular as optimistic.

| stage | budget | seq/row (256/529/1089 img + 234 tgt + 32 scaffold) | est. s/step | est. h @ 500 steps |
|---|---|---|---|---|
| precursor probe (no training) | 280/560/1120 | — | — | **~1 h total** |
| gate — `h17_randstr_560` | 560 | ~575 | — | ~2 h @ 600 steps |
| **baseline — d3 @ 280** | 280 | **522** | ~28 | **~3.9 h** |
| **leg 1 — d3 @ 560** | 560 | **795** | ~43 | **~6.0 h** |
| gate — `h17_randstr_1120` | 1120 | ~1135 | — | ~4 h @ 600 steps |
| **leg 2 — d3 @ 1120** | 1120 | **1355** | ~73 | **~10.2 h** |

Kill at leg 1 ≈ **12 h** (precursor + gate 560 + baseline + leg 1). Full sequence ≈ **26 h**. That is
~3× the original "~8–10 h", which costed only a single 1120 run. The extra spend buys the difference
between an interpretable and an uninterpretable refutation — and the refute branch is the one that
promotes an architectural bet, so it is exactly the branch that must not be ambiguous.

**Two bookkeeping changes forced by the budget, both verified CPU-side:**

- `max_length` 2048 → **4096 on both legs**. This is not cosmetic. `_filter_training_tokens`
  estimates a row as `original_token_length + n_images * image_token_budget(max_soft_tokens) + 256`,
  and `image_token_budget(1120) = 1122`, so a median d4 row is `929 + 1122 + 256 = 2307 > 2048`:
  at `max_length: 2048` the filter **drops all 20,000 d4 train rows** — the densest rung, i.e.
  exactly the one K is measured on — while d1–d3/d5 are untouched. Measured on the real splits: at
  4096 every rung retains 20000/20000 at all three budgets, so both legs train and evaluate on
  identical data. It costs nothing (`HybridCollator` pads to the batch max, not to `max_length`).
  **Keep `max_length: 4096` after the repair, even though the trained lane is now d3-only.** The
  repair moves *where the trap bites*, it does not remove it. d3 at 1120 estimates
  `234 + 1122 + 256 = 1612 < 2048` and would survive — but (a) the moment d4 is reintroduced to
  training the filter silently deletes it again, and (b) **the probes have the same default and are
  the more likely victim now**: `scratchpad/h13_analyze.py`, `hybrid_4lane_position_decay.py` and
  `hybrid_pretrained_ablation.py` all default to `--max-length 2048`, and a d4 probe row at 1120 is
  `1089 + 932 + 32 ≈ 2053` — over the line. **Pass `--max-length 4096` to every probe at 560/1120.**
  A silently dropped or truncated d4 rung produces a K trend that reads like a result.
- Per-device batch 4 → 4 (leg 1) and 4 → **2** (leg 2), accum 16 → 32, holding the **effective batch
  at 64** (H07 exposure — the invariant; per-device size is free and must be confirmed against the
  26 GB ceiling over the first ~20 steps).

**Rejected: raising the render canvas** (see the [confound section](#fifth-correction-2026-07-28--the-resolution-confound-quantified-and-the-design-fixed)
for the numbers). Rejected on *design* grounds, not cost — re-rendering is cheap (measured: 0.19 h
CPU for a 20.5k-row d3 rung at canvas 1024, 0.32 h at 1584; the whole ladder ≈ 2 h CPU, offline and
non-blocking). It is rejected because it would (a) *re-confound* leg 2, which exists precisely to
hold optics fixed, (b) shift chars/page by +5–8% (integer font metrics do not scale exactly:
6612 → 6962 → 7140 at canvas 1024/1536/2048), (c) break comparability with H13's own K values, which
are the 280-token baseline the whole ladder is measured against, and (d) invalidate the sanity gate,
which replicates H07's +115% on H07's exact pixels. A canvas raise is kept as a **conditional third
arm** of one rung — see the criterion.

Zero-code complement: **re-render at font 28–52** to cut lines-per-patch, and split documents across
more images (rows currently average 1.43; `max_train_images: 4` allows more). **Re-scoped 2026-07-29
by [H13](H13-density-ladder-long-targets.md) §6:** its d5 rung (font 40, byte-identical
targets to d3) is not a cheap preview of this intervention after all — under the H07 checkpoint d5
reads **nothing** (pos-0 **+0.0**, Δperm +0.01%) where d3 reads (pos-0 **+28.7**, Δperm +3.05%). But
H07 never *trained* at font 40, so that is a **scale-generalisation failure, not a bandwidth
result**, and the "D5 ≫ D3 ⇒ line-demux" branch is **untested, not refuted**. Whether the free lever
works at all when it is trained is now [H21](../done/H21-vision-path-not-scale-invariant.md)'s question, and
H21 is the cheaper experiment (~3 h). See also
[the cheaper lever](#the-cheaper-lever-already-has-a-partial-result-in-flight), whose "partial result
in flight" framing is superseded by this paragraph.

## Pre-registered criterion

**K** is [H13](H13-density-ladder-long-targets.md)'s: the character position beyond which
positional reading gain drops below 50% of the 0–100-char gain, measured per rung with
`scratchpad/hybrid_4lane_position_decay.py` / `scratchpad/h13_analyze.py` — **not** aggregate eval CE.
(H13 §7 is the reason that qualifier is in bold: its `floor.json` computed the no-reading floor over
*target* tokens while response-only masking also supervises the deterministic `<|im_end|>\n` trailer,
so every published floor was too high by ~`T/(T−2)` and every `*`-marked "below floor ⇒ reading" in
H13's trajectory table was an artifact. The ablation-based §4 verdict was untouched. **Never state a
criterion in eval CE.**)

**Measured on the same rung at three budgets**, all trained identically except for
`max_soft_tokens` — 280 baseline, 560, 1120. Each leg is a ~2.07× / 2.06× token increase, so the
original "≥ 3× for 4× the tokens" is split geometrically: **each leg must give K a factor ≥ 1.73×**
(√3 = 1.732), which compounds to ≥ 3× across the full 4.25× range. (Correction 3's note stands: the
real increase in *emitted* tokens is 4.25×, and against *inked* tokens 3.82×.)

The 280 baseline's own K must additionally land within **±20%** of H13 §4's **48.5 tokens** on d3.
If it does not, d3-training itself moved K and the three legs are no longer a clean budget series —
report that rather than the K trend.

| leg 1 (280→560) | leg 2 (560→1120) | verdict |
|---|---|---|
| K ≥ 1.73× | K ≥ 1.73× | **(A′) fixed absolute capacity CONFIRMED · (C) fixed scan depth REFUTED.** The model extracts a fixed quantity per page and the quantity is set by the token count — leg 2 holds optics fixed, so it is the tokens and not the pixels. [H18](../todo/H18-no-learned-scan.md) drops as the primary account. Scale the budget. |
| K ≥ 1.73× | flat | **Bounded, not decisive.** The gain came from the 280→560 step and saturates at native resolution. Two sub-readings this design cannot separate — the recovered resolution did it, or tokens help to ~2× then saturate — but the resolution component is bounded at **< 1 pp of single-glyph legibility** (measured below), which makes the second reading much more likely. Reads as *partial* capacity relief with (C) still binding above 560. Does **not** license the ViT bet. Next step is the conditional third arm. |
| flat | K ≥ 1.73× | Non-monotone; treat as a bug or noise and re-run before interpreting. |
| flat | flat | **(C) fixed scan depth CONFIRMED · (A′) REFUTED** — *conditional on the sanity gate passing and on every leg having trained at its own budget.* 4.25× the tokens, with the lost resolution restored at the halfway rung and leg 2 optically controlled, moved K by < 3×. [H18](../todo/H18-no-learned-scan.md) then becomes the main line and the fix is architectural or curricular, not more tokens. A different pretrained front-end (a real ViT/SigLIP encoder) stays on the table but for a **changed reason**: not "more capacity" but "cross-patch attention supplies the spatial integration `univi/hybrid/vision.py` structurally cannot" — a (C)-motivated bet, and it should be weighed against H18's cheaper interventions (chunked crop curriculum, localization supervision via [H14](H14-masked-region-targets.md)) first. |

**Sanity gate — both rungs, 560 first.** `configs/h17_randstr_560.yaml` then
`configs/h17_randstr_1120.yaml` must each replicate **≥ +100% Δperm** on the prior-proof randstr lane
(H07 measured +114.7%). A failure at 560 kills the whole hypothesis cheaply: at a 1.08× resize the
pretrained embedder is being asked for almost nothing. A failure at 1120 only kills leg 2.
*Blank-token caveat, registered before the fact:* the 25-letter gate page inks ~5/256 cells at 280
and ~18/1089 at 1120, so a Δperm drop there could be dilution by blank tokens rather than a
budget-specific embedder failure. Both gate configs therefore carry a **zero-training-cost secondary
read-out** — ablate the same checkpoint on `randstr-d3/validation`, which inks 9.5% of the grid at
1120 — and the gate is only failed if *both* pages fail.

**Conditional third arm (only if leg 2 is flat while leg 1 scales).** Re-materialize **one** rung —
d3 at `--canvas 1584 --font-size 22`, ≈ 0.3 h CPU — and rerun leg 2 on it. Canvas 1584 is the
smallest square canvas at which budget 1120 stops interpolating (its resize target is exactly 1584),
and it roughly doubles the real optical samples per character cell (143 → 316 px). If K moves there
and not on the 1024 canvas, the extra cells needed *real pixels*, and the render canvas — not the
architecture — is the lever. This arm is **not** part of the main ladder and its K is not comparable
to H13's.

**What no branch licenses.**

1. **A null at 1120 *without* the 560 rung licenses nothing about the architecture** — that is the
   confound this design exists to remove, and it was the state of the hypothesis before 2026-07-28.
2. **A null at a budget the model never trained at licenses nothing at all** (added 2026-07-29). The
   cheap precursor probe of the H07 checkpoint at 560/1120 is *asymmetric*: a K rise is strong early
   evidence for (A′); a flat K is confounded with out-of-distribution grid geometry and position ids
   the checkpoint never saw. It is not a kill signal and must not be reported as one.
3. **The (A′)/(C) separation is the best available, not a clean one** (added 2026-07-29). Raising the
   budget does two things at once on this architecture: it adds capacity *and* it refines the
   **spatial addressing grid** from 16×16 to 33×33 — and a finer addressing grid is plausibly some
   relief for a scan limit too, which is exactly what (C) says is missing. A pure scan failure
   predicts no benefit and a pure capacity limit predicts proportional benefit, so the extremes stay
   diagnostic; the genuinely ambiguous zone is a *partial* rise (between 1× and 1.73× per leg), and
   the 1.73× bar exists to stop that zone being read as confirmation. Report a partial rise as
   partial, never as (A′).
4. **This re-tests nothing about density.** [H13](H13-density-ladder-long-targets.md) settled
   that: K is invariant over a 16× density span. The d2/d4 probe rungs here are a replication check,
   not a new density arm.
5. Nothing here measures whether *blank* soft tokens are harmful, neutral or a useful positional
   scaffold. The legibility bound below assumes perfect glyph segmentation, which the model does not
   get. And **none of it transfers to audio**: measured, the 1000×160 spectrogram page
   is *upsampled at every budget* (1.97× even at 280), so for audio the entire ladder is grid
   subdivision with zero optical gain — [H20](../todo/H20-audio-phoneme-resolution.md) must derive its own
   criterion rather than reuse this one. (**Independently replicated 2026-07-28** on real materialized
   pages by a different code path — 1.968× on the time axis, 1.800× on the frequency axis, with a flat
   round-trip residual across all three budgets — and H20's criterion has now been re-derived; see its
   [render-geometry measurement](../todo/H20-audio-phoneme-resolution.md#render-geometry-measured-2026-07-28--the-budget-is-the-weaker-of-two-levers).)
6. **Nothing here tests glyph scale.** Font size is a separate lever and H13 §6 shows it fails
   untrained; that is [H21](../done/H21-vision-path-not-scale-invariant.md), not this.

## Prep status (2026-07-28) — plumbing DONE and verified, nothing run

`model.max_soft_tokens` is threaded end to end and defaults to 280 everywhere:
`vision.py` (`DEFAULT_MAX_SOFT_TOKENS`), `pretrained.py` (config field, serialized so a checkpoint
trained at 1120 reloads at 1120, plus `resolve_max_soft_tokens()` / `build_image_processor()`),
`train_pretrained.py`, and all 5 probe scripts (`--max-soft-tokens`, defaulting to the *checkpoint's*
recorded budget — a probe silently running at a different budget than training would produce garbage
that reads like a result). `configs/h17_randstr_1120.yaml` is written but not launched.

Verified CPU-only: the 280 path is **byte-identical** (`torch.equal` on `pixel_values` and
`image_position_ids`); a legacy checkpoint without the key resolves to 280; round-trip
save/reload at 1120 holds; `tests/test_hybrid.py` green.

**Every token-count claim in this doc reproduced exactly:**

| page | 280 | 560 | 1120 | max pos id (280/560/1120) |
|---|---|---|---|---|
| text 1024×1024 | 256 | **529** | **1089** | 15 / 22 / 32 |
| audio 1000×160 | 246 | **531** | **1079** | 40 / 58 / **82** |

### Three corrections to this doc, found by measurement

1. **`max_length: 4096` is marginal, but the original framing cited the wrong split.** 4 × 1079 =
   **4316 > 4096** is arithmetically right, but `{1: 2136, 2: 506, 3: 52, 4: 9}` is librispeech
   **validation**. **Train never reaches 4 pages**: `{1: 20548, 2: 83449, 3: 17}` over 104,014 rows.
   So 4096 *is* sufficient for training and bites only **9 validation rows**. Prefer
   `max_length: 8192` (it costs nothing — `HybridCollator` pads to the batch max, not `max_length`)
   over `max_train_images: 3`, which would *drop* those 9 eval rows rather than keep them. The
   randstr sanity gate is unaffected either way (single image, seq 1337).
2. ~~**`univi/trainer._IMAGE_TOKEN_BUDGET = 282` is hard-coded**~~ — **FIXED 2026-07-28.** It drove
   `_filter_training_tokens` in every dataset path; at 1120 it under-counted per-image cost ~3.9×, so
   overlong multi-image rows passed the filter, the collator truncated their placeholders, and
   `forward` raised `Image placeholder count != projected soft tokens`. Now
   `univi.trainer.image_token_budget(max_soft_tokens)` (= `max_soft_tokens + 2`) is derived from
   `model.max_soft_tokens` and threaded through `load_dataset` → `_load_local`/`_load_hub` and
   `_load_eval_datasets` → `_filter_training_tokens(..., per_image_tokens=…)`.
   Verified: `image_token_budget(280) == 282 == ` the legacy constant, and on 2000 d4 rows at
   `max_length 2048` the default and an explicit 282 both retain **2000/2000** while a 1122 budget
   correctly rejects all of them (931 + 1122 + 256 = 2309 > 2048) — exactly the rows that would
   otherwise have crashed mid-run. Tests: 33 + 8 passed.
3. **280 is the *padded* budget, not tokens used** — a text page really uses 256 and audio 246. So
   1120 is a **4.25×** (text) / 4.39× (audio) real increase, not 4×. The chars/soft-token table
   above divides by the budget rather than the used count; the direction of the argument is
   unchanged, but the "≥ 3× for 4× the tokens" criterion should be read against 4.25×.

### Interpretation of the sanity gate, fixed before it runs

"**frozen-vision** randstr at 1120" is ambiguous: the +115% reference
([H07](H07-pretrained-vision-adapter-qwen.md)) ran the 3-group LR with **no** freeze, and
"frozen" in the notes describes the *observed* ≈0 rel-delta of the vision tower, not
`freeze_vision: true`. The config therefore uses **H07's recipe verbatim so `max_soft_tokens` is the
only changed variable** — which is what a replication gate requires. If the literal frozen recipe
were used instead, the ≥ +100% bar would be compared against a recipe that was never measured.

## ~~Do not run this before H13~~ — H13 has reported (2026-07-29)

The original text read: *"If [H13] shows the collapse is fixed at a token index rather than a
density, this experiment is predicted to do nothing, and its cost is better spent on
[H18](../todo/H18-no-learned-scan.md)."*

**That gate is now resolved, and it resolved the other way round from how it was written.** H13 found
K fixed at token ~48.5 across a 16× density span — which under the old framing would have said
"predicted to do nothing". H13 §5.1 and §8 explicitly reject that inference: a K constant across
rungs is *equally* consistent with (C) fixed scan depth and with (A′) fixed absolute capacity,
because chars/token ≈ 2.06 on every rung makes character position and token index proportional by
construction. H13's finding kills only **density-proportional** bandwidth.

So this experiment is **not** predicted to do nothing — it is the only experiment that separates the
two survivors, and [H18](../todo/H18-no-learned-scan.md) is the thing that now waits on *it*, rather than the
reverse. See [H18's conditional status](../todo/H18-no-learned-scan.md#status-conditional-on-h17-2026-07-29).

## Fourth correction (2026-07-28): "chars / soft token" divides by the budget, not by INKED tokens

Measured CPU-only by `scratchpad/h13_ink_occupancy.py` →
`data/eval/h13-ink-occupancy{,-560,-1120}.json`; full method, self-test and caveats in
[H13's ink-occupancy observation](H13-density-ladder-long-targets.md). No GPU, no
weights (H13 was training; this repo's rule is that training runs SOLO).

Correction 3 above already noted that 280 is the padded budget and a text page really uses 256.
There is a second, larger denominator error underneath it: **most of those 256 tokens are blank
paper.** A 1024² page resizes to 768² and yields a 16×16 grid of 48px cells (one cell = **64
original page pixels**); on a fineweb-edu page only about half of them contain any ink, and on the
prior-proof randstr lane only 2%.

### The table's row, with each denominator substituted (fineweb-edu validation, page 0, n=30)

| denominator | value | what it is |
|---|---|---|
| ÷ 280 (the budget) — **the row above** | **10.55** | reproduces this doc's 10.7 |
| ÷ 256 (tokens actually emitted) | 11.54 | correction 3's version |
| **÷ tokens containing ink (mean 148.5)** | **19.16** | the quantity that could plausibly bind |

Inked fraction 58.0% (sd 26%, range 15.6%–99.6% over rows — short documents fill less of the page,
so this mean hides a large spread and should not be quoted as a per-page constant).
**Estimator note:** every "chars / inked" figure here is the **mean of the per-row ratio**, not
(mean chars)/(mean inked). The two coincide when the inked count is constant across rows — it is, on
the randstr rungs — but not on fineweb, where inked varies 40–255 and the ratio-of-means would read
19.89 instead of 19.16. The per-row mean is the honest one; both estimators are used consistently
throughout, so every ratio-of-ratios below is unaffected.
Measured lines per 48px patch is **3.76**, i.e. this doc's 3.8 was already right.

### The bigger correction is the reference line, not the table

> "Reference: the prior-proof lane that reads at +115% runs at **0.1 chars/token** — a ~100× gap."

That lane is H13's d1 geometry (25 letters, font 14). Measured on the same footing it runs at
**5.80 characters per inked soft token**, against fineweb's 19.16 — a **3.3× gap, not ~100×**. The
100× is an artifact of dividing a nearly-empty page (5 inked cells of 256) by the full budget. The
lane that demonstrably reads is *not* operating two orders of magnitude below fineweb in the
corrected quantity; it is operating a factor of three below it.

### Would raising the budget 4× even change chars-per-INKED-token? Yes — measured

The obvious worry about this correction is that it might dissolve the intervention: if the extra
tokens all landed on blank paper, chars-per-inked-token would not move. **It does move, essentially
proportionally**, because the inked *fraction* is roughly budget-invariant — a finer grid resolves
the blank gutters between text lines slightly better, so the fraction drifts down a little, but the
extra cells land on ink in about the same proportion.

| budget | grid | tokens | cell = orig px | fineweb inked | inked frac | fineweb ch/inked | d3 ch/inked | d4 ch/inked | lines/cell |
|---|---|---|---|---|---|---|---|---|---|
| 280 | 16×16 | 256 | 64.0 | 137.2 | 53.6% | **19.1** | 14.97 | 23.99 | 3.76 |
| 560 | 23×23 | 529 | 44.5 | 269.7 | 51.0% | **9.84** | 9.78 | 12.46 | 2.62 |
| 1120 | 33×33 | 1089 | 31.0 | 523.5 | 48.1% | **5.14** | 4.65 | 6.00 | 1.83 |

(fineweb rows are the same 20 documents at all three budgets; the randstr rungs are deterministic.)

So 280 → 1120 buys **3.82×** more inked tokens on fineweb and cuts chars-per-inked-token **3.73×** —
tracking the 4.25× rise in emitted tokens closely. **The correction changes the level of the
quantity, not its scaling, and the pre-registered criterion survives unchanged:** "K must scale ≥ 3×
for 4× the tokens" is still the right shape of test, and correction 3's note to read it against
4.25× stands (against inked tokens the real factor is 3.82×). Do **not** read this correction as
"raising the budget would not help" — it moves the corrected quantity by very nearly the factor it
moves the uncorrected one.

### But the optical benefit is exhausted well before 1120

The render is a **1024×1024** PNG. The processor's aspect-preserving resize targets 768 at budget
280, 1104 at 560 and 1584 at 1120. So:

- at **280 the page is DOWNSAMPLED 0.75×** — real glyph detail is destroyed before the model sees it
  (**"destroyed" is too strong — measured and softened in the [fifth correction](#fifth-correction-2026-07-28--the-resolution-confound-quantified-and-the-design-fixed)**);
- at **560 (1104px) the source resolution is already fully recovered**;
- at **1120 (1584px) the image is upsampled 1.55×** — those extra tokens subdivide *interpolated*
  pixels and carry no optical information the 1024 render did not already hold.

Honest reading: raising the budget is two interventions bundled together — recovering lost
resolution (real, and complete by 560) and subdividing the page into more, smaller units (continues
past 560, but on interpolated pixels). This does **not** show 1120 is useless: a finer grid may help
purely by giving the decoder more and smaller units to attend over, and that is **not measured
here**. It does mean that **if the goal is more legible glyphs per token, the render canvas must be
raised above 1024 alongside the budget** (free, offline) — otherwise the 2× sequence cost of 1120
over 560 buys grid units, not sharpness.

### The cheaper lever already has a partial result in flight

Font size moves the same quantity harder and at zero token cost. H13's **d5** (font 40 vs d3's 14,
same canvas, same 280 budget, byte-identical targets) runs at **3.23 chars per inked token vs d3's
14.97 — a 4.64× improvement**, larger than the 3.73× a 4.25× budget raise buys, and it raises real
glyph resolution instead of interpolating. This doc's own "zero-code complement: re-render at font
28–52" is therefore not a complement but a **cheap preview of the main intervention**.

Its result so far is a **null** — d5 ≡ d3 to three decimals through seven consecutive evals — but
that is on aggregate eval CE with both rungs still at the no-reading floor, and the per-position
discriminator has not been run. So this is **not yet evidence against H17**. It does raise the stakes
on this doc's sanity gate: if d5's per-position reading gain also matches d3's, then the cheap
version of the intervention failed, and the gate ("frozen-vision randstr at 1120 must replicate
≥ +100% Δperm") becomes the load-bearing part of the design rather than a formality.

Two things this comparison is **not**: the levers are complementary, not substitutes — font size buys
resolution per glyph and spends page area (fewer characters fit per page), while the budget buys grid
units and spends sequence length. And none of this measures the *information content* of a soft
token; a blank cell still emits one, and whether blank tokens are harmful, neutral or a useful
positional scaffold is unmeasured and needs the GPU.

## Fifth correction (2026-07-28) — the resolution confound, quantified, and the design fixed

The subsection above identified a confound but left it unmeasured, and it is expensive to leave that
way: the pre-registered *refute* branch promotes a much larger architectural bet, so a null must be
attributable. Measured CPU-only by `scratchpad/h17_render_resolution.py` →
`data/eval/h17-render-resolution.json` (no GPU, no weights — [H13](H13-density-ladder-long-targets.md)
was training and this repo's rule is that training runs SOLO). Five self-tests gate the numbers,
including a raster round-trip of the reconstructed model-visible image against `pixel_values` and a
mutation test that a deliberately shifted glyph-crop origin fails.

### The structural fact that drives everything

**For a square page the processor's resize target is a function of the BUDGET ALONE — the render
canvas never enters it.** Verified over canvases 768/1024/1104/1536/1584/2048: every one of them
resizes to **768 px at 280, 1104 px at 560, 1584 px at 1120**.

| source | 280 | 560 | 1120 |
|---|---|---|---|
| 1024² text page | 768² · **0.75× down** · 256 tok · 16×16 · maxpos 15 | 1104² · 1.08× up · 529 tok · 23×23 · maxpos 22 | 1584² · 1.55× up · 1089 tok · 33×33 · maxpos 32 |
| 1536² | 768² · 0.50× down | 1104² · 0.72× down | 1584² · 1.03× up |
| 2048² | 768² · 0.375× down | 1104² · 0.54× down | 1584² · 0.77× down |
| 1000×160 audio page | 1968×288 · **1.97× up** · 246 tok · 41×6 · maxpos 40/5 | 2832×432 · 2.83× up · 531 tok · 59×9 | 3984×624 · 3.98× up · 1079 tok · 83×13 · maxpos 82 |

This independently reproduces the token counts and max position ids in the prep-status table, by a
different code path. Two consequences the doc did not previously draw:

1. **The canvas is the only lever on optical fidelity, and it is decoupled from token cost.** Raising
   the canvas changes nothing about how many soft tokens a page costs; it changes only whether the
   budget's resize is a down- or an up-sample. **1584 is the smallest canvas at which 1120 stops
   interpolating.**
2. **The audio page is upsampled at every budget — 1.97× even at 280.** The 1000×160 render is 1000
   mel frames at 1 px/frame and 80 mel bins already stretched 2×, so there is no lost audio detail to
   recover at *any* rung: for audio the whole ladder is grid subdivision of interpolated pixels.
   [H20](../todo/H20-audio-phoneme-resolution.md) cannot inherit this doc's criterion; raising *its* optical
   information means a smaller hop / more mels, not a bigger budget.

### How big is the optical half of the bundle? Bounded, and small

The worry was that 280 is illegible and 1120 fixes it. Measured on real page crops at their true
sub-pixel phases — leave-one-out 1-NN letter identity over ≈2,550–2,600 characters per condition,
where the *only* source of confusion is real aliasing (no synthetic noise):

| canvas | budget | model-visible x-height | real px / char cell | 1-NN letter identity | min between-class RMS | Fisher ratio | identity @ noise σ=0.3 |
|---|---|---|---|---|---|---|---|
| 1024 | 280 | 6.00 px | 80.5 | **99.26%** | 0.0768 | 0.577 | 50.7% |
| 1024 | 560 | 8.62 px | 143.2 | **99.92%** | 0.0904 | 0.676 | 87.2% |
| 1024 | 1120 | 12.38 px (interpolated) | **143.2** | **100.00%** | 0.0986 | 0.889 | 96.6% |
| 1536 | 1120 | 11.34 px | **316.0** | 100.00% | 0.1223 | 1.070 | 99.3% |
| 2048 | 1120 | 11.60 px | **332.8** | 100.00% | 0.1147 | 1.047 | 98.7% |

("Fisher ratio" here = the minimum between-class RMS distance divided by the mean within-class RMS
spread — a scale-free margin, not an accuracy.)

Font 14 on a 1024 canvas measures: line height 17 px, advance 8.4219 px, x-height 8 px, cap height
10 px, `l`-stem ink-mass width 3.29 px. So at 280 the model sees a 6 px x-height — small, but
**letters remain 99.26% identifiable**. "Real glyph detail is destroyed before the model sees it" is
too strong: the 0.75× downsample costs **0.74 pp of single-glyph legibility** across the entire
280→1120 range. It does cost *margin* — the Fisher ratio at 280 is 65% of the 1120 value (0.577 vs
0.889) — and that margin is what shows up once per-pixel noise is added, where identity falls to
50.7% at 280 against 96.6% at 1120. But the clean channel is not the bottleneck.

**This shrinks the confound rather than dissolving it.** Leg 1 (280→560) bundles 2.07× tokens with an
optical gain bounded at < 1 pp of glyph legibility; leg 2 (560→1120) is optically identical by
construction. So the bundle was always lopsided — but "bounded and small" is an argument, not a
control, and leg 2 is the control.

**What this legibility number is not.** It assumes *perfect glyph segmentation*: each crop is placed
on the character grid the renderer used. The model gets no such thing. A 48 px cell multiplexes
`48/(advance·s) × 48/(line_height·s)` character slots — **28.6 at 280, 13.8 at 560, 6.7 at 1120**, a
4.25× reduction that is exactly the token ratio, because it is a pure area argument. Cell-internal
demultiplexing is the plausible binding constraint and it is **not measured here**; identifying a
letter you have already isolated is much easier than isolating it.

### Is the ink picture canvas-invariant? Yes — measured

If raising the canvas changed inked-token counts, the canvas would be a confound in its own right. It
does not. Same targets, font scaled with the canvas, `n=12` rows per cell (fineweb documents here fit
one page at every canvas, so the character counts are exactly paired):

| lane | chars | ch/inked @280 · 560 · 1120, canvas **1024** | same, canvas **1536** | inked@1120: 1024 vs 1536 |
|---|---|---|---|---|
| randstr d3 | 479 | 14.97 · 9.78 · 4.65 | 14.97 · 9.78 · 4.65 | 103.0 vs 103.0 |
| randstr d4 | 1919 | 23.99 · 12.46 · 6.00 | 23.99 · 12.46 · 6.06 | 320.0 vs 316.8 |
| fineweb-edu | 2658 | 19.62 · 10.05 · 5.19 | 20.07 · 10.46 · 5.31 | 511.6 vs 500.6 |

d3 and d4 reproduce the fourth correction's table to the decimal by an independent implementation.
The canvas moves these numbers by ≤ 3% (slightly *fewer* inked cells at a larger canvas, because
integer font metrics make the glyph a little smaller relative to the page). **So the canvas raise
does not change the quantity H17's argument is about — it only changes optical fidelity, which is
already near ceiling.** (One artifact worth naming so it is not mistaken for signal: the single-line
d1 rung reads 18 inked cells at 1024/1120 but 9 at 1536/1120, purely because at 1024 the text band
straddles a 48 px grid boundary and at 1536 it does not. That is a 1-line-page quirk, not a canvas
effect.)

### Why the fix is the 560 rung and not the bigger canvas

Three options were weighed:

- **Add a 560 rung — CHOSEN.** Zero re-materialization, keeps H07/H13 comparability intact, and turns
  the ladder into a decomposition whose second leg holds optics *exactly* fixed. It is the only
  option that makes a null attributable without changing any pixels.
- **Raise the render canvas — REJECTED**, on design grounds rather than cost. The compute is cheap
  (measured: 0.19 h CPU per 20.5k-row d3 rung at canvas 1024, 0.32 h at 1584; the full 5-rung ladder
  ≈ 2 h CPU, offline). It is rejected because it re-confounds the leg built to control optics, shifts
  chars/page +5–8% (6612 → 6962 → 7140 at 1024/1536/2048 — integer font metrics do not scale
  exactly), breaks comparability with H13's 280-token K values, and — since the whole ladder would
  have to be re-run at all three budgets on the new pixels — costs *three* ladder runs, not two. It
  survives as a one-rung **conditional third arm**.
- **Keep 1120 alone and re-scope the criterion to "bounded/ambiguous" — REJECTED AS THE PRIMARY
  FIX**, but adopted as part of the criterion: the "leg 1 scales, leg 2 flat" cell is reported as
  bounded, not as refutation. On its own this option is not enough, because the branch that matters
  most (a flat null) would still license nothing, and the ~12 h spent would buy no decision.

### One thing this fix does *not* repair

`scratchpad/h13_ink_occupancy.py` and `scratchpad/h13_analyze.py` call the layout helpers with
`canvas=1024` hard-coded. They are correct for every lane that exists today (all canvas 1024) and for
the whole design above, but if the conditional third arm is ever materialized, their line/word
geometry would be silently wrong on it. `lane_specs[<subset>]["canvas"]` is now recorded in the
manifest for exactly that reason.

### Renderer change and its proof

`data/preprocessing/random_strings.py` gained `--canvas` (a `LaneSpec.canvas` field, default
`CANVAS = 1024`), threaded through the render call, the `render_config` JSON, the one-page guard's
error message, and the manifest — which now records `canvas` per lane, since a root may hold lanes at
different canvases while the root-level `canvas` key is only written at creation.

The default path is **unchanged, proved against the production data rather than a temporary copy**:
`scratchpad/h17_canvas_render_identity.py` → `data/eval/h17-canvas-render-identity.json` replays each
lane's RNG and re-renders with the post-edit code, byte-comparing against the PNGs H07/H13 actually
trained on. **150 rows over 6 lanes** — `h3-randstr-v0/random-strings` train+validation and
`h13-density-v0/randstr-d{1,3,4,5}` validation — match on **targets 150/150, raster pixels 150/150
(`PIL.Image.tobytes`), and the stored `render_config` JSON 150/150**. A positive control confirms
`--canvas 1584` does change the pixels, the image size and the render config, so "identical" is not
vacuous. A separate run of the pre-edit file against the post-edit file on freshly generated splits
also matched arrow-shard SHA-256, every non-image column and `floor.json`, with the manifest
differing by exactly one added key (`canvas: 1024`).
