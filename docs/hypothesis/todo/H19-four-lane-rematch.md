# H19 — Curriculum + anchors + weighted loss makes the real mixture read

**Status:** TODO — **BLOCKED on H15/H16/H17, and the mixture as written was measured to be H13's
death configuration** ([Pre-launch audit](#pre-launch-audit-2026-07-29)). Config drafted with four
fields marked pending: `configs/h19_four_lane.yaml`.
· **Cost:** ~5.8 h GPU at the repaired mixture (was budgeted ~16 h) · Integration run · Related:
[H09](../done/H09-balanced-mixture-prevents-drowning.md), [H13](../done/H13-density-ladder-long-targets.md),
[H15](../done/H15-prior-poisoned-text.md), [H16](../done/H16-prior-gap-weighted-loss.md),
[H17](../done/H17-raise-soft-token-budget.md)

## Claim

[H09](../done/H09-balanced-mixture-prevents-drowning.md) failed from a cold start with an
unmonitored objective. Warm-starting from a checkpoint that already reads, keeping prior-proof
**anchor lanes** in the mixture, and using a grounding-aware loss will make the real four-lane
mixture read deeply.

## Design — changed on every axis the evidence demands

1. **Warm start** from the best [H15](../done/H15-prior-poisoned-text.md)/[H16](../done/H16-prior-gap-weighted-loss.md)
   checkpoint. Both prior-proof successes showed the grad_norm dip-then-rise recruitment signature;
   [H09](../done/H09-balanced-mixture-prevents-drowning.md)'s cold start never recruited (flat
   0.42–0.58 for 4000 steps).
2. **Anchor lanes retained** — randstr 15% + spoken-digits 15%, four real lanes ~17.5% each.
   [H05](../done/H05-isolated-ocr-from-scratch.md)'s lesson is that removing prior-proof pressure
   lets reading decay to zero; the replay share follows the continual-learning
   [5–25% window](../../literature/replay-prevents-forgetting.md).
3. **Grounding-aware loss** from [H16](../done/H16-prior-gap-weighted-loss.md) on all lanes.
4. **Live Δblank monitor** — **already built** (2026-07-28, as part of
   [H16](../done/H16-prior-gap-weighted-loss.md)'s prep). Setting `training.log_blank_ce: true` runs a
   no-grad blank forward per eval batch and injects `eval_<lane>_aligned_ce`,
   `eval_<lane>_blank_ce`, `eval_<lane>_prior_gap_ce` and `eval_<lane>_blank_ce_inflation` into the
   metrics before they are logged, per lane. It can be enabled **without** `gap_weighted_loss`, so
   the monitor costs one extra forward per eval batch and leaves the loss untouched. **Mandatory**:
   eval CE is proven orthogonal to grounding
   ([H09](../done/H09-balanced-mixture-prevents-drowning.md)), and this run must not be able to fail
   silently for 27 hours again.

   Two integration caveats for this run specifically: imageless microbatches carry no blank tensors
   and fall through unweighted (fine for a monitor, relevant if `gap_weighted_loss` is on for a
   mixture), and the `poisoned-text` / `masked-randstr` lanes store extra columns that break
   `concatenate_datasets` — `remove_columns` them before mixing.

2000–2500 steps at 280 soft tokens, or 1120 if [H17](../done/H17-raise-soft-token-budget.md) passed.

## Pre-registered criterion

> ### ⚠️ REPAIRED 2026-07-30 — both original bars were unreachable by arithmetic
>
> The original criterion read:
>
> > - **Confirms:** fineweb Δperm **≥ +30%** (an order above the saturated +2.5%) **and** positional gain
> >   at tokens 10–100 ≥ +5 pts; librispeech position-0 gain flips positive and Δblank ≥ +20%; anchors
> >   retain ≥ 80% of their single-lane Δperm.
> > - **Diagnostic failure:** anchors hold but real lanes stay **< +10%** ⇒ real-lane density exceeds the
> >   H13 ceiling ⇒ H17 becomes **mandatory rather than optional**.
>
> **Aggregate Δperm is a function of target length** (see
> [the calibration table](../README.md#criterion-defects-found-by-running-the-criteria--audit-any-new-bar-against-these)):
> on a checkpoint that unambiguously reads it measures **+14.31% at 60 supervised tokens/row, +3.05% at
> 234, +0.63% at 932.** This mixture's lanes run **fineweb 298.4** (993.3 as originally built),
> **densefusion 238.4, smoltalk 198.0, librispeech 59.1**. So:
> - **"fineweb Δperm ≥ +30%" sits ~10× (at 298 tok) to ~48× (at 993 tok) above what *full* reading
>   produces at that length.** Unreachable however well the model reads.
> - **"real lanes stay < +10%" fires deterministically** — three of the four real lanes are below +10%
>   by arithmetic regardless of what they read — and its consequence was a queue decision ("H17 becomes
>   mandatory") on a 26 GPU-hour ladder that has since come back **VOID**.
>
> The `≥ 80% of single-lane Δperm` anchor bar is **fine and is kept**: it is a same-lane, same-length
> *ratio*, so the dilution cancels.

- **Confirms:** on **fineweb** — `pos0_gain` **≥ +40 pts** ([H10](../done/H10-reading-concentrated-at-start.md)
  measured +70 on this lane; [H21](../done/H21-vision-path-not-scale-invariant.md)'s trained fonts
  +52…+64) **and** `char_bins[10-25]` **≥ +5 pts** with a row-clustered CI excluding 0 **and**
  prefix-matched Δperm over the **first 5–10 answer tokens** ≥ +30%. On **librispeech** — `pos0_gain`
  turns significantly positive **and** prefix-matched Δblank (first 5–10 tokens) ≥ +20%. **Anchors**
  retain ≥ 80% of their single-lane Δperm (unchanged — same-lane ratio).
- **Diagnostic failure:** anchors hold but the real lanes' **prefix-matched** Δperm (first 5–10 tokens)
  stays < +10% **and** their `pos0_gain` stays below +10 pts. Only then is the real-lane null about the
  lanes rather than about the statistic.
- **Report aggregate Δperm per lane as context**, alongside the lane's supervised-tokens/row, so the
  number can be compared with the calibration table instead of against a fixed bar.

⚠️ **"librispeech position-0 gain flips positive" needs its baseline re-measured before it can be
scored.** The −12.7 pts it refers to was measured on the **RGB** librispeech lane against a **dark-red
(128, 0, 0)** blank; the current instrument (`hybrid_4lane_position_decay.py` → `blank_like()`) emits
**true gray**. The criterion's reference point cannot be reproduced by the tool that will evaluate it.
Re-measure librispeech `pos0_gain` on `hybrid-4lane-v0/final` with the repaired gray blank (~25 min GPU,
no training) and restate the bar against that number.

## Do not run before the upstream single-lane results

This is the expensive integration. Everything upstream of it is single-lane, bounded, and carries a
kill criterion. Run SOLO.

---

## Pre-launch audit (2026-07-29)

CPU-only, no GPU touched. `scratchpad/h19_mixture_audit.py` (`--self-test`, 6 checks) →
`data/eval/h19-mixture-audit.json`; `scratchpad/h19_build_mixture.py` (`--self-test`, 3 checks) →
`data/eval/h19-mixture-build-planB.json`. Config: `configs/h19_four_lane.yaml`.

### 1. The mixture as specified is [H13](../done/H13-density-ladder-long-targets.md)'s death configuration

Design point 2 gives **row** shares. Loss is normalised per **supervised token**, so gradient share
is `rows × tokens/row`. Measured with the real Qwen3 tokenizer over 400 train rows per lane:

| lane | rows | row % | sup tok/row | **TOKEN %** | readable within depth 48 |
|---|---|---|---|---|---|
| fineweb-edu | 28,000 | 17.5% | **993.3** | **63.2%** | **4.8%** |
| smoltalk | 28,000 | 17.5% | 249.4 | 15.9% | 18.0% |
| densefusion | 28,000 | 17.5% | 239.6 | 15.3% | 20.0% |
| librispeech | 28,000 | 17.5% | 59.4 | 3.8% | 74.9% |
| spoken-digits | 24,000 | 15.0% | 17.0 | **0.9%** | 100% |
| random-strings | 24,000 | 15.0% | 16.5 | **0.9%** | 100% |
| | | | | | **overall readable: 13.6%** |

H13 died at **14.2%** readable with its d4 rung on **63.04%** of the gradient. H19-as-written is
**13.6%** readable with fineweb on **63.2%**. That is the same failure twice. Worse, the two
prior-proof anchors — the lanes whose entire job is to stop reading decaying to zero
([H05](../done/H05-isolated-ocr-from-scratch.md)) — would each have carried **0.9%** of the gradient;
H13's d1 got 1.13% and lost the bootstrap race by step 110. The replay share the design cites from
[the continual-learning literature](../../literature/replay-prevents-forgetting.md) is 5–25%; read in
tokens, 15% + 15% of rows is **1.8%**.

### 2. The repair: the same numbers, read as TOKEN shares

Nothing about the design changes — the same six lanes at the same 17.5/17.5/17.5/17.5/15/15 — only
which quantity those numbers balance. Row counts become a consequence, computed by
`scratchpad/h19_build_mixture.py` (lanes measured on the rows that survive the length filter, i.e.
the rows a run actually trains on), at 2000 steps × effective batch 64 = 128,000 rows:

| lane | rows | row % | sup tok/row | TOKEN % | readable |
|---|---|---|---|---|---|
| librispeech | 16,419 | 15.1% | 59.1 | 18.6% | 75.0% |
| densefusion | 4,070 | 3.7% | 238.4 | 18.6% | 20.1% |
| smoltalk | 4,900 | 4.5% | 198.0 | 18.6% | 22.3% |
| fineweb-edu | 3,252 | 3.0% | 298.4 | 18.6% | 16.1% |
| random-strings | 50,436 | 46.2% | 16.5 | 15.9% | 100% |
| spoken-digits | 30,000 | 27.5% | 17.0 | 9.8% (clamped) | 100% |
| | | | | | **overall readable: 50.5%** |

**13.6% → 50.5%**, a 3.7× repair, with no new lane, no reweighting scheme and no code change.

> ⚠️ **Every percentage in that table is computed against a depth of 48 answer tokens, which is retired**
> (2026-07-30). 48 was the K estimator's structural floor, not a measurement; the measured readable depth
> is **~5–12 answer tokens**, so these figures are overstated roughly **4–10×**. **Recomputed on the real
> splits** (`scratchpad/h19_mixture_audit.py --scan-depth {12,5} -n 400 --prefilter`; the depth-48 control
> reproduces this table to 0.02 pp, so only the depth changed):
>
> | mixture | at depth 48 | at depth 12 | at depth 5 |
> |---|---|---|---|
> | A — row shares, as originally written | 25.8% | **8.3%** | **3.5%** |
> | B — token shares | 53.3% | **27.6%** | **11.5%** |
> | C — split text lane (**the repair this table describes**) | 62.4% | **30.0%** | **12.5%** |
>
> Per lane (depth 48 → 12 → 5): fineweb-edu 16.1 / 4.0 / 1.7 · densefusion 20.1 / 5.0 / 2.1 · smoltalk
> 22.0 / 5.9 / 2.5 · librispeech 74.9 / 20.2 / 8.4 · random-strings 100 / 72.7 / 30.3 · spoken-digits
> 100 / 70.6 / 29.4. **The four real lanes collapse to 2–20%; only the prior-proof anchors survive**, and
> reaching even 33.9% at depth 12 needs the anchors to hold ~40% of the token share.
>
> So the "repaired" mixture lands back inside
> [H13](../done/H13-density-ladder-long-targets.md)'s death band (it died at 14.2% *as computed at depth
> 48*), not above [H14](../done/H14-masked-region-targets.md)'s repaired 75.2%.
>
> **Do not act on that reversal mechanically.** The reference points (13.6% here, 14.2% for H13, 75.2%
> for H14, 100% for H07) were *also* computed at depth 48, so recomputing one side only makes the
> comparison depth-inconsistent. Recompute both sides at the same depth before using any of them as a
> gate. And note the counter-evidence:
> [H21](../done/H21-vision-path-not-scale-invariant.md)'s leg 1 trained **healthily** (`grad_norm`
> 0.65–0.86, the only clean curve in its batch) on a lane whose corrected readable fraction is also
> **~20%** — so ~20% corrected-readable is empirically survivable. The gate's *ordering* remains useful;
> its absolute thresholds do not.
>
> `scratchpad/h19_mixture_audit.py` and `h19_build_mixture.py` hardcode `SCAN_DEPTH = 48`. Regenerate
> with an explicit depth before quoting a readable fraction anywhere.

### 3. What the repair costs — three things, all recorded before the run

1. **Exposure.** densefusion sees 4,070 distinct natural images instead of 28,000. Equal gradient and
   equal example variety are in direct conflict when target lengths differ 18×, and H13's lesson
   picks gradient. If captioning needs variety more than gradient, the fix is **shorter densefusion
   targets** (a materializer change), not more rows — more rows put the token share back where it
   killed H13.
2. **spoken-digits is clamped.** The token target asks for 48,923 rows and 30,000 exist, so its
   realised share is 9.8% and the readable fraction lands at 50.5% instead of 53.3%. Prep step, CPU,
   ~28 min (the FSDD source is already at `data/raw/fsdd`):
   `uv run python -m data.preprocessing.spoken_digits --out data/materialized/spoken-digits-v1
   --train-rows 60000 --val-rows 1000`.
3. **50.5% is between the only two calibration points that exist** (H13 died at 14.2%,
   [H14](../done/H14-masked-region-targets.md) confirmed at 75.2%). Nobody knows where the threshold
   is. densefusion and smoltalk cap it — their targets are ~20% readable and no row count changes
   that. An alternative "mixture C" (H15's two-length-arm split of the text lane: 13% token share to
   the readable short arm, 4.5% to the dense page) measures **62.4%**, but trains mostly on *sparse*
   pages, because `poisoned_text.py` renders exactly the target, so a 64-token cap draws a nearly
   empty page. `scratchpad/h19_build_mixture.py --mix C` builds it.

   **An assumption this inherits, which has never been tested.** H13's scan-depth rule was measured
   on **transcription** lanes, where answer-token position maps to page position. A densefusion
   caption token at position 200 still depends on the image (a photo is not scanned left to right),
   and a smoltalk answer depends on a short rendered instruction near the top of the page. The rule
   is probably over-conservative for those two lanes. Treat 50.5% as a **lower bound** and say so;
   do not quote it as if it were measured on them.

### 4. `remove_columns` — the doc's claim above is wrong, and the truth is worse

Design point 4 says the extra columns "break `concatenate_datasets`". Measured on **datasets 4.3.0**
they do not: `concatenate_datasets` aligns by name and **null-fills**, silently returning a 13-column
(`poisoned-text`: `poison_mask`, `poison_rate`, `source_row_id`) or 22-column (`masked-randstr`:
`words`, `mask_*`, `visible_words`, `sentinel_*`, `mask_b_*`) table. Row payloads survive — `messages`,
`row_id` and image geometry were checked on both sides of the join — and the standard lane's rows just
get `None`. So the failure mode is **silence, not an exception**, and it is version-dependent: nothing
in this repo pins `datasets`, and a version that raises would present as "H19 crashed at startup".
Column **order** also differs (smoltalk stores `messages` before `images`) and is tolerated.
**Handled:** the build script strips every non-base column and then asserts that all lanes' Features
are identical and the merged table carries exactly the 10 base columns.

### 5. Retention, `max_length`, and the single-root problem

`_filter_training_tokens` **silently drops** overflowing rows. Retention computed on the **full**
splits with the filter's own arrow predicate, cross-checked against the real filter on 2,000 rows per
lane (identical counts, 9/9 lanes):

| max_length | fineweb-edu | smoltalk | densefusion | librispeech | randstr / spoken-digits / poisoned / masked |
|---|---|---|---|---|---|
| 1024 | **38.96%** | 87.73% | 99.96% | 99.98% | 100.00% |
| 2048 | 79.91% | 99.60% | 100.00% | 100.00% | 100.00% |
| 4096 | 95.07% | 99.89% | 100.00% | 100.00% | 100.00% |

**Full-page fineweb-edu can never reach 100.00% at any practical budget** — its longest row estimates
at 168,616 tokens. So "100.00% retention" is only achievable by *pre-selecting* rows from the passing
set, which is what the build script does; retention through the trainer's own filter is then 100.00%
by construction on every lane, and the per-lane token statistics above are measured on those same
rows. Chosen `max_length: 1024`; real assembled sequences through `HybridCollator` max at **752**
(fineweb 752, smoltalk 692, densefusion 639, librispeech 576, randstr 292, spoken-digits 282; mixture
mean ~354), so nothing truncates.

Separately: `univi.trainer._load_local` reads **one** `dataset.path`, but these lanes live in three
materialization roots, and `max_train_rows_per_subset` is a single **global** cap that cannot express
per-lane counts. Both are why the mixture must be built into one root rather than expressed in YAML.

### 6. VRAM

Same accounting as `scratchpad/h17_config_verify.estimate_vram`, calibrated against H13 (estimate
20.05 GB, measured ~21 GB). At the mixture's max assembled sequence of 752:

| per-device batch | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| GB | 12.26 | 13.69 | **16.54** | 22.24 | 33.65 |
| GB (pessimistic 14 B/logit) | 12.47 | 14.11 | **17.39** | 23.94 | 37.05 |

Budget ceiling 26 GB: bs4 and bs8 both fit. **Total-card feasibility:** the 40 GB A100 currently
shows ~16 GB held by another container (~24 GB free), so bs8 at 23.94 GB pessimistic would very
likely OOM and bs4 is chosen (~6.6 GB headroom). Note that a 26 GB config and that container's 16 GB
**do not both fit on a 40 GB card** — anything written to the full ceiling is only launchable once it
releases. `log_blank_ce` costs one extra no-grad forward per *eval* batch, no training-path memory.
Wall clock: mean sequence 354 = 0.72× H14's 493, and H14 measured 14.5 s/step at this same 4 × 16
geometry ⇒ ~10.5 s/step ⇒ **~5.8 h** for 2000 steps.

### 7. What is still PENDING — this config cannot be finalised yet

| field | pending on | placeholder | consequence of changing it |
|---|---|---|---|
| `model.init_from` | H15 / H16 verdicts | `hybrid-pretrained-randstr-v0/final` (H07 — the only checkpoint that currently exists *and* reads) | none for the data build; changes what "warm start" means and must be stated in the write-up. [H14](../done/H14-masked-region-targets.md)'s checkpoint (Δperm +23.90%) is now also a candidate |
| `model.max_soft_tokens` | [H17](../done/H17-raise-soft-token-budget.md) | 280 | at 1120 the per-image budget goes 282 → 1122, `max_length` must reach ≥ 2048, retention must be re-measured, the VRAM table is void (expect bs1–2), **and the data build must be redone** |
| `training.max_steps` | H17 (280 ⇒ 2000–2500, 1120 ⇒ 1120) | 2000 | sets `--total-rows`, so **every row count in the mixture changes** |
| `training.gap_weighted_loss` | [H16](../done/H16-prior-gap-weighted-loss.md) | off (`log_blank_ce` on) | turning on an unvalidated objective in the expensive integration run is how H09 lost 27 h |

Because the row counts are a function of `max_steps` and `max_length`, **the data build is downstream
of H17**, not merely the YAML. `data/materialized/h19-mix-v0` is deliberately not built yet.

### 8. What this audit does NOT license

1. It is arithmetic on measured token counts, not a result. It says the *as-written* mixture sits at
   H13's death point; it does not prove the repaired one reads. Only the run does.
2. The 48-token scan depth is [H13](../done/H13-density-ladder-long-targets.md) §4's number, measured
   on **one checkpoint** (H07) trained at **one density**. If [H17](../done/H17-raise-soft-token-budget.md)
   moves it, every readable fraction here moves with it.
3. Per-lane target-token means come from 400-row samples (the retention and clamp numbers are
   full-split). Lane means are stable to well under 1%, but the fineweb tail is heavy — its mean fell
   993.3 → 298.4 once the length pre-filter was applied, which is a selection effect, not noise, and
   is exactly why the pre-filtered numbers are the ones quoted.
4. Nothing here touches the criterion. Confirms/refutes stay in Δperm / Δblank / positional reading
   gain, never eval CE.
