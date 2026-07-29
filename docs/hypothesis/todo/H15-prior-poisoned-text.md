# H15 — Making the prior wrong at every position restores deep reading

**Status:** TODO — **REVISED 2026-07-29 after [H13](../done/H13-density-ladder-long-targets.md)'s
verdict; the original design would have collapsed and the original claim is largely pre-refuted.**
· **Cost:** ~4 h GPU (3 arms) + ~30 min CPU materialization · Attacks **(B) objective**
· Related: [H14](../in-progress/H14-masked-region-targets.md), [H16](H16-prior-gap-weighted-loss.md),
[H11](../done/H11-position-decay-is-prior-induced.md), [H13](../done/H13-density-ladder-long-targets.md)

## Claim (original)

[H10](../done/H10-reading-concentrated-at-start.md) showed reading is bought only where the prior
fails, and the prior only fails at position 0. Corrupt the *source text before rendering* so the
prior is wrong at **every** position, and deep reading should appear.

## Revision forced by H13 (2026-07-29, pre-launch)

Three separate problems, all measured rather than argued. Every number below is from the real Qwen3
tokenizer (`data/checkpoints/encoder-free-v0/best`) on real `fineweb-edu` rows pushed through
`poisoned_text.py`'s own target-building path — `scratchpad/h15_readable_fraction.py` and
`scratchpad/h15_position_profile.py`, outputs `data/eval/h15-readable-fraction.json`,
`data/eval/h15-position-profile{,-short,-long}.json`.

### 1. The lane as decided was 5.2% readable — worse than the run that died

[H13](../done/H13-density-ladder-long-targets.md) §3: the model reads only ~**48** answer tokens
deep; a run whose supervised tokens fall mostly beyond that depth collapses into the
uniform-marginal basin, because every unreachable token teaches "emit the marginal, ignore the
image" and a from-scratch adapter loses the bootstrap race. H13 died at **14.2%** readable;
[H07](../done/H07-pretrained-vision-adapter-qwen.md) succeeded at **100%**;
[H14](../in-progress/H14-masked-region-targets.md) was repaired from 22.9% to **75.2%** before launch.

Measured over 200 real validation rows (`readable = Σ min(48, T_i) / Σ T_i`):

| H15 variant | target tok/row | chars/tok | **readable** |
|---|---|---|---|
| as *designed* (4 pages, no cap) | 1507.9 | 2.97 | **3.2 %** |
| as *decided* (`--max-target-tokens 1400`) | **923.2** | 2.99 | **5.2 %** |
| 1 page, no cap | 1029.0 | 2.98 | 4.7 % |
| cap 400 | 366.2 | 2.99 | 13.1 % |
| cap 200 | 181.7 | 2.98 | 26.4 % |
| cap 128 | 111.9 | 2.96 | 42.9 % |
| cap 96 | 78.5 | 2.96 | 61.1 % |
| cap 64 | 44.3 | 2.96 | 92.8 % |
| cap 48 | 33.6 | 2.96 | 100 % |

**H15 as decided is H13's `d4` rung wearing English.** d4 measured 932 target tokens/row and 5.2%
readable; H15 measures 923.2 and 5.2%. d4 was the single largest contributor to H13's collapse (63%
of its gradient); H15 would have run that regime **alone, at 100% of the gradient**. The prep note's
reasoning — "trimming at 1400 keeps the longest documents instead of biasing the sample against
depth" — optimised for exactly the quantity that killed H13.

### 2. "The prior is wrong at every position" is false in *token* terms at p = 0.15

The loss is per **token**; the poisoning is per **character**. Measured at depth 20–50 (200 rows):
of 30 target tokens/row, **9.68 (32.3%) touch a poisoned character and only 1.99 (6.6%) lie
entirely inside poisoned characters**. Across all bins only **~20% of poison-touching tokens are
pure**. So at p = 0.15 more than two thirds of tokens at depth remain fully prior-predictable, the
"pure-reading metric" has ~5× less data than the doc implies, and a null would not distinguish *"the
prior is not the constraint"* from *"the dose was too weak"*. Hence the dose ladder below.

### 3. The strong form of the claim is already pre-refuted, so the hypothesis is re-scoped

`random-strings` **is the p → 1 limit of this design**: a target of uniform i.i.d. letters, where the
prior is wrong at every position by construction and blank-branch accuracy is ~0 everywhere.

- [H11](../done/H11-position-decay-is-prior-induced.md) ran the position probe on two fully
  prior-proof lanes and found the decay **persists with the prior removed entirely**.
- [H13](../done/H13-density-ladder-long-targets.md) §4 put a number on it: on the H07 checkpoint,
  **K = 48.8 / 48.5 / 48.5 tokens** across a 16× span of page density, on prior-proof lanes.

Removing the prior *completely* does not produce deep reading; it produces a ceiling at token ~48.
A 15% character substitution on real text is a strictly **weaker** intervention on the same axis.
"Deep reading appears" is therefore not a live outcome, and a criterion demanding it is close to
unfalsifiable.

**What is still open, and worth measuring:** real text currently dies at ~10 tokens
([H10](../done/H10-reading-concentrated-at-start.md): fineweb gain +0.13 pts by position 10–19)
while prior-proof text sustains to ~48. That **~10 → ~48 gap is the prior-attributable share of the
depth ceiling**, and nothing has measured it, because H11's prior-proof lanes had 14.5-token targets
and H10's real lane had a different checkpoint. H15 is re-scoped to measure exactly that, which makes
it the **gate for [H16](H16-prior-gap-weighted-loss.md)**: if the prior costs ~38 tokens of depth,
an objective-side fix has something to win; if it costs ~0, H16 is dead too and the queue goes
entirely to (A′)/(C) via [H17](H17-raise-soft-token-budget.md).

## The depth-vs-collapse tension, and how it is resolved

**State it plainly.** H15 exists to test reading at *deep* positions. H13 says supervised tokens at
deep positions are precisely what collapses a run on this architecture. A lane short enough to be
safe looks too short to test the hypothesis. This is a real conflict and it cannot be waved away by
"warm-starting will hold it" — see the H14 dependency below.

It resolves into three moves, in order of how much they buy:

1. **Separate the depth you *supervise* from the depth you *measure*.** Measurement is free:
   teacher-forced position-resolved reading gain on a validation split of any length costs no
   gradient and cannot collapse anything. The precedent is H13 §4 itself — H07 was trained on
   14-token targets and its K was measured at 48.5 on 931-token d4 rows. H15 therefore **measures**
   on a 923-tok/row deep validation split that is never trained on, out to bin 400+.
   *Cost of this move:* deep positions are off-distribution for a model trained to stop earlier, so
   its stop-token habit inflates absolute CE there. Reading gain is an aligned − blank *difference*
   at the same position, so that penalty cancels to first order; H13 §4 produced a clean +28 pt
   position-0 signal this way. It is still a caveat, not a nullity.
2. **Buy supervision depth with token weight, not row count.** H13's generalisable lesson is that
   *balancing rows does not balance gradient*. Run it in reverse: a **minority of long rows** buys
   deep supervision at a *controlled* gradient share. 4,000 of 50,000 rows (8.0%) at 181.7 tokens
   supplies 26.3% of the token-weighted gradient and supervises to **~182 tokens = 3.8× the scan
   depth**, while the mix stays at **75.4% readable** — H14's repaired level.
3. **Accept a bounded claim.** Even repaired, H15 cannot demonstrate reading at 400+ tokens; nothing
   in this programme ever has, including a 100%-prior-proof lane. The re-scoped question — *how much
   of the ~48-token ceiling is prior-attributable* — is bounded a priori between H10's ~10 tokens and
   H13's ~48, and that is the whole size of the prize. Anyone hoping for more should read §3 above
   before spending the GPU hours.

**Rejected alternatives, and why.** *Curriculum (short → long)* adds a schedule variable to a run
that already changes the lane, and H13's collapse happened by step 110 — a curriculum that reaches
long targets at step 200 would collapse at step 200. *Warm-start alone at 900-token targets* is not
licensed by anything (below). *Measuring at ~100 tokens only* is subsumed: the deep val split gives
100-token bins for free, so there is no reason to cap the measurement.

## H14 dependency — not blocked, but with a pre-registered contingency

H15 is **scientifically independent of H14** and can launch as soon as the GPU frees. It does not
rely on H14's result, and it must not: **H14 changed two things at once** — readable density
(22.9% → 75.2%) *and* warm-start (`init_from`) — so whatever H14 reports, it cannot license the
claim "warm-start makes long targets survivable". That claim remains untested by anything, which is
why H15's repair fixes the readable fraction by design instead of betting on the warm start.

Pre-registered contingency, recorded before either result:

- **H14 reads (any positive grounding) ⇒ launch H15 as specified.** 75.4% readable + warm-start is
  then a demonstrated-survivable configuration.
- **H14 returns VOID / collapse at 75.2% readable ⇒ drop H15's long arm** and run the short arm
  alone (`poisoned-text` only, 92.8% readable, supervision to ~46 tokens), measuring depth on the
  deep val split exactly as before. That costs supervision depth, not measurement depth.

The only hard blocker is GPU serialisation — H14 holds the card, and this repo's rule is that
training runs SOLO.

## Revised design

**Lane.** `data/preprocessing/poisoned_text.py`, unchanged (it already supports everything needed;
manifest merging was built so several rates can share one root).

**Two length arms per dose**, sized by token weight (measured at p = 0.15):

| subset | `--max-target-tokens` | tgt tok/row | readable | rows | token share |
|---|---|---|---|---|---|
| `poisoned-text` | 64 | 44.3 | 92.8 % | 46,000 | 73.7 % |
| `poisoned-text-long` | 200 | 181.7 | 26.4 % | 4,000 | 26.3 % |
| **mix** | | | **75.4 %** | **50,000** | |

The same 46k/4k row split lands at **73.7%** readable at p = 0.00 and **76.5%** at p = 0.50, so the
design is robust across the ladder. `"poisoned-text-long"` was added to `univi.trainer.VALID_SUBSETS`
(one additive line) so both arms load and `concatenate_datasets` merges them — their `Features` are
identical because both come from this materializer.

**Do not use `--max-target-tokens 48`.** `_fit_token_budget` returns `""` — and `_make_row` then
silently drops the row — whenever the *first rendered line alone* exceeds the cap. Measured row loss
at cap 48: **1/200 at p = 0.00, 4/200 at p = 0.15, 16/200 at p = 0.30, 72/200 (36%) at p = 0.50.**
The drop rate is **dose-dependent**, i.e. a selection confound that runs along the treatment axis. At
cap 64 it is 0/200 up to p = 0.30 and 1/200 at p = 0.50.

**Dose ladder — three runs, not one** (`configs/h15_poisoned_p{00,15,50}.yaml`):

| arm | poison rate | why |
|---|---|---|
| control | **0.00** | the missing control (see defect 2). Same source rows, same recipe, same lengths, same warm start. |
| pre-registered | **0.15** | the dose the hypothesis was written around. |
| strong | **0.50** | measured at depth 20–50: **67.9% of tokens touch** a poisoned char and **21.5% are pure**, against 32.3% / 6.6% at p = 0.15 — 3.2× the prior-proof power (6.45 vs 1.99 pure tokens/row) — so a null is interpretable. Word shapes, layout and punctuation survive. |
| *(already run)* | *1.00* | `random-strings` — the p → 1 limit, K = 48.5 tok (H13 §4). Not re-run. |

If only two arms fit the budget, run **0.00 and 0.50** — the endpoints carry the contrast.

**Matching.** Arms are matched on **target-token length**, not characters/page: at a fixed token cap
clean text runs 4.51 chars/token against poisoned text's 2.96, so the control page carries ~1.5×
more characters. H13 licenses this — position-0 gain is statistically identical at d3 (+28.7 ± 3.7)
and d4 (+28.0 ± 3.7) across a 4× density change, and K is invariant over 16×. Matching on tokens
keeps the readable fraction and gradient structure comparable, which is the quantity that killed H13.

**Recipe.** H07 verbatim (3-group LR 3e-4 / 5e-5 / 2e-5, **no freeze**, vision trainable), effective
batch 64 as 4 × 16, `max_soft_tokens: 280`, `max_length: 1024` (worst case 200 + 282 + 256 = 738 ⇒
100% retention), 400 steps ≈ 0.51 epochs. Warm-start `init_from:
data/checkpoints/hybrid-pretrained-randstr-v0/final`. Note that checkpoint read *random strings*,
never English — which is another reason the p = 0.00 control is mandatory.

**Live collapse alarm.** `log_blank_ce: true` costs one no-grad blank forward per **eval** batch
(the training path early-returns to the stock loss while `gap_weighted_loss` is off) and emits
`eval_*_aligned_ce / _blank_ce / _prior_gap_ce / _blank_ce_inflation`. **Read it as an alarm, never
as grounding**: `prior_gap_ce` is a Δblank quantity, and H13 §1 showed a fully dead run with
Δperm = 0 still carried Δblank of +1.0% … +8.7% (ink-presence, not content). Pre-registered use: if
`eval_prior_gap_ce` < 0.05 nats for three consecutive evals, kill the run rather than spend the
remaining hours — the failure H13 only discovered post hoc.

## Pre-registered criteria (revised)

**Primary instrument: position-resolved reading gain, aligned vs blank**
(`scratchpad/hybrid_4lane_position_decay.py --val-path data/materialized/h15-deepval-p<rate>/poisoned-text/validation -n 150`),
reported per position bin. **Not** aggregate CE, **not** `floor.json`, **not** aggregate Δperm.

Two readouts per arm:

- **all-token gain** — defined on every arm including the p = 0.00 control, so the arms are
  comparable;
- **pure-poisoned-token gain** — restricted to tokens lying *entirely* inside poisoned characters
  (map `poison_mask` through `return_offsets_mapping`). Prior-proof by construction: no model can
  beat 1/26 there without reading pixels. Power on the deep val split at n = 150 rows: **480 / 963 /
  1859 / 5421** pure tokens in bins 50–100 / 100–200 / 200–400 / 400+.

Define **D50** = the deepest position bin whose all-token reading gain is ≥ 50% of the position-0–1
gain. (This is H13's K in *token* units; H13 defined K in characters with a 0–100-char reference
window, and on this lane 100 chars ≈ 34 tokens. Report both so the numbers stay comparable.)

**VOID guard, evaluated first.** If prefix-matched Δperm over the first 48 answer tokens is
< +10% on any arm, that arm did not read at all and **none of its position numbers may be
interpreted** — exactly the H13 failure. Report it and stop. (Prefix-matched, because H13 §4 showed
aggregate Δperm collapses 14.31% → 0.63% for *identical* reading purely as targets lengthen; the old
"Δperm ≥ +30%" bar was measuring target length, not grounding.)

Then, comparing the poisoned arm(s) against the p = 0.00 control on the same deep val geometry:

- **CONFIRMS (the prior is the binding constraint at depth):** poisoned D50 ≥ 2× control D50, **and**
  all-token gain in bin 20–50 exceeds the control by ≥ +10 pts with a row-clustered bootstrap CI
  excluding +2 pts, **and** pure-poisoned-token gain in bin 20–50 ≥ +10 pts with its CI excluding 0.
  ⇒ a data-side prior intervention buys depth ⇒ **promote [H16](H16-prior-gap-weighted-loss.md)**,
  which does the same thing with a loss weight instead of a data rebuild.
- **PARTIAL (the expected outcome, per §3):** depth improves over the control but D50 saturates at
  ≲ 48 tokens — the poisoned arm converges to the prior-proof ceiling and stops. ⇒ the prior costs
  real depth *below* 48 but a second constraint sets the ceiling ⇒ H16 is worth running only if its
  target is depth < 48; the ceiling itself is [H17](H17-raise-soft-token-budget.md)'s (A′)-vs-(C)
  question.
- **REFUTES:** poisoned and control depth profiles are statistically indistinguishable — the CI on
  the gain *difference* contains 0 in **both** bins 10–20 and 20–50 — with both arms passing the
  VOID guard. ⇒ prior competition is not what limits depth on real text ⇒ **(B) is dead as an
  account of the positional ceiling; kill H16 as well** and spend the queue on H17.
- **DOSE-INCOHERENT:** p = 0.50 shows no more effect than p = 0.15, or the ordering inverts. ⇒ report
  as such; a monotone dose-response is what would license a causal reading, and its absence means the
  three runs measured something other than the intended manipulation.

**What changed from the original criteria, and why**

| original | problem | replacement |
|---|---|---|
| "poisoned-position gain ≥ +20 pts at answer positions > 50 (currently ~+1 pt)" | "currently ~+1 pt" is H10's number from a *different checkpoint* (`hybrid-4lane-v0/final`), different mixture, length and recipe — a cross-run historical comparison, not a control. And positions > 50 sit beyond the ceiling a **100%** prior-proof lane exhibits (K = 48.5), so +20 pts there was near-unfalsifiable. | control-referenced D50 and bin-20–50 gain deltas against a matched p = 0.00 run, with CIs. |
| "Δperm ≥ +30%" | H13 §4: aggregate Δperm is a function of *target length* (14.31% → 0.63% for the same reading). A fixed threshold measures length. | prefix-matched Δperm over the first 48 tokens, used only as a **VOID guard**. |
| (none) | no control arm at all | p = 0.00 arm, mandatory. |
| floor as "crossing it proves reading" | H13 §7: response-only masking supervises `target + <|im_end|>\n` (T+2 tokens) while the floor divides by T, so measured CE is deflated by T/(T+2) — the bias runs *toward false positives*. Measured here: the published loose floor is **6.06% too high at cap 48** (1.1330 → 1.0643), 1.78% at cap 128, 0.30% at cap 1400. | no criterion touches CE or `floor.json` at all. `floor.json` should be regenerated over T+2 before it is quoted anywhere. |

## What an H15 result would and would not license

**Would license.**

- A **magnitude for the prior-attributable share of the depth ceiling** on real text, measured within
  one run family, one recipe and one warm start — the number H10 and H11 between them could not
  produce.
- A **go/no-go on [H16](H16-prior-gap-weighted-loss.md)**: H16 reweights loss by the blind-branch gap,
  i.e. it is the objective-side version of the same intervention. If poisoning the data cannot buy
  depth, reweighting the loss almost certainly cannot either, and 6 h are saved.
- With the dose ladder, a **dose-response curve** interpolating between two already-measured
  endpoints (p = 0 real text; p → 1 = `random-strings`, K = 48.5).

**Would NOT license.**

1. **Nothing about full-page density.** Both arms render short targets, so the page is d2-like, not a
   full fineweb page. H13 says depth is density-invariant, which is what makes this legitimate — but
   it means a positive result reads "at d2 density", exactly the caveat H14 accepted.
2. **Nothing about depth beyond ~182 tokens as a *trainable* regime.** Deeper positions are measured,
   never supervised. A gain that appears at 200–400 would be transfer from shallower supervision, and
   should be reported as such.
3. **No claim that the prior is "removed".** At p = 0.15 only 32.3% of tokens at depth touch a
   poisoned character (6.6% purely); at p = 0.50 it is 67.9% / 21.5%. The manipulation is graded, and the
   pure-token metric is the only prior-*proof* readout — the all-token readout is contaminated by
   partially-predictable tokens.
4. **No architectural conclusion.** A confirm result says the objective can buy depth *up to* the
   ceiling; it says nothing about whether the ceiling is (A′) capacity or (C) scan. Only
   [H17](H17-raise-soft-token-budget.md) separates those.
5. **Nothing about the warm start.** All arms warm-start from the same checkpoint, so the design
   cannot attribute anything to `init_from` — deliberately, since that is a variable H14 already
   confounds.

## Queue recommendation

- **Keep H15 behind [H17](H17-raise-soft-token-budget.md)** — unchanged. H17 discriminates (C) vs
  (A′), which is the ceiling H15's own outcome is bounded by.
- **Keep it ahead of [H16](H16-prior-gap-weighted-loss.md)**, and re-label it *H16's gate* rather
  than a fix in its own right.
- **As originally framed it is not worth running.** The strong claim ("deep reading appears") is
  pre-refuted by H11 + H13 §4, and the design would have collapsed before measuring anything. As
  re-scoped — measure the prior-attributable share of a ceiling whose value is already known — it is
  worth ~4 h, mostly because it decides a 6 h follow-up.
- **Optional cheap pre-check (~25 min GPU, no training).** Re-run the position probe on
  `hybrid-4lane-v0/final` with a **per-token dump**, and stratify fineweb tokens at *fixed* depth by
  blank-branch CE (prior strength). If, at depth 20–50, weak-prior tokens show materially more gain
  than strong-prior tokens, the prior is binding at depth and H15/H16 are promoted. **This can
  promote but cannot refute** — a null is equally consistent with "never learned to read there", and
  [H12](../done/H12-contrastive-decoding-probe.md) already predicts the null. The existing artifact
  `data/eval/hybrid-4lane-position-decay-4lane-final.json` holds bin aggregates only, so the probe
  must be re-run to emit per-token values.

## Materialization — NOT run, awaiting a decision

CPU only, no GPU. Measured cost on this box: **83 ms/row** at cap 64, 88 ms at cap 128, 172 ms at
cap 1400 (single core, one page/row at every cap tested). At `--num-proc 8`: **~9 min and ~0.6 GB
per dose**, ~30 min and ~1.8 GB for the three-arm ladder — against ~3.8 GB for the original single
1400-token lane. Exact commands are in the header of each config
(`configs/h15_poisoned_p{00,15,50}.yaml`); each dose needs three invocations — short arm, long arm
(same root, the manifest merges), and a 500-row **measurement-only deep val split** at
`--max-target-tokens 1400` in its own root (`data/materialized/h15-deepval-p<rate>`), which is never
listed in a config and therefore never trained on.

## Prep status (2026-07-28, still valid) — materializer BUILT and verified

`data/preprocessing/poisoned_text.py` is written and smoke-verified. Verified on 32 rows: schema
byte-identical to the fineweb-edu lane on all 10 shared columns (plus `poison_mask`, `poison_rate`,
`source_row_id`); image PNG-byte-identical to `render(target)` 32/32; target fully visible 32/32;
poison rate **0.1493** over alphabetic chars.

To get exact target↔image agreement the renderer's layout pass was extracted into a public
`wrap_text_pages()` (`render_utils.py`) and the target is now the *wrapped lines*, so whitespace is
normalized — this differs from the existing fineweb-edu lane, which supervises raw text the renderer
may reflow. At `--poison-rate 0.0` this is the only difference between the control arm and ordinary
fineweb-edu transcription, and it is deliberate: the two H15 arms must differ *only* in poisoning.

**Decisions taken at prep time, with their current status:**

- ~~**Use `--max-target-tokens 1400`, keep `max_length: 2048`.**~~ **REVERSED** — see §1. The
  retention argument was sound (2048 retains only 67.5% of uncapped rows, dropping the longest) but
  it optimised for depth of *supervision*, which is what H13 showed to be fatal. Superseded by the
  two-arm design at cap 64 / cap 200 with `max_length: 1024`.
- **Poisoning is uniform over all 26 letters, not rejection-sampled** — unchanged and still correct:
  the masked character is exactly uniform, so the `ln 26` / `1/26` bounds are exact. ~3.85% of masked
  positions land back on the original letter, so the invariant is "every **un**masked index is
  unchanged", not "every masked index differs".
- **The floor is not a pass/fail gate here** — unchanged, and now *stronger*: after H13 §7 the loose
  per-token floor is also **numerically wrong** at short targets (computed over T target tokens while
  T+2 are supervised: 6.06% too high at cap 48, 1.78% at cap 128, 0.30% at cap 1400). Even the
  "crossing it proves reading" direction is unsafe at the repaired lengths, because the bias runs
  toward false positives. No criterion touches it.
- `poison_mask` is **character**-indexed — a token-level probe must map through
  `return_offsets_mapping`, and a token straddling masked and unmasked characters is not purely
  prior-proof. **Now quantified:** only ~20% of poison-touching tokens are pure (§2).
- The extra columns make this lane's `Features` incompatible with `concatenate_datasets` against
  *other* lanes. Fine for single-lane H15 (the two length arms share identical Features and
  concatenate cleanly); a future mixture ([H19](H19-four-lane-rematch.md)) must drop them.

## Defects found while revising (2026-07-29)

1. **5.2% readable as decided** (§1) — the run would have collapsed; the lane is H13's d4 in disguise.
2. **No control arm.** The confirm bar's "currently ~+1 pt" reference came from a different
   checkpoint and mixture. Fixed by `configs/h15_poisoned_p00.yaml`.
3. **The confirm bar asked for something no checkpoint in this programme has produced**, including
   100%-prior-proof ones (§3).
4. **`Δperm ≥ +30%` is length-confounded** (H13 §4) — it would have been passed or failed by target
   length as much as by grounding.
5. **"prior wrong at every position" is a character-level claim used as a token-level one** (§2):
   32.3% of tokens at depth touched, 6.6% pure, at p = 0.15.
6. **`_fit_token_budget` silently drops rows whose first rendered line exceeds the cap**, at a
   **dose-dependent** rate (36% at p = 0.50 / cap 48). Avoided by staying at cap ≥ 64; worth a guard
   in the materializer if tighter caps are ever wanted (truncate mid-line, or log the drop rate).
7. **`floor.json`'s per-token floor divides by T, not T+2** — inherited from the `random_strings.py`
   defect H13 §7 found; measured here at 6.06% / 1.78% / 0.30% for caps 48 / 128 / 1400.
8. **Minor:** the prep note's "1.28 pages/row" holds only for the uncapped design; every capped
   variant measures **1.00 pages/row**, so `max_pages: 4` / `max_train_images: 4` are moot and the
   configs set 1.

## Relationship to H14

[H14](../in-progress/H14-masked-region-targets.md) and H15 attack the same constraint from different
directions — H14 makes the *target* mask-dependent, H15 makes the *source* unpredictable. H14
additionally trains localization; H15 preserves clean real-text statistics. They are not mutually
exclusive. See the dependency section above: H15 does **not** wait on H14's verdict, but its
long arm is contingent on it.

## Design lineage

"Make the prior wrong" is the principle behind
[VQA-CP-style debiasing](../../literature/blind-branch-subtraction.md); the character-poisoning
instantiation is ours, not a published result.

## Artifacts (CPU, produced during this revision)

- `scratchpad/h15_readable_fraction.py` → `data/eval/h15-readable-fraction.json` — readable fraction
  vs target-length cap, real tokenizer, 200 real rows.
- `scratchpad/h15_position_profile.py` → `data/eval/h15-position-profile{,-short,-long,-p50}.json` —
  per-position poisoned/pure token counts (criterion power) and the T-vs-T+2 floor correction.
- `configs/h15_poisoned_p{00,15,50}.yaml` — the three arms, with materialization commands in-header.
- `univi/trainer.py` — `"poisoned-text-long"` added to `VALID_SUBSETS` (one additive line).
