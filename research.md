# Univi — Research Status

Hub document. Current status, committed decisions, and everything we have established.
Kept short on purpose: details live in [`docs/hypothesis/`](docs/hypothesis/README.md) and
[`docs/literature/`](docs/literature/README.md), and this file is re-synced from the
hypothesis index whenever a hypothesis changes status.

- Glossary of metrics (Δperm, Δblank, Retention, Decodability Gate): [`CONTEXT.md`](CONTEXT.md)
- Original programme framing: [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md)
- Closed thread postmortem: [`docs/encoder-free-thread-postmortem.md`](docs/encoder-free-thread-postmortem.md)

Last synced: 2026-07-29 (H13, H14 closed; K correction applied; H17 interim).

**If you are picking this up cold, read in this order:** [Thesis](#thesis) →
[What is established](#what-is-established) → [The two live constraints](#the-two-live-constraints)
→ [Next up](#next-up). The two *Methodological caveat* sections at the bottom record measurement
defects that produced wrong numbers; check them before quoting any figure.

## Thesis

Render text as images, render audio as log-mel spectrogram images, pass natural images
through unchanged — so a single model consumes **all answer-bearing input through the
vision path**. The question is whether a model can read pixels instead of consuming native
text/audio tokens.

## Committed architecture (decided 2026-07-28)

**Pretrained vision embedder + reused pretrained Qwen3-1.7B decoder.**

```
Gemma-4-12B unified vision embedder   (PRETRAINED, patchify+linear, 280 soft tokens, 3840-d)
  -> Linear(3840 -> 2048)             (from-scratch adapter)
  -> Qwen3-1.7B decoder               (PRETRAINED)
```

Code: `univi/hybrid/pretrained.py`, `univi/hybrid/train_pretrained.py`.
Rationale: this is the only configuration that has ever produced strong, prior-proof
reading ([H07](docs/hypothesis/done/H07-pretrained-vision-adapter-qwen.md),
[H08](docs/hypothesis/done/H08-audio-needs-trainable-vision.md)). Every from-scratch vision
front-end collapsed to ignoring the image
([H03](docs/hypothesis/done/H03-from-scratch-embedder-shallow-reading.md)–[H06](docs/hypothesis/done/H06-mlp-connector-escalation.md)).

## What is established

1. **Rendering is not the bottleneck.** A log-mel PNG round-trip costs a strong reader 0% WER.
   [H02](docs/hypothesis/done/H02-render-destroys-audio-signal.md)
2. **A from-scratch vision front-end cannot be recruited**, at any connector capacity, with or
   without decoder freezing. [H03](docs/hypothesis/done/H03-from-scratch-embedder-shallow-reading.md),
   [H04](docs/hypothesis/done/H04-frozen-decoder-warmup.md),
   [H05](docs/hypothesis/done/H05-isolated-ocr-from-scratch.md),
   [H06](docs/hypothesis/done/H06-mlp-connector-escalation.md)
3. **A pretrained vision front-end makes a *different* pretrained decoder read** — text (Δperm
   +115%) and audio (Δperm +135%), both prior-proof.
   [H07](docs/hypothesis/done/H07-pretrained-vision-adapter-qwen.md),
   [H08](docs/hypothesis/done/H08-audio-needs-trainable-vision.md)
4. **Text/image reading transfers with vision frozen; audio requires training the vision tower**
   (spectrograms are out-of-distribution for Gemma vision).
   [H08](docs/hypothesis/done/H08-audio-needs-trainable-vision.md)
5. **Balanced lane shares fix audio drowning** — librispeech went from the ignore-floor (+0.16%)
   to the *best* lane (+12.9%). [H09](docs/hypothesis/done/H09-balanced-mixture-prevents-drowning.md)
6. **But on real lanes, reading is shallow and saturates.** All four lanes read (Δperm +0.7% to
   +12.9%), grounding did not grow between step 2500 and 4000, and the decodes show fluent
   prior-driven fabrication. [H09](docs/hypothesis/done/H09-balanced-mixture-prevents-drowning.md)
7. **Reading is concentrated at the START of the image.** fineweb reading gain is **+70 points at
   answer position 0** and ~+1 point beyond position 10.
   [H10](docs/hypothesis/done/H10-reading-concentrated-at-start.md)
8. **That decay is NOT purely a language-prior artifact** — it also appears on prior-proof lanes
   where the prior is useless, at ~100× lower character density.
   [H11](docs/hypothesis/done/H11-position-decay-is-prior-induced.md)
9. **Per-lane eval CE is nearly orthogonal to grounding.** Our fineweb 2.470 ≈ encoder-free H2's
   2.482, and H2 was ablation-proven to read zero pixels. Never use eval loss as a grounding
   metric; use Δperm / Δblank / reading gain.
10. **Mid-page content is not merely out-voted at argmax.** PMI contrastive decoding
    (`(1+α)·aligned − α·blank`) recovers nothing: overlap at answer tokens 10–100 rises only
    1.10×/1.19× against a pre-registered 2×, and ~95% of raw overlap is generic function words a
    *random* target earns equally. The apparent lift is the α=0 baseline's phrase-level looping
    being broken into fluent hallucination, not content.
    [H12](docs/hypothesis/done/H12-contrastive-decoding-probe.md)
11. **The readable depth is ~5–12 answer tokens, and it is the same with or without a language
    prior.** On the prior-proof `randstr-d3` lane the reading gain is **+18.6 pts over characters
    0–10**, +0 over 10–25, and **≈0 beyond character 25** (≈12 tokens at 2.06 chars/token). On real
    text, [H10](docs/hypothesis/done/H10-reading-concentrated-at-start.md) independently measured
    **+70 pts at answer position 0 and ~+1 pt beyond position 10**. Two different lanes, two
    different metrics, same shape and roughly the same cliff.
    ⚠️ **Corrected 2026-07-29.** This was previously recorded throughout the repo as
    **"K ≈ 48.5 tokens"**. That figure was an artifact of the K estimator, not a measurement — see
    [the caveat below](#methodological-caveat--k-was-an-estimator-floor-not-a-measurement).
    [H13](docs/hypothesis/done/H13-density-ladder-long-targets.md)
12. **More training buys reading *amplitude*, not reading *depth*.** H17's leg-280 warm-started from
    [H07](docs/hypothesis/done/H07-pretrained-vision-adapter-qwen.md) and trained 500 further steps
    on `randstr-d3` alone. Gain over chars 0–10 nearly tripled, **+18.6 → +55.7 pts** — and gain
    beyond char 25 stayed at **zero**. The cliff did not move. Whatever sets the depth is not
    relieved by more gradient on the same lane.
13. **Depth does not scale with page density** — the density-proportional form of (A) is dead.
    Across a **16× density span** (119 → 1919 chars/page) position-0 gain falls only 35.3 → 28.0 pts
    and is statistically identical at the top two rungs (479 chars +28.7±3.7 vs 1919 chars
    +28.0±3.7). Density-proportional bandwidth predicts a 16× reduction; the observed factor is
    2–3×. More characters per soft token does not degrade reading — it only shrinks the readable
    *fraction* of a longer target.
    [H13](docs/hypothesis/done/H13-density-ladder-long-targets.md) **REFUTED**
14. **The model extracts coarse layout reliably and fine content barely at all** — a ~100×
    *where*-vs-*what* dissociation. With page regions occluded it localizes the occluded span at
    **95.65%**, and changes its output on **100%** of rows when only the mask position moves (which
    no mask-independent strategy can do), while transcribing plainly visible words at **0.86%**.
    [H14](docs/hypothesis/done/H14-masked-region-targets.md) **CONFIRMS**
15. **The vision path is not scale-invariant.** Two lanes with byte-identical targets differing only
    in font (14 vs 40) are read at +28.7 pts and **+0.0 pts** respectively. "Just render it bigger"
    is not available as a fix. [H13](docs/hypothesis/done/H13-density-ladder-long-targets.md),
    now [H21](docs/hypothesis/todo/H21-vision-path-not-scale-invariant.md)

## The two live constraints

Everything now points at two coupled limits, and the open programme is designed to separate them:

- **(A) Representation** — 280 soft tokens over a full page is **10.7 chars/soft-token** (vs 0.1
  for the prior-proof lane that reads), each 48px patch mixes **3.8 text lines**, and audio gets
  **244 ms/token-column** against 50–150 ms phonemes.
  Headroom exists: the processor supports up to **1120** soft tokens and the pretrained position
  table is `(1120, 2, 3840)`. [H17](docs/hypothesis/todo/H17-raise-soft-token-budget.md)
  **Split in two on 2026-07-29** by [H13](docs/hypothesis/done/H13-density-ladder-long-targets.md):
  reading stops at **~5–12 answer tokens regardless of page density**, and position-0 gain is
  statistically identical at 479 and 1919 chars/page. So the *density-proportional* form of (A) is
  **excluded** — more chars per token does not degrade reading, it only shortens the readable
  fraction of a longer target. What survives is **(A′) fixed absolute capacity**: a fixed quantity
  extracted per page regardless of what is on it.
- **(B) Objective** — reading is purchased only where the language prior fails. At position 0 the
  prior is useless and the model captures 83% of the available gap; at position 400+ it captures
  ~4%. [H15](docs/hypothesis/todo/H15-prior-poisoned-text.md),
  [H16](docs/hypothesis/todo/H16-prior-gap-weighted-loss.md)
  **Weakened 2026-07-28** by [H12](docs/hypothesis/done/H12-contrastive-decoding-probe.md): if the
  content were present-but-suppressed, contrastive decoding would surface it, and it does not. (B)
  survives only in its *training-time* form — an objective change that puts content into the stream
  — not as a decoding-time suppression story.
  **Bounded 2026-07-29, then bounded much harder the same day.** The `random-strings` lane **is** the
  p→1 limit of "make the prior wrong at every position" — the prior is wrong there *by construction*.
  So whatever depth that lane reaches is the ceiling on what any objective-side intervention can buy.
  - The *first* version of this bound used **K = 48.5 tokens** for the prior-proof lane against
    ~10 tokens for real text, implying (B) had ~4.8× of headroom to chase.
  - **That 48.5 was an estimator floor, not a measurement** (see caveat below). The corrected
    prior-proof depth is **~5–12 tokens** — which is the *same* depth
    [H10](docs/hypothesis/done/H10-reading-concentrated-at-start.md) measured on ordinary English
    text, where the prior is fully available.

  **So removing the language prior entirely appears to buy approximately nothing.** The headroom (B)
  was supposed to unlock has largely closed, and the depth limit looks like a property of (A′)/(C).
  *Hedge, and it matters:* the two depths come from different lanes with different metrics and
  different scales, so "approximately equal" is an eyeball comparison, not a measured equality —
  which is exactly what [H15](docs/hypothesis/todo/H15-prior-poisoned-text.md) exists to turn into a
  number. It remains **the gate on whether
  [H16](docs/hypothesis/todo/H16-prior-gap-weighted-loss.md) is worth 6 h**, but it is now a gate
  expected to *close*, and it should be read as a cheap confirmation of a near-refutation rather than
  a search for headroom.

A third mechanism is now suspected and untested: **(C) no learned scan** — the model may lack any
way to track *where* it is in the image as it generates.
[H18](docs/hypothesis/todo/H18-no-learned-scan.md)
**First direct evidence against (C), 2026-07-29:**
[H14](docs/hypothesis/done/H14-masked-region-targets.md) found a ~100× dissociation between *where*
and *what* — the model **localizes occluded spans at 95.65%** (and changes its output on **100%** of
rows when only the mask moves, which no mask-independent strategy can do) while transcribing
**visible words at 0.86%**. Coarse layout is extracted reliably; fine content is not. That is the
signature of a **resolution/capacity** limit, not of a missing cursor. Hedge: a black rectangle is a
*salient* target, so "finds a box" is weaker than "knows it is at character 300 of running text" —
and it is one 400-step run, not a budget manipulation.

**After [H13](docs/hypothesis/done/H13-density-ladder-long-targets.md), (A′) and (C) are the two
survivors and they are indistinguishable in H13's data** — chars/token ≈ 2.06 on every rung, so
character position and token index are proportional *by construction*. They differ in exactly one
testable way: **raising the soft-token budget relieves a capacity limit but not a scan limit.** That
is what makes [H17](docs/hypothesis/todo/H17-raise-soft-token-budget.md) the critical-path
experiment rather than a contingent fix.

**H17 interim — 2 of 3 legs, budget is not buying depth (2026-07-29).** Matched 500-step runs on
`randstr-d3`, same warm start, effective batch 64, probed at their own training budget:

| | budget 280 | budget 560 |
|---|---|---|
| gain, chars 0–10 | **+55.7** ±1.6 | **+19.7** ±1.1 |
| gain, chars 10–25 | +3.0 ±0.8 | −0.1 ±0.3 |
| gain, chars 25+ | ≈0 | ≈0 |
| position-0 gain | +64.7 ±3.9 | +68.0 ±3.8 |
| Δperm | +7.36% | +2.29% |
| final grad_norm | 0.25 | 0.08 |

Doubling the budget left position 0 intact and **collapsed everything behind it**; depth past 25
chars was zero either way. **Do not read this as a verdict yet** — the 1120 leg is the pre-registered
4× point and is still running, and the 560 leg carries a confound I cannot remove: it warm-started
from a 280-trained checkpoint and had to re-learn a 23×23 positional grid in 500 steps, with a 3×
lower grad_norm. "Undertrained into a new positional regime" is still live against "budget does not
help".

## Next up

Ordered by information per GPU-hour, **re-ranked 2026-07-29 after
[H13](docs/hypothesis/done/H13-density-ladder-long-targets.md) excluded density-proportional
bandwidth**. H17 is promoted from "the fix if (A)" to **the discriminator between (A′) and (C)**.
Full detail in the [hypothesis index](docs/hypothesis/README.md).

| | hypothesis | cost | separates | queue status |
|---|---|---|---|---|
| 1 | [H17 raise soft-token budget](docs/hypothesis/todo/H17-raise-soft-token-budget.md) | ~26 h full | **(A′) capacity vs (C) scan** | **legs 280 + 560 DONE, leg 1120 RUNNING** |
| 2 | [H15 prior-poisoned text](docs/hypothesis/todo/H15-prior-poisoned-text.md) — H16's gate, now expected to *close* | 4 h | (B) at every position | queued (stage 1, 3 doses) |
| 3 | [H16 prior-gap-weighted loss](docs/hypothesis/todo/H16-prior-gap-weighted-loss.md) | 6 h | (B) at the objective | queued (stage 2) — **skip if H15 closes** |
| 4 | [H21 vision path is not scale-invariant](docs/hypothesis/todo/H21-vision-path-not-scale-invariant.md) | 3 h | render geometry | queued (stage 2) |
| 5 | [H20 audio at phoneme resolution](docs/hypothesis/todo/H20-audio-phoneme-resolution.md) | 9 h | (A) for audio | queued (stage 2, 2 legs) |
| 6 | [H18 no learned scan](docs/hypothesis/todo/H18-no-learned-scan.md) | 0 h | (A′) vs (C) | **resolves from H17** — no separate run |
| 7 | [H19 four-lane rematch](docs/hypothesis/todo/H19-four-lane-rematch.md) | ~5.8 h | integration | **blocked** — 6 unfilled config fields, and its data build is downstream of the budget verdict |

Execution is an unattended two-stage queue (`scratchpad/gpu_queue.sh`, `gpu_queue_stage2.sh`), one
job on the card at a time per the SOLO rule, logging every step and every skip with a reason to
`data/eval/gpu-queue.log`.

## Two rules that came out of H13 and bind every future run

1. **Balancing rows does not balance gradient.** Loss is normalised per supervised token, so a
   mixture's gradient share follows *target length*, not row count. H13 balanced 20k rows per rung
   and still sent **63% of the gradient to one rung** and **1.1% to the only rung that could be
   read**; 85.8% of its supervised tokens lay beyond the model's reach, every one of them teaching
   "emit the marginal, ignore the image". The run collapsed by step 110. **Before launching, compute
   the token-weighted fraction of supervised tokens that is actually within reading depth.** H13
   died at 14%; [H07](docs/hypothesis/done/H07-pretrained-vision-adapter-qwen.md) succeeded at 100%.
   ⚠️ **Those percentages were computed against the wrong depth.** Every readable-fraction figure in
   this repo — H13 14%, H07 100%, H14 75.2%, H15 75.4%, H19 50.5% — used the bogus 48-token depth.
   At the corrected ~5–12 tokens they are all **overstated, roughly 4–10×**. The gate's *ordering*
   was right, which is why it still worked as a design tool (it correctly ranked H07 safest and H13
   most dangerous, and correctly flagged H19). The absolute numbers were not. **Recompute before
   quoting, and re-check any queued run whose target cap was justified by one of them** — in
   particular H15's 64-token cap, which scored 75.4% under the old depth and lands nearer 10–20%
   under the corrected one, i.e. close to H13's death configuration rather than H07's success.
2. **"CE below the analytic floor ⇒ reading" is unsafe.** The floor must be computed over *exactly*
   the masked token set. `floor.json` divided by target tokens while response-only masking also
   supervises the deterministic `<|im_end|>\n` trailer (2 tokens, ~0 nats), inflating the floor by
   ~`T/(T−2)` — wrong-by-most on short targets. It made a rung that read **nothing** look 0.586 nats
   "below floor". Regenerate `floor.json` for any lane built by `random_strings.py` before quoting
   it. This is why criteria in this repo are stated in **Δperm / Δblank / reading gain and never eval
   CE** — H13's verdict is ablation-based and survived the bug untouched.

New seam for (1): **`init_from`** in `univi/hybrid/train_pretrained.py` warm-starts a run from a
checkpoint's weights (fresh optimizer and LR schedule), so a run tests its own variable instead of
re-running the bootstrap lottery. Verified round-tripping — 322 tensors; Qwen3 ties `lm_head` to
`embed_tokens` so that one key is never serialized and is skipped by design.

## Methodological caveat — K was an estimator floor, not a measurement (found 2026-07-29)

**Every K ever reported in this repo returned the smallest value the estimator can emit.** Not one of
them measured anything.

`k_from_buckets` in `scratchpad/h13_analyze.py` defines K as *the first character bucket **past** the
0–`baseline_chars` window where reading gain drops below ½ the in-window gain*. With the
pre-registered `baseline_chars=100` and 25-char buckets, `k_bucket ≥ 100//25 = 4`, so
**`k_chars = k_bucket × 25 ≥ 100` is structurally guaranteed.** H13's four rungs and both finished
H17 legs all returned exactly 100. "K = 48.5 tokens" is that constant 100 divided by 2.06
chars/token — a unit conversion of the estimator's floor.

The defect is that the baseline window was **wider than the effect it was normalising against**. Real
reading lives in chars 0–10, entirely *inside* the 0–100 baseline window, so the window average is
dominated by ~90 chars of dead zone and "half of baseline" is crossed at the very first bucket past
it, always. `K_alt_baseline25` reproduces this exactly one bucket lower (K=25 with a 25-char window).

The true curve was in the data the whole time, in `char_bins`, which splits the baseline window:

| rung | chars/page | pos-0 | 0–10 | 10–25 | 25–50 | 50+ |
|---|---|---|---|---|---|---|
| randstr-d1 | 29 | +62.7 | +45.9 | +2.2 | −0.7 | — |
| randstr-d2 | 119 | +35.3 | +27.5 | +2.8 | +0.2 | ≈0 |
| randstr-d3 | 479 | +28.7 | +18.6 | −0.3 | +0.1 | ≈0 |
| randstr-d4 | 1919 | +28.0 | +15.6 | +1.2 | +0.1 | ≈0 |
| randstr-d5 | 479 (font 40) | +0.0 | +0.2 | ≈0 | ≈0 | ≈0 |

**What survives:** [H13](docs/hypothesis/done/H13-density-ladder-long-targets.md)'s REFUTED verdict.
It rested on position-0 gain at d3 (+28.7±3.7) vs d4 (+28.0±3.7), which K never entered. The
aggregate Δperm collapse and the d5 font result are likewise untouched.

**What does not:** the claim that K is *invariant across density*. Four rungs all pinned to an
estimator floor is not evidence of invariance. The honest version is item 13 above — early reading
*does* decline with density (45.9 → 15.6 over 16×), just far sub-proportionally, flattening between
d3 and d4.

**Rule going forward:** a depth statistic whose baseline window is wider than the effect it measures
is censored from below. Report the **`char_bins` curve**, not a scalar K; if a scalar is needed, its
baseline window must sit strictly *inside* the region where gain is still significantly positive.

## Methodological caveat — the "blank" control was dark red on RGB lanes (found 2026-07-28)

Every probe built its blank as `Image.new(mode, size, 128)`. **PIL reads a bare int in a multi-band
mode as the first band only**, so an RGB source got **(128, 0, 0) — dark red**, not gray. Verified
directly. Lane modes: `fineweb-edu` **L**, `smoltalk` **L**, `random-strings` **L**,
`spoken-digits` **L**, `randstr-d1…d5` **L** — but **`densefusion` RGB** and **`librispeech` RGB**
(spectrograms are stored RGB).

- **Unaffected** (mode L ⇒ the blank really was gray): the headline prior-proof results —
  [H07](docs/hypothesis/done/H07-pretrained-vision-adapter-qwen.md) randstr +115%,
  [H08](docs/hypothesis/done/H08-audio-needs-trainable-vision.md) spoken-digits +135% — plus every
  fineweb/smoltalk number and all of [H13](docs/hypothesis/done/H13-density-ladder-long-targets.md).
- **Affected** (blank was dark red): all `densefusion` and `librispeech` Δblank / reading-gain
  figures — [H01](docs/hypothesis/done/H01-gemma-e2b-mixture-audio-drowns.md),
  [H09](docs/hypothesis/done/H09-balanced-mixture-prevents-drowning.md),
  [H10](docs/hypothesis/done/H10-reading-concentrated-at-start.md),
  [H11](docs/hypothesis/done/H11-position-decay-is-prior-induced.md), and
  [H12](docs/hypothesis/done/H12-contrastive-decoding-probe.md)'s librispeech branch.
- **How to read them:** a dark-red constant is still content-free, so the *direction* of each result
  stands. But it is further out of distribution than mid-gray, which should if anything *inflate*
  Δblank — which makes librispeech's persistently **negative** Δblank (blank beating the real
  spectrogram) a stronger finding, not a weaker one. Magnitudes are **not comparable across L and
  RGB lanes**, and any RGB-lane figure should be re-measured before it is quoted quantitatively.
- **Fixed:** all five probes now call `univi.hybrid.data.blank_like()`, which emits true gray in
  both modes. `scratchpad/h3_ablation.py` was always correct (it passed the `(128,128,128)` tuple).

## Standing rules

- Training runs go out **SOLO** (a concurrent GPU probe caused a host-RAM OOM).
- Every run pre-registers a success/failure criterion in Δperm / Δblank / reading gain.
- Every run ships a **live Δblank monitor**; eval CE alone is not evidence of grounding.
- Do not re-propose anything in the DONE-and-refuted list without new reasoning.
