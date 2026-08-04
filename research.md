# Univi — Research Status

Hub document. Current status, committed decisions, and everything we have established.
Kept short on purpose: details live in [`docs/hypothesis/`](docs/hypothesis/README.md) and
[`docs/literature/`](docs/literature/README.md), and this file is re-synced from the
hypothesis index whenever a hypothesis changes status.

- Glossary of metrics (Δperm, Δblank, Retention, Decodability Gate): [`CONTEXT.md`](CONTEXT.md)
- Original programme framing: [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md)
- Closed thread postmortem: [`docs/encoder-free-thread-postmortem.md`](docs/encoder-free-thread-postmortem.md)

Last synced: 2026-07-30 (**H15, H16, H17, H21 all closed** — H17 VOID, H16 + H21 REFUTED, H15 PARTIAL;
**(B) closed as an account of the positional ceiling**; the grad_norm-collapse pattern promoted to the
live blocker; four pre-registered criteria found mis-specified — see the caveat sections).

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
    ⚠️ **This holds; the *budget* extension of it does not** (2026-07-30). H17's 560 and 1120 legs
    did not merely fail to move the cliff — they **erased the amplitude too** (0–10 gain 55.7 → 19.7
    → −0.1) while `grad_norm` collapsed 0.26 → 0.08 → 0.04, so they measured an optimization failure
    rather than a budget. Only the 280 leg above is a usable data point.
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
15. **Reading is memorised per glyph scale — it does not even interpolate.** Two lanes with
    byte-identical targets differing only in font (14 vs 40) are read at +28.7 and **+0.0 pts**, and
    training on fonts {14, 24, 40} does **not** generalise: those three read at **+57.3 / +52.7 /
    +64.0 pts** while held-out {18, 31} sit at **+0.0 / +0.7** and the *interpolated* 20 and 28 sit at
    the floor too. **Font 18 lies between two trained fonts and reads nothing.** So whatever happens at
    the start of a page is template-like, not a general perceptual routine, and "just render it bigger"
    is not available as a fix. [H13](docs/hypothesis/done/H13-density-ladder-long-targets.md) §6,
    [H21](docs/hypothesis/done/H21-vision-path-not-scale-invariant.md) **REFUTED**
16. **The language prior costs about four answer tokens of reading, and nothing deeper.** Poisoning the
    source text before rendering moves reading at answer positions **1–4 only**, monotonically in dose
    (position-1 gain **+0.0 → +13.3 → +36.7 pts** at poison rate 0.00 / 0.15 / 0.50), with every arm
    flat past position ~5. Verified not to be a headroom artifact: the strong arm's *absolute* aligned
    accuracy at position 1 is **higher** than the control's (38.7% vs 27.3%) on a far higher-entropy
    target. [H15](docs/hypothesis/done/H15-prior-poisoned-text.md) **PARTIAL**
17. **A loss weight cannot buy even those four tokens.** Upweighting by the blind-branch gap
    (`w = 1 + β·max(0, CE_blank − CE_aligned)`) produced gain ≈ **0** at answer tokens 10–100 against a
    +10 pts bar, and *abolished* position-0 reading (**+70.0 → +0.0**) while receiving 4× the gradient
    share of the baseline. The weight is largest exactly where the model already reads, so it
    concentrates loss on near-zero-gradient tokens. **The prose variant — upweight where the prior
    *succeeds* — is the opposite intervention and remains untested.**
    [H16](docs/hypothesis/done/H16-prior-gap-weighted-loss.md) **REFUTED**
18. **Scale buys reading *amplitude*, not *depth*.** On identical targets a scale-trained checkpoint
    reads font 40 at **+66.0** vs font 14 at **+55.3** at position 0 — but depth is ≈ 0 past chars
    10–24 on every rung in the font sweep (+3.2 pts at best). Every lever found so far moves the height
    of the first-token spike and none moves the cliff.
    [H21](docs/hypothesis/done/H21-vision-path-not-scale-invariant.md)

## The two live constraints

Everything now points at two coupled limits, and the open programme is designed to separate them:

- **(A) Representation** — 280 soft tokens over a full page is **10.7 chars/soft-token** (vs 0.1
  for the prior-proof lane that reads), each 48px patch mixes **3.8 text lines**, and audio gets
  **244 ms/token-column** against 50–150 ms phonemes.
  Headroom exists: the processor supports up to **1120** soft tokens and the pretrained position
  table is `(1120, 2, 3840)`. [H17](docs/hypothesis/done/H17-raise-soft-token-budget.md)
  **Spending that headroom has now been tried once and it went backwards** (2026-07-30, H17 VOID):
  the plumbing is sound (256/529/1089 tokens emitted as predicted, no rows truncated) but training at
  560 and 1120 destroyed the reading the 280 baseline had. So the headroom remains *unspent*, not
  *disproved* — nothing yet shows what a converged 1089-token run would read.
  **Split in two on 2026-07-29** by [H13](docs/hypothesis/done/H13-density-ladder-long-targets.md):
  reading stops at **~5–12 answer tokens regardless of page density**, and position-0 gain is
  statistically identical at 479 and 1919 chars/page. So the *density-proportional* form of (A) is
  **excluded** — more chars per token does not degrade reading, it only shortens the readable
  fraction of a longer target. What survives is **(A′) fixed absolute capacity**: a fixed quantity
  extracted per page regardless of what is on it.
- **(B) Objective** — reading is purchased only where the language prior fails. At position 0 the
  prior is useless and the model captures 83% of the available gap; at position 400+ it captures
  ~4%. [H15](docs/hypothesis/done/H15-prior-poisoned-text.md),
  [H16](docs/hypothesis/done/H16-prior-gap-weighted-loss.md)
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

  **MEASURED AND CLOSED 2026-07-30.** The eyeball comparison above has been turned into a number, from
  two independent directions, and both agree: **(B) is real and worth about four answer tokens.**

  - **[H15](docs/hypothesis/done/H15-prior-poisoned-text.md) — PARTIAL.** Three arms
    (poison rate 0.00 / 0.15 / 0.50), all three optimizing (`grad_norm` 0.275 / 0.330 / 0.350).
    Removing the prior moves reading at **answer positions 1–4 and nowhere else**, monotonically in
    dose — position-1 gain **+0.0 → +13.3 → +36.7 pts**, positions 2–4 **−1.8 → +1.3 → +10.0** — and
    every arm is flat by position ~5. **Not a headroom artifact:** p50's *absolute* aligned accuracy at
    position 1 (38.7%) exceeds p00's (27.3%) on a far higher-entropy target, and normalised for
    remaining headroom the effect is +0.0% → +15.2% → +37.4%. The page confound runs *against* the
    result (the control carries 1.2 images/row vs p50's 1.0, i.e. more bandwidth and less depth).
  - **[H16](docs/hypothesis/done/H16-prior-gap-weighted-loss.md) — REFUTED.** The loss-weight version
    of the same intervention captured **none** of those four tokens: gain at answer tokens 10–100 came
    out at **≈ 0** against a **≥ +10 pts** bar and Δperm at **+0.687%** against **≥ +30%**, and it
    *abolished* position-0 reading (**+70.0 → +0.0** vs the H09 baseline) rather than deepening it. Its
    degeneracy guard passed, so this is a refutation and not a VOID. Caveat: no matched control was
    run (fineweb-only, 600 steps, weighting off), so the *mechanism* story is confounded even though
    the absolute bars are not.

  **So (B) is closed as an account of the positional ceiling** — the only role it was still being kept
  alive for. It is not dead as a phenomenon; it is small. Two follow-ups survive: H16's **prose
  variant** (upweight where the prior *succeeds*) is the opposite intervention and remains untested,
  and H16 still owes the matched control it never ran.

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
was [H17](docs/hypothesis/done/H17-raise-soft-token-budget.md), the critical-path experiment — and
**it came back VOID.** The separation is still open.

**H17 FINAL — VOID, the ladder measured its own training failure (2026-07-30).** All three legs
trained, 500 steps each, `randstr-d3`, effective batch 64, identical except `max_soft_tokens`, each
probed at its own training budget with `max_length 4096` (zero rows truncated). All three
warm-started from the **same** `hybrid-pretrained-randstr-v0/final` — they do **not** chain, which
retires the "560 inherited a 280-trained checkpoint" confound stated in the interim entry and
replaces it with a worse one.

| | budget 280 | budget 560 | budget 1120 |
|---|---|---|---|
| gain, chars 0–10 | **+55.7** ±1.6 | **+19.7** ±1.1 | **−0.1** ±0.4 |
| gain, chars 10–25 | +3.0 ±0.8 | −0.1 ±0.3 | −0.1 |
| gain, chars 25+ | ≈0 | ≈0 | ≈0 |
| position-0 gain | +64.7 ±3.9 | +68.0 ±3.8 | **−1.3 ±0.9 (ns)** |
| Δperm | +7.36% | +2.29% | **+0.006%** |
| final grad_norm | 0.26 | 0.08 | **0.04** |
| eval CE vs no-reading floor 5.6086 | 5.531 (below) | 5.606 (below) | **5.643 (ABOVE)** |

**Why VOID rather than a verdict.** Legs 560 and 1120 both miss the pre-registered step-400 hard bar
(Δperm ≥ +3.05%, the reading this same init already achieved), and H17's own gate says a 560 failure
"kills the whole hypothesis cheaply". Leg 1120 is a training collapse, not a capacity measurement:
grad_norm 0.038 is *below* H13's collapse signature, aligned is statistically indistinguishable from
blank (`frac_rows_aligned_gt_blank` 0.353, `sign_test_p` 0.573), and eval CE never even reached the
floor a non-reader hits by predicting the letter marginal. The failure is **monotone in the
treatment**, so "more tokens don't help" and "a 280-native adapter cannot be re-addressed onto a
33×33 grid in 500 steps" predict the same data. **(A′) is not refuted, (C) is not confirmed, and
[H18](docs/hypothesis/todo/H18-no-learned-scan.md) is not promoted.**

**The one surviving sub-finding leans (C):** at 560, position-0 gain is *unchanged* (+68.0 vs +64.7,
CIs overlap) while `pos0_4` collapses 58.3 → 21.3. First-fixation legibility survives the budget
change; place-keeping does not. A hint from a leg that failed its own bar, not evidence.

**The blocker moved upstream: optimization, not representation.** `grad_norm < 0.1` with Δperm ≈ 0 has
now voided or degraded **four** independent runs — H13, H16, H17-560, H17-1120 — and in H17 it scaled
with the budget. Until a raised-budget run is shown to *optimize*, no budget ladder is interpretable.
A successor needs its own recruitment schedule (longer adapter warmup, more steps, or a chained
280→560→1120 curriculum), must abort on sustained `grad_norm < 0.10`, and must probe Δperm at step 200
— **all three legs' eval CE descended smoothly while reading went to zero.**

## Next up

**Re-ranked 2026-07-30, after H15, H16, H17 and H21 all closed.** Five hypotheses reported in 24 h.
(B) is closed as an account of the ceiling, the render-geometry lever is dead, and the discriminator
between (A′) and (C) went VOID — so **the blocking problem is no longer which constraint binds, it is
that runs at raised budgets stop optimizing.** Full detail in the
[hypothesis index](docs/hypothesis/README.md).

| | hypothesis | cost | separates | status |
|---|---|---|---|---|
| 1 | **the `grad_norm` collapse itself** — `< 0.1` with Δperm ≈ 0 in H13, H16, H17-560, H17-1120, and it *tracks the soft-token budget* | — | **upstream of (A′)/(B)/(C)** | **the live blocker; no doc yet** |
| 2 | [H18 no learned scan](docs/hypothesis/todo/H18-no-learned-scan.md) | — | (A′) vs (C) | **no longer resolves from H17** — needs its own design or a repaired budget ladder |
| 3 | [H20 audio at phoneme resolution](docs/hypothesis/todo/H20-audio-phoneme-resolution.md) | 9 h | (A) for audio | **both legs died on CUDA OOM** 2026-07-30 (05:56Z / 06:00Z), checkpoint dirs empty. Needs resizing against *measured* free VRAM — the card is shared — plus a `grad_norm` abort, since it raises the budget |
| 4 | H16's **prose variant** — upweight where the prior *succeeds* (the opposite of the formula H16 ran) | ~6 h | (B), the untested half | flagged before H16 ran as "a distinct, untested intervention"; now the only live objective-side rung |
| 5 | H13's **line-demux branch**, revived by [H21](docs/hypothesis/done/H21-vision-path-not-scale-invariant.md) | ? | (A′) | newly *askable*: a scale-trained model reads d5 (font 40) at +66.0 vs d3 at +55.3 on identical targets, where H07 read +0.0 |
| 6 | [H19 four-lane rematch](docs/hypothesis/todo/H19-four-lane-rematch.md) | ~5.8 h | integration | **blocked** — 6 unfilled config fields whose values were to come from H17's verdict |

Execution is an unattended queue (`scratchpad/gpu_queue.sh`, `gpu_queue_stage2.sh`, and
`gpu_queue_stage3.sh` for the read-out probes), one job on the card at a time per the SOLO rule,
logging every step and every skip with a reason to `data/eval/gpu-queue.log`. All three stages have run
to completion; nothing is armed. **Stages 1–2 queue training only** — grounding probes are stage 3, so
a checkpoint that exists is not a measured checkpoint.

⚠️ **The A100 is shared and the neighbour grows.** A neighbour container held ~15.7 GB on 2026-07-29 and
**22.5 GB** by 2026-07-30 06:00Z. Both H20 legs died of CUDA OOM with our own processes at only 15.9 GB
and 17.5 GB — *inside* the 26 GB budget, but 22 + 17.5 > 39.49. **Size every run against
`nvidia-smi --query-gpu=memory.free` at launch, not against 40 GB or the budget**, and cap our own
process with `torch.cuda.set_per_process_memory_fraction` so we OOM before the neighbour does.
`_accumulate_blank_ce` in `train_pretrained.py` costs a second full forward pass and is where H20
leg 1 died — first lever to cut.

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

**Every K that a verdict was ever stated on returned the smallest value the estimator can emit.** Not
one of those measured anything.

`k_from_buckets` in `scratchpad/h13_analyze.py` defines K as *the first character bucket **past** the
0–`baseline_chars` window where reading gain drops below ½ the in-window gain*. With the
pre-registered `baseline_chars=100` and 25-char buckets, `k_bucket ≥ 100//25 = 4`, so
**`k_chars = k_bucket × 25 ≥ 100` is structurally guaranteed.** "K = 48.5 tokens" is that constant 100
divided by 2.06 chars/token — a unit conversion of the estimator's floor.

**Scoped precisely, 2026-07-30** (an earlier version of this section said "every K ever reported",
which is too strong — the audit checked every artifact):

| artifact | K per rung | floored? |
|---|---|---|
| `h13-poscontrol-h07.json` — **the artifact H13's verdict was stated on** | d2/d3/d4 = **100 chars** (48.84 / 48.49 / 48.47 tok) | **yes, all of them** |
| `h17-K-h17-d3-{280,560}.json` — **H17's pre-registered bar** | both **100 chars** (48.4913 tok, identical to 4 dp) | **yes** |
| `h17-K-h17-d3-1120.json` | `no_reading_ns` | n/a — nothing to floor |
| `h13-density-ladder.json` — the **VOID** trained run | d3 450, d4 1075, d5 450 chars | **no** |

So unfloored K values do exist — but only in the artifact from the run that **read nothing**
(`grad_norm` 0.061, Δperm ≈ 0 on all five rungs). **Where K was floored it measured the window; where
it was unfloored it measured noise. Neither case measured reading.** Two further tells: `K_alt_baseline25`
is *also* at its own floor (25 chars) on both H17 legs, so shrinking the window does not rescue the
estimator — which independently corroborates a readable depth below ~12 tokens. And the *same rung* d3
carries K = 450 chars (~218 tok) in one artifact and 100 chars (48.5 tok) in another: a 4.5× spread
across checkpoints, so K was never a stable property of anything.

**H17 then demonstrated the cost of this concretely (2026-07-30):** its pre-registered criterion was
*"each leg must give K a factor ≥ 1.73×"*, and K returned **48.4913 tokens at budget 280 and 48.4913
at budget 560** — identical to four decimals — for two rungs whose actual reading differed by
**2.8×** (0–10-char gain 55.7 vs 19.7). The bootstrap reported `ci_lo = ci_hi = 100.0`, agreeing with
itself because it was resolving to a floor rather than converging on a value. A criterion phrased in K
would have scored that leg "flat, factor 1.00×" — a *destroyed* leg reported as an unchanged one.
**Never state a criterion in K.** Use `char_bins` / `pos0_gain`.

### The general defect, and the four criteria it has now broken (2026-07-30)

K is one instance of a class: **a criterion stated in a statistic that is not comparable across the
configurations it is applied to.** Four pre-registered criteria have now been found broken this way, and
**three of the four would have published a wrong verdict** on data that reads clearly position-resolved.

| # | criterion | defect | what it did |
|---|---|---|---|
| 1 | H17: "K must rise ≥ 1.73× per leg" | structural floor (`k_chars ≥ 100` always) | identical **48.4913 tok at budgets 280 and 560** for rungs differing **2.8×** in real reading |
| 2 | H17: "≥ +100% Δperm" sanity gate | length dilution — H07's ~25-char anchor on a 479-char target | a baseline reading **+64.7 pts** (p = 1.7e-34) scored **+7.36%** and "failed" |
| 3 | H15: VOID guard over the **first 48 answer tokens** | inherited the retired 48-token depth; reading lives at positions 0–4, so ~10× dilution | declared the control "did not read" while it shows **+54.0 pts** at position 0 |
| 4 | H21: `trained_read` ANDs pos-0 ≥ 20 pts with **Δperm ≥ 30%** | same H07 import, on a 119-char / ~60-token lane where H13 measured *established* reading at only **+14.31%** | self-reported `branch: VOID` on a perfect **~80×** trained-vs-held-out dissociation |

Also retired: **D50** (H15's token-unit K) is degenerate when depth is 0–1 bins, so "poisoned D50 ≥ 2×
control D50" reduces to `≥ 2 × 0` and is satisfied by anything. And H15's CONFIRMS/REFUTES windows sat
at bins 10–20 and 20–50, **2–10× deeper than any arm reads**, so the manipulation's real effect at
positions 1–4 fell outside every branch — the decision procedure could not have returned the right
answer regardless of the data.

**The rules that follow.** State every bar in **`pos0_gain`, `pos0_4_gain`, or `char_bins`** — neither
floored nor diluted. If a Δperm bar is unavoidable, prefix-match it to the **first 5–10 tokens** (not 48)
and anchor it to a lane of the **same target length**. Never state a criterion in K, D50, or aggregate
eval CE. Place the measurement window **where reading is expected**, not where it would be nice to find
it. And note the common root: two of the four inherited the 48-token number *directly*, so any bar
written before 2026-07-29 that mentions 48 tokens, K, or a ~50-token depth is suspect by default.

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
