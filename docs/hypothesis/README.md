# Hypothesis Index

Status board for the whole programme. One file per hypothesis, grouped into
[`done/`](done) · [`in-progress/`](in-progress) · [`todo/`](todo).
Sync material changes up to [`research.md`](../../research.md).

**Conventions**
- Every hypothesis states a falsifiable claim and a **pre-registered** success/failure criterion
  *before* it runs.
- Criteria are written in **position-resolved reading gain — `pos0_gain`, `pos0_4_gain`,
  `char_bins`.** Never eval CE (see [H09](done/H09-balanced-mixture-prevents-drowning.md) for why CE is
  orthogonal to grounding), and never **K**, **D50**, or **aggregate Δperm/Δblank**. If a Δ statistic is
  unavoidable, **prefix-match it to the first 5–10 answer tokens** and state both the window and the
  target length it will be evaluated at.
  ⚠️ **This line used to read "Δperm / Δblank / reading gain — never eval CE", which banned eval CE
  while explicitly licensing *aggregate* Δperm — and that is the loophole all four mis-specified
  criteria came through** (tightened 2026-07-30; see
  [the defect table](#criterion-defects-found-by-running-the-criteria--audit-any-new-bar-against-these)).
- A criterion must **name its statistic's window and the target length it will be applied to.** All four
  defects would have been caught at pre-registration by that one requirement.
- A hypothesis moves `todo/` → `in-progress/` at launch and → `done/` at verdict. Update the status
  line inside the file, this index, and `research.md` together.
- When a hypothesis closes, record the verdict and the *learning*, not just the numbers.

---

## [done/](done)

| id | claim | verdict |
|---|---|---|
| [H01](done/H01-gemma-e2b-mixture-audio-drowns.md) | A full-Gemma mixture reads all four lanes | **PARTIAL** — 3 lanes read, audio at the ignore-floor |
| [H02](done/H02-render-destroys-audio-signal.md) | Spectrogram rendering destroys the audio signal | **REFUTED** — 0% WER loss through the PNG round-trip |
| [H03](done/H03-from-scratch-embedder-shallow-reading.md) | A from-scratch pixel embedder + pretrained Qwen learns to read | **PARTIAL** — uniformly shallow (~3% Δperm) |
| [H04](done/H04-frozen-decoder-warmup.md) | Freezing the decoder first lets the embedder catch up | **REFUTED** — content-blind soft-prompt collapse |
| [H05](done/H05-isolated-ocr-from-scratch.md) | Isolating a prior-proof OCR lane rescues the from-scratch embedder | **REFUTED** — grounding collapsed to exactly zero |
| [H06](done/H06-mlp-connector-escalation.md) | The connector's expressivity is the bottleneck | **REFUTED** — MLP changed nothing (+0.00 pts) |
| [H07](done/H07-pretrained-vision-adapter-qwen.md) | A *pretrained* vision front-end makes a reused Qwen decoder read | **VALIDATED** — Δperm +115%, gain +22 pts |
| [H08](done/H08-audio-needs-trainable-vision.md) | Audio reads only if the vision tower is trainable | **VALIDATED** — frozen flat, trainable Δperm +135% |
| [H09](done/H09-balanced-mixture-prevents-drowning.md) | Balanced lane shares stop audio drowning in the real mixture | **PARTIAL** — drowning fixed, reading shallow + saturated |
| [H10](done/H10-reading-concentrated-at-start.md) | Reading is spread evenly across the answer | **REFUTED** — +70 pts at position 0, ~+1 pt after position 10 |
| [H11](done/H11-position-decay-is-prior-induced.md) | The position decay is caused by the language prior | **REFUTED** — it persists on prior-proof lanes |
| [H12](done/H12-contrastive-decoding-probe.md) | Mid-page content is present but out-voted by the prior | **REFUTED** — no recovery at any α (1.10×/1.19× vs 2×) |
| [H13](done/H13-density-ladder-long-targets.md) | The readable ceiling is set by character density, not token position | **REFUTED** — readable depth ~5–12 answer tokens, invariant over a 16× density span (the "K = 48.5 tok" it originally reported was an estimator floor) |
| [H14](done/H14-masked-region-targets.md) | Mask-dependent targets force reading *and* localization | **CONFIRMS** — all 3 bars cleared; localizes masks 95.65% vs 0.86% visible-word accuracy |
| [H15](done/H15-prior-poisoned-text.md) | Making the prior wrong at every position restores deep reading | **PARTIAL** — the prior costs reading at positions **1–4 only**, monotone in dose (pos-1 gain +0.0→+13.3→+36.7 pts), flat past ~5. Buys ~4 tokens, then stops. Its VOID guard fired and is mis-sized |
| [H16](done/H16-prior-gap-weighted-loss.md) | Reweighting loss by the blind-branch gap restores deep reading | **REFUTED** — gain at tokens 10–100 ≈ 0 vs a ≥+10 pts bar; Δperm +0.687% vs ≥+30%. Abolished position-0 reading (+70.0 → +0.0) instead of deepening it |
| [H17](done/H17-raise-soft-token-budget.md) | 280 soft tokens is the page ceiling; 1120 lifts it | **VOID** — all 3 legs trained; reading *fell* 55.7→19.7→−0.1 pts as `grad_norm` collapsed 0.264→0.079→0.038. (A′) not refuted, (C) not confirmed |
| [H21](done/H21-vision-path-not-scale-invariant.md) | Reading survives a glyph-scale change it was not trained on | **REFUTED** — trained fonts read +52.7…+64.0 pts, held-out fonts +0.0/+0.7. Fails even by *interpolation* (font 18 sits between trained 14 and 24). Repairs H13 §6 and revives its line-demux branch |

## [in-progress/](in-progress)

*(empty)*

## [todo/](todo)

**Re-ranked 2026-07-30, after H15, H16, H17 and H21 all reported.** Three things changed the ranking:

1. **(B) is closed as an account of the positional ceiling.** [H15](done/H15-prior-poisoned-text.md)
   removed the language prior *by construction* and bought ~4 tokens of depth;
   [H16](done/H16-prior-gap-weighted-loss.md) attacked the same target with a loss weight and captured
   none of it. Two independent directions, same answer: the prior is real but small.
2. **The render-geometry lever is dead.** [H21](done/H21-vision-path-not-scale-invariant.md) shows
   reading is memorised *per glyph scale* and does not even interpolate.
3. **The live blocker is upstream of (A′)/(B)/(C): optimization.** `grad_norm` collapse below ~0.1 with
   Δperm ≈ 0 has now voided or degraded **four** runs (H13, H16, H17-560, H17-1120), and in H17 it
   scaled monotonically with the treatment. Until a raised-budget run is shown to *optimize*, no budget
   ladder is interpretable.

| id | claim | cost | separates |
|---|---|---|---|
| **(unwritten)** | **the `grad_norm` collapse itself** — 4 runs, and it tracks the soft-token budget | — | **upstream of (A′)/(B)/(C)** — needs a doc |
| [H18](todo/H18-no-learned-scan.md) | The model has no mechanism to track position in the image | — | (C) — **no longer resolves from H17**; needs its own design |
| [H20](todo/H20-audio-phoneme-resolution.md) | Audio needs ≤120 ms/token-column to transcribe | 9 h | (A) for audio — both legs **died on CUDA OOM 2026-07-30**; needs resizing against *measured* free VRAM (the card is shared) |
| [H19](todo/H19-four-lane-rematch.md) | Curriculum + anchors + weighted loss makes the real mixture read | ~5.8 h | integration — **blocked: H17's VOID left `max_soft_tokens`/`init_from` unresolved** |

### Criterion defects found by running the criteria — audit any new bar against these

Four criteria in this programme have now been found mis-specified in **one** way: *stated in a
statistic that is not comparable across the configurations it is applied to.* **Three of the four
would have published a wrong verdict** on data that reads clearly when taken position-resolved. Same
class as [H13 §7](done/H13-density-ladder-long-targets.md)'s floor bug.

| # | criterion | defect | what it did |
|---|---|---|---|
| 1 | H17: "K must rise ≥ 1.73× per leg" | **structural floor** — `k_from_buckets` needs the first 25-char bucket *past* a 100-char window, so `k_chars ≥ 100` always | returned an identical **48.4913 tokens at budgets 280 and 560** for rungs whose real reading differed **2.8×** |
| 2 | H17: "≥ +100% Δperm" sanity gate | **length dilution** — imported from [H07](done/H07-pretrained-vision-adapter-qwen.md)'s ~25-char lane, applied to a 479-char / 234-token target | a baseline reading at **+64.7 pts** (p = 1.7e-34) scored **+7.36%** and "failed" |
| 3 | H15: VOID guard over the **first 48 answer tokens** | **inherited the retired 48-token depth**; real reading lives at positions 0–4, so the window dilutes ~10× | declared the control arm "did not read" while it shows **+54.0 pts** at position 0 |
| 4 | H21: `trained_read` ANDs pos-0 ≥ 20 pts with **Δperm ≥ 30%** | **length dilution** — same H07 import, applied to a 119-char / ~60-token lane where H13 measured *established* reading at only **+14.31%** | self-reported `branch: VOID` on a perfect **~80×** trained-vs-held-out dissociation |

Also retired: **D50** (H15's token-unit K) is degenerate when depth is 0–1 bins — "poisoned D50 ≥ 2×
control D50" reduces to `≥ 2 × 0`. And H15's CONFIRMS/REFUTES windows were placed at bins 10–20 and
20–50, **2–10× deeper than any arm reads**, so the manipulation's real effect at positions 1–4 fell
outside every branch.

**Rules that follow.** State every bar in **`pos0_gain`, `pos0_4_gain`, or `char_bins`** — these are
neither floored nor diluted. If a Δperm bar is unavoidable, prefix-match it to the **first 5–10 tokens**
(**not 48** — H15's 48-token window is itself defect #3, so "prefix-matched, as H15 does" is *not* a safe
remedy) and anchor it to a lane of the **same target length**. Never state a criterion in K, D50, or
aggregate eval CE. And place the measurement window **where reading is expected**, not where it would
be nice to find it.

**The calibration table any Δ bar must be checked against.** All rows are the same H07 checkpoint that
unambiguously reads, on the H13 ladder, plus H21's best-trained font:

| lane | chars/page | supervised tok/row | aggregate Δperm | pos-0 gain |
|---|---|---|---|---|
| randstr-d1 (H07's own lane) | 29 | **17** | **+113.84%** | +62.7 |
| randstr-d2 | 119 | **60** | **+14.31%** | +35.3 |
| randstr-d3 | 479 | **234** | **+3.05%** | +28.7 |
| randstr-d4 | 1919 | **932** | **+0.63%** | +28.0 |
| H21 leg-1 font 14 (d2 geometry) | 119 | **60** | **25.80%** | **+57.3** |

**The largest aggregate Δperm ever recorded at T ≈ 60 supervised tokens is 25.80%, by a model reading at
+57.3 pts at position 0.** So a "≥ +30% Δperm" bar sits *above the observed ceiling* at T ≥ 60 and is
unreachable by construction at T ≥ 234. Two open docs still carry bars in that range — see
[todo/](#todo).

**Also retired: every "token-weighted readable fraction" figure in the programme.** They were all
computed against depth 48 and are overstated ~4–10×. Recompute at the measured ~5–12 tokens **and**
recompute the reference points (H13 died at 14.2%, H14 confirmed at 75.2%, H07 succeeded at 100%) at the
same depth, or the comparison is depth-inconsistent on both sides. Useful anchor:
[H21](done/H21-vision-path-not-scale-invariant.md)'s leg 1 trained **healthily** (`grad_norm` 0.65–0.86)
on a lane whose corrected readable fraction is ~20%, so ~20% is empirically survivable.

---

## The three candidate constraints

Shorthand used throughout:

- **(A) Representation / bandwidth** — the soft-token stream physically cannot carry the content.
- **(B) Objective / gradient allocation** — the information is there, but the language prior makes
  reading unprofitable, so it is never learned.
- **(C) Scan / cursor** — the information is there and reading *is* profitable, but the model has no
  learned way to track *where* in the image it currently is.

Current reading of the evidence: **(A) and (B) are coupled and both real**
([H11](done/H11-position-decay-is-prior-induced.md)); **(C)** is newly suspected and untested
([H18](todo/H18-no-learned-scan.md)).

[H12](done/H12-contrastive-decoding-probe.md) sharpens this: mid-page content cannot be recovered by
a **decoding-time** intervention, so it is not merely out-voted at argmax. That weighs against a
pure-(B) story and toward (A)/(C) — with the caveat that H12 only bounds decoding-time fixes, so a
*training-time* objective change could still put content into the stream that is not there today.

**That caveat is now closed, 2026-07-30.** Both training-time objective interventions have run.
[H15](done/H15-prior-poisoned-text.md) removed the prior *by construction* (uniform poisoning of the
source text) and moved reading at answer positions **1–4 only** — monotone in dose, +0.0 → +13.3 →
+36.7 pts at position 1, and verified not to be a headroom artifact — while every arm stayed flat past
position ~5. [H16](done/H16-prior-gap-weighted-loss.md) attacked the identical target with a per-token
loss weight and captured **none** of it: gain at tokens 10–100 ≈ 0 against a ≥ +10 pts bar, and it
*abolished* position-0 reading (+70.0 → +0.0). **So (B) is real but worth ~4 tokens.** It is closed as
an account of the positional ceiling — the only role it was still being kept alive for.

[H13](done/H13-density-ladder-long-targets.md) splits **(A)** in two and kills one half. Readable depth
sits at **~5–12 answer tokens across a 16× span of page density** (119 → 1919 chars/page), and
position-0 reading gain is statistically identical at d3 (+28.7) and d4 (+28.0) despite d4
carrying ~24 chars per inked soft token against d3's ~15. So:

- **(A) density-proportional bandwidth** — *excluded*. More characters per token does not degrade
  reading at the start of the page; it only shortens the fraction of a longer target that gets read.
- **(A′) fixed absolute capacity** — alive. The model extracts a fixed quantity of content per page
  regardless of how much is on it.
- **(C) fixed scan depth** — alive, and indistinguishable from (A′) in H13's data, because
  chars/token ≈ 2.06 on every rung makes character position and token index proportional *by
  construction*.

**(A′) and (C) differ in exactly one testable way:** raising the soft-token budget relieves a
capacity limit but not a scan limit. That was [H17](done/H17-raise-soft-token-budget.md) — and
**H17 came back VOID (2026-07-30)**, so the separation is still open.

H17 trained all three legs (280 / 560 / 1120, identical except `max_soft_tokens`, all warm-started
from the same H07 checkpoint) and reading *fell* monotonically with the budget — 0–10-char gain
+55.7 → +19.7 → −0.1 pts — while `grad_norm` fell 0.264 → 0.079 → 0.038 and the 1120 leg's eval CE
never even reached the no-reading floor. "More tokens don't help" and "a 280-native adapter cannot be
re-addressed onto a 33×33 grid in 500 steps" predict the same data, so **(A′) is not refuted and (C)
is not confirmed**. The one surviving hint leans (C): at 560, position-0 gain is *unchanged*
(+68.0 vs +64.7, CIs overlap) while the continuation collapses — first-fixation legibility survives
the budget change and place-keeping does not.

**The live blocker is now optimization, not representation.** `grad_norm < 0.1` with Δperm ≈ 0 has
appeared in H13, H16 and both raised-budget H17 legs. Any successor at a raised budget needs its own
recruitment schedule (longer adapter warmup, more steps, or a chained 280→560→1120 budget curriculum —
which H17's legs deliberately did *not* use), must abort on sustained `grad_norm < 0.10`, and must
probe Δperm at step 200 rather than trusting a smoothly-descending eval CE, which all three legs had
while reading went to zero.

### What (C) now has to explain — and one new lever

[H21](done/H21-vision-path-not-scale-invariant.md) adds a hard constraint that any account has to fit:
reading is **memorised per glyph scale** and does not transfer *even by interpolation* — trained fonts
{14, 24, 40} read at +52.7…+64.0 pts while held-out {18, 31} and interpolated {20, 28} sit at the ignore
floor. Whatever the model does at the start of a page is template-like, not a general perceptual
routine. Combined with [H14](done/H14-masked-region-targets.md)'s ~100× *where*-vs-*what* dissociation,
the picture is: coarse layout extracted reliably and scale-robustly, fine content read only at
memorised scales and only for ~5 tokens.

**The one new lever** comes from H21's secondary read-out. On H13's own d3/d5 splits — byte-identical
targets differing only in font — the scale-trained checkpoint reads **d5 (font 40) at +66.0** against
**d3 (font 14) at +55.3**, where [H07](done/H07-pretrained-vision-adapter-qwen.md) read **+0.0** on d5.
That repairs [H13](done/H13-density-ladder-long-targets.md) §6 (its d5 null was a *scale-generalisation*
failure, not a property of the manipulation) and **revives H13's untested "D5 ≫ D3 ⇒ line-demux"
branch**, which was unanswerable only because no model could read font 40. Note the limits: font 40 was
a *trained* font, so this is not generalisation, and +66.0 vs +55.3 is position-0 **amplitude** — depth
is ≈ 0 past chars 10–24 on every rung. **Scale buys amplitude, not depth.**
