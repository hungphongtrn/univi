# Hypothesis Index

Status board for the whole programme. One file per hypothesis, grouped into
[`done/`](done) · [`in-progress/`](in-progress) · [`todo/`](todo).
Sync material changes up to [`research.md`](../../research.md).

**Conventions**
- Every hypothesis states a falsifiable claim and a **pre-registered** success/failure criterion
  *before* it runs.
- Criteria are written in Δperm / Δblank / reading gain — never eval CE (see
  [H09](done/H09-balanced-mixture-prevents-drowning.md) for why CE is orthogonal to grounding).
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
| [H13](done/H13-density-ladder-long-targets.md) | The readable ceiling is set by character density, not token position | **REFUTED** — K = 48.8/48.5/48.5 tok over a 16× density span |

## [in-progress/](in-progress)

| id | claim | launched |
|---|---|---|
| [H14](in-progress/H14-masked-region-targets.md) | Mask-dependent targets force reading *and* localization | 2026-07-29 (readable-density variant) |

## [todo/](todo)

Ranked by information per GPU-hour. **Re-ranked 2026-07-29 by
[H13](done/H13-density-ladder-long-targets.md)**: K is invariant to page density, which excludes
*density-proportional* bandwidth and leaves **(C) fixed scan depth** vs **(A′) fixed absolute
capacity**. H17 is the only experiment that separates those two, so it stays first — but its job has
changed from "the fix if (A)" to "the discriminator".

| id | claim | cost | separates |
|---|---|---|---|
| [H17](todo/H17-raise-soft-token-budget.md) | 280 soft tokens is the page ceiling; 1120 lifts it | ~12 h to kill at leg 1, ~26 h full | **(C) vs (A′)** |
| [H15](todo/H15-prior-poisoned-text.md) | **H16's gate** — measures the *prior-attributable share* of the 10→48-token depth gap (strong form pre-refuted: `random-strings` IS the p→1 limit and caps at 48) | 4 h | (B) |
| [H16](todo/H16-prior-gap-weighted-loss.md) | Reweighting loss by the blind-branch gap restores deep reading | 6 h | (B) |
| [H18](todo/H18-no-learned-scan.md) | The model has no mechanism to track position in the image | — | (C) |
| [H20](todo/H20-audio-phoneme-resolution.md) | Audio needs ≤120 ms/token-column to transcribe | 9 h | (A) for audio |
| [H19](todo/H19-four-lane-rematch.md) | Curriculum + anchors + weighted loss makes the real mixture read | 16 h | integration |
| [H21](todo/H21-vision-path-not-scale-invariant.md) | Reading survives a glyph-scale change it was not trained on | 3 h | render geometry |

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
*training-time* objective change (H15/H16) could still put content into the stream that is not
there today.

[H13](done/H13-density-ladder-long-targets.md) splits **(A)** in two and kills one half. The
collapse point sits at **token ~48.5 across a 16× span of page density** (119 → 1919 chars/page),
and position-0 reading gain is statistically identical at d3 (+28.7) and d4 (+28.0) despite d4
carrying ~24 chars per inked soft token against d3's ~15. So:

- **(A) density-proportional bandwidth** — *excluded*. More characters per token does not degrade
  reading at the start of the page; it only shortens the fraction of a longer target that gets read.
- **(A′) fixed absolute capacity** — alive. The model extracts a fixed quantity of content per page
  regardless of how much is on it.
- **(C) fixed scan depth** — alive, and indistinguishable from (A′) in H13's data, because
  chars/token ≈ 2.06 on every rung makes character position and token index proportional *by
  construction*.

**(A′) and (C) differ in exactly one testable way:** raising the soft-token budget relieves a
capacity limit but not a scan limit. That is [H17](todo/H17-raise-soft-token-budget.md), which is why
it now heads the queue.
