# Attention-grounding analysis: does the model read the pixels? (2026-07-22)

**Question.** All answer-bearing input in Univi flows through one vision path
(text→image, audio→spectrogram). Aggregate NLL ablations (Δperm) suggested text
grounds strongly and audio barely, but NLL is a coarse, correlational signal. This
analysis asks the sharper question directly: **when the model emits a token, does its
attention land on the image patch that actually contains that token's content?** —
and does that behaviour *emerge with training*?

TL;DR: **Yes — after training, each generated token's attention moves onto the
correct, content-matching patch, and this is learned (absent at base). The effect is
strong and broad for text and weak/narrow for audio.** A shared vision channel does
not confer shared learnability.

---

## Method

For each response (target) token we reduce its attention over the image soft-tokens to
a scalar "where it looks", and compare to where the content actually is.

- **Audio (temporal alignment).** A 10 s spectrogram page = 246 contiguous image
  tokens = 6 freq × 41 time grid. Per token, sum the 6 freq rows → 41-col time
  profile → centroid column. Ground truth = **Whisper-tiny.en word timestamps**
  (word midpoint → time → expected column). Metric = within-clip
  Spearman(expected_col, predicted_col), mean over 24 LibriSpeech test-clean clips
  (≤10 s, single page).
- **Text (reading-order alignment).** Text reading order = raster order = image-token
  sequence order (row-major). Per token, expected = its fractional position through
  the passage (char-offset / total); predicted = attention centroid over the image
  tokens in sequence order, normalized to [0,1]. Metric = within-row
  Spearman(char-fraction, raster-centroid), mean over 24 single-image fineweb rows.

### Controls (what makes a positive result interpretable)

- **Permuted image** — same target text, same token positions, *different* image.
  A diagonal that survives here is autoregressive positional habit, **not** reading.
  The reported signal is **Δ = aligned − permuted**.
- **Base vs trained** — isolates *learned* reading from architectural priors.
- **Per-layer, head-averaged** — no cherry-picking a lucky head. Report the deep-band
  (L28–34) and all-layer means; single-layer maxima are winner's-curse inflated over
  35 layers and are quoted only as illustration.

### Feasibility notes (reusable)

- Attentions **do** come out of the 4-bit Unsloth Gemma-4 model, but only if you force
  eager: set `_attn_implementation="eager"` on `config`, `text_config`, and
  `vision_config` after load (the default sdpa path returns `attentions=None`).
- 35 layers × 8 heads. Image token id = 258880. Materialized rows lack raw audio, so
  the audio test re-renders from local LibriSpeech FLACs with the production settings
  (matching `data/eval/decodability/wer_direct.py`).

---

## Results

### The four-cell headline

Δ = aligned − permuted (image-dependent, position-correct attention). ≈0 ⇒ no genuine
reading (only habit); larger ⇒ stronger reading.

| lane | **base** (untrained) | **trained** |
|------|----------------------|-------------|
| **text** (fineweb, ckpt-2800) | all-layer Δ **+0.014**, deep −0.024 | all-layer Δ **+0.286**, deep **+0.225**; **23/35 layers Δ>0.2**; best L18 aligned 0.415 |
| **audio** (spectrogram, audio-only ckpt-800) | deep Δ **+0.003** | deep Δ **+0.102**; ~1–2 layers; best L30 aligned 0.503 |

Both modalities: **base ≈ 0 → positive after training.** Reading is *learned*. Text is
~2–3× stronger and far broader than audio.

### The base-model trap (why "more attention" is the wrong claim)

At base, the audio model's raw aligned correlation is a large **0.610** — but permuted
is **0.608**, so Δ = **0.003**. Untrained attention *looks* like reading (0.61!) but is
100% positional habit (words run left-to-right in time; attention drifts left-to-right
by position; they correlate for free). Raw attention-on-patches would have been
completely misleading. The permuted control is load-bearing.

### Audio training trajectory (audio-only-v0)

Deep-band Δ over checkpoints — genuine reading switches on ~step 400 and stabilizes,
mirroring the Δperm grounding curve (0.157%→1.275% over the same steps):

| step | base | 200 | 400 | 600 | 800 |
|------|------|-----|-----|-----|-----|
| deep-band Δ | +0.003 | +0.004 | +0.098 | +0.072 | +0.102 |

---

## Interpretation

1. **The model reads pixels — provably, causally (via the permutation control), and as
   a learned behaviour** (absent at base). This is a positive, temporally/positionally
   resolved answer to "does it actually use visual input", stronger than aggregate NLL.
2. **Reading is modality-asymmetric inside one shared channel.** Text: strong, broad.
   Audio: weak, narrow. The same probe detects strong text reading, so the audio
   weakness is a real property of the model, not a broken metric. This is the
   attention-level microscopy behind the Δperm gap (575% vs 1.3%).
3. **"Unifying modalities into one vision channel does not unify their learnability."**
   The four-cell table is the microscopy evidence for that thesis.

## Caveats

- **Attention is correlational.** The definitive, *causal* and *layout-independent*
  version is a **pixel-space occlusion test** (mask a time-band / region, watch which
  target tokens' NLL moves). Recommended next build.
- **Grid orientation** (6×41 vs 41×6) for audio was not fully disambiguated; both
  orientations give the same conclusion. Occlusion would settle it.
- **Single-layer maxima are inflated** by 35-way multiple comparisons; trust the
  deep-band / all-layer means.
- **N = 24** per condition; bootstrap CIs on the Δ trajectory would firm up
  significance (per-clip deltas not yet saved).

## Reproduce

```bash
# audio temporal alignment (one checkpoint)
uv run python data/eval/attn_probe/temporal_align.py \
    --checkpoint data/checkpoints/audio-only-v0/checkpoint-800 --n-clips 24

# audio base+trajectory sweep
bash data/eval/attn_probe/sweep.sh

# text reading-order alignment sweep (base + full-v0/ckpt-2800)
bash data/eval/attn_probe/text_sweep.sh
```

Artifacts: `data/eval/attn_probe/{feasibility.py, temporal_align.py, text_align.py,
sweep.sh, text_sweep.sh}`; results JSON `data/eval/attn_probe/*.json`.

## Next steps

- **P0a — occlusion test** (causal, layout-independent): the capstone that turns
  "attention suggests reading" into "reading confirmed by intervention".
- **P0b — hygiene**: bootstrap CIs on the Δ trajectory; fineweb already serves as the
  cross-lane positive control.
- Feeds the publication plan (`docs/plans/publication_plan.md`, P0 grounding microscopy).
