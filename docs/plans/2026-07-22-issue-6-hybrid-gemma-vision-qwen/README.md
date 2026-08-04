# Issue #6 — Random-init hybrid: Gemma 4 image embedder + Qwen 3 1.7B

Replicate the **Gemma 4 image-embedder architecture** (random weights, no
pretrained load), attach a smaller/faster **Qwen 3 1.7B** language backbone
(also random init), and train on the **existing materialized data**
(`univi-3M-v0-split`).

## Why (research motivation)

Every prior Univi finding was confounded by the pretrained language prior of
`gemma-4-E2B-it`. The CLAUDE.md "learnability-limited" verdict turned on the model
falling into an *ignore-the-image* basin **because** it already had strong text
priors (negative Δblank on librispeech: a blank image beat the real spectrogram).

A **from-scratch** model removes that confound. With random weights there is no
inherited text prior to fall back on, so any pixel-reading the model acquires is
learned here, from these images. This directly tests whether the visual-
unification objective is learnable *in principle*, independent of a donor model's
priors — the open question E3/E4 in CLAUDE.md were circling.

## What is replicated (architecture only)

**Correction (superseding an earlier cut of this issue):** the Gemma 4 12B
"unified" line is **encoder-free** — no ViT, no self-attention between patches.
The first cut of this issue used the E2B `gemma4_vision` config (a 16-layer ViT),
which is architecturally distinct. `univi/hybrid/vision.py` now replicates the
actual 12B `gemma4_unified_vision` path instead — see its module docstring for
the full provenance note.

| Component | Source of architecture | Weights |
|-----------|------------------------|---------|
| Vision tower | `Gemma4UnifiedVisionEmbedder` (`univi/hybrid/vision.py`), replicating `google/gemma-4-12B-it`'s encoder-free `gemma4_unified_vision` path: patchify → merge k×k teacher patches → Dense/LayerNorm → factorized row/col positional embeddings → LayerNorm. No ViT layers. | random |
| Projector | `Gemma4UnifiedMultimodalEmbedder(vision_config, text_config)` — RMSNorm → Linear, baked into the vision embedder's last step, parameterized by the text hidden size so it targets Qwen's 2048-dim space unchanged. | random |
| LLM | `Qwen3ForCausalLM` from the Qwen3-1.7B config (`hidden_size=2048`, 28 layers). | random |

Soft-token merge mirrors Gemma 4 exactly: embed `input_ids`, then `masked_scatter`
the projected image soft tokens into the image-placeholder positions.

## Code

`univi/hybrid/` — a subpackage (moved out of flat `univi/hybrid_*.py` files):

- `univi/hybrid/model.py` — `UniViHybridConfig` + `UniViHybridForConditionalGeneration`.
- `univi/hybrid/vision.py` — `Gemma4UnifiedVisionEmbedder` (the encoder-free image
  embedder) + `Gemma4UnifiedImageProcessor` (patchify + merge, vendored since this
  repo's pinned `transformers` doesn't yet ship `gemma4_unified`).
- `univi/hybrid/build.py` — assemble the random-init model + Qwen tokenizer (with
  an added `<|univi_image|>` placeholder token) + Gemma image processor from
  *configs only* (no weight download).
- `univi/hybrid/data.py` — `HybridCollator`: Gemma image processor for pixels,
  Qwen tokenizer for text, per-image soft-token expansion, response-only masking.
- `univi/hybrid/train.py` — plain `transformers.Trainer` entry (the model is not
  an Unsloth `FastVisionModel`), reusing `univi.trainer.load_dataset`. Run with
  `uv run python -m univi.hybrid.train --config <yaml>`.
- `configs/hybrid_smoke.yaml` — 20-step plumbing smoke.
- `configs/hybrid_bounded.yaml` — bounded (3000-step) training run on the full
  `univi-3M-v0-split` mixture (E4).
- `tests/test_hybrid.py` — CPU tests (tiny faithful configs).

## Status

- **Architecture replicated and validated.** CPU tests pass (`tests/test_hybrid.py`,
  4/4): config composition, text-only forward/backward, placeholder/soft-token
  mismatch guard, and full collator↔model integration with real image processor +
  tokenizer (placeholder count == projected soft-token count; gradients reach the
  vision tower, projector, and LM).
- **GPU smoke** on the A100 with the *full-size* random-init model over real
  `univi-3M-v0-split` rows, re-run after the encoder-free correction — see
  `decisions.md` for the recorded loss trajectory.
- **Bounded training run (E4) in progress** — `configs/hybrid_bounded.yaml`,
  3000 steps over the real mixture, launched via `data/eval/hybrid_bounded_run.sh`;
  checkpoints land in `data/checkpoints/hybrid-bounded-v0/`.

## Setup note

Qwen3-1.7B config + tokenizer were `curl`ed into `data/hybrid/qwen3-1.7b/`
(config.json, tokenizer.*, vocab/merges) to sidestep the Xet hub hang documented
in CLAUDE.md. Only tiny files are fetched — **no model weights** are downloaded,
which is exactly what random init requires. `data/hybrid/` is gitignored;
reproduce it with:

```bash
mkdir -p data/hybrid/qwen3-1.7b
base="https://huggingface.co/Qwen/Qwen3-1.7B/resolve/main"
for f in config.json tokenizer.json tokenizer_config.json vocab.json \
         merges.txt generation_config.json; do
  curl -sfL "$base/$f" -o "data/hybrid/qwen3-1.7b/$f"
done
```

The `vision_source` (`google/gemma-4-12B-it`) resolves its tiny `config.json` +
`preprocessor_config.json` from the normal HF cache — those are already present on
the box and are not Xet-gated.

## Not done here (next steps)

- Let the bounded run (E4) finish, then re-run the grounding ablation
  (`eval_ablation.py` seams) against the resulting checkpoint to compare
  Δperm/Δblank against the `gemma-4-E2B` baseline — the from-scratch model is
  the clean control that experiment always wanted.
- A rebalanced/audio-first-curriculum follow-up run, if E4's librispeech lane
  still shows the flat grounding signal seen in the pretrained-baseline ablations.
