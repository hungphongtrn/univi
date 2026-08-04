# CLAUDE.md

Guidance for Claude Code working in this repo.

> **Start with [`research.md`](research.md)** — the hub for current status, the committed
> architecture, and everything established. Hypotheses (with pre-registered criteria and verdicts)
> live in [`docs/hypothesis/`](docs/hypothesis/README.md), grouped `done/` · `in-progress/` ·
> `todo/`; literature claims in [`docs/literature/`](docs/literature/README.md).
> The experiment-state sections lower down in this file are **historical narrative** — where they
> disagree with `research.md`, `research.md` wins.

## What this project is

**Univi** is a research experiment in **visual modality unification**: render text as
images, render audio as log-mel spectrogram images, and pass ordinary images unchanged,
so a single vision-language model consumes *all* answer-bearing input through the vision
path. The thesis is that a model can read rendered pixels instead of native text/audio
tokens.

- Baseline model: `unsloth/gemma-4-E2B-it` (Gemma 4 E2B), 4-bit QLoRA via **Unsloth** `FastVisionModel` + TRL `SFTTrainer`.
- Phase-0 training mixture (concatenate-and-shuffle, no hand-tuned ratios):
  `fineweb-edu` (text→image transcription), `densefusion` (natural-image captioning),
  `smoltalk` (rendered-instruction following), `librispeech` (spectrogram→transcript).
- Read **`CONTEXT.md`** first — it is the canonical glossary (Image-Only Lane, Native
  Upper Bound, Retention Metric, Linguistic Prior Dependence, Visual Decodability Gate,
  Modality-Permutation Control). Also `docs/RESEARCH_PLAN.md`, `docs/DATASETS.md`,
  `docs/plans/`, `docs/adr/`.

## Stack / environment

- Python `>=3.12,<3.14`, managed with **`uv`** (always `uv run …`, never bare `python`).
- `torch==2.10.0+cu130`, unsloth, trl, peft, bitsandbytes, flash-attn (prebuilt wheel).
- Heavy GPU imports (unsloth/torch CUDA) are **deferred inside functions** so the modules
  import CPU-only (e.g. on a Mac for editing/tests). Keep it that way — top-level imports
  in `eval_lane.py` / `eval_ablation.py` are CPU-safe; unsloth is imported lazily in
  `load_model`.
- **The probe scripts default to `--max-length 2048`, and at raised soft-token budgets that
  silently DROPS ROWS — biased, not random.** The per-image cost goes 282 → 562 → 1122 at budgets
  280/560/1120, so long-target rows stop fitting and the filter keeps **exactly the shortest
  utterances**. Measured: `randstr-d4` validation loses 31/500 rows at 1120; **librispeech train
  loses 80.24%** (only 20,548 of 104,014 survive). Nothing errors — the run just reports a
  confident number computed on a short-biased subsample, at precisely the deep positions K is
  measured at. **Pass `--max-length 4096` for the randstr rungs and `--max-length 8192` for audio**,
  and re-verify retention whenever a budget changes.
- **`render_utils.py` silently clamps `font_size = max(font_size, 14)`** in all four render entry
  points (found 2026-07-29). Fonts 8, 10, 12 and 14 therefore produce a **byte-identical** image.
  Any experiment that varies font below 14 is not varying anything: H21 was designed to train on
  {10, 20, 40} and hold out {14, 28}, which would have trained the "held-out" font 20,000 times and
  fired a CONFIRMS verdict on a memorised font, with nothing in the logs to show it. Repaired to
  train {14, 24, 40} / hold out {18, 31}. **14 is the smallest page this renderer can draw** — check
  pixel-distinctness (not just the parameter) before trusting any render-geometry sweep.
- **`concatenate_datasets` does NOT raise on mismatched schemas** (`datasets` 4.3.0, unpinned): it
  aligns by column name and **silently null-fills** the missing ones. Lanes carrying extra columns
  (`poisoned-text` +3, `masked-randstr` +12) will merge into a wide null-padded table rather than
  failing. Strip to the shared base columns and assert `Features` equality across lanes before
  concatenating — the hazard is silence, not an error, and it is version-dependent.
- **Exception — `univi/__init__.py` is NOT CPU-importable** (verified 2026-07-29, tripped over
  independently by three separate probes). It imports `univi.trainer` at module scope, which does
  `from unsloth import …` at *its* module scope, and unsloth raises
  `NotImplementedError: Unsloth cannot find any torch accelerator` when no CUDA device is visible.
  So **any** `import univi.*` — including `univi.hybrid.pretrained`, which needs no unsloth — fails
  under `CUDA_VISIBLE_DEVICES=""`. The CPU-only claim above holds for the standalone probe scripts,
  not for the package. Workarounds for CPU-side analysis: stub `sys.modules["unsloth"]` before
  importing, or don't import `univi` at all.

## Repository layout

- `univi/` — the package: `train.py`/`cli.py` (entry points), `trainer.py` (model load,
  LoRA/PEFT, SFTConfig, collator, response-only masking), `evaluation.py` (SFTTrainer
  subclass adding macro-avg `eval_mean_total_loss` = the `metric_for_best_model`),
  `config.py`, `validation.py`, `metrics.py`, `wandb_utils.py`, `smoke.py`, `manifest.py`,
  `fingerprint.py`.
- `eval_lane.py` — image-lane vs native-lane eval (Retention Metric). Reusable seams:
  `load_model`, `_merge_images`, `compute_loss`, `load_source_dataset`.
- `eval_ablation.py` — **visual-grounding ablation**: per-subset target NLL under
  aligned / permuted (Modality-Permutation Control) / blank image, on base + trained
  models. `Δperm = loss(permuted) − loss(aligned)` is the grounding signal; ≈0 ⇒ the
  model ignores pixels and generates from language priors. Built on `eval_lane`'s seams.
- `data/preprocessing/` — offline materialization (renders text/audio to images ONCE):
  `render_utils.py`, `fineweb_edu.py`, `smoltalk.py`, `librispeech_asr.py`,
  `densefusion.py`, `merge_mixture.py`, `prepare_eval.py` (builds the eval-lane datasets
  with native columns). The trainer only *consumes* pre-rendered datasets.
- `configs/` — `3060_full.yaml` (the live full run, despite the "3060" name — see below),
  `eval.yaml` / `3060_eval.yaml` (eval-lane), `smoke.yaml`, `full.yaml`, `3060_1epoch.yaml`.
- `tests/` — pytest; GPU tests marked `requires_gpu`.

## Common commands

```bash
# Train (full run) — long-running
uv run python -m univi.train --config configs/3060_full.yaml

# Tests (skip GPU-only on a CPU box)
uv run pytest -m "not requires_gpu"

# Eval lane (image vs native, retention) — needs materialized eval datasets
uv run python eval_lane.py --config configs/eval.yaml

# Visual-grounding ablation on the best checkpoint
uv run python eval_ablation.py --config configs/eval.yaml \
  --checkpoint data/checkpoints/full-v0/checkpoint-2800 --max-samples 150
```

## GPU environment (A100) — how to run training/eval

All GPU work runs on an **A100-PCIE-40GB**. The project lives at **`/workspace/univi`**
inside the Docker container **`phong-pytorch-machine`** (`uv`, venv `.venv`, Python 3.12,
zsh). The GPU is **40 GB**; a base+trained 4-bit E2B ablation fits comfortably, and (as of
2026-07-21) the GPU is idle — no training is running.

**Claude may be running directly inside the container** (check `hostname` —
`phong-pytorch-machine` means you're on the box). In that case just run commands natively
from `/workspace/univi`, no SSH needed:

```bash
uv run python eval_ablation.py --config configs/eval.yaml \
  --checkpoint data/checkpoints/full-v0/checkpoint-2800 --max-samples 150
```

Gotchas that will bite you either way:
- The shell is **zsh**; avoid `echo ===marker===` (leading `=` triggers zsh `=`-expansion
  and errors) — use plain markers like `echo STEP1`.
- There is **no CUDA on a Mac** — run GPU work here, not locally. Model weights are cached
  on this box.
- **HuggingFace downloads hang on this box.** The **Xet** transfer client silently stalls
  on large LFS blobs (model weights, dataset audio shards) — a `snapshot_download` /
  `from_pretrained` / streaming `load_dataset` freezes at ~1 MB with 0% GPU and no error.
  Plain **`curl` to the same `https://huggingface.co/<repo>/resolve/main/<file>` URLs works
  fine and fast** (the network is not the problem). Workarounds: set `HF_HUB_DISABLE_XET=1`
  (and `HF_HUB_OFFLINE=1` once cached), or bypass the hub entirely — `curl` the files into a
  local dir and `from_pretrained(local_dir)`. Example: `whisper-tiny.en` weights + tokenizer
  were curled into `data/eval/decodability/whisper-tiny.en/`, and LibriSpeech `test-clean`
  was pulled from `openslr.org` (not HF streaming) into
  `data/eval/decodability/LibriSpeech/`.

**If instead working remotely** (from a laptop, container reachable via SSH host
`aiv_viettel1` → host `a100-public-srv`): `ssh aiv_viettel1` drops a human into the
container shell (`RequestTTY yes` + a `RemoteCommand` running `docker exec -it
phong-pytorch-machine zsh`). For scripted/non-interactive use, override that
`RemoteCommand` and run `docker exec` yourself:

```bash
ssh -o RemoteCommand=none -o RequestTTY=no -o ClearAllForwardings=yes aiv_viettel1 \
  'docker exec phong-pytorch-machine zsh -c "cd /workspace/univi && <COMMAND>"'
```

- `-o RemoteCommand=none` is **required** — else *"Cannot execute command-line and remote
  command"*.
- `-o ClearAllForwardings=yes` silences the harmless `bind [127.0.0.1]:4096: Address
  already in use` LocalForward warning (config forwards port 4096); or pipe through
  `grep -v -E "bind \[|channel_setup|Could not request"`.

## Encoder-free thread (as of 2026-07-24) — ACTIVE

Separate experiment from the Gemma run below and from `univi/hybrid/`. **Pretrained
Qwen3-1.7B decoder (full FT) + a from-scratch tiny pixel embedder** (`train_encoder_free.py`,
`configs/encoder_free.yaml`). Goal: teach the pretrained Qwen3 to SEE all modalities through
pixels — **REUSE Qwen3, do NOT random-init it** (user rejected the hybrid/random-init path
for this thread as too costly).

- **Original run saturated + died.** W&B `9hb9aike`. Train loss hit ~1.85 by step ~70 and
  never descended; fineweb-edu eval frozen ~2.64 (≈ no-image English LM perplexity). Died at
  step 2890 — **host-RAM OOM** (swap 7/7 full, 38GB RSS) when a concurrent GPU probe piled on.
  **LESSON: run training SOLO.** Checkpoints: `data/checkpoints/encoder-free-v0/{best(=step2500),
  checkpoint-2000,checkpoint-2500}`.
- **Three probes (verified).** (1) OCR gate: rendered text fully legible at the 512²/32px the
  model sees → resolution ruled out. (2) 4-lane Δperm/Δblank ablation on `best/`
  (`scratchpad/h1_ablation.py`, `scratchpad/h1_ablation.json`): ALL lanes read pixels but
  UNIFORMLY SHALLOW — Δperm fineweb +3.15%, densefusion +3.57%, smoltalk +3.43%, **librispeech
  +1.87%** (vs Gemma strong ≈+575%, ignore-floor ≈+0.1%). (3) KEY: **audio Δblank POSITIVE
  (+1.75%)** — spectrogram beats blank, REVERSING Gemma (audio there was ignore-floor w/
  negative Δblank). Reusing Qwen3 through a shared pixel path makes audio ground.
- **Diagnosis:** uniformly shallow reading — the pretrained LM prior out-competes the
  from-scratch vision path for gradient (uniform 3e-4 let the decoder win). Weak, not broken.
- **H2 RAN + FAILED (2026-07-25 analysis).** `configs/encoder_free_h2.yaml`, W&B `xc2vypxx`,
  `data/checkpoints/encoder-free-h2-v0/{best==final}`. Warmup+split-LR (freeze decoder 800 steps,
  embedder 3e-4 / decoder 2e-5). Final macro **1.875 — WORSE than H1's 1.743**; warmup barely
  moved loss, all descent came after unfreeze, hard plateau after step 1250. **Ablation reversed
  both signals: Δperm COLLAPSED to the ignore-floor** (fineweb 3.15%→0.22%, densefusion 3.57%→0.14%,
  smoltalk 3.43%→1.38%, libri 1.87%→0.57%) while **Δblank EXPLODED** (fineweb→+23%, densefusion→+68%,
  smoltalk→+95%). Mechanism = **content-blind SOFT-PROMPT COLLAPSE**: against a frozen decoder the
  only way a from-scratch embedder lowers loss is a constant content-independent steering vector;
  the frozen-decoder warmup CAUSED it. fineweb aligned dropped below the 2.64 floor (→2.47) but
  that's the steering prefix, NOT reading. **LESSON: never freeze the decoder for a from-scratch
  embedder; H1's plain single-3e-4 recipe was the better grounding state.**
- **H3 RAN + FAILED — grounding COLLAPSED TO ZERO (2026-07-26 analysis).** Prior-proof ISOLATED OCR
  diagnostic (design vetted with Fable): train on ONE lane — render **random lowercase-letter strings
  → transcribe** (charset a-z, 5×5 letters, fineweb geometry). Zero LM shortcut ⇒ a steering prefix
  can't cheat; separates (A) gradient competition vs (B) linear-embedder capacity. Recipe = H1 EXACTLY
  (single LR 3e-4, no freeze), **warm-started decoder+embedder from `encoder-free-v0/best`**. New:
  materializer `data/preprocessing/random_strings.py` → `data/materialized/h3-randstr-v0/`
  (100k/2k + `floor.json`); `"random-strings"` added to `univi/trainer.VALID_SUBSETS`; embedder
  auto-warm-start in `build_encoder_free_univi`; `configs/h3_randstr.yaml` (+`_smoke`). W&B `gshey7fa`
  (`h3-randstr-ocr-warmstart`), `data/checkpoints/h3-randstr-v0/{best(=step500),checkpoint-500/1000/1500,
  final}`, log `…-run.log`, 1500 steps, ran SOLO, finished clean.
  - **Training:** eval macro 3.71 (step 0) → **2.214 (best, step 500)** → drifted UP to 2.249 (step
    1500). Saturated early, mildly regressed; train loss flat ~2.13. NEVER dived toward 0.
  - **Ablation VERDICT = hyp B, capacity/optimization (NOT reading).** Rebuilt `scratchpad/h3_ablation.py`
    (the original was in uncommitted scratchpad, cleared — new one reuses `train_encoder_free` seams:
    answer-only CE + teacher-forced tok-acc under aligned/permuted/blank, warm-loads decoder+embedder
    from the ckpt). Outputs `data/eval/h3-ablation-{h3,h1-baseline}.json` (200 rows each). **H3-trained:
    aligned=permuted=blank IDENTICAL to 5–6 sig figs (CE 4.023 all three; Δperm +0.0005%, Δblank +0.001%;
    tok-acc 32.07% aligned==blank, bit-identical).** The image is PROVABLY IGNORED. Not a plumbing bug:
    same code+rows shows **H1-baseline** responding (CE aligned 6.196 < blank 6.296, Δblank **+1.61%**,
    Δperm +0.25%, reading gain +0.26 pts).
  - **Mechanism:** H3 training DESTROYED the weak reading H1 had (regression, not stall). CE fell
    6.20→4.02 (below the 5.61 floor) ENTIRELY via format-modeling — the decoder learned the deterministic
    scaffold (fixed 5-letter groups, spaces, `im_end`, uniform letter marginal, guess at chance) while the
    from-scratch embedder's signal decayed to nothing. This is the **H2 soft-prompt-collapse basin
    reappearing WITHOUT a language shortcut**: a strong pretrained decoder + from-scratch linear embedder,
    on random targets, finds "model the output marginal + format, ignore pixels" as the lowest-loss basin
    and out-competes the embedder for gradient. The mixture's other lanes had been keeping the embedder
    marginally alive; isolation removed them. Vindicates the pre-registered "crossing the floor is not the
    clean signal" warning — H3 sits below floor reading NOTHING.
  - **NEXT (pre-registered, stall ⇒ hyp B): escalate connector capacity (linear → MLP → conv stem).**
  - **H3-MLP RAN + NULL (2026-07-26).** First escalation rung: replaced the linear Fuyu-style connector
    with an **MLP (Linear→GELU→Linear, mlp_ratio 4, +30M params)**, config-gated via `model.connector`/
    `mlp_ratio` in `train_encoder_free.py` (`_build_vision_embedder`/`EFConfig`). `configs/h3_randstr_mlp.yaml`
    — ONLY changed variable vs H3 (same single-LR 3e-4, no freeze, warm-start decoder+lower-embedder from
    encoder-free-v0/best; MLP connector fresh-inits, lower layers reused). W&B `h3-randstr-ocr-mlp-connector`,
    `data/checkpoints/h3-randstr-mlp-v0/{best(=step200)}`. **Killed early at step ~300 (pre-registered):**
    eval REPLICATED H3's plateau to the decimal — step0 3.45 → step100 2.2245 → step200 **2.2217(best)** →
    step300 2.2241 (H3 was 2.231→2.224→2.225; if anything a hair worse than H3's eventual 2.214).
    **Ablation (`--connector mlp`, `data/eval/h3-ablation-h3-mlp.json`): IDENTICAL collapse — aligned=
    permuted=blank 4.031 all three, Δperm −0.000%, Δblank +0.000%, tok-acc 32.32% aligned==blank bit-
    identical, reading gain +0.00 pts.** The image is STILL fully ignored. +30M non-linear capacity changed
    NOTHING. **Rules out connector-expressivity as the cause** — linear and MLP land in the identical
    ignore-pixels basin. The bottleneck is the OPTIMIZATION DYNAMIC (decoder's format shortcut wins gradient
    from step 0; vision path never recruited), upstream of the embedder function class. Conv stem (next rung)
    would likely fail the same way. **VERDICT: option-1 (connector escalation) EXHAUSTED as a cheap fix.**
  - **Deeper open question RESOLVED by the pretrained-vision hybrid below** — the from-scratch embedder was
    the bottleneck, not the reused-decoder idea. Swapping in a PRETRAINED vision front-end makes the same
    Qwen decoder read. See `docs/encoder-free-thread-postmortem.md` (its "needs a capable pretrained vision
    embedder" recommendation is now VALIDATED).

## Pretrained-vision hybrid (2026-07-26) — THESIS VALIDATED, ACTIVE

Follow-up to the encoder-free collapse (user's architecture proposal). **PRETRAINED Gemma-4-12B unified
vision embedder + from-scratch Linear(3840→2048) adapter + PRETRAINED Qwen3-1.7B**, 3-group LR (adapter
3e-4 > vision 5e-5 > decoder 2e-5, NO freeze). The encoder-free diagnosis said the *un-pretrained* vision
path was the bottleneck; this swaps in a capable pretrained front-end and keeps the reused Qwen decoder.

- **Key correction (verified, don't trust the old memory):** Gemma-4 **E2B vision = a 16-layer ViT**
  (`gemma4_vision`, hidden 768); **12B vision = the encoder-free `gemma4_unified` patchify+linear embedder**
  (mm_embed_dim 3840, model_patch_size 48, 280 soft tokens). They are DIFFERENT architectures. User chose
  the **12B unified** embedder (the "unified" single-path design = the univi thesis; and it tests whether
  *pretraining* rescues the linear design that failed from-scratch).
- **Weights:** 12B repo is a single 24GB safetensors, not downloaded. Range-downloaded ONLY the 10 vision
  tensors (~100MB) via curl byte-ranges (`scratchpad/fetch_gemma12b_vision.py` → `data/hybrid/gemma12b-vision/
  vision_embedder.pt`); load into `Gemma4UnifiedVisionEmbedder` clean (10/10, no missing/unexpected).
- **Code:** `univi/hybrid/pretrained.py` (`UniViHybridPretrained` = pretrained vision@3840 + adapter + pretrained
  Qwen, masked_scatter merge; `build_pretrained_hybrid` loads both towers pretrained), `univi/hybrid/train_pretrained.py`
  (Trainer + 3-group bnb-8bit AdamW, reuses `HybridCollator`), `configs/hybrid_pretrained_randstr.yaml`,
  `scratchpad/hybrid_pretrained_ablation.py`. Decoder src `unsloth/Qwen3-1.7B` (cached), vision config
  `google/gemma-4-12B-it` (config cached).
- **RESULT — random-string OCR diagnostic (600 steps, eff batch 64, SOLO). READS STRONGLY.** W&B run_name
  `hybrid-pretrained-vision-randstr`, `data/checkpoints/hybrid-pretrained-randstr-v0/{checkpoint-200/400/600,
  final}`. Eval descended 5.06→**3.735** (still descending when the cosine LR decayed out; grad_norm healthy
  ~4-5 throughout — NOT the H3 dead-flat). **Ablation (`data/eval/hybrid-pretrained-ablation-*.json`):**

  | | Δperm | Δblank | reading gain | |
  |---|---|---|---|---|
  | H3 / H3-MLP (from-scratch) | ~0% | ~0% | +0.00 pts | image ignored |
  | hybrid-pretrained @200 | +27.4% | +39.5% | +4.6 pts | emerging |
  | **hybrid-pretrained final** | **+114.7%** | **+90.8%** | **+22.2 pts** | **reads** |

  Aligned CE 3.69; permuted 7.92 (a WRONG image is worse than blank 7.03 → genuine content-conditioning,
  Gemma-like); tok-acc 32% aligned vs 9.8% blank (guess floor). Grounding GREW with training (27%→115%).
  +115% sits in Gemma's strong-reading band (its lanes were +39–575%). **Thesis supported: a reused Qwen
  decoder DOES read pixels through a pretrained vision front-end; the from-scratch embedder was the whole
  problem.** Caveat: PARTIAL reading (tok-acc 32%, CE 3.69, not →0) — a bounded 600-step run whose LR ran
  out mid-descent, not converged.
- **FROZEN-TRANSFER characterized (2026-07-26):** the OCR-trained vision embedder barely moved (rel-delta
  ~0 across patch_dense/pos/LN; only Gemma's final projector +4% and the from-scratch adapter changed).
  So the pretrained Gemma vision FEATURES are reused essentially FROZEN; a from-scratch LINEAR adapter is
  enough to make them readable by a DISTINCT decoder (Qwen). This is LLaVA-style cross-decoder transfer,
  and it's why the from-scratch encoder-free embedder failed (no pretrained features to transfer).
- **AUDIO reads too — PRIOR-PROOF (2026-07-26). Strongest result in the program.** Built a prior-proof
  spoken-digit AUDIO lane (`data/preprocessing/spoken_digits.py` → `data/materialized/spoken-digits-v0/`,
  30k/1k, floor 1.228 nats/tok): random digit sequences from FSDD real speech (curled from GitHub) →
  production log-mel spectrogram → transcribe. Uniform i.i.d. digits ⇒ ZERO language-prior shortcut (fixes
  the user's worry that librispeech English transcripts let the Qwen prior cheat). `"spoken-digits"` added
  to `univi/trainer.VALID_SUBSETS`; `freeze_vision` knob added to `train_pretrained.py`/`pretrained.py`;
  ablation `scratchpad/hybrid_pretrained_ablation.py` generalized (`--val-dataset`/`--floor`).
  - **FROZEN vision (`configs/hybrid_pretrained_spokendigits.yaml`): NO reading** — eval flat at ~1.08
    (≈floor), grad_norm small. Mechanism: spectrograms are OUT-OF-DISTRIBUTION for Gemma's vision (never
    pretrained on them), so frozen features don't encode them; the adapter alone can't recover it.
  - **TRAINABLE vision (`configs/hybrid_pretrained_spokendigits_trainvis.yaml`, lr_vision 1e-4): READS.**
    W&B `hybrid-pretrained-spoken-digits-trainvision`, `data/checkpoints/hybrid-pretrained-spokendigits-trainvis-v0/`.
    Eval descended 1.084→**0.726** (below the 1.228 floor; grad_norm rising ~2.4). Ablation
    (`data/eval/hybrid-pretrained-ablation-spdigit-{final,200}.json`): grounding grew @200 Δperm +7.4% →
    **final Δperm +134.8%, Δblank +129.9%, reading gain +17.6 pts, tok-acc 75% vs 57% blank**; permuted
    (1.67) worse than blank (1.63) = content-conditioning. Prior-proof ⇒ this is genuine spectrogram-reading.
  - **MECHANISTIC FINDING:** text/image reading transfers FROZEN (in-distribution for Gemma vision); AUDIO
    reading requires ADAPTING the vision tower (OOD). Either way the reused-Qwen + pretrained-Gemma-vision
    path reads text, images, AND audio — all confirmed prior-proof. Caveat: partial (tok-acc 75%, CE 0.71,
    still descending at 600-step LR-out); frozen-audio "no reading" is from flat eval, not a separate ablation.
- **NEXT (open, awaiting direction):** (1) longer/scaled runs (both OCR and audio were still descending) to
  push partial→full reading; (2) the real 4-lane mixture — read everything at once, no lane drowning, with
  vision trainable (needed for audio); (3) compare vs full-Gemma. E3 (audio-only on plain Gemma) also came
  back POSITIVE (see below) — a lower-risk audio path, now somewhat superseded by this stronger prior-proof result.

## Current experiment state (as of 2026-07-21)

- **The full run has STOPPED — GPU is idle** (no `univi.train` process, ~4 MiB used).
  It ran `configs/3060_full.yaml` (W&B run `k3ceh56s`, project `univi-gemma4`) from
  2026-07-17 and reached ~step 19,600 before ending. Config is *named* `3060_full` but ran
  on the A100. `data/checkpoints/full-v0/` now holds: `checkpoint-2800` (best),
  `checkpoint-19400`, `checkpoint-19600` (rolling latest). Do **not** clobber
  `full-v0/checkpoint-2800` without intent.
- **Best checkpoint = `data/checkpoints/full-v0/checkpoint-2800`** (`best_model_checkpoint`
  in `trainer_state.json`; `best_metric`/`mean_total_loss` = **1.361**). `save_total_limit=3`
  + `load_best_model_at_end=true` protected it through the run.
- **The run plateaued/regressed before it ended**: the macro `best_metric` never beat the
  step-2800 value — ~16k steps of no macro improvement. Text (`fineweb-edu`) kept dropping
  while `densefusion`/`smoltalk`/`librispeech` peaked ~step 2–3k and drifted up. Finishing
  1 epoch (`num_train_epochs=1`, ~183k steps) would have taken weeks. Recommended path:
  salvage `checkpoint-2800`, diagnose grounding (below), then relaunch with a bounded
  `max_steps`, lower LR, and a rebalanced mixture.
- **Grounding ablation done** (`data/eval/val-ablation/ablation.json`, checkpoint-2800,
  120 samples): grounding is **modality-specific, not a global prior-exploitation issue**
  (this *supersedes* the older "global prior-dependence" framing in the risk bullet above).
  Relative Δperm (trained): fineweb **+575%**, densefusion **+63%**, smoltalk **+39%** — all
  genuinely read pixels. **librispeech +0.16%** (flag `IGNORES IMAGE`) with a *negative*
  Δblank (−0.106, blank image beats the real spectrogram) — the spectrogram is an active
  distractor: the model learned to *ignore* it and lean on the text prior.
- **E1 — audio never grounds at any exposure.** Re-ablated checkpoint-19600 (7× more audio,
  ~11k vs ~1.6k spectrograms seen): `data/eval/val-ablation/ablation-ckpt19600.json`.
  librispeech rel Δperm went **0.16% → 0.07%** (still floor), Δblank still negative. The
  whole model regressed 2800→19600 (that's the known plateau), so read it as: audio grounding
  sat at ~0.1% at *both* checkpoints while every other lane reached 34–795%. Data-limited-at-
  current-mixture (a *rising* slope with exposure) is falsified; audio is categorically
  different, not merely lagging. librispeech is only **3.52%** of the mixture (104k/2.96M rows;
  the other three are ~30% each; effective batch 16).
- **E2 — the render is NOT the bottleneck (decisive).** Model-free decodability gate,
  `data/eval/decodability/wer_direct.py` → `wer_gate.json`: 30 real test-clean clips, Whisper
  (tiny.en) fed three ways — NATIVE (raw audio), CONTROL (librosa mel direct), GATE (mel
  recovered from the rendered PNG). **All three WER = 3.75%**, identical transcripts including
  identical errors. The log-mel PNG round-trip loses **0%** for a strong reader — "rendering
  destroys the signal" is hard-falsified. (Production render is the paged path:
  `page_duration_sec=10`, `output_width=1000` = 1000 mel frames = 10 ms/px, no time-warp;
  80→160 vertical stretch only. The earlier agent's "GATE FAIL" was a Griffin-Lim artifact.)
- **Combined verdict: LEARNABILITY-LIMITED.** Readable pixels the model won't learn to read.
  Mechanism: audio at 3.5% share + no spectrogram pretraining prior + easy lanes dominating
  the gradient → E2B fell into an "ignore-the-image" basin (negative Δblank) early and never
  climbed out. This *reopens* curriculum/reweighting as a fix (E1 looked like "data can't
  help", but E2 proves the signal is there to be learned).
- **Open next step — E3:** audio-only bounded LoRA fine-tune (librispeech alone, few hundred
  steps) → re-ablate. If audio-only training produces grounding ⇒ mixture drowned audio ⇒ fix
  with audio-first curriculum + higher share (bounded rebalanced run = E4). If still flat ⇒
  capacity/architecture, bigger rethink.
- **[RESOLVED — see E1/E2 below]** The original open risk was framed as *global* prior-
  dependence (whether the model reads pixels at all vs exploits language priors). The
  ablations disproved the global framing: three of four lanes read pixels strongly; the
  problem is **audio-specific and learnability-limited**, not prior-exploitation and not a
  render defect. `librispeech` audio is routed through the **vision** encoder (it's a PNG);
  `finetune_audio_layers:false` is irrelevant to it.
- **`eval-v0` datasets are NOT materialized** on the remote (`data/materialized/` has
  `univi-3M-v0`, `univi-3M-v0-split`, `smoke-v0/v1` only). `configs/eval.yaml` points at
  `data/materialized/eval-v0/*`; before running eval-lane/ablation, either materialize
  them (`data/preprocessing/prepare_eval.py`) or point `--config` at a config whose
  `eval_datasets` map to an existing split with `messages`+`images` columns. Missing
  sources are skipped with a `[skip]` line, not an error.
- **`configs/eval.yaml` default `checkpoint_path` is `.../full-v0/final`**, which does not
  exist until training completes — pass `--checkpoint .../checkpoint-2800` explicitly.
- The full 40 GB A100 is currently free (training stopped), so eval/ablation has the whole
  GPU. If a training run is relaunched, a concurrent base+trained 4-bit E2B ablation still
  fits alongside it (~16 GB for training); use `--no-base` if memory gets tight.

## Conventions

- Branch work off `main`; the active branch is `issue-1-phase-1-preprocessing`.
- Commit/push only when asked. Keep GPU imports lazy. Match surrounding code style.
- `metric_for_best_model="mean_total_loss"` is a **macro-average across subsets** computed
  in `univi/evaluation.py`; W&B logs it as `eval/mean_total_loss` (`wandb_utils.py`).
