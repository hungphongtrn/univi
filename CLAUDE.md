# CLAUDE.md

Guidance for Claude Code working in this repo.

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

## Remote GPU (A100) — how to run training/eval

The real GPU work runs on an **A100-PCIE-40GB** reachable via SSH host **`aiv_viettel1`**
(host `a100-public-srv`). The project lives at **`/workspace/univi`** inside the running
Docker container **`phong-pytorch-machine`** (`uv`, venv `.venv`, Python 3.12).

**Interactive** (what a human does): `ssh aiv_viettel1` drops you straight into the
container shell (the SSH `Host` block sets `RequestTTY yes` + a `RemoteCommand` that runs
`docker exec -it phong-pytorch-machine zsh`). Then `cd /workspace/univi`.

**Non-interactive** (scripted / for Claude): you must override that `RemoteCommand` and
run `docker exec` yourself. Working pattern:

```bash
ssh -o RemoteCommand=none -o RequestTTY=no -o ClearAllForwardings=yes aiv_viettel1 \
  'docker exec phong-pytorch-machine zsh -c "cd /workspace/univi && <COMMAND>"'
```

Notes / gotchas that will bite you:
- `-o RemoteCommand=none` is **required** — otherwise you get *"Cannot execute
  command-line and remote command"*.
- `-o ClearAllForwardings=yes` silences the harmless `bind [127.0.0.1]:4096: Address
  already in use` LocalForward warning; the config forwards port 4096. You can also pipe
  through `grep -v -E "bind \[|channel_setup|Could not request"`.
- The shell is **zsh**; avoid `echo ===marker===` (leading `=` triggers zsh `=`-expansion
  and errors) — use plain markers like `echo STEP1`.
- Run eval on the A100, not the Mac (no CUDA locally). Model weights are already cached
  there.

Example — run the grounding ablation remotely (single line):

```bash
ssh -o RemoteCommand=none -o RequestTTY=no -o ClearAllForwardings=yes aiv_viettel1 \
  'docker exec phong-pytorch-machine zsh -c "cd /workspace/univi && uv run python eval_ablation.py --config configs/eval.yaml --checkpoint data/checkpoints/full-v0/checkpoint-2800 --max-samples 150"'
```

## Current experiment state (as of 2026-07-20)

- **A full run is LIVE on the A100** (PID 754260, started 2026-07-17):
  `uv run python -m univi.train --config configs/3060_full.yaml`. W&B run `k3ceh56s`
  (project `univi-gemma4`). Do **not** kill it or clobber `data/checkpoints/full-v0/`
  without intent. Config is *named* `3060_full` but is running on the A100.
- **Best checkpoint = `data/checkpoints/full-v0/checkpoint-2800`** (`best_model_checkpoint`
  in `trainer_state.json`; `mean_total_loss` = **1.361**). `save_total_limit=3` +
  `load_best_model_at_end=true` protect it; the other retained dirs are the rolling latest.
- **The run has plateaued / regressed**: at step ~19,600 the macro `best_metric` is still
  the step-2800 value — ~16k steps of no macro improvement. Text (`fineweb-edu`) keeps
  dropping while `densefusion`/`smoltalk`/`librispeech` peaked ~step 2–3k and drift up.
  Finishing 1 epoch (`num_train_epochs=1`, ~183k steps) would take weeks. Recommended
  path: stop, salvage `checkpoint-2800`, diagnose grounding (below), then relaunch with a
  bounded `max_steps`, lower LR, and a rebalanced mixture.
- **Open risk under investigation** (`eval_ablation.py`): whether the model reads pixels
  or exploits language priors — the run's fingerprint (fake-able text learns, must-read
  spectrogram stalls) is consistent with prior-dependence. `librispeech` audio is routed
  through the **vision** encoder (it's a PNG); `finetune_audio_layers:false` is irrelevant
  to it.
- **`eval-v0` datasets are NOT materialized** on the remote (`data/materialized/` has
  `univi-3M-v0`, `univi-3M-v0-split`, `smoke-v0/v1` only). `configs/eval.yaml` points at
  `data/materialized/eval-v0/*`; before running eval-lane/ablation, either materialize
  them (`data/preprocessing/prepare_eval.py`) or point `--config` at a config whose
  `eval_datasets` map to an existing split with `messages`+`images` columns. Missing
  sources are skipped with a `[skip]` line, not an error.
- **`configs/eval.yaml` default `checkpoint_path` is `.../full-v0/final`**, which does not
  exist until training completes — pass `--checkpoint .../checkpoint-2800` explicitly.
- Running eval concurrently with training shares the 40 GB A100 (~16 GB used by training).
  A base+trained 4-bit E2B ablation fits, but if memory is tight use `--no-base` or run
  after the training stops.

## Conventions

- Branch work off `main`; the active branch is `issue-1-phase-1-preprocessing`.
- Commit/push only when asked. Keep GPU imports lazy. Match surrounding code style.
- `metric_for_best_model="mean_total_loss"` is a **macro-average across subsets** computed
  in `univi/evaluation.py`; W&B logs it as `eval/mean_total_loss` (`wandb_utils.py`).
