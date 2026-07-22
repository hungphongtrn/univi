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
