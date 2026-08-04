"""
Training entry point for the PRETRAINED-vision hybrid
(:class:`univi.hybrid.pretrained.UniViHybridPretrained`): pretrained Gemma-12B
unified vision embedder + from-scratch adapter + pretrained Qwen3-1.7B.

Same Trainer/collator path as :mod:`univi.hybrid.train`, but with:
  * both towers loaded PRETRAINED (only the adapter is random),
  * a **3-group learning rate** (adapter > vision > decoder, NO freezing — the H2
    lesson: freezing the decoder against a fresh component causes soft-prompt
    collapse), built by a Trainer subclass overriding ``create_optimizer``,
  * bitsandbytes 8-bit AdamW so full-FT of both pretrained towers fits on 40GB,
  * an optional held-out eval split for the loss trajectory (the decisive read is
    the post-hoc grounding ablation, run separately).

Usage:
    HF_HUB_OFFLINE=1 uv run python -m univi.hybrid.train_pretrained \
        --config configs/hybrid_pretrained_randstr.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_training_args(tc: dict):
    from transformers import TrainingArguments

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    has_eval = bool(tc.get("do_eval", True))
    return TrainingArguments(
        output_dir=tc["output_dir"],
        per_device_train_batch_size=tc.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=tc.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=tc.get("gradient_accumulation_steps", 1),
        # learning_rate here is a nominal default; real LRs come from the 3 groups.
        learning_rate=tc.get("lr_adapter", 3e-4),
        warmup_ratio=tc.get("warmup_ratio", 0.03),
        lr_scheduler_type=tc.get("lr_scheduler_type", "cosine"),
        weight_decay=tc.get("weight_decay", 0.01),
        max_grad_norm=tc.get("max_grad_norm", 1.0),
        num_train_epochs=tc.get("num_train_epochs", 1),
        max_steps=tc.get("max_steps", -1),
        logging_steps=tc.get("logging_steps", 10),
        eval_strategy=("steps" if has_eval else "no"),
        eval_steps=tc.get("eval_steps", 100),
        save_steps=tc.get("save_steps", 500),
        save_total_limit=tc.get("save_total_limit", 3),
        bf16=bf16,
        fp16=not bf16 and torch.cuda.is_available(),
        gradient_checkpointing=tc.get("gradient_checkpointing", True),
        dataloader_num_workers=tc.get("dataloader_num_workers", 0),
        report_to=tc.get("report_to", ["wandb"]),
        remove_unused_columns=False,
        seed=tc.get("seed", 3407),
    )


# ---------------------------------------------------------------------------
# H16 — prior-gap-weighted loss (blind-branch reweighting)
# ---------------------------------------------------------------------------
# Per batch a no-grad BLANK-image forward gives a per-token "what the prior alone
# costs". Tokens the sighted model already buys with the image get upweighted:
#
#     w_t = 1 + beta * max(0, CE_blank_t - CE_aligned_t.detach())   [mean-1 over
#                                                                    supervised]
#     loss = sum_t w_t * CE_aligned_t / denom
#
# Everything below is inert unless ``training.gap_weighted_loss`` (or
# ``training.log_blank_ce``) is set: ``_make_trainer_class`` early-returns into
# the stock ``Trainer`` path, byte for byte as before.


def per_token_ce(logits, labels):
    """Per-token CE and the supervised mask, matching ``ForCausalLMLoss`` exactly.

    Same shift (right-pad the labels with the ignore index, drop the first), same
    float upcast, same ``ignore_index``. Returns ``(ce, valid)`` flattened over
    ``batch * seq``; ``ce`` is 0 wherever ``valid`` is False.
    """
    import torch.nn.functional as F

    from univi.hybrid.data import IGNORE_INDEX

    labels = F.pad(labels, (0, 1), value=IGNORE_INDEX)
    shift_labels = labels[..., 1:].contiguous().view(-1)
    flat_logits = logits.float().view(-1, logits.size(-1))
    shift_labels = shift_labels.to(flat_logits.device)
    ce = F.cross_entropy(
        flat_logits, shift_labels, ignore_index=IGNORE_INDEX, reduction="none"
    )
    return ce, shift_labels != IGNORE_INDEX


def gap_weights(ce_aligned, ce_blank, valid, beta):
    """``w_t = 1 + beta * relu(CE_blank_t - CE_aligned_t)``, mean-1, DETACHED.

    - masked (``-100``) tokens get weight **0** so padding cannot dilute the mean;
    - the mean is taken over the supervised tokens only;
    - ``beta = 0`` (or a gap that is 0 everywhere) ⇒ all weights exactly 1 ⇒ the
      loss collapses to the ordinary token-mean CE.
    """
    with torch.no_grad():
        keep = valid.to(ce_aligned.dtype)
        w = torch.ones_like(ce_aligned)
        if beta:
            w = w + float(beta) * torch.clamp(ce_blank - ce_aligned, min=0.0)
        w = w * keep
        n = keep.sum()
        if n > 0:
            mean_w = w.sum() / n
            if mean_w > 0:
                w = w / mean_w
        return w.detach()


def _make_trainer_class(
    lr_adapter,
    lr_vision,
    lr_decoder,
    weight_decay,
    gap_weighted_loss: bool = False,
    gap_beta: float = 1.0,
    log_blank_ce: bool = False,
):
    from transformers import Trainer

    from univi.hybrid.data import BLANK_PREFIX, IGNORE_INDEX

    # A single flag decides whether ANY of the H16 code below can run. When it is
    # False every override delegates straight to ``Trainer`` on its first line.
    blind_branch = bool(gap_weighted_loss or log_blank_ce)

    class ThreeGroupTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._blank_baseline: dict[str, float] = {}
            self._reset_blank_eval_stats()
            self._reset_gap_train_stats()

        def create_optimizer(self):
            if self.optimizer is None:
                import bitsandbytes as bnb

                groups = self.model.param_groups(lr_adapter, lr_vision, lr_decoder)
                self.optimizer = bnb.optim.AdamW8bit(groups, weight_decay=weight_decay)
                logger.info(
                    "3-group AdamW8bit: adapter=%g vision=%g decoder=%g",
                    lr_adapter, lr_vision, lr_decoder,
                )
            return self.optimizer

        # -- H16 bookkeeping ------------------------------------------------
        def _reset_blank_eval_stats(self):
            self._blank_eval = {"aligned": 0.0, "blank": 0.0, "tokens": 0}

        def _reset_gap_train_stats(self):
            self._gap_train = {"gap_sum": None, "w_max": None, "tokens": 0}

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            if not blind_branch:
                return super().compute_loss(
                    model, inputs, return_outputs, num_items_in_batch
                )

            # The blank tensors are collator extras; the model must never see them.
            blank = {
                k[len(BLANK_PREFIX):]: v
                for k, v in inputs.items()
                if k.startswith(BLANK_PREFIX)
            }
            if blank:
                inputs = {
                    k: v for k, v in inputs.items() if not k.startswith(BLANK_PREFIX)
                }

            # EVAL: never reweight — eval_loss must stay comparable to every other
            # run. Optionally measure the blind branch for the degeneracy guard.
            if not model.training:
                loss, outputs = super().compute_loss(
                    model, inputs, return_outputs=True,
                    num_items_in_batch=num_items_in_batch,
                )
                if log_blank_ce and blank:
                    self._accumulate_blank_ce(
                        model, inputs, blank, loss, num_items_in_batch
                    )
                return (loss, outputs) if return_outputs else loss

            labels = inputs.get("labels")
            if not gap_weighted_loss or not blank or labels is None:
                return super().compute_loss(
                    model, inputs, return_outputs, num_items_in_batch
                )

            fwd = {k: v for k, v in inputs.items() if k != "labels"}

            # BLIND BRANCH FIRST, forward-only: no_grad ⇒ no graph, no retained
            # activations, no contribution to any parameter gradient. Running it
            # BEFORE the aligned pass means its logits are freed again before the
            # aligned graph exists, so peak memory stays ~= the unweighted path
            # and the cost is the doc's ≈ +35% wall-clock, not +35% VRAM.
            with torch.no_grad():
                blank_out = model(**{**fwd, **blank})
                ce_blank, _ = per_token_ce(blank_out.logits, labels)
                del blank_out

            outputs = model(**fwd)
            ce_aligned, valid = per_token_ce(outputs.logits, labels)
            n_valid = int(valid.sum())
            if n_valid == 0:
                loss = ce_aligned.sum()
                return (loss, outputs) if return_outputs else loss

            w = gap_weights(ce_aligned.detach(), ce_blank, valid, gap_beta)
            self._record_gap_stats(ce_aligned.detach(), ce_blank, valid, w)

            denom = num_items_in_batch if num_items_in_batch is not None else n_valid
            if torch.is_tensor(denom):
                denom = denom.to(ce_aligned.device)
            loss = (w * ce_aligned).sum() / denom
            # Same cross-device rescale the stock path applies (a no-op at
            # world_size 1, which is every run here — kept so the two paths do
            # not silently diverge if this is ever run multi-GPU).
            if (
                self.args.average_tokens_across_devices
                and self.model_accepts_loss_kwargs
                and num_items_in_batch is not None
            ):
                loss = loss * (
                    self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu
                )
            return (loss, outputs) if return_outputs else loss

        def _record_gap_stats(self, ce_aligned, ce_blank, valid, w):
            """Accumulate on-device; converted to floats once, in ``log``."""
            with torch.no_grad():
                gap = torch.clamp(ce_blank - ce_aligned, min=0.0)[valid]
                st = self._gap_train
                gs = gap.sum()
                wm = w.max()
                st["gap_sum"] = gs if st["gap_sum"] is None else st["gap_sum"] + gs
                st["w_max"] = wm if st["w_max"] is None else torch.maximum(st["w_max"], wm)
                st["tokens"] += int(gap.numel())

        def _accumulate_blank_ce(self, model, inputs, blank, aligned_loss, num_items):
            """One extra no-grad forward on the blank image, reduced EXACTLY like
            the aligned loss the Trainer just computed, so the two are comparable."""
            labels = inputs.get("labels")
            if labels is None:
                return
            n = int((labels != IGNORE_INDEX).sum())
            if n == 0:
                return
            kwargs = {**inputs, **blank}
            if num_items is not None and self.model_accepts_loss_kwargs:
                kwargs["num_items_in_batch"] = num_items
            with torch.no_grad():
                out = model(**kwargs)
            st = self._blank_eval
            st["aligned"] += float(aligned_loss.detach()) * n
            st["blank"] += float(out.loss.detach()) * n
            st["tokens"] += n

        def evaluation_loop(
            self, dataloader, description, prediction_loss_only=None,
            ignore_keys=None, metric_key_prefix="eval",
        ):
            if not log_blank_ce:
                return super().evaluation_loop(
                    dataloader, description, prediction_loss_only,
                    ignore_keys, metric_key_prefix,
                )
            self._reset_blank_eval_stats()
            output = super().evaluation_loop(
                dataloader, description, prediction_loss_only,
                ignore_keys, metric_key_prefix,
            )
            st = self._blank_eval
            if st["tokens"]:
                aligned = st["aligned"] / st["tokens"]
                bl = st["blank"] / st["tokens"]
                base = self._blank_baseline.setdefault(metric_key_prefix, bl)
                output.metrics[f"{metric_key_prefix}_aligned_ce"] = aligned
                output.metrics[f"{metric_key_prefix}_blank_ce"] = bl
                output.metrics[f"{metric_key_prefix}_prior_gap_ce"] = bl - aligned
                output.metrics[f"{metric_key_prefix}_blank_ce_inflation"] = (
                    bl / base - 1.0 if base else float("nan")
                )
            return output

        def log(self, logs, start_time=None):
            st = self._gap_train if blind_branch else None
            if st and st["tokens"] and "loss" in logs:
                logs["gap_prior_gap_mean"] = float(st["gap_sum"]) / st["tokens"]
                logs["gap_weight_max"] = float(st["w_max"])
                self._reset_gap_train_stats()
            super().log(logs, start_time)

    return ThreeGroupTrainer


def main(argv: list[str] | None = None) -> None:
    # force=True is load-bearing: importing torch/transformers installs a root
    # handler, and a bare basicConfig() is a silent no-op once one exists. Without
    # it EVERY logger.info in this module was swallowed — including the WARM-START
    # and pretrained-weight-load confirmations, which then had to be established by
    # deduction after the fact (2026-07-29). Run logs had 0 INFO lines.
    logging.basicConfig(level=logging.INFO, force=True)
    parser = argparse.ArgumentParser(description="Train the pretrained-vision hybrid.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    # The A100 is SHARED with a neighbour container whose footprint GROWS (~15.7 GB
    # on 2026-07-29, 22.5 GB by 2026-07-30). Both H20 legs died of CUDA OOM with our
    # own processes at only 15.9 / 17.5 GB — inside the 26 GB budget, but 22 + 17.5 >
    # 39.49. The OOM landed on us that time; the same arithmetic could land on the
    # neighbour instead, which would break someone else's job. Setting a hard cap
    # makes THIS process fail cleanly rather than take memory the neighbour needs.
    #
    # UNIVI_CUDA_MEM_FRACTION is a fraction of TOTAL device memory (torch's own
    # units), so a launcher that wants a 15 GB ceiling on a 40.96 GB card passes
    # 15/40.96 = 0.366. Unset ⇒ unchanged behaviour.
    frac = os.environ.get("UNIVI_CUDA_MEM_FRACTION")
    if frac and torch.cuda.is_available():
        f = float(frac)
        if not 0.0 < f <= 1.0:
            raise ValueError(f"UNIVI_CUDA_MEM_FRACTION must be in (0, 1], got {f}")
        total = torch.cuda.get_device_properties(0).total_memory
        free, _ = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(f)
        logger.info(
            "CUDA memory cap: fraction=%.4f (%.2f GiB of %.2f GiB total); "
            "%.2f GiB currently free on the device",
            f, f * total / 2**30, total / 2**30, free / 2**30,
        )
        if f * total > free:
            logger.warning(
                "CAP (%.2f GiB) EXCEEDS CURRENTLY-FREE VRAM (%.2f GiB) — another "
                "process holds the difference. This run may OOM, and if it does not "
                "it may starve the neighbour. Re-check memory.free before launching.",
                f * total / 2**30, free / 2**30,
            )

    config = load_config(args.config)
    mc = config["model"]
    tc = config["training"]
    if args.max_steps is not None:
        tc["max_steps"] = args.max_steps

    # Honour the config's ``wandb:`` block. Without this the HF Trainer defaults to
    # project "huggingface" with a random run name, so a config's `run_name`/`project`
    # were silently ignored and runs had to be labelled by exporting WANDB_* by hand.
    # Env vars already set by the launcher win, so an explicit export still overrides.
    wc = config.get("wandb", {})
    if wc.get("project"):
        os.environ.setdefault("WANDB_PROJECT", str(wc["project"]))
    if wc.get("run_name"):
        os.environ.setdefault("WANDB_NAME", str(wc["run_name"]))
    if wc.get("tags"):
        os.environ.setdefault("WANDB_TAGS", ",".join(str(t) for t in wc["tags"]))

    from univi.hybrid.data import HybridCollator
    from univi.hybrid.pretrained import build_pretrained_hybrid
    from univi.hybrid.vision import DEFAULT_MAX_SOFT_TOKENS
    from univi.trainer import _load_eval_datasets, load_dataset

    # H17 knob: per-image soft-token budget. Absent ⇒ 280, the historical default
    # every run before H17 used.
    max_soft_tokens = int(mc.get("max_soft_tokens", DEFAULT_MAX_SOFT_TOKENS))

    logger.info("Assembling PRETRAINED-vision hybrid model")
    model, tokenizer, image_processor, image_token_id = build_pretrained_hybrid(
        vision_source=mc["vision_source"],
        text_source=mc["text_source"],
        vision_weights=mc.get("vision_weights", "data/hybrid/gemma12b-vision/vision_embedder.pt"),
        max_soft_tokens=max_soft_tokens,
    )
    # The collator below reuses THIS processor, so the budget the model records and
    # the budget the pixels are cut at cannot drift apart.
    assert image_processor.max_soft_tokens == max_soft_tokens
    logger.info(
        "Soft-token budget: %d per image (config max_soft_tokens=%d)",
        image_processor.max_soft_tokens, model.config.max_soft_tokens,
    )

    # H13 lesson: the adapter is the ONLY from-scratch part, and it has to bootstrap
    # reading against whatever fraction of the supervised tokens is actually within
    # the model's scan depth. On the density ladder that fraction was 14% and the run
    # collapsed (grad_norm 0.06 by step 110, loss pinned at the no-reading floor).
    # `init_from` starts instead from a checkpoint that ALREADY reads, so a run tests
    # its own variable rather than re-running the bootstrap lottery.
    #
    # Weights only — no optimizer/scheduler state, so the LR schedule starts fresh
    # (this is warm-START, not `resume_from_checkpoint`).
    init_from = mc.get("init_from")
    if init_from:
        from safetensors.torch import load_file as _load_safetensors

        init_path = Path(init_from)
        st = init_path / "model.safetensors"
        if not st.is_file():
            raise FileNotFoundError(f"init_from: {st} not found")
        sd = _load_safetensors(str(st))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # Qwen3 ties lm_head to embed_tokens, so safetensors never serializes it and
        # it ALWAYS reports missing. Tolerate exactly that key; anything else is a
        # real shape/name drift, which would look exactly like "the run just failed to
        # learn" — the failure mode this knob exists to avoid. Refuse those.
        tied = {"language_model.lm_head.weight"}
        real_missing = [k for k in missing if k not in tied]
        if unexpected:
            raise RuntimeError(f"init_from {init_path}: unexpected tensors {unexpected[:8]}")
        if real_missing:
            raise RuntimeError(f"init_from {init_path}: missing tensors {real_missing[:8]}")
        logger.info("WARM-START: loaded %d tensors from %s (weights only, fresh optimizer; "
                    "%d tied key(s) skipped)", len(sd), init_path, len(missing))
    if tc.get("freeze_vision", False):
        model.freeze_vision()
        logger.info("Vision embedder FROZEN (LLaVA-style; trains adapter + decoder only)")
    if tc.get("gradient_checkpointing", True):
        model.config.use_cache = False
        # Feed the decoder inputs_embeds (text + scattered image features); the
        # embedding output must require grad so checkpointing backprops into the
        # vision path.
        if hasattr(model.language_model, "enable_input_require_grads"):
            model.language_model.enable_input_require_grads()
    n_tot = sum(p.numel() for p in model.parameters())
    n_ad = sum(p.numel() for p in model.adapter.parameters())
    n_vis = sum(p.numel() for p in model.vision_tower.parameters())
    logger.info(
        "Model ready: %.0fM params (adapter %.2fM, vision %.1fM, decoder %.0fM); image_token_id=%d",
        n_tot / 1e6, n_ad / 1e6, n_vis / 1e6,
        (n_tot - n_ad - n_vis) / 1e6, image_token_id,
    )

    dataset = load_dataset(config, source="auto")
    logger.info("Train dataset: %d rows", len(dataset))

    eval_sets = _load_eval_datasets(config) if tc.get("do_eval", True) else {}
    eval_dataset = None
    if len(eval_sets) == 1:
        # Single lane: one eval set, metrics logged as eval_loss.
        eval_dataset = next(iter(eval_sets.values()))
        logger.info("Eval dataset: %d rows (%s)", len(eval_dataset), list(eval_sets))
    elif len(eval_sets) > 1:
        # Multi-lane mixture: pass the dict so the Trainer reports a SEPARATE
        # eval_<lane>_loss per subset — the direct read on whether any lane
        # drowns while the others train.
        eval_dataset = eval_sets
        logger.info(
            "Eval datasets (per-lane): %s",
            {k: len(v) for k, v in eval_sets.items()},
        )

    max_length = tc.get("max_length", 2048)
    if max_soft_tokens != DEFAULT_MAX_SOFT_TOKENS:
        from univi.trainer import image_token_budget

        # The dataset filter now derives its per-image budget from
        # model.max_soft_tokens, so it tracks this setting automatically. Logged
        # (not warned) so a non-default run states its real per-image cost.
        logger.info(
            "max_soft_tokens=%d ⇒ dataset length filter budgets %d tokens/image "
            "against max_length=%d.",
            max_soft_tokens, image_token_budget(max_soft_tokens), max_length,
        )

    # H16 knobs. Absent ⇒ off ⇒ the stock loss path, unchanged. ``log_blank_ce``
    # defaults to ON whenever the weighting is on (it is the pre-registered
    # degeneracy guard) but can be enabled alone to measure a control run.
    gap_weighted_loss = bool(tc.get("gap_weighted_loss", False))
    gap_beta = float(tc.get("gap_beta", 1.0))
    log_blank_ce = bool(tc.get("log_blank_ce", gap_weighted_loss))
    if gap_weighted_loss or log_blank_ce:
        logger.info(
            "H16 blind branch ON: gap_weighted_loss=%s beta=%g log_blank_ce=%s "
            "(+1 forward-only blank pass per step)",
            gap_weighted_loss, gap_beta, log_blank_ce,
        )

    collator = HybridCollator(
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_token_id=image_token_id,
        max_length=max_length,
        response_only=tc.get("loss_masking", "response_only") == "response_only",
        emit_blank_images=gap_weighted_loss or log_blank_ce,
    )

    TrainerCls = _make_trainer_class(
        lr_adapter=float(tc.get("lr_adapter", 3e-4)),
        lr_vision=float(tc.get("lr_vision", 5e-5)),
        lr_decoder=float(tc.get("lr_decoder", 2e-5)),
        weight_decay=float(tc.get("weight_decay", 0.01)),
        gap_weighted_loss=gap_weighted_loss,
        gap_beta=gap_beta,
        log_blank_ce=log_blank_ce,
    )
    trainer = TrainerCls(
        model=model,
        args=build_training_args(tc),
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    out = tc["output_dir"] + "/final"
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    image_processor.save_pretrained(out)
    logger.info("Saved to %s", out)


if __name__ == "__main__":
    main()
