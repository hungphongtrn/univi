"""
train_encoder_free.py — a self-contained encoder-free VLM trainer for Univi.

This is a faithful port of the HuggingFace M4 tutorial *"Train Your Own
Encoder-Free VLM in $100"* (https://huggingface.co/spaces/HuggingFaceM4/
encoder-free-vlm), adapted to consume Univi's already-materialized
``messages``+``images`` rows and swapping a few pieces for the memory-frugal
kernels the task asked for:

- **Model** (:class:`EncoderFreeUnivi`): a Qwen 3 1.7B decoder loaded with
  Unsloth ``FastModel`` (``full_finetuning=True``) — *no* pretrained vision
  encoder. Images enter through the tutorial's tiny **embedder**: patchify →
  LayerNorm → one ``Linear`` (pixel space → LM hidden) → LayerNorm → factorized
  positional embeddings → LayerNorm, followed by a 1-layer connector. Where the
  tutorial uses ``nn.LayerNorm`` we use Liger's fused ``LigerLayerNorm`` (the
  "use the Liger-optimized modules where they exist" instruction).
- **Tokenize / image tokens** exactly as the tutorial: every image content part
  is replaced in the chat string by ``num_patches`` copies of the ``<|image|>``
  placeholder; ``apply_chat_template`` then formats the turn, and the image
  placeholders are overwritten by the projected patch embeddings at forward time
  via ``masked_scatter``.
- **Image processing** exactly as the tutorial: resize shorter side to 512,
  center-crop 512×512, to-tensor — so every image is a fixed 16×16 = 256-patch
  grid.
- **Packing** exactly as the tutorial: concatenate whole samples back-to-back
  into fixed-length knapsacks, padding once at the tail. Each knapsack carries
  its members' images in order, so the flattened image-token positions line up
  one-to-one with the flattened patch embeddings.
- **Loss**: next-token cross-entropy with image/pad positions masked to -100,
  computed with :class:`LigerFusedLinearCrossEntropyLoss` — which fuses the LM
  head matmul into the CE kernel and never materializes the ``[N, vocab]`` logit
  tensor (the win that keeps a full-finetune of a 1.7B decoder on one 40 GB GPU).
- **Optimizer**: bitsandbytes ``AdamW8bit``.
- A hand-written PyTorch training loop (grad accumulation, cosine schedule,
  clipping, checkpointing) — no ``Trainer``.

Data is loaded through the repo's existing :func:`univi.trainer.load_dataset`,
so the same materialized mixture feeds this experiment and the others.

Usage::

    uv run python train_encoder_free.py --config configs/encoder_free_smoke.yaml
    uv run python train_encoder_free.py --config configs/encoder_free_smoke.yaml --max-steps 20

Heavy GPU imports (unsloth, bitsandbytes, torch CUDA) are deferred into the
functions/classes that need them so this module imports CPU-only for tests.

Copyright (c) 2026, hungphongtrn.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100

# Standard ChatML template (what Qwen 3 uses). The *base* ``unsloth/Qwen3-1.7B``
# ships no ``chat_template``, so we install this when one is absent — the tutorial
# assumes an instruction-tuned tokenizer whose template is already set.
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class EFConfig:
    """Flattened view of the ``model``/``training`` config knobs used here.

    ``dataset`` is passed straight through to :func:`univi.trainer.load_dataset`,
    so it is kept as the raw dict rather than mirrored field-by-field.
    """

    # model / embedder
    model_name: str = "unsloth/Qwen3-1.7B"
    max_seq_length: int = 8192
    load_in_4bit: bool = True
    image_token: str = "<|image_pad|>"  # Qwen3 already ships this token (no resize)
    image_size: int = 512
    patch_size: int = 32
    # Vision-embedder connector: "linear" (Fuyu-style, original) or "mlp"
    # (Linear→GELU→Linear, the H3 capacity escalation). mlp_ratio sizes the hidden.
    connector: str = "linear"
    mlp_ratio: int = 4
    # "auto" → use the tokenizer's own chat template (Qwen 3's native ChatML);
    # any other value → override via Unsloth's ``get_chat_template`` (e.g.
    # "qwen-2.5" for plain ChatML with no reasoning scaffolding).
    chat_template: str = "auto"

    # optimization
    per_device_train_batch_size: int = 1  # knapsacks per micro-batch
    per_device_eval_batch_size: int = 1   # knapsacks per eval micro-batch
    gradient_accumulation_steps: int = 8
    max_steps: int = -1
    num_train_epochs: int = 1
    max_length: int = 2048  # knapsack length (the tutorial's 2048)
    learning_rate: float = 3.0e-4  # embedder (vision-path) LR
    # Decoder LR. None → reuse ``learning_rate`` (the original single-LR behavior).
    # For reusing a pretrained decoder, set this well below ``learning_rate``
    # (e.g. 2e-5) so the trillion-token prior is refined, not overwritten, while
    # the from-scratch embedder trains fast.
    decoder_lr: float | None = None
    # Vision-path warmup: for the first N optimizer steps, freeze the decoder and
    # train ONLY the embedder (LLaVA stage-1 style alignment), so the vision
    # features mature before the decoder's LM prior can win the gradient race.
    # 0 → no warmup (decoder trainable from step 0).
    freeze_decoder_steps: int = 0
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    reset_position_ids: bool = True

    # evaluation (held-out per-subset validation)
    eval_steps: int = 0                    # 0 → eval only at start+end; >0 → periodic
    max_eval_samples_per_subset: int | None = 64  # small bounded set per subset

    # bookkeeping
    logging_steps: int = 1
    save_steps: int = 1000
    output_dir: str = "data/checkpoints/encoder-free-v0"
    seed: int = 3407
    gradient_checkpointing: bool = True

    dataset: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config: dict) -> "EFConfig":
        mc = config.get("model", {})
        tc = config.get("training", {})
        known = {
            # model
            "model_name": mc.get("model_name", cls.model_name),
            "max_seq_length": mc.get("max_seq_length", cls.max_seq_length),
            "load_in_4bit": mc.get("load_in_4bit", cls.load_in_4bit),
            "image_token": mc.get("image_token", cls.image_token),
            "image_size": mc.get("image_size", cls.image_size),
            "patch_size": mc.get("patch_size", cls.patch_size),
            "connector": mc.get("connector", cls.connector),
            "mlp_ratio": mc.get("mlp_ratio", cls.mlp_ratio),
            "chat_template": mc.get("chat_template", cls.chat_template),
            # training
            "per_device_train_batch_size": tc.get(
                "per_device_train_batch_size", cls.per_device_train_batch_size
            ),
            "per_device_eval_batch_size": tc.get(
                "per_device_eval_batch_size", cls.per_device_eval_batch_size
            ),
            "gradient_accumulation_steps": tc.get(
                "gradient_accumulation_steps", cls.gradient_accumulation_steps
            ),
            "max_steps": tc.get("max_steps", cls.max_steps),
            "num_train_epochs": tc.get("num_train_epochs", cls.num_train_epochs),
            "max_length": tc.get("max_length", cls.max_length),
            "learning_rate": float(tc.get("learning_rate", cls.learning_rate)),
            "decoder_lr": (
                float(tc["decoder_lr"]) if tc.get("decoder_lr") is not None else None
            ),
            "freeze_decoder_steps": tc.get(
                "freeze_decoder_steps", cls.freeze_decoder_steps
            ),
            "warmup_ratio": tc.get("warmup_ratio", cls.warmup_ratio),
            "weight_decay": tc.get("weight_decay", cls.weight_decay),
            "max_grad_norm": tc.get("max_grad_norm", cls.max_grad_norm),
            "reset_position_ids": tc.get("reset_position_ids", cls.reset_position_ids),
            "eval_steps": tc.get("eval_steps", cls.eval_steps),
            "max_eval_samples_per_subset": tc.get(
                "max_eval_samples_per_subset", cls.max_eval_samples_per_subset
            ),
            "logging_steps": tc.get("logging_steps", cls.logging_steps),
            "save_steps": tc.get("save_steps", cls.save_steps),
            "output_dir": tc.get("output_dir", cls.output_dir),
            "seed": tc.get("seed", cls.seed),
            "gradient_checkpointing": tc.get(
                "gradient_checkpointing", cls.gradient_checkpointing
            ),
        }
        return cls(dataset=config.get("dataset", {}), **known)


# ---------------------------------------------------------------------------
# The embedder (tutorial: "Building the Embedder")
# ---------------------------------------------------------------------------


def _layer_norm(dim: int):
    """LayerNorm, preferring Liger's fused kernel per the task instruction."""
    from torch import nn

    try:
        from liger_kernel.transformers import LigerLayerNorm

        return LigerLayerNorm(dim)
    except Exception:  # pragma: no cover - Liger optional / CPU fallback
        return nn.LayerNorm(dim)


def _build_vision_embedder(
    image_size: int,
    patch_size: int,
    hidden_size: int,
    connector: str = "linear",
    mlp_ratio: int = 4,
):
    """Construct the encoder-free image embedder as an ``nn.Module``.

    patchify → LayerNorm → Linear(pixel→hidden) → LayerNorm → + factorized
    positional embeddings → LayerNorm, then a connector.
    Defined inside a function so ``torch``/``torch.nn`` stay deferred imports.

    ``connector``:
      * ``"linear"`` (default) — the original Fuyu-style single ``Linear``
        (hidden→hidden), NO non-linearity; the decoder supplies all non-linearity.
      * ``"mlp"`` — a 2-layer MLP ``Linear(D→D·mlp_ratio) → GELU → Linear(→D)``.
        The H3 escalation: gives the from-scratch vision path its own non-linear
        capacity so it isn't wholly out-competed by the pretrained decoder. Keeps
        the lower layers (patch proj + pos) warm-start-compatible with H1 — only
        the ``connector.*`` keys change shape, so a strict=False embedder warm-start
        reuses everything up to the connector and fresh-inits the MLP.
    """
    import torch
    from torch import nn

    class VisionEmbedder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            assert image_size % patch_size == 0, "image_size must divide patch_size"
            self.patch_size = patch_size
            self.grid = image_size // patch_size          # 16
            self.num_patches = self.grid * self.grid       # 256
            patch_dim = 3 * patch_size * patch_size        # 3 * 32 * 32 = 3072

            self.ln1 = _layer_norm(patch_dim)              # normalize each patch
            self.fc = nn.Linear(patch_dim, hidden_size)    # pixel space → LM dim
            self.ln2 = _layer_norm(hidden_size)            # normalize projection
            # Factorized positional tables (row + col), tutorial's 8×-cheaper form.
            self.y_pos_emb = nn.Parameter(torch.zeros(1, self.grid, hidden_size))
            self.x_pos_emb = nn.Parameter(torch.zeros(1, self.grid, hidden_size))
            self.ln3 = _layer_norm(hidden_size)            # final normalization
            # Connector: linear (Fuyu-style, original) or a non-linear MLP (H3 escalation).
            if connector == "mlp":
                self.connector = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * mlp_ratio),
                    nn.GELU(),
                    nn.Linear(hidden_size * mlp_ratio, hidden_size),
                )
            elif connector == "linear":
                self.connector = nn.Linear(hidden_size, hidden_size)
            else:
                raise ValueError(f"unknown connector {connector!r} (want 'linear' or 'mlp')")

            nn.init.normal_(self.y_pos_emb, std=0.02)
            nn.init.normal_(self.x_pos_emb, std=0.02)

        @staticmethod
        def _extract_flattened_patches(x: "torch.Tensor", P: int) -> "torch.Tensor":
            # (B, C, H, W) → (B, N, C·P²) via one reshape → permute → reshape.
            B, C, H, W = x.shape
            nh, nw = H // P, W // P
            x = x.reshape(B, C, nh, P, nw, P)
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(B, nh * nw, C * P * P)
            return x

        def forward(self, pixel_values: "torch.Tensor") -> "torch.Tensor":
            # pixel_values: (num_images, 3, image_size, image_size)
            x = self._extract_flattened_patches(pixel_values, self.patch_size)
            x = self.ln1(x)
            x = self.fc(x)
            x = self.ln2(x)
            nh = nw = self.grid
            row = self.y_pos_emb[:, :nh, :]                # (1, nh, D)
            col = self.x_pos_emb[:, :nw, :]                # (1, nw, D)
            pos = (row.unsqueeze(2) + col.unsqueeze(1)).reshape(1, nh * nw, -1)
            x = x + pos
            x = self.ln3(x)
            x = self.connector(x)                          # (num_images, N, D)
            return x

    return VisionEmbedder()


# ---------------------------------------------------------------------------
# The model (tutorial: "Mixing Images and Text")
# ---------------------------------------------------------------------------


def build_encoder_free_univi(cfg: EFConfig):
    """Load Qwen 3 via Unsloth ``FastModel`` and attach the vision embedder.

    Deferred entirely inside this function — it triggers the heavy Unsloth /
    torch-CUDA import path and downloads weights.
    """
    import torch
    from torch import nn
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        full_finetuning=True,
        # Unsloth's offloading gradient checkpointing ("smart" gradient offload +
        # double-buffered parallel H2D/compute) — strictly more VRAM-frugal than
        # stock HF checkpointing. Configured at load so the decoder is patched
        # once. FlashAttention-2 is auto-selected by Unsloth (no separate knob).
        use_gradient_checkpointing=(
            "unsloth" if cfg.gradient_checkpointing else False
        ),
    )

    # Chat template resolution:
    #   * "auto" (default) → use the tokenizer's own template. Qwen 3's instruct
    #     tokenizer ships its native ChatML template, so this is what training
    #     sees. (Only if the tokenizer genuinely has none — e.g. an incomplete
    #     cache — do we install the hand-written ChatML fallback.)
    #   * any other value → override via Unsloth's ``get_chat_template`` helper
    #     (e.g. "qwen-2.5" for plain ChatML with no reasoning scaffolding).
    if cfg.chat_template and cfg.chat_template != "auto":
        from unsloth.chat_templates import get_chat_template

        tokenizer = get_chat_template(tokenizer, chat_template=cfg.chat_template)
    elif getattr(tokenizer, "chat_template", None) is None:
        logger.warning(
            "tokenizer has no chat template (incomplete cache?); "
            "installing ChatML fallback"
        )
        tokenizer.chat_template = CHATML_TEMPLATE

    # Reserve the image placeholder token. Qwen 3's tokenizer already ships the
    # unused vision tokens (``<|image_pad|>`` etc.), so the common path reuses an
    # existing embedding row and never touches the table. Only a genuinely novel
    # token triggers a resize (+ mean warm-start of the new row).
    image_token_id = tokenizer.convert_tokens_to_ids(cfg.image_token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if image_token_id is None or image_token_id == unk_id:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [cfg.image_token]}
        )
        model.resize_token_embeddings(len(tokenizer))
        image_token_id = tokenizer.convert_tokens_to_ids(cfg.image_token)
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            emb[image_token_id] = emb[:image_token_id].mean(dim=0)

    hidden_size = model.config.hidden_size
    param_dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    embedder = _build_vision_embedder(
        cfg.image_size, cfg.patch_size, hidden_size,
        connector=cfg.connector, mlp_ratio=cfg.mlp_ratio,
    )
    embedder = embedder.to(device=device, dtype=param_dtype)

    # Warm-start the embedder when ``model_name`` points at a prior checkpoint dir
    # (i.e. it ships an ``embedder.pt`` beside the decoder weights). The decoder is
    # already warm-started by ``FastModel.from_pretrained`` above; loading the
    # matching embedder pairs them back into the same basin they were saved in.
    # Base HF repos have no ``embedder.pt`` → this is a no-op for fresh runs.
    _emb_ckpt = Path(cfg.model_name) / "embedder.pt"
    if _emb_ckpt.is_file():
        sd = torch.load(_emb_ckpt, map_location="cpu")
        before = embedder.fc.weight.detach().float().norm().item()
        missing, unexpected = embedder.load_state_dict(sd, strict=False)
        embedder = embedder.to(device=device, dtype=param_dtype)
        after = embedder.fc.weight.detach().float().norm().item()
        logger.info(
            "Warm-started embedder from %s (fc.weight norm %.4f → %.4f, "
            "missing=%s unexpected=%s)",
            _emb_ckpt, before, after, missing, unexpected,
        )

    class EncoderFreeUnivi(nn.Module):
        """Qwen 3 decoder + tiny pixel embedder, merged via ``masked_scatter``."""

        def __init__(self) -> None:
            super().__init__()
            self.model = model                 # Unsloth-wrapped Qwen3ForCausalLM
            self.tokenizer = tokenizer
            self.embedder = embedder
            self.image_token_id = image_token_id
            self.num_patches = embedder.num_patches

        def get_lm_head_weight(self) -> "torch.Tensor":
            return self.model.get_output_embeddings().weight

        def forward(
            self,
            input_ids: "torch.Tensor",
            attention_mask: "torch.Tensor",
            pixel_values: "torch.Tensor | None" = None,
            position_ids: "torch.Tensor | None" = None,
        ) -> "torch.Tensor":
            """Return the decoder's last hidden state (LM head fused into loss)."""
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

            if pixel_values is not None and pixel_values.numel() > 0:
                patch_embeds = self.embedder(
                    pixel_values.to(device=inputs_embeds.device, dtype=self.embedder.fc.weight.dtype)
                )
                patch_embeds = patch_embeds.reshape(-1, patch_embeds.shape[-1])
                mask = input_ids == self.image_token_id
                n_slots = int(mask.sum())
                if n_slots != patch_embeds.shape[0]:
                    raise ValueError(
                        f"image placeholder count ({n_slots}) != projected patch "
                        f"embeddings ({patch_embeds.shape[0]}) — packing/patch "
                        "accounting is out of sync."
                    )
                inputs_embeds = inputs_embeds.clone()
                inputs_embeds[mask] = patch_embeds.to(inputs_embeds.dtype)

            outputs = self.model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            return outputs.last_hidden_state

    vlm = EncoderFreeUnivi()
    return vlm, tokenizer, image_token_id


# ---------------------------------------------------------------------------
# Tokenize + pack (tutorial: "Mixing Images..." + "Packing samples into batches")
# ---------------------------------------------------------------------------


def _message_to_string(content: Any, image_token_block: str) -> str:
    """Flatten one message's content parts into a single string.

    Image content parts are replaced *in place* by ``image_token_block`` (the
    placeholder repeated ``num_patches`` times); text parts pass through. Keeping
    images in position (rather than always prepending to the first turn) supports
    the multi-part / multi-image rows the Univi mixture contains.
    """
    if isinstance(content, str):
        return content
    out: list[str] = []
    for part in content:
        if part.get("type") == "image":
            out.append(image_token_block)
        else:
            out.append(part.get("text") or "")
    return "".join(out)


def tokenize_sample(
    messages: list[dict], tokenizer, image_token: str, num_patches: int
) -> list[int]:
    """Tutorial's ``tokenize_sample``: build the chat string, then encode.

    Each image becomes ``num_patches`` copies of the ``<|image|>`` string, so
    ``input_ids.count(image_token_id) == num_patches × (#images in the row)``.
    """
    block = image_token * num_patches
    chat = [
        {"role": m["role"], "content": _message_to_string(m["content"], block)}
        for m in messages
    ]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    return tokenizer.encode(text, add_special_tokens=False)


@dataclass
class PackedKnapsack:
    """One fixed-length training sequence assembled from ≥1 samples."""

    input_ids: list[int]
    seg_lengths: list[int]  # token length of each member sample (for position ids)
    images: list[Any]       # PIL images, in the order their placeholders appear


class KnapsackPacker:
    """Greedy sequential packer (the tutorial's line-193 description).

    Walk the (shuffled) rows, tokenizing each; keep concatenating whole samples
    into the current knapsack until the next one would overflow ``max_length``,
    then emit the knapsack and start a new one. A single oversized sample is
    truncated to ``max_length`` and emitted on its own.
    """

    def __init__(self, tokenizer, image_token: str, num_patches: int, max_length: int):
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        self.num_patches = num_patches
        self.max_length = max_length

    def _row_images(self, row: dict) -> list[Any]:
        return [im.convert("RGB") for im in (row.get("images") or [])]

    def pack(self, rows: Iterator[dict]) -> Iterator[PackedKnapsack]:
        cur_ids: list[int] = []
        cur_segs: list[int] = []
        cur_imgs: list[Any] = []

        for row in rows:
            ids = tokenize_sample(
                row["messages"], self.tokenizer, self.image_token, self.num_patches
            )
            imgs = self._row_images(row)
            # Defensive: a row's placeholder count must match its image count.
            n_ph = sum(1 for t in ids if t == self.image_token_id)
            if n_ph != len(imgs) * self.num_patches:
                logger.warning(
                    "skipping row: %d placeholders != %d images × %d patches",
                    n_ph, len(imgs), self.num_patches,
                )
                continue

            if len(ids) > self.max_length:
                # Truncating a sample would sever an image's placeholder block
                # from its pixels; drop oversized rows rather than corrupt the
                # image↔placeholder alignment.
                logger.warning(
                    "skipping oversized sample (%d > max_length %d)",
                    len(ids), self.max_length,
                )
                continue

            if cur_ids and len(cur_ids) + len(ids) > self.max_length:
                yield PackedKnapsack(cur_ids, cur_segs, cur_imgs)
                cur_ids, cur_segs, cur_imgs = [], [], []

            cur_ids += ids
            cur_segs.append(len(ids))
            cur_imgs += imgs

        if cur_ids:
            yield PackedKnapsack(cur_ids, cur_segs, cur_imgs)


def _image_transform(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(image_size),      # shorter side → image_size
            transforms.CenterCrop(image_size),  # exact image_size × image_size
            transforms.ToTensor(),              # H×W×C uint8 → C×H×W float[0,1]
        ]
    )


class PackedIterableDataset:
    """Torch ``IterableDataset`` yielding padded, packed training tensors.

    Reshuffles the underlying map-style HF dataset each epoch (seeded), tokenizes
    + packs on the fly, and pads each knapsack once to ``max_length``. Labels copy
    ``input_ids`` with image and pad positions masked to ``-100`` (the tutorial's
    masking). ``position_ids`` restart at 0 per member sample when
    ``reset_position_ids`` is set, so RoPE treats packed members independently.
    """

    def __init__(self, hf_dataset, tokenizer, cfg: EFConfig, num_patches: int):
        import torch
        from torch.utils.data import IterableDataset

        self._torch = torch
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.num_patches = num_patches
        self.transform = _image_transform(cfg.image_size)
        self.image_token_id = tokenizer.convert_tokens_to_ids(cfg.image_token)
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id
        self._epoch = 0

        # Make this object an IterableDataset without a module-level torch import.
        self.__class__ = type(
            "PackedIterableDataset", (PackedIterableDataset, IterableDataset), {}
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _rows(self) -> Iterator[dict]:
        ds = self.hf_dataset.shuffle(seed=self.cfg.seed + self._epoch)
        for i in range(len(ds)):
            yield ds[i]

    def __iter__(self):
        torch = self._torch
        packer = KnapsackPacker(
            self.tokenizer, self.cfg.image_token, self.num_patches, self.cfg.max_length
        )
        L = self.cfg.max_length
        for kn in packer.pack(self._rows()):
            ids = kn.input_ids
            n = len(ids)

            input_ids = torch.full((L,), self.pad_id, dtype=torch.long)
            input_ids[:n] = torch.tensor(ids, dtype=torch.long)

            attention_mask = torch.zeros((L,), dtype=torch.long)
            attention_mask[:n] = 1

            labels = input_ids.clone()
            labels[labels == self.image_token_id] = IGNORE_INDEX
            labels[labels == self.pad_id] = IGNORE_INDEX
            labels[n:] = IGNORE_INDEX  # guard: pad_id may equal a real token id

            if self.cfg.reset_position_ids:
                pos = torch.zeros((L,), dtype=torch.long)
                cursor = 0
                for seg in kn.seg_lengths:
                    pos[cursor : cursor + seg] = torch.arange(seg)
                    cursor += seg
            else:
                pos = torch.arange(L, dtype=torch.long)

            pixel_values = None
            if kn.images:
                pixel_values = torch.stack([self.transform(im) for im in kn.images])

            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "position_ids": pos,
                "pixel_values": pixel_values,
            }


def collate_packed(features: list[dict]) -> dict:
    """Stack same-length knapsacks; concatenate their images batch-then-order."""
    import torch

    batch = {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
        "position_ids": torch.stack([f["position_ids"] for f in features]),
    }
    pix = [f["pixel_values"] for f in features if f["pixel_values"] is not None]
    batch["pixel_values"] = torch.cat(pix, dim=0) if pix else None
    return batch


# ---------------------------------------------------------------------------
# Forward + loss (shared by the training loop and the eval loop)
# ---------------------------------------------------------------------------


def _forward_loss(vlm, batch: dict, device: str, use_amp: bool, amp_dtype, loss_fn):
    """Run one micro-batch forward and the fused-linear CE loss.

    Returns ``(loss, n_valid_tokens)`` where ``loss`` is the mean CE over the
    micro-batch's supervised tokens and ``n_valid_tokens`` is how many there
    were — the caller multiplies them back out to accumulate a token-weighted
    corpus mean (so eval batches of unequal supervised length combine correctly).
    """
    import torch

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    position_ids = batch["position_ids"].to(device)
    pixel_values = batch["pixel_values"]
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)

    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        hidden = vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            position_ids=position_ids,
        )
    # Next-token shift; fuse LM head into the CE kernel (never build logits).
    hidden_size = hidden.shape[-1]
    shift_hidden = hidden[:, :-1, :].reshape(-1, hidden_size)
    shift_labels = labels[:, 1:].reshape(-1)
    # Match the LM-head weight dtype to the (autocast) hidden states; the cast
    # stays differentiable so grads reach the fp32 parameter.
    lm_w = vlm.get_lm_head_weight().to(shift_hidden.dtype)
    loss = loss_fn(lm_w, shift_hidden, shift_labels)
    n_valid = int((shift_labels != IGNORE_INDEX).sum())
    return loss, n_valid


def evaluate(
    vlm, tokenizer, eval_sets: dict, cfg: EFConfig, loss_fn, device: str,
    use_amp: bool, amp_dtype,
) -> tuple[dict, float]:
    """Compute held-out CE loss on a small bounded set from each subset.

    Each subset's validation rows are tokenized + packed with the *same*
    machinery as training (so the loss is directly comparable), then run through
    the model with grads disabled. Returns ``(per_subset_loss, macro_mean)``
    where ``macro_mean`` averages the per-subset losses — the macro-average that
    the rest of the repo treats as ``mean_total_loss`` (``univi/evaluation.py``),
    so audio isn't drowned by the larger text/caption subsets.
    """
    import torch
    from torch.utils.data import DataLoader

    if not eval_sets:
        return {}, float("nan")

    was_training = vlm.training
    vlm.eval()
    results: dict[str, float] = {}
    with torch.no_grad():
        for name, ds in eval_sets.items():
            packed = PackedIterableDataset(ds, tokenizer, cfg, vlm.num_patches)
            packed.set_epoch(0)  # deterministic packing for a stable eval number
            loader = DataLoader(
                packed,
                batch_size=cfg.per_device_eval_batch_size,
                collate_fn=collate_packed,
                num_workers=0,
                pin_memory=(device == "cuda"),
            )
            tot_loss, tot_tok = 0.0, 0
            for batch in loader:
                loss, n = _forward_loss(
                    vlm, batch, device, use_amp, amp_dtype, loss_fn
                )
                tot_loss += loss.item() * n
                tot_tok += n
            results[name] = tot_loss / max(tot_tok, 1)
    if was_training:
        vlm.train()
    macro = sum(results.values()) / len(results) if results else float("nan")
    return results, macro


def _log_eval(global_step: int, total_steps: int, per_subset: dict, macro: float) -> None:
    parts = " | ".join(f"{k} {v:.4f}" for k, v in sorted(per_subset.items()))
    logger.info(
        "[eval] step %d/%d | macro %.4f | %s",
        global_step, total_steps, macro, parts,
    )


def _init_wandb(config: dict, cfg: EFConfig, total_steps: int):
    """Start a W&B run for the hand-written loop (best-effort, never fatal).

    Enabled when ``training.report_to`` contains ``"wandb"`` (the repo default).
    A missing package, absent credentials, or an init failure downgrades to a
    warning and returns ``None`` so training proceeds untracked.
    """
    tc = config.get("training", {})
    if "wandb" not in tc.get("report_to", ["wandb"]):
        return None
    try:
        import wandb
    except Exception:  # pragma: no cover - wandb optional
        logger.warning("wandb not installed; training will run untracked")
        return None

    wc = config.get("wandb", {})
    run_config = {
        # the knobs that define this run, flattened for the W&B config panel
        "model_name": cfg.model_name,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "effective_batch_size": (
            cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
        ),
        "learning_rate": cfg.learning_rate,
        "warmup_ratio": cfg.warmup_ratio,
        "weight_decay": cfg.weight_decay,
        "max_grad_norm": cfg.max_grad_norm,
        "max_length": cfg.max_length,
        "image_size": cfg.image_size,
        "patch_size": cfg.patch_size,
        "num_train_epochs": cfg.num_train_epochs,
        "max_steps": cfg.max_steps,
        "total_steps": total_steps,
        "seed": cfg.seed,
        "gradient_checkpointing": cfg.gradient_checkpointing,
    }
    try:
        run = wandb.init(
            project=wc.get("project", "univi-encoder-free"),
            entity=wc.get("entity"),
            name=wc.get("run_name"),
            tags=wc.get("tags", []),
            mode=wc.get("mode", "online"),
            config=run_config,
        )
        logger.info("W&B run started: %s", getattr(run, "name", "?"))
        return run
    except Exception as exc:  # pragma: no cover - network/credentials
        logger.warning("wandb.init failed (%s); training will run untracked", exc)
        return None


# ---------------------------------------------------------------------------
# Training loop (tutorial: "Actual training!!!")
# ---------------------------------------------------------------------------


def train(config: dict, max_steps_override: int | None = None) -> None:
    import torch
    from torch.utils.data import DataLoader
    import bitsandbytes as bnb
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    from transformers import get_cosine_schedule_with_warmup

    from univi.trainer import _load_eval_datasets, load_dataset

    cfg = EFConfig.from_yaml(config)
    if max_steps_override is not None:
        cfg.max_steps = max_steps_override

    torch.manual_seed(cfg.seed)
    # TF32 matmuls (Ampere+): speeds the fp32 embedding / LM-head GEMMs Unsloth
    # keeps in fp32, at no meaningful accuracy cost for this training.
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- model -------------------------------------------------------------
    logger.info("Building encoder-free model on %s", cfg.model_name)
    vlm, tokenizer, image_token_id = build_encoder_free_univi(cfg)
    if cfg.gradient_checkpointing:
        # Gradient checkpointing (Unsloth offload mode) is configured at load in
        # build_encoder_free_univi. Here we only enforce the two preconditions
        # the loop needs and add a fallback if the load-time enable didn't stick.
        #
        # use_cache and checkpointing are mutually exclusive — the backward
        # recompute is incompatible with a populated KV cache.
        vlm.model.config.use_cache = False
        if not getattr(vlm.model.model, "gradient_checkpointing", False):
            # Fallback (e.g. older Unsloth): turn on stock HF checkpointing so we
            # still get the memory saving.
            if hasattr(vlm.model, "gradient_checkpointing_enable"):
                vlm.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
        # Checkpointing recomputes each block's activations during backward; for
        # that recomputed graph to connect, the *entry* activations must require
        # grad. We feed the decoder ``inputs_embeds`` directly (masked_scatter of
        # text + patch embeds), so register the canonical hook that forces the
        # embedding output to require grad — otherwise checkpointing can silently
        # produce no gradient for the layers.
        if hasattr(vlm.model, "enable_input_require_grads"):
            vlm.model.enable_input_require_grads()
    n_params = sum(p.numel() for p in vlm.parameters())
    n_train = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    logger.info(
        "Model ready: %.0fM params (%.0fM trainable), image_token_id=%d, "
        "num_patches=%d",
        n_params / 1e6, n_train / 1e6, image_token_id, vlm.num_patches,
    )

    # --- data --------------------------------------------------------------
    hf_dataset = load_dataset(config, source="auto")
    logger.info("Dataset loaded: %d rows", len(hf_dataset))
    packed = PackedIterableDataset(hf_dataset, tokenizer, cfg, vlm.num_patches)
    loader = DataLoader(
        packed,
        batch_size=cfg.per_device_train_batch_size,
        collate_fn=collate_packed,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # Small bounded held-out set per subset (training.validation_subsets +
    # training.max_eval_samples_per_subset). Empty dict ⇒ eval is skipped.
    eval_sets = _load_eval_datasets(config)
    if eval_sets:
        logger.info(
            "Eval sets: %s",
            {k: len(v) for k, v in eval_sets.items()},
        )

    # --- optimizer / schedule ---------------------------------------------
    # Two param groups so the from-scratch embedder and the pretrained decoder can
    # learn at different rates: embedder at ``learning_rate`` (fast), decoder at
    # ``decoder_lr`` (slow — refine the trillion-token prior, don't overwrite it).
    # ``decoder_lr=None`` reproduces the original single-LR behavior.
    decoder_lr = cfg.decoder_lr if cfg.decoder_lr is not None else cfg.learning_rate
    embedder_param_ids = {id(p) for p in vlm.embedder.parameters()}
    embedder_params = [p for p in vlm.parameters() if id(p) in embedder_param_ids]
    decoder_params = [p for p in vlm.parameters() if id(p) not in embedder_param_ids]
    optimizer = bnb.optim.AdamW8bit(
        [
            {"params": embedder_params, "lr": cfg.learning_rate},
            {"params": decoder_params, "lr": decoder_lr},
        ],
        weight_decay=cfg.weight_decay,
    )
    # Vision-path warmup: freeze the decoder for the first ``freeze_decoder_steps``
    # optimizer steps (train the embedder only), then unfreeze in the loop. Frozen
    # params get no grad, so the optimizer skips them; gradients still flow THROUGH
    # the decoder to the embedder (activations require grad via the input hook).
    decoder_frozen = cfg.freeze_decoder_steps > 0
    if decoder_frozen:
        for p in decoder_params:
            p.requires_grad_(False)
        logger.info(
            "Vision-path warmup: decoder frozen for the first %d steps "
            "(embedder-only; %.1fM embedder params trainable)",
            cfg.freeze_decoder_steps,
            sum(p.numel() for p in embedder_params) / 1e6,
        )
    # The packer's knapsack count is data-dependent, so target step counts come
    # from max_steps when set; otherwise estimate one epoch conservatively.
    if cfg.max_steps and cfg.max_steps > 0:
        total_steps = cfg.max_steps
    else:
        est_knaps = max(1, len(hf_dataset) // 4)  # rough packing factor
        total_steps = math.ceil(
            est_knaps
            / (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)
            * cfg.num_train_epochs
        )
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = LigerFusedLinearCrossEntropyLoss(ignore_index=IGNORE_INDEX)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Training: total_steps=%d warmup=%d grad_accum=%d micro_bs=%d | "
        "lr_embedder=%.2e lr_decoder=%.2e freeze_decoder_steps=%d",
        total_steps, warmup_steps, cfg.gradient_accumulation_steps,
        cfg.per_device_train_batch_size,
        cfg.learning_rate, decoder_lr, cfg.freeze_decoder_steps,
    )

    # --- loop --------------------------------------------------------------
    # Unsloth keeps the embedding / LM-head matrix in fp32 while the transformer
    # body is bf16, and relies on autocast to reconcile them (``Trainer`` sets
    # this up; a hand-written loop must do it explicitly).
    use_amp = device == "cuda"
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    import time

    run = _init_wandb(config, cfg, total_steps)

    vlm.train()
    global_step = 0
    micro = 0
    accum_loss = 0.0
    step_tokens = 0          # tokens seen in the current optimizer step (throughput)
    best_macro = float("inf")
    optimizer.zero_grad(set_to_none=True)
    done = False
    t_step = time.perf_counter()

    # Baseline eval before any optimizer step (a random-init reference point).
    if eval_sets:
        per_subset, macro = evaluate(
            vlm, tokenizer, eval_sets, cfg, loss_fn, device, use_amp, amp_dtype
        )
        _log_eval(0, total_steps, per_subset, macro)
        if run is not None:
            metrics = {f"eval/{k}_loss": v for k, v in per_subset.items()}
            metrics["eval/mean_total_loss"] = macro
            run.log(metrics, step=0)

    for epoch in range(cfg.num_train_epochs):
        packed.set_epoch(epoch)
        for batch in loader:
            loss, _ = _forward_loss(
                vlm, batch, device, use_amp, amp_dtype, loss_fn
            )

            (loss / cfg.gradient_accumulation_steps).backward()
            accum_loss += loss.item()
            step_tokens += batch["input_ids"].numel()
            micro += 1

            if micro % cfg.gradient_accumulation_steps == 0:
                # clip_grad_norm_ both caps the norm at max_grad_norm (=1.0) and
                # returns the pre-clip total norm — the value we log.
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    vlm.parameters(), cfg.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # End of vision-path warmup: unfreeze the decoder so it starts
                # refining (at its slower group LR) alongside the now-aligned
                # embedder.
                if decoder_frozen and global_step >= cfg.freeze_decoder_steps:
                    for p in decoder_params:
                        p.requires_grad_(True)
                    decoder_frozen = False
                    logger.info(
                        "Vision-path warmup complete at step %d — decoder unfrozen "
                        "(decoder lr=%.2e)",
                        global_step, decoder_lr,
                    )

                now = time.perf_counter()
                tokens_per_sec = step_tokens / max(now - t_step, 1e-6)
                t_step = now
                lrs = scheduler.get_last_lr()
                lr_emb, lr_dec = lrs[0], lrs[-1]

                if global_step % cfg.logging_steps == 0:
                    avg = accum_loss / cfg.gradient_accumulation_steps
                    logger.info(
                        "step %d/%d | loss %.4f | grad_norm %.3f | lr_emb %.2e | "
                        "lr_dec %.2e%s | %.0f tok/s",
                        global_step, total_steps, avg, float(grad_norm),
                        lr_emb, lr_dec, " (frozen)" if decoder_frozen else "",
                        tokens_per_sec,
                    )
                    if run is not None:
                        run.log(
                            {
                                "train/loss": avg,
                                "train/grad_norm": float(grad_norm),
                                "train/lr": lr_emb,
                                "train/lr_decoder": lr_dec,
                                "train/decoder_frozen": int(decoder_frozen),
                                "train/tokens_per_sec": tokens_per_sec,
                                "train/epoch": epoch,
                            },
                            step=global_step,
                        )
                accum_loss = 0.0
                step_tokens = 0

                if cfg.save_steps and global_step % cfg.save_steps == 0:
                    _save(vlm, tokenizer, out_dir / f"checkpoint-{global_step}")

                if (
                    eval_sets
                    and cfg.eval_steps
                    and global_step % cfg.eval_steps == 0
                ):
                    per_subset, macro = evaluate(
                        vlm, tokenizer, eval_sets, cfg, loss_fn, device,
                        use_amp, amp_dtype,
                    )
                    _log_eval(global_step, total_steps, per_subset, macro)
                    if run is not None:
                        metrics = {
                            f"eval/{k}_loss": v for k, v in per_subset.items()
                        }
                        metrics["eval/mean_total_loss"] = macro
                        run.log(metrics, step=global_step)
                    if macro < best_macro:
                        best_macro = macro
                        logger.info(
                            "[eval] new best macro %.4f — saving best/", macro
                        )
                        _save(vlm, tokenizer, out_dir / "best")

                if cfg.max_steps and cfg.max_steps > 0 and global_step >= cfg.max_steps:
                    done = True
                    break
        if done:
            break

    # Final eval + checkpoint.
    if eval_sets:
        per_subset, macro = evaluate(
            vlm, tokenizer, eval_sets, cfg, loss_fn, device, use_amp, amp_dtype
        )
        _log_eval(global_step, total_steps, per_subset, macro)
        if run is not None:
            metrics = {f"eval/{k}_loss": v for k, v in per_subset.items()}
            metrics["eval/mean_total_loss"] = macro
            run.log(metrics, step=global_step)
        if macro < best_macro:
            best_macro = macro
            _save(vlm, tokenizer, out_dir / "best")
    _save(vlm, tokenizer, out_dir / "final")
    logger.info("Done. Final checkpoint at %s", out_dir / "final")
    if run is not None:
        run.finish()


def _save(vlm, tokenizer, path: Path) -> None:
    import torch

    path.mkdir(parents=True, exist_ok=True)
    # Persist the Unsloth decoder + the bespoke embedder separately: the decoder
    # via HF save_pretrained, the embedder as a plain state dict (it is not an
    # HF module). Reload pairs them back up.
    vlm.model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    torch.save(vlm.embedder.state_dict(), path / "embedder.pt")
    logger.info("Saved checkpoint to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Train the encoder-free Univi VLM.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    train(config, max_steps_override=args.max_steps)


if __name__ == "__main__":
    main()
