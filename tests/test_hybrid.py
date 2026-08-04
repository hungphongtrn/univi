"""CPU tests for the issue-#6 random-init hybrid model.

These use tiny-but-structurally-faithful configs (small embedding/position-table
widths) so they run on a CPU box in seconds. The real (vendored) Gemma 4 12B
image processor and Qwen 3 tokenizer are used so the soft-token accounting is
exercised exactly as in training.
"""

from __future__ import annotations

import pytest
import torch

from transformers import Qwen3Config

from univi.hybrid.model import (
    UniViHybridConfig,
    UniViHybridForConditionalGeneration,
)
from univi.hybrid.data import HybridCollator, IGNORE_INDEX, blank_like
from univi.hybrid.vision import Gemma4UnifiedImageProcessor, Gemma4UnifiedVisionConfig

QWEN_SRC = "data/hybrid/qwen3-1.7b"


def _tiny_vision() -> Gemma4UnifiedVisionConfig:
    return Gemma4UnifiedVisionConfig(
        patch_size=16,
        pooling_kernel_size=3,
        mm_embed_dim=64,
        mm_posemb_size=64,
        output_proj_dims=64,
    )


def _tiny_text(vocab_size: int = 1000) -> Qwen3Config:
    return Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=vocab_size,
        tie_word_embeddings=True,
    )


def _tiny_model(image_token_id: int, vocab_size: int = 1000):
    cfg = UniViHybridConfig(
        vision_config=_tiny_vision(),
        text_config=_tiny_text(vocab_size),
        image_token_id=image_token_id,
    )
    return UniViHybridForConditionalGeneration(cfg)


def test_config_composes_subconfigs():
    cfg = UniViHybridConfig(
        vision_config=_tiny_vision(), text_config=_tiny_text(), image_token_id=42
    )
    assert cfg.vision_config.model_type == "gemma4_unified_vision"
    assert cfg.text_config.model_type == "qwen3"
    assert cfg.image_token_id == 42
    # round-trips through dict form (how it deserializes from disk).
    cfg2 = UniViHybridConfig(**cfg.to_dict())
    assert cfg2.vision_config.mm_embed_dim == 64
    assert cfg2.text_config.hidden_size == 64


def test_text_only_forward_and_backward():
    model = _tiny_model(image_token_id=999)
    ids = torch.tensor([[1, 5, 6, 7]])
    labels = torch.tensor([[IGNORE_INDEX, 5, 6, 7]])
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert model.language_model.get_input_embeddings().weight.grad is not None


def test_placeholder_soft_token_mismatch_raises():
    model = _tiny_model(image_token_id=999)
    # one image but wrong number of placeholders should raise.
    from PIL import Image

    ip = Gemma4UnifiedImageProcessor()
    enc = ip(images=[Image.new("RGB", (256, 256))], return_tensors="pt")
    ids = torch.tensor([[1, 999, 999, 5]])  # only 2 placeholders, not the true count
    with pytest.raises(ValueError, match="does not match"):
        model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            pixel_values=enc["pixel_values"],
            image_position_ids=enc["image_position_ids"],
        )


def test_collator_aligns_and_model_runs():
    from transformers import AutoTokenizer
    from PIL import Image

    tok = AutoTokenizer.from_pretrained(QWEN_SRC)
    tok.add_special_tokens({"additional_special_tokens": ["<|univi_image|>"]})
    img_id = tok.convert_tokens_to_ids("<|univi_image|>")
    ip = Gemma4UnifiedImageProcessor()

    model = _tiny_model(image_token_id=img_id, vocab_size=len(tok))

    coll = HybridCollator(
        tokenizer=tok, image_processor=ip, image_token_id=img_id, max_length=2048
    )
    rows = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "text": None},
                        {"type": "text", "text": "Transcribe the image."},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
            ],
            "images": [Image.new("RGB", (320, 240))],
        },
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi there"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "yo"}]},
            ],
            "images": [],
        },
    ]
    batch = coll(rows)
    # placeholder count equals total soft tokens.
    n_ph = int((batch["input_ids"] == img_id).sum())
    assert n_ph == int(batch["num_soft_tokens_per_image"].sum())
    # response-only masking left some active labels.
    assert int((batch["labels"] != IGNORE_INDEX).sum()) > 0
    out = model(**batch)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    # both the patch embedder and the projector receive gradient.
    assert model.vision_tower.patch_dense.weight.grad is not None
    assert model.vision_tower.multimodal_embedder.embedding_projection.weight.grad is not None


# ---------------------------------------------------------------------------
# H16 — prior-gap-weighted loss (blind branch). Everything here must be a no-op
# unless the feature is switched on.
# ---------------------------------------------------------------------------


def _rows_with_image():
    from PIL import Image

    return [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "text": None},
                        {"type": "text", "text": "Transcribe the image."},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
            ],
            "images": [Image.new("L", (320, 240), 200)],
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "text": None},
                        {"type": "text", "text": "Transcribe the image."},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "world"}]},
            ],
            "images": [Image.new("L", (320, 240), 40)],
        },
    ]


def test_blank_like_is_mid_gray():
    from PIL import Image

    assert blank_like(Image.new("L", (4, 4), 7)).getpixel((0, 0)) == 128
    # NOTE PIL reads a bare int in RGB mode as band 0 only -> (128, 0, 0);
    # blank_like must not reproduce that.
    assert blank_like(Image.new("RGB", (4, 4))).getpixel((0, 0)) == (128, 128, 128)


def test_collator_blank_images_off_by_default_and_aligned_unchanged():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN_SRC)
    tok.add_special_tokens({"additional_special_tokens": ["<|univi_image|>"]})
    img_id = tok.convert_tokens_to_ids("<|univi_image|>")
    ip = Gemma4UnifiedImageProcessor(max_soft_tokens=70)
    kw = dict(tokenizer=tok, image_processor=ip, image_token_id=img_id, max_length=256)

    rows = _rows_with_image()
    plain = HybridCollator(**kw)(rows)
    withb = HybridCollator(**kw, emit_blank_images=True)(rows)

    assert not any(k.startswith("blank_") for k in plain)
    # the aligned half of the batch is untouched by the new option
    for k, v in plain.items():
        assert torch.equal(v, withb[k]), k
    # blanks line up with the real pixels and are uniform mid-gray (128/255)
    assert withb["blank_pixel_values"].shape == plain["pixel_values"].shape
    assert torch.equal(
        withb["blank_num_soft_tokens_per_image"], plain["num_soft_tokens_per_image"]
    )
    nz = withb["blank_pixel_values"][withb["blank_pixel_values"] != 0]
    assert torch.allclose(nz, torch.full_like(nz, 128 / 255), atol=1e-6)


def test_gap_weights_formula_normalization_and_masking():
    from univi.hybrid.train_pretrained import gap_weights, per_token_ce

    ce_a = torch.tensor([1.0, 2.0, 0.5, 0.0, 3.0])
    ce_b = torch.tensor([4.0, 1.0, 0.5, 9.9, 5.0])
    valid = torch.tensor([True, True, True, False, True])

    w = gap_weights(ce_a, ce_b, valid, beta=1.0)
    raw = torch.tensor([4.0, 1.0, 1.0, 0.0, 3.0])  # 1 + relu(gap), 0 on masked
    expected = raw / (raw[valid].sum() / int(valid.sum()))
    expected[~valid] = 0.0
    assert torch.allclose(w, expected)
    assert torch.isclose(w[valid].mean(), torch.tensor(1.0))
    assert float(w[~valid].abs().max()) == 0.0        # masked ⇒ zero weight
    assert w.requires_grad is False                    # weights never carry grad

    # degenerate: no gap anywhere ⇒ plain mean CE
    flat = gap_weights(ce_a, ce_a - 1.0, valid, beta=1.0)
    assert torch.equal(flat, valid.float())
    # beta = 0 ⇒ identical to the degenerate case whatever the gap
    assert torch.equal(gap_weights(ce_a, ce_b, valid, beta=0.0), valid.float())

    # per-token CE: masked positions contribute exactly nothing
    logits = torch.randn(1, 6, 11)
    labels = torch.tensor([[IGNORE_INDEX, 3, IGNORE_INDEX, 5, 7, IGNORE_INDEX]])
    ce, v = per_token_ce(logits, labels)
    assert v.tolist() == [True, False, True, True, False, False]  # HF's label shift
    assert float(ce[~v].abs().max()) == 0.0


def test_gap_weighted_loss_is_a_no_op_when_off_or_beta_zero(tmp_path):
    """`gap_weighted_loss: false` must be bit-identical to the stock Trainer loss,
    and beta = 0 must reduce to the unweighted loss."""
    from transformers import AutoTokenizer, Qwen3Config, Trainer, TrainingArguments

    from univi.hybrid.pretrained import (
        UniViHybridPretrained,
        UniViHybridPretrainedConfig,
    )
    from univi.hybrid.train_pretrained import _make_trainer_class, per_token_ce

    tok = AutoTokenizer.from_pretrained(QWEN_SRC)
    tok.add_special_tokens({"additional_special_tokens": ["<|univi_image|>"]})
    img_id = tok.convert_tokens_to_ids("<|univi_image|>")
    ip = Gemma4UnifiedImageProcessor(max_soft_tokens=70)
    kw = dict(tokenizer=tok, image_processor=ip, image_token_id=img_id, max_length=256)
    rows = _rows_with_image()
    plain = HybridCollator(**kw)(rows)
    withb = HybridCollator(**kw, emit_blank_images=True)(rows)

    torch.manual_seed(0)
    cfg = UniViHybridPretrainedConfig(
        vision_config=_tiny_vision(),
        text_config=Qwen3Config(
            hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            vocab_size=len(tok), tie_word_embeddings=True,
        ),
        image_token_id=img_id,
        max_soft_tokens=70,
    )
    model = UniViHybridPretrained(cfg)
    model.train()

    args = TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, report_to=[],
        remove_unused_columns=False,
    )

    def trainer(**gap):
        return _make_trainer_class(3e-4, 5e-5, 2e-5, 0.01, **gap)(model=model, args=args)

    n = int((plain["labels"] != IGNORE_INDEX).sum())
    off = trainer()
    with torch.no_grad():
        stock = Trainer.compute_loss(off, model, dict(plain), num_items_in_batch=n)
        got = off.compute_loss(model, dict(plain), num_items_in_batch=n)
        beta0 = trainer(gap_weighted_loss=True, gap_beta=0.0).compute_loss(
            model, dict(withb), num_items_in_batch=n
        )
        weighted = trainer(gap_weighted_loss=True, gap_beta=1.0).compute_loss(
            model, dict(withb), num_items_in_batch=n
        )
        # reference: the unweighted token-sum built from the SAME per-token CEs
        fwd = {k: v for k, v in plain.items() if k != "labels"}
        ce, valid = per_token_ce(model(**fwd).logits, plain["labels"])
        reference = ce.sum() / n

    assert torch.equal(stock, got)          # feature OFF ⇒ bit-identical
    assert torch.equal(reference, beta0)    # beta = 0 ⇒ exactly the unweighted sum
    # vs the stock loss, beta = 0 differs only by float32 reduction order
    # (cross_entropy(reduction="sum") vs sum of reduction="none").
    assert torch.allclose(stock, beta0, rtol=1e-5, atol=1e-5)
    assert torch.isfinite(weighted)


def test_blind_branch_contributes_no_gradient(tmp_path):
    from transformers import AutoTokenizer, Qwen3Config, TrainingArguments

    from univi.hybrid.pretrained import (
        UniViHybridPretrained,
        UniViHybridPretrainedConfig,
    )
    from univi.hybrid.train_pretrained import _make_trainer_class

    tok = AutoTokenizer.from_pretrained(QWEN_SRC)
    tok.add_special_tokens({"additional_special_tokens": ["<|univi_image|>"]})
    img_id = tok.convert_tokens_to_ids("<|univi_image|>")
    ip = Gemma4UnifiedImageProcessor(max_soft_tokens=70)
    batch = HybridCollator(
        tokenizer=tok, image_processor=ip, image_token_id=img_id,
        max_length=256, emit_blank_images=True,
    )(_rows_with_image())

    torch.manual_seed(0)
    model = UniViHybridPretrained(
        UniViHybridPretrainedConfig(
            vision_config=_tiny_vision(),
            text_config=Qwen3Config(
                hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                vocab_size=len(tok), tie_word_embeddings=True,
            ),
            image_token_id=img_id,
            max_soft_tokens=70,
        )
    )
    model.train()
    args = TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, report_to=[],
        remove_unused_columns=False,
    )
    n = int((batch["labels"] != IGNORE_INDEX).sum())

    def grads(blank_pixels, beta):
        tr = _make_trainer_class(
            3e-4, 5e-5, 2e-5, 0.01, gap_weighted_loss=True, gap_beta=beta
        )(model=model, args=args)
        b = {k: v.clone() for k, v in batch.items()}
        b["blank_pixel_values"] = blank_pixels.clone()
        model.zero_grad(set_to_none=True)
        tr.compute_loss(model, b, num_items_in_batch=n).backward()
        return {k: p.grad.clone() for k, p in model.named_parameters()
                if p.grad is not None}

    gray = batch["blank_pixel_values"]
    torch.manual_seed(7)
    noise = torch.rand_like(gray)
    # beta = 0 ⇒ the weights cannot depend on the blind branch, so a completely
    # different blind input must leave every gradient bit-identical.
    g1, g2 = grads(gray, 0.0), grads(noise, 0.0)
    assert g1 and all(torch.equal(g1[k], g2[k]) for k in g1)

    # ... and the blind input itself never receives gradient (it is inside no_grad).
    tr = _make_trainer_class(
        3e-4, 5e-5, 2e-5, 0.01, gap_weighted_loss=True, gap_beta=1.0
    )(model=model, args=args)
    b = {k: v.clone() for k, v in batch.items()}
    b["blank_pixel_values"] = b["blank_pixel_values"].clone().requires_grad_(True)
    b["pixel_values"] = b["pixel_values"].clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    tr.compute_loss(model, b, num_items_in_batch=n).backward()
    assert b["blank_pixel_values"].grad is None
    assert b["pixel_values"].grad is not None
