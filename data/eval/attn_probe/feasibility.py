"""Feasibility probe for the attention-grounding test.

Answers three make-or-break questions before building the full pipeline:
  1. Can we get attention tensors out of the 4-bit Unsloth model at all?
     (flash/xformers/sdpa silently return attentions=None; we may need eager.)
  2. What is the image-token layout? How many soft tokens per spectrogram,
     are they contiguous, what is the id?
  3. For the response (target) tokens, what fraction of attention lands on the
     image patch tokens vs the text tokens — the crude "does it look at pixels".

Loss-only harness seams are reused from eval_lane so preprocessing/masking match
training exactly. Run:
    uv run python data/eval/attn_probe/feasibility.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from datasets import load_from_disk

from eval_lane import _collate_one, _merge_images, load_model

CKPT = "data/checkpoints/audio-only-v0/checkpoint-800"
LIBRI = "data/materialized/univi-3M-v0-split/librispeech/validation"


def main() -> None:
    import torch

    print("[data] loading librispeech validation ...", flush=True)
    ds = load_from_disk(LIBRI)
    print(f"[data] {len(ds)} rows; columns={ds.column_names}", flush=True)
    row = ds[0]
    print(f"[data] messages structure: {type(row['messages'])}", flush=True)
    # peek at the target text (assistant turn)
    for m in row["messages"]:
        role = m.get("role")
        content = m.get("content")
        txt = None
        if isinstance(content, list):
            txt = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
            )
        print(f"   role={role} text={(txt or '')[:80]!r} n_content={len(content) if isinstance(content, list) else '?'}")
    print(f"[data] n_images={len(row.get('images', []))}", flush=True)
    if row.get("images"):
        im = row["images"][0]
        print(f"[data] image0 size={getattr(im, 'size', '?')} mode={getattr(im, 'mode', '?')}")

    print(f"\n[model] loading {CKPT} ...", flush=True)
    model, tokenizer = load_model(CKPT)
    model.eval()

    # Try to force eager so attentions are actually returned.
    forced = []
    for cfg in (getattr(model, "config", None),
                getattr(getattr(model, "config", None), "text_config", None),
                getattr(getattr(model, "config", None), "vision_config", None)):
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            try:
                cfg._attn_implementation = "eager"
                forced.append(type(cfg).__name__)
            except Exception:
                pass
    print(f"[model] forced eager on: {forced}", flush=True)

    img_token_id = getattr(getattr(model, "config", None), "image_token_id", 258880)
    print(f"[model] image_token_id={img_token_id}", flush=True)

    batch = _collate_one(model, tokenizer, _merge_images(row["messages"], row.get("images", [])))
    input_ids = batch["input_ids"][0]
    labels = batch.get("labels")
    seq_len = input_ids.shape[0]
    img_mask = (input_ids == img_token_id)
    n_img = int(img_mask.sum())
    img_pos = torch.where(img_mask)[0]
    contiguous = bool((img_pos[1:] - img_pos[:-1]).eq(1).all()) if n_img > 1 else False
    print(f"\n[seq] len={seq_len}  n_image_tokens={n_img}  contiguous={contiguous}", flush=True)
    if n_img:
        print(f"[seq] image token span: [{int(img_pos[0])}, {int(img_pos[-1])}]", flush=True)
    if labels is not None:
        tgt_mask = labels[0] != -100
        print(f"[seq] n_target(response) tokens={int(tgt_mask.sum())}", flush=True)

    print("\n[fwd] forward with output_attentions=True ...", flush=True)
    with torch.no_grad():
        out = model(**batch, output_attentions=True)
    attns = getattr(out, "attentions", None)
    if attns is None:
        print("[fwd] !!! outputs.attentions is None — eager did not take. "
              "Will need plain-transformers+PEFT load. ABORT full build until fixed.", flush=True)
        return
    print(f"[fwd] GOT attentions: n_layers={len(attns)} shape[0]={tuple(attns[0].shape)}", flush=True)

    # Attention fraction on image tokens, averaged over target tokens, per layer.
    tgt_mask = (labels[0] != -100) if labels is not None else torch.ones(seq_len, dtype=torch.bool)
    tgt_pos = torch.where(tgt_mask)[0]
    print(f"\n[attn] mean attention mass on image tokens, per layer (target tokens as queries):", flush=True)
    for li, a in enumerate(attns):
        # a: (batch, heads, q, k) -> average over heads
        am = a[0].float().mean(0)  # (q, k)
        # queries = target positions; sum key-mass over image positions
        q_att = am[tgt_pos]                    # (n_tgt, k)
        img_frac = q_att[:, img_pos].sum(-1)   # (n_tgt,)
        # exclude BOS sink (position 0) from the comparison denominator sanity
        print(f"   layer {li:2d}: img_frac mean={img_frac.mean():.4f}  "
              f"min={img_frac.min():.4f} max={img_frac.max():.4f}")

    print("\n[done] feasibility OK — attentions available, image tokens located.", flush=True)


if __name__ == "__main__":
    main()
