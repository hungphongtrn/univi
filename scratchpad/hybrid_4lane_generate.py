"""
Qualitative decodability probe for the 4-lane hybrid: greedy-decode the answer
with the ALIGNED image vs a BLANK image and print both against the target.

CE/Δblank say how much probability mass the image moves; this says WHAT the model
actually writes. If the OCR lanes were "already learned", aligned decodes should
reproduce the target text and blank decodes should fall apart.

``--val-path`` loads a split directly (skipping the SPLIT_ROOT/lane/filters path),
which is what any lane outside ``univi-3M-v0-split`` needs — e.g. H20's WIDE
re-render at ``data/materialized/h20-audio-wide-v0/librispeech/validation``.
``--out-json`` additionally writes the decodes plus a **word-level overlap** with
the target, which is the third component of H20's CONFIRMS criterion
(``docs/hypothesis/todo/H20-audio-phoneme-resolution.md``: "greedy decodes show
nonzero word-level overlap with targets (currently zero — length-only)"). Overlap
is reported for the BLANK decode too, because a blind model that emits common
English words scores a nonzero overlap by itself — the aligned number only means
something above the blank one.

Usage:
  HF_HUB_OFFLINE=1 uv run python scratchpad/hybrid_4lane_generate.py \
      --checkpoint data/checkpoints/hybrid-4lane-v0/final --lanes fineweb-edu librispeech -n 3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SPLIT_ROOT = "data/materialized/univi-3M-v0-split"

_WORD = re.compile(r"[a-z0-9']+")


def word_overlap(decode: str, target: str) -> dict:
    """Type-level word overlap of ``decode`` with ``target`` (case-folded).

    Type-level, not token-level: a model that repeats one correct word 40 times
    should not score 40 hits. ``recall`` is the fraction of the target's distinct
    words that appear at all — the "nonzero overlap" H20 asks about.
    """
    t = set(_WORD.findall(target.lower()))
    d = set(_WORD.findall(decode.lower()))
    hit = t & d
    return {
        "n_target_words": len(t),
        "n_decode_words": len(d),
        "n_overlap": len(hit),
        "recall": (len(hit) / len(t)) if t else None,
        "precision": (len(hit) / len(d)) if d else None,
        "overlap_words": sorted(hit)[:20],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lanes", nargs="*", default=["fineweb-edu", "librispeech"])
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-soft-tokens", type=int, default=None,
                    help="Per-image soft-token budget. Default: whatever the checkpoint "
                         "records (280 for pre-H17 checkpoints).")
    ap.add_argument("--max-length", type=int, default=2048,
                    help="Collator truncation budget; raise with --max-soft-tokens.")
    ap.add_argument("--val-path", default=None,
                    help="Load this split directly (skips the SPLIT_ROOT/lane/filter "
                         "path). Needed for any lane outside univi-3M-v0-split, e.g. "
                         "data/materialized/h20-audio-wide-v0/librispeech/validation.")
    ap.add_argument("--out-json", default=None,
                    help="Write the decodes + word-level overlap here (H20's third "
                         "CONFIRMS component). Without it this prints only.")
    a = ap.parse_args()

    import torch
    from datasets import load_from_disk
    from PIL import Image
    from transformers import AutoTokenizer

    from univi.hybrid.data import HybridCollator, blank_like
    from univi.hybrid.pretrained import (
        UniViHybridPretrained,
        build_image_processor,
        resolve_max_soft_tokens,
    )
    from univi.trainer import _filter_training_images, _filter_training_tokens

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UniViHybridPretrained.from_pretrained(a.checkpoint, dtype=torch.bfloat16).to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    # Process images at the budget the CHECKPOINT was trained at unless overridden.
    mst = resolve_max_soft_tokens(model, a.max_soft_tokens)
    print(f"[soft-tokens] budget = {mst} "
          f"(checkpoint records {getattr(model.config, 'max_soft_tokens', '<absent -> 280>')})")
    col = HybridCollator(
        tokenizer=tok,
        image_processor=build_image_processor(mst),
        image_token_id=model.image_token_id,
        max_length=a.max_length,
        response_only=True,
    )

    @torch.no_grad()
    def greedy(row, images, max_new):
        r = {k: row[k] for k in row}
        r["images"] = images
        batch = col([r])
        labels = batch["labels"][0]
        nz = (labels != -100).nonzero()
        if len(nz) == 0:
            return "<no supervised tokens>"
        start = int(nz[0])                       # first answer token position
        ids = batch["input_ids"][:, :start].to(device)
        attn = batch["attention_mask"][:, :start].to(device)
        extra = {
            k: batch[k].to(device)
            for k in ("pixel_values", "image_position_ids", "num_soft_tokens_per_image")
            if k in batch
        }
        out_ids = []
        for _ in range(max_new):
            o = model(input_ids=ids, attention_mask=attn, **extra)
            nxt = int(o.logits[0, -1].argmax(-1))
            if nxt == tok.eos_token_id or tok.decode([nxt]) == "<|im_end|>":
                break
            out_ids.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
            attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype, device=device)], dim=1)
        return tok.decode(out_ids)

    report = {"checkpoint": a.checkpoint, "max_soft_tokens": mst,
              "max_length": a.max_length, "max_new": a.max_new, "seed": a.seed,
              "val_path": a.val_path, "lanes": {}}
    for lane in a.lanes:
        if a.val_path:
            ds = load_from_disk(a.val_path)
        else:
            ds = load_from_disk(f"{SPLIT_ROOT}/{lane}/validation")
            ds = _filter_training_tokens(
                _filter_training_images(ds, lane, 4), lane, a.max_length)
        ds = ds.shuffle(seed=a.seed).select(range(min(a.n, len(ds))))
        rows = []
        print(f"\n{'=' * 78}\nLANE: {lane}\n{'=' * 78}")
        for i in range(len(ds)):
            row = ds[i]
            imgs = list(row["images"])
            blanks = [
                blank_like(im)
                for im in imgs
            ]
            # target = the assistant turn's text
            tgt = ""
            for m in row["messages"]:
                if m["role"] == "assistant":
                    c = m["content"]
                    tgt = c if isinstance(c, str) else " ".join(
                        p.get("text", "") for p in c if p.get("type") == "text"
                    )
            dec_a = greedy(row, imgs, a.max_new)
            dec_b = greedy(row, blanks, a.max_new)
            ov_a = word_overlap(dec_a, tgt)
            ov_b = word_overlap(dec_b, tgt)
            print(f"\n--- row {i} ({len(imgs)} img) ---")
            print(f"TARGET : {tgt[:240]!r}")
            print(f"ALIGNED: {dec_a[:240]!r}")
            print(f"        overlap {ov_a['n_overlap']}/{ov_a['n_target_words']} words")
            print(f"BLANK  : {dec_b[:240]!r}")
            print(f"        overlap {ov_b['n_overlap']}/{ov_b['n_target_words']} words")
            rows.append({"idx": i, "n_images": len(imgs), "target": tgt,
                         "aligned": dec_a, "blank": dec_b,
                         "overlap_aligned": ov_a, "overlap_blank": ov_b})

        n_nonzero_a = sum(1 for r in rows if r["overlap_aligned"]["n_overlap"] > 0)
        n_nonzero_b = sum(1 for r in rows if r["overlap_blank"]["n_overlap"] > 0)
        report["lanes"][lane] = {
            "n_rows": len(rows),
            "rows_with_nonzero_overlap_aligned": n_nonzero_a,
            "rows_with_nonzero_overlap_blank": n_nonzero_b,
            "mean_recall_aligned": (
                sum(r["overlap_aligned"]["recall"] or 0.0 for r in rows) / len(rows)
                if rows else None),
            "mean_recall_blank": (
                sum(r["overlap_blank"]["recall"] or 0.0 for r in rows) / len(rows)
                if rows else None),
            "note": "A blind model emitting common English words scores nonzero "
                    "overlap by itself; the aligned number only means something "
                    "ABOVE the blank one.",
            "decodes": rows,
        }
        print(f"\n[{lane}] rows with nonzero word overlap: aligned {n_nonzero_a}/"
              f"{len(rows)} | blank {n_nonzero_b}/{len(rows)}")

    if a.out_json:
        out = Path(a.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
