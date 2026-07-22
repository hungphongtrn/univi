"""Reading-order attention-alignment test for the TEXT lane (positive control).

Text reading order = raster order = the image-token sequence order (row-major,
top-left -> bottom-right). So for each response token we take its fractional
position through the passage (char offset / total chars) as the EXPECTED reading
position, and the attention centroid over the image tokens (sequence order,
normalized to [0,1]) as the PREDICTED position. If the model reads the rendered
text, expected and predicted sweep 0->1 together.

Same controls as the audio test:
  aligned   : the row's own rendered-text image
  permuted  : another row's image, SAME target text + token positions
              (a diagonal here = autoregressive habit, not reading)
  base vs trained : learned vs architectural

Uses full-v0/checkpoint-2800 (the mixture model where fineweb Δperm = +575%).
Reuses eval_lane seams + temporal_align helpers. No re-render / no grid dims
needed (unlike audio: text raster order is monotonic in reading order).

  uv run python data/eval/attn_probe/text_align.py \
      --checkpoint data/checkpoints/full-v0/checkpoint-2800 --n-rows 24
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from datasets import load_from_disk

from eval_lane import _collate_one, _merge_images, load_model
from temporal_align import spearman, image_block_positions  # reuse

FINEWEB = "data/materialized/univi-3M-v0-split/fineweb-edu/validation"


def target_text(messages):
    for m in messages:
        if m["role"] == "assistant":
            return " ".join(c.get("text", "") for c in m["content"]
                            if isinstance(c, dict) and c.get("type") == "text")
    return ""


def response_char_fracs(tokenizer, input_ids, labels):
    """Fraction-through-text (0..1) for each response token, from cumulative
    decoded char length (the non-uniform reading-position ground truth)."""
    import torch

    tok = getattr(tokenizer, "tokenizer", tokenizer)
    resp_pos = torch.where(labels != -100)[0]
    ids = [int(input_ids[p]) for p in resp_pos]
    lens = [len(tok.decode([i])) for i in ids]
    cum = np.cumsum([0] + lens[:-1], dtype=np.float64)  # char offset at token START
    total = float(sum(lens)) or 1.0
    return resp_pos, cum / total


def per_layer_raster_centroid(attns, resp_pos, img_pos):
    """(n_layers, n_resp) fractional raster centroid over image tokens in
    sequence order, normalized to [0,1]."""
    n_layers = len(attns)
    N = len(img_pos)
    frac = np.arange(N, dtype=np.float64) / max(N - 1, 1)  # 0..1 reading position
    out = np.full((n_layers, len(resp_pos)), np.nan)
    for l, a in enumerate(attns):
        am = a[0].float().mean(0)               # (q, k)
        block = am[resp_pos][:, img_pos].cpu().numpy()  # (n_resp, N)
        s = block.sum(axis=1)
        ok = s > 0
        out[l, ok] = (block[ok] * frac).sum(axis=1) / s[ok]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="data/checkpoints/full-v0/checkpoint-2800")
    ap.add_argument("--n-rows", type=int, default=24)
    ap.add_argument("--max-chars", type=int, default=1400, help="single-page-ish cap")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch

    ds = load_from_disk(FINEWEB)
    rows = []
    for i in range(len(ds)):
        r = ds[i]
        if len(r.get("images", [])) != 1:
            continue
        t = target_text(r["messages"])
        if not (200 <= len(t) <= args.max_chars):
            continue
        rows.append(r)
        if len(rows) >= args.n_rows:
            break
    print(f"[data] {len(rows)} single-image fineweb rows (<= {args.max_chars} chars)", flush=True)

    print(f"[model] loading {args.checkpoint} ...", flush=True)
    model, tokenizer = load_model(args.checkpoint)
    model.eval()
    for cfg in (model.config, getattr(model.config, "text_config", None),
                getattr(model.config, "vision_config", None)):
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"

    n_layers = None
    clip_corr = {"aligned": [], "permuted": []}
    for ci, r in enumerate(rows):
        for lane in ("aligned", "permuted"):
            donor = rows[(ci + 1) % len(rows)]
            img = r["images"][0] if lane == "aligned" else donor["images"][0]
            batch = _collate_one(model, tokenizer, _merge_images(r["messages"], [img]))
            input_ids = batch["input_ids"][0]
            labels = batch["labels"][0]
            img_pos = image_block_positions(input_ids)
            if len(img_pos) < 16:
                print(f"   [warn] row {ci} {lane}: {len(img_pos)} image tokens, skip", flush=True)
                continue
            resp_pos, exp_frac = response_char_fracs(tokenizer, input_ids, labels)
            with torch.no_grad():
                out = model(**batch, output_attentions=True)
            attns = out.attentions
            if n_layers is None:
                n_layers = len(attns)
            cents = per_layer_raster_centroid(attns, resp_pos, img_pos)
            per_layer_c = np.array([spearman(exp_frac, cents[l]) for l in range(n_layers)])
            clip_corr[lane].append(per_layer_c)
        print(f"   [{ci+1}/{len(rows)}] done  (n_img={len(img_pos)}, n_resp={len(resp_pos)})", flush=True)

    def mean_over(lane):
        arr = np.array(clip_corr[lane])
        return np.nanmean(arr, axis=0), int(arr.shape[0])

    al, n_used = mean_over("aligned")
    pm, _ = mean_over("permuted")
    report = {"checkpoint": args.checkpoint, "n_used": n_used, "lane": "text/fineweb",
              "metric": "within-row Spearman(char-fraction, raster-centroid), mean over rows",
              "layers": []}
    print("\n" + "=" * 78, flush=True)
    print("TEXT READING-ORDER ALIGNMENT  Spearman(char-fraction, raster centroid)", flush=True)
    print(f"  mean over {n_used} rows.  aligned>>permuted => reads text in reading order", flush=True)
    print("=" * 78, flush=True)
    print(f"  {'layer':>5} {'aligned':>9} {'permuted':>9} {'Δ':>9}", flush=True)
    for l in range(n_layers or 0):
        d = float(al[l] - pm[l])
        report["layers"].append({"layer": l, "aligned": float(al[l]), "permuted": float(pm[l]), "delta": d})
        star = " *" if d > 0.2 else ""
        print(f"  {l:>5} {al[l]:>9.3f} {pm[l]:>9.3f} {d:>9.3f}{star}", flush=True)

    band = [l for l in range(n_layers) if l >= 28]
    db_al = float(np.nanmean([al[l] for l in band])); db_pm = float(np.nanmean([pm[l] for l in band]))
    allb_al = float(np.nanmean(al)); allb_pm = float(np.nanmean(pm))
    best_l = int(np.nanargmax([report["layers"][l]["delta"] for l in range(n_layers)]))
    report["deep_band"] = {"aligned": db_al, "permuted": db_pm, "delta": db_al - db_pm}
    report["all_layers"] = {"aligned": allb_al, "permuted": allb_pm, "delta": allb_al - allb_pm}
    report["best_delta_layer"] = best_l
    print(f"\n  deep band L{band[0]}-{band[-1]}: aligned={db_al:.3f} permuted={db_pm:.3f} Δ={db_al-db_pm:.3f}", flush=True)
    print(f"  all layers        : aligned={allb_al:.3f} permuted={allb_pm:.3f} Δ={allb_al-allb_pm:.3f}", flush=True)
    print(f"  best Δ layer = {best_l} (Δ={report['layers'][best_l]['delta']:.3f}, aligned={report['layers'][best_l]['aligned']:.3f})", flush=True)

    out_path = args.out or str(Path(__file__).resolve().parent / f"text_align_{Path(args.checkpoint).name}.json")
    Path(out_path).write_text(json.dumps(report, indent=2))
    print(f"[done] {out_path}", flush=True)


if __name__ == "__main__":
    main()
