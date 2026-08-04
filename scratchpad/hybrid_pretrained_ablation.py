"""
Grounding ablation for the PRETRAINED-vision hybrid (univi/hybrid/pretrained.py),
on the random-string OCR lane. Mirrors scratchpad/h3_ablation.py but for
UniViHybridPretrained + HybridCollator.

Per val row, response-only CE + teacher-forced token accuracy under three image
conditions: aligned / permuted-donor / blank. Δperm and Δblank ≈ 0 (aligned ==
permuted == blank) ⇒ the model ignores pixels (the H3 collapse). Δblank/Δperm ≫ 0
⇒ the pretrained vision front-end reads where the from-scratch one couldn't.

READ THIS BEFORE QUOTING A Δperm NUMBER FROM HERE (added 2026-07-29)
-------------------------------------------------------------------
The Δperm/Δblank this script reports are AGGREGATES over every supervised token,
and reading is concentrated in the first ~5-12 answer tokens, so the aggregate is a
function of TARGET LENGTH as much as of grounding.  Measured for IDENTICAL reading:
+113.84% at 17 supervised tok/row, +14.31% at 60, +3.05% at 234, +0.63% at 932; the
largest aggregate ever recorded at T≈60 is +25.80%, from a model reading at +57.3
pts at position 0.  Every "Δperm >= X%" bar in this repo was read off this script
or `hybrid_4lane_ablation.py`, and neither could produce a prefix-matched value —
that is the mechanical root cause of the mis-specified criteria.  `--prefix-tokens
N` now emits `delta_perm_rel_prefixN` / `delta_blank_rel_prefixN` ALONGSIDE the
aggregate (never replacing it, so old artifacts stay comparable).  A bar of the form
"Δperm >= 30%" is unreachable by construction at T >= 234 — use the prefix numbers,
or better, the position-resolved instrument
(`scratchpad/hybrid_4lane_position_decay.py`).

Usage:
  HF_HUB_OFFLINE=1 uv run python scratchpad/hybrid_pretrained_ablation.py \
      --checkpoint data/checkpoints/hybrid-pretrained-randstr-v0/checkpoint-200 \
      --tag hybrid-pre-200 --prefix-tokens 10
"""
from __future__ import annotations

import argparse, json, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("hybrid_pre_ablation")

FLOOR_PATH = "data/materialized/h3-randstr-v0/floor.json"
TEXT_SOURCE = "unsloth/Qwen3-1.7B"
IMAGE_PLACEHOLDER = "<|univi_image|>"


def _mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def run(checkpoint, tag, max_samples, seed, output, val_dataset, floor_path,
        max_soft_tokens=None, max_length=4096, prefix_tokens=None):
    import torch
    from datasets import load_from_disk
    from PIL import Image
    from transformers import AutoTokenizer

    from univi.hybrid.pretrained import (
        UniViHybridPretrained,
        build_image_processor,
        resolve_max_soft_tokens,
    )
    from univi.hybrid.data import HybridCollator, blank_like

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading trained hybrid from %s", checkpoint)
    model = UniViHybridPretrained.from_pretrained(checkpoint, dtype=torch.bfloat16).to(device)
    model.eval()
    model.config.use_cache = True

    # Tokenizer: prefer the checkpoint's; else reconstruct deterministically.
    try:
        tok = AutoTokenizer.from_pretrained(checkpoint)
    except Exception:
        tok = AutoTokenizer.from_pretrained(TEXT_SOURCE)
        if IMAGE_PLACEHOLDER not in tok.get_vocab():
            tok.add_special_tokens({"additional_special_tokens": [IMAGE_PLACEHOLDER]})
    image_token_id = model.image_token_id
    # Process images at the budget the CHECKPOINT was trained at unless overridden
    # — a probe run at a different budget yields garbage that looks like a result.
    mst = resolve_max_soft_tokens(model, max_soft_tokens)
    logger.info("Soft-token budget: %d (checkpoint records %s)", mst,
                getattr(model.config, "max_soft_tokens", "<absent → 280>"))
    ip = build_image_processor(mst)
    col = HybridCollator(tokenizer=tok, image_processor=ip, image_token_id=image_token_id,
                         max_length=max_length, response_only=True)

    ds = load_from_disk(val_dataset)
    ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))
    n = len(ds)
    donor = [(i + 1) % n for i in range(n)]
    # Blank matches the lane's image geometry (spectrogram 1000x160, OCR 1024x1024).
    _ref = ds[0]["images"][0]
    blank = blank_like(_ref)
    logger.info("Evaluating %d rows (blank %s %s)", n, blank.mode, blank.size)

    import torch.nn.functional as F

    def eval_one(row, image):
        r = {k: row[k] for k in row}
        r["images"] = [image]
        batch = col([r])
        labels = batch["labels"]
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        # Forward WITHOUT labels so the LM returns full logits (passing labels
        # triggers the logits_to_keep optimization → empty logits).
        fwd = {k: v for k, v in batch.items() if k != "labels"}
        with torch.no_grad():
            out = model(**fwd)
        logits = out.logits[0].float()          # [S, V]
        tg = labels[0].to(logits.device)
        sl = logits[:-1]
        tg = tg[1:]
        valid = tg != -100
        if int(valid.sum()) == 0:
            return float("nan"), float("nan"), float("nan"), float("nan")
        # per-token CE first, so the PREFIX window can be taken from the same pass.
        # The aggregate is the mean of these, i.e. bit-identical to the previous
        # reduction="mean" — old artifacts stay comparable.
        ce_tok = F.cross_entropy(sl[valid], tg[valid], reduction="none")
        corr = (sl[valid].argmax(-1) == tg[valid]).float()
        ce, acc = ce_tok.mean().item(), corr.mean().item()
        if not prefix_tokens:
            return ce, acc, float("nan"), float("nan")
        k = min(prefix_tokens, ce_tok.numel())
        return ce, acc, ce_tok[:k].mean().item(), corr[:k].mean().item()

    keys = ["a_ce", "p_ce", "b_ce", "a_acc", "p_acc", "b_acc",
            "a_ce_pre", "p_ce_pre", "b_ce_pre", "a_acc_pre", "b_acc_pre"]
    rows = {k: [] for k in keys}
    for i in range(n):
        row = ds[i]
        img = row["images"][0]
        dn = ds[donor[i]]["images"][0]
        a_ce, a_acc, a_ce_pre, a_acc_pre = eval_one(row, img)
        p_ce, p_acc, p_ce_pre, _ = eval_one(row, dn)
        b_ce, b_acc, b_ce_pre, b_acc_pre = eval_one(row, blank)
        rows["a_ce"].append(a_ce); rows["p_ce"].append(p_ce); rows["b_ce"].append(b_ce)
        rows["a_acc"].append(a_acc); rows["p_acc"].append(p_acc); rows["b_acc"].append(b_acc)
        rows["a_ce_pre"].append(a_ce_pre); rows["p_ce_pre"].append(p_ce_pre)
        rows["b_ce_pre"].append(b_ce_pre)
        rows["a_acc_pre"].append(a_acc_pre); rows["b_acc_pre"].append(b_acc_pre)
        if (i + 1) % 25 == 0:
            logger.info("  %d/%d aligned ce %.3f acc %.3f", i + 1, n, _mean(rows["a_ce"]), _mean(rows["a_acc"]))

    a, p, b = _mean(rows["a_ce"]), _mean(rows["p_ce"]), _mean(rows["b_ce"])
    dperm = (p - a) / a if a else float("nan")
    dblank = (b - a) / a if a else float("nan")
    floor = None
    try:
        floor = json.loads(Path(floor_path).read_text()).get("no_reading_floor_nats_per_token")
    except Exception:
        pass
    res = {
        "tag": tag, "checkpoint": checkpoint, "n_samples": n,
        "max_soft_tokens": mst, "max_length": max_length,
        "prefix_tokens": prefix_tokens,
        "response_ce": {"aligned": a, "permuted": p, "blank": b},
        "tok_acc": {"aligned": _mean(rows["a_acc"]), "permuted": _mean(rows["p_acc"]), "blank": _mean(rows["b_acc"])},
        "delta_perm_rel": dperm, "delta_blank_rel": dblank, "no_reading_floor_nats": floor,
        "aggregate_note": (
            "delta_perm_rel / delta_blank_rel average over ALL supervised tokens and "
            "are therefore LENGTH-DILUTED (identical reading measures +113.84% at 17 "
            "sup tok/row, +14.31% at 60, +3.05% at 234, +0.63% at 932). Compare them "
            "only within one target length; for a criterion use the prefix numbers "
            "below or a position-resolved probe."),
    }
    if prefix_tokens:
        ap_, pp_, bp_ = (_mean(rows["a_ce_pre"]), _mean(rows["p_ce_pre"]),
                         _mean(rows["b_ce_pre"]))
        res[f"prefix{prefix_tokens}_response_ce"] = {"aligned": ap_, "permuted": pp_,
                                                     "blank": bp_}
        res[f"prefix{prefix_tokens}_tok_acc"] = {
            "aligned": _mean(rows["a_acc_pre"]), "blank": _mean(rows["b_acc_pre"])}
        res[f"delta_perm_rel_prefix{prefix_tokens}"] = (
            (pp_ - ap_) / ap_ if ap_ else float("nan"))
        res[f"delta_blank_rel_prefix{prefix_tokens}"] = (
            (bp_ - ap_) / ap_ if ap_ else float("nan"))
        res[f"reading_gain_pts_prefix{prefix_tokens}"] = (
            (_mean(rows["a_acc_pre"]) - _mean(rows["b_acc_pre"])) * 100)
    print(f"\n===== HYBRID-PRETRAINED ABLATION ({tag}) =====")
    print(f"checkpoint : {checkpoint}   n={n}   max_length {max_length}")
    print(f"response-only CE : aligned {a:.3f} | permuted {p:.3f} (Δperm {dperm*100:+.2f}%) | blank {b:.3f} (Δblank {dblank*100:+.2f}%)")
    ta = res["tok_acc"]
    print(f"tok-acc          : aligned {ta['aligned']*100:.2f}% | blank {ta['blank']*100:.2f}% (guess) | reading gain {(ta['aligned']-ta['blank'])*100:+.2f} pts")
    if prefix_tokens:
        print(f"PREFIX ({prefix_tokens} tok)     : CE aligned {ap_:.3f} | permuted {pp_:.3f} "
              f"(Δperm {res[f'delta_perm_rel_prefix{prefix_tokens}']*100:+.2f}%) | "
              f"blank {bp_:.3f} "
              f"(Δblank {res[f'delta_blank_rel_prefix{prefix_tokens}']*100:+.2f}%)  "
              f"reading gain {res[f'reading_gain_pts_prefix{prefix_tokens}']:+.2f} pts")
        print("  ^ prefix-matched: use THIS (or a position-resolved probe) for any "
              "criterion; the aggregate above is length-diluted.")
    print("=" * 48)
    out_path = Path(output) if output else Path(f"data/eval/hybrid-pretrained-ablation-{tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    logger.info("Wrote %s", out_path)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", default="hybrid-pre")
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--output", default=None)
    p.add_argument("--val-dataset", default="data/materialized/h3-randstr-v0/random-strings/validation",
                   help="Path to the validation split (load_from_disk).")
    p.add_argument("--floor", default=FLOOR_PATH, help="floor.json for the no-reading floor.")
    p.add_argument("--max-soft-tokens", type=int, default=None,
                   help="Per-image soft-token budget. Default: whatever the checkpoint "
                        "records (280 for pre-H17 checkpoints).")
    # 2048 silently DROPPED the longest rows (the collator truncates, so the deep
    # supervised positions simply vanish and the row is scored on what is left).
    # 4096 fits rendered text at a 280-1120 soft-token budget; AUDIO needs 8192.
    # Pass --max-length 2048 to reproduce the pre-2026-07-29 artifacts.
    p.add_argument("--max-length", type=int, default=4096,
                   help="Collator truncation budget. 4096 for rendered text, 8192 "
                        "for audio (spectrogram pages are longer); 2048 reproduces "
                        "the old artifacts but truncates deep positions.")
    p.add_argument("--prefix-tokens", type=int, default=None,
                   help="Also report Δperm/Δblank over the FIRST N answer tokens "
                        "(prefix-matched), emitted as delta_perm_rel_prefixN "
                        "beside the aggregate. Reading lives in the first ~5-12 "
                        "tokens; the aggregate is length-diluted, so any criterion "
                        "should be read off the prefix. Suggested: 10.")
    a = p.parse_args(argv)
    run(a.checkpoint, a.tag, a.max_samples, a.seed, a.output, a.val_dataset, a.floor,
        a.max_soft_tokens, a.max_length, a.prefix_tokens)


if __name__ == "__main__":
    main()
