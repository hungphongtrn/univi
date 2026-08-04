"""
Per-lane grounding ablation for the 4-lane pretrained-vision hybrid run
(configs/hybrid_4lane.yaml -> data/checkpoints/hybrid-4lane-v0).

Generalizes scratchpad/hybrid_pretrained_ablation.py in two ways needed by the
real mixture:

1. **Per-lane**: iterates the four validation splits of univi-3M-v0-split and
   reports Δperm / Δblank separately, so "does audio drown?" is answerable.
2. **Multi-image rows**: fineweb-edu / librispeech rows render to several pages.
   The permuted condition supplies exactly ``len(row["images"])`` donor images
   (cycling the donor's list) and the blank condition builds one blank per image
   at the *original* size/mode, so the soft-token count is identical across the
   three conditions and only image CONTENT varies.

Rows are filtered with the trainer's own ``_filter_training_images`` /
``_filter_training_tokens`` so the pool matches what training/eval actually saw
(max_train_images=4, ``--filter-max-length``, default = the training budget 2048).

READ THIS BEFORE QUOTING A Δperm NUMBER FROM HERE (added 2026-07-29)
-------------------------------------------------------------------
The Δperm/Δblank here are AGGREGATES over every supervised token, and reading is
concentrated in the first ~5-12 answer tokens, so the aggregate is a function of
TARGET LENGTH as much as of grounding.  Measured for IDENTICAL reading: +113.84% at
17 supervised tok/row, +14.31% at 60, +3.05% at 234, +0.63% at 932; the largest
aggregate ever recorded at T≈60 is +25.80%, from a model reading at +57.3 pts at
position 0.  Since this lane mixture spans 60→930 tok/row, the four lanes' aggregate
Δperm are NOT comparable with each other.  Every "Δperm >= X%" bar in this repo was
read off this script or ``hybrid_pretrained_ablation.py``, and neither could produce
a prefix-matched value — the mechanical root cause of the mis-specified criteria.
``--prefix-tokens N`` now emits ``delta_perm_rel_prefixN`` beside the aggregate
(never replacing it, so old artifacts stay comparable).

Two length knobs, deliberately separated: ``--max-length`` truncates the assembled
sequence (raising it stops the deep supervised positions from vanishing) while
``--filter-max-length`` selects the ROW POOL (keep it at the run's training budget
or the eval scores rows training never saw).

Usage:
  HF_HUB_OFFLINE=1 uv run python scratchpad/hybrid_4lane_ablation.py \
      --checkpoint data/checkpoints/hybrid-4lane-v0/final --tag 4lane-final \
      --prefix-tokens 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("hybrid_4lane_ablation")

TEXT_SOURCE = "unsloth/Qwen3-1.7B"
IMAGE_PLACEHOLDER = "<|univi_image|>"
SPLIT_ROOT = "data/materialized/univi-3M-v0-split"
LANES = ["fineweb-edu", "densefusion", "smoltalk", "librispeech"]


def _mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def _load_lane(lane: str, max_samples: int, seed: int, max_images: int, max_length: int):
    from datasets import load_from_disk

    from univi.trainer import _filter_training_images, _filter_training_tokens

    ds = load_from_disk(f"{SPLIT_ROOT}/{lane}/validation")
    ds = _filter_training_images(ds, lane, max_images)
    ds = _filter_training_tokens(ds, lane, max_length)
    ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))
    return ds


def run(checkpoint, tag, max_samples, seed, output, lanes, max_images, max_length,
        max_soft_tokens=None, prefix_tokens=None, filter_max_length=None):
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoTokenizer

    from univi.hybrid.data import HybridCollator, blank_like
    from univi.hybrid.pretrained import (
        UniViHybridPretrained,
        build_image_processor,
        resolve_max_soft_tokens,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading trained hybrid from %s", checkpoint)
    model = UniViHybridPretrained.from_pretrained(checkpoint, dtype=torch.bfloat16).to(device)
    model.eval()
    model.config.use_cache = True

    try:
        tok = AutoTokenizer.from_pretrained(checkpoint)
    except Exception:
        tok = AutoTokenizer.from_pretrained(TEXT_SOURCE)
        if IMAGE_PLACEHOLDER not in tok.get_vocab():
            tok.add_special_tokens({"additional_special_tokens": [IMAGE_PLACEHOLDER]})
    # Process images at the budget the CHECKPOINT was trained at unless overridden.
    mst = resolve_max_soft_tokens(model, max_soft_tokens)
    logger.info("Soft-token budget: %d (checkpoint records %s)", mst,
                getattr(model.config, "max_soft_tokens", "<absent → 280>"))
    ip = build_image_processor(mst)
    col = HybridCollator(
        tokenizer=tok,
        image_processor=ip,
        image_token_id=model.image_token_id,
        max_length=max_length,
        response_only=True,
    )

    def eval_one(row, images):
        r = {k: row[k] for k in row}
        r["images"] = images
        batch = col([r])
        labels = batch["labels"]
        seq_len = int(batch["input_ids"].shape[1])
        fwd = {
            k: (v.to(device) if hasattr(v, "to") else v)
            for k, v in batch.items()
            if k != "labels"
        }
        with torch.no_grad():
            out = model(**fwd)
        logits = out.logits[0].float()
        tg = labels[0].to(logits.device)
        sl = logits[:-1]
        tg = tg[1:]
        valid = tg != -100
        nan = float("nan")
        if int(valid.sum()) == 0:
            return {"ce": nan, "acc": nan, "n": 0, "ce_pre": nan, "acc_pre": nan,
                    "truncated": seq_len >= max_length}
        # per-token CE first so the PREFIX window comes from the same pass; the
        # aggregate is the mean of these, i.e. identical to the old
        # reduction="mean" — existing artifacts stay comparable.
        ce_tok = F.cross_entropy(sl[valid], tg[valid], reduction="none")
        corr = (sl[valid].argmax(-1) == tg[valid]).float()
        rec = {"ce": ce_tok.mean().item(), "acc": corr.mean().item(),
               "n": int(valid.sum()), "ce_pre": nan, "acc_pre": nan,
               "truncated": seq_len >= max_length}
        if prefix_tokens:
            k = min(prefix_tokens, ce_tok.numel())
            rec["ce_pre"] = ce_tok[:k].mean().item()
            rec["acc_pre"] = corr[:k].mean().item()
        return rec

    results = {}
    for lane in lanes:
        # the row POOL is selected at filter_max_length (the training budget), while
        # the collator truncates at max_length — two different jobs, two knobs.
        ds = _load_lane(lane, max_samples, seed, max_images,
                        filter_max_length or max_length)
        n = len(ds)
        logger.info("[%s] %d rows", lane, n)
        donor = [(i + 1) % n for i in range(n)]
        acc = {k: [] for k in ["a_ce", "p_ce", "b_ce", "a_acc", "p_acc", "b_acc",
                               "n_img", "n_tgt", "a_ce_pre", "p_ce_pre", "b_ce_pre",
                               "a_acc_pre", "b_acc_pre"]}
        n_trunc = 0

        for i in range(n):
            row = ds[i]
            imgs = list(row["images"])
            k = len(imgs)
            # permuted: exactly k donor images (cycle if the donor has fewer)
            d_imgs = list(ds[donor[i]]["images"])
            p_imgs = [d_imgs[j % len(d_imgs)] for j in range(k)]
            # blank: one per image, same size/mode -> identical soft-token count
            b_imgs = [
                blank_like(im)
                for im in imgs
            ]

            ra = eval_one(row, imgs)
            rp = eval_one(row, p_imgs)
            rb = eval_one(row, b_imgs)
            n_trunc += int(ra["truncated"])
            acc["a_ce"].append(ra["ce"]); acc["p_ce"].append(rp["ce"])
            acc["b_ce"].append(rb["ce"])
            acc["a_acc"].append(ra["acc"]); acc["p_acc"].append(rp["acc"])
            acc["b_acc"].append(rb["acc"])
            acc["a_ce_pre"].append(ra["ce_pre"]); acc["p_ce_pre"].append(rp["ce_pre"])
            acc["b_ce_pre"].append(rb["ce_pre"])
            acc["a_acc_pre"].append(ra["acc_pre"])
            acc["b_acc_pre"].append(rb["acc_pre"])
            acc["n_img"].append(k); acc["n_tgt"].append(ra["n"])
            if (i + 1) % 50 == 0:
                logger.info(
                    "  [%s] %d/%d aligned ce %.3f | perm %.3f | blank %.3f",
                    lane, i + 1, n, _mean(acc["a_ce"]), _mean(acc["p_ce"]), _mean(acc["b_ce"]),
                )

        a, p, b = _mean(acc["a_ce"]), _mean(acc["p_ce"]), _mean(acc["b_ce"])
        results[lane] = {
            "n_samples": n,
            "mean_images_per_row": _mean(acc["n_img"]),
            "mean_target_tokens": _mean(acc["n_tgt"]),
            "n_rows_truncated_at_max_length": n_trunc,
            "response_ce": {"aligned": a, "permuted": p, "blank": b},
            "tok_acc": {
                "aligned": _mean(acc["a_acc"]),
                "permuted": _mean(acc["p_acc"]),
                "blank": _mean(acc["b_acc"]),
            },
            "delta_perm_rel": (p - a) / a if a else float("nan"),
            "delta_blank_rel": (b - a) / a if a else float("nan"),
            "reading_gain_pts": (_mean(acc["a_acc"]) - _mean(acc["b_acc"])) * 100,
        }
        if prefix_tokens:
            ap_, pp_, bp_ = (_mean(acc["a_ce_pre"]), _mean(acc["p_ce_pre"]),
                             _mean(acc["b_ce_pre"]))
            results[lane][f"prefix{prefix_tokens}_response_ce"] = {
                "aligned": ap_, "permuted": pp_, "blank": bp_}
            results[lane][f"prefix{prefix_tokens}_tok_acc"] = {
                "aligned": _mean(acc["a_acc_pre"]), "blank": _mean(acc["b_acc_pre"])}
            results[lane][f"delta_perm_rel_prefix{prefix_tokens}"] = (
                (pp_ - ap_) / ap_ if ap_ else float("nan"))
            results[lane][f"delta_blank_rel_prefix{prefix_tokens}"] = (
                (bp_ - ap_) / ap_ if ap_ else float("nan"))
            results[lane][f"reading_gain_pts_prefix{prefix_tokens}"] = (
                (_mean(acc["a_acc_pre"]) - _mean(acc["b_acc_pre"])) * 100)
        r = results[lane]
        print(f"\n----- {lane} (n={n}, {r['mean_images_per_row']:.2f} img/row, "
              f"{r['mean_target_tokens']:.0f} sup tok/row) -----")
        print(
            f"  CE  aligned {a:.3f} | permuted {p:.3f} (Δperm {r['delta_perm_rel']*100:+.2f}%)"
            f" | blank {b:.3f} (Δblank {r['delta_blank_rel']*100:+.2f}%)"
        )
        print(
            f"  acc aligned {r['tok_acc']['aligned']*100:.2f}% | blank {r['tok_acc']['blank']*100:.2f}%"
            f" | reading gain {r['reading_gain_pts']:+.2f} pts"
        )
        if prefix_tokens:
            print(f"  PREFIX ({prefix_tokens} tok): Δperm "
                  f"{r[f'delta_perm_rel_prefix{prefix_tokens}']*100:+.2f}% | Δblank "
                  f"{r[f'delta_blank_rel_prefix{prefix_tokens}']*100:+.2f}% | reading "
                  f"gain {r[f'reading_gain_pts_prefix{prefix_tokens}']:+.2f} pts")
        if n_trunc:
            print(f"  !! {n_trunc}/{n} rows hit --max-length {max_length}: their deep "
                  f"supervised positions were CUT, so this lane's aggregate is "
                  f"computed on a short-biased slice of those rows. Raise "
                  f"--max-length (8192 for audio).")

    res = {
        "tag": tag, "checkpoint": checkpoint, "seed": seed,
        "max_soft_tokens": mst, "max_length": max_length,
        "filter_max_length": filter_max_length or max_length,
        "prefix_tokens": prefix_tokens,
        "aggregate_note": (
            "delta_perm_rel / delta_blank_rel average over ALL supervised tokens and "
            "are LENGTH-DILUTED: identical reading measures +113.84% at 17 sup "
            "tok/row, +14.31% at 60, +3.05% at 234, +0.63% at 932. These lanes span "
            "60-930 tok/row, so their aggregate Δperm are NOT comparable with each "
            "other and no fixed '>= X%' bar applies across them. Use the prefix "
            "numbers or a position-resolved probe."),
        "lanes": results,
    }
    print(f"\n===== 4-LANE GROUNDING ABLATION ({tag}) =====")
    hdr = (f"{'lane':<14}{'tok/row':>9}{'aligned':>9}{'Δperm':>10}{'Δblank':>10}"
           f"{'gain':>9}")
    if prefix_tokens:
        hdr += f"{'Δperm@' + str(prefix_tokens):>12}{'gain@' + str(prefix_tokens):>12}"
    print(hdr)
    for lane, r in results.items():
        line = (
            f"{lane:<14}{r['mean_target_tokens']:>9.0f}"
            f"{r['response_ce']['aligned']:>9.3f}"
            f"{r['delta_perm_rel']*100:>9.2f}%{r['delta_blank_rel']*100:>9.2f}%"
            f"{r['reading_gain_pts']:>8.2f}p"
        )
        if prefix_tokens:
            line += (f"{r[f'delta_perm_rel_prefix{prefix_tokens}']*100:>11.2f}%"
                     f"{r[f'reading_gain_pts_prefix{prefix_tokens}']:>11.2f}p")
        print(line)
    print("Δperm/Δblank columns are LENGTH-DILUTED aggregates — the tok/row column is "
          "why they are not comparable across lanes.")
    print("=" * 52)

    out_path = Path(output) if output else Path(f"data/eval/hybrid-4lane-ablation-{tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    logger.info("Wrote %s", out_path)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", default="4lane")
    p.add_argument("--max-samples", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    p.add_argument("--lanes", nargs="*", default=LANES)
    p.add_argument("--max-images", type=int, default=4)
    # 2048 silently truncated the longest rows: the collator cuts the sequence, so
    # the deep supervised positions vanish and the row is scored on a short-biased
    # slice. 4096 fits rendered text; AUDIO (spectrogram pages) needs 8192. Pass
    # --max-length 2048 to reproduce the pre-2026-07-29 artifacts.
    p.add_argument("--max-length", type=int, default=4096,
                   help="Collator truncation budget. 4096 for rendered text, 8192 "
                        "for audio; 2048 reproduces the old artifacts but truncates "
                        "deep positions.")
    p.add_argument("--filter-max-length", type=int, default=2048,
                   help="Budget used to SELECT the row pool via the trainer's own "
                        "length filter. Keep it at the run's training max_length "
                        "(2048 for hybrid-4lane) so the pool matches what training "
                        "saw; --max-length only controls truncation.")
    p.add_argument("--prefix-tokens", type=int, default=None,
                   help="Also report Δperm/Δblank over the FIRST N answer tokens "
                        "(prefix-matched), emitted as delta_perm_rel_prefixN beside "
                        "the aggregate. Reading lives in the first ~5-12 tokens and "
                        "the aggregate is length-diluted, so cross-lane comparisons "
                        "and any '>= X%%' bar need this. Suggested: 10.")
    p.add_argument("--max-soft-tokens", type=int, default=None,
                   help="Per-image soft-token budget. Default: whatever the checkpoint "
                        "records (280 for pre-H17 checkpoints).")
    a = p.parse_args(argv)
    run(a.checkpoint, a.tag, a.max_samples, a.seed, a.output, a.lanes, a.max_images,
        a.max_length, a.max_soft_tokens, a.prefix_tokens, a.filter_max_length)


if __name__ == "__main__":
    main()
