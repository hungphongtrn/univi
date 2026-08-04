"""
H17 — prove ``random_strings.py``'s new ``--canvas`` knob did not change the
DEFAULT render.

This repo has been bitten by silent render changes, so a renderer edit is only
acceptable with a byte-level proof.  The strongest available reference is not a
temporary copy of the pre-edit file — it is the **production data already on
disk**: ``data/materialized/h3-randstr-v0`` and ``data/materialized/h13-density-v0``
were materialized by the pre-edit renderer, and their PNGs are the exact pixels
H07/H13 trained on.

So the proof is: **regenerate those rows with the post-edit code at its default
canvas and byte-compare against the stored artifacts.**  The lane RNG is
``random.Random(seed)`` consumed in row order, so replaying the first N rows
reproduces the same targets deterministically; the images are then re-rendered
and compared raster-exactly (``PIL.Image.tobytes``), along with the target string
and the stored ``render_config`` JSON.

A positive control renders the same rows at ``--canvas 1584`` and asserts they
DIFFER and carry the raised geometry — otherwise "identical" could mean the knob
does nothing.

CPU ONLY.  No model weights, no GPU (a training run holds the card; this repo's
rule is that training runs SOLO).

Usage:
  CUDA_VISIBLE_DEVICES="" uv run python scratchpad/h17_canvas_render_identity.py
  CUDA_VISIBLE_DEVICES="" uv run python scratchpad/h17_canvas_render_identity.py -n 40
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("h17_canvas_render_identity")

#: (root, subset, split, seed).  ``_build_split`` uses ``seed`` for train and
#: ``seed + 10**6`` for validation; the default ``--seed`` is 3407.
LANES = [
    ("data/materialized/h3-randstr-v0", "random-strings", "train", 3407),
    ("data/materialized/h3-randstr-v0", "random-strings", "validation", 3407 + 10**6),
    ("data/materialized/h13-density-v0", "randstr-d1", "validation", 3407 + 10**6),
    ("data/materialized/h13-density-v0", "randstr-d3", "validation", 3407 + 10**6),
    ("data/materialized/h13-density-v0", "randstr-d4", "validation", 3407 + 10**6),
    ("data/materialized/h13-density-v0", "randstr-d5", "validation", 3407 + 10**6),
]
#: h3-randstr-v0 predates ``lane_specs``; its geometry is the module defaults.
FALLBACK_SPEC = {"group_len": 5, "n_groups": 5, "font_size": 14}


def stored_target(row) -> str:
    for msg in row["messages"]:
        if msg["role"] == "assistant":
            c = msg["content"]
            return ("".join(p.get("text") or "" for p in c) if isinstance(c, list) else str(c))
    raise ValueError("no assistant message")


def check_lane(root, subset, split, seed, n) -> dict:
    from datasets import load_from_disk

    from data.preprocessing.random_strings import (
        LaneSpec, _make_target, _render_config_json,
    )
    from data.preprocessing.render_utils import render_text_pages

    path = ROOT / root / subset / split
    if not path.exists():
        logger.warning("missing %s — skipped", path)
        return {"lane": f"{root}:{subset}/{split}", "skipped": str(path)}

    manifest = json.loads((ROOT / root / "manifest.json").read_text())
    ls = manifest.get("lane_specs", {}).get(subset, FALLBACK_SPEC)
    spec = LaneSpec(group_len=ls["group_len"], n_groups=ls["n_groups"],
                    font_size=ls["font_size"], subset_name=subset)

    ds = load_from_disk(str(path))
    k = min(n, len(ds))
    rng = random.Random(seed)
    rcfg = _render_config_json(spec)

    n_target_ok = n_px_ok = n_cfg_ok = 0
    first_bad = None
    for i in range(k):
        target = _make_target(rng, spec)          # replay the lane RNG
        row = ds[i]
        stored = stored_target(row)
        t_ok = target == stored
        n_target_ok += t_ok
        pages = render_text_pages(target, canvas_width=spec.canvas,
                                  canvas_height=spec.canvas, font_size=spec.font_size)
        img = row["images"][0]
        p_ok = (len(pages) == 1
                and pages[0].convert("L").tobytes() == img.convert("L").tobytes()
                and pages[0].size == img.size)
        n_px_ok += p_ok
        c_ok = rcfg == row["render_config"]
        n_cfg_ok += c_ok
        if first_bad is None and not (t_ok and p_ok and c_ok):
            first_bad = {"row": i, "target_ok": t_ok, "pixels_ok": p_ok, "cfg_ok": c_ok}

    res = {
        "lane": f"{root.split('/')[-1]}:{subset}/{split}",
        "n_checked": k, "n_rows": len(ds),
        "spec": {"group_len": spec.group_len, "n_groups": spec.n_groups,
                 "font_size": spec.font_size, "canvas": spec.canvas},
        "targets_match": n_target_ok, "pixels_match": n_px_ok, "render_config_match": n_cfg_ok,
        "image_size": list(ds[0]["images"][0].size),
        "all_match": n_target_ok == n_px_ok == n_cfg_ok == k,
        "first_mismatch": first_bad,
    }
    logger.info("%-38s n=%-4d targets %d/%d  pixels %d/%d  render_config %d/%d  %s",
                res["lane"], k, n_target_ok, k, n_px_ok, k, n_cfg_ok, k,
                "OK" if res["all_match"] else "MISMATCH")
    return res


def positive_control(n=6) -> dict:
    """``--canvas 1584`` must change the pixels, the size and the render_config."""
    from data.preprocessing.random_strings import LaneSpec, _render_config_json, _make_target
    from data.preprocessing.render_utils import render_text_pages

    base = LaneSpec()
    raised = LaneSpec(canvas=1584, font_size=22)
    rng = random.Random(3407)
    differ = size_ok = 0
    for _ in range(n):
        t = _make_target(rng, base)
        a = render_text_pages(t, canvas_width=base.canvas, canvas_height=base.canvas,
                              font_size=base.font_size)[0]
        b = render_text_pages(t, canvas_width=raised.canvas, canvas_height=raised.canvas,
                              font_size=raised.font_size)[0]
        differ += a.tobytes() != b.tobytes()
        size_ok += (a.size == (1024, 1024) and b.size == (1584, 1584))
    cfg_a = json.loads(_render_config_json(base))
    cfg_b = json.loads(_render_config_json(raised))
    res = {
        "n": n, "n_pixels_differ": differ, "n_sizes_as_expected": size_ok,
        "render_config_default": cfg_a, "render_config_raised": cfg_b,
        "passed": differ == n and size_ok == n
                  and cfg_a["canvas_width"] == 1024 and cfg_b["canvas_width"] == 1584,
    }
    logger.info("positive control: pixels differ %d/%d, sizes right %d/%d, "
                "render_config 1024 -> 1584  %s", differ, n, size_ok, n,
                "OK" if res["passed"] else "FAIL")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--n", type=int, default=25, help="rows per lane")
    ap.add_argument("--out", default="data/eval/h17-canvas-render-identity.json")
    args = ap.parse_args(argv)

    lanes = [check_lane(r, s, sp, sd, args.n) for r, s, sp, sd in LANES]
    ctl = positive_control()
    checked = [l for l in lanes if "skipped" not in l]
    ok = bool(checked) and all(l["all_match"] for l in checked) and ctl["passed"]

    res = {
        "claim": ("random_strings.py's --canvas knob does not change the default "
                  "render: replaying each lane's RNG and re-rendering with the "
                  "post-edit code reproduces the materialized PNGs raster-exactly."),
        "reference": "the production datasets on disk, built by the pre-edit renderer",
        "lanes": lanes, "positive_control": ctl, "all_passed": ok,
        "device": "cpu (no model weights; a training run holds the GPU)",
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    logger.info("wrote %s", out)
    print("\nDEFAULT RENDER UNCHANGED:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
