"""H21 — materialize the glyph-scale (font) lanes.

Two products, one script:

**leg 0** (`--mode leg0`) — validation-ONLY rungs at several font sizes, all d2
geometry (20 groups x 5 letters = 119 chars) and all drawn from the **same seed**,
so every rung carries **byte-identical targets** and every cross-font comparison
is paired by row.  Written to ``data/materialized/h21-fonts-v0/randstr-f<F>/validation``.

**leg 1** (`--mode leg1`) — ONE training lane whose rows are drawn at several
fonts (a *per-row* font mixture, not one subset per font).  Written to
``data/materialized/h21-fontmix-v0/random-strings/{train,validation}``.

Why one mixed lane rather than one subset per font: ``univi.trainer.VALID_SUBSETS``
is a closed set and none of its names mean "font F".  Mixing inside a single
already-valid subset (``random-strings``) needs **no code change**, and the font of
each row stays recoverable from ``source_dataset_id`` (``synthetic/randstr-f24``)
and from ``render_config.font_size``.  H21's design note said per-row jitter
"would need a per-row font draw in the materializer" — this is that draw, done
outside the materializer by concatenating per-font blocks.

THREE THINGS THIS SCRIPT CHECKS RATHER THAN ASSUMES
---------------------------------------------------
1. **The font clamp.** ``render_utils`` does ``font_size = max(font_size, 14)``, so
   every font below 14 renders **byte-identically to font 14**.  A rung set that
   contains 10 *and* 14 is therefore two copies of the same image with two labels.
   ``--mode leg0`` asserts that every pair of rungs differs in pixels on every row
   and FAILS LOUDLY if two fonts collide.
2. **The one-page guard.** ``render_text_pages`` pages out; ``random_strings``
   raises if a row needs more than one page (it would otherwise show a truncated
   image with a full target).  Verified implicitly on every row.
3. **The floor.** H13 §7: ``random_strings.py`` divides the string entropy by
   *target* tokens while response-only masking also supervises ``<|im_end|>\\n``,
   so the published floor is too high by ~T/(T-2).  This script writes BOTH the
   legacy number (so existing readers do not silently change meaning) and
   ``no_reading_floor_nats_per_supervised_token``, and says which is which.
   Criteria are stated in Delta-perm / Delta-blank / reading gain, never in CE.

CPU ONLY — no torch, no GPU, no model.

    CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 uv run python \\
        scratchpad/h21_materialize_fonts.py --mode leg0 --val-rows 300

    CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 uv run python \\
        scratchpad/h21_materialize_fonts.py --mode leg1 --train-rows-per-font 20000

    uv run python scratchpad/h21_materialize_fonts.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("h21_materialize")

# d2 geometry (H13): 20 groups x 5 letters = 100 random letters, 119 chars,
# ~58 target tokens => ~60 supervised => ~80% readable inside the 48-token scan
# depth.  H13 died at 14.2% readable; H07 succeeded at 100%.
GROUP_LEN = 5
N_GROUPS = 20
CANVAS = 1024

#: leg 0 sweep.  10 is DELIBERATELY ABSENT — it is clamped to 14 (see module doc).
LEG0_FONTS = [14, 18, 20, 24, 28, 31, 40, 52]

#: leg 1 trained fonts and the held-out interior fonts the criterion is stated on.
LEG1_TRAIN_FONTS = [14, 24, 40]
LEG1_HELDOUT_FONTS = [18, 31]

LEG0_ROOT = "data/materialized/h21-fonts-v0"
LEG1_ROOT = "data/materialized/h21-fontmix-v0"

TRAILER_TOKENS = 2  # response-only masking also supervises `<|im_end|>\n`
SCAN_DEPTH = 48     # H13 §4: K = 48.5 answer tokens over a 16x density span

VAL_SEED = 3407 + 10**6        # one seed for EVERY validation rung => paired rows
TRAIN_SEED_BASE = 3407         # per-font train seeds are distinct (design choice 5)


def lane_spec(font: int, subset_name: str):
    from data.preprocessing.random_strings import LaneSpec

    return LaneSpec(
        group_len=GROUP_LEN,
        n_groups=N_GROUPS,
        font_size=font,
        subset_name=subset_name,
        canvas=CANVAS,
    )


def _png_hashes(ds, k: int) -> list[str]:
    """md5 of the first ``k`` rows' stored PNG bytes, without decoding images."""
    sub = ds.select(range(min(k, len(ds)))).select_columns(["images"]).with_format("arrow")
    recs = sub[:].column("images").to_pylist()
    return [hashlib.md5(r[0]["bytes"]).hexdigest() for r in recs]


def _targets(ds, k: int) -> list[str]:
    sub = ds.select(range(min(k, len(ds)))).select_columns(["messages"])
    return ["".join(p["text"] or "" for p in r["messages"][1]["content"]) for r in sub]


def floor_json(spec, tok_total: int, n_rows: int) -> dict:
    """Analytic no-reading floor, in BOTH conventions (H13 §7)."""
    from data.preprocessing.random_strings import CHARSET

    n_letters = spec.n_random_letters
    h_char = math.log(len(CHARSET))
    tgt_per_row = tok_total / max(n_rows, 1)
    sup_per_row = tgt_per_row + TRAILER_TOKENS
    return {
        "charset_size": len(CHARSET),
        "n_random_letters_per_string": n_letters,
        "H_char_nats": h_char,
        "val_avg_tokens_per_string": tgt_per_row,
        "val_avg_supervised_tokens_per_string": sup_per_row,
        # legacy key, kept so existing readers do not silently change meaning
        "no_reading_floor_nats_per_token": h_char * n_letters / max(tgt_per_row, 1e-9),
        # H13 §7 correction: the supervised set includes `<|im_end|>\n`
        "no_reading_floor_nats_per_supervised_token":
            h_char * n_letters / max(sup_per_row, 1e-9),
        "note": (
            "H13 §7: `no_reading_floor_nats_per_token` divides the string entropy by "
            "TARGET tokens, but response-only masking also supervises the "
            "deterministic `<|im_end|>\\n` trailer (2 tokens, ~0 nats), so that number "
            "is too high by ~T/(T-2). Use "
            "`no_reading_floor_nats_per_supervised_token` if you must compare a CE — "
            "but H21's criteria are Delta-perm / Delta-blank / reading gain and touch "
            "NEITHER."
        ),
    }


def write_manifest(root: Path, entries: list[dict], lane_specs: dict) -> None:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {
        "artifact": root.name,
        "canvas": {"height": CANVAS, "width": CANVAS},
        "subsets": [],
    }
    names = {e["config_name"] for e in entries}
    manifest["subsets"] = [
        s for s in manifest["subsets"]
        if not (s["config_name"] in names and s["split"] in
                {e["split"] for e in entries if e["config_name"] == s["config_name"]})
    ] + entries
    manifest.setdefault("lane_specs", {}).update(lane_specs)
    path.write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# leg 0 — validation-only font sweep, paired by row
# ---------------------------------------------------------------------------
def build_leg0(args) -> dict:
    from transformers import AutoTokenizer

    from data.preprocessing.random_strings import _build_split

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    root = Path(args.leg0_root)
    root.mkdir(parents=True, exist_ok=True)

    report, entries, specs = {}, [], {}
    hashes, targets_ref = {}, None
    for font in args.fonts:
        name = f"randstr-f{font}"
        spec = lane_spec(font, name)
        logger.info("leg0 rung %s: font %d, %d letters, canvas %d, %d rows",
                    name, font, spec.n_random_letters, CANVAS, args.val_rows)
        ds, tok_total = _build_split("validation", args.val_rows, VAL_SEED, tok, spec)
        out = root / name / "validation"
        out.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(out))
        (root / name / "floor.json").write_text(
            json.dumps(floor_json(spec, tok_total, args.val_rows), indent=2)
        )
        entries.append({"config_name": name, "path": f"{name}/validation",
                        "split": "validation", "is_training_split": False})
        specs[name] = {"group_len": GROUP_LEN, "n_groups": N_GROUPS,
                       "font_size": font, "n_random_letters": spec.n_random_letters,
                       "canvas": CANVAS}
        tgts = _targets(ds, args.check_rows)
        if targets_ref is None:
            targets_ref = tgts
        elif tgts != targets_ref:
            raise AssertionError(
                f"{name}: targets differ from the reference rung — the rungs are NOT "
                "paired by row. Every rung must use the same seed."
            )
        hashes[name] = _png_hashes(ds, args.check_rows)
        report[name] = {
            "font": font, "rows": len(ds),
            "target_tokens_mean": tok_total / max(args.val_rows, 1),
            "supervised_tokens_mean": tok_total / max(args.val_rows, 1) + TRAILER_TOKENS,
            "path": str(out),
        }

    # --- the check that catches the font-<14 clamp -------------------------
    collisions = []
    names = list(hashes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            same = sum(x == y for x, y in zip(hashes[a], hashes[b]))
            if same:
                collisions.append({"a": a, "b": b, "identical_rows": same,
                                   "checked_rows": len(hashes[a])})
    if collisions:
        raise AssertionError(
            "PIXEL COLLISION between font rungs — render_utils clamps font_size to "
            f">= 14, so these rungs are the SAME image under two labels: {collisions}"
        )
    logger.info("pixel-distinctness: all %d rungs differ on all %d checked rows",
                len(names), args.check_rows)
    logger.info("pairing: targets byte-identical across all rungs on %d checked rows",
                args.check_rows)

    write_manifest(root, entries, specs)
    return {"mode": "leg0", "root": str(root), "fonts": args.fonts,
            "val_rows": args.val_rows, "seed": VAL_SEED,
            "targets_paired_rows_checked": args.check_rows,
            "pixel_collisions": collisions, "rungs": report}


# ---------------------------------------------------------------------------
# leg 1 — ONE training lane, rows drawn at several fonts
# ---------------------------------------------------------------------------
def build_leg1(args) -> dict:
    from datasets import concatenate_datasets, load_from_disk
    from transformers import AutoTokenizer

    from data.preprocessing.random_strings import _build_split

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    root = Path(args.leg1_root)
    (root / "random-strings").mkdir(parents=True, exist_ok=True)
    tmp = root / "_blocks"
    tmp.mkdir(parents=True, exist_ok=True)

    report = {}
    for split, rows_per_font, seed_off in (
        ("train", args.train_rows_per_font, 0),
        ("validation", args.val_rows_per_font, 10**6),
    ):
        parts, tok_total, n_total = [], 0, 0
        for j, font in enumerate(args.train_fonts):
            # DISTINCT train seed per font (design choice 5: otherwise the run sees
            # each 119-char string once per font, i.e. ~3 epochs of target content
            # on a lane whose premise is that targets are unguessable).
            # Validation shares ONE seed across fonts is NOT wanted here — this is a
            # mixed lane, not a paired sweep; the paired sweep is leg 0.
            seed = TRAIN_SEED_BASE + seed_off + 7919 * (j + 1)
            spec = lane_spec(font, f"randstr-f{font}")
            logger.info("leg1 %s block: font %d, %d rows, seed %d",
                        split, font, rows_per_font, seed)
            ds, t = _build_split(split, rows_per_font, seed, tok, spec)
            p = tmp / f"{split}-f{font}"
            if p.exists():
                shutil.rmtree(p)
            ds.save_to_disk(str(p))
            parts.append(str(p))
            tok_total += t
            n_total += len(ds)
            del ds
        merged = concatenate_datasets([load_from_disk(p) for p in parts])
        out = root / "random-strings" / split
        if out.exists():
            shutil.rmtree(out)
        merged.save_to_disk(str(out))
        report[split] = {
            "rows": len(merged),
            "rows_per_font": rows_per_font,
            "fonts": args.train_fonts,
            "target_tokens_mean": tok_total / max(n_total, 1),
            "supervised_tokens_mean": tok_total / max(n_total, 1) + TRAILER_TOKENS,
            "readable_fraction_at_depth_48":
                min(1.0, SCAN_DEPTH / (tok_total / max(n_total, 1) + TRAILER_TOKENS)),
            "path": str(out),
        }
        del merged
        for p in parts:
            shutil.rmtree(p)
    shutil.rmtree(tmp, ignore_errors=True)

    spec = lane_spec(args.train_fonts[0], "random-strings")
    (root / "random-strings" / "floor.json").write_text(json.dumps(
        floor_json(spec,
                   int(report["validation"]["target_tokens_mean"] * report["validation"]["rows"]),
                   report["validation"]["rows"]), indent=2))
    write_manifest(root, [
        {"config_name": "random-strings", "path": "random-strings/train",
         "split": "train", "is_training_split": True},
        {"config_name": "random-strings", "path": "random-strings/validation",
         "split": "validation", "is_training_split": False},
    ], {"random-strings": {"group_len": GROUP_LEN, "n_groups": N_GROUPS,
                           "font_sizes": args.train_fonts, "canvas": CANVAS,
                           "n_random_letters": GROUP_LEN * N_GROUPS,
                           "note": "per-ROW font mixture; font recoverable from "
                                   "source_dataset_id / render_config"}})
    return {"mode": "leg1", "root": str(root), "splits": report,
            "train_fonts": args.train_fonts, "heldout_fonts": LEG1_HELDOUT_FONTS}


# ---------------------------------------------------------------------------
def self_test() -> None:
    """CPU, no datasets on disk: the two invariants this script exists to enforce."""
    from data.preprocessing.render_utils import render_text_pages

    rng_target = " ".join("abcde" for _ in range(N_GROUPS))
    assert len(rng_target) == 119, len(rng_target)
    print(f"  ok  d2 geometry is {len(rng_target)} chars (20 groups x 5 letters)")

    def h(f):
        p = render_text_pages(rng_target, canvas_width=CANVAS, canvas_height=CANVAS,
                              font_size=f)
        assert len(p) == 1, f"font {f} needs {len(p)} pages"
        return hashlib.md5(p[0].convert("L").tobytes()).hexdigest()

    # 1. the clamp is real and this script's guard would catch it
    assert h(10) == h(14), "expected render_utils to clamp font_size to >= 14"
    print("  ok  font 10 renders BYTE-IDENTICALLY to font 14 (max(font_size, 14))")

    # 2. every font actually used is distinct, and every one fits a single page
    hs = {f: h(f) for f in LEG0_FONTS}
    assert len(set(hs.values())) == len(hs), f"font rungs collide: {hs}"
    print(f"  ok  the {len(LEG0_FONTS)} leg-0 fonts {LEG0_FONTS} are pairwise "
          f"pixel-distinct and each fits ONE 1024px page")

    for f in LEG1_TRAIN_FONTS + LEG1_HELDOUT_FONTS:
        assert f in LEG0_FONTS, f"leg-1 font {f} is not probed by leg 0"
    lo, hi = min(LEG1_TRAIN_FONTS), max(LEG1_TRAIN_FONTS)
    for f in LEG1_HELDOUT_FONTS:
        assert lo < f < hi, f"held-out font {f} is not INTERIOR to {LEG1_TRAIN_FONTS}"
        assert f not in LEG1_TRAIN_FONTS
    print(f"  ok  held-out fonts {LEG1_HELDOUT_FONTS} are interior to the trained "
          f"range {LEG1_TRAIN_FONTS} and are never trained")

    # 3. the floor correction is the H13 §7 factor
    class _S:
        n_random_letters = GROUP_LEN * N_GROUPS
    fj = floor_json(_S(), 58 * 100, 100)
    ratio = fj["no_reading_floor_nats_per_token"] / \
        fj["no_reading_floor_nats_per_supervised_token"]
    assert abs(ratio - 60 / 58) < 1e-9, ratio
    print(f"  ok  floor.json carries both conventions; legacy/corrected = "
          f"{ratio:.4f} = T/(T-2) at T=60 (H13 §7)")

    print("\nSELF-TEST PASSED")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["leg0", "leg1"], default=None)
    ap.add_argument("--fonts", type=int, nargs="*", default=LEG0_FONTS)
    ap.add_argument("--train-fonts", type=int, nargs="*", default=LEG1_TRAIN_FONTS)
    ap.add_argument("--val-rows", type=int, default=300)
    ap.add_argument("--train-rows-per-font", type=int, default=20000)
    ap.add_argument("--val-rows-per-font", type=int, default=500)
    ap.add_argument("--check-rows", type=int, default=64,
                    help="rows used for the pairing / pixel-collision assertions")
    ap.add_argument("--tokenizer", default="data/checkpoints/encoder-free-v0/best")
    ap.add_argument("--leg0-root", default=LEG0_ROOT)
    ap.add_argument("--leg1-root", default=LEG1_ROOT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        print("SELF-TEST (CPU, no dataset written)")
        self_test()
        return
    if a.mode is None:
        ap.error("--mode leg0|leg1 is required (or --self-test)")

    bad = [f for f in (a.fonts if a.mode == "leg0" else a.train_fonts) if f < 14]
    if bad:
        ap.error(f"fonts {bad} are below render_utils' clamp (max(font_size, 14)) and "
                 f"would render byte-identically to font 14")

    report = build_leg0(a) if a.mode == "leg0" else build_leg1(a)
    out = Path(a.out or f"data/eval/h21-materialize-{a.mode}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
