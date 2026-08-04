"""H19 — assemble the four-lane rematch mixture into ONE loadable root.

``univi.trainer._load_local`` takes a SINGLE ``dataset.path`` and reads the
manifest under it, so a mixture whose lanes live in different materialization
roots (univi-3M-v0-split, h3-randstr-v0, spoken-digits-v0, h15-poisoned-p00, …)
cannot be expressed by a config alone.  This builds the merged root.

It does four things a plain symlink farm cannot:

1. **Per-lane row counts.**  ``dataset.max_train_rows_per_subset`` is a single
   GLOBAL cap, so it cannot give lane A 3,263 rows and lane B 50,496.  H13's
   lesson (*balancing rows does not balance gradient*) requires exactly that:
   token share is ``rows x supervised_tokens_per_row``, and this repo's lanes
   differ by 18x in target length.  Counts come from
   ``scratchpad/h19_mixture_audit.rows_for_token_share``.

2. **Pre-filtering at ``max_length``.**  ``_filter_training_tokens`` SILENTLY
   DROPS overflowing rows.  On fineweb-edu that is 61% of the split at
   max_length 1024 and it can never reach 100% at any practical budget (the
   longest row estimates at 168,616 tokens).  Selecting rows from the PASSING
   set makes retention through the trainer's own filter 100.00% BY CONSTRUCTION,
   and makes the measured per-lane token statistics the ones the run will
   actually see.

3. **``remove_columns``.**  ``poisoned-text`` carries 3 extra columns and
   ``masked-randstr`` 12.  Measured on datasets 4.3.0: ``concatenate_datasets``
   does **NOT** raise — it aligns by name and NULL-FILLS the missing columns,
   silently producing a 13- or 22-column table (row data itself is preserved).
   So the hazard is silence, not failure, and it is version-dependent.  Stripping
   makes the Features identical on every lane, which is checked here.

4. **A realised share table.**  Row counts get CLAMPED by availability
   (spoken-digits has only 30,000 rows), so the mixture that lands on disk is not
   always the mixture that was asked for.  The script reports what it actually
   built and refuses to pretend otherwise.

CPU ONLY.  Writes datasets; run it once, not while another job needs the disk.

    CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 uv run python \\
        scratchpad/h19_build_mixture.py --plan-only --total-rows 128000

    CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 uv run python \\
        scratchpad/h19_build_mixture.py --out data/materialized/h19-mix-v0 \\
        --total-rows 128000 --max-length 1024

    uv run python scratchpad/h19_build_mixture.py --self-test
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h19_mixture_audit import (  # noqa: E402
    LANES, MIX_B_TOKEN_SHARES, MIX_C_TOKEN_SHARES, SCAN_DEPTH,
    SCAN_DEPTH_MEASURED_RANGE, SCAN_DEPTH_RETIRED, TRAILER_TOKENS,
    prefilter_indices, rows_for_token_share, share_table, stub_unsloth,
    warn_scan_depth,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("h19_build")

MIXES = {"B": MIX_B_TOKEN_SHARES, "C": MIX_C_TOKEN_SHARES}

#: Columns every lane must end up with (the univi materialized schema).
BASE_COLUMNS = [
    "images", "messages", "source_dataset_id", "split", "row_id", "render_config",
    "modality_label", "preprocessing_version", "original_token_length", "source_split",
]


def lane_stats(name, root, split, n, tokenizer, max_length, per_image,
               scan_depth=SCAN_DEPTH):
    """Supervised-token geometry on the rows that SURVIVE the length filter.

    ``scan_depth`` only affects ``readable_tokens_mean`` (the reported readable
    fraction); the token-share solver is driven by ``sup_tokens_mean``, so the row
    counts this script builds are INDEPENDENT of the depth. Default 48 is the
    retired K floor — see ``h19_mixture_audit.SCAN_DEPTH``.
    """
    import statistics

    from datasets import load_from_disk

    ds = load_from_disk(str(Path(root) / split))
    idx = prefilter_indices(ds, max_length, per_image)
    kept = len(idx)
    sub = ds.select(idx[:n]).select_columns(["messages"])
    sup = []
    for row in sub:
        t = 0
        for m in row["messages"]:
            if m["role"] != "assistant":
                continue
            c = m["content"]
            txt = ("".join(p.get("text") or "" for p in c)
                   if isinstance(c, list) else str(c))
            t += len(tokenizer.encode(txt, add_special_tokens=False)) + TRAILER_TOKENS
        sup.append(t)
    return {
        "rows_passing_filter": kept,
        "sup_tokens_mean": statistics.fmean(sup),
        "scan_depth": scan_depth,
        "readable_tokens_mean": statistics.fmean(min(scan_depth, t) for t in sup),
    }


def plan(stats: dict, target: dict, total_rows: int) -> dict:
    """Row counts for the token-share target, CLAMPED to what exists."""
    want = rows_for_token_share(stats, target, total_rows)
    got, clamped = {}, {}
    for k, v in want.items():
        avail = stats[k]["rows_passing_filter"]
        got[k] = min(v, avail)
        if v > avail:
            clamped[k] = {"wanted": v, "available": avail,
                          "epochs_if_repeated": round(v / avail, 2)}
    return {"target_token_share": target, "rows_wanted": want, "rows_built": got,
            "clamped": clamped, "realised": share_table(stats, got)}


def build(out: Path, stats: dict, rows: dict, split: str, max_length: int,
          per_image: int, seed: int, val_rows: int) -> list[dict]:
    from datasets import load_from_disk

    entries = []
    out.mkdir(parents=True, exist_ok=True)
    for name, n in rows.items():
        src = Path(LANES[name])
        dst = out / name / split
        if dst.exists():
            shutil.rmtree(dst)
        ds = load_from_disk(str(src / split))
        idx = prefilter_indices(ds, max_length, per_image)
        # deterministic, and NOT the first-N (which would correlate with whatever
        # order the source was written in)
        sel = ds.select(idx).shuffle(seed=seed).select(range(min(n, len(idx))))
        extra = [c for c in sel.column_names if c not in BASE_COLUMNS]
        if extra:
            logger.info("%s: removing %d extra columns %s", name, len(extra), extra)
            sel = sel.remove_columns(extra)
        missing = [c for c in BASE_COLUMNS if c not in sel.column_names]
        if missing:
            raise RuntimeError(f"{name}: missing base columns {missing}")
        sel.save_to_disk(str(dst))
        entries.append({"config_name": name, "path": f"{name}/{split}",
                        "split": split, "is_training_split": True,
                        "rows": len(sel)})
        logger.info("%s: wrote %d rows -> %s", name, len(sel), dst)

        # validation: same treatment, capped small (the trajectory monitor only)
        vsrc = src / "validation"
        if vsrc.exists():
            vdst = out / name / "validation"
            if vdst.exists():
                shutil.rmtree(vdst)
            vds = load_from_disk(str(vsrc))
            vidx = prefilter_indices(vds, max_length, per_image)
            v = vds.select(vidx).shuffle(seed=seed).select(
                range(min(val_rows, len(vidx))))
            ve = [c for c in v.column_names if c not in BASE_COLUMNS]
            if ve:
                v = v.remove_columns(ve)
            v.save_to_disk(str(vdst))
            entries.append({"config_name": name, "path": f"{name}/validation",
                            "split": "validation", "is_training_split": False,
                            "rows": len(v)})
    return entries


def verify(out: Path, entries: list[dict]) -> dict:
    """Features identical on every lane, and concatenate_datasets is clean."""
    from datasets import concatenate_datasets, load_from_disk

    train = [e for e in entries if e["is_training_split"]]
    heads = {e["config_name"]: load_from_disk(str(out / e["path"])).select(range(2))
             for e in train}
    ref = next(iter(heads.values()))
    same = {k: bool(v.features == ref.features) for k, v in heads.items()}
    merged = concatenate_datasets(list(heads.values()))
    return {"features_identical_per_lane": same,
            "all_identical": all(same.values()),
            "merged_columns": list(merged.column_names),
            "merged_rows": len(merged),
            "merged_has_only_base_columns":
                sorted(merged.column_names) == sorted(BASE_COLUMNS)}


def self_test(scan_depth: int = SCAN_DEPTH) -> None:
    # readable_tokens_mean is min(scan_depth, T) per row — derived from the depth,
    # never hardcoded, so the fixture follows --scan-depth like the real lanes do.
    stats = {
        "long": {"rows_passing_filter": 10_000, "sup_tokens_mean": 300.0,
                 "readable_tokens_mean": float(min(scan_depth, 300))},
        "short": {"rows_passing_filter": 10_000, "sup_tokens_mean": 20.0,
                  "readable_tokens_mean": float(min(scan_depth, 20))},
    }
    p = plan(stats, {"long": 0.5, "short": 0.5}, 10_000)
    r = p["realised"]["per_lane"]
    assert abs(r["long"]["token_share"] - 0.5) < 0.01, r
    assert p["rows_built"]["short"] > 10 * p["rows_built"]["long"], p["rows_built"]
    assert not p["clamped"]
    print(f"  ok  equal TOKEN share needs {p['rows_built']['short']} short vs "
          f"{p['rows_built']['long']} long rows (15x), and the solver finds it")

    stats["short"]["rows_passing_filter"] = 100
    p = plan(stats, {"long": 0.5, "short": 0.5}, 10_000)
    assert "short" in p["clamped"] and p["rows_built"]["short"] == 100
    assert p["clamped"]["short"]["epochs_if_repeated"] > 1
    assert abs(p["realised"]["per_lane"]["short"]["token_share"] - 0.5) > 0.4
    print(f"  ok  an under-supplied lane is CLAMPED, and the realised share table "
          f"shows the damage ({100 * p['realised']['per_lane']['short']['token_share']:.1f}% "
          f"instead of 50%), it is not silently rescaled")

    assert sorted(BASE_COLUMNS) == sorted(set(BASE_COLUMNS)) and len(BASE_COLUMNS) == 10
    print("  ok  the base schema is the 10 columns every univi lane carries")

    # the row counts the mixture is BUILT from must not move with the scan depth
    # (they come from sup_tokens_mean); only the reported readable fraction does
    # fresh fixtures (the clamp test above mutated `stats`), and pinned to depths
    # 48 vs 5 so the comparison does not depend on --scan-depth
    def _fix(depth):
        return {"long": {"rows_passing_filter": 10_000, "sup_tokens_mean": 300.0,
                         "readable_tokens_mean": float(min(depth, 300))},
                "short": {"rows_passing_filter": 10_000, "sup_tokens_mean": 20.0,
                          "readable_tokens_mean": float(min(depth, 20))}}

    plan48 = plan(_fix(SCAN_DEPTH_RETIRED), {"long": 0.5, "short": 0.5}, 10_000)
    plan5 = plan(_fix(5), {"long": 0.5, "short": 0.5}, 10_000)
    assert plan5["rows_built"] == plan48["rows_built"], (plan5, plan48)
    assert (plan5["realised"]["readable_fraction_overall"]
            < plan48["realised"]["readable_fraction_overall"])
    print(f"  ok  --scan-depth changes only the READABLE FRACTION "
          f"({100 * plan48['realised']['readable_fraction_overall']:.1f}% at depth "
          f"{SCAN_DEPTH_RETIRED} -> "
          f"{100 * plan5['realised']['readable_fraction_overall']:.1f}% at depth 5), "
          f"never the row counts")
    print("\nSELF-TEST PASSED (token-share solver, availability clamp, base schema, "
          "scan-depth independence of the row plan)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/materialized/h19-mix-v0")
    ap.add_argument("--mix", choices=list(MIXES), default="B")
    ap.add_argument("--total-rows", type=int, default=128_000,
                    help="max_steps x effective batch (2000 x 64 by default)")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-soft-tokens", type=int, default=280)
    ap.add_argument("--val-rows", type=int, default=256)
    ap.add_argument("--stat-rows", type=int, default=400)
    ap.add_argument("--scan-depth", type=int, default=SCAN_DEPTH,
                    help=f"answer-token depth for the REPORTED readable fraction "
                         f"(the row plan is driven by sup_tokens_mean and does not "
                         f"move with it). Default {SCAN_DEPTH} is the RETIRED K "
                         f"floor; measured readable depth is "
                         f"~{SCAN_DEPTH_MEASURED_RANGE[0]}-"
                         f"{SCAN_DEPTH_MEASURED_RANGE[1]} answer tokens.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--report", default="data/eval/h19-mixture-build.json")
    a = ap.parse_args(argv)

    if a.self_test:
        print("SELF-TEST (CPU, no data)")
        self_test(a.scan_depth)
        return

    depth_note = warn_scan_depth(a.scan_depth, logger)
    stub_unsloth()
    from transformers import AutoTokenizer

    from univi.trainer import image_token_budget

    per_image = image_token_budget(a.max_soft_tokens)
    tok = AutoTokenizer.from_pretrained("data/checkpoints/encoder-free-v0/best")

    target = MIXES[a.mix]
    stats = {}
    for name in target:
        logger.info("stats for %s", name)
        stats[name] = lane_stats(name, LANES[name], "train", a.stat_rows, tok,
                                 a.max_length, per_image, a.scan_depth)

    p = plan(stats, target, a.total_rows)
    print(f"\nPLAN — mixture {a.mix}, {a.total_rows:,} rows, max_length {a.max_length}")
    hdr = (f"{'lane':<20}{'rows':>10}{'row %':>9}{'sup tok/row':>13}{'TOKEN %':>10}"
           f"{'target %':>10}{'lane readable %':>17}")
    print(hdr)
    print("-" * len(hdr))
    for k, v in sorted(p["realised"]["per_lane"].items(),
                       key=lambda kv: -kv[1]["token_share"]):
        print(f"{k:<20}{v['rows']:>10,}{100 * v['row_share']:>8.1f}%"
              f"{v['sup_tokens_mean']:>13.1f}{100 * v['token_share']:>9.1f}%"
              f"{100 * target[k]:>9.1f}%"
              f"{100 * v['readable_fraction_within_lane']:>16.1f}%")
    print(f"\n  token-weighted readable fraction (depth {a.scan_depth}): "
          f"{100 * p['realised']['readable_fraction_overall']:.1f}%  "
          f"(H13 died 14.2% | H14 confirmed 75.2% | H07 100% — all three measured "
          f"at depth {SCAN_DEPTH_RETIRED})")
    if a.scan_depth == SCAN_DEPTH_RETIRED:
        print(f"  !! {depth_note}")
    for k, c in p["clamped"].items():
        print(f"  !! {k} CLAMPED: wanted {c['wanted']:,}, only {c['available']:,} "
              f"exist ({c['epochs_if_repeated']} epochs would be needed)")

    report = {"mix": a.mix, "total_rows": a.total_rows, "max_length": a.max_length,
              "max_soft_tokens": a.max_soft_tokens, "per_image_tokens": per_image,
              "scan_depth": a.scan_depth,
              "scan_depth_is_retired_default":
                  bool(a.scan_depth == SCAN_DEPTH_RETIRED),
              "scan_depth_note": depth_note,
              "stats": stats, "plan": p}
    if not a.plan_only:
        out = Path(a.out)
        entries = build(out, stats, p["rows_built"], "train", a.max_length,
                        per_image, a.seed, a.val_rows)
        manifest = {"artifact": out.name, "canvas": {"height": 1024, "width": 1024},
                    "subsets": entries,
                    "note": "built by scratchpad/h19_build_mixture.py; rows are "
                            "pre-filtered at max_length so univi.trainer's own "
                            "filter retains 100.00%, and extra columns are removed "
                            "so every lane has identical Features."}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        report["entries"] = entries
        report["verify"] = verify(out, entries)
        print(f"\nVERIFY: {json.dumps(report['verify'], indent=2)}")

    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", a.report)


if __name__ == "__main__":
    main()
