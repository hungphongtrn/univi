"""H19 — pre-launch audit of the four-lane rematch mixture.

Everything H13 §3 says must be checked before a multi-lane run is launched, plus
the two integration hazards H19's own doc names.  CPU ONLY: no GPU, no model
weights, no training.

WHAT IT MEASURES
----------------
1. **Token-weighted gradient share.**  HF's ``ForCausalLMLoss`` normalises over
   SUPERVISED TOKENS, so a lane's share of the gradient is
   ``rows_L * supervised_tokens_per_row_L`` / the sum over lanes — NOT its row
   share.  H13 balanced rows 20k/rung and d4 still took 63.04% of the gradient;
   85.8% of all supervised tokens lay past the ~48-token scan depth and the run
   collapsed to the uniform-letter marginal.
2. **Token-weighted readable fraction**, per lane and overall:
   ``sum_rows min(48, T_row) / sum_rows T_row``.  Reference points: H13 died at
   14.2%, H14 was repaired to 75.2% before launching and CONFIRMED, H07 succeeded
   at 100%.
3. **Retention** through the REAL ``univi.trainer._filter_training_tokens`` at the
   configured ``max_length`` and per-image budget — the filter silently DROPS
   overflowing rows, it does not truncate them.
4. **Assembled sequence length** through the REAL ``univi.hybrid.data.HybridCollator``
   (which truncates at ``max_length``), including the per-lane soft-token count,
   which is NOT 256 everywhere: a 1024x1024 page emits 256, but densefusion's
   arbitrary aspect ratios emit up to ~272 and librispeech's 1000x160
   spectrogram emits 246.
5. **``concatenate_datasets`` against mismatched schemas** — what ACTUALLY happens
   when ``poisoned-text`` (3 extra columns) or ``masked-randstr`` (12 extra
   columns) meets a standard lane, and that ``remove_columns`` repairs it.
6. **Peak VRAM**, analytically, against the 26 GB ceiling.
7. **A repair solver**: the per-lane ROW COUNTS that hit a requested TOKEN-share
   target, reported with the resulting readable fraction.

    CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 uv run python \\
        scratchpad/h19_mixture_audit.py -n 400

    uv run python scratchpad/h19_mixture_audit.py --self-test
"""
from __future__ import annotations

import argparse
import importlib.machinery
import json
import logging
import statistics
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("h19_audit")

#: DEFAULT ONLY, AND IT IS THE RETIRED NUMBER.  48 came from H13 §4's K
#: ("K = 48.5 answer tokens over a 16x density span"), and K is now known to be
#: STRUCTURALLY FLOORED: `scratchpad/h13_analyze.k_from_buckets` scans buckets
#: strictly past a 100-char baseline window, so k_chars >= 100 BY CONSTRUCTION and
#: every K ever reported sat on that floor (an identical 48.4913 tokens at
#: soft-token budgets 280 AND 560, for rungs whose real reading differed 2.8x).
#: The position-resolved read-outs put the readable depth at ~5-12 answer tokens.
#: It stays 48 by default so existing artifacts remain reproducible; pass
#: `--scan-depth` to audit a mixture at the depth actually measured.
SCAN_DEPTH = 48
SCAN_DEPTH_RETIRED = 48
SCAN_DEPTH_MEASURED_RANGE = (5, 12)
TRAILER_TOKENS = 2    # response-only masking also supervises `<|im_end|>\n`
CEILING_GB = 26.0


def warn_scan_depth(depth: int, log=logger) -> str:
    """One loud line about what the chosen scan depth is worth."""
    if depth == SCAN_DEPTH_RETIRED:
        msg = (f"scan depth {depth} is the RETIRED K FLOOR (H13's K is structurally "
               f"floored at the 100-char baseline window, so every reported K was "
               f"that floor). The position-resolved read-outs measure the readable "
               f"depth at ~{SCAN_DEPTH_MEASURED_RANGE[0]}-"
               f"{SCAN_DEPTH_MEASURED_RANGE[1]} answer tokens, so every readable "
               f"fraction below is an UPPER BOUND — re-run with --scan-depth 5 and "
               f"--scan-depth 12 before using it as a launch gate.")
        log.warning(msg)
    else:
        msg = (f"scan depth {depth} (default {SCAN_DEPTH_RETIRED} is the retired K "
               f"floor; measured readable depth is ~{SCAN_DEPTH_MEASURED_RANGE[0]}-"
               f"{SCAN_DEPTH_MEASURED_RANGE[1]} answer tokens)")
        log.info(msg)
    return msg

#: Every lane H19 might mix, and where it lives.  Roots DIFFER — `_load_local`
#: takes a single `dataset.path`, so a real H19 needs one merged root (symlinks
#: for schema-clean lanes, a `remove_columns` rewrite for the rest).
LANES = {
    "fineweb-edu":        "data/materialized/univi-3M-v0-split/fineweb-edu",
    "densefusion":        "data/materialized/univi-3M-v0-split/densefusion",
    "smoltalk":           "data/materialized/univi-3M-v0-split/smoltalk",
    "librispeech":        "data/materialized/univi-3M-v0-split/librispeech",
    "random-strings":     "data/materialized/h3-randstr-v0/random-strings",
    "spoken-digits":      "data/materialized/spoken-digits-v0/spoken-digits",
    "poisoned-text":      "data/materialized/h15-poisoned-p00/poisoned-text",
    "poisoned-text-long": "data/materialized/h15-poisoned-p00/poisoned-text-long",
    "masked-randstr":     "data/materialized/h14-masked-short-v0/masked-randstr",
}

#: The mixture exactly as H19's doc specifies it (ROW shares).
MIX_AS_DESIGNED = {
    "fineweb-edu": 0.175, "densefusion": 0.175, "smoltalk": 0.175,
    "librispeech": 0.175, "random-strings": 0.15, "spoken-digits": 0.15,
}

#: MIXTURE B — the doc's OWN numbers, read as TOKEN shares instead of row shares.
#: Nothing about the design changes except which quantity is balanced.
MIX_B_TOKEN_SHARES = dict(MIX_AS_DESIGNED)

#: MIXTURE C — B, plus H15's two-length-arm move on the text lane: most of the
#: text token share goes to the readable short arm, the rest to the dense page.
MIX_C_TOKEN_SHARES = {
    "poisoned-text": 0.13, "fineweb-edu": 0.045, "densefusion": 0.175,
    "smoltalk": 0.175, "librispeech": 0.175, "random-strings": 0.15,
    "spoken-digits": 0.15,
}


def stub_unsloth() -> None:
    """`univi/__init__.py` imports unsloth at module scope and hard-fails without
    CUDA.  Stub it (with a real __spec__ — trl calls importlib.util.find_spec)."""
    if "unsloth" in sys.modules:
        return
    m = types.ModuleType("unsloth")
    m.__spec__ = importlib.machinery.ModuleSpec("unsloth", None)
    m.__path__ = []
    m.FastVisionModel = object
    m.UnslothVisionDataCollator = object
    m.is_bfloat16_supported = lambda: False
    sys.modules["unsloth"] = m


# ---------------------------------------------------------------------------
# pure share / readable maths (self-tested; no data needed)
# ---------------------------------------------------------------------------
def share_table(lanes: dict, rows: dict) -> dict:
    """``lanes`` = {name: {"sup_tokens_mean", "readable_tokens_mean"}}, ``rows`` =
    {name: n_rows}.  Returns row share, TOKEN share, and readable fractions."""
    tok = {k: rows[k] * lanes[k]["sup_tokens_mean"] for k in rows}
    rdb = {k: rows[k] * lanes[k]["readable_tokens_mean"] for k in rows}
    n_rows = sum(rows.values())
    n_tok = sum(tok.values())
    n_rdb = sum(rdb.values())
    per_lane = {}
    for k in rows:
        per_lane[k] = {
            "rows": rows[k],
            "row_share": rows[k] / n_rows if n_rows else 0.0,
            "sup_tokens_mean": lanes[k]["sup_tokens_mean"],
            "supervised_tokens": tok[k],
            "token_share": tok[k] / n_tok if n_tok else 0.0,
            "readable_tokens_mean": lanes[k]["readable_tokens_mean"],
            "readable_fraction_within_lane":
                lanes[k]["readable_tokens_mean"] / lanes[k]["sup_tokens_mean"],
        }
    return {
        "per_lane": per_lane,
        "total_rows": n_rows,
        "total_supervised_tokens": n_tok,
        "readable_fraction_overall": (n_rdb / n_tok) if n_tok else 0.0,
        # the three anchors were all computed at depth 48, so they are comparable
        # with a depth-48 measurement and NOT with a depth-5/12 one
        "reference_points": {"H13_died_at": 0.142, "H14_confirmed_at": 0.752,
                             "H07_succeeded_at": 1.0},
        "reference_points_measured_at_scan_depth": SCAN_DEPTH_RETIRED,
    }


def rows_for_token_share(lanes: dict, target_share: dict, total_rows: int) -> dict:
    """Per-lane row counts that realise ``target_share`` of the SUPERVISED TOKENS
    while summing to ``total_rows``.  n_L is proportional to share_L / T_L."""
    w = {k: target_share[k] / lanes[k]["sup_tokens_mean"] for k in target_share}
    s = sum(w.values())
    return {k: int(round(total_rows * w[k] / s)) for k in w}


def rows_from_row_share(row_share: dict, total_rows: int) -> dict:
    return {k: int(round(total_rows * v)) for k, v in row_share.items()}


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------
def prefilter_indices(ds, max_length: int | None, per_image: int, scan: int = 400_000):
    """Row indices that survive ``_filter_training_tokens`` — the rows a run at this
    ``max_length`` would ACTUALLY train on.  Measuring the lane on the unfiltered
    split overstates its target length whenever the filter drops the long tail."""
    import pyarrow.compute as pc

    from univi.trainer import _TEMPLATE_TOKEN_OVERHEAD

    if max_length is None:
        return None
    otl = ds.data.column("original_token_length")
    n_img = pc.list_value_length(ds.data.column("images"))
    est = pc.add(pc.add(otl, pc.multiply(n_img, per_image)), _TEMPLATE_TOKEN_OVERHEAD)
    keep = pc.or_(pc.less_equal(otl, 0), pc.less_equal(est, max_length)).to_pylist()
    return [i for i, v in enumerate(keep[:scan]) if v]


def measure_lane(name: str, root: str, split: str, n: int, tokenizer,
                 image_processor=None, n_images_sample: int = 64,
                 prefilter_max_length: int | None = None,
                 per_image: int = 282, scan_depth: int = SCAN_DEPTH) -> dict:
    from datasets import load_from_disk

    path = Path(root) / split
    if not path.exists():
        return {"name": name, "status": "MISSING", "path": str(path)}
    ds = load_from_disk(str(path))
    total = len(ds)
    retained = total
    if prefilter_max_length is not None:
        import pyarrow.compute as pc

        from univi.trainer import _TEMPLATE_TOKEN_OVERHEAD

        otl = ds.data.column("original_token_length")
        ni = pc.list_value_length(ds.data.column("images"))
        est = pc.add(pc.add(otl, pc.multiply(ni, per_image)), _TEMPLATE_TOKEN_OVERHEAD)
        keep = pc.or_(pc.less_equal(otl, 0),
                      pc.less_equal(est, prefilter_max_length))
        retained = int(pc.sum(pc.cast(keep, "int64")).as_py() or 0)
        ds = ds.select(prefilter_indices(ds, prefilter_max_length, per_image))
    k = min(n, len(ds))
    sub = ds.select(range(k)).select_columns(["messages", "original_token_length"])

    sup, turns = [], []
    for row in sub:
        t, c = 0, 0
        for m in row["messages"]:
            if m["role"] != "assistant":
                continue
            c += 1
            cc = m["content"]
            txt = ("".join(p.get("text") or "" for p in cc)
                   if isinstance(cc, list) else str(cc))
            t += len(tokenizer.encode(txt, add_special_tokens=False)) + TRAILER_TOKENS
        sup.append(t)
        turns.append(c)

    # images / soft tokens on a smaller sample (decoding pixels is the expensive part)
    ki = min(n_images_sample, total)
    imgs = ds.select(range(ki)).select_columns(["images"])
    n_img, soft, sizes = [], [], []
    for row in imgs:
        ims = row["images"] or []
        n_img.append(len(ims))
        sizes.extend(im.size for im in ims)
        if image_processor is not None and ims:
            out = image_processor(images=[im.convert("RGB") for im in ims],
                                  return_tensors="pt")
            soft.append(int(out["num_soft_tokens_per_image"].sum()))
    srt = sorted(sup)
    return {
        "name": name, "status": "ok", "path": str(path),
        "rows_available": total, "n_sampled": k,
        "prefilter_max_length": prefilter_max_length,
        "rows_available_after_prefilter": retained,
        "sup_tokens_mean": statistics.fmean(sup),
        "sup_tokens_p50": statistics.median(sup),
        "sup_tokens_p99": srt[int(0.99 * (len(srt) - 1))],
        "sup_tokens_max": max(sup),
        "assistant_turns_mean": statistics.fmean(turns),
        "scan_depth": scan_depth,
        # exactly the quantity the gradient sees: mean over rows of min(depth, T)
        "readable_tokens_mean": statistics.fmean(min(scan_depth, t) for t in sup),
        "readable_fraction_token_weighted":
            sum(min(scan_depth, t) for t in sup) / max(sum(sup), 1),
        "rows_fully_readable": sum(t <= scan_depth for t in sup) / len(sup),
        "n_images_mean": statistics.fmean(n_img),
        "n_images_max": max(n_img),
        "n_images_sampled": ki,
        "image_sizes_seen": sorted({s for s in sizes})[:6],
        "soft_tokens_per_row_mean": (statistics.fmean(soft) if soft else None),
        "soft_tokens_per_row_max": (max(soft) if soft else None),
    }


def retention(name: str, root: str, split: str, max_length: int, per_image: int,
              num_proc: int, crosscheck_rows: int = 2000) -> dict:
    """Retention over the WHOLE split, plus a cross-check against the real filter.

    ``_filter_training_tokens`` reads two columns and drops rows whose
    ``otl + n_images * per_image + 256`` exceeds ``max_length``.  Running the real
    ``.filter()`` over fineweb/densefusion/smoltalk would page ~280 GB of image
    bytes off disk, so the predicate is evaluated directly on the memory-mapped
    Arrow columns (offsets only for ``images``) for the full split, and the REAL
    filter is run on the first ``crosscheck_rows`` rows to prove the two agree.
    """
    import pyarrow.compute as pc
    from datasets import load_from_disk

    from univi.trainer import _TEMPLATE_TOKEN_OVERHEAD, _filter_training_tokens

    path = Path(root) / split
    if not path.exists():
        return {"status": "MISSING"}
    ds = load_from_disk(str(path))
    before = len(ds)
    otl = ds.data.column("original_token_length")
    n_img = pc.list_value_length(ds.data.column("images"))
    est = pc.add(pc.add(otl, pc.multiply(n_img, per_image)), _TEMPLATE_TOKEN_OVERHEAD)
    keep = pc.or_(pc.less_equal(otl, 0), pc.less_equal(est, max_length))
    after = int(pc.sum(pc.cast(keep, "int64")).as_py() or 0)

    k = min(crosscheck_rows, before)
    real = len(_filter_training_tokens(ds.select(range(k)), name, max_length,
                                       num_proc=None, per_image_tokens=per_image))
    mine = int(pc.sum(pc.cast(
        pc.or_(pc.less_equal(otl.slice(0, k), 0),
               pc.less_equal(pc.add(pc.add(otl.slice(0, k),
                                           pc.multiply(n_img.slice(0, k), per_image)),
                                    _TEMPLATE_TOKEN_OVERHEAD), max_length)),
        "int64")).as_py() or 0)
    return {"rows_before": before, "rows_after": after,
            "retention": after / before if before else None,
            "dropped": before - after, "max_length": max_length,
            "per_image_tokens": per_image,
            "estimated_len_max": int(pc.max(est).as_py()),
            "crosscheck_rows": k, "crosscheck_real_filter_kept": real,
            "crosscheck_predicate_kept": mine,
            "crosscheck_agrees": bool(real == mine)}


def assembled_lengths(name: str, root: str, split: str, n: int, collator,
                      prefilter_max_length: int | None = None,
                      per_image: int = 282) -> dict:
    """Real HybridCollator, real rows: the sequence the model actually sees.

    With ``prefilter_max_length`` the rows are taken from the set the length
    filter would KEEP — otherwise the longest rows are ones the run never sees and
    the reported max is a phantom truncation.
    """
    from datasets import load_from_disk

    path = Path(root) / split
    if not path.exists():
        return {"status": "MISSING"}
    full = load_from_disk(str(path))
    if prefilter_max_length is not None:
        full = full.select(prefilter_indices(full, prefilter_max_length, per_image))
    ds = full.select(range(min(n, len(full))))
    lens, sup = [], []
    for i in range(len(ds)):
        b = collator([ds[i]])
        lens.append(int(b["input_ids"].shape[1]))
        sup.append(int((b["labels"] != -100).sum()))
    return {"n": len(lens), "assembled_mean": statistics.fmean(lens),
            "assembled_max": max(lens), "supervised_mean": statistics.fmean(sup),
            "supervised_max": max(sup)}


# ---------------------------------------------------------------------------
# the concatenate_datasets hazard
# ---------------------------------------------------------------------------
def concat_check(tokenizer=None) -> dict:
    """What ACTUALLY happens when lanes with different Features are concatenated."""
    from datasets import concatenate_datasets, load_from_disk

    std = "data/materialized/h3-randstr-v0/random-strings/validation"
    out = {"standard_lane": std, "cases": []}
    if not Path(std).exists():
        return {"status": "MISSING", **out}
    a = load_from_disk(std).select(range(2))
    base_cols = list(a.column_names)
    out["standard_columns"] = base_cols

    for name, root in (("poisoned-text", LANES["poisoned-text"]),
                       ("masked-randstr", LANES["masked-randstr"])):
        p = Path(root) / "validation"
        case = {"lane": name, "path": str(p)}
        if not p.exists():
            case["status"] = "MISSING"
            out["cases"].append(case)
            continue
        b = load_from_disk(str(p)).select(range(2))
        case["columns"] = list(b.column_names)
        case["extra_columns"] = [c for c in b.column_names if c not in base_cols]
        case["missing_columns"] = [c for c in base_cols if c not in b.column_names]
        try:
            m = concatenate_datasets([a, b])
            case["raw_concat"] = {
                "raised": False, "rows": len(m),
                "columns": list(m.column_names),
                "WARNING": "silently produced a table — inspect for null padding",
            }
        except Exception as e:  # noqa: BLE001 — the point is to record the type
            case["raw_concat"] = {"raised": True, "error_type": type(e).__name__,
                                  "error": str(e)[:400]}
        stripped = b.remove_columns(case["extra_columns"])
        try:
            m = concatenate_datasets([a, stripped])
            case["after_remove_columns"] = {
                "raised": False, "rows": len(m), "columns": list(m.column_names),
                "features_equal": bool(m.features == a.features),
            }
        except Exception as e:  # noqa: BLE001
            case["after_remove_columns"] = {"raised": True,
                                            "error_type": type(e).__name__,
                                            "error": str(e)[:400]}
        out["cases"].append(case)

    # column ORDER: smoltalk stores messages before images
    st = Path(LANES["smoltalk"]) / "validation"
    if st.exists():
        s = load_from_disk(str(st)).select(range(2))
        try:
            m = concatenate_datasets([a, s])
            out["column_order_case"] = {
                "smoltalk_columns": list(s.column_names),
                "raised": False, "rows": len(m),
                "note": "column ORDER differs from the randstr lane and is tolerated",
            }
        except Exception as e:  # noqa: BLE001
            out["column_order_case"] = {"raised": True, "error_type": type(e).__name__,
                                        "error": str(e)[:400]}
    return out


# ---------------------------------------------------------------------------
# VRAM
# ---------------------------------------------------------------------------
def estimate_vram(seq_len: int, batch: int, max_soft_tokens: int = 280) -> dict:
    """Same accounting as scratchpad/h17_config_verify.estimate_vram."""
    import torch
    from transformers import AutoTokenizer, Qwen3Config

    from univi.hybrid.build import load_vision_config
    from univi.hybrid.pretrained import (
        IMAGE_PLACEHOLDER, UniViHybridPretrained, UniViHybridPretrainedConfig,
    )

    tok = AutoTokenizer.from_pretrained("unsloth/Qwen3-1.7B")
    if IMAGE_PLACEHOLDER not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [IMAGE_PLACEHOLDER]})
    tc = Qwen3Config.from_pretrained("unsloth/Qwen3-1.7B")
    tc.vocab_size = len(tok)
    cfg = UniViHybridPretrainedConfig(
        vision_config=load_vision_config("google/gemma-4-12B-it"),
        text_config=tc, image_token_id=tok.convert_tokens_to_ids(IMAGE_PLACEHOLDER),
        max_soft_tokens=max_soft_tokens,
    )
    with torch.device("meta"):
        m = UniViHybridPretrained(cfg)
    n_par = sum(p.numel() for p in m.parameters())
    del m
    V, H, L = tc.vocab_size, tc.hidden_size, tc.num_hidden_layers
    GB = 1024.0 ** 3
    static = n_par * (2 + 2 + 2) / GB
    ctx = 0.9

    def leg(bs, s, slots, bpl=12):
        logits = bs * s * V * bpl / GB
        dec = L * bs * s * H * 2 / GB + 20 * bs * s * H * 2 / GB
        vis = bs * slots * (6912 + 5 * 3840) * 2 / GB
        return {"per_device_batch": bs, "max_seq_len": s,
                "tokens_per_microbatch": bs * s,
                "static_GB": round(static, 2), "logits_GB": round(logits, 2),
                "decoder_activations_GB": round(dec, 2),
                "vision_activations_GB": round(vis, 2), "cuda_context_GB": ctx,
                "total_GB": round(static + logits + dec + vis + ctx, 2),
                "total_GB_pessimistic_14B_per_logit":
                    round(static + logits * 14 / 12 + dec + vis + ctx, 2)}

    return {
        "params_total": n_par, "vocab_size": V, "ceiling_GB": CEILING_GB,
        "max_soft_tokens": max_soft_tokens,
        "sweep": {f"bs{bs}": leg(bs, seq_len, max_soft_tokens)
                  for bs in (1, 2, 4, 8, 16)},
        "chosen": leg(batch, seq_len, max_soft_tokens),
        "calibration_anchors": {
            "H07 (ran OK) bs16 seq~303": leg(16, 303, 280)["total_GB"],
            "H13 (ran OK, ~21GB measured) bs4 seq~1220": leg(4, 1220, 280)["total_GB"],
            "hybrid_4lane (ran OK) bs16 seq~2048": leg(16, 2048, 280)["total_GB"],
        },
    }


# ---------------------------------------------------------------------------
def print_share(title: str, tbl: dict, scan_depth: int = SCAN_DEPTH) -> None:
    print(f"\n{'=' * 108}")
    print(title)
    print(f"{'=' * 108}")
    hdr = (f"{'lane':<20}{'rows':>10}{'row %':>9}{'sup tok/row':>13}"
           f"{'TOKEN %':>10}{'readable/row':>14}{'lane readable %':>17}")
    print(hdr)
    print("-" * len(hdr))
    for k, v in sorted(tbl["per_lane"].items(), key=lambda kv: -kv[1]["token_share"]):
        print(f"{k:<20}{v['rows']:>10,}{100 * v['row_share']:>8.1f}%"
              f"{v['sup_tokens_mean']:>13.1f}{100 * v['token_share']:>9.1f}%"
              f"{v['readable_tokens_mean']:>14.1f}"
              f"{100 * v['readable_fraction_within_lane']:>16.1f}%")
    print("-" * len(hdr))
    print(f"{'TOTAL':<20}{tbl['total_rows']:>10,}{'100.0%':>9}"
          f"{tbl['total_supervised_tokens'] / max(tbl['total_rows'], 1):>13.1f}"
          f"{'100.0%':>10}")
    r = tbl["readable_fraction_overall"]
    ref = tbl["reference_points"]
    verdict = ("DEATH REGIME (H13 died at 14.2%)" if r < 0.30 else
               "MARGINAL — above H13's 14.2% but below H14's 75.2%" if r < 0.60 else
               "CLEAR of the failure regime, still below H14's 75.2%" if r < 0.752 else
               "AT OR ABOVE H14's confirmed 75.2%")
    if scan_depth != SCAN_DEPTH_RETIRED:
        # H13/H14/H07 were all measured at depth 48; the bands mean nothing against
        # a fraction computed at another depth, so do not pretend to name a regime.
        verdict = (f"NO VERDICT — the 14.2/75.2/100% bands were measured at depth "
                   f"{SCAN_DEPTH_RETIRED}; at depth {scan_depth} only RANKINGS "
                   f"between mixtures are meaningful, not the bands ({verdict!r} "
                   f"would be the depth-48 reading of this number)")
    print(f"\n  TOKEN-WEIGHTED READABLE FRACTION (depth {scan_depth}): "
          f"{100 * r:.1f}%   -> {verdict}")
    print(f"  reference: H13 died {100 * ref['H13_died_at']:.1f}% | "
          f"H14 confirmed {100 * ref['H14_confirmed_at']:.1f}% | "
          f"H07 succeeded {100 * ref['H07_succeeded_at']:.0f}%")
    if scan_depth == SCAN_DEPTH_RETIRED:
        print(f"  !! depth {scan_depth} is the RETIRED K floor; the reference points "
              f"above were all computed at it, so the COMPARISON is valid but the "
              f"absolute number is an upper bound (measured depth "
              f"~{SCAN_DEPTH_MEASURED_RANGE[0]}-{SCAN_DEPTH_MEASURED_RANGE[1]} tok)")


def self_test() -> None:
    # H13 §3's own table must come back out of share_table()
    lanes = {
        "d1": {"sup_tokens_mean": 17, "readable_tokens_mean": 17},
        "d2": {"sup_tokens_mean": 60, "readable_tokens_mean": 48},
        "d3": {"sup_tokens_mean": 234, "readable_tokens_mean": 48},
        "d4": {"sup_tokens_mean": 932, "readable_tokens_mean": 48},
        "d5": {"sup_tokens_mean": 234, "readable_tokens_mean": 48},
    }
    rows = {k: 20000 for k in lanes}
    t = share_table(lanes, rows)
    got = {k: round(100 * v["token_share"], 2) for k, v in t["per_lane"].items()}
    # H13 §3 published 1.13 / 4.11 / 15.86 / 63.04 / 15.86 from unrounded per-lane
    # means; this recomputation uses the doc's ROUNDED tok/row column, so agreement
    # is to ~0.1 pp, not exact.
    want = {"d1": 1.13, "d2": 4.11, "d3": 15.86, "d4": 63.04, "d5": 15.86}
    for k, v in want.items():
        assert abs(got[k] - v) <= 0.2, (k, got[k], v)
    print(f"  ok  share_table reproduces H13 §3's gradient-share table "
          f"(within 0.2 pp of the published 1.13/4.11/15.86/63.04/15.86): {got}")
    assert abs(t["readable_fraction_overall"] - 0.142) < 0.006, \
        t["readable_fraction_overall"]
    print(f"  ok  H13's 14.2% readable fraction reproduced: "
          f"{100 * t['readable_fraction_overall']:.1f}%")

    # balancing ROWS does not balance GRADIENT — the whole lesson, as an assertion
    assert t["per_lane"]["d4"]["row_share"] == t["per_lane"]["d1"]["row_share"]
    assert t["per_lane"]["d4"]["token_share"] > 50 * t["per_lane"]["d1"]["token_share"]
    print("  ok  equal ROW share -> 55x unequal TOKEN share (H13's lesson)")

    # the solver inverts it
    target = {k: 0.2 for k in lanes}
    r = rows_for_token_share(lanes, target, 100_000)
    t2 = share_table(lanes, r)
    for k, v in t2["per_lane"].items():
        assert abs(v["token_share"] - 0.2) < 0.002, (k, v["token_share"])
    print(f"  ok  rows_for_token_share equalises TOKEN share to 20.0% each: "
          f"{ {k: v for k, v in r.items()} }")

    # a share target that is not uniform
    target = {"d1": 0.1, "d2": 0.1, "d3": 0.3, "d4": 0.3, "d5": 0.2}
    r = rows_for_token_share(lanes, target, 100_000)
    t3 = share_table(lanes, r)
    for k in target:
        assert abs(t3["per_lane"][k]["token_share"] - target[k]) < 0.002, k
    print("  ok  the solver hits a NON-uniform token-share target too")

    # H07's single 100%-readable lane
    t4 = share_table({"x": {"sup_tokens_mean": 16.5, "readable_tokens_mean": 16.5}},
                     {"x": 100})
    assert t4["readable_fraction_overall"] == 1.0
    print("  ok  a fully-readable single lane scores 100% (H07)")

    # the scan depth is a knob now, and the default announces what it is worth
    quiet = logging.getLogger("h19_audit.selftest")
    quiet.setLevel(logging.CRITICAL)
    assert "RETIRED K FLOOR" in warn_scan_depth(SCAN_DEPTH_RETIRED, quiet)
    assert "retired K floor" in warn_scan_depth(5, quiet)
    assert t4["reference_points_measured_at_scan_depth"] == SCAN_DEPTH_RETIRED
    # the same lane at depth 5 must score 5/16.5, not 100%
    t5 = share_table({"x": {"sup_tokens_mean": 16.5, "readable_tokens_mean": 5.0}},
                     {"x": 100})
    assert abs(t5["readable_fraction_overall"] - 5.0 / 16.5) < 1e-12
    print(f"  ok  --scan-depth is honoured end-to-end (H07's lane: 100% at depth 48, "
          f"{100 * t5['readable_fraction_overall']:.1f}% at depth 5) and depth 48 "
          f"announces itself as the retired K floor")
    print("\nSELF-TEST PASSED (gradient share, readable fraction, row solver, "
          "scan-depth knob)")


# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-n", type=int, default=400, help="rows sampled per lane for tokens")
    ap.add_argument("--image-sample", type=int, default=32)
    ap.add_argument("--collator-sample", type=int, default=12)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--max-soft-tokens", type=int, default=280)
    ap.add_argument("--total-rows", type=int, default=160_000,
                    help="rows the run will see (max_steps x effective batch)")
    ap.add_argument("--num-proc", type=int, default=4)
    ap.add_argument("--scan-depth", type=int, default=SCAN_DEPTH,
                    help=f"answer-token depth the readable fraction is computed to. "
                         f"Default {SCAN_DEPTH} keeps existing artifacts "
                         f"reproducible, but it is the RETIRED K floor (K is "
                         f"structurally floored at the 100-char baseline window); "
                         f"the measured readable depth is "
                         f"~{SCAN_DEPTH_MEASURED_RANGE[0]}-"
                         f"{SCAN_DEPTH_MEASURED_RANGE[1]} answer tokens.")
    ap.add_argument("--tokenizer", default="data/checkpoints/encoder-free-v0/best")
    ap.add_argument("--lanes", nargs="*", default=list(LANES))
    ap.add_argument("--prefilter", action="store_true",
                    help="measure each lane on the rows that SURVIVE the length "
                         "filter at --max-length (what a run actually trains on)")
    ap.add_argument("--skip-retention", action="store_true")
    ap.add_argument("--skip-collator", action="store_true")
    ap.add_argument("--out", default="data/eval/h19-mixture-audit.json")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        print("SELF-TEST (CPU, no data)")
        self_test()
        return

    depth_note = warn_scan_depth(a.scan_depth)

    stub_unsloth()
    from transformers import AutoTokenizer

    from univi.hybrid.data import HybridCollator
    from univi.hybrid.pretrained import build_image_processor
    from univi.trainer import image_token_budget

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    proc = build_image_processor(a.max_soft_tokens)
    per_image = image_token_budget(a.max_soft_tokens)

    lanes = {}
    for name in a.lanes:
        logger.info("measuring %s", name)
        lanes[name] = measure_lane(
            name, LANES[name], a.split, a.n, tok, proc, a.image_sample,
            prefilter_max_length=(a.max_length if a.prefilter else None),
            per_image=per_image, scan_depth=a.scan_depth)

    ok = {k: v for k, v in lanes.items() if v.get("status") == "ok"}

    print(f"\n{'=' * 118}")
    print(f"PER-LANE TOKEN GEOMETRY  (Qwen3 tokenizer, split={a.split}, "
          f"n={a.n} rows/lane, scan depth {a.scan_depth})")
    print(f"{'=' * 118}")
    print(f"  NOTE: {depth_note}")
    if a.prefilter:
        print(f"  (lanes measured on the rows that SURVIVE the length filter at "
              f"max_length={a.max_length} — i.e. the rows a run would train on)")
    hdr = (f"{'lane':<20}{'rows avail':>12}{'sup tok mean':>14}{'p50':>7}{'p99':>7}"
           f"{'max':>8}{'readable %':>12}{'img/row':>9}{'soft/row':>10}")
    print(hdr)
    print("-" * len(hdr))
    for k, v in ok.items():
        print(f"{k:<20}"
              f"{v.get('rows_available_after_prefilter', v['rows_available']):>12,}"
              f"{v['sup_tokens_mean']:>14.1f}"
              f"{v['sup_tokens_p50']:>7.0f}{v['sup_tokens_p99']:>7.0f}"
              f"{v['sup_tokens_max']:>8.0f}"
              f"{100 * v['readable_fraction_token_weighted']:>11.1f}%"
              f"{v['n_images_mean']:>9.2f}"
              f"{(v['soft_tokens_per_row_max'] or 0):>10.0f}")

    mixes = {}
    have = set(ok)

    if set(MIX_AS_DESIGNED) <= have:
        rows = rows_from_row_share(MIX_AS_DESIGNED, a.total_rows)
        t = share_table(ok, rows)
        mixes["as_designed_row_shares"] = {"rows": rows, "table": t}
        print_share("MIXTURE A — H19's doc, VERBATIM (row shares 17.5/17.5/17.5/17.5/15/15)",
                    t, a.scan_depth)

    for tag, spec, title in (
        ("B_token_shares", MIX_B_TOKEN_SHARES,
         "MIXTURE B — the doc's OWN 17.5/17.5/17.5/17.5/15/15, read as TOKEN shares"),
        ("C_split_text_lane", MIX_C_TOKEN_SHARES,
         "MIXTURE C — B + H15's two-length-arm split of the text lane "
         "(13% short arm / 4.5% dense page)"),
    ):
        if not set(spec) <= have:
            continue
        rows = rows_for_token_share(ok, spec, a.total_rows)
        t = share_table(ok, rows)
        mixes[tag] = {"target_token_share": spec, "rows": rows, "table": t}
        print_share(title, t, a.scan_depth)
        short = {k: v for k, v in rows.items()
                 if v > ok[k].get("rows_available_after_prefilter",
                                  ok[k]["rows_available"])}
        if short:
            print("\n  !! lanes asking for MORE rows than are available "
                  "(would repeat epochs):")
            for k, v in short.items():
                av = ok[k].get("rows_available_after_prefilter",
                               ok[k]["rows_available"])
                print(f"     {k}: needs {v:,}, has {av:,} ({v / av:.2f} epochs)")

    # sensitivity: what does the anchors' TOKEN share buy in readable fraction?
    if {"fineweb-edu", "densefusion", "smoltalk", "librispeech",
        "random-strings", "spoken-digits"} <= have:
        print("\n  -- sensitivity: anchor TOKEN share vs overall readable fraction")
        sens = {}
        for anchor in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
            real = (1.0 - anchor) / 4.0
            spec = {"fineweb-edu": real, "densefusion": real, "smoltalk": real,
                    "librispeech": real, "random-strings": anchor / 2,
                    "spoken-digits": anchor / 2}
            r = rows_for_token_share(ok, spec, a.total_rows)
            f = share_table(ok, r)["readable_fraction_overall"]
            sens[f"{anchor:.0%}"] = round(100 * f, 1)
            print(f"     anchors {anchor:>4.0%} of tokens -> readable {100 * f:5.1f}%"
                  + ("   [literature replay window 5-25%]"
                     if 0.05 <= anchor <= 0.25 else ""))
        mixes["anchor_share_sensitivity_readable_pct"] = sens

    ret = {}
    if not a.skip_retention:
        for k in ok:
            ret[k] = retention(k, LANES[k], a.split, a.max_length, per_image,
                               a.num_proc)
        print(f"\n{'=' * 96}")
        print(f"RETENTION through univi.trainer._filter_training_tokens "
              f"(max_length={a.max_length}, {per_image} tok/image)")
        print(f"{'=' * 96}")
        for k, v in ret.items():
            if v.get("status") == "MISSING":
                continue
            flag = "" if v["retention"] == 1.0 else "   <-- DROPS ROWS"
            print(f"  {k:<20}{v['rows_after']:>9,}/{v['rows_before']:<9,} "
                  f"= {100 * v['retention']:6.2f}%{flag}")

    asm = {}
    if not a.skip_collator:
        from univi.hybrid.pretrained import IMAGE_PLACEHOLDER

        if IMAGE_PLACEHOLDER not in tok.get_vocab():
            tok.add_special_tokens({"additional_special_tokens": [IMAGE_PLACEHOLDER]})
        col = HybridCollator(tokenizer=tok, image_processor=proc,
                             image_token_id=tok.convert_tokens_to_ids(IMAGE_PLACEHOLDER),
                             max_length=a.max_length, response_only=True)
        print(f"\n{'=' * 96}")
        print(f"ASSEMBLED SEQUENCE through the REAL HybridCollator "
              f"(max_length={a.max_length})")
        print(f"{'=' * 96}")
        for k in ok:
            asm[k] = assembled_lengths(
                k, LANES[k], a.split, a.collator_sample, col,
                prefilter_max_length=(a.max_length if a.prefilter else None),
                per_image=per_image)
            v = asm[k]
            flag = "  <-- TRUNCATED" if v["assembled_max"] >= a.max_length else ""
            print(f"  {k:<20} mean {v['assembled_mean']:8.1f}  max {v['assembled_max']:6d}"
                  f"   supervised mean {v['supervised_mean']:7.1f} "
                  f"max {v['supervised_max']:6d}{flag}")

    cc = concat_check()
    print(f"\n{'=' * 96}")
    print("concatenate_datasets AGAINST MISMATCHED SCHEMAS")
    print(f"{'=' * 96}")
    for case in cc.get("cases", []):
        if case.get("status") == "MISSING":
            print(f"  {case['lane']}: MISSING")
            continue
        raw = case["raw_concat"]
        print(f"  {case['lane']}: {len(case['extra_columns'])} extra columns "
              f"{case['extra_columns'][:4]}{'...' if len(case['extra_columns']) > 4 else ''}")
        print(f"     raw concat            -> "
              + (f"RAISED {raw['error_type']}: {raw['error'][:150]}"
                 if raw["raised"] else
                 f"NO ERROR ({raw['rows']} rows, {len(raw['columns'])} columns) "
                 f"** SILENT **"))
        af = case["after_remove_columns"]
        print(f"     after remove_columns  -> "
              + (f"RAISED {af['error_type']}" if af["raised"] else
                 f"OK, {af['rows']} rows, features identical: {af['features_equal']}"))
    co = cc.get("column_order_case")
    if co:
        print(f"  column ORDER (smoltalk stores messages before images) -> "
              + ("RAISED " + co["error_type"] if co.get("raised") else "tolerated"))

    seq = max((v["assembled_max"] for v in asm.values()), default=a.max_length)
    vram = estimate_vram(min(seq, a.max_length), 8, a.max_soft_tokens)
    print(f"\n{'=' * 96}")
    print(f"VRAM (analytic) at max seq {min(seq, a.max_length)}, "
          f"budget {a.max_soft_tokens}, ceiling {CEILING_GB} GB")
    print(f"{'=' * 96}")
    for k, v in vram["sweep"].items():
        mark = "OK " if v["total_GB_pessimistic_14B_per_logit"] <= CEILING_GB else "OVER"
        print(f"  {k:<6} {v['total_GB']:6.2f} GB  "
              f"(pessimistic {v['total_GB_pessimistic_14B_per_logit']:6.2f} GB)  {mark}")
    print(f"  calibration: {vram['calibration_anchors']}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": {"n": a.n, "split": a.split, "max_length": a.max_length,
                 "max_soft_tokens": a.max_soft_tokens, "per_image_tokens": per_image,
                 "total_rows": a.total_rows, "scan_depth": a.scan_depth,
                 "scan_depth_is_retired_default":
                     bool(a.scan_depth == SCAN_DEPTH_RETIRED),
                 "scan_depth_note": depth_note,
                 "tokenizer": a.tokenizer},
        "lanes": lanes, "mixtures": mixes, "retention": ret,
        "assembled": asm, "concat_check": cc, "vram": vram,
    }, indent=2))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
