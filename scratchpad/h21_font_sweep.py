"""H21 — glyph-scale (font) sweep of a checkpoint, paired by row.

LEG 0 (zero-training): probe ``hybrid-pretrained-randstr-v0/final`` across a font
ladder at fixed d2 geometry (119 chars) and turn H13 §6's two points into a curve.
LEG 1 (post-training): probe the font-mixture checkpoint and evaluate H21's
pre-registered CONFIRMS / REFUTES / PARTIAL / VOID table on the HELD-OUT fonts.

This is deliberately thin: every statistic comes from ``scratchpad/h13_analyze.py``
(``collect_rung`` / ``analyse_rung`` / ``_compare_pos`` / ``cluster_stats``), so the
numbers here are produced by exactly the code that produced H13 §4 and §6.

THE SANITY GATE COMES FIRST AND IS NOT OPTIONAL
----------------------------------------------
Before any new rung is believed, the same invocation must reproduce H13 §6 on the
**existing** ``randstr-d3`` / ``randstr-d5`` validation splits:

    d3 (font 14) pos-0 gain  +28.7 +/- 3.7 pts,  Delta-perm +3.05%
    d5 (font 40) pos-0 gain   +0.0 pts,          Delta-perm +0.01%

Same checkpoint, same rows, same seed, same code => the result should replicate to
within GPU nondeterminism.  The anchors are read from
``data/eval/h13-poscontrol-h07.json`` when it is present (so the gate compares
against the *recorded* run rather than against numbers retyped from a doc) and
fall back to the doc's published values otherwise.  **If the gate fails, the sweep
below means nothing** — the script says so, marks every downstream rung
``gate_failed``, and exits non-zero unless ``--force``.

THE LEG-1 BRANCH IS DECIDED BY POSITION-RESOLVED GAIN, NOT AGGREGATE Delta-perm
-------------------------------------------------------------------------------
(repaired 2026-07-29).  ``leg1_branch`` used to AND a hardcoded ``Delta-perm >=
30%`` conjunct into the trained-font reading bar with no CLI override.  Aggregate
Delta-perm averages over ALL supervised tokens while reading lives in the first
~5-12, so it measures TARGET LENGTH as much as grounding (H13 §4: 14.31% -> 0.63%
for IDENTICAL reading; the largest value ever recorded at these ~60-token targets
is 25.80%).  The 30% bar was therefore unreachable by construction and produced a
spurious ``branch: VOID`` on ``data/eval/h21-leg1.json``.  ``--bar-pos0`` (default
20 pts) now decides; ``--bar-dperm`` defaults to 0 = reporting-only and can be set
to 30 to reproduce the retired criterion.  ``criterion_form`` in the JSON records
which bars were active.

CE IS NEVER A CRITERION.  H13 §7: ``random_strings.py``'s ``floor.json`` divides
the string entropy by *target* tokens while response-only masking also supervises
``<|im_end|>\\n``, inflating every published floor by ~T/(T-2) (~3.4% at these
~60-token targets).  CE and the floor are printed as context and are used by no
branch.

GPU.  This loads a checkpoint.  Do NOT run it while a training job holds the card.

    # leg 0 — gate + sweep, ~0.5 h on a free A100
    HF_HUB_OFFLINE=1 uv run python scratchpad/h21_font_sweep.py \\
        --checkpoint data/checkpoints/hybrid-pretrained-randstr-v0/final \\
        --tag h21-leg0 -n 150

    # leg 1 — same sweep on the font-mixture checkpoint, with the branch table
    HF_HUB_OFFLINE=1 uv run python scratchpad/h21_font_sweep.py \\
        --checkpoint data/checkpoints/h21-fontmix-v0/final \\
        --tag h21-leg1 -n 150 --trained-fonts 14 24 40 --heldout-fonts 18 31

    # no GPU, no model: exercise the gate / pairing / branch logic
    uv run python scratchpad/h21_font_sweep.py --self-test
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h13_analyze import (  # noqa: E402
    Z95,
    Z_MDE,
    _compare_pos,
    _fin,
    _fmt,
    analyse_rung,
    cluster_stats,
    collect_rung,
    make_scorer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("h21_font_sweep")

FONT_ROOT = "data/materialized/h21-fonts-v0"
GATE_ROOT = "data/materialized/h13-density-v0"
GATE_ANCHOR_FILE = "data/eval/h13-poscontrol-h07.json"

#: H13 §6, as published.  Used only when GATE_ANCHOR_FILE is absent.
GATE_ANCHORS_FALLBACK = {
    "randstr-d3": {"pos0_gain_pts": 28.7, "pos0_se_pts": 3.7, "delta_perm_rel": 0.0305},
    "randstr-d5": {"pos0_gain_pts": 0.0, "pos0_se_pts": 0.0, "delta_perm_rel": 0.0001},
}

REFERENCE_FONT = 14  # the font H07 trained on; every paired comparison is vs this


# ---------------------------------------------------------------------------
# paired per-row statistics across two rungs that share targets
# ---------------------------------------------------------------------------
def paired_row_delta(rows_a, rows_b, key_hi, key_lo):
    """Paired mean of ``(hi - lo)`` per row, rung B minus rung A.

    ``key_hi``/``key_lo`` name per-row CE fields in ``rows_info`` — e.g.
    ``("ce_perm_all", "ce_aligned_all")`` gives the per-row Delta-perm in nats.
    Pairing is CHECKED on the stored target text, never assumed; when it does not
    hold the comparison falls back to two independent means with a pooled SE and
    says so.
    """
    import numpy as np

    n = min(len(rows_a), len(rows_b))
    if n < 2:
        return None
    paired = all(rows_a[i]["target_text"] == rows_b[i]["target_text"] for i in range(n))
    va = np.array([r[key_hi] - r[key_lo] for r in rows_a[:n]], dtype=float)
    vb = np.array([r[key_hi] - r[key_lo] for r in rows_b[:n]], dtype=float)
    ok = np.isfinite(va) & np.isfinite(vb)
    va, vb = va[ok], vb[ok]
    if len(va) < 2:
        return None
    if paired:
        d = vb - va
        mean = float(d.mean())
        se = float(d.std(ddof=1) / math.sqrt(len(d)))
        method = "paired-by-row (byte-identical targets, font differs)"
    else:
        mean = float(vb.mean() - va.mean())
        se = float(math.hypot(va.std(ddof=1) / math.sqrt(len(va)),
                              vb.std(ddof=1) / math.sqrt(len(vb))))
        method = "UNPAIRED (targets did not match row-by-row)"
    return {
        "method": method, "n_rows": int(len(va)),
        "mean_a_nats": float(va.mean()), "mean_b_nats": float(vb.mean()),
        "diff_nats": _fin(mean), "se_nats": _fin(se),
        "ci95": [_fin(mean - Z95 * se), _fin(mean + Z95 * se)],
        "mde_80pct_power_nats": _fin(Z_MDE * se),
        "significant": bool(math.isfinite(se) and se > 0 and abs(mean) > Z95 * se),
    }


def overall_reading_gain(arrays):
    """acc(aligned) - acc(blank) over ALL scored content tokens, +/- cluster SE."""
    d = arrays["a_c"] - arrays["b_c"]
    g, se, n = cluster_stats(d, arrays["row"])
    return {"gain_pts": _fin(g * 100), "se_pts": _fin(se * 100), "n_rows": n,
            "ci95": [_fin((g - Z95 * se) * 100), _fin((g + Z95 * se) * 100)],
            "significant": bool(math.isfinite(g) and math.isfinite(se)
                                and g - Z95 * se > 0)}


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def load_gate_anchors(path: str) -> tuple[dict, str]:
    p = Path(path)
    if not p.is_file():
        return GATE_ANCHORS_FALLBACK, f"H13 §6 published values ({path} absent)"
    d = json.loads(p.read_text())
    out = {}
    for rung in ("randstr-d3", "randstr-d5"):
        R = d.get("rungs", {}).get(rung)
        if R is None:
            return GATE_ANCHORS_FALLBACK, f"H13 §6 published values ({rung} missing)"
        out[rung] = {
            "pos0_gain_pts": R["pos0_gain"]["gain_pts"],
            "pos0_se_pts": R["pos0_gain"]["se_pts"],
            "delta_perm_rel": R["delta_perm_rel"],
        }
    return out, f"{path} (checkpoint {d.get('meta', {}).get('checkpoint')})"


def evaluate_gate(measured: dict, anchors: dict, tol_pts: float, tol_dperm_pp: float):
    """PASS only if BOTH anchors replicate, on BOTH statistics."""
    checks, ok = [], True
    for rung in ("randstr-d3", "randstr-d5"):
        if rung not in measured:
            checks.append({"rung": rung, "status": "MISSING", "pass": False})
            ok = False
            continue
        R, A = measured[rung], anchors[rung]
        p0 = R["pos0_gain"]["gain_pts"]
        dp = (R["delta_perm_rel"] or 0.0) * 100
        a_p0 = A["pos0_gain_pts"]
        a_dp = A["delta_perm_rel"] * 100
        p0_ok = p0 is not None and abs(p0 - a_p0) <= tol_pts
        dp_ok = abs(dp - a_dp) <= tol_dperm_pp
        # d3 must still READ (CI excludes 0); d5 must still be at the ignore floor.
        sig = bool(R["pos0_gain"]["significant"])
        sig_ok = sig if rung == "randstr-d3" else (not sig)
        checks.append({
            "rung": rung,
            "pos0_measured_pts": _fin(p0), "pos0_anchor_pts": a_p0,
            "pos0_abs_diff_pts": _fin(None if p0 is None else abs(p0 - a_p0)),
            "pos0_tolerance_pts": tol_pts, "pos0_pass": bool(p0_ok),
            "dperm_measured_pct": _fin(dp), "dperm_anchor_pct": _fin(a_dp),
            "dperm_abs_diff_pp": _fin(abs(dp - a_dp)),
            "dperm_tolerance_pp": tol_dperm_pp, "dperm_pass": bool(dp_ok),
            "pos0_significant": sig,
            "significance_expected": (rung == "randstr-d3"),
            "significance_pass": bool(sig_ok),
            "pass": bool(p0_ok and dp_ok and sig_ok),
        })
        ok = ok and p0_ok and dp_ok and sig_ok
    return {"pass": bool(ok), "checks": checks}


# ---------------------------------------------------------------------------
# the sweep read-out (leg 0) and the pre-registered branch table (leg 1)
# ---------------------------------------------------------------------------
def sweep_profile(results, arrays_by_rung, rows_by_rung, fonts, ref_font):
    """Per-font pos-0 gain / Delta-perm / overall gain, each paired against ref."""
    ref = f"randstr-f{ref_font}"
    prof, vs_ref = {}, {}
    for f in fonts:
        name = f"randstr-f{f}"
        if name not in results:
            continue
        R = results[name]
        prof[f] = {
            "rung": name,
            "pos0_gain_pts": R["pos0_gain"]["gain_pts"],
            "pos0_se_pts": R["pos0_gain"]["se_pts"],
            "pos0_ci95": R["pos0_gain"]["ci95"],
            "pos0_significant": R["pos0_gain"]["significant"],
            "pos0_4_gain_pts": R["pos0_4_gain"]["gain_pts"],
            "overall_gain": overall_reading_gain(arrays_by_rung[name]),
            "delta_perm_pct": _fin((R["delta_perm_rel"] or 0) * 100),
            "delta_blank_pct": _fin((R["delta_blank_rel"] or 0) * 100),
            "frac_rows_aligned_gt_blank": R["row_sanity"]["frac_rows_aligned_gt_blank"],
            "sign_test_p": R["row_sanity"]["sign_test_p"],
            "ce_aligned": R["ce_all_tokens"]["aligned"],
            "target_tokens_mean": R["target_tokens_mean"],
        }
        if name != ref and ref in results:
            vs_ref[f] = {
                "pos0": _compare_pos(ref, name, arrays_by_rung, rows_by_rung, 0, 1),
                "pos0_4": _compare_pos(ref, name, arrays_by_rung, rows_by_rung, 0, 5),
                "delta_perm_nats": paired_row_delta(
                    rows_by_rung[ref], rows_by_rung[name],
                    "ce_perm_all", "ce_aligned_all"),
                "delta_blank_nats": paired_row_delta(
                    rows_by_rung[ref], rows_by_rung[name],
                    "ce_blank_all", "ce_aligned_all"),
            }
    return {"reference_font": ref_font, "per_font": prof, "vs_reference": vs_ref}


def find_cliff(prof):
    """Largest drop in pos-0 gain between ADJACENT fonts, and whether the decay
    looks like a cliff or a gradual slope.  Descriptive only — leg 0 names no
    branch (H21: "measures the shape of the failure ... decides nothing on its own")."""
    fonts = sorted(f for f in prof if prof[f]["pos0_gain_pts"] is not None)
    if len(fonts) < 2:
        return {"status": "underpowered", "n_fonts": len(fonts)}
    g = [prof[f]["pos0_gain_pts"] for f in fonts]
    drops = [(fonts[i], fonts[i + 1], g[i] - g[i + 1]) for i in range(len(fonts) - 1)]
    big = max(drops, key=lambda d: d[2])
    total = max(g) - min(g)
    share = (big[2] / total) if total > 0 else float("nan")
    reading = [f for f in fonts if prof[f]["pos0_significant"]]
    return {
        "status": "ok",
        "fonts": fonts,
        "pos0_gain_pts": [_fin(x) for x in g],
        "fonts_with_significant_pos0_gain": reading,
        "largest_adjacent_drop": {"from_font": big[0], "to_font": big[1],
                                  "drop_pts": _fin(big[2])},
        "largest_drop_share_of_total_range": _fin(share),
        "shape": ("CLIFF" if math.isfinite(share) and share >= 0.6 else
                  "GRADUAL" if math.isfinite(share) else "FLAT"),
        "monotone_decreasing_in_font": all(g[i] >= g[i + 1] for i in range(len(g) - 1)),
    }


def leg1_branch(prof, trained, heldout, bar_pos0=20.0, bar_dperm=0.0,
                floor_pos0=3.0, floor_dperm=1.0, frac_confirm=0.5, frac_partial=0.15):
    """H21's pre-registered branch table, evaluated on POSITION-RESOLVED evidence.

    CONFIRMS: every trained font reads (pos-0 >= bar_pos0) AND every held-out font
              reaches >= frac_confirm of the TRAINED-FONT MEAN pos-0 gain.
    REFUTES : trained fonts read while BOTH held-out fonts sit at the ignore floor
              (pos-0 <= floor_pos0).
    PARTIAL : held-out between frac_partial and frac_confirm of the trained mean.
    VOID    : the trained fonts themselves fail to read.

    WHY AGGREGATE Delta-perm IS NO LONGER LOAD-BEARING (repaired 2026-07-29)
    -----------------------------------------------------------------------
    ``bar_dperm`` used to default to **30.0** and was ANDed into ``tr_read`` with no
    CLI override.  Aggregate Delta-perm averages over ALL supervised tokens, and
    reading lives in the first ~5-12 of them, so the statistic is a function of
    TARGET LENGTH, not of grounding: H13 §4 measured 14.31% -> 0.63% for IDENTICAL
    reading as targets lengthened (+113.84% at 17 sup tok/row, +14.31% at 60,
    +3.05% at 234, +0.63% at 932).  The largest aggregate Delta-perm ever recorded
    at T~60 tokens in this repo is 25.80% — by a model reading at +57.3 pts at
    position 0 — so a 30% bar sits ABOVE the observed ceiling at T>=60 and is
    unreachable BY CONSTRUCTION at T>=234.  It duly fired ``branch: VOID`` on
    ``data/eval/h21-leg1.json``, whose trained fonts read at +57.3 / +52.7 / +64.0
    pts at position 0 against held-out fonts at +0.0 / +0.7 — an ~80x dissociation
    that is exactly the REFUTES pattern.

    So every gate here is now decided by ``pos0_gain_pts`` alone.  Aggregate
    Delta-perm is still computed, emitted and printed as CONTEXT.  Passing
    ``--bar-dperm > 0`` re-arms it as a conjunct (which reproduces the old
    behaviour with ``--bar-dperm 30``); ``criterion_form`` in the JSON records
    which form was used, so a future reader can tell which numbers were gating.
    """
    def g(f, k):
        return (prof.get(f) or {}).get(k)

    # bar_dperm <= 0 => aggregate Delta-perm is reporting-only (the default).
    dperm_load_bearing = bool(bar_dperm and bar_dperm > 0)
    tr = [f for f in trained if f in prof]
    ho = [f for f in heldout if f in prof]
    out = {"trained_fonts": tr, "heldout_fonts": ho, "bars": {
        "trained_pos0_pts": bar_pos0, "trained_delta_perm_pct": bar_dperm,
        "ignore_floor_pos0_pts": floor_pos0, "ignore_floor_delta_perm_pct": floor_dperm,
        "confirm_fraction": frac_confirm, "partial_fraction_low": frac_partial}}
    out["criterion_form"] = {
        "trained_read_gated_on": (["pos0_gain_pts", "delta_perm_pct"]
                                  if dperm_load_bearing else ["pos0_gain_pts"]),
        "heldout_gated_on": (["pos0_gain_pts", "delta_perm_pct"]
                             if dperm_load_bearing else ["pos0_gain_pts"]),
        "aggregate_delta_perm_role": ("LOAD-BEARING (re-armed with --bar-dperm "
                                      f"{bar_dperm})" if dperm_load_bearing
                                      else "reporting-only"),
        "bar_pos0_pts": bar_pos0, "bar_dperm_pct": bar_dperm,
        "note": ("Aggregate Delta-perm is length-diluted (H13 §4: 14.31% -> 0.63% "
                 "for identical reading as targets lengthen; observed ceiling "
                 "25.80% at T~60 tok/row). The default form decides the branch "
                 "from position-resolved evidence (pos-0 gain) only."),
    }
    if not tr or not ho:
        out["branch"] = "VOID"
        out["reason"] = "trained and/or held-out fonts missing from the sweep"
        return out

    tr_read = {f: bool((g(f, "pos0_gain_pts") or 0) >= bar_pos0
                       and (not dperm_load_bearing
                            or (g(f, "delta_perm_pct") or 0) >= bar_dperm))
               for f in tr}
    out["trained_read"] = tr_read
    # reporting-only: whether each trained font would ALSO have cleared a 30%
    # aggregate bar. Kept so the retired criterion stays auditable, never gating.
    out["trained_delta_perm_vs_retired_30pct_bar"] = {
        f: {"delta_perm_pct": _fin(g(f, "delta_perm_pct")),
            "would_pass_30pct": bool((g(f, "delta_perm_pct") or 0) >= 30.0)}
        for f in tr}
    if not all(tr_read.values()):
        out["branch"] = "VOID"
        out["reason"] = (
            "the trained fonts themselves fail the reading bar "
            f"(pos-0 >= {bar_pos0} pts"
            + (f" AND Delta-perm >= {bar_dperm}%" if dperm_load_bearing else "")
            + "): "
            + ", ".join(f"font {f}: pos0 {_fmt(g(f, 'pos0_gain_pts'))} pts, "
                        f"Dperm {_fmt(g(f, 'delta_perm_pct'))}%" for f in tr)
            + ". The run collapsed — check grad_norm and the token-weighted readable "
              "fraction. SAYS NOTHING ABOUT SCALE.")
        return out

    mean_pos0 = sum(g(f, "pos0_gain_pts") for f in tr) / len(tr)
    mean_dperm = sum(g(f, "delta_perm_pct") for f in tr) / len(tr)
    out["trained_mean"] = {"pos0_gain_pts": _fin(mean_pos0),
                           "delta_perm_pct": _fin(mean_dperm)}
    ratios = {}
    for f in ho:
        ratios[f] = {
            "pos0_ratio": _fin((g(f, "pos0_gain_pts") or 0) / mean_pos0)
            if mean_pos0 else None,
            "dperm_ratio": _fin((g(f, "delta_perm_pct") or 0) / mean_dperm)
            if mean_dperm else None,
            "at_ignore_floor": bool(
                (g(f, "pos0_gain_pts") or 0) <= floor_pos0
                and (not dperm_load_bearing
                     or (g(f, "delta_perm_pct") or 0) <= floor_dperm)),
        }
    out["heldout"] = ratios
    # the PARTIAL band is judged on the position-resolved ratio; the Delta-perm
    # ratio is reported beside it (and only folded in when explicitly re-armed).
    worst = min((r["pos0_ratio"] or 0) if not dperm_load_bearing
                else min(r["pos0_ratio"] or 0, r["dperm_ratio"] or 0)
                for r in ratios.values())
    out["worst_heldout_ratio"] = _fin(worst)
    out["worst_heldout_ratio_incl_aggregate_dperm"] = _fin(
        min(min(r["pos0_ratio"] or 0, r["dperm_ratio"] or 0) for r in ratios.values()))

    if all(r["pos0_ratio"] is not None and r["pos0_ratio"] >= frac_confirm
           and (not dperm_load_bearing
                or (r["dperm_ratio"] is not None
                    and r["dperm_ratio"] >= frac_confirm))
           for r in ratios.values()):
        out["branch"] = "CONFIRMS"
        out["reason"] = ("every held-out font reaches >= "
                         f"{frac_confirm:.0%} of the trained-font mean pos-0 gain"
                         + (" and Delta-perm" if dperm_load_bearing else "")
                         + " => scale generalisation is "
                         "learnable by interpolation inside the trained range.")
    elif all(r["at_ignore_floor"] for r in ratios.values()):
        out["branch"] = "REFUTES"
        out["reason"] = ("both held-out fonts sit at the ignore floor "
                         f"(pos-0 <= {floor_pos0} pts"
                         + (f" AND Delta-perm <= {floor_dperm}%"
                            if dperm_load_bearing else "")
                         + ") while the trained fonts read => reading is memorised "
                         "per glyph scale and does not transfer even by "
                         "interpolation.")
    elif frac_partial <= worst < frac_confirm:
        out["branch"] = "PARTIAL"
        out["reason"] = ("held-out performance is between "
                         f"{frac_partial:.0%} and {frac_confirm:.0%} of the trained "
                         "mean. NO BRANCH IS NAMED — report the decay curve against "
                         "|font - nearest trained font| and stop.")
    else:
        out["branch"] = "UNDECIDED"
        out["reason"] = ("the held-out fonts match no pre-registered branch: not at "
                         "the ignore floor, not above the confirm fraction, and not "
                         "inside the PARTIAL band on every statistic. Report the "
                         "numbers; do not force a branch.")
    return out


# ---------------------------------------------------------------------------
def probe(model_bits, root: str, rungs: list[str], n: int, args):
    """Run collect+analyse over a list of rungs under one root."""
    from datasets import load_from_disk

    score, tokchars = model_bits
    results, arrays, rows = {}, {}, {}
    specs = {}
    mpath = Path(root) / "manifest.json"
    if mpath.exists():
        specs = json.loads(mpath.read_text()).get("lane_specs", {})
    for rung in rungs:
        vpath = Path(root) / rung / "validation"
        if not vpath.exists():
            logger.warning("[skip] %s: no validation split at %s", rung, vpath)
            continue
        floor = None
        fpath = Path(root) / rung / "floor.json"
        if fpath.exists():
            floor = json.loads(fpath.read_text()).get("no_reading_floor_nats_per_token")
        ds = load_from_disk(str(vpath))
        ds = ds.shuffle(seed=args.seed).select(range(min(n, len(ds))))
        logger.info("[%s] %d rows", rung, len(ds))
        A, RI, meta = collect_rung(score, tokchars, ds, args.max_length, False)
        results[rung] = analyse_rung(A, RI, meta, floor, specs.get(rung, {}), args)
        arrays[rung], rows[rung] = A, RI
    return results, arrays, rows


def print_report(rep) -> None:
    g = rep["gate"]
    print(f"\n{'=' * 112}")
    print("SANITY GATE — replicate H13 §6 on the EXISTING randstr-d3 / randstr-d5 splits")
    print(f"  anchors from: {rep['gate_anchor_source']}")
    print(f"{'=' * 112}")
    hdr = (f"{'rung':<14}{'pos0 measured':>16}{'pos0 anchor':>14}{'|diff|':>9}"
           f"{'tol':>7}{'Dperm meas%':>13}{'anchor%':>10}{'|diff|pp':>10}{'verdict':>10}")
    print(hdr)
    print("-" * len(hdr))
    for c in g["checks"]:
        print(f"{c['rung']:<14}{_fmt(c.get('pos0_measured_pts'), '+.2f'):>16}"
              f"{_fmt(c.get('pos0_anchor_pts'), '+.2f'):>14}"
              f"{_fmt(c.get('pos0_abs_diff_pts'), '.2f'):>9}"
              f"{_fmt(c.get('pos0_tolerance_pts'), '.1f'):>7}"
              f"{_fmt(c.get('dperm_measured_pct'), '+.3f'):>13}"
              f"{_fmt(c.get('dperm_anchor_pct'), '+.3f'):>10}"
              f"{_fmt(c.get('dperm_abs_diff_pp'), '.3f'):>10}"
              f"{('PASS' if c['pass'] else 'FAIL'):>10}")
    print(f"\n  GATE: {'PASS' if g['pass'] else 'FAIL'}")
    if not g["pass"]:
        print("  !! THE SWEEP BELOW MEANS NOTHING. The probe or the checkpoint has "
              "drifted from the run that produced H13 §6: the same code on the same "
              "rows of the same checkpoint no longer gives the same answer. Fix that "
              "first; do not interpret any font rung.")

    p = rep.get("sweep", {}).get("per_font", {})
    if p:
        print(f"\n{'=' * 112}")
        print("FONT SWEEP — d2 geometry (119 chars), byte-identical targets, paired by row")
        print(f"{'=' * 112}")
        hdr = (f"{'font':>6}{'pos0 gain':>18}{'sig':>5}{'pos0-4':>10}"
               f"{'overall gain':>16}{'Dperm%':>10}{'Dblank%':>10}"
               f"{'rows a>b':>10}{'CE':>9}")
        print(hdr)
        print("-" * len(hdr))
        for f in sorted(p):
            e = p[f]
            og = e["overall_gain"]
            print(f"{f:>6}"
                  f"{_fmt(e['pos0_gain_pts'], '+.2f') + '+/-' + _fmt(e['pos0_se_pts'], '.2f'):>18}"
                  f"{('*' if e['pos0_significant'] else ' '):>5}"
                  f"{_fmt(e['pos0_4_gain_pts'], '+.2f'):>10}"
                  f"{_fmt(og['gain_pts'], '+.2f') + '+/-' + _fmt(og['se_pts'], '.2f'):>16}"
                  f"{_fmt(e['delta_perm_pct'], '+.3f'):>10}"
                  f"{_fmt(e['delta_blank_pct'], '+.3f'):>10}"
                  f"{_fmt(e['frac_rows_aligned_gt_blank'] and e['frac_rows_aligned_gt_blank'] * 100, '.0f') + '%':>10}"
                  f"{_fmt(e['ce_aligned'], '.3f'):>9}")
        print("\npos0 gain: acc(aligned)-acc(blank) at answer token 0, +/- cluster SE; "
              "* = 95% CI excludes 0.")
        print("Dperm/Dblank are RELATIVE CE deltas. CE is context ONLY — no branch "
              "uses it (H13 §7: floor.json is inflated by ~T/(T-2)).")

        vr = rep["sweep"]["vs_reference"]
        if vr:
            print(f"\n  -- paired against font {rep['sweep']['reference_font']} "
                  f"(same rows, same targets)")
            for f in sorted(vr):
                c = vr[f]["pos0"]
                dp = vr[f]["delta_perm_nats"]
                if c:
                    print(f"     font {f:>3}: pos-0 diff {_fmt(c['diff_pts'], '+.2f')} "
                          f"+/- {_fmt(c['se_pts'], '.2f')} pts  95% CI "
                          f"[{_fmt(c['ci95'][0], '+.2f')}, {_fmt(c['ci95'][1], '+.2f')}]"
                          f"  MDE(80%) {_fmt(c['mde_80pct_power_pts'], '.2f')} pts"
                          f"  [{c['method']}]")
                if dp:
                    print(f"              Delta-perm diff {_fmt(dp['diff_nats'], '+.4f')} "
                          f"+/- {_fmt(dp['se_nats'], '.4f')} nats  95% CI "
                          f"[{_fmt(dp['ci95'][0], '+.4f')}, {_fmt(dp['ci95'][1], '+.4f')}]"
                          f"  MDE {_fmt(dp['mde_80pct_power_nats'], '.4f')} nats")

    cl = rep.get("cliff", {})
    if cl.get("status") == "ok":
        print(f"\n  -- SHAPE OF THE FAILURE (descriptive; leg 0 names no branch)")
        print(f"     fonts reading at pos-0: {cl['fonts_with_significant_pos0_gain']}")
        d = cl["largest_adjacent_drop"]
        print(f"     largest adjacent drop: font {d['from_font']} -> {d['to_font']}, "
              f"{_fmt(d['drop_pts'], '.2f')} pts "
              f"({_fmt(cl['largest_drop_share_of_total_range'] and cl['largest_drop_share_of_total_range'] * 100, '.0f')}% "
              f"of the whole range) => {cl['shape']}")
        print(f"     monotone decreasing in font size: {cl['monotone_decreasing_in_font']}")

    b = rep.get("branch")
    if b:
        print(f"\n{'=' * 112}")
        print("H21 PRE-REGISTERED BRANCH (leg 1)")
        print(f"{'=' * 112}")
        print(f"  trained {b['trained_fonts']}  held-out {b['heldout_fonts']}")
        cf = b.get("criterion_form") or {}
        if cf:
            print(f"  criterion form: gated on {cf.get('trained_read_gated_on')} "
                  f"(bar pos-0 {cf.get('bar_pos0_pts')} pts); aggregate Delta-perm is "
                  f"{cf.get('aggregate_delta_perm_role')}")
        for f, r in (b.get("trained_delta_perm_vs_retired_30pct_bar") or {}).items():
            print(f"    [context] trained font {f}: Delta-perm "
                  f"{_fmt(r['delta_perm_pct'], '+.2f')}% "
                  f"(retired 30% bar: {'pass' if r['would_pass_30pct'] else 'FAIL'} "
                  f"— length-diluted, decides nothing)")
        if "trained_mean" in b:
            print(f"  trained-font mean: pos-0 "
                  f"{_fmt(b['trained_mean']['pos0_gain_pts'], '+.2f')} pts, "
                  f"Delta-perm {_fmt(b['trained_mean']['delta_perm_pct'], '+.2f')}%")
        for f, r in (b.get("heldout") or {}).items():
            print(f"    font {f}: pos-0 ratio {_fmt(r['pos0_ratio'], '.3f')}, "
                  f"Delta-perm ratio {_fmt(r['dperm_ratio'], '.3f')}, "
                  f"at ignore floor: {r['at_ignore_floor']}")
        print(f"\n  >>> BRANCH: {b['branch']} — {b['reason']}")

    print("\nWHAT NO NUMBER HERE LICENSES:")
    for i, c in enumerate(rep["caveats"], 1):
        print(f"  {i}. {c}")


CAVEATS = [
    "Font is not a single-factor dial (H13 §5.3): changing it changes "
    "chars-per-inked-token AND stroke width, lines per cell, grid rows occupied and "
    "the fraction of the token grid in use. A null does not say WHICH component "
    "failed to transfer.",
    "Nothing here touches the soft-token budget — it is pinned at whatever the "
    "checkpoint records (280 for every pre-H17 checkpoint). Capacity vs scan is H17.",
    "randstr is prior-proof BY CONSTRUCTION; fineweb-edu at ~2,832 chars/page cannot "
    "change font without also changing the page count, so no branch transfers to the "
    "real text lanes without a separate test.",
    "Fonts below 14 do NOT exist: render_utils clamps `font_size = max(font_size, 14)`, "
    "so a 'font 10' rung is byte-identical to font 14. H21's design as written asked "
    "for one; the materializer refuses to build it.",
    "The gate is a REPLICATION check, not a calibration: it says the probe and the "
    "checkpoint still behave as they did for H13 §6. It cannot detect a defect that "
    "was already present in H13 §6.",
    "Every CI here is a row-level cluster interval; tokens inside one page share a "
    "model state and an image and are not independent.",
]


# ---------------------------------------------------------------------------
def self_test() -> None:
    import numpy as np

    # --- gate: replication passes, drift fails ------------------------------
    anchors = {"randstr-d3": {"pos0_gain_pts": 28.7, "pos0_se_pts": 3.7,
                              "delta_perm_rel": 0.0305},
               "randstr-d5": {"pos0_gain_pts": 0.0, "pos0_se_pts": 0.0,
                              "delta_perm_rel": 0.0001}}

    def meas(p3, dp3, sig3, p5, dp5, sig5):
        return {
            "randstr-d3": {"pos0_gain": {"gain_pts": p3, "significant": sig3},
                           "delta_perm_rel": dp3},
            "randstr-d5": {"pos0_gain": {"gain_pts": p5, "significant": sig5},
                           "delta_perm_rel": dp5},
        }

    g = evaluate_gate(meas(28.7, 0.0305, True, 0.0, 0.0001, False), anchors, 2.0, 0.5)
    assert g["pass"], g
    print("  ok  gate PASSES on an exact replication of H13 §6")

    g = evaluate_gate(meas(29.9, 0.0290, True, 0.7, 0.0009, False), anchors, 2.0, 0.5)
    assert g["pass"], g
    print("  ok  gate tolerates GPU-nondeterminism-sized wobble (+/-2 pts, +/-0.5 pp)")

    g = evaluate_gate(meas(4.0, 0.0020, False, 0.0, 0.0001, False), anchors, 2.0, 0.5)
    assert not g["pass"] and not g["checks"][0]["pass"]
    print("  ok  gate FAILS when d3 stops reading (the checkpoint/probe drifted)")

    g = evaluate_gate(meas(28.7, 0.0305, True, 25.0, 0.0300, True), anchors, 2.0, 0.5)
    assert not g["pass"] and not g["checks"][1]["pass"]
    print("  ok  gate FAILS when d5 suddenly reads (H13 §6 would not replicate)")

    # --- paired row delta ---------------------------------------------------
    ra = [{"target_text": "a", "ce_perm_all": 5.2, "ce_aligned_all": 5.0},
          {"target_text": "b", "ce_perm_all": 5.4, "ce_aligned_all": 5.0}]
    rb = [{"target_text": "a", "ce_perm_all": 5.0, "ce_aligned_all": 5.0},
          {"target_text": "b", "ce_perm_all": 5.0, "ce_aligned_all": 5.0}]
    d = paired_row_delta(ra, rb, "ce_perm_all", "ce_aligned_all")
    assert d["method"].startswith("paired") and abs(d["diff_nats"] + 0.3) < 1e-9, d
    rb[0]["target_text"] = "zzz"
    d = paired_row_delta(ra, rb, "ce_perm_all", "ce_aligned_all")
    assert d["method"].startswith("UNPAIRED"), d
    print("  ok  paired Delta-perm pairs rows only when targets match, else UNPAIRED")

    # --- cliff detection ----------------------------------------------------
    prof = {14: {"pos0_gain_pts": 30.0, "pos0_significant": True},
            20: {"pos0_gain_pts": 28.0, "pos0_significant": True},
            28: {"pos0_gain_pts": 26.0, "pos0_significant": True},
            40: {"pos0_gain_pts": 0.5, "pos0_significant": False}}
    c = find_cliff(prof)
    assert c["shape"] == "CLIFF" and c["largest_adjacent_drop"]["to_font"] == 40, c
    print("  ok  cliff detected between the last two fonts (85% of the range)")
    prof = {14: {"pos0_gain_pts": 30.0, "pos0_significant": True},
            20: {"pos0_gain_pts": 22.0, "pos0_significant": True},
            28: {"pos0_gain_pts": 14.0, "pos0_significant": True},
            40: {"pos0_gain_pts": 6.0, "pos0_significant": True}}
    c = find_cliff(prof)
    assert c["shape"] == "GRADUAL" and c["monotone_decreasing_in_font"], c
    print("  ok  an even slope is reported GRADUAL, not a cliff")

    # --- the pre-registered branch table ------------------------------------
    def mk(p0, dp):
        return {"pos0_gain_pts": p0, "delta_perm_pct": dp}

    prof = {14: mk(40, 60), 24: mk(35, 50), 40: mk(30, 40), 18: mk(30, 40),
            31: mk(25, 35)}
    b = leg1_branch(prof, [14, 24, 40], [18, 31])
    assert b["branch"] == "CONFIRMS", b
    print("  ok  branch CONFIRMS when both held-out fonts clear 50% of the trained mean")

    prof = {14: mk(40, 60), 24: mk(35, 50), 40: mk(30, 40), 18: mk(1.0, 0.4),
            31: mk(0.5, 0.2)}
    b = leg1_branch(prof, [14, 24, 40], [18, 31])
    assert b["branch"] == "REFUTES", b
    print("  ok  branch REFUTES when both held-out fonts sit at the ignore floor")

    prof = {14: mk(40, 60), 24: mk(35, 50), 40: mk(30, 40), 18: mk(9, 14),
            31: mk(8, 12)}
    b = leg1_branch(prof, [14, 24, 40], [18, 31])
    assert b["branch"] == "PARTIAL", b
    print("  ok  branch PARTIAL between 15% and 50% of the trained mean (names nothing)")

    prof = {14: mk(5, 2), 24: mk(4, 1), 40: mk(2, 0.5), 18: mk(1, 0.2), 31: mk(1, 0.2)}
    b = leg1_branch(prof, [14, 24, 40], [18, 31])
    assert b["branch"] == "VOID" and "SAYS NOTHING ABOUT SCALE" in b["reason"], b
    print("  ok  branch VOID when the TRAINED fonts fail to read (says nothing "
          "about scale)")

    # REGRESSION (2026-07-29): the real data/eval/h21-leg1.json profile. The
    # retired 30% aggregate-Delta-perm conjunct called this VOID even though the
    # trained fonts read at +57/+53/+64 pts at position 0 and the held-out fonts
    # sit at +0.0/+0.7 — an ~80x dissociation, i.e. textbook REFUTES.
    real = {14: mk(57.33, 25.800), 24: mk(52.67, 8.854), 40: mk(64.00, 11.148),
            18: mk(0.00, 0.012), 31: mk(0.67, 0.086)}
    b = leg1_branch(real, [14, 24, 40], [18, 31])
    assert b["branch"] == "REFUTES", b
    assert b["criterion_form"]["trained_read_gated_on"] == ["pos0_gain_pts"], b
    assert all(b["trained_read"].values()), b
    print("  ok  REGRESSION: h21-leg1's real profile -> REFUTES (was a spurious VOID "
          "from the length-diluted 30% aggregate-Delta-perm bar)")

    b30 = leg1_branch(real, [14, 24, 40], [18, 31], bar_dperm=30.0)
    assert b30["branch"] == "VOID", b30
    assert b30["criterion_form"]["aggregate_delta_perm_role"].startswith("LOAD-BEARING")
    print("  ok  --bar-dperm 30 still reproduces the retired criterion (and its VOID), "
          "and criterion_form records that it was armed")

    b0 = leg1_branch(real, [14, 24, 40], [18, 31], bar_pos0=70.0)
    assert b0["branch"] == "VOID", b0
    print("  ok  --bar-pos0 is honoured (a 70-pt bar voids the same profile)")

    # --- overall reading gain uses the cluster (per-row) SE -----------------
    A = {"a_c": np.array([1.0, 1, 0, 0]), "b_c": np.zeros(4),
         "row": np.array([0, 0, 1, 1])}
    og = overall_reading_gain(A)
    assert abs(og["gain_pts"] - 50.0) < 1e-9 and abs(og["se_pts"] - 50.0) < 1e-9, og
    print("  ok  overall reading gain uses the per-ROW cluster SE (n_rows, not tokens)")

    print("\nSELF-TEST PASSED (gate, pairing, cliff, branch table, cluster SE)")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint")
    ap.add_argument("--fonts", type=int, nargs="*",
                    default=[14, 18, 20, 24, 28, 31, 40, 52])
    ap.add_argument("--reference-font", type=int, default=REFERENCE_FONT)
    ap.add_argument("--trained-fonts", type=int, nargs="*", default=None)
    ap.add_argument("--heldout-fonts", type=int, nargs="*", default=None)
    ap.add_argument("--bar-pos0", type=float, default=20.0,
                    help="reading bar on the POSITION-RESOLVED statistic (pos-0 gain, "
                         "pts). This is what decides the leg-1 branch.")
    ap.add_argument("--bar-dperm", type=float, default=0.0,
                    help="aggregate-Delta-perm conjunct, in %%. DEFAULT 0 = "
                         "REPORTING-ONLY: aggregate Delta-perm is length-diluted "
                         "(H13 §4: 14.31%% -> 0.63%% for identical reading; observed "
                         "ceiling 25.80%% at ~60 sup tok/row), so the old hardcoded "
                         "30%% bar was unreachable by construction and voided a run "
                         "reading at +57 pts at position 0. Pass 30 to reproduce that "
                         "retired criterion.")
    ap.add_argument("--floor-pos0", type=float, default=3.0,
                    help="ignore-floor bar on pos-0 gain (pts) for the held-out fonts")
    ap.add_argument("--floor-dperm", type=float, default=1.0,
                    help="ignore-floor bar on aggregate Delta-perm (%%); only "
                         "consulted when --bar-dperm > 0 re-arms the aggregate")
    ap.add_argument("-n", type=int, default=150, help="validation rows per rung")
    ap.add_argument("--gate-n", type=int, default=150,
                    help="rows for the d3/d5 replication gate (H13 used 150)")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--font-root", default=FONT_ROOT)
    ap.add_argument("--gate-root", default=GATE_ROOT)
    ap.add_argument("--gate-anchors", default=GATE_ANCHOR_FILE)
    ap.add_argument("--gate-tol-pts", type=float, default=2.0)
    ap.add_argument("--gate-tol-dperm-pp", type=float, default=0.5)
    ap.add_argument("--skip-gate", action="store_true",
                    help="NOT recommended; the gate is what makes the sweep believable")
    ap.add_argument("--force", action="store_true",
                    help="continue (and exit 0) even if the gate fails")
    ap.add_argument("--max-soft-tokens", type=int, default=None)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--char-bin-width", type=int, default=25)
    ap.add_argument("--baseline-chars", type=int, default=100)
    ap.add_argument("--min-bin-tokens", type=int, default=50)
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--logit-chunk", type=int, default=256)
    ap.add_argument("--tag", default="h21-font-sweep")
    ap.add_argument("--output", default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        print("SELF-TEST (no model, no GPU)")
        self_test()
        return 0
    if not a.checkpoint:
        ap.error("--checkpoint is required (or use --self-test)")

    import torch
    from transformers import AutoTokenizer

    from h13_analyze import TokenChars
    from univi.hybrid.data import HybridCollator
    from univi.hybrid.pretrained import (
        UniViHybridPretrained, build_image_processor, resolve_max_soft_tokens,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading %s on %s", a.checkpoint, device)
    model = UniViHybridPretrained.from_pretrained(a.checkpoint, dtype=torch.bfloat16).to(device)
    model.eval()
    model.config.use_cache = False
    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    mst = resolve_max_soft_tokens(model, a.max_soft_tokens)
    logger.info("Soft-token budget: %d (checkpoint records %s)",
                mst, getattr(model.config, "max_soft_tokens", "<absent -> 280>"))
    col = HybridCollator(tokenizer=tok, image_processor=build_image_processor(mst),
                         image_token_id=model.image_token_id,
                         max_length=a.max_length, response_only=True)
    bits = (make_scorer(model, col, device, a.logit_chunk), TokenChars(tok))

    anchors, src = load_gate_anchors(a.gate_anchors)
    if a.skip_gate:
        gate = {"pass": None, "checks": [], "skipped": True}
    else:
        logger.info("GATE: replicating H13 §6 on randstr-d3 / randstr-d5")
        gres, _, _ = probe(bits, a.gate_root, ["randstr-d3", "randstr-d5"], a.gate_n, a)
        gate = evaluate_gate(gres, anchors, a.gate_tol_pts, a.gate_tol_dperm_pp)
        gate["measured"] = {k: {"pos0_gain_pts": v["pos0_gain"]["gain_pts"],
                                "pos0_se_pts": v["pos0_gain"]["se_pts"],
                                "delta_perm_rel": v["delta_perm_rel"],
                                "delta_blank_rel": v["delta_blank_rel"]}
                            for k, v in gres.items()}

    rungs = [f"randstr-f{f}" for f in a.fonts]
    results, arrays, rows = probe(bits, a.font_root, rungs, a.n, a)
    sweep = sweep_profile(results, arrays, rows, a.fonts, a.reference_font)
    rep = {
        "meta": {"checkpoint": a.checkpoint, "tag": a.tag, "n_per_rung": a.n,
                 "gate_n": a.gate_n, "seed": a.seed, "max_soft_tokens": mst,
                 "max_length": a.max_length, "font_root": a.font_root,
                 "gate_root": a.gate_root, "fonts": a.fonts,
                 "reference_font": a.reference_font,
                 "bar_pos0_pts": a.bar_pos0, "bar_dperm_pct": a.bar_dperm,
                 "floor_pos0_pts": a.floor_pos0, "floor_dperm_pct": a.floor_dperm},
        "gate": gate, "gate_anchor_source": src, "gate_anchors": anchors,
        "sweep": sweep, "cliff": find_cliff(sweep["per_font"]),
        "rungs": results, "caveats": CAVEATS,
    }
    if a.trained_fonts and a.heldout_fonts:
        rep["branch"] = leg1_branch(sweep["per_font"], a.trained_fonts, a.heldout_fonts,
                                    bar_pos0=a.bar_pos0, bar_dperm=a.bar_dperm,
                                    floor_pos0=a.floor_pos0, floor_dperm=a.floor_dperm)
    if gate.get("pass") is False:
        rep["gate_failed"] = True
        for f in rep["sweep"]["per_font"]:
            rep["sweep"]["per_font"][f]["gate_failed"] = True

    print_report(rep)
    out = Path(a.output or f"data/eval/{a.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    logger.info("Wrote %s", out)
    if gate.get("pass") is False and not a.force:
        logger.error("GATE FAILED — exiting non-zero. Nothing above is interpretable.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
