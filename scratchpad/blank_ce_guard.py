"""Extract the pre-registered BLIND-BRANCH guards out of finished run logs.

`log_blank_ce: true` makes `univi/hybrid/train_pretrained.py` inject four metrics
per lane at every eval:

    eval_<lane>_aligned_ce   eval_<lane>_blank_ce
    eval_<lane>_prior_gap_ce eval_<lane>_blank_ce_inflation

Two DIFFERENT pre-registrations read them, and neither is a grounding metric:

1. **Degeneracy guard (H16, H20).** `blank_ce_inflation > +0.10` ⇒ the run is being
   made *worse blind* rather than *better sighted* and the result is **VOID**
   (`docs/hypothesis/todo/H16-prior-gap-weighted-loss.md`, "Guard (degeneracy
   check)"; `docs/hypothesis/todo/H20-audio-phoneme-resolution.md`,
   "Blind-branch guard"). Inflation is relative to the FIRST eval of the run.
2. **Collapse alarm (H15).** `prior_gap_ce < 0.05` nats for three consecutive
   evals ⇒ the arm is dead and should have been killed
   (`docs/hypothesis/todo/H15-prior-poisoned-text.md`, "Live collapse alarm").
   **That 0.05 is LENGTH-DEPENDENT** — `prior_gap_ce` is a per-token mean while
   reading lives in the first ~5-12 answer tokens, so the same reading yields a ~4x
   smaller number on H15's 181.7-tok/row arm than on its 44.3-tok/row arm and the
   long arm trips first. The bar is therefore configurable per lane
   (`--prior-gap-dead-below-lane poisoned-text-long=0.012`) and the report prints a
   length-scaled suggestion. Guard 1 is a within-run same-lane RATIO and is immune.

`prior_gap_ce` is a Δblank quantity. H13 §1 found a fully dead run (Δperm = 0)
still carrying Δblank of +1.0%…+8.7% from ink-presence alone, so a PASS here is
**not** evidence of reading — it only means the pre-registered VOID condition did
not fire. Grounding still has to come from Δperm / position-resolved gain.

CPU only: this reads text files. No GPU, no model, no dataset.

    uv run python scratchpad/blank_ce_guard.py \
        --logs data/checkpoints/h15-poisoned-p00-v0-run.log ... \
        --prior-gap-dead-below-lane poisoned-text-long=0.012 \
        --out data/eval/stage3-blank-ce-guard.json
    uv run python scratchpad/blank_ce_guard.py --self-test
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

#: Longest metric name first — `blank_ce` is a prefix of `blank_ce_inflation`.
METRICS = ("blank_ce_inflation", "prior_gap_ce", "aligned_ce", "blank_ce")

#: Emitted as `eval_<lane>_<metric>` on a multi-lane run and as plain
#: `eval_<metric>` on a single-lane one (H16 is fineweb-only and logs the latter),
#: so the lane group is OPTIONAL — requiring it silently found nothing in the H16
#: log. Lane names in this repo never contain "_" (fineweb-edu,
#: poisoned-text-long, spoken-digits, librispeech), so excluding it from the lane
#: class keeps the metric suffix unambiguous.
SINGLE_LANE = "<single-lane>"
PATTERN = re.compile(
    r"eval_(?:(?P<lane>[A-Za-z0-9.\-]+)_)?(?P<metric>"
    + "|".join(METRICS)
    + r")'?\s*[:=]\s*'?(?P<value>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)

#: H16 / H20. A WITHIN-RUN, SAME-LANE RATIO against the run's first eval, so it is
#: dimensionless and length-independent — this one is fine as an absolute constant.
INFLATION_VOID_AT = 0.10

#: H15. An ABSOLUTE nats threshold on `prior_gap_ce`, which is a per-token MEAN over
#: all supervised tokens — so it is LENGTH-DEPENDENT and NOT comparable across arms
#: of different target length. Reading lives in the first ~5-12 answer tokens, so an
#: arm with T tokens/row dilutes the same reading by ~1/T: H15's arms sit at 44.3 and
#: 181.7 supervised tok/row, a ~4.1x ratio, so the SAME reading yields a ~4x smaller
#: prior_gap_ce on the long arm and the long arm trips a fixed 0.05-nat bar first.
#: Configure it per lane (`--prior-gap-dead-below LANE=VALUE`) — a defensible scaling
#: is bar_lane = 0.05 * (44.3 / T_lane), i.e. ~0.012 at 181.7 tok/row — or read the
#: alarm as length-confounded.
PRIOR_GAP_DEAD_BELOW = 0.05
#: Observed supervised tokens/row per lane, for the scaling above. The reference is
#: the arm H15's 0.05 was written against.
PRIOR_GAP_REFERENCE_TOKENS_PER_ROW = 44.3
LANE_TOKENS_PER_ROW = {"poisoned-text": 44.3, "poisoned-text-long": 181.7}
PRIOR_GAP_DEAD_RUN = 3         # consecutive evals


def length_scaled_prior_gap_bar(tokens_per_row: float,
                                bar: float = PRIOR_GAP_DEAD_BELOW,
                                ref: float = PRIOR_GAP_REFERENCE_TOKENS_PER_ROW
                                ) -> float:
    """H15's 0.05-nat bar rescaled for a lane of ``tokens_per_row``.

    `prior_gap_ce` is a per-token mean and reading is front-loaded, so the same
    reading shrinks the mean roughly in proportion to 1/T. Comparing two arms
    against ONE absolute bar therefore compares their target lengths.
    """
    if not tokens_per_row or tokens_per_row <= 0:
        return bar
    return bar * (ref / float(tokens_per_row))


def parse_text(text: str) -> dict:
    """{lane: {metric: [values in file order]}} — order is eval order."""
    out: dict[str, dict[str, list[float]]] = {}
    for m in PATTERN.finditer(text):
        lane = m.group("lane") or SINGLE_LANE
        out.setdefault(lane, {}).setdefault(m.group("metric"), []).append(
            float(m.group("value"))
        )
    return out


def judge(series: dict, dead_below: float = PRIOR_GAP_DEAD_BELOW,
          tokens_per_row: float | None = None) -> dict:
    """Apply both pre-registered rules to one lane's series.

    ``dead_below`` is per-lane on purpose: it is an ABSOLUTE nats bar on a per-token
    mean, so it is length-dependent (see ``PRIOR_GAP_DEAD_BELOW``).
    """
    infl = series.get("blank_ce_inflation") or []
    gap = series.get("prior_gap_ce") or []
    tail = gap[-PRIOR_GAP_DEAD_RUN:]
    dead = len(tail) == PRIOR_GAP_DEAD_RUN and all(
        g < dead_below for g in tail
    )
    return {
        "n_evals": max(len(v) for v in series.values()) if series else 0,
        "blank_ce_inflation": {
            "max": max(infl) if infl else None,
            "last": infl[-1] if infl else None,
            "void_threshold": INFLATION_VOID_AT,
            # The doc's wording is "must not inflate by more than +10%" at ANY
            # eval, so the max is what decides, not the final value.
            "verdict": (
                "NO-DATA" if not infl
                else "VOID" if max(infl) > INFLATION_VOID_AT
                else "PASS"
            ),
        },
        "prior_gap_ce": {
            "min": min(gap) if gap else None,
            "last": gap[-1] if gap else None,
            "last_3": tail,
            "dead_threshold": dead_below,
            "dead_threshold_default": PRIOR_GAP_DEAD_BELOW,
            "tokens_per_row": tokens_per_row,
            "length_scaled_threshold_suggestion": (
                length_scaled_prior_gap_bar(tokens_per_row)
                if tokens_per_row else None),
            "length_dependence_warning": (
                "prior_gap_ce is a per-token MEAN over all supervised tokens while "
                "reading is concentrated in the first ~5-12 answer tokens, so this "
                "absolute nats bar is length-dependent: the same reading yields a "
                "~4x smaller value at 181.7 tok/row than at 44.3. A single bar "
                "across arms of different target length compares LENGTHS, not "
                "collapse. Set --prior-gap-dead-below LANE=VALUE per lane."),
            "verdict": (
                "NO-DATA" if not gap
                else "COLLAPSE-ALARM" if dead
                else "PASS"
            ),
        },
        "aligned_ce_last": (series.get("aligned_ce") or [None])[-1],
        "blank_ce_last": (series.get("blank_ce") or [None])[-1],
    }


def run(logs, out_path, prior_gap_bars: dict | None = None,
        default_prior_gap_bar: float = PRIOR_GAP_DEAD_BELOW,
        lane_tokens: dict | None = None) -> dict:
    prior_gap_bars = prior_gap_bars or {}
    lane_tokens = dict(LANE_TOKENS_PER_ROW, **(lane_tokens or {}))
    report = {
        "note": (
            "Blind-branch guards only. blank_ce_inflation > +0.10 = VOID (H16/H20); "
            "prior_gap_ce < the lane's bar for 3 consecutive evals = collapse alarm "
            "(H15). prior_gap_ce IS A Δblank QUANTITY AND IS NOT A GROUNDING METRIC "
            "— H13 §1 saw +1.0%…+8.7% Δblank in a run with Δperm = 0. AND IT IS "
            "LENGTH-DILUTED: it is a per-token mean while reading lives in the first "
            "~5-12 answer tokens, so one absolute nats bar applied to arms of "
            "different target length (H15: 44.3 vs 181.7 tok/row, ~4x) fires on the "
            "LONG arm first for the same reading. blank_ce_inflation is a within-run "
            "same-lane RATIO and is not affected."
        ),
        "thresholds": {
            "blank_ce_inflation_void_above": INFLATION_VOID_AT,
            "prior_gap_ce_dead_below_default": default_prior_gap_bar,
            "prior_gap_ce_dead_below_per_lane": prior_gap_bars,
            "prior_gap_ce_consecutive_evals": PRIOR_GAP_DEAD_RUN,
            "prior_gap_ce_is_length_dependent": True,
            "lane_tokens_per_row": lane_tokens,
            "length_scaled_bar_suggestion": {
                k: round(length_scaled_prior_gap_bar(v, default_prior_gap_bar), 5)
                for k, v in lane_tokens.items()},
        },
        "runs": {},
    }
    for log in logs:
        p = Path(log)
        if not p.exists():
            report["runs"][log] = {"status": "MISSING"}
            print(f"[skip] {log}: no such file")
            continue
        lanes = parse_text(p.read_text(errors="replace"))
        if not lanes:
            report["runs"][log] = {
                "status": "NO-BLANK-CE-METRICS",
                "note": "log_blank_ce was off for this run, or it logs under other keys.",
            }
            print(f"[none] {log}: no blank-CE metrics present")
            continue
        report["runs"][log] = {
            "status": "ok",
            "lanes": {lane: judge(s,
                                  prior_gap_bars.get(lane, default_prior_gap_bar),
                                  lane_tokens.get(lane))
                      for lane, s in lanes.items()},
        }
        for lane, j in report["runs"][log]["lanes"].items():
            print(
                f"  {p.name:<44} {lane:<20} "
                f"inflation max {j['blank_ce_inflation']['max']} "
                f"[{j['blank_ce_inflation']['verdict']}]  "
                f"prior_gap min {j['prior_gap_ce']['min']} "
                f"(bar {j['prior_gap_ce']['dead_threshold']:g}"
                + (f", length-scaled suggestion "
                   f"{j['prior_gap_ce']['length_scaled_threshold_suggestion']:.4f} at "
                   f"{j['prior_gap_ce']['tokens_per_row']:g} tok/row"
                   if j["prior_gap_ce"]["length_scaled_threshold_suggestion"] else "")
                + f") [{j['prior_gap_ce']['verdict']}]"
            )
    if out_path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"Wrote {out}")
    return report


def self_test() -> None:
    txt = (
        "{'eval_poisoned-text_aligned_ce': '1.2000', "
        "'eval_poisoned-text_blank_ce': '2.2000', "
        "'eval_poisoned-text_prior_gap_ce': '1.0000', "
        "'eval_poisoned-text_blank_ce_inflation': '0.07017'}\n"
        "{'eval_poisoned-text-long_prior_gap_ce': '0.0400', "
        "'eval_poisoned-text-long_blank_ce_inflation': '0.20000'}\n"
        "{'eval_poisoned-text-long_prior_gap_ce': '0.0300', "
        "'eval_poisoned-text-long_blank_ce_inflation': '0.10000'}\n"
        "{'eval_poisoned-text-long_prior_gap_ce': '0.0200', "
        "'eval_poisoned-text-long_blank_ce_inflation': '0.05000'}\n"
    )
    lanes = parse_text(txt)
    assert set(lanes) == {"poisoned-text", "poisoned-text-long"}, lanes
    # `blank_ce` must NOT swallow `blank_ce_inflation`.
    assert lanes["poisoned-text"]["blank_ce"] == [2.2], lanes["poisoned-text"]
    assert lanes["poisoned-text"]["blank_ce_inflation"] == [0.07017]
    print("  ok  metric names disambiguated (blank_ce vs blank_ce_inflation)")

    j = judge(lanes["poisoned-text"])
    assert j["blank_ce_inflation"]["verdict"] == "PASS", j
    assert j["prior_gap_ce"]["verdict"] == "PASS", j
    print("  ok  a healthy lane passes both guards")

    j = judge(lanes["poisoned-text-long"])
    # max inflation 0.20 > 0.10 even though the LAST eval is 0.05.
    assert j["blank_ce_inflation"]["verdict"] == "VOID", j
    assert j["blank_ce_inflation"]["max"] == 0.2
    print("  ok  VOID fires on the MAX inflation, not the last value")
    assert j["prior_gap_ce"]["verdict"] == "COLLAPSE-ALARM", j
    print("  ok  collapse alarm fires on 3 consecutive prior_gap_ce < 0.05")

    j = judge({"prior_gap_ce": [0.04, 0.06, 0.04]})
    assert j["prior_gap_ce"]["verdict"] == "PASS", j
    print("  ok  a non-consecutive dip does not fire the alarm")

    # the collapse bar is per-lane, because it is an ABSOLUTE nats bar on a
    # length-diluted per-token mean: the long arm's 0.02 is the short arm's ~0.08
    long_series = {"prior_gap_ce": [0.02, 0.02, 0.02]}
    assert judge(long_series)["prior_gap_ce"]["verdict"] == "COLLAPSE-ALARM"
    scaled = length_scaled_prior_gap_bar(181.7)
    assert abs(scaled - 0.05 * 44.3 / 181.7) < 1e-12
    assert scaled < 0.02, scaled
    j = judge(long_series, scaled, 181.7)
    assert j["prior_gap_ce"]["verdict"] == "PASS", j
    assert j["prior_gap_ce"]["dead_threshold"] == scaled
    print(f"  ok  the collapse bar is configurable per lane: 0.02 nats at 181.7 "
          f"tok/row fires against the fixed 0.05 bar but PASSES the length-scaled "
          f"{scaled:.4f} (the same reading as 0.08 at 44.3 tok/row)")
    assert _parse_kv_floats(["a=0.01", "b=2"], "--x") == {"a": 0.01, "b": 2.0}
    print("  ok  LANE=VALUE overrides parse")

    assert judge({})["blank_ce_inflation"]["verdict"] == "NO-DATA"
    print("  ok  absent metrics report NO-DATA rather than a number")

    # A single-lane run (H16 is fineweb-only) logs the keys with NO lane infix.
    single = parse_text(
        "{'eval_aligned_ce': '2.0', 'eval_blank_ce': '2.5', "
        "'eval_prior_gap_ce': '0.5', 'eval_blank_ce_inflation': '-0.01228'}"
    )
    assert list(single) == [SINGLE_LANE], single
    assert single[SINGLE_LANE]["blank_ce_inflation"] == [-0.01228], single
    assert single[SINGLE_LANE]["blank_ce"] == [2.5], single
    print("  ok  un-laned `eval_blank_ce_inflation` (single-lane run) is picked up")
    print("\nSELF-TEST PASSED")


def _parse_kv_floats(items, flag) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"{flag} expects LANE=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = float(v)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logs", nargs="*", default=[])
    ap.add_argument("--out", default="data/eval/stage3-blank-ce-guard.json")
    ap.add_argument("--prior-gap-dead-below", type=float,
                    default=PRIOR_GAP_DEAD_BELOW,
                    help="default collapse-alarm bar in NATS. It is an absolute bar "
                         "on a per-token mean and therefore LENGTH-DEPENDENT — the "
                         "same reading gives a ~4x smaller value at 181.7 tok/row "
                         "than at 44.3. Prefer per-lane bars.")
    ap.add_argument("--prior-gap-dead-below-lane", nargs="*", default=[],
                    metavar="LANE=NATS",
                    help="per-lane override, e.g. poisoned-text-long=0.012 "
                         "(= 0.05 * 44.3/181.7).")
    ap.add_argument("--lane-tokens-per-row", nargs="*", default=[],
                    metavar="LANE=TOKENS",
                    help="supervised tokens/row per lane; used only to print the "
                         "length-scaled bar suggestion.")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        print("SELF-TEST (no GPU, no files)")
        self_test()
        return 0
    if not a.logs:
        ap.error("--logs is required (or use --self-test)")
    run(a.logs, a.out,
        _parse_kv_floats(a.prior_gap_dead_below_lane, "--prior-gap-dead-below-lane"),
        a.prior_gap_dead_below,
        _parse_kv_floats(a.lane_tokens_per_row, "--lane-tokens-per-row"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
