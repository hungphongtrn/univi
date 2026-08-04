"""H20 — token-weighted READABLE FRACTION + target-length audit for the audio lanes.

H13 §4/§8: a run whose supervised tokens fall mostly beyond the readable depth
collapses into the uniform-marginal basin.  H13 died at 14.2% readable, H14 was
repaired to 75.2%, H07 succeeded at 100% — all three measured at depth 48.  This
measures where H20's librispeech lane sits BEFORE the GPU is spent, and prices the
repairs (duration cap / target cap).

THE DEPTH ITSELF IS NOW A FLAG, NOT A CONSTANT (2026-07-29).  ~48 came from H13's
K, and K is structurally floored (see ``SCAN_DEPTH`` below): position-resolved
read-outs put the readable depth at ~5-12 answer tokens.  ``--scan-depth`` keeps 48
as the default so ``data/eval/h20-lane-audit.json`` stays reproducible, but every
fraction at 48 is an UPPER BOUND on the honest gate.

Also audits the stored ``original_token_length`` column (which the trainer's
length filter trusts) against the REAL Qwen3 tokenizer the collator uses --
they were produced by different tokenizers and a mismatch would make every
retention number in the H20 doc wrong.

CPU only, no model weights, no GPU.

  CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 uv run python \
      scratchpad/h20_lane_audit.py -n 4000
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: DEFAULT ONLY, AND IT IS THE RETIRED NUMBER.  48 came from H13 §4's K
#: (48.8 / 48.5 / 48.5 answer tokens over a 16x density span), and K is now known to
#: be STRUCTURALLY FLOORED: `h13_analyze.k_from_buckets` scans buckets strictly past
#: a 100-char baseline window, so k_chars >= 100 BY CONSTRUCTION — the identical
#: 48.4913 at soft-token budgets 280 AND 560 was that floor, not a measurement.
#: Position-resolved read-outs put the readable depth at ~5-12 answer tokens.
#: Kept at 48 so `data/eval/h20-lane-audit.json` stays reproducible; use
#: `--scan-depth` for the honest gate.
SCAN_DEPTH = 48
SCAN_DEPTH_RETIRED = 48
SCAN_DEPTH_MEASURED_RANGE = (5, 12)
TRAILER_TOKENS = 2   # response-only masking also supervises `<|im_end|>\n` (H13 §7)


def _target_text(messages) -> str:
    for m in messages:
        if m["role"] == "assistant":
            return "".join(
                p.get("text") or "" for p in m["content"] if p.get("type") == "text"
            )
    return ""


def _readable(tokens: list[int], scan_depth: int = SCAN_DEPTH) -> dict:
    """H13/H14 readable-fraction conventions, all three of them."""
    sup = [t + TRAILER_TOKENS for t in tokens]
    mean_tok = statistics.fmean(tokens)
    return {
        "scan_depth": scan_depth,
        "target_tokens_mean": mean_tok,
        "target_tokens_p50": statistics.median(tokens),
        "target_tokens_p90": sorted(tokens)[int(0.90 * (len(tokens) - 1))],
        "target_tokens_p99": sorted(tokens)[int(0.99 * (len(tokens) - 1))],
        "target_tokens_min": min(tokens),
        "target_tokens_max": max(tokens),
        # H13/H14 headline convention: depth / mean target tokens
        "readable_frac_h13_convention": min(1.0, scan_depth / mean_tok),
        # exact token-weighted: what fraction of supervised tokens is in reach
        "readable_frac_token_weighted": sum(min(scan_depth, t) for t in tokens)
        / sum(tokens),
        "readable_frac_incl_trailer": sum(min(scan_depth, s) for s in sup) / sum(sup),
        "rows_fully_readable": sum(t <= scan_depth for t in tokens) / len(tokens),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=4000, help="rows to tokenize per split")
    ap.add_argument("--tokenizer", default="data/checkpoints/encoder-free-v0/best")
    ap.add_argument("--out", default="data/eval/h20-lane-audit.json")
    ap.add_argument("--scan-depth", type=int, default=SCAN_DEPTH,
                    help=f"answer-token depth the readable fraction is computed to. "
                         f"Default {SCAN_DEPTH} reproduces the existing artifact but "
                         f"is the RETIRED K floor; the measured readable depth is "
                         f"~{SCAN_DEPTH_MEASURED_RANGE[0]}-"
                         f"{SCAN_DEPTH_MEASURED_RANGE[1]} answer tokens.")
    a = ap.parse_args()

    depth = a.scan_depth
    if depth == SCAN_DEPTH_RETIRED:
        print(f"WARNING: scan depth {depth} is the RETIRED K floor (H13's K is "
              f"structurally floored at its 100-char baseline window, so every "
              f"reported K WAS that floor). Measured readable depth is "
              f"~{SCAN_DEPTH_MEASURED_RANGE[0]}-{SCAN_DEPTH_MEASURED_RANGE[1]} "
              f"answer tokens, so every fraction below is an UPPER BOUND. Re-run "
              f"with --scan-depth 5 / 12 before using this as a launch gate.")
    else:
        print(f"scan depth {depth} (the default {SCAN_DEPTH_RETIRED} is the retired "
              f"K floor; the H13/H14/H07 reference points were all measured at 48, "
              f"so they are NOT comparable with this number)")

    from datasets import load_from_disk
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)

    lanes = {
        "librispeech/train": "data/materialized/univi-3M-v0-split/librispeech/train",
        "librispeech/validation": "data/materialized/univi-3M-v0-split/librispeech/validation",
        "spoken-digits/train": "data/materialized/spoken-digits-v0/spoken-digits/train",
        "spoken-digits/validation": "data/materialized/spoken-digits-v0/spoken-digits/validation",
    }

    out: dict = {
        "scan_depth_tokens": depth,
        "scan_depth_is_retired_default": bool(depth == SCAN_DEPTH_RETIRED),
        "scan_depth_note": (
            f"48 is H13's K, which is structurally floored at its 100-char baseline "
            f"window; the position-resolved measurement puts the readable depth at "
            f"~{SCAN_DEPTH_MEASURED_RANGE[0]}-{SCAN_DEPTH_MEASURED_RANGE[1]} answer "
            f"tokens. The reference points below were all computed at depth 48."),
        "tokenizer": a.tokenizer,
        "reference_points": {
            "H13_died_at": 0.142,
            "H14_repaired_to": 0.752,
            "H07_succeeded_at": 1.0,
            "measured_at_scan_depth": SCAN_DEPTH_RETIRED,
        },
        "lanes": {},
    }

    for name, path in lanes.items():
        ds = load_from_disk(path)
        n_total = len(ds)
        n = min(a.n, n_total)
        sub = ds.select(range(n)).select_columns(
            ["messages", "original_token_length", "render_config"]
        )
        qwen_tokens: list[int] = []
        stored: list[int] = []
        durations: list[float] = []
        n_pages: list[int] = []
        texts: list[str] = []
        for row in sub:
            t = _target_text(row["messages"])
            texts.append(t)
            stored.append(int(row["original_token_length"]))
            rc = json.loads(row["render_config"])
            durations.append(float(rc.get("source_duration_ms") or 0.0) / 1000.0)
            n_pages.append(int(rc.get("n_pages", 1)))
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        qwen_tokens = [len(e) for e in enc]

        rec = {
            "path": path,
            "rows_total": n_total,
            "rows_sampled": n,
            "qwen3": _readable(qwen_tokens, depth),
            "stored_original_token_length": {
                "mean": statistics.fmean(stored),
                "max": max(stored),
                "equals_qwen3_rows": sum(
                    int(s == q) for s, q in zip(stored, qwen_tokens)
                ),
                "stored_minus_qwen3_mean": statistics.fmean(
                    s - q for s, q in zip(stored, qwen_tokens)
                ),
                "stored_lt_qwen3_rows": sum(
                    int(s < q) for s, q in zip(stored, qwen_tokens)
                ),
                "max_qwen3_minus_stored": max(
                    q - s for s, q in zip(stored, qwen_tokens)
                ),
            },
            "chars_per_token": statistics.fmean(len(t) for t in texts)
            / statistics.fmean(qwen_tokens),
            "pages_per_row": {str(k): n_pages.count(k) for k in sorted(set(n_pages))},
        }
        if any(durations):
            rec["duration_sec"] = {
                "mean": statistics.fmean(durations),
                "p50": statistics.median(durations),
                "max": max(durations),
            }
            # Repair pricing: cap utterance duration -> keeps only short rows.
            rec["duration_caps"] = {}
            for cap_s in (4.0, 5.0, 6.0, 8.0, 10.0):
                keep = [
                    (q, d) for q, d in zip(qwen_tokens, durations) if 0 < d <= cap_s
                ]
                if not keep:
                    continue
                kt = [q for q, _ in keep]
                rec["duration_caps"][f"<= {cap_s:g}s"] = {
                    "rows_frac": len(keep) / n,
                    "rows_estimate_full_split": round(len(keep) / n * n_total),
                    **_readable(kt, depth),
                }
        out["lanes"][name] = rec

        q = rec["qwen3"]
        print(
            f"{name:26s} n={n:6d}  tok/row mean {q['target_tokens_mean']:7.1f} "
            f"p50 {q['target_tokens_p50']:5.0f} max {q['target_tokens_max']:5d}  "
            f"READABLE(tok-weighted) {100 * q['readable_frac_token_weighted']:5.1f}%  "
            f"(h13-conv {100 * q['readable_frac_h13_convention']:5.1f}%)  "
            f"rows fully readable {100 * q['rows_fully_readable']:5.1f}%"
        )
        s = rec["stored_original_token_length"]
        print(
            f"{'':26s} stored otl mean {s['mean']:7.1f} max {s['max']:5d}  "
            f"stored==qwen3 on {s['equals_qwen3_rows']}/{n} rows  "
            f"stored<qwen3 on {s['stored_lt_qwen3_rows']} rows "
            f"(worst under-count {s['max_qwen3_minus_stored']} tok)"
        )

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
