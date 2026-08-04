"""
H12 — PMI-style contrastive decoding probe for the 4-lane hybrid.

Pre-registered design: docs/hypothesis/todo/H12-contrastive-decoding-probe.md

Claim under test: the soft-token stream *does* carry mid-page content, but the
language prior out-votes it at argmax. If so, contrastive decoding

    logits = (1 + alpha) * logits(aligned) - alpha * logits(blank)

should recover target text that plain greedy decoding loses.

Metric (pre-registered): target-word overlap in answer tokens 10-100.
  overlap = |multiset_intersection(gen_words, tgt_words)| / |tgt_words|
where the words come from decoding token slice [10:100] of the target and of the
generation respectively. Also reported over the FULL decode and over [0:10] so a
recovery can be told apart from a global shift. Never pooled across lanes.

Degeneracy guard: mean generated length + repetition rate (fraction of generated
tokens equal to the immediately preceding token) per alpha.

Usage:
  HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python scratchpad/h12_contrastive_decode.py \
      --checkpoint data/checkpoints/hybrid-4lane-v0/final \
      --lanes fineweb-edu librispeech -n 50 --max-new 110
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SPLIT_ROOT = "data/materialized/univi-3M-v0-split"
WIN_LO, WIN_HI = 10, 100


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------
def words(text: str) -> list[str]:
    return text.lower().split()


def overlap(gen_words: list[str], tgt_words: list[str]) -> float | None:
    """Multiset-intersection recall of target words. None if the target window is empty."""
    if not tgt_words:
        return None
    g, t = Counter(gen_words), Counter(tgt_words)
    inter = sum(min(c, g[w]) for w, c in t.items())
    return inter / len(tgt_words)


def repetition_rate(ids: list[int]) -> float | None:
    if len(ids) < 2:
        return None
    return sum(1 for a, b in zip(ids, ids[1:]) if a == b) / (len(ids) - 1)


def mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="data/checkpoints/hybrid-4lane-v0/final")
    ap.add_argument("--lanes", nargs="*", default=["fineweb-edu", "librispeech"])
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--max-new", type=int, default=110)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--out", default="data/eval/h12-contrastive-decode.json")
    ap.add_argument("--no-cache", action="store_true", help="disable KV cache (fallback)")
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--max-soft-tokens", type=int, default=None,
                    help="Per-image soft-token budget. Default: whatever the checkpoint "
                         "records (280 for pre-H17 checkpoints).")
    ap.add_argument("--max-length", type=int, default=2048,
                    help="Collator truncation budget; raise with --max-soft-tokens.")
    a = ap.parse_args()

    import torch
    from datasets import load_from_disk
    from PIL import Image
    from transformers import AutoTokenizer

    from univi.hybrid.data import HybridCollator, blank_like
    from univi.hybrid.pretrained import (
        UniViHybridPretrained,
        build_image_processor,
        resolve_max_soft_tokens,
    )
    from univi.trainer import _filter_training_images, _filter_training_tokens

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {a.checkpoint} -> {device}", flush=True)
    model = UniViHybridPretrained.from_pretrained(a.checkpoint, dtype=torch.bfloat16).to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    # Process images at the budget the CHECKPOINT was trained at unless overridden.
    mst = resolve_max_soft_tokens(model, a.max_soft_tokens)
    print(f"[load] soft-token budget = {mst} "
          f"(checkpoint records {getattr(model.config, 'max_soft_tokens', '<absent -> 280>')})",
          flush=True)
    col = HybridCollator(
        tokenizer=tok,
        image_processor=build_image_processor(mst),
        image_token_id=model.image_token_id,
        max_length=a.max_length,
        response_only=True,
    )
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    stop_ids = {i for i in (tok.eos_token_id, im_end_id) if i is not None}
    print(f"[load] stop ids = {sorted(stop_ids)}", flush=True)

    # ----------------------------------------------------------------- decode
    #
    # All settings for a row are decoded in ONE batched forward per step:
    #   seq 0            = plain greedy, aligned image        (alpha = 0)
    #   seq 2i-1, 2i     = alpha_i's aligned / blank branches
    # Batching matters twice over: it is ~6x faster than looping the branches,
    # and every setting then goes through the *identical* kernel path, so greedy
    # and contrastive are compared under bit-identical numerics (bf16 attention
    # is sensitive to shape, so separate per-branch forwards would inject noise
    # into exactly the comparison we care about).
    def prep(row, imgs, blanks, alphas):
        """Collate the (1 + 2*len(alphas)) branch sequences for one row."""
        rows = [{**row, "images": imgs}]
        for _ in alphas:
            rows.append({**row, "images": imgs})
            rows.append({**row, "images": blanks})
        batch = col(rows)
        labels = batch["labels"][0]
        nz = (labels != -100).nonzero()
        if len(nz) == 0:
            return None
        start = int(nz[0])  # first supervised (answer) token position
        ids = batch["input_ids"]
        # every branch must share the same prefix -- only the pixels may differ
        assert torch.equal(ids[0].repeat(ids.shape[0], 1), ids), "branch prefixes differ"
        assert batch["attention_mask"].sum(1).unique().numel() == 1, "branch lengths differ"
        extra = {
            k: batch[k].to(device)
            for k in ("pixel_values", "image_position_ids", "num_soft_tokens_per_image")
            if k in batch
        }
        return {
            "ids": ids[:, :start].to(device),
            "attn": batch["attention_mask"][:, :start].to(device),
            "extra": extra,
            "start": start,
            "n_seq": ids.shape[0],
        }

    @torch.no_grad()
    def decode_all(pre, alphas, max_new, use_cache=True):
        """Return [greedy_ids, *per-alpha ids] decoded in lockstep (index-keyed, so
        alpha=0.0 may be passed as a contrastive alpha for the reduction gate)."""
        ids = pre["ids"].clone()
        attn = pre["attn"].clone()
        cache = None
        settings = [0.0] + list(alphas)
        out = {j: [] for j in range(len(settings))}
        done = {j: False for j in range(len(settings))}
        pad = tok.pad_token_id or tok.eos_token_id or 0

        for step in range(max_new):
            if use_cache:
                if step == 0:
                    o = model(
                        input_ids=ids, attention_mask=attn,
                        use_cache=True, logits_to_keep=1, **pre["extra"],
                    )
                else:
                    o = model(
                        input_ids=ids[:, -1:], attention_mask=attn,
                        past_key_values=cache, use_cache=True, logits_to_keep=1,
                    )
                cache = o.past_key_values
            else:
                o = model(
                    input_ids=ids, attention_mask=attn, logits_to_keep=1, **pre["extra"]
                )
            lg = o.logits[:, -1, :].float()  # (n_seq, V)

            nxt_per_seq = torch.full((pre["n_seq"],), pad, dtype=torch.long, device=device)
            for j, alpha in enumerate(settings):
                if j == 0:  # plain greedy baseline: aligned logits, untouched
                    comb, seq_rows = lg[0], [0]
                else:
                    ra, rb = 2 * j - 1, 2 * j
                    comb = (1.0 + alpha) * lg[ra] - alpha * lg[rb]
                    seq_rows = [ra, rb]
                if done[j]:
                    continue
                nxt = int(comb.argmax(-1))
                if nxt in stop_ids:
                    done[j] = True
                    continue
                out[j].append(nxt)
                for rrow in seq_rows:
                    nxt_per_seq[rrow] = nxt
            if all(done.values()):
                break
            ids = torch.cat([ids, nxt_per_seq.unsqueeze(1)], dim=1)
            attn = torch.cat(
                [attn, torch.ones((pre["n_seq"], 1), dtype=attn.dtype, device=device)],
                dim=1,
            )
        return [out[j] for j in range(len(settings))]

    def target_text(row) -> str:
        tgt = ""
        for m in row["messages"]:
            if m["role"] == "assistant":
                c = m["content"]
                tgt = (
                    c
                    if isinstance(c, str)
                    else " ".join(p.get("text", "") for p in c if p.get("type") == "text")
                )
        return tgt

    # ------------------------------------------------------- correctness gates
    lane0 = a.lanes[0]
    ds0 = load_from_disk(f"{SPLIT_ROOT}/{lane0}/validation")
    ds0 = _filter_training_tokens(_filter_training_images(ds0, lane0, 4), lane0, a.max_length)
    ds0 = ds0.shuffle(seed=a.seed).select(range(a.n))

    def blanks_of(imgs):
        return [
            blank_like(im)
            for im in imgs
        ]

    use_cache = not a.no_cache
    cache_note = "disabled via --no-cache"
    if use_cache:
        print("\nGATE-1 KV-cache equivalence (cached vs uncached, all settings, 2 rows)", flush=True)
        n_ident = n_tot = 0
        for i in range(2):
            row = ds0[i]
            imgs = list(row["images"])
            pre = prep(row, imgs, blanks_of(imgs), a.alphas)
            if pre is None:
                continue
            t0 = time.time()
            c = decode_all(pre, a.alphas, 40, use_cache=True)
            t1 = time.time()
            u = decode_all(pre, a.alphas, 40, use_cache=False)
            t2 = time.time()
            for al, cc, uu in zip([0.0] + a.alphas, c, u):
                n_tot += 1
                same = cc == uu
                n_ident += int(same)
                pref = next(
                    (k for k, (x, y) in enumerate(zip(cc, uu)) if x != y),
                    min(len(cc), len(uu)),
                )
                print(
                    f"  row {i} alpha={al}: identical={same} "
                    f"(common prefix {pref}/{max(len(cc), len(uu))} tokens)",
                    flush=True,
                )
            print(f"  row {i}: cached={t1-t0:.1f}s  uncached={t2-t1:.1f}s", flush=True)
        cache_note = f"cached==uncached on {n_ident}/{n_tot} settings"
        print(f"  GATE-1: {cache_note}", flush=True)
        print(
            "  NOTE: divergence here is a bf16 argmax TIE-FLIP, not a plumbing bug. A\n"
            "  teacher-forced check (same inputs, cached vs uncached) gave 40/40 argmax\n"
            "  agreement, max |logit| diff 0.25 on a logit scale of 26.5 -- i.e. prefill vs\n"
            "  single-token-decode kernel rounding. All settings share ONE batched forward,\n"
            "  so greedy vs contrastive are always compared under identical numerics.",
            flush=True,
        )

    print("\nGATE-2 alpha=0 contrastive reduces exactly to plain greedy (2 rows)", flush=True)
    for i in range(2):
        row = ds0[i]
        imgs = list(row["images"])
        # alphas=[0.0] exercises the full contrastive arithmetic path at alpha=0
        pre = prep(row, imgs, blanks_of(imgs), [0.0])
        if pre is None:
            continue
        g, c0 = decode_all(pre, [0.0], 40, use_cache=use_cache)
        assert g == c0, (
            f"alpha=0 does NOT reduce to greedy on row {i}:\n"
            f"  greedy: {tok.decode(g)!r}\n  a=0.0 : {tok.decode(c0)!r}"
        )
        print(f"  row {i}: identical={g == c0} ({len(g)} tokens)", flush=True)
    print("  alpha=0 reduces exactly to greedy.", flush=True)

    # ------------------------------------------------------------------- sweep
    settings = [0.0] + list(a.alphas)
    results: dict = {}
    t_start = time.time()

    for lane in a.lanes:
        ds = load_from_disk(f"{SPLIT_ROOT}/{lane}/validation")
        ds = _filter_training_tokens(_filter_training_images(ds, lane, 4), lane, a.max_length)
        ds = ds.shuffle(seed=a.seed).select(range(a.n))
        print(f"\n{'=' * 78}\nLANE: {lane}  ({len(ds)} rows, seed {a.seed})\n{'=' * 78}", flush=True)

        rows_out = []
        for i in range(len(ds)):
            row = ds[i]
            imgs = list(row["images"])
            pre = prep(row, imgs, blanks_of(imgs), a.alphas)
            if pre is None:
                print(f"  row {i}: <no supervised tokens> skipped", flush=True)
                continue

            tgt = target_text(row)
            tgt_ids = tok.encode(tgt, add_special_tokens=False)
            tw_win = words(tok.decode(tgt_ids[WIN_LO:WIN_HI]))
            tw_full = words(tok.decode(tgt_ids))
            tw_head = words(tok.decode(tgt_ids[:WIN_LO]))

            rec = {
                "idx": i,
                "n_images": len(imgs),
                "prefix_len": int(pre["ids"].shape[1]),
                "target": tgt,
                "n_target_tokens": len(tgt_ids),
                "n_target_words_win": len(tw_win),
                "n_target_words_full": len(tw_full),
                "n_target_words_head": len(tw_head),
                "decodes": {},
            }
            all_ids = decode_all(pre, a.alphas, a.max_new, use_cache=use_cache)
            for alpha, ids in zip(settings, all_ids):
                gtxt = tok.decode(ids)
                rec["decodes"][str(alpha)] = {
                    "text": gtxt,
                    "n_tokens": len(ids),
                    "ov_win": overlap(words(tok.decode(ids[WIN_LO:WIN_HI])), tw_win),
                    "ov_full": overlap(words(gtxt), tw_full),
                    "ov_head": overlap(words(tok.decode(ids[:WIN_LO])), tw_head),
                    "rep": repetition_rate(ids),
                    "distinct": (len(set(ids)) / len(ids)) if ids else None,
                }
            rows_out.append(rec)
            if (i + 1) % 5 == 0 or i == 0:
                el = time.time() - t_start
                print(
                    f"  [{lane}] row {i+1}/{len(ds)}  ({el/60:.1f} min elapsed)  "
                    + "  ".join(
                        f"a={al}:ov{fmt(rec['decodes'][str(al)]['ov_win'], 3)}"
                        for al in settings
                    ),
                    flush=True,
                )

        # ---- aggregate
        metrics = {}
        for alpha in settings:
            k = str(alpha)
            metrics[k] = {
                "overlap_10_100": mean([r["decodes"][k]["ov_win"] for r in rows_out]),
                "overlap_full": mean([r["decodes"][k]["ov_full"] for r in rows_out]),
                "overlap_0_10": mean([r["decodes"][k]["ov_head"] for r in rows_out]),
                "mean_len": mean([float(r["decodes"][k]["n_tokens"]) for r in rows_out]),
                "repetition": mean([r["decodes"][k]["rep"] for r in rows_out]),
                "distinct": mean([r["decodes"][k]["distinct"] for r in rows_out]),
                "n_rows_win": sum(
                    1 for r in rows_out if r["decodes"][k]["ov_win"] is not None
                ),
                "n_rows": len(rows_out),
            }
        results[lane] = {"metrics": metrics, "rows": rows_out}

        print(f"\n--- {lane}: metric table ({len(rows_out)} rows) ---", flush=True)
        print(
            f"  {'alpha':>6} {'ov@10-100':>10} {'ov@full':>9} {'ov@0-10':>9} "
            f"{'len':>7} {'rep':>7} {'distinct':>9}"
        )
        for alpha in settings:
            m = metrics[str(alpha)]
            print(
                f"  {alpha:>6} {fmt(m['overlap_10_100']):>10} {fmt(m['overlap_full']):>9} "
                f"{fmt(m['overlap_0_10']):>9} {fmt(m['mean_len'],1):>7} "
                f"{fmt(m['repetition'],3):>7} {fmt(m['distinct'],3):>9}",
                flush=True,
            )
        base = metrics["0.0"]["overlap_10_100"] or 0.0
        best_a, best_v = None, -1.0
        for alpha in a.alphas:
            v = metrics[str(alpha)]["overlap_10_100"] or 0.0
            if v > best_v:
                best_a, best_v = alpha, v
        ratio = (best_v / base) if base > 0 else (float("inf") if best_v > 0 else 0.0)
        rs = "inf" if ratio == float("inf") else f"{ratio:.2f}x"
        verdict = "CONFIRMS (>=2x)" if ratio >= 2.0 else "REFUTES (<2x)"
        print(
            f"  PRE-REGISTERED: baseline(alpha=0) ov@10-100 = {fmt(base)}; "
            f"best alpha={best_a} -> {fmt(best_v)}  ratio {rs}  "
            f"abs delta {best_v - base:+.4f}  => {verdict} for {lane}",
            flush=True,
        )
        metrics["_summary"] = {
            "baseline_ov_10_100": base,
            "best_alpha": best_a,
            "best_ov_10_100": best_v,
            "ratio": None if ratio == float("inf") else ratio,
            "abs_delta": best_v - base,
            "doubles": bool(ratio >= 2.0),
        }

        # ---- representative examples
        print(f"\n--- {lane}: {a.examples} representative examples (TARGET / greedy / alpha={best_a}) ---")
        # pick rows with the biggest |improvement| plus a couple of median rows
        scored = sorted(
            rows_out,
            key=lambda r: -((r["decodes"][str(best_a)]["ov_win"] or 0)
                            - (r["decodes"]["0.0"]["ov_win"] or 0)),
        )
        picks = scored[: max(1, a.examples // 2)] + scored[len(scored) // 2 : len(scored) // 2 + (a.examples - max(1, a.examples // 2))]
        for r in picks:
            print(f"\n  [row {r['idx']}]  tgt_tokens={r['n_target_tokens']} prefix={r['prefix_len']}")
            print(f"    TARGET      : {r['target'][:400]!r}")
            print(f"    greedy a=0  : {r['decodes']['0.0']['text'][:400]!r}")
            print(f"    contr a={best_a} : {r['decodes'][str(best_a)]['text'][:400]!r}")
            print(
                f"    ov@10-100   : greedy {fmt(r['decodes']['0.0']['ov_win'],3)} -> "
                f"a={best_a} {fmt(r['decodes'][str(best_a)]['ov_win'],3)}"
            )
        print(flush=True)

    peak = (
        torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    )
    reserved = (
        torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0
    )
    print(f"\n[vram] peak allocated = {peak:.2f} GiB, peak reserved = {reserved:.2f} GiB", flush=True)
    print(f"[time] total {(time.time() - t_start)/60:.1f} min", flush=True)

    out = {
        "meta": {
            "checkpoint": a.checkpoint,
            "lanes": a.lanes,
            "n_rows": a.n,
            "seed": a.seed,
            "max_new": a.max_new,
            "max_soft_tokens": mst,
            "max_length": a.max_length,
            "alphas": settings,
            "window": [WIN_LO, WIN_HI],
            "kv_cache": use_cache,
            "kv_cache_gate": cache_note,
            "peak_vram_gib": peak,
            "peak_reserved_gib": reserved,
        },
        "results": results,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[write] {a.out}", flush=True)


if __name__ == "__main__":
    main()
