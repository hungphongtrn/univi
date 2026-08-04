"""
Post-hoc analysis of data/eval/h12-contrastive-decode.json.

Two things the headline metric cannot tell you on its own:

1. **Window coverage.** The pre-registered window is target tokens 10-100. If a
   lane's targets are shorter than 10 tokens the window is EMPTY and the metric
   is computed on a handful of rows. Report how many rows actually contribute.

2. **The diversity confound.** Plain greedy on this checkpoint degenerates into
   repeated phrases. A degenerate decode emits few DISTINCT words, so its
   multiset overlap with the target is mechanically low; anything that breaks the
   loop raises overlap without reading anything. Control: score each generation
   against ANOTHER row's target (a derangement). Grounded recovery raises
   own-target overlap MORE than shuffled-target overlap; a pure diversity/vocab
   shift raises both equally. The grounded excess is (own - shuffled).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/workspace/univi")

IN = sys.argv[1] if len(sys.argv) > 1 else "data/eval/h12-contrastive-decode.json"
WIN_LO, WIN_HI = 10, 100


def words(t):
    return t.lower().split()


def ov(g, t):
    if not t:
        return None
    gc, tc = Counter(g), Counter(t)
    return sum(min(c, gc[w]) for w, c in tc.items()) / len(t)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def f(x, n=4):
    return "n/a" if x is None else f"{x:.{n}f}"


def main():
    from transformers import AutoTokenizer

    d = json.load(open(IN))
    ckpt = d["meta"]["checkpoint"]
    tok = AutoTokenizer.from_pretrained(ckpt)
    alphas = d["meta"]["alphas"]

    for lane, r in d["results"].items():
        rows = r["rows"]
        n = len(rows)
        print(f"\n{'=' * 78}\nLANE {lane}  ({n} rows)\n{'=' * 78}")

        # --- 1. window coverage / target length distribution
        lens = sorted(x["n_target_tokens"] for x in rows)
        n_win = sum(1 for x in rows if x["n_target_words_win"] > 0)
        print(
            f"target token lengths: min {lens[0]}  p25 {lens[n // 4]}  median "
            f"{lens[n // 2]}  p75 {lens[3 * n // 4]}  max {lens[-1]}"
        )
        print(
            f"rows with a NON-EMPTY 10-100 target window: {n_win}/{n} "
            f"({100 * n_win / n:.0f}%)  <-- ov@10-100 is averaged over these only"
        )

        # --- 2. own-target vs shuffled-target control
        # precompute per-row target windows and full targets
        tw, tf = [], []
        for x in rows:
            tid = tok.encode(x["target"], add_special_tokens=False)
            tw.append(words(tok.decode(tid[WIN_LO:WIN_HI])))
            tf.append(words(x["target"]))
        perm = [(i + 1) % n for i in range(n)]  # derangement

        print(
            f"\n{'alpha':>6} | {'ov@10-100':>9} {'shuf':>7} {'excess':>8} | "
            f"{'ov@full':>8} {'shuf':>7} {'excess':>8} | {'rep':>6} {'distinct':>8} {'len':>6}"
        )
        base_ex_w = base_ex_f = None
        for al in alphas:
            k = str(al)
            gw, gf = [], []
            for x in rows:
                gid = tok.encode(x["decodes"][k]["text"], add_special_tokens=False)
                gw.append(words(tok.decode(gid[WIN_LO:WIN_HI])))
                gf.append(words(x["decodes"][k]["text"]))
            own_w = mean([ov(gw[i], tw[i]) for i in range(n)])
            shu_w = mean([ov(gw[i], tw[perm[i]]) for i in range(n)])
            own_f = mean([ov(gf[i], tf[i]) for i in range(n)])
            shu_f = mean([ov(gf[i], tf[perm[i]]) for i in range(n)])
            ex_w = None if own_w is None or shu_w is None else own_w - shu_w
            ex_f = None if own_f is None or shu_f is None else own_f - shu_f
            if al == 0.0:
                base_ex_w, base_ex_f = ex_w, ex_f
            m = r["metrics"][k]
            print(
                f"{al:>6} | {f(own_w)} {f(shu_w,3):>7} {f(ex_w):>8} | "
                f"{f(own_f):>8} {f(shu_f,3):>7} {f(ex_f):>8} | "
                f"{f(m['repetition'],3):>6} {f(m['distinct'],3):>8} {f(m['mean_len'],1):>6}"
            )

        print("\ngrounded-excess ratio vs alpha=0 (the diversity-corrected version of")
        print("the pre-registered '>=2x' test):")
        for al in alphas:
            if al == 0.0:
                continue
            k = str(al)
            gw = []
            for x in rows:
                gid = tok.encode(x["decodes"][k]["text"], add_special_tokens=False)
                gw.append(words(tok.decode(gid[WIN_LO:WIN_HI])))
            own_w = mean([ov(gw[i], tw[i]) for i in range(n)])
            shu_w = mean([ov(gw[i], tw[perm[i]]) for i in range(n)])
            ex = None if own_w is None or shu_w is None else own_w - shu_w
            if ex is None or base_ex_w in (None, 0):
                print(f"  alpha={al}: n/a")
            else:
                print(
                    f"  alpha={al}: excess {f(base_ex_w)} -> {f(ex)}  "
                    f"({ex / base_ex_w:.2f}x, abs {ex - base_ex_w:+.4f})"
                )


if __name__ == "__main__":
    main()
