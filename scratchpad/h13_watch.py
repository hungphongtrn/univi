"""Poll the live H13 W&B run and emit one line per NEW eval step.

The HF ProgressCallback routes loss/eval dicts into the tqdm bar, so the nohup log
file contains no metrics at all — a file monitor on it is blind. This polls the
W&B history instead and prints only on new eval rows (plus a periodic grad_norm
heartbeat), so the event stream stays sparse.

Emits on: new eval, dead-gradient warning, run finish/crash.
"""
from __future__ import annotations

import sys
import time

RUN = "yuuart/huggingface/a6kv5ysx"
FLOOR = 5.60  # nats/tok, ~constant across all 5 rungs
POLL_S = 300

import wandb  # noqa: E402

api = wandb.Api()
seen_eval: set[int] = set()
last_beat = 0.0

while True:
    try:
        run = api.run(RUN)
        run.load(force=True)
        h = run.history(pandas=True)
    except Exception as e:  # transient API failure must not kill the watch
        print(f"[warn] wandb poll failed: {e}", flush=True)
        time.sleep(POLL_S)
        continue

    if len(h):
        # W&B logs EACH subset's eval as its own sparse row sharing one
        # train/global_step, so a per-row read sees only one rung. Group by step
        # and collapse, else four of five rungs are silently invisible.
        ev = sorted(c for c in h.columns if c.startswith("eval/") and c.endswith("_loss"))
        if ev and "train/global_step" in h.columns:
            g = h.dropna(subset=ev, how="all").groupby("train/global_step")[ev].first()
            for step, row in g.iterrows():
                step = int(step)
                if step in seen_eval:
                    continue
                have = {c: row[c] for c in ev if row[c] == row[c]}
                if len(have) < len(ev):      # wait until every rung has reported
                    continue
                seen_eval.add(step)
                parts = [f"{c.split('/')[1].replace('_loss', '')}={v:.3f}"
                         f"{'*' if v < FLOOR else ''}" for c, v in sorted(have.items())]
                print(f"EVAL step {step}: " + " ".join(parts)
                      + f"  (floor ~{FLOOR}; * = below floor ⇒ reading)", flush=True)

        now = time.time()
        if now - last_beat > 1800:
            last_beat = now
            t = h.dropna(subset=["train/grad_norm"]) if "train/grad_norm" in h else h.iloc[0:0]
            if len(t):
                last = t.iloc[-1]
                gn = float(last["train/grad_norm"])
                st = int(last.get("train/global_step") or 0)
                tl = float(last.get("train/loss") or float("nan"))
                flag = "  DEAD-GRAD" if gn < 0.35 else ""
                print(f"beat step {st}: loss={tl:.3f} grad_norm={gn:.3f}{flag}", flush=True)

    if run.state != "running":
        print(f"RUN {run.state.upper()} at step {int(h['train/global_step'].iloc[-1]) if len(h) else 0}", flush=True)
        sys.exit(0)

    time.sleep(POLL_S)
