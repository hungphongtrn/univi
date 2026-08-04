# H16 — Reweighting loss by the blind-branch gap restores deep reading

**Status:** DONE (2026-07-30) — **REFUTED**, on both conjuncts of its own criterion, one of which is
position-resolved and therefore not subject to the length-dilution defect that voided
[H17](H17-raise-soft-token-budget.md). Gain at answer tokens 10–100 came out at **≈ 0**
against a **≥ +10 pts** bar, and Δperm at **+0.687%** against **≥ +30%**. The degeneracy guard
**passed** (`blank_ce_inflation` −0.0123, bar +0.10), so this is a genuine refutation and not a VOID.
The pre-registered failure mode recorded in [the RULING below](#ruling--this-docs-prose-and-its-formula-point-in-opposite-directions)
is confirmed and then some: the weighting did not merely fail to buy depth, it **abolished the
position-0 reading that already existed** (+70.0 → +0.0 pts vs the
[H09](H09-balanced-mixture-prevents-drowning.md) baseline). The prose variant of the
intervention remains **untested and is not refuted by this**. See [Verdict](#verdict-2026-07-30).
· **Cost:** ~6.2 h GPU (600 steps) + ~5 min probes · Attacks **(B) objective** · Related:
[H15](H15-prior-poisoned-text.md), [H12](H12-contrastive-decoding-probe.md)

## Verdict (2026-07-30)

Ran as `configs/h16_gap_weighted.yaml`, fineweb-only, 600 steps, effective batch 64, formula as
written (`gap_beta` 1.0). Probed with `scratchpad/hybrid_4lane_ablation.py` →
`data/eval/h16-ablation-gap-weighted.json` and `scratchpad/hybrid_4lane_position_decay.py` →
`data/eval/hybrid-4lane-position-decay-h16-gap-weighted.json`, 150 rows, `max_length 4096`.

**Both conjuncts fail, and the position-resolved one fails decisively.**

| answer position | H09 4-lane baseline | **H16 gap-weighted** |
|---|---|---|
| 0 | **+70.0** | **+0.0** |
| 1 | +6.7 | +3.3 |
| 2–4 | +3.1 | +0.7 |
| 5–9 | +2.3 | −0.5 |
| **10–19** | +0.1 | **−0.3** |
| **20–49** | +0.8 | **+0.1** |
| **50–99** | +0.7 | **+0.2** |
| 100–199 | +1.2 | +0.4 |

The criterion's window (tokens 10–100) reads **−0.3 / +0.1 / +0.2 pts** against a **+10 pts** bar.
Aggregate Δperm +0.687%, Δblank +0.276%, reading gain +0.254 pts — against the same lane's
unweighted baseline of Δperm **+2.513%**, Δblank +3.477%, gain +1.246 pts
(`data/eval/hybrid-4lane-ablation-4lane-final.json`). So the intervention moved fineweb reading
**down by ~3.7×** while receiving **4× the gradient share** (fineweb-only vs ~25% of a 4-lane
mixture). `grad_norm` collapsed 0.734 → 0.099 → 0.100 and eval moved only 2.566 → 2.522.

**Why this is REFUTED and not VOID.** The doc's own degeneracy guard — eval blank-CE must not inflate
by more than +10% — **passes** at −0.0123, so the model was not pushed to be *worse blind*. And unlike
H17, the load-bearing conjunct here is stated in **position-resolved gain at tokens 10–100**, which is
immune to the aggregate-Δperm length dilution that
[H13 §4](H13-density-ladder-long-targets.md) measured. The Δperm ≥ +30% conjunct *is* subject
to it and should not be leaned on, but it is not needed: the position-resolved conjunct fails by two
orders of magnitude.

### The mechanism, stated as a hypothesis — and the control that is missing

The weight `w_t = 1 + β·max(0, CE_blank_t − CE_aligned_t)` is large **exactly where the model already
reads** and ≈ 1 everywhere else. So the loss is concentrated on tokens that already carry near-zero
CE and therefore near-zero gradient, while the deep tokens holding almost all the loss mass are
relatively starved. That predicts an *under-trained* run, which is what `grad_norm` 0.100 shows, and it
predicts no new depth, which is what the bins show. It does **not** by itself predict the destruction
of position-0 reading.

**This is a hypothesis, because H16 has no matched control.** The +70.0 → +0.0 comparison is against
H09's *4-lane* final checkpoint — a different data diet and a different schedule. The correct control
is fineweb-only, 600 steps, `gap_weighted_loss: false`, and it was never run. The doc already
identifies it: `log_blank_ce` can be enabled **alone** for "a zero-risk instrumented control run". Until
that exists, "gap-weighting destroyed position-0 reading" is confounded with "fineweb-only for 600
steps does not produce reading either". The **criterion** verdict does not depend on the control — the
bars are absolute — but the mechanism story does.

### What this does NOT license

- **It does not refute the prose variant.** The [RULING](#ruling--this-docs-prose-and-its-formula-point-in-opposite-directions)
  recorded before the run that the doc's prose ("upweight tokens where a blind model already
  *succeeds*") is the **opposite** intervention to the formula, and that a failure here leaves the
  prose variant as "a distinct, untested intervention [that] deserves its own rung". That stands.
  Under the corrected depth picture it is also the more plausible of the two.
- **It does not refute (B) on its own** — but combined with [H15](H15-prior-poisoned-text.md), which
  removed the language prior *by construction* and bought only ~4 extra tokens of depth, the
  objective-side account is now closed from two independent directions. See
  [H15's verdict](H15-prior-poisoned-text.md#verdict-2026-07-30).
- **It does not cleanly separate "the intervention is wrong" from "the run collapsed."**
  `grad_norm` 0.100 is the same signature as H13 and H17's raised-budget legs. A rerun of the prose
  variant must carry a `grad_norm < 0.10` abort and a step-200 Δperm probe.

## Claim

If reading is only learned where the prior fails
([H10](H10-reading-concentrated-at-start.md)), then explicitly **upweighting the tokens where
a blind model already succeeds** should force the model to buy reading at depth too.

## Design

Per batch, run a no-grad **blank-image** forward. Per token:

```
w_t = 1 + β · max(0, CE_blank_t − CE_aligned_t.detach())      # normalized to mean 1
loss = Σ w_t · CE_aligned_t
```

Implement as a `compute_loss` override in `univi/hybrid/train_pretrained.py` (`_make_trainer_class`),
plus a `HybridCollator` option emitting the blank pixel tensors (gray-128, matching the probes).
Overhead ≈ +35%/step — the blind branch is forward-only.

This is the autoregressive-token transfer of
[blind-branch subtraction](../../literature/blind-branch-subtraction.md) (RUBi, Learned-Mixin), which
is established for VQA. **The per-token adaptation is ours, not a published result.**

Deliberate design choice: **reweighting**, not an adversarial "make the blank branch worse" term.
The adversarial form has a degenerate solution (inflate blank loss without improving aligned
reading); reweighting does not.

## Pre-registered criterion

fineweb-only, 600 steps:

- **Confirms:** positional reading gain at tokens 10–100 **≥ +10 pts**, and Δperm ≥ +30%.
- **Guard (degeneracy check):** eval blank-CE must not inflate by more than +10%. If it does, the
  model is being pushed to be *worse blind* rather than *better sighted*, and the result is void.

## Prep status (2026-07-28) — implemented and verified, not yet run

`univi/hybrid/data.py` gained `blank_like()` + `HybridCollator.emit_blank_images` (blank tensors go
through the *same* image-processor path, and it raises if blank soft-token counts ever diverge from
aligned). `univi/hybrid/train_pretrained.py` gained module-level `per_token_ce()` / `gap_weights()`
and `compute_loss` / `evaluation_loop` / `log` overrides, all gated behind
`training.gap_weighted_loss` / `gap_beta` / `log_blank_ce` (**default OFF**, first-line early return
into stock `Trainer`). Config: `configs/h16_gap_weighted.yaml` (fineweb-only, 600 steps, effective
batch 64 via 4×16, `< 26 GB` noted for the launcher). 5 CPU tests added.

Verified: **off-path is bit-identical** to stock (`11.947490692138672` both, `torch.equal` True);
β=0 identical; hand-checked weights `[1.778, 0.444, 0.444, 0.0, 1.333]` with max error 0.0, mean 1
over supervised tokens, `requires_grad False`, weight 0 at `-100`; **gradient isolation proven** —
with β=0, swapping the blind input gray→noise leaves all 36 parameter grads bit-identical, and
`blank_pixel_values.grad is None`; eval is never reweighted, so `eval_loss` stays comparable to every
prior run. Pytest 16 passed. The blind branch runs **before** the aligned forward so its logits are
freed first — peak VRAM ≈ the unweighted path, cost is wall-clock not memory.

**Guard is live, not reconstructed afterwards:** `eval[_<lane>]_aligned_ce`, `_blank_ce`,
`_prior_gap_ce`, and `_blank_ce_inflation` are injected into `output.metrics` before logging —
**kill the run if `_blank_ce_inflation` exceeds +0.10**, which is the doc's pre-registered void
condition. Training-side, `gap_prior_gap_mean` / `gap_weight_max` log at `logging_steps`.

### RULING — this doc's prose and its formula point in opposite directions

The claim sentence says upweight "tokens where a blind model already **succeeds**". The formula
upweights `max(0, CE_blank − CE_aligned)` — tokens where the blind model **fails** and the sighted
model already reads them. After mean-1 normalization, tokens the prior solves are *down*weighted.
These are opposite interventions.

**Decision: run the formula as written.** It is the precise, literature-grounded statement (it is the
RUBi / Learned-Mixin direction the cited
[blind-branch subtraction](../../literature/blind-branch-subtraction.md) note describes), and a
pre-registration is the formula, not its prose gloss. The prose sentence is treated as an imprecise
summary.

**But note the predicted failure mode, recorded before the run:** by
[H10](H10-reading-concentrated-at-start.md), deep tokens have gap ≈ 0 *precisely because* the
model does not read there, so they receive the minimum weight — this weighting amplifies the
position-0 reading that already exists rather than forcing depth. If H16 fails its criterion, the
prose variant (upweight where the prior *succeeds*) is a **distinct, untested intervention** and
deserves its own rung rather than being treated as refuted by this result.

### Other judgement calls

- **`gap_beta` default 1.0** — weights are then literally `1 + gap-in-nats`; on fineweb the per-token
  gap runs ~0 (deep) to ~6 (position 0), giving a ~0.5×–5× spread after normalization. No upper clip
  (not in the doc); `gap_weight_max` is logged so runaway weights are visible.
- **Weighting is off at eval** (gated on `model.training`), so the *training* loss series is not
  magnitude-comparable to unweighted runs — the eval series is.
- **mean(w)=1 is per-microbatch**, not per-accumulation-window (the window's token count isn't
  available inside `compute_loss`); the shared `num_items_in_batch` denominator keeps overall scale
  right.
- `log_blank_ce` can be enabled **alone** for a zero-risk instrumented control run (guard metrics,
  stock loss).
- Imageless microbatches carry no blank tensors and fall through to the unweighted loss — irrelevant
  for fineweb-only, relevant if this is ever run on a mixture ([H19](../todo/H19-four-lane-rematch.md)).
- The ≈ +35%/step overhead is the doc's estimate and is **not benchmarked** (no GPU available during
  prep).

## Sequencing

Run [H15](H15-prior-poisoned-text.md) first — it is cheaper and carries no loss-code risk. Promote
H16 if H15 moves deep-position gain but plateaus below criterion.
