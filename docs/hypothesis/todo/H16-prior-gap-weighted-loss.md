# H16 — Reweighting loss by the blind-branch gap restores deep reading

**Status:** TODO · **Cost:** ~6 h GPU · Attacks **(B) objective** · Related: [H15](H15-prior-poisoned-text.md), [H12](../done/H12-contrastive-decoding-probe.md)

## Claim

If reading is only learned where the prior fails
([H10](../done/H10-reading-concentrated-at-start.md)), then explicitly **upweighting the tokens where
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
[H10](../done/H10-reading-concentrated-at-start.md), deep tokens have gap ≈ 0 *precisely because* the
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
  for fineweb-only, relevant if this is ever run on a mixture ([H19](H19-four-lane-rematch.md)).
- The ≈ +35%/step overhead is the doc's estimate and is **not benchmarked** (no GPU available during
  prep).

## Sequencing

Run [H15](H15-prior-poisoned-text.md) first — it is cheaper and carries no loss-code risk. Promote
H16 if H15 moves deep-position gain but plateaus below criterion.
