# H19 — Curriculum + anchors + weighted loss makes the real mixture read

**Status:** TODO · **Cost:** ~16 h GPU · Integration run · Related: [H09](../done/H09-balanced-mixture-prevents-drowning.md), [H15](H15-prior-poisoned-text.md), [H16](H16-prior-gap-weighted-loss.md)

## Claim

[H09](../done/H09-balanced-mixture-prevents-drowning.md) failed from a cold start with an
unmonitored objective. Warm-starting from a checkpoint that already reads, keeping prior-proof
**anchor lanes** in the mixture, and using a grounding-aware loss will make the real four-lane
mixture read deeply.

## Design — changed on every axis the evidence demands

1. **Warm start** from the best [H15](H15-prior-poisoned-text.md)/[H16](H16-prior-gap-weighted-loss.md)
   checkpoint. Both prior-proof successes showed the grad_norm dip-then-rise recruitment signature;
   [H09](../done/H09-balanced-mixture-prevents-drowning.md)'s cold start never recruited (flat
   0.42–0.58 for 4000 steps).
2. **Anchor lanes retained** — randstr 15% + spoken-digits 15%, four real lanes ~17.5% each.
   [H05](../done/H05-isolated-ocr-from-scratch.md)'s lesson is that removing prior-proof pressure
   lets reading decay to zero; the replay share follows the continual-learning
   [5–25% window](../../literature/replay-prevents-forgetting.md).
3. **Grounding-aware loss** from [H16](H16-prior-gap-weighted-loss.md) on all lanes.
4. **Live Δblank monitor** — **already built** (2026-07-28, as part of
   [H16](H16-prior-gap-weighted-loss.md)'s prep). Setting `training.log_blank_ce: true` runs a
   no-grad blank forward per eval batch and injects `eval_<lane>_aligned_ce`,
   `eval_<lane>_blank_ce`, `eval_<lane>_prior_gap_ce` and `eval_<lane>_blank_ce_inflation` into the
   metrics before they are logged, per lane. It can be enabled **without** `gap_weighted_loss`, so
   the monitor costs one extra forward per eval batch and leaves the loss untouched. **Mandatory**:
   eval CE is proven orthogonal to grounding
   ([H09](../done/H09-balanced-mixture-prevents-drowning.md)), and this run must not be able to fail
   silently for 27 hours again.

   Two integration caveats for this run specifically: imageless microbatches carry no blank tensors
   and fall through unweighted (fine for a monitor, relevant if `gap_weighted_loss` is on for a
   mixture), and the `poisoned-text` / `masked-randstr` lanes store extra columns that break
   `concatenate_datasets` — `remove_columns` them before mixing.

2000–2500 steps at 280 soft tokens, or 1120 if [H17](H17-raise-soft-token-budget.md) passed.

## Pre-registered criterion

- **Confirms:** fineweb Δperm **≥ +30%** (an order above the saturated +2.5%) **and** positional gain
  at tokens 10–100 ≥ +5 pts; librispeech position-0 gain flips positive and Δblank ≥ +20%; anchors
  retain ≥ 80% of their single-lane Δperm.
- **Diagnostic failure:** anchors hold but real lanes stay < +10% ⇒ real-lane density exceeds the
  [H13](../done/H13-density-ladder-long-targets.md) ceiling ⇒ [H17](H17-raise-soft-token-budget.md) becomes
  mandatory rather than optional.

## Do not run before the upstream single-lane results

This is the expensive integration. Everything upstream of it is single-lane, bounded, and carries a
kill criterion. Run SOLO.
