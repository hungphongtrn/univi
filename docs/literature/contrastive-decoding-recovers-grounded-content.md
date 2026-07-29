# Contrastive decoding recovers grounded content

**Confidence:** Established · **Used by:** [H12](../hypothesis/done/H12-contrastive-decoding-probe.md)

## Claim

When a VLM's language prior outvotes its visual evidence at argmax, contrasting the logits under a
real image against the logits under a degraded/absent image recovers image-grounded content that
greedy decoding loses. The contrast is a pointwise-mutual-information estimate: it amplifies exactly
the tokens the image made *more* likely.

## Evidence

- **VCD — Visual Contrastive Decoding** (Leng et al., CVPR 2024): contrast original vs distorted
  visual inputs at decode time; reduces object hallucination without retraining.
- The general form `logits = (1+α)·logits(cond) − α·logits(uncond)` is the standard
  classifier-free-guidance / contrastive-decoding construction.

## Why it matters for univi

It is a **zero-training diagnostic** for the question we most need answered: is mid-page content
*absent* from the soft-token stream, or *present but suppressed*?

- Present but suppressed ⇒ the constraint is the objective ⇒
  [H15](../hypothesis/todo/H15-prior-poisoned-text.md),
  [H16](../hypothesis/todo/H16-prior-gap-weighted-loss.md).
- Absent ⇒ the constraint is bandwidth or scan ⇒
  [H13](../hypothesis/done/H13-density-ladder-long-targets.md),
  [H17](../hypothesis/todo/H17-raise-soft-token-budget.md).

We already generate the aligned/blank logit pair for every ablation, so the marginal cost is
essentially zero.

## Caveat

This is a *decoding-time* method. A positive result proves the information exists in the
representation; it does **not** prove training can be made to use it. It re-ranks our experiments —
it is not itself a fix.
