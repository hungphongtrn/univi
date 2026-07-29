# Subtracting a blind branch removes the language prior

**Confidence:** Established for VQA · **per-token autoregressive form: our inference** · **Used by:** [H15](../hypothesis/todo/H15-prior-poisoned-text.md), [H16](../hypothesis/todo/H16-prior-gap-weighted-loss.md)

## Claim

When a multimodal model can answer from a unimodal shortcut, the reliable fix is to train an
explicit **blind branch** (one that sees only the non-image input) and remove its contribution from
the main model's objective — by ensembling, reweighting, or causal subtraction. This measurably
shifts models from prior-following to image-following.

## Evidence

- **VQA-CP** (Agrawal et al., CVPR 2018) — the benchmark that exposed the problem: answer priors
  differ between train and test, and standard VQA models collapse.
- **RUBi** (Cadène et al., NeurIPS 2019) — question-only branch used to reweight the main loss;
  **+8.65 pts** on VQA-CP v2.
- **Learned-Mixin / LMH** (Clark et al., 2019, arXiv:1909.03683) — bias-ensemble with an entropy
  penalty; **~+12 pts** on VQA-CP.
- **CF-VQA** (Niu et al., CVPR 2021, arXiv:2006.04315) — counterfactual causal framing: subtract the
  question-only direct effect at inference.
- **GenB** (Cho & Kim, CVPR 2023) — generative bias modelling of the blind branch.

## Why it matters for univi

Our failure has exactly this shape.
[H10](../hypothesis/done/H10-reading-concentrated-at-start.md) showed reading is purchased only where
the prior fails: at answer position 0 the blind model scores 0.67% and the model captures ~83% of the
available gap; by position 400+ the blind model already scores ~46% and the model captures ~4%.
That is a gradient-allocation problem, which is precisely what this family addresses.

Our blank-image forward is a natural blind branch, and we already compute it for every ablation.

## What is ours, not theirs

The literature reweights **per question/example**. Univi needs it **per token** of an autoregressive
answer, because the prior's strength varies *within* a single target
([H10](../hypothesis/done/H10-reading-concentrated-at-start.md)). The formulation in
[H16](../hypothesis/todo/H16-prior-gap-weighted-loss.md) is our adaptation and has no published
validation — it must be treated as a hypothesis, including its degeneracy guard.
