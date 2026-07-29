# Replay at 5–25% prevents forgetting in continual pretraining

**Confidence:** Established · **Used by:** [H19](../hypothesis/todo/H19-four-lane-rematch.md)

## Claim

When continuing pretraining on a new distribution, mixing in a modest fraction of the original or
anchor distribution — roughly **5–25%** — retains the earlier capability at little cost to the new
one. Below that band capability decays; above it, new learning is slowed without much added
retention.

## Evidence

- **Simple and Scalable Strategies to Continually Pre-train LLMs** (Ibrahim et al., arXiv:2403.08763)
  — LR re-warming plus a small replay fraction matches full retraining across distribution shifts.
- **GeRe** (arXiv:2508.04676) — general-sample replay for sustained capability retention.
  *Flagged as a newer lead; the 5–25% window itself is the established part.*

## Why it matters for univi

We have direct in-project evidence of the failure this prevents.
[H05](../hypothesis/done/H05-isolated-ocr-from-scratch.md): when the prior-proof OCR lane was trained
in **isolation**, grounding collapsed to exactly zero — the mixture's other lanes had been what kept
the vision path alive. Removing grounding pressure let it decay.

[H19](../hypothesis/todo/H19-four-lane-rematch.md) therefore retains the two prior-proof lanes
(randstr, spoken-digits) as **anchors at 15% each**, inside the established window, and
pre-registers that anchors must retain ≥ 80% of their single-lane Δperm. If anchors decay, the run is
reproducing [H05](../hypothesis/done/H05-isolated-ocr-from-scratch.md) and should be killed.

## Caveat

This literature concerns **capability retention** under distribution shift. Our use is slightly
different: the anchors are meant to supply continuous *grounding pressure* during training, not only
to preserve a previously learned skill. That extension is our inference.
