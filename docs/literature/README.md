# Literature Index

One file per **core claim**, named for the claim. Each file summarizes the insight, the evidence
behind it, and what it implies for univi — not a paper-by-paper reading list.

**Confidence convention.** Every claim is tagged:
- **Established** — multiple peer-reviewed sources, safe to cite.
- **Lead** — plausible and relevant but not verified by us; do not quote numbers without checking.
- **Our inference** — our extrapolation, not a published result. Must be labelled as such wherever
  it appears in [hypothesis docs](../hypothesis/README.md).

| claim | confidence | used by |
|---|---|---|
| [Full-page OCR needs ~1k–4k visual tokens, or learned compression](full-page-ocr-token-budget.md) | Established | [H13](../hypothesis/done/H13-density-ladder-long-targets.md), [H17](../hypothesis/todo/H17-raise-soft-token-budget.md) |
| [Subtracting a blind branch removes the language prior](blind-branch-subtraction.md) | Established (per-token form: our inference) | [H15](../hypothesis/todo/H15-prior-poisoned-text.md), [H16](../hypothesis/todo/H16-prior-gap-weighted-loss.md) |
| [Contrastive decoding recovers grounded content](contrastive-decoding-recovers-grounded-content.md) | Established | [H12](../hypothesis/done/H12-contrastive-decoding-probe.md) |
| [Synthetic prior-proof pretraining bootstraps reading](synthetic-pretraining-bootstraps-reading.md) | Established | [H13](../hypothesis/done/H13-density-ladder-long-targets.md), [H18](../hypothesis/todo/H18-no-learned-scan.md) |
| [Spectrogram patch models classify but do not transcribe](spectrogram-patch-models-do-not-transcribe.md) | Established | [H20](../hypothesis/todo/H20-audio-phoneme-resolution.md) |
| [Replay at 5–25% prevents forgetting in continual pretraining](replay-prevents-forgetting.md) | Established | [H19](../hypothesis/todo/H19-four-lane-rematch.md) |

## Unverified leads

Surfaced during the 2026-07-28 survey but **not** verified — treat as pointers, verify before
quoting: MASS (arXiv:2501.11469), CICD (arXiv:2505.10634), CF-VLM (arXiv:2506.17267), OViP
(arXiv:2505.15963), DocPedia's tokens-vs-DocVQA ablation figures.
