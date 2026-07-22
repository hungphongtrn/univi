# Phase 4: Diagnostics & Publication

## Phase Goal
Compute retention metrics, shuffled-option diagnostics, invalid output rates, and publish results. Document render settings and source splits for reproduction.

## Stub Note
This phase will be detailed after Phase 3 produces evaluation results. The sections below are high-level placeholders.

## Files to Touch (rough sketch)
- `analysis/report.ipynb` — Analysis notebook
- `analysis/metrics.py` — Metric computation utilities
- `results/` — Output directory for tables and figures

## Tasks (to be detailed)
1. Compute retention metric for each modality (image-only acc ÷ native upper-bound acc).
2. Run shuffled-options control and low-prior diagnostic.
3. Compute invalid output rate under strict first-letter parsing.
4. Generate tables/figures for all metrics.
5. Document exact render settings used (font, canvas size, colormap, audio window).
6. Document source splits and subset sizes.
7. Push final materialized dataset to HF Hub with documentation.
8. Update issue #1 with results summary.

## Completion Criteria (rough)
- Retention ≥80 % for text, ≥50 % for audio.
- Shuffled-option diagnostic reported.
- Invalid output rate reported.
- All render settings and source splits documented in a reproducible format.
- Results published on the GitHub issue.
