# Publication plan — grounding analysis & positioning (2026-07-22)

Synthesis of four literature sweeps (text-as-pixels; VLM grounding diagnostics;
modality competition; spectrogram-reading VLMs) + the E1/E2/E3 experiment state.

## 1. Novelty map — what is and is not ours

**Already published (cannot be the contribution):**
- Permuted/blank-image NLL-style ablations: "Beyond Accuracy" (arXiv 2603.03437)
  has a Visual Reliance Score incl. *negative* VRS (image-as-distractor); "VLMs Are
  Blind" (ACCV 2024) reports blank-beats-real.
- "VLMs fail at spectrograms": "Seeing isn't Hearing" (Loakman et al., IJCNLP-AACL
  2025, arXiv 2511.13225) — zero-shot ≈ chance AND finetuned shows no with-image vs
  without-image gain (their version of Δperm≈0). Multiple-choice word ID only.
- Modality competition mechanisms + fixes: gradient starvation (Pezeshki, NeurIPS
  2021), greedy modal learning (Wu, ICML 2022), OGM-GE (CVPR 2022), alternating
  unimodal adaptation (CVPR 2024), data remixing (ICML 2025), DynCIM — but ALL in
  multi-encoder fusion settings (separate audio/vision encoders).
- Text-as-pixels lineage: PIXEL, CLIPPO, Pix2Struct, PIXAR, Chai et al. (EMNLP
  2024). Rendering cost benchmark: "Text-Printed Image" (CVPR 2026) ≈ 7–8 pt drop
  for text→image (Qwen3-VL MMLU 76→68.5).

**Defensibly ours:**
1. **Modality competition inside a single vision channel.** Everything is pixels,
   one input path — yet the competition phenomenon persists. Headline framing:
   *"Unifying modalities into one channel does not unify their learnability."*
   The competition literature assumes separate encoders; we show the same
   suppression dynamics among *renderings* sharing one encoder-free vision path.
2. **The causal elimination chain**, not the diagnosis: E2 model-free decodability
   gate (render lossless, itself a reusable method) → E1 exposure falsification
   (7× audio, grounding flat) → E3 isolation rescue (8× Δperm growth, Δblank
   sign-flip, unsaturated) → E4 intervention. "Seeing isn't Hearing" stops at
   "VLMs lack parametric knowledge"; E3 shows the knowledge is acquirable and the
   mixture is the suppressor — we continue where they end.
3. **First transcription-level (seq2seq) spectrogram-reading attempt** in a
   general VLM (prior art: classification / 4-way multiple choice).
4. **One consistent grounding metric across 4 lanes over training dynamics**
   (0.1% vs 39/63/575%; exposure curves; checkpoint trajectory), vs one-shot
   benchmark audits.
5. Retention metric per modality, directly comparable to Text-Printed Image's
   7–8 pt text-rendering cost.

## 2. Concern: "does the model actually use visual input?" — proof hierarchy

All current evidence is aggregate NLL (weak: consistent with a global cue).
Planned causal tests, strongest first:

- **T1 Counterfactual generation** (headline): generate with aligned / permuted /
  blank image on ckpt-800. Score permuted generations against BOTH the original
  and the shown-image transcript. If gen follows the shown image → literal
  transcription of pixels. Reuses E2 WER code.
- **T2 Temporal occlusion alignment** (killer figure): mask a sliding 2s column
  band of the spectrogram; per-token NLL on transcript; heatmap mask-window ×
  token-position. Diagonal ridge = time-aligned reading; smear = global cue;
  flat = ignoring. Run fineweb as positive control (crisp diagonal expected).
  ~40 fwd passes × 30 samples — cheap.
- **T3 Content/function-word Δperm decomposition**: prior predicts function words;
  pixels should matter for content words. Aggregate +1.3% may hide +8–15% on
  content words. Near-free if per-token losses are captured.
- **T4 Attention/attribution over the 41×6 soft-token grid** (supplementary):
  monotonic alignment path emerging base→ckpt-800. Ties to layer-wise PID
  literature (arXiv 2602.15580, 2603.29676) for an optional depth analysis.

## 3. Experiment matrix to submittable

| # | Experiment | Cost | Decision rule |
|---|-----------|------|---------------|
| P0 | T1+T2(+T3) grounding microscopy on ckpt-800 | ~1 day | Reading confirmed → grounding claims stand; global-cue only → reframe honestly |
| P1 | E4a mixture re-entry from audio-only ckpt-800 (~400–800 steps, ablate every 200) | 1 night | Δperm holds ≥1% → curriculum viable (cite alternating-unimodal); collapses → need per-lane gradient treatment (OGM-GE/CAGrad class) |
| P2 | Retention metric: materialize eval-v0, run eval_lane.py on full-v0/ckpt-2800 | ~1 day | Compare per-modality rendering cost to CVPR-2026's 7–8 pt |
| P3 | E4b audio-only extension (saturation curve, +1600 steps) | 1 night | Ceiling estimate for audio grounding |
| P4 | Mechanism: H1 prior-proof audio (TTS digits) vs H3 resolution (5s pages) vs H2' soft-token probe | 2–3 days | Pick after P0/P1; H1 result = strongest scientific point if positive |
| P5 | Hygiene: bootstrap CIs on all Δ; 2–3 seeds on the E3 probe (central causal claim) | cheap | Required for review |

## 4. Framing + key citations

Working title direction: *"One Channel, Unequal Modalities: Gradient Competition
When Everything Is Pixels"* (or Loakman-adjacent: *"Why Can't VLMs Learn to Read
Spectrograms? A Causal Analysis"*).

Cite: Loakman 2025 (the gap we explain); Dixit 2024 (coarse spectrogram reading
exists); Pezeshki 2021 + Wu 2022 + Geirhos 2020 (mechanism names: gradient
starvation, greedy modal learning, shortcut learning); OGM-GE / data remixing /
alternating-unimodal / DynCIM (fix families E4 tests against); Beyond Accuracy
2603.03437 + VLMs-Are-Blind (metric precedents — position Δperm/Δblank as
per-modality, training-dynamics extensions); PIXEL/CLIPPO/PIXAR/Chai (lineage);
Text-Printed Image CVPR 2026 (retention benchmark); Fuyu/EVE (encoder-free);
AST/SSAST (spectrogram ViTs with audio pretraining — sharpens our question);
Whisper (native upper bound: tiny.en 5.6% WER test-clean).

Framing gift from the diagnostics sweep: audio lane failure is *arbitration/
learnability*, not perceptual blindness (cf. arXiv 2604.09364) — and E2 proves
the percept is intact.
