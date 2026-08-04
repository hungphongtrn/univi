# Encoder-free thread — post-mortem & program reassessment (2026-07-26)

Written after H3 concluded. Purpose: synthesize H1/H2/H3 into a thread verdict and
weigh the encoder-free bet against the sibling Gemma and hybrid (issue #6) threads
before committing more GPU. **No new run was launched** — this is the "pause &
reassess" deliverable.

## The program question

Can a single VLM read **text + natural images + audio**, all through the vision
path (text→image, audio→log-mel image, images unchanged)? Three architectural bets
have been tried:

| Thread | Vision path | Decoder | Training | Status |
|---|---|---|---|---|
| **Gemma** | **pretrained** E2B vision tower | pretrained Gemma-4 E2B | 4-bit QLoRA | ran to ~19.6k steps |
| **Encoder-free** | **from-scratch** tiny linear embedder | **pretrained** Qwen3-1.7B | full FT | H1/H2/H3 done |
| **Hybrid (#6)** | **random-init** Gemma vision tower | **random-init** Qwen3-1.7B | full FT | 250-step smoke only |

## Encoder-free thread verdict (H1 → H2 → H3)

- **H1** (plain single-LR 3e-4 mixture): all four lanes read, but **uniformly
  shallow** (Δperm ~3%; librispeech ~1.9%). One genuinely novel positive — **audio
  grounded** (Δblank +1.75%), *reversing* Gemma, where audio sits at the ignore-floor
  and actively avoids the spectrogram. Reusing the decoder through a shared pixel path
  gave audio a foothold Gemma never found.
- **H2** (freeze decoder 800 steps + split LR): **FAILED** — content-blind
  soft-prompt collapse. Against a frozen decoder the only way a from-scratch embedder
  lowers loss is a constant steering vector; Δperm → ignore-floor, Δblank exploded.
- **H3** (pure random-string OCR, zero language shortcut, warm-start from H1):
  **FAILED, decisively** — grounding **collapsed to exactly zero**. aligned = permuted
  = blank identical to 5–6 sig figs; the image is provably ignored. CE fell 6.20 → 4.02
  (below the 5.61 no-reading floor) *entirely* by the decoder modeling the answer's
  deterministic format, while the embedder's signal decayed to nothing. Same code+rows
  shows H1 still responding (Δblank +1.6%), so this is a real regression, not a bug.

**Core finding:** with a **from-scratch tiny linear embedder + a strong pretrained
decoder**, H1's weak reading was the *ceiling, not a floor to grow from*. Both attempts
to deepen it collapsed it. The decoder out-competes the tiny embedder for gradient and
settles into a non-reading basin — using the LM prior for structured text (H1/H2), or
pure format-modeling for random text (H3). The collapse is **repeatable** (H2 via
frozen warmup, H3 via isolation), which indicts the pairing, not the hyperparameters.

## Cross-thread synthesis — the decisive variable is *vision-side pretraining*

Line the three threads up and the pattern is clean:

- **Gemma (pretrained vision)** → reads 3 of 4 modalities *strongly* (Δperm +575 /
  +63 / +39%). Render proven lossless (E2: 0% WER round-trip). Only **audio** is the
  hold-out, and E1/E2 localized that to **learnability** (audio at 3.5% mixture share
  fell into an ignore basin), *not* a render defect and *not* a global prior problem.
- **Encoder-free (from-scratch vision)** → reads *nothing* robustly; collapses to zero
  under pressure.
- **Hybrid (random vision)** → unstarted; would train *both* towers from scratch.

The variable that flips "reads well" vs "won't read" is **whether the vision embedder
is pretrained**, not whether the decoder is reused. Gemma's "learnability-limited"
audio verdict and encoder-free's "collapse-to-zero" point the same direction: **pixel-
reading capacity must be built into / pretrained on the vision side.** You can't cheaply
bolt a tiny from-scratch projector onto a strong LM and expect emergent OCR — especially
absent a task that forces per-sample pixel dependence (and even *with* one, H3 shows the
decoder finds a non-reading shortcut first).

## Implications for what's worth doing next

- **More encoder-free architecture escalation (MLP → conv stem)** — the pre-registered
  next step — might squeeze out marginally more reading, but the collapse dynamic looks
  fundamental to the from-scratch-embedder + strong-decoder pairing, not a capacity knob
  away. Low expected value. **[UPDATE 2026-07-26 — TESTED, NULL] The MLP rung was run
  (`configs/h3_randstr_mlp.yaml`, +30M non-linear connector, only-changed-variable vs H3).
  Eval replicated H3's plateau to the decimal (best 2.222 vs H3's 2.214) and the ablation
  (`data/eval/h3-ablation-h3-mlp.json`) showed the IDENTICAL collapse — aligned=permuted=
  blank 4.031, Δperm/Δblank 0.000%, tok-acc 32.3% aligned==blank bit-identical. Connector
  expressivity is RULED OUT: linear and MLP land in the same ignore-pixels basin, so the
  bottleneck is the optimization dynamic (decoder shortcut wins gradient from step 0),
  upstream of the embedder function class. A conv stem would very likely fail the same way.
  Option-1 is exhausted as a cheap fix.]**
- **The fully-random hybrid (#6)** removes the language-prior confound cleanly, but also
  removes the *one thing that makes Gemma read* (vision pretraining) and doubles the
  from-scratch training burden. Given encoder-free localized the bottleneck to the
  embedder side, a random-init hybrid faces an even harder version of the same problem —
  high compute, high risk of never grounding within budget.
- **The Gemma thread's open audio problem is the highest-value lever.** Reading already
  works for 3 modalities on that architecture and the render is proven lossless; the
  single unsolved thing is audio, and it's been narrowed to a *fixable* learnability
  issue (share/curriculum), per the E3 (audio-only probe) → E4 (rebalanced bounded run)
  path already sketched in CLAUDE.md.

**One caveat worth preserving:** encoder-free H1 is the *only* configuration in the whole
program where **audio grounded**. The shared-pixel-path idea isn't dead — it just needs a
*capable* vision embedder, not a from-scratch linear one. If the shared-path audio result
matters, the way to chase it is a pretrained vision embedder feeding the reused decoder,
not a tiny random projector.

**[UPDATE 2026-07-26 — this recommendation was TESTED and VALIDATED.]** Built the pretrained
path (`univi/hybrid/pretrained.py`): PRETRAINED Gemma-12B unified vision embedder → from-scratch
`Linear(3840→2048)` adapter → PRETRAINED Qwen3-1.7B, 3-group LR, no freeze. On the same
random-string OCR diagnostic that H3 failed at exact-zero, this **reads strongly** — Δperm
+114.7%, Δblank +90.8%, reading gain +22.2 pts (vs H3/H3-MLP ~0), and grounding *grew* with
training. The from-scratch embedder was the whole bottleneck; a pretrained vision front-end
feeding the reused Qwen decoder works. See the "Pretrained-vision hybrid" section of `CLAUDE.md`
and the `hybrid-pretrained-vision-findings` memory.

## Artifacts

- Ablation code (rebuilt): `scratchpad/h3_ablation.py`
- Ablation results: `data/eval/h3-ablation-{h3,h1-baseline}.json`
- Checkpoints: `data/checkpoints/h3-randstr-v0/{best,checkpoint-500/1000/1500,final}`
- W&B: `gshey7fa` (`h3-randstr-ocr-warmstart`, project `univi-encoder-free`)
- Full per-hypothesis detail: `CLAUDE.md` (encoder-free thread section) and the
  `encoder-free-vlm-findings` memory.
