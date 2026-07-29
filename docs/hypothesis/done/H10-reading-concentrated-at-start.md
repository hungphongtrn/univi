# H10 — Reading is spread evenly across the answer

**Status:** DONE · **Verdict: REFUTED** (2026-07-28) · Related: [H09](H09-balanced-mixture-prevents-drowning.md), [H11](H11-position-decay-is-prior-induced.md), [H18](../todo/H18-no-learned-scan.md)

## Claim

The small aggregate reading gains in [H09](H09-balanced-mixture-prevents-drowning.md) (+0.4 to +2.0
pts) reflect a little reading spread evenly over the answer.

## Setup

`scratchpad/hybrid_4lane_position_decay.py` — teacher-forced token accuracy and CE under aligned vs
blank image, **binned by answer-token position**. n=150 rows/lane, `hybrid-4lane-v0/final`.

## Result — reading gain (pts) by position

| position | fineweb | librispeech | densefusion | smoltalk |
|---|---|---|---|---|
| 0 | **+70.00** | **−12.67** | 0.00 | **+18.67** |
| 1 | +6.67 | +2.00 | +0.67 | +8.67 |
| 2–4 | +3.11 | +2.67 | +5.33 | +6.00 |
| 5–9 | +2.27 | +3.56 | +3.20 | +2.41 |
| 10–19 | +0.13 | **+4.60** | +1.13 | +1.32 |
| 50–99 | +0.65 | −0.48 | +0.60 | +0.16 |
| 400+ | +1.07 | — | 0.00 | +0.04 |

On fineweb the **first answer token is 70.67% correct with the image and 0.67% without** (CE 1.207
vs 7.261). Reading then collapses within ten tokens and never recovers beyond ~+1 pt.

## Learning

- The aggregate gain was hiding an extreme distribution: the model reads the **start** of the image
  almost perfectly and essentially nothing after.
- The representation therefore **contains high-fidelity detail** for at least one region — which
  argues against a naive "the pixels are mush" bandwidth story.
- librispeech is the exception in shape: **negative** at position 0 (no phonetic onset reading), with
  a broad weak peak at 10–19 — consistent with the duration-conditioning finding in
  [H09](H09-balanced-mixture-prevents-drowning.md).
- The gain tracks prior-unexplained loss: at position 0 blank CE is 7.26 and the model captures ~83%
  of the gap; at position 400+ blank CE is 2.51 and it captures ~4%. **Reading is purchased where,
  and only where, the prior fails.**
- This profile is predicted by *both* a bandwidth limit and prior competition — see
  [H11](H11-position-decay-is-prior-induced.md) for the control that separates them.

## Artifacts

`data/eval/hybrid-4lane-position-decay-4lane-final.json`
