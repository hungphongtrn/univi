# Visual Decodability Gate

**Purpose**: Determine whether Univi's rendered log-mel-spectrogram PNGs preserve enough phonetic information to be transcribed, independent of the E2B model.

## Research Question

Can a strong ASR model (Whisper) read spectrograms AFTER they have been:
1. Rendered to PNG using production parameters
2. Recovered from the PNG pixels
3. Reconstructed to audio via Griffin-Lim

**Decision Rule**:
- **GATE PASS** (WER gap < 5%): Render preserves phonetic info → audio training failure is data-limited
- **GATE PARTIAL** (5-15%): Moderate info loss → consider higher-resolution render
- **GATE FAIL** (> 15%): Render destroys signal → rendering must be redesigned

## Rendering Pipeline

Production parameters from `data/preprocessing/librispeech_asr.py`:

```python
output_width=1000
output_height=160
page_duration_sec=10.0
n_mels=80
n_fft=400
hop_length=160
window="hann"
power=2.0
normalization="whisper_log_mel"
```

### Render → PNG Process
1. Audio (sr=16kHz) → mel spectrogram (80 × T_frames)
2. Apply Whisper log-mel normalization
3. Encode to pixels (grayscale, inverted: dark=high energy)
4. **LANCZOS resize** to 1000×160 (frequency 80→160, time T_frames→1000)

### PNG → Audio Recovery
1. Resize PNG pixels back to (1000, 80)
2. Decode pixels → normalized log-mel → mel spectrogram
3. **Griffin-Lim** with n_iter=64 to reconstruct audio
4. Feed to Whisper for transcription

## Aspect Ratio Distortion

For a typical 5-10 second clip:
- T_frames = duration × 16000 / 160 = duration × 100
- For 10s: T_frames ≈ 1000 (no time-axis stretch)
- For 5s: T_frames ≈ 500 (time-axis stretched 2×)

The frequency axis is always stretched 80→160 (2×).

## Files

- `run_gate_test.py` — Main test script; runs Whisper ceiling & gate WER
- `probe.py` — Original end-to-end probe (WIP: requires full dataset + Whisper)
- `probe_simple.py` — Simpler render-invert-reconstruct cycle test
- `test_whisper_mel_recovery.py` — Validates mel recovery fidelity before Griffin-Lim
- `diagnose_render.py` — Diagnostic script to isolate pixel encoding vs. time-axis effects
- `gate_results.json` — Test results (clip-by-clip WER and aggregate statistics)

## Expected Results

- **Ceiling WER** (raw audio + Whisper): ~3-5% on LibriSpeech test.clean
- **Gate WER** (render→PNG→recover→audio→Whisper): TBD, measure the gap
- **Gap**: The information destroyed by rendering

## Technical Notes

### Pixel Encoding Fidelity
- 8-bit quantization introduces error, but is lossless at log-mel scale
- Log-mel correlation after render-recover cycle: ~0.98 (high)
- Log-mel MSE after double LANCZOS resize: ~26 (non-negligible distortion)

### Griffin-Lim Reconstruction
- Uses 64 iterations (standard)
- Reconstructs audio from mel without access to phase
- May introduce harmonic artifacts but captures envelope/formants

### Time-Axis Warping
The LANCZOS resize from (80 T_frames) → (160, 1000) → (80, 1000) introduces:
- Frequency stretching: 80→160→80 (2× up, 2× down, compound interpolation error)
- Time stretching: T_frames→1000→T_frames (artifact-prone for clips < 10s)

The worst case is short clips (e.g., 5s), where time axis is warped 2×.

## Running Tests

```bash
# 10 clips (quick test)
uv run python data/eval/decodability/run_gate_test.py --max-clips 10

# 30 clips (comprehensive)
uv run python data/eval/decodability/run_gate_test.py --max-clips 30

# Render-invert cycle validation
uv run python data/eval/decodability/test_whisper_mel_recovery.py

# Pixel encoding diagnostic
uv run python data/eval/decodability/diagnose_render.py
```

## Results Format

`gate_results.json`:
```json
{
  "test_type": "visual_decodability_gate",
  "num_clips": 10,
  "render_params": { ... },
  "ceiling_stats": {
    "mean_wer": 0.0450,
    "median_wer": 0.0400,
    "std_wer": 0.0200
  },
  "gate_stats": {
    "mean_wer": 0.2500,
    "median_wer": 0.2000,
    "std_wer": 0.1500
  },
  "gap_stats": {
    "mean_gap": 0.2050,
    "median_gap": 0.1600,
    "std_gap": 0.1400
  },
  "verdict": "GATE FAIL: Render destroys signal...",
  "results": [
    {
      "clip_id": "...",
      "duration_sec": 5.2,
      "ceiling_wer": 0.05,
      "gate_wer": 0.25,
      "wer_gap": 0.20,
      "reference_text": "...",
      "ceiling_transcript": "...",
      "gate_transcript": "..."
    },
    ...
  ]
}
```

## Related Files

- Rendering: `data/preprocessing/render_utils.py`, `data/preprocessing/librispeech_asr.py`
- Training: `univi/train.py`, `univi/trainer.py`
- Eval lane (image vs native): `eval_lane.py`
- Ablation (permutation control): `eval_ablation.py`
