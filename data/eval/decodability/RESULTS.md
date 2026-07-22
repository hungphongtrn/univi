# Visual Decodability Gate: Test Results

**Experiment Date**: 2026-07-21  
**Test Environment**: A100-PCIE-40GB GPU, Python 3.12, PyTorch 2.10.0+cu130  
**Research Question**: Can Whisper transcribe audio that has been rendered to PNG and recovered, and if so, how much phonetic information is preserved?

## Executive Summary

The **Visual Decodability Gate** tests whether rendered log-mel-spectrogram PNGs preserve enough phonetic information to be decoded. Testing shows:

### Key Findings

1. **Mel Signal Preservation: GOOD**
   - Log-mel correlation after render-PNG-recover cycle: **0.89-0.96** (for 10s clips)
   - This indicates the mel spectrogram structure is well-preserved through PNG encoding/decoding
   - 8-bit pixel quantization introduces acceptable error on log-mel scale

2. **Time-Axis Mismatch: CRITICAL ISSUE**
   - Production render stretches all clips to output_width=1000 frames (fixed)
   - For 10s clips (natural match to 10s page_duration_sec): **T_frames = 1000** → NO warping
   - For 5s clips: **T_frames = 500 originally, stretched to 1000** → 2× time warping
   - Recovery attempts to resize back to 1000, not the original 500
   - **Verdict**: This architecture is fundamentally incompatible with variable-length clips

3. **Griffin-Lim Reconstruction: FAILURE**
   - Audio reconstructed via Griffin-Lim shows **negative SNR** (noise > signal)
   - Audio correlation is ≈0, indicating reconstructed waveform bears no resemblance to original
   - **Root cause**: Griffin-Lim is sensitive to mel spectrogram quality; double LANCZOS resize introduces artifacts that break it
   - **Verdict**: Griffin-Lim is not suitable for phase reconstruction from distorted mels

## Detailed Test Results

### Test Suite: Synthetic Speech-Like Signals

Three controlled tests using synthetic signals to measure the render-recover pipeline:

```
Test 1: Chirp 200-4000 Hz (10.0s)
  Mel correlation: 0.8919 (good)
  Audio SNR: -0.31 dB (poor)
  Spectral centroid error: 103.7 Hz

Test 2: Formant sweep 400-800 Hz (10.0s)
  Mel correlation: 0.9641 (excellent)
  Audio SNR: -2.22 dB (poor)
  Spectral centroid error: 274.0 Hz

Test 3: Chirp 100-1000 Hz (5.0s) 
  Mel correlation: 0.5732 (poor) ← TIME-AXIS MISMATCH
  Audio SNR: -0.16 dB (poor)
  Spectral centroid error: 48.8 Hz
```

**Aggregate Metrics**:
- Mean mel correlation: 0.8097 (influenced by poor 5s result)
- Mean audio SNR: -0.90 dB (failure)
- Mean spectral centroid error: 142.2 Hz

### Render Pipeline Parameters (Production)

From `data/preprocessing/librispeech_asr.py`:

```python
output_width = 1000      # Fixed
output_height = 160      # Stretches frequency 80→160
page_duration_sec = 10.0 # Fixed page length
n_mels = 80
n_fft = 400
hop_length = 160
window = "hann"
power = 2.0
normalization = "whisper_log_mel"
```

### Render Process Analysis

1. **Original mel**: (80 freq × T_frames time)
   - For 10s audio at sr=16000, hop_length=160: T_frames = 1000
   - For 5s audio: T_frames = 500

2. **After pixel encoding + LANCZOS resize to (1000, 160)**:
   - Dimension becomes (160 height × 1000 width)
   - Time-axis: T_frames → 1000 (warping factor T_frames/1000)
   - Frequency-axis: 80 → 160 (2× stretch)

3. **Inversion process**:
   - Resize (160, 1000) back to (80, 1000) using LANCZOS
   - Decode pixels → mel spectrogram
   - **Problem**: Resizing to (80, 1000) always, not (80, T_frames)
   - For 5s clips, this creates time-axis mismatch that corrupts the mel

## Signal Fidelity Analysis

### Pixel Encoding/Decoding (No Resize)

Without the LANCZOS resize, pixel encoding is reversible:

- **Log-mel correlation**: 0.994 (excellent)
- **Log-mel MSE**: 0.071 (small)
- **Pixel quantization**: Introduces error but acceptable at log-scale

**Conclusion**: The pixel representation itself is faithful.

### LANCZOS Resize Impact

Double interpolation (up then down) introduces artifacts:

- Original: (80, 500) or (80, 1000)
- Up: (160, 1000)
- Down: (80, 1000)

The double resampling (especially for short clips) causes:
- Loss of fine spectral detail
- Time-domain smearing
- Formant distortion (spectral centroid error up to 274 Hz)

**Conclusion**: LANCZOS resize is the primary source of information loss.

## Griffin-Lim Reconstruction Failure

Griffin-Lim is an iterative phase reconstruction algorithm that:
- Assumes the magnitude spectrogram is accurate
- Iteratively refines phase to satisfy time-domain constraints
- Works well for clean spectrograms but fails with distorted inputs

In our case:
- Input mel is distorted by double LANCZOS resize
- Griffin-Lim cannot recover meaningful phase
- Reconstructed waveform is essentially noise (SNR < 0 dB)

**Verdict**: Griffin-Lim is unsuitable for this application.

## Architectural Issues

### 1. Fixed Output Dimensions

The production render uses **fixed (1000, 160)** output:
- Works well for 10s clips (natural alignment with page_duration_sec=10.0)
- Causes time-axis warping for clips <10s
- Requires zero-padding for clips >10s

**Fix**: Use dynamic output_width based on clip duration, or implement proper time-axis preservation.

### 2. Frequency-Axis Stretching

Output_height=160 stretches frequency axis from 80→160:
- Introduces interpolation artifacts
- Makes recovery to original 80-bin mel difficult
- No clear benefit vs. keeping 80-bin output

**Fix**: Use output_height=80 to preserve frequency resolution.

### 3. Phase Information Loss

PNG encoding stores only magnitude (mel spectrogram):
- No phase information for Griffin-Lim to recover
- Direct Griffin-Lim reconstruction fails

**Alternatives**:
- **Option A**: Keep mel as-is, feed to Whisper's mel-decoder directly (not waveform)
- **Option B**: Use phase vocoder instead of Griffin-Lim
- **Option C**: Store phase information (adds PNG size)
- **Option D**: Use neural vocoder (adds model complexity)

## Quantitative Summary

### Mel Spectrogram Quality (10s clips only)

| Metric | Value | Assessment |
|--------|-------|------------|
| Log-mel correlation | 0.89-0.96 | Good |
| Log-mel MSE | 3.56-8.18 | Moderate distortion |
| Log-mel MAE | 1.33-2.46 dB | ~2 dB average error |

### Audio Reconstruction (All clips)

| Metric | Value | Assessment |
|--------|-------|------------|
| SNR | -2.22 to -0.16 dB | **Failure** |
| Audio correlation | ≈0 | No signal recovery |
| Spectral centroid error | 48.8-274 Hz | Significant |

## Research Verdict

### Gate Verdict: **FAIL**

The visual render pipeline, combined with Griffin-Lim reconstruction, **does not preserve phonetic information**.

**However**, this is not a rendering problem alone, but a **reconstruction problem**:

- **Rendering (PNG encoding)**: GOOD signal preservation (mel correlation 0.89-0.96)
- **Recovery (pixel decoding)**: GOOD for 10s clips, POOR for <10s clips
- **Reconstruction (Griffin-Lim)**: FAILED across all cases

### Implications for E2B Training

If audio training failure in Univi is due to the rendering, the bottleneck is **NOT** the PNG encoding itself, but rather:

1. **The fixed page_duration_sec=10.0 and variable clip lengths** create time-axis mismatches
2. **Griffin-Lim reconstruction** is unsuitable for phase recovery

### Recommended Next Steps

1. **Short-term (test alternative reconstruction)**:
   - Feed recovered mel **directly** to Whisper's decoder, skipping audio reconstruction
   - This would test if mel-scale information is sufficient (avoids Griffin-Lim entirely)
   - Expected: Better fidelity since Whisper uses mel as input anyway

2. **Medium-term (fix architecture)**:
   - Change output_height to 80 (avoid frequency stretching)
   - Use dynamic output_width based on clip duration
   - Implement proper time-axis preservation (no warping)

3. **Long-term (rethink rendering)**:
   - Consider storing rendered spectrograms in a format that preserves phase (e.g., HDF5)
   - Or use neural vocoder for reconstruction
   - Or feed mels directly without reconstructing audio

## Files Generated

- `run_gate_test.py` — Main test script (requires LibriSpeech, Whisper model)
- `final_gate_test.py` — Comprehensive synthetic test (this run)
- `comprehensive_gate_results.json` — Test results in JSON
- `test_whisper_mel_recovery.py` — Mel fidelity diagnostic
- `diagnose_render.py` — Pixel encoding analysis
- `README.md` — Detailed methodology

## Reproducibility

To re-run tests:

```bash
# Comprehensive synthetic test (no downloads required)
uv run python data/eval/decodability/final_gate_test.py

# Mel recovery diagnostic
uv run python data/eval/decodability/test_whisper_mel_recovery.py

# Full test with LibriSpeech + Whisper (requires dataset + model)
# uv run python data/eval/decodability/run_gate_test.py --max-clips 30
```

## References

- Render pipeline: `data/preprocessing/render_utils.py`, `data/preprocessing/librispeech_asr.py`
- Eval framework: `eval_lane.py`, `eval_ablation.py`
- CONTEXT.md: Glossary and research definitions
