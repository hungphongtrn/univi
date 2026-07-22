#!/usr/bin/env python3
"""
Visual Decodability Gate: measure whether rendered log-mel spectrograms preserve phonetic info.

EXPERIMENTAL DESIGN:
- Ceiling: Whisper WER on raw audio (what Whisper can achieve)
- Gate: Whisper WER on audio recovered from rendered PNG (what survives the render+invert)
- Gap = WER_gate - WER_ceiling is the information destroyed by rendering

If gap is small, render preserves phonetic info and audio training failure is data-limited.
If gap is large, render destroys signal and rendering must be redesigned.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import librosa
from PIL import Image
from datasets import load_dataset

# Deferred GPU imports
_whisper_pipeline = None


def _get_whisper_pipeline():
    global _whisper_pipeline
    if _whisper_pipeline is None:
        from transformers import pipeline
        _whisper_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base")
    return _whisper_pipeline


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute word error rate (0-1 scale)."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # Simple edit distance for WER (Levenshtein at word level)
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0

    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n] / m


def transcribe_audio_array(audio_array: np.ndarray, sr: int = 16000) -> str:
    """Transcribe audio array using Whisper."""
    pipeline = _get_whisper_pipeline()
    try:
        result = pipeline({"sampling_rate": sr, "raw": audio_array})
        return result["text"]
    except Exception as e:
        print(f"Whisper transcription error: {e}")
        return ""


def render_log_mel_spectrogram_to_png(
    audio_array: np.ndarray,
    sr: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    window: str = "hann",
    power: float = 2.0,
    output_width: int = 1000,
    output_height: int = 160,
) -> Image.Image:
    """Render audio to log-mel spectrogram PNG (matching production params)."""
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_array,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        power=power,
    )

    # Trim to expected frames
    expected_frames = max(1, len(audio_array) // hop_length)
    mel_spec = mel_spec[:, :expected_frames]

    # Apply Whisper normalization
    log_mel = np.log10(np.maximum(mel_spec, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    whisper_values = (log_mel + 4.0) / 4.0
    normalized = np.clip((whisper_values + 1.0) / 2.0, 0.0, 1.0)

    # Convert to pixel values (inverted: dark = high energy)
    pixels = np.rint(255 * (1.0 - normalized)).astype(np.uint8)

    # Create image and resize to output size
    img = Image.fromarray(pixels, mode="L")
    img = img.resize((output_width, output_height), Image.LANCZOS)

    return img


def invert_png_to_mel(
    img: Image.Image,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    sr: int = 16000,
) -> np.ndarray:
    """
    Invert PNG back to log-mel (80 x T).

    Pipeline:
    1. Resize PNG back to (80, T_frames) where T_frames = duration / (hop_length / sr)
    2. Pixel -> normalized log-mel: normalized = 1 - pixel/255
    3. Normalized -> whisper_values: whisper_values = normalized * 2 - 1
    4. Whisper values -> log_mel: log_mel = whisper_values * 4 - 4
    5. Log-mel -> mel: mel = 10^log_mel
    """
    # Convert to grayscale if needed
    if img.mode != "L":
        img = img.convert("L")

    # Get pixel array
    pixels = np.array(img, dtype=np.float32)

    # Resize back to (n_mels, T_frames)
    # We don't know exact T_frames, so infer from 10s default page duration
    # page_duration_sec = 10.0, so T_frames = (10 * sr) / hop_length = (10 * 16000) / 160 = 1000
    pil_img = Image.fromarray(pixels.astype(np.uint8), mode="L")

    # Estimate frame count from duration. For now, use the production default.
    # The PNG was created from 10s page at hop_length=160, so T_frames ~ 1000
    # But be flexible and use the actual resized shape
    T_estimated = 1000  # Default for 10s page
    pil_resized = pil_img.resize((T_estimated, n_mels), Image.LANCZOS)

    pixels_resized = np.array(pil_resized, dtype=np.float32)

    # Invert pipeline
    normalized = 1.0 - pixels_resized / 255.0
    whisper_values = normalized * 2.0 - 1.0
    log_mel = whisper_values * 4.0 - 4.0
    mel = 10.0 ** log_mel

    # Clamp to avoid numerical issues
    mel = np.maximum(mel, 1e-10)

    # mel is now (n_mels, T) = (80, 1000)
    return mel, T_estimated


def reconstruct_audio_from_mel(
    mel: np.ndarray,
    sr: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    window: str = "hann",
) -> np.ndarray:
    """Reconstruct audio from mel spectrogram using Griffin-Lim."""
    # Convert mel back to power spectrogram
    # mel = librosa.feature.melspectrogram(...), so we need the inverse
    inverse_mel_filter = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=mel.shape[0],
    )
    # Solve: mel = inverse_mel_filter @ power_spec
    # Use least-squares: power_spec = pinv(inverse_mel_filter) @ mel
    power_spec = np.linalg.pinv(inverse_mel_filter) @ mel

    # Apply Griffin-Lim to recover phase
    audio = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        n_iter=64,  # Standard Griffin-Lim iterations
    )

    return audio


@dataclass
class DecodabilityResult:
    """Single clip result."""
    clip_id: str
    duration_sec: float
    n_frames: int
    aspect_ratio_distortion: float  # T_frames / output_width

    # Ceiling (raw audio)
    reference_text: str
    ceiling_transcript: str
    ceiling_wer: float

    # Gate (rendered + inverted)
    gate_transcript: str
    gate_wer: float

    # Gap
    wer_gap: float


def run_decodability_test(
    max_clips: int = 30,
    output_dir: Path | None = None,
    use_griffin_lim: bool = True,
) -> dict:
    """Run the Visual Decodability Gate test."""
    if output_dir is None:
        output_dir = Path("/workspace/univi/data/eval/decodability")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading LibriSpeech clean test split (max {max_clips} clips)...")
    ds = load_dataset(
        "openslr/librispeech_asr",
        "clean",
        split="test",
    )

    # Use first N clips
    if len(ds) > max_clips:
        ds = ds.select(range(max_clips))

    print(f"Loaded {len(ds)} clips")

    results: list[DecodabilityResult] = []

    for idx, row in enumerate(ds):
        print(f"\n[{idx + 1}/{len(ds)}] Processing clip {row.get('id', idx)}")

        # Decode audio
        audio_dict = row["audio"]
        audio_array = np.array(audio_dict["array"], dtype=np.float32)
        audio_sr = audio_dict["sampling_rate"]
        duration_sec = len(audio_array) / audio_sr

        reference_text = row["text"]
        print(f"  Duration: {duration_sec:.2f}s | Text: {reference_text[:60]}...")

        # CEILING: Transcribe raw audio
        print("  [CEILING] Transcribing raw audio...")
        ceiling_transcript = transcribe_audio_array(audio_array, sr=audio_sr)
        ceiling_wer = compute_wer(reference_text, ceiling_transcript)
        print(f"  Ceiling WER: {ceiling_wer:.4f}")
        print(f"  Ceiling transcript: {ceiling_transcript[:60]}...")

        # GATE: Render -> Invert -> Transcribe
        print("  [GATE] Rendering to PNG...")
        png = render_log_mel_spectrogram_to_png(audio_array, sr=audio_sr)

        # Compute aspect ratio distortion
        T_frames_original = max(1, len(audio_array) // 160)  # hop_length=160
        output_width = png.width
        aspect_ratio_distortion = T_frames_original / output_width

        print("  Inverting PNG to mel...")
        mel, T_estimated = invert_png_to_mel(png, sr=audio_sr)

        print("  Reconstructing audio (Griffin-Lim)...")
        reconstructed_audio = reconstruct_audio_from_mel(mel, sr=audio_sr)

        # Trim to match original length if needed
        if len(reconstructed_audio) > len(audio_array):
            reconstructed_audio = reconstructed_audio[:len(audio_array)]
        elif len(reconstructed_audio) < len(audio_array):
            # Pad with zeros
            reconstructed_audio = np.pad(
                reconstructed_audio,
                (0, len(audio_array) - len(reconstructed_audio))
            )

        print("  Transcribing reconstructed audio...")
        gate_transcript = transcribe_audio_array(reconstructed_audio, sr=audio_sr)
        gate_wer = compute_wer(reference_text, gate_transcript)
        print(f"  Gate WER: {gate_wer:.4f}")
        print(f"  Gate transcript: {gate_transcript[:60]}...")

        wer_gap = gate_wer - ceiling_wer
        print(f"  WER Gap: {wer_gap:+.4f}")

        result = DecodabilityResult(
            clip_id=str(row.get("id", idx)),
            duration_sec=duration_sec,
            n_frames=T_frames_original,
            aspect_ratio_distortion=aspect_ratio_distortion,
            reference_text=reference_text,
            ceiling_transcript=ceiling_transcript,
            ceiling_wer=ceiling_wer,
            gate_transcript=gate_transcript,
            gate_wer=gate_wer,
            wer_gap=wer_gap,
        )
        results.append(result)

    # Aggregate results
    ceiling_wers = [r.ceiling_wer for r in results]
    gate_wers = [r.gate_wer for r in results]
    gaps = [r.wer_gap for r in results]
    durations = [r.duration_sec for r in results]
    distortions = [r.aspect_ratio_distortion for r in results]

    summary = {
        "test_type": "visual_decodability_gate",
        "render_method": "librispeech_production_params",
        "render_params": {
            "output_width": 1000,
            "output_height": 160,
            "page_duration_sec": 10.0,
            "n_mels": 80,
            "n_fft": 400,
            "hop_length": 160,
            "window": "hann",
            "power": 2.0,
            "normalization": "whisper_log_mel",
        },
        "inversion_method": "griffin_lim" if use_griffin_lim else "unknown",
        "num_clips": len(results),
        "ceiling_stats": {
            "mean_wer": float(np.mean(ceiling_wers)),
            "median_wer": float(np.median(ceiling_wers)),
            "min_wer": float(np.min(ceiling_wers)),
            "max_wer": float(np.max(ceiling_wers)),
            "std_wer": float(np.std(ceiling_wers)),
        },
        "gate_stats": {
            "mean_wer": float(np.mean(gate_wers)),
            "median_wer": float(np.median(gate_wers)),
            "min_wer": float(np.min(gate_wers)),
            "max_wer": float(np.max(gate_wers)),
            "std_wer": float(np.std(gate_wers)),
        },
        "gap_stats": {
            "mean_gap": float(np.mean(gaps)),
            "median_gap": float(np.median(gaps)),
            "min_gap": float(np.min(gaps)),
            "max_gap": float(np.max(gaps)),
            "std_gap": float(np.std(gaps)),
        },
        "audio_characteristics": {
            "mean_duration_sec": float(np.mean(durations)),
            "median_duration_sec": float(np.median(durations)),
            "mean_n_frames": float(np.mean([r.n_frames for r in results])),
            "mean_aspect_ratio_distortion": float(np.mean(distortions)),
        },
        "verdict": _compute_verdict(np.mean(gaps), np.std(gaps)),
        "individual_results": [
            {
                "clip_id": r.clip_id,
                "duration_sec": r.duration_sec,
                "n_frames": r.n_frames,
                "aspect_ratio_distortion": round(r.aspect_ratio_distortion, 4),
                "ceiling_wer": round(r.ceiling_wer, 4),
                "gate_wer": round(r.gate_wer, 4),
                "wer_gap": round(r.wer_gap, 4),
                "reference_text": r.reference_text[:100],
                "ceiling_transcript": r.ceiling_transcript[:100],
                "gate_transcript": r.gate_transcript[:100],
            }
            for r in results
        ],
    }

    return summary


def _compute_verdict(mean_gap: float, std_gap: float) -> str:
    """Compute the research verdict based on WER gap."""
    if mean_gap < 0.05:
        return (
            "GATE PASS (phonetic info preserved): Render preserves phonetic signal. "
            "Audio training failure is data/learnability-limited. Scaling data justified."
        )
    elif mean_gap < 0.15:
        return (
            "GATE PARTIAL (moderate information loss): Some phonetic info lost but "
            "transcription still feasible. Consider higher-resolution render."
        )
    else:
        return (
            "GATE FAIL (phonetic info destroyed): Render destroys signal. "
            "Rendering must be redesigned (higher time resolution, no square-warp) before more training."
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-clips", type=int, default=30, help="Max clips to test")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--no-griffin-lim", action="store_true", help="Disable Griffin-Lim (not yet supported)")
    args = parser.parse_args()

    results = run_decodability_test(
        max_clips=args.max_clips,
        output_dir=args.output_dir,
        use_griffin_lim=not args.no_griffin_lim,
    )

    # Save results
    output_path = (args.output_dir or Path("/workspace/univi/data/eval/decodability")) / "results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Clips tested: {results['num_clips']}")
    print(f"\nCeiling (raw audio):")
    print(f"  Mean WER: {results['ceiling_stats']['mean_wer']:.4f}")
    print(f"  Median WER: {results['ceiling_stats']['median_wer']:.4f}")
    print(f"\nGate (rendered + inverted):")
    print(f"  Mean WER: {results['gate_stats']['mean_wer']:.4f}")
    print(f"  Median WER: {results['gate_stats']['median_wer']:.4f}")
    print(f"\nWER Gap (gate - ceiling):")
    print(f"  Mean gap: {results['gap_stats']['mean_gap']:+.4f}")
    print(f"  Median gap: {results['gap_stats']['median_gap']:+.4f}")
    print(f"  Std gap: {results['gap_stats']['std_gap']:.4f}")
    print(f"\nAudio Characteristics:")
    print(f"  Mean duration: {results['audio_characteristics']['mean_duration_sec']:.2f}s")
    print(f"  Mean n_frames: {results['audio_characteristics']['mean_n_frames']:.0f}")
    print(f"  Mean aspect ratio distortion (T_frames/width): {results['audio_characteristics']['mean_aspect_ratio_distortion']:.4f}")
    print(f"\nVerdict: {results['verdict']}")
    print(f"\nFull results saved to: {output_path}")
