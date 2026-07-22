#!/usr/bin/env python3
"""
Visual Decodability Gate: Test whether Whisper can transcribe rendered spectrograms.

This is the actual end-to-end test:
1. Load real LibriSpeech clips
2. Ceiling WER: Whisper on raw audio
3. Gate WER: Whisper on (render -> recover mel -> Griffin-Lim audio)
4. Report the gap
"""

from __future__ import annotations

import json
import numpy as np
import librosa
from PIL import Image
from pathlib import Path
from dataclasses import dataclass

import sys
sys.path.insert(0, "/workspace/univi")
from data.preprocessing.render_utils import _whisper_log_mel

# Deferred GPU import
_whisper_pipeline = None


def _get_whisper_pipeline():
    global _whisper_pipeline
    if _whisper_pipeline is None:
        try:
            from transformers import pipeline
            print("Loading Whisper model (base)...")
            _whisper_pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                device=0,  # GPU
            )
            print("Whisper model loaded")
        except Exception as e:
            print(f"Error loading Whisper: {e}")
            return None
    return _whisper_pipeline


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute word error rate."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

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


def transcribe_audio(audio_array: np.ndarray, sr: int = 16000) -> str:
    """Transcribe audio using Whisper."""
    pipeline = _get_whisper_pipeline()
    if pipeline is None:
        return ""

    try:
        result = pipeline({"sampling_rate": sr, "raw": audio_array})
        return result.get("text", "").strip()
    except Exception as e:
        print(f"  Transcription error: {e}")
        return ""


def render_to_png(audio_array: np.ndarray, sr: int = 16000) -> Image.Image:
    """Render audio to PNG using production parameters."""
    # Compute mel
    mel_spec = librosa.feature.melspectrogram(
        y=audio_array,
        sr=sr,
        n_mels=80,
        n_fft=400,
        hop_length=160,
        window="hann",
        power=2.0,
    )

    # Trim to expected frames
    expected_frames = max(1, len(audio_array) // 160)
    mel_spec = mel_spec[:, :expected_frames]

    # Apply Whisper normalization
    log_mel = np.log10(np.maximum(mel_spec, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    whisper_values = (log_mel + 4.0) / 4.0
    normalized = np.clip((whisper_values + 1.0) / 2.0, 0.0, 1.0)

    # Encode to pixels
    pixels = np.rint(255 * (1.0 - normalized)).astype(np.uint8)

    # Create image and resize to production size (1000x160)
    img = Image.fromarray(pixels, mode="L")
    img = img.resize((1000, 160), Image.LANCZOS)

    return img


def recover_mel_from_png(img: Image.Image, sr: int = 16000) -> np.ndarray:
    """Recover mel spectrogram from PNG."""
    if img.mode != "L":
        img = img.convert("L")

    pixels = np.array(img, dtype=np.float32)

    # Resize back to (80, T_frames)
    # Estimate T_frames based on default 10s page duration at hop_length=160
    T_frames = max(1000, (sr * 10) // 160)

    img_back = Image.fromarray(pixels.astype(np.uint8), mode="L")
    img_back = img_back.resize((T_frames, 80), Image.LANCZOS)
    pixels_recovered = np.array(img_back, dtype=np.float32)

    # Invert normalization
    normalized = 1.0 - pixels_recovered / 255.0
    whisper_values = normalized * 2.0 - 1.0
    log_mel = whisper_values * 4.0 - 4.0
    mel = 10.0 ** log_mel

    return np.maximum(mel, 1e-10)


def griffin_lim_reconstruct(mel: np.ndarray, sr: int = 16000, n_iter: int = 64) -> np.ndarray:
    """Reconstruct audio from mel using Griffin-Lim."""
    try:
        audio = librosa.feature.inverse.mel_to_audio(
            mel,
            sr=sr,
            n_fft=400,
            hop_length=160,
            window="hann",
            n_iter=n_iter,
        )
        return audio
    except Exception as e:
        print(f"  Griffin-Lim error: {e}")
        return np.zeros(len(mel[0]) * 160, dtype=np.float32)


@dataclass
class ClipResult:
    """Result for a single clip."""
    clip_id: str
    duration_sec: float
    reference_text: str
    ceiling_transcript: str
    ceiling_wer: float
    gate_transcript: str
    gate_wer: float
    wer_gap: float


def run_gate_test(max_clips: int = 10) -> dict:
    """Run the decodability gate test."""
    from datasets import load_dataset

    print("="*80)
    print(f"Visual Decodability Gate Test ({max_clips} clips)")
    print("="*80)

    print("Loading LibriSpeech clean test set...")
    try:
        ds = load_dataset(
            "openslr/librispeech_asr",
            "clean",
            split="test",
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {}

    if len(ds) > max_clips:
        ds = ds.select(range(max_clips))

    print(f"Loaded {len(ds)} clips\n")

    results: list[ClipResult] = []

    for idx, row in enumerate(ds):
        clip_id = str(row.get("id", idx))
        reference_text = row["text"]

        # Decode audio
        audio_dict = row["audio"]
        audio_array = np.array(audio_dict["array"], dtype=np.float32)
        sr = audio_dict["sampling_rate"]
        duration = len(audio_array) / sr

        print(f"[{idx + 1}/{len(ds)}] Clip {clip_id} ({duration:.2f}s)")
        print(f"  Reference: {reference_text[:80]}")

        # CEILING: Transcribe raw audio
        print(f"  [CEILING] Transcribing raw audio...")
        ceiling_text = transcribe_audio(audio_array, sr=sr)
        ceiling_wer = compute_wer(reference_text, ceiling_text)
        print(f"  Ceiling WER: {ceiling_wer:.4f}")
        print(f"  Transcribed: {ceiling_text[:80]}")

        # GATE: Render -> Recover -> Griffin-Lim -> Transcribe
        print(f"  [GATE] Rendering to PNG...")
        png = render_to_png(audio_array, sr=sr)

        print(f"  Recovering mel from PNG...")
        mel = recover_mel_from_png(png, sr=sr)

        print(f"  Reconstructing audio (Griffin-Lim)...")
        reconstructed = griffin_lim_reconstruct(mel, sr=sr, n_iter=64)

        # Trim to match original length
        if len(reconstructed) > len(audio_array):
            reconstructed = reconstructed[:len(audio_array)]
        elif len(reconstructed) < len(audio_array):
            reconstructed = np.pad(reconstructed, (0, len(audio_array) - len(reconstructed)))

        print(f"  Transcribing reconstructed audio...")
        gate_text = transcribe_audio(reconstructed, sr=sr)
        gate_wer = compute_wer(reference_text, gate_text)
        wer_gap = gate_wer - ceiling_wer

        print(f"  Gate WER: {gate_wer:.4f}")
        print(f"  WER gap: {wer_gap:+.4f}")
        print(f"  Transcribed: {gate_text[:80]}\n")

        result = ClipResult(
            clip_id=clip_id,
            duration_sec=duration,
            reference_text=reference_text,
            ceiling_transcript=ceiling_text,
            ceiling_wer=ceiling_wer,
            gate_transcript=gate_text,
            gate_wer=gate_wer,
            wer_gap=wer_gap,
        )
        results.append(result)

    # Aggregate
    ceiling_wers = [r.ceiling_wer for r in results]
    gate_wers = [r.gate_wer for r in results]
    gaps = [r.wer_gap for r in results]

    summary = {
        "test_type": "visual_decodability_gate",
        "num_clips": len(results),
        "render_method": "librispeech_production",
        "render_params": {
            "output_width": 1000,
            "output_height": 160,
            "n_mels": 80,
            "n_fft": 400,
            "hop_length": 160,
            "window": "hann",
            "power": 2.0,
        },
        "inversion_method": "lanczos_resize_griffin_lim",
        "ceiling_stats": {
            "mean_wer": float(np.mean(ceiling_wers)) if ceiling_wers else 0,
            "median_wer": float(np.median(ceiling_wers)) if ceiling_wers else 0,
            "std_wer": float(np.std(ceiling_wers)) if ceiling_wers else 0,
        },
        "gate_stats": {
            "mean_wer": float(np.mean(gate_wers)) if gate_wers else 0,
            "median_wer": float(np.median(gate_wers)) if gate_wers else 0,
            "std_wer": float(np.std(gate_wers)) if gate_wers else 0,
        },
        "gap_stats": {
            "mean_gap": float(np.mean(gaps)) if gaps else 0,
            "median_gap": float(np.median(gaps)) if gaps else 0,
            "std_gap": float(np.std(gaps)) if gaps else 0,
        },
        "results": [
            {
                "clip_id": r.clip_id,
                "duration_sec": round(r.duration_sec, 2),
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

    # Compute verdict
    if ceiling_wers and gate_wers:
        mean_gap = np.mean(gaps)
        summary["verdict"] = _compute_verdict(mean_gap)
    else:
        summary["verdict"] = "INCONCLUSIVE (no results)"

    return summary


def _compute_verdict(mean_gap: float) -> str:
    """Compute the research verdict."""
    if mean_gap < 0.05:
        return "GATE PASS: Render preserves phonetic info (gap < 5%). Audio failure is data-limited."
    elif mean_gap < 0.15:
        return "GATE PARTIAL: Moderate info loss (5-15%). Consider higher-resolution render."
    else:
        return "GATE FAIL: Render destroys signal (gap > 15%). Rendering must be redesigned."


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-clips", type=int, default=10)
    args = parser.parse_args()

    summary = run_gate_test(max_clips=args.max_clips)

    # Save results
    output_dir = Path("/workspace/univi/data/eval/decodability")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "gate_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Clips tested: {summary['num_clips']}")
    print(f"\nCeiling (raw audio):")
    print(f"  Mean WER: {summary['ceiling_stats']['mean_wer']:.4f}")
    print(f"  Median WER: {summary['ceiling_stats']['median_wer']:.4f}")
    print(f"\nGate (rendered + recovered):")
    print(f"  Mean WER: {summary['gate_stats']['mean_wer']:.4f}")
    print(f"  Median WER: {summary['gate_stats']['median_wer']:.4f}")
    print(f"\nWER Gap:")
    print(f"  Mean gap: {summary['gap_stats']['mean_gap']:+.4f}")
    print(f"  Median gap: {summary['gap_stats']['median_gap']:+.4f}")
    print(f"  Std dev: {summary['gap_stats']['std_gap']:.4f}")
    print(f"\nVerdict: {summary['verdict']}")
    print(f"\nFull results: {results_path}")
    print("="*80)
