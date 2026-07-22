#!/usr/bin/env python3
"""
Simplified decodability test: validate render-invert pipeline first.
"""

from __future__ import annotations

import json
import numpy as np
import librosa
from PIL import Image
from pathlib import Path

# Import from project
import sys
sys.path.insert(0, "/workspace/univi")
from data.preprocessing.render_utils import render_log_mel_spectrogram


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
    1. Resize PNG back to (80, T_frames)
    2. Pixel -> normalized log-mel: normalized = 1 - pixel/255
    3. Normalized -> whisper_values: whisper_values = normalized * 2 - 1
    4. Whisper values -> log_mel: log_mel = whisper_values * 4 - 4
    5. Log-mel -> mel: mel = 10^log_mel
    """
    if img.mode != "L":
        img = img.convert("L")

    pixels = np.array(img, dtype=np.float32)
    print(f"  Pixel array shape: {pixels.shape}")

    # We don't know exact T_frames, so infer from known duration
    # The default page_duration_sec=10.0, so T_frames = (10 * sr) / hop_length
    T_estimated = (10 * sr) // hop_length  # = 1000
    pil_resized = img.resize((T_estimated, n_mels), Image.LANCZOS)

    pixels_resized = np.array(pil_resized, dtype=np.float32)
    print(f"  Resized pixel array shape: {pixels_resized.shape}")

    # Invert pipeline
    normalized = 1.0 - pixels_resized / 255.0
    whisper_values = normalized * 2.0 - 1.0
    log_mel = whisper_values * 4.0 - 4.0
    mel = 10.0 ** log_mel

    mel = np.maximum(mel, 1e-10)

    return mel


def test_render_invert_cycle():
    """Test that render -> invert -> Griffin-Lim -> reconstruct works."""
    print("="*80)
    print("Testing render-invert cycle")
    print("="*80)

    # Generate a simple test signal: a chirp (frequency sweep)
    sr = 16000
    duration = 5.0  # seconds
    t = np.linspace(0, duration, int(sr * duration))
    # Chirp from 200 Hz to 2000 Hz
    f0, f1 = 200, 2000
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
    audio = 0.5 * np.sin(phase)

    print(f"Test signal: {duration}s chirp from {f0} to {f1} Hz")
    print(f"Audio array shape: {audio.shape}, min={audio.min():.4f}, max={audio.max():.4f}")

    # Step 1: Render using production params
    print("\n1. Rendering to PNG...")
    png = render_log_mel_spectrogram(
        audio,
        sample_rate=sr,
        n_mels=80,
        n_fft=400,
        hop_length=160,
        window="hann",
        output_size=512,
        power=2.0,
        source_sample_rate=sr,
        output_width=1000,
        output_height=160,
    )

    if isinstance(png, list):
        print(f"  Got {len(png)} PNG pages, using first")
        png = png[0]

    print(f"  PNG size: {png.size} (width x height)")

    # Save for inspection
    output_dir = Path("/workspace/univi/data/eval/decodability")
    output_dir.mkdir(parents=True, exist_ok=True)
    png.save(output_dir / "test_render.png")
    print(f"  Saved to {output_dir / 'test_render.png'}")

    # Step 2: Invert back to mel
    print("\n2. Inverting PNG to mel...")
    mel = invert_png_to_mel(png, n_mels=80, sr=sr)
    print(f"  Mel shape: {mel.shape}, min={mel.min():.4f}, max={mel.max():.4f}")

    # Step 3: Reconstruct audio using Griffin-Lim
    print("\n3. Reconstructing audio (Griffin-Lim)...")
    reconstructed = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=sr,
        n_fft=400,
        hop_length=160,
        window="hann",
        n_iter=64,
    )
    print(f"  Reconstructed audio shape: {reconstructed.shape}")
    print(f"  Reconstructed audio min={reconstructed.min():.4f}, max={reconstructed.max():.4f}")

    # Trim to match original length
    if len(reconstructed) > len(audio):
        reconstructed = reconstructed[:len(audio)]
    elif len(reconstructed) < len(audio):
        reconstructed = np.pad(reconstructed, (0, len(audio) - len(reconstructed)))

    # Step 4: Compute spectral similarity (not WER, just sanity check)
    print("\n4. Computing spectral metrics...")
    orig_mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, n_fft=400, hop_length=160)
    orig_mel_log = np.log10(np.maximum(orig_mel, 1e-10))

    recon_mel = librosa.feature.melspectrogram(y=reconstructed, sr=sr, n_mels=80, n_fft=400, hop_length=160)
    recon_mel_log = np.log10(np.maximum(recon_mel, 1e-10))

    # Trim to same length
    min_frames = min(orig_mel_log.shape[1], recon_mel_log.shape[1])
    orig_mel_log = orig_mel_log[:, :min_frames]
    recon_mel_log = recon_mel_log[:, :min_frames]

    mse = np.mean((orig_mel_log - recon_mel_log)**2)
    mae = np.mean(np.abs(orig_mel_log - recon_mel_log))
    correlation = np.corrcoef(orig_mel_log.flatten(), recon_mel_log.flatten())[0, 1]

    print(f"  Log-mel MSE: {mse:.6f}")
    print(f"  Log-mel MAE: {mae:.6f}")
    print(f"  Log-mel correlation: {correlation:.6f}")

    # Results summary
    results = {
        "test_type": "render_invert_cycle",
        "test_signal": "chirp_200_2000Hz",
        "duration_sec": duration,
        "audio_shape": audio.shape,
        "png_size": list(png.size),
        "mel_shape": list(mel.shape),
        "reconstructed_shape": reconstructed.shape,
        "spectral_metrics": {
            "log_mel_mse": float(mse),
            "log_mel_mae": float(mae),
            "log_mel_correlation": float(correlation),
        },
        "status": "PASS" if mse < 0.5 else "FAIL",
    }

    results_path = output_dir / "test_cycle_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Status: {results['status']}")
    return results


if __name__ == "__main__":
    results = test_render_invert_cycle()
    print("\n" + "="*80)
    print(f"Cycle test status: {results['status']}")
    print("="*80)
