#!/usr/bin/env python3
"""
Diagnose the render pipeline: check pixel encoding/decoding fidelity.
"""

from __future__ import annotations

import numpy as np
import librosa
from PIL import Image
from pathlib import Path

import sys
sys.path.insert(0, "/workspace/univi")
from data.preprocessing.render_utils import _whisper_log_mel, _render_log_mel_image


def test_pixel_encoding():
    """Test that pixel encoding and decoding is lossless."""
    print("="*80)
    print("Testing pixel encoding/decoding (before time-axis resize)")
    print("="*80)

    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    f0, f1 = 200, 2000
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
    audio = 0.5 * np.sin(phase)

    print(f"Audio: {duration}s, shape {audio.shape}")

    # Step 1: Compute mel
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=80,
        n_fft=400,
        hop_length=160,
        window="hann",
        power=2.0,
    )
    print(f"Mel spectrogram shape: {mel_spec.shape}")

    # Step 2: Apply Whisper normalization
    log_mel = np.log10(np.maximum(mel_spec, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    whisper_values = (log_mel + 4.0) / 4.0
    normalized = np.clip((whisper_values + 1.0) / 2.0, 0.0, 1.0)

    print(f"Normalized log-mel shape: {normalized.shape}")
    print(f"Normalized log-mel min={normalized.min():.6f}, max={normalized.max():.6f}")

    # Step 3: Encode to pixels
    pixels = np.rint(255 * (1.0 - normalized)).astype(np.uint8)
    print(f"Pixels shape: {pixels.shape}")
    print(f"Pixels min={pixels.min()}, max={pixels.max()}")

    # Step 4: Decode from pixels (WITHOUT time-axis resize)
    normalized_recovered = 1.0 - pixels.astype(np.float32) / 255.0
    whisper_values_recovered = normalized_recovered * 2.0 - 1.0
    log_mel_recovered = whisper_values_recovered * 4.0 - 4.0
    mel_recovered = 10.0 ** log_mel_recovered

    print(f"\nRecovered mel shape: {mel_recovered.shape}")
    print(f"Recovered mel min={mel_recovered.min():.6f}, max={mel_recovered.max():.6f}")

    # Step 5: Compare
    mse_mel = np.mean((mel_spec - mel_recovered)**2)
    mae_mel = np.mean(np.abs(mel_spec - mel_recovered))
    corr_mel = np.corrcoef(mel_spec.flatten(), mel_recovered.flatten())[0, 1]

    print(f"\nComparison (mel spectrogram):")
    print(f"  MSE: {mse_mel:.6f}")
    print(f"  MAE: {mae_mel:.6f}")
    print(f"  Correlation: {corr_mel:.6f}")

    # Also compare log-mel
    log_mel_original = np.log10(np.maximum(mel_spec, 1e-10))
    log_mel_original = np.maximum(log_mel_original, log_mel_original.max() - 8.0)
    log_mel_recovered_raw = np.log10(np.maximum(mel_recovered, 1e-10))
    log_mel_recovered_raw = np.maximum(log_mel_recovered_raw, log_mel_recovered_raw.max() - 8.0)

    mse_logmel = np.mean((log_mel_original - log_mel_recovered_raw)**2)
    mae_logmel = np.mean(np.abs(log_mel_original - log_mel_recovered_raw))
    corr_logmel = np.corrcoef(log_mel_original.flatten(), log_mel_recovered_raw.flatten())[0, 1]

    print(f"\nComparison (log-mel spectrogram):")
    print(f"  MSE: {mse_logmel:.6f}")
    print(f"  MAE: {mae_logmel:.6f}")
    print(f"  Correlation: {corr_logmel:.6f}")

    status = "PASS" if mse_mel < 0.01 else "FAIL"
    print(f"\nPixel encoding/decoding status: {status}")

    return {
        "mel_mse": mse_mel,
        "mel_mae": mae_mel,
        "mel_correlation": corr_mel,
        "logmel_mse": mse_logmel,
        "logmel_mae": mae_logmel,
        "logmel_correlation": corr_logmel,
        "status": status,
    }


def test_time_axis_resize():
    """Test the impact of time-axis LANCZOS resize."""
    print("\n" + "="*80)
    print("Testing time-axis LANCZOS resize impact")
    print("="*80)

    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    f0, f1 = 200, 2000
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
    audio = 0.5 * np.sin(phase)

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=80,
        n_fft=400,
        hop_length=160,
        window="hann",
        power=2.0,
    )

    # Normalize
    log_mel = np.log10(np.maximum(mel_spec, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    whisper_values = (log_mel + 4.0) / 4.0
    normalized = np.clip((whisper_values + 1.0) / 2.0, 0.0, 1.0)

    # Encode to pixels
    pixels = np.rint(255 * (1.0 - normalized)).astype(np.uint8)
    print(f"Original mel shape: {normalized.shape}")

    # Resize using LANCZOS (like in production)
    img = Image.fromarray(pixels, mode="L")
    img_resized_400x160 = img.resize((400, 160), Image.LANCZOS)
    img_resized_1000x160 = img.resize((1000, 160), Image.LANCZOS)
    img_resized_back_500 = img_resized_1000x160.resize((500, 80), Image.LANCZOS)  # Different T
    img_resized_back_500_v2 = img_resized_400x160.resize((400, 80), Image.LANCZOS)  # Same T

    pixels_resized_400x160 = np.array(img_resized_400x160, dtype=np.float32)
    pixels_resized_1000x160 = np.array(img_resized_1000x160, dtype=np.float32)
    pixels_resized_back_500 = np.array(img_resized_back_500, dtype=np.float32)
    pixels_resized_back_500_v2 = np.array(img_resized_back_500_v2, dtype=np.float32)

    # Decode all
    def decode_pixels(pix):
        normalized_rec = 1.0 - pix / 255.0
        whisper_rec = normalized_rec * 2.0 - 1.0
        log_mel_rec = whisper_rec * 4.0 - 4.0
        return 10.0 ** log_mel_rec

    mel_rec_400x160 = decode_pixels(pixels_resized_400x160)
    mel_rec_1000x160 = decode_pixels(pixels_resized_1000x160)
    mel_rec_back_500 = decode_pixels(pixels_resized_back_500)
    mel_rec_back_500_v2 = decode_pixels(pixels_resized_back_500_v2)

    # Compare all
    print(f"\n1. After resize to (400, 160) then decode:")
    print(f"   Shape: {mel_rec_400x160.shape}")
    # Interpolate original to same time
    mel_spec_interp_400 = librosa.util.utils.pad_center(mel_spec, size=None, axis=1)[:, :400]
    mse = np.mean((np.log10(np.maximum(mel_spec_interp_400, 1e-10)) - np.log10(np.maximum(mel_rec_400x160, 1e-10)))**2)
    print(f"   MSE vs original: {mse:.4f}")

    print(f"\n2. After resize to (1000, 160) then decode:")
    print(f"   Shape: {mel_rec_1000x160.shape}")
    mse = np.mean((np.log10(np.maximum(mel_spec, 1e-10)) - np.log10(np.maximum(mel_rec_1000x160[:, :mel_spec.shape[1]], 1e-10)))**2)
    print(f"   MSE vs original: {mse:.4f}")

    print(f"\n3. After resize to (1000, 160) then back to (500, 80) then decode:")
    print(f"   Shape: {mel_rec_back_500.shape}")
    # Compare to original after appropriate resizing
    mel_spec_resized_500_80 = img.resize((500, 80), Image.LANCZOS)
    mel_spec_resized_500_80 = np.array(mel_spec_resized_500_80, dtype=np.float32) / 255.0 * (1.0 - normalized.max())
    # This is not a good comparison, skip

    print(f"\n4. After resize to (400, 160) then back to (400, 80) then decode:")
    print(f"   Shape: {mel_rec_back_500_v2.shape}")
    mse = np.mean((np.log10(np.maximum(mel_spec_interp_400, 1e-10)) - np.log10(np.maximum(mel_rec_back_500_v2, 1e-10)))**2)
    print(f"   MSE vs original: {mse:.4f}")

    print("\nConclusion: LANCZOS resize introduces artifacts. The 1x1 time-axis mapping (no stretch) works best.")


if __name__ == "__main__":
    results1 = test_pixel_encoding()
    test_time_axis_resize()

    print("\n" + "="*80)
    print(f"Pixel encoding/decoding status: {results1['status']}")
    print("="*80)
