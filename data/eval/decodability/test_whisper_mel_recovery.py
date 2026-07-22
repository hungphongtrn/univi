#!/usr/bin/env python3
"""
Direct test: Can Whisper read recovered mel spectrograms from rendered PNGs?

This approach:
1. Renders audio -> PNG using production params
2. Recovers mel from PNG pixels
3. Feeds mel directly to Whisper's decoder (bypassing audio->mel)
4. Compares transcriptions
"""

from __future__ import annotations

import json
import numpy as np
import librosa
from PIL import Image
from pathlib import Path

import sys
sys.path.insert(0, "/workspace/univi")
from data.preprocessing.render_utils import _whisper_log_mel


def transcribe_with_whisper_mel(mel_log: np.ndarray, sr: int = 16000) -> str:
    """
    Transcribe using Whisper's decoder with pre-computed mel.

    This bypasses Whisper's audio->mel computation, feeding mels directly.
    """
    try:
        # Import Whisper components
        from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration
        from transformers.models.whisper.tokenizer import MULTILINGUAL_LANGUAGE_TOKENS
        from transformers import AutoTokenizer

        # Load model (use base to keep it reasonable)
        model_id = "openai/whisper-base"
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Prepare mel input (shape should be (1, n_mels, T_frames))
        mel_input = torch.from_numpy(mel_log).unsqueeze(0).float()  # (1, 80, T)

        # Get encoder output
        encoder_outputs = model.encoder(mel_input)

        # Prepare decoder input (start token)
        decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

        # Generate with greedy search
        generated_ids = model.generate(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
        )

        # Decode
        transcription = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription

    except Exception as e:
        print(f"Error in Whisper mel transcription: {e}")
        import traceback
        traceback.print_exc()
        return ""


def test_mel_recovery_simple():
    """Test mel recovery from rendered PNG on a simple test signal."""
    print("="*80)
    print("Testing mel recovery and transcription")
    print("="*80)

    sr = 16000
    duration = 10.0  # Use 10s to match production page_duration_sec
    t = np.linspace(0, duration, int(sr * duration))
    # Chirp
    f0, f1 = 200, 2000
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
    audio = 0.5 * np.sin(phase)

    print(f"Test signal: {duration}s chirp from {f0} to {f1} Hz")
    print(f"Audio shape: {audio.shape}")

    # Render using production params
    print("\n1. Rendering mel using Whisper pipeline...")
    mel_original = _whisper_log_mel(
        audio,
        sample_rate=sr,
        n_mels=80,
        n_fft=400,
        hop_length=160,
        window="hann",
        power=2.0,
    )
    print(f"   Original mel shape: {mel_original.shape}")
    print(f"   Original mel min={mel_original.min():.6f}, max={mel_original.max():.6f}")

    # Encode to pixels (like rendering)
    print("\n2. Encoding mel to pixels...")
    pixels = np.rint(255 * (1.0 - mel_original)).astype(np.uint8)

    # Create image and resize to production size
    img = Image.fromarray(pixels, mode="L")
    print(f"   Image before resize: {img.size}")

    # Resize to production params (1000x160)
    output_width, output_height = 1000, 160
    img_resized = img.resize((output_width, output_height), Image.LANCZOS)
    print(f"   Image after resize: {img_resized.size}")

    # Decode from pixels, resizing back to original frequency resolution
    print("\n3. Decoding pixels back to mel...")
    pixels_resized = np.array(img_resized, dtype=np.float32)

    # IMPORTANT: resize back to (80, 1000) to match original mel resolution
    # pixels_resized is (160, 1000), we need to resize to (80, 1000)
    # Use PIL to resize: (width, height) = (1000, 80)
    img_back = Image.fromarray(pixels_resized.astype(np.uint8), mode="L")
    img_back = img_back.resize((1000, 80), Image.LANCZOS)
    pixels_recovered = np.array(img_back, dtype=np.float32)

    # Invert: pixel -> normalized -> whisper_values -> log_mel -> mel
    normalized = 1.0 - pixels_recovered / 255.0
    whisper_values = normalized * 2.0 - 1.0
    log_mel = whisper_values * 4.0 - 4.0
    mel_recovered = 10.0 ** log_mel

    print(f"   Recovered mel shape: {mel_recovered.shape}")
    print(f"   Recovered mel min={mel_recovered.min():.6f}, max={mel_recovered.max():.6f}")

    # Clamp to valid range
    mel_recovered = np.maximum(mel_recovered, 1e-10)

    # Compare
    print("\n4. Comparing original vs recovered mel...")

    # Now both should be (80, 1000)
    mel_orig_trim = mel_original[:, :1000]
    mel_recov_trim = mel_recovered

    print(f"   Original trimmed shape: {mel_orig_trim.shape}")
    print(f"   Recovered trimmed shape: {mel_recov_trim.shape}")

    # Convert both to log-mel for comparison
    log_mel_orig = np.log10(np.maximum(mel_orig_trim, 1e-10))
    log_mel_recov = np.log10(np.maximum(mel_recov_trim, 1e-10))

    mse = np.mean((log_mel_orig - log_mel_recov)**2)
    mae = np.mean(np.abs(log_mel_orig - log_mel_recov))
    corr = np.corrcoef(log_mel_orig.flatten(), log_mel_recov.flatten())[0, 1]

    print(f"   Log-mel MSE: {mse:.6f}")
    print(f"   Log-mel MAE: {mae:.6f}")
    print(f"   Log-mel correlation: {corr:.6f}")

    results = {
        "test_type": "mel_recovery_from_png",
        "duration_sec": duration,
        "render_params": {
            "output_width": output_width,
            "output_height": output_height,
            "n_mels": 80,
        },
        "original_mel_shape": list(mel_original.shape),
        "recovered_mel_shape": list(mel_recovered.shape),
        "metrics": {
            "log_mel_mse": float(mse),
            "log_mel_mae": float(mae),
            "log_mel_correlation": float(corr),
        },
    }

    output_dir = Path("/workspace/univi/data/eval/decodability")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "mel_recovery_results.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    # Visualize
    print("\n5. Saving mel visualizations...")
    def mel_to_image(mel, name):
        mel_clipped = np.clip(mel, 0, 1)
        pixels = (255 * (1.0 - mel_clipped)).astype(np.uint8)
        img = Image.fromarray(pixels, mode="L")
        img_path = output_dir / f"{name}.png"
        img.save(img_path)
        print(f"   Saved {img_path}")

    # Save log-mels for visualization
    log_mel_orig_vis = np.clip((log_mel_orig + 4.0) / 8.0, 0, 1)
    log_mel_recov_vis = np.clip((log_mel_recov + 4.0) / 8.0, 0, 1)

    mel_to_image(log_mel_orig_vis, "mel_original")
    mel_to_image(log_mel_recov_vis, "mel_recovered")

    return results


if __name__ == "__main__":
    results = test_mel_recovery_simple()
    print("\n" + "="*80)
    print("Summary:")
    print(f"  Log-mel correlation: {results['metrics']['log_mel_correlation']:.4f}")
    print("="*80)
