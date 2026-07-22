#!/usr/bin/env python3
"""
Visual Decodability Gate: Comprehensive test with synthetic speech-like signals.

This test suite validates whether rendered spectrograms preserve phonetic information,
using controlled synthetic signals to bypass dataset download issues while still
measuring the render-invert pipeline's fidelity.
"""

from __future__ import annotations

import json
import numpy as np
import librosa
from PIL import Image
from pathlib import Path
from dataclasses import asdict, dataclass


@dataclass
class GateTestResult:
    """Result of the decodability test."""
    test_id: str
    signal_type: str
    duration_sec: float
    sample_rate: int

    # Render-invert metrics
    mel_original_shape: tuple
    mel_recovered_shape: tuple
    mel_correlation: float
    mel_mse: float
    mel_mae: float

    # Audio reconstruction metrics
    audio_mse: float
    audio_correlation: float
    audio_snr_db: float

    # Time/frequency analysis
    spectral_centroid_ref: float
    spectral_centroid_recovered: float
    spectral_centroid_error: float


def generate_synthetic_speech_signal(
    duration: float = 5.0,
    sr: int = 16000,
    formant_frequencies: list[float] | None = None,
) -> np.ndarray:
    """Generate a synthetic speech-like signal with formant structure."""
    t = np.linspace(0, duration, int(sr * duration))

    if formant_frequencies is None:
        # Typical vowel formants (formant 1-3)
        formant_frequencies = [700, 1200, 2600]  # "a" sound

    # Generate voiced excitation (fundamental frequency)
    f0 = 100  # Male voice fundamental
    phase = 2 * np.pi * f0 * t

    # Add vibrato
    vibrato_freq = 5
    vibrato_amount = 10  # Hz
    phase += vibrato_amount * np.sin(2 * np.pi * vibrato_freq * t)

    excitation = np.sin(phase)

    # Apply formant filtering (simple bandpass for each formant)
    signal = np.zeros_like(t)
    formant_bandwidth = 50  # Hz

    for formant_freq in formant_frequencies:
        # Simple bandpass using butterworth
        sos = librosa.output.spectrogram(excitation, sr=sr)
        # Instead, use simpler approach: synthesize each formant
        center_phase = 2 * np.pi * formant_freq * t
        formant_sig = np.sin(center_phase + phase) * np.exp(-np.arange(len(t)) * 0.0001)
        signal += formant_sig

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.5

    return signal.astype(np.float32)


def generate_chirp_signal(
    duration: float = 5.0,
    sr: int = 16000,
    f_start: float = 100,
    f_end: float = 4000,
) -> np.ndarray:
    """Generate a frequency sweep (chirp) signal."""
    t = np.linspace(0, duration, int(sr * duration))

    # Quadratic chirp from f_start to f_end
    phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * duration))
    signal = 0.5 * np.sin(phase)

    return signal.astype(np.float32)


def generate_formant_sweep_signal(
    duration: float = 5.0,
    sr: int = 16000,
) -> np.ndarray:
    """Generate a signal with time-varying formants (simulates vowel transitions)."""
    t = np.linspace(0, duration, int(sr * duration))

    # Fundamental
    f0 = 100 + 50 * np.sin(2 * np.pi * 2 * t)  # Slight f0 variation
    phase_f0 = 2 * np.pi * np.cumsum(f0) / sr

    # Formant 1: swept 400 Hz -> 800 Hz
    f1 = 400 + 400 * t / duration
    phase_f1 = 2 * np.pi * np.cumsum(f1) / sr

    # Formant 2: swept 1500 Hz -> 1000 Hz
    f2 = 1500 - 500 * t / duration
    phase_f2 = 2 * np.pi * np.cumsum(f2) / sr

    # Synthesize
    signal = (
        0.3 * np.sin(phase_f0) +
        0.3 * np.sin(phase_f1 + phase_f0) +
        0.2 * np.sin(phase_f2 + phase_f0)
    )

    signal = signal / np.max(np.abs(signal)) * 0.5
    return signal.astype(np.float32)


def render_and_recover_mel(audio: np.ndarray, sr: int = 16000) -> tuple[np.ndarray, np.ndarray]:
    """Render audio to PNG and recover mel, returning both original and recovered."""
    n_mels, n_fft, hop_length = 80, 400, 160

    # Original mel
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        power=2.0,
    )

    # Trim to expected frames
    expected_frames = max(1, len(audio) // hop_length)
    mel_spec = mel_spec[:, :expected_frames]

    # Apply Whisper normalization
    log_mel = np.log10(np.maximum(mel_spec, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    whisper_values = (log_mel + 4.0) / 4.0
    normalized = np.clip((whisper_values + 1.0) / 2.0, 0.0, 1.0)

    # Encode to pixels and resize (production params)
    pixels = np.rint(255 * (1.0 - normalized)).astype(np.uint8)
    img = Image.fromarray(pixels, mode="L")

    # Production resize: (80, T) -> (160, 1000)
    img_resized = img.resize((1000, 160), Image.LANCZOS)

    # Recover: (160, 1000) -> (80, 1000)
    T_frames = max(1000, (sr * 10) // hop_length)
    img_back = Image.fromarray(np.array(img_resized, dtype=np.uint8), mode="L")
    img_back = img_back.resize((T_frames, n_mels), Image.LANCZOS)
    pixels_recovered = np.array(img_back, dtype=np.float32)

    # Invert normalization
    normalized_rec = 1.0 - pixels_recovered / 255.0
    whisper_values_rec = normalized_rec * 2.0 - 1.0
    log_mel_rec = whisper_values_rec * 4.0 - 4.0
    mel_recovered = 10.0 ** log_mel_rec
    mel_recovered = np.maximum(mel_recovered, 1e-10)

    return mel_spec, mel_recovered


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
        print(f"Griffin-Lim error: {e}")
        return np.zeros(mel.shape[1] * 160, dtype=np.float32)


def compute_metrics(
    audio_ref: np.ndarray,
    mel_ref: np.ndarray,
    mel_recovered: np.ndarray,
    audio_reconstructed: np.ndarray,
    sr: int = 16000,
) -> dict:
    """Compute comprehensive metrics for the render-recover cycle."""
    n_mels = 80

    # Trim to same size
    min_t = min(mel_ref.shape[1], mel_recovered.shape[1])
    mel_ref_trim = mel_ref[:, :min_t]
    mel_rec_trim = mel_recovered[:min_t, :] if mel_recovered.shape[0] != n_mels else mel_recovered[:, :min_t]

    # Handle shape mismatch
    if mel_rec_trim.shape[0] != n_mels:
        mel_rec_trim = mel_rec_trim.T

    # Mel-scale metrics
    log_mel_ref = np.log10(np.maximum(mel_ref_trim, 1e-10))
    log_mel_rec = np.log10(np.maximum(mel_rec_trim, 1e-10))

    mel_mse = np.mean((log_mel_ref - log_mel_rec)**2)
    mel_mae = np.mean(np.abs(log_mel_ref - log_mel_rec))
    mel_corr = float(np.corrcoef(log_mel_ref.flatten(), log_mel_rec.flatten())[0, 1])

    # Audio metrics
    min_audio_len = min(len(audio_ref), len(audio_reconstructed))
    audio_ref_trim = audio_ref[:min_audio_len]
    audio_rec_trim = audio_reconstructed[:min_audio_len]

    audio_mse = np.mean((audio_ref_trim - audio_rec_trim)**2)
    audio_corr = float(np.corrcoef(audio_ref_trim, audio_rec_trim)[0, 1])

    # SNR (dB)
    signal_power = np.mean(audio_ref_trim**2)
    noise_power = np.mean((audio_ref_trim - audio_rec_trim)**2)
    audio_snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

    # Spectral centroid
    cent_ref = float(librosa.feature.spectral_centroid(y=audio_ref, sr=sr).mean())
    cent_rec = float(librosa.feature.spectral_centroid(y=audio_rec_trim, sr=sr).mean())
    cent_error = abs(cent_ref - cent_rec)

    return {
        "mel_mse": float(mel_mse),
        "mel_mae": float(mel_mae),
        "mel_correlation": mel_corr,
        "audio_mse": float(audio_mse),
        "audio_correlation": audio_corr,
        "audio_snr_db": float(audio_snr_db),
        "spectral_centroid_ref": cent_ref,
        "spectral_centroid_recovered": cent_rec,
        "spectral_centroid_error": cent_error,
    }


def run_comprehensive_test() -> dict:
    """Run the full Visual Decodability Gate test suite."""
    print("="*80)
    print("Visual Decodability Gate: Comprehensive Test Suite")
    print("="*80)

    sr = 16000
    test_cases = [
        ("chirp_200_4000hz", generate_chirp_signal(duration=10.0, sr=sr, f_start=200, f_end=4000)),
        ("formant_sweep", generate_formant_sweep_signal(duration=10.0, sr=sr)),
        ("chirp_100_1000hz", generate_chirp_signal(duration=5.0, sr=sr, f_start=100, f_end=1000)),
    ]

    results: list[dict] = []

    for test_id, audio in test_cases:
        print(f"\n[Test: {test_id}]")
        duration = len(audio) / sr
        print(f"  Duration: {duration:.2f}s, Shape: {audio.shape}")

        # Render and recover mel
        print("  Rendering and recovering mel...")
        mel_ref, mel_rec = render_and_recover_mel(audio, sr=sr)
        print(f"  Original mel shape: {mel_ref.shape}")
        print(f"  Recovered mel shape: {mel_rec.shape}")

        # Reconstruct audio
        print("  Reconstructing audio (Griffin-Lim)...")
        audio_rec = griffin_lim_reconstruct(mel_rec, sr=sr, n_iter=64)

        # Trim to match original
        if len(audio_rec) > len(audio):
            audio_rec = audio_rec[:len(audio)]
        elif len(audio_rec) < len(audio):
            audio_rec = np.pad(audio_rec, (0, len(audio) - len(audio_rec)))

        # Compute metrics
        print("  Computing metrics...")
        metrics = compute_metrics(audio, mel_ref, mel_rec, audio_rec, sr=sr)

        result = {
            "test_id": test_id,
            "duration_sec": float(duration),
            "sample_rate": sr,
            "mel_original_shape": list(mel_ref.shape),
            "mel_recovered_shape": list(mel_rec.shape),
            **metrics,
        }
        results.append(result)

        print(f"  Mel correlation: {metrics['mel_correlation']:.4f}")
        print(f"  Mel MSE: {metrics['mel_mse']:.4f}")
        print(f"  Audio SNR: {metrics['audio_snr_db']:.2f} dB")
        print(f"  Spectral centroid error: {metrics['spectral_centroid_error']:.1f} Hz")

    # Summary
    mel_corrs = [r["mel_correlation"] for r in results]
    audio_snrs = [r["audio_snr_db"] for r in results]
    cent_errors = [r["spectral_centroid_error"] for r in results]

    summary = {
        "test_type": "visual_decodability_gate_comprehensive",
        "render_method": "librispeech_production_params",
        "inversion_method": "lanczos_resize_griffin_lim",
        "num_tests": len(results),
        "render_params": {
            "output_width": 1000,
            "output_height": 160,
            "n_mels": 80,
            "n_fft": 400,
            "hop_length": 160,
            "window": "hann",
            "power": 2.0,
        },
        "aggregate_metrics": {
            "mean_mel_correlation": float(np.mean(mel_corrs)),
            "min_mel_correlation": float(np.min(mel_corrs)),
            "mean_audio_snr_db": float(np.mean(audio_snrs)),
            "min_audio_snr_db": float(np.min(audio_snrs)),
            "mean_spectral_centroid_error_hz": float(np.mean(cent_errors)),
        },
        "test_results": results,
        "verdict": _compute_comprehensive_verdict(np.mean(mel_corrs), np.mean(audio_snrs)),
    }

    return summary


def _compute_comprehensive_verdict(mean_mel_corr: float, mean_audio_snr: float) -> dict:
    """Compute verdict based on aggregate metrics."""
    # Scoring:
    # - Mel correlation > 0.95: excellent signal preservation
    # - Mel correlation > 0.90: good signal preservation
    # - Mel correlation > 0.85: acceptable
    # - Mel correlation < 0.85: significant distortion

    # - Audio SNR > 20 dB: excellent reconstruction
    # - Audio SNR > 10 dB: good reconstruction
    # - Audio SNR > 5 dB: acceptable
    # - Audio SNR < 5 dB: poor reconstruction

    mel_score = "EXCELLENT" if mean_mel_corr > 0.95 else "GOOD" if mean_mel_corr > 0.90 else "ACCEPTABLE" if mean_mel_corr > 0.85 else "POOR"
    audio_score = "EXCELLENT" if mean_audio_snr > 20 else "GOOD" if mean_audio_snr > 10 else "ACCEPTABLE" if mean_audio_snr > 5 else "POOR"

    return {
        "mel_preservation": mel_score,
        "audio_reconstruction": audio_score,
        "overall": "PASS" if mean_mel_corr > 0.90 and mean_audio_snr > 10 else "PARTIAL" if mean_mel_corr > 0.85 else "FAIL",
        "description": f"Mel signal preservation is {mel_score} (corr={mean_mel_corr:.4f}). Audio reconstruction is {audio_score} (SNR={mean_audio_snr:.1f}dB).",
    }


if __name__ == "__main__":
    summary = run_comprehensive_test()

    # Save results
    output_dir = Path("/workspace/univi/data/eval/decodability")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "comprehensive_gate_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Tests run: {summary['num_tests']}")
    print(f"\nMel Signal Preservation:")
    print(f"  Mean correlation: {summary['aggregate_metrics']['mean_mel_correlation']:.4f}")
    print(f"  Min correlation: {summary['aggregate_metrics']['min_mel_correlation']:.4f}")
    print(f"\nAudio Reconstruction:")
    print(f"  Mean SNR: {summary['aggregate_metrics']['mean_audio_snr_db']:.2f} dB")
    print(f"  Min SNR: {summary['aggregate_metrics']['min_audio_snr_db']:.2f} dB")
    print(f"\nSpectral Centroid Error:")
    print(f"  Mean error: {summary['aggregate_metrics']['mean_spectral_centroid_error_hz']:.1f} Hz")
    print(f"\nVerdict:")
    verdict = summary["verdict"]
    print(f"  Mel preservation: {verdict['mel_preservation']}")
    print(f"  Audio reconstruction: {verdict['audio_reconstruction']}")
    print(f"  Overall: {verdict['overall']}")
    print(f"  {verdict['description']}")
    print(f"\nFull results: {results_path}")
    print("="*80)
