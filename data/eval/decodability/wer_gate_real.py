#!/usr/bin/env python3
"""
Visual Decodability Gate: WER on real LibriSpeech clips via Whisper mel decoder.

Three conditions on SAME clips:
  NATIVE:  raw audio → Whisper.transcribe → WER_native
  CONTROL: raw audio → librosa mel (repo pipeline) → Whisper.decode(mel) → WER_control
  GATE:    raw audio → render PNG → recover mel → Whisper.decode(mel) → WER_gate
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
from data.preprocessing.render_utils import render_log_mel_spectrogram


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


def get_whisper_model():
    """Load Whisper model (base)."""
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    print("Loading Whisper base model...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model


def transcribe_native(audio_array: np.ndarray, sr: int = 16000) -> str:
    """Condition 1: NATIVE - Whisper.transcribe on raw audio."""
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=0)
    result = asr({"sampling_rate": sr, "raw": audio_array})
    return result.get("text", "").strip()


def transcribe_mel_direct(mel: np.ndarray, processor, model) -> str:
    """Transcribe via mel directly (conditions 2 and 3)."""
    import torch

    # Pad/trim to Whisper's expected length (3000 frames for 30s at hop_length=160)
    # Whisper expects: (batch, n_mels, time_steps) where time_steps ≤ 3000
    expected_frames = 3000
    if mel.shape[1] < expected_frames:
        mel = np.pad(mel, ((0, 0), (0, expected_frames - mel.shape[1])), mode='constant')
    else:
        mel = mel[:, :expected_frames]

    # Prepare input (add batch dimension, move to device)
    mel_input = torch.from_numpy(mel).unsqueeze(0).float()
    if torch.cuda.is_available():
        mel_input = mel_input.cuda()

    # Encode: mel -> encoder output
    with torch.no_grad():
        encoder_outputs = model.encoder(mel_input)

        # Decode: greedy decoding starting from BOS token
        decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], device=mel_input.device)
        generated_ids = model.generate(encoder_outputs=encoder_outputs, decoder_input_ids=decoder_input_ids)

    # Decode to text
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription.strip()


def build_librosa_mel(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Build mel using repo's exact pipeline (CONTROL condition)."""
    n_mels, n_fft, hop_length = 80, 400, 160

    # Build mel
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, window="hann", power=2.0,
    )

    # Trim to expected frames
    expected_frames = max(1, len(audio) // hop_length)
    mel_spec = mel_spec[:, :expected_frames]

    # Whisper normalization
    log_mel = np.log10(np.maximum(mel_spec, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    whisper_values = (log_mel + 4.0) / 4.0
    normalized = np.clip((whisper_values + 1.0) / 2.0, 0.0, 1.0)

    # Convert back to mel (undo normalization to get log-mel Whisper expects)
    log_mel_recovered = (normalized * 2.0 - 1.0) * 4.0 - 4.0
    mel_recovered = 10.0 ** log_mel_recovered

    return np.maximum(mel_recovered, 1e-10)


def render_and_recover_mel(audio: np.ndarray, sr: int = 16000, output_width: int = 1000, output_height: int = 160) -> np.ndarray:
    """GATE condition: render to PNG and recover mel."""
    n_mels, n_fft, hop_length = 80, 400, 160

    # Step 1: Render to PNG using production params
    png = render_log_mel_spectrogram(
        audio,
        sample_rate=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        output_width=output_width,
        output_height=output_height,
        power=2.0,
        source_sample_rate=sr,
    )

    # Handle multi-page return
    if isinstance(png, list):
        png = png[0]

    # Step 2: Recover mel from PNG pixels
    if png.mode != "L":
        png = png.convert("L")

    pixels = np.array(png, dtype=np.float32)

    # Step 3: CRITICAL - resize back to original mel dimensions
    # Original mel is (80, T_frames) where T_frames = len(audio) // hop_length
    T_frames_original = len(audio) // hop_length

    # PNG is (output_width, output_height) = (1000, 160) in PIL coords
    # Resize back to (T_frames_original, 80) to DE-WARP time axis
    png_resized = Image.fromarray(pixels.astype(np.uint8), mode="L")
    png_resized = png_resized.resize((T_frames_original, n_mels), Image.LANCZOS)
    pixels_recovered = np.array(png_resized, dtype=np.float32)

    # Step 4: Invert pixel encoding to recover mel
    # pixels = 255 * (1 - normalized_log_mel)
    # normalized_log_mel = 1 - pixels/255
    normalized = 1.0 - pixels_recovered / 255.0

    # Undo Whisper normalization: normalized = (whisper_values + 1) / 2
    # whisper_values = 2 * normalized - 1
    whisper_values = 2.0 * normalized - 1.0

    # Undo Whisper normalization: whisper_values = (log_mel + 4) / 4
    # log_mel = 4 * whisper_values - 4
    log_mel = 4.0 * whisper_values - 4.0

    # Recover mel: log_mel = log10(mel)
    # mel = 10^log_mel
    mel = 10.0 ** log_mel

    return np.maximum(mel, 1e-10)


@dataclass
class ClipResult:
    clip_id: str
    reference_text: str
    native_transcript: str
    native_wer: float
    control_transcript: str
    control_wer: float
    gate_transcript: str
    gate_wer: float


def run_wer_gate_test(max_clips: int = 30) -> dict:
    """Run WER gate test on real LibriSpeech."""
    from datasets import load_dataset

    print("="*80)
    print(f"WER Gate Test: Real LibriSpeech (up to {max_clips} clips)")
    print("="*80)

    # Load Whisper
    processor, model = get_whisper_model()

    # Load dataset with streaming to avoid full download
    print("Loading LibriSpeech clean test split (streaming)...")
    try:
        ds = load_dataset(
            "openslr/librispeech_asr",
            "clean",
            split="test",
            streaming=True,
        )
        ds = ds.take(max_clips)
        ds_list = list(ds)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative split...")
        try:
            ds = load_dataset(
                "openslr/librispeech_asr",
                "clean",
                split="validation",
                streaming=True,
            )
            ds = ds.take(max_clips)
            ds_list = list(ds)
        except:
            print("Could not load dataset. Ensure network connectivity.")
            return {}

    print(f"Loaded {len(ds_list)} clips\n")

    results: list[ClipResult] = []
    sr = 16000

    for idx, row in enumerate(ds_list):
        clip_id = str(row.get("id", idx))
        reference_text = row["text"].strip()

        # Extract audio
        audio_dict = row["audio"]
        if isinstance(audio_dict, dict):
            audio_array = np.array(audio_dict["array"], dtype=np.float32)
            audio_sr = audio_dict.get("sampling_rate", sr)
        else:
            # Try to handle as object with array attribute
            audio_array = np.array(audio_dict.array, dtype=np.float32)
            audio_sr = audio_dict.sampling_rate

        # Resample if needed
        if audio_sr != sr:
            audio_array = librosa.resample(audio_array, orig_sr=audio_sr, target_sr=sr)

        print(f"[{idx + 1}/{len(ds_list)}] {clip_id}: {reference_text[:60]}")

        # CONDITION 1: NATIVE
        print(f"  [NATIVE] Whisper.transcribe on raw audio...")
        try:
            native_text = transcribe_native(audio_array, sr=sr)
            native_wer = compute_wer(reference_text, native_text)
            print(f"    WER: {native_wer:.4f}")
        except Exception as e:
            print(f"    Error: {e}")
            native_text = ""
            native_wer = 1.0

        # CONDITION 2: CONTROL
        print(f"  [CONTROL] Librosa mel → Whisper.decode...")
        try:
            mel_control = build_librosa_mel(audio_array, sr=sr)
            control_text = transcribe_mel_direct(mel_control, processor, model)
            control_wer = compute_wer(reference_text, control_text)
            print(f"    WER: {control_wer:.4f}")
        except Exception as e:
            print(f"    Error: {e}")
            control_text = ""
            control_wer = 1.0

        # CONDITION 3: GATE
        print(f"  [GATE] Render PNG → recover mel → Whisper.decode...")
        try:
            mel_gate = render_and_recover_mel(audio_array, sr=sr)
            gate_text = transcribe_mel_direct(mel_gate, processor, model)
            gate_wer = compute_wer(reference_text, gate_text)
            print(f"    WER: {gate_wer:.4f}")
        except Exception as e:
            print(f"    Error: {e}")
            gate_text = ""
            gate_wer = 1.0

        print()

        result = ClipResult(
            clip_id=clip_id,
            reference_text=reference_text,
            native_transcript=native_text,
            native_wer=native_wer,
            control_transcript=control_text,
            control_wer=control_wer,
            gate_transcript=gate_text,
            gate_wer=gate_wer,
        )
        results.append(result)

    # Aggregate
    native_wers = [r.native_wer for r in results]
    control_wers = [r.control_wer for r in results]
    gate_wers = [r.gate_wer for r in results]

    mean_native = np.mean(native_wers) if native_wers else 0.0
    mean_control = np.mean(control_wers) if control_wers else 0.0
    mean_gate = np.mean(gate_wers) if gate_wers else 0.0

    # Compute gaps
    gap_control_vs_native = mean_control - mean_native  # Probe artifact
    gap_gate_vs_control = mean_gate - mean_control      # Render loss

    # Verdict
    verdict = compute_verdict(mean_native, mean_control, mean_gate, gap_control_vs_native, gap_gate_vs_control)

    summary = {
        "test_type": "wer_gate_real_librispeech",
        "num_clips": len(results),
        "whisper_model": "openai/whisper-base",
        "render_params": {
            "output_width": 1000,
            "output_height": 160,
        },
        "wer_native_mean": float(mean_native),
        "wer_control_mean": float(mean_control),
        "wer_gate_mean": float(mean_gate),
        "gap_control_vs_native": float(gap_control_vs_native),
        "gap_gate_vs_control": float(gap_gate_vs_control),
        "verdict": verdict,
        "individual_results": [
            {
                "clip_id": r.clip_id,
                "reference_text": r.reference_text[:100],
                "native_wer": round(r.native_wer, 4),
                "control_wer": round(r.control_wer, 4),
                "gate_wer": round(r.gate_wer, 4),
            }
            for r in results
        ],
    }

    return summary


def compute_verdict(wer_native: float, wer_control: float, wer_gate: float, gap_control: float, gap_gate: float) -> str:
    """Compute verdict based on WER numbers."""
    # Check if probe is broken
    if gap_control > 0.10:
        return f"PROBE-INCONCLUSIVE: Librosa-mel→Whisper path is broken (gap {gap_control:.4f} >> WER_native {wer_native:.4f}). Mel filterbank/normalization mismatch needs investigation before rendering can be evaluated."

    # Check if render preserves signal
    if gap_gate < 0.05:
        return f"RENDERING-PASS (learnability-limited): WER_gate {wer_gate:.4f} ≈ WER_control {wer_control:.4f} (gap {gap_gate:.4f}). Render preserves phonetic info. Audio training failure is data/training-side, not render-side."

    if gap_gate < 0.15:
        return f"RENDERING-PARTIAL: WER_gate {wer_gate:.4f} vs WER_control {wer_control:.4f} (gap {gap_gate:.4f}). Moderate render loss. Consider architectural improvements (dynamic dimensions, better resize method)."

    return f"RENDERING-FAIL (rendering-limited): WER_gate {wer_gate:.4f} >> WER_control {wer_control:.4f} (gap {gap_gate:.4f}). Render destroys signal. Rendering must be redesigned (higher time resolution, no square-warp, or alternative representation)."


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-clips", type=int, default=30)
    args = parser.parse_args()

    summary = run_wer_gate_test(max_clips=args.max_clips)

    # Save result
    output_dir = Path("/workspace/univi/data/eval/decodability")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "wer_gate.json"

    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("="*80)
    print("WER GATE RESULTS")
    print("="*80)
    print(f"Clips tested: {summary['num_clips']}")
    print(f"\nWER_native (Whisper on raw audio):     {summary['wer_native_mean']:.4f}")
    print(f"WER_control (librosa-mel→Whisper):   {summary['wer_control_mean']:.4f}")
    print(f"WER_gate (PNG-render→mel→Whisper):   {summary['wer_gate_mean']:.4f}")
    print(f"\nGap (control - native):                {summary['gap_control_vs_native']:+.4f}  [probe artifact]")
    print(f"Gap (gate - control):                  {summary['gap_gate_vs_control']:+.4f}  [render loss]")
    print(f"\nVERDICT:\n{summary['verdict']}")
    print(f"\nResults saved: {results_file}")
    print("="*80)
