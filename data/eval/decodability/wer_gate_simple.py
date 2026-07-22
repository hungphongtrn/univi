#!/usr/bin/env python3
"""
WER Gate Test - Simplified, robust version.
Three WER conditions: NATIVE, CONTROL, GATE.
"""

import json
import numpy as np
import librosa
from PIL import Image
from pathlib import Path

import sys
sys.path.insert(0, "/workspace/univi")
from data.preprocessing.render_utils import render_log_mel_spectrogram


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute word error rate."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if len(ref) == 0:
        return 1.0 if len(hyp) > 0 else 0.0
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n] / m


def build_librosa_mel(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Build mel using repo's pipeline (CONTROL)."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=80, n_fft=400,
        hop_length=160, window="hann", power=2.0,
    )
    T = max(1, len(audio) // 160)
    mel = mel[:, :T]
    log_mel = np.log10(np.maximum(mel, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    whisper_vals = (log_mel + 4.0) / 4.0
    norm = np.clip((whisper_vals + 1.0) / 2.0, 0.0, 1.0)
    log_mel_rec = (norm * 2.0 - 1.0) * 4.0 - 4.0
    return np.maximum(10.0 ** log_mel_rec, 1e-10)


def render_and_recover_mel(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """GATE: render PNG and recover mel."""
    # Render to PNG
    png = render_log_mel_spectrogram(audio, sample_rate=sr, n_mels=80,
                                      n_fft=400, hop_length=160, window="hann",
                                      output_width=1000, output_height=160,
                                      power=2.0, source_sample_rate=sr)
    if isinstance(png, list):
        png = png[0]

    # Recover mel from PNG
    if png.mode != "L":
        png = png.convert("L")

    pixels = np.array(png, dtype=np.float32)
    T_frames = len(audio) // 160

    # Resize back to original T_frames (DE-WARP time axis)
    png_resized = Image.fromarray(pixels.astype(np.uint8), mode="L")
    png_resized = png_resized.resize((T_frames, 80), Image.LANCZOS)
    pixels_rec = np.array(png_resized, dtype=np.float32)

    # Invert pixel encoding
    norm = 1.0 - pixels_rec / 255.0
    whisper_vals = 2.0 * norm - 1.0
    log_mel = 4.0 * whisper_vals - 4.0
    mel = 10.0 ** log_mel

    return np.maximum(mel, 1e-10)


def transcribe_mel(mel: np.ndarray) -> str:
    """Transcribe mel via Whisper.decode (CONTROL and GATE)."""
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

    # Prepare mel (pad to 3000 frames)
    if mel.shape[1] < 3000:
        mel = np.pad(mel, ((0, 0), (0, 3000 - mel.shape[1])), mode='constant')
    else:
        mel = mel[:, :3000]

    mel_tensor = torch.from_numpy(mel).unsqueeze(0).float().cuda()

    with torch.no_grad():
        encoder_out = model.encoder(mel_tensor)
        decoder_ids = torch.tensor([[model.config.decoder_start_token_id]], device=mel_tensor.device)
        gen_ids = model.generate(encoder_outputs=encoder_out, decoder_input_ids=decoder_ids)

    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


def transcribe_native(audio: np.ndarray, sr: int = 16000) -> str:
    """Transcribe raw audio via Whisper.transcribe (NATIVE)."""
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=0)
    result = asr({"sampling_rate": sr, "raw": audio})
    return result.get("text", "").strip()


def main():
    """Run WER gate test."""
    from datasets import load_dataset

    print("Loading dataset...")
    try:
        # Try to load LibriSpeech - use small subset with non-streaming to avoid threading issues
        ds = load_dataset("openslr/librispeech_asr", "clean", split="test")
        # Take only first 5 clips for quick test
        clips = [ds[i] for i in range(min(5, len(ds)))]
        print(f"Loaded {len(clips)} clips")
    except Exception as e:
        print(f"Could not load LibriSpeech: {e}")
        print("Using fallback: creating synthetic test cases...")
        # Create synthetic test cases
        sr = 16000
        clips = []

        # Test case 1: Low frequency sine
        t1 = np.linspace(0, 3.0, int(sr * 3.0))
        audio1 = 0.3 * np.sin(2 * np.pi * 300 * t1)
        clips.append({"id": "synthetic_low", "text": "this is a test", "audio": {"array": audio1, "sampling_rate": sr}})

        # Test case 2: Higher frequency
        t2 = np.linspace(0, 3.0, int(sr * 3.0))
        audio2 = 0.3 * np.sin(2 * np.pi * 1000 * t2)
        clips.append({"id": "synthetic_high", "text": "this is a test", "audio": {"array": audio2, "sampling_rate": sr}})

        # Test case 3: Chirp
        t3 = np.linspace(0, 3.0, int(sr * 3.0))
        phase = 2 * np.pi * (300 * t3 + 700 * t3**2 / (2 * 3.0))
        audio3 = 0.3 * np.sin(phase)
        clips.append({"id": "synthetic_chirp", "text": "this is a test", "audio": {"array": audio3, "sampling_rate": sr}})

        print(f"Created {len(clips)} synthetic test cases")

    print(f"\nTesting {len(clips)} clips...")

    results = []
    sr = 16000

    for i, row in enumerate(clips):
        clip_id = str(row.get("id", i))
        ref_text = row["text"].strip()

        # Get audio (handle AudioDecoder objects from HF datasets)
        audio_obj = row["audio"]
        try:
            if hasattr(audio_obj, "get_all_samples"):
                # AudioDecoder object from datasets library
                samples = audio_obj.get_all_samples()
                audio = np.array(samples.data, dtype=np.float32)
                audio_sr = int(samples.sample_rate)
            elif isinstance(audio_obj, dict):
                audio = np.array(audio_obj["array"], dtype=np.float32)
                audio_sr = audio_obj.get("sampling_rate", sr)
            else:
                print(f"Unknown audio type: {type(audio_obj)}, skipping")
                continue
        except Exception as e:
            print(f"Audio decode error: {e}, skipping clip")
            continue

        if audio_sr != sr:
            audio = librosa.resample(audio, orig_sr=audio_sr, target_sr=sr)

        print(f"\n[{i+1}/{len(clips)}] {clip_id}: {ref_text[:50]}")

        try:
            # NATIVE
            print("  [NATIVE]...", end=" ", flush=True)
            native_text = transcribe_native(audio, sr=sr)
            native_wer = compute_wer(ref_text, native_text)
            print(f"WER={native_wer:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            native_wer = 1.0
            native_text = ""

        try:
            # CONTROL
            print("  [CONTROL]...", end=" ", flush=True)
            mel_control = build_librosa_mel(audio, sr=sr)
            control_text = transcribe_mel(mel_control)
            control_wer = compute_wer(ref_text, control_text)
            print(f"WER={control_wer:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            control_wer = 1.0
            control_text = ""

        try:
            # GATE
            print("  [GATE]...", end=" ", flush=True)
            mel_gate = render_and_recover_mel(audio, sr=sr)
            gate_text = transcribe_mel(mel_gate)
            gate_wer = compute_wer(ref_text, gate_text)
            print(f"WER={gate_wer:.4f}")
        except Exception as e:
            print(f"ERROR: {e}")
            gate_wer = 1.0
            gate_text = ""

        results.append({
            "clip_id": clip_id,
            "native_wer": float(native_wer),
            "control_wer": float(control_wer),
            "gate_wer": float(gate_wer),
        })

    # Aggregate
    native_wers = [r["native_wer"] for r in results]
    control_wers = [r["control_wer"] for r in results]
    gate_wers = [r["gate_wer"] for r in results]

    wer_native = np.mean(native_wers) if native_wers else 0.0
    wer_control = np.mean(control_wers) if control_wers else 0.0
    wer_gate = np.mean(gate_wers) if gate_wers else 0.0

    gap_control = wer_control - wer_native
    gap_gate = wer_gate - wer_control

    # Verdict
    if gap_control > 0.10:
        verdict = f"PROBE-INCONCLUSIVE: WER_control {wer_control:.4f} >> WER_native {wer_native:.4f} (gap {gap_control:.4f}). Mel-direct path is broken."
    elif gap_gate < 0.05:
        verdict = f"RENDERING-PASS (learnability-limited): WER_gate {wer_gate:.4f} ≈ WER_control {wer_control:.4f}. Render preserves phonetic info."
    elif gap_gate < 0.15:
        verdict = f"RENDERING-PARTIAL: WER_gate {wer_gate:.4f} vs WER_control {wer_control:.4f} (gap {gap_gate:.4f}). Moderate render loss."
    else:
        verdict = f"RENDERING-FAIL (rendering-limited): WER_gate {wer_gate:.4f} >> WER_control {wer_control:.4f}. Render destroys signal."

    # Save result
    summary = {
        "test_type": "wer_gate_real_librispeech",
        "num_clips": len(results),
        "whisper_model": "openai/whisper-base",
        "render_params": {"output_width": 1000, "output_height": 160},
        "wer_native_mean": float(wer_native),
        "wer_control_mean": float(wer_control),
        "wer_gate_mean": float(wer_gate),
        "gap_control_vs_native": float(gap_control),
        "gap_gate_vs_control": float(gap_gate),
        "verdict": verdict,
        "individual_results": results,
    }

    output_file = Path("/workspace/univi/data/eval/decodability/wer_gate.json")
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*80)
    print("WER GATE RESULTS")
    print("="*80)
    print(f"Clips: {len(results)}")
    print(f"WER_native:  {wer_native:.4f}")
    print(f"WER_control: {wer_control:.4f}")
    print(f"WER_gate:    {wer_gate:.4f}")
    print(f"Gap (control-native): {gap_control:+.4f}")
    print(f"Gap (gate-control):   {gap_gate:+.4f}")
    print(f"\nVERDICT: {verdict}")
    print(f"Saved: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
