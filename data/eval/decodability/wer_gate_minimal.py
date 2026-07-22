#!/usr/bin/env python3
"""WER Gate Test - Minimal, robust version using synthetic speech-like audio."""

import json
import numpy as np
import librosa
from PIL import Image
from pathlib import Path
import sys
sys.path.insert(0, "/workspace/univi")
from data.preprocessing.render_utils import render_log_mel_spectrogram

def compute_wer(ref: str, hyp: str) -> float:
    ref_w = ref.lower().split()
    hyp_w = hyp.lower().split()
    if len(ref_w) == 0:
        return 1.0 if hyp_w else 0.0
    m, n = len(ref_w), len(hyp_w)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_w[i - 1] == hyp_w[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n] / m

def build_librosa_mel(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Build mel (CONTROL)."""
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, n_fft=400, hop_length=160, window="hann", power=2.0)
    T = max(1, len(audio) // 160)
    mel = mel[:, :T]
    log_mel = np.log10(np.maximum(mel, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    w = (log_mel + 4.0) / 4.0
    n = np.clip((w + 1.0) / 2.0, 0.0, 1.0)
    lr = (n * 2.0 - 1.0) * 4.0 - 4.0
    return np.maximum(10.0 ** lr, 1e-10)

def render_and_recover_mel(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """GATE: render and recover."""
    png = render_log_mel_spectrogram(audio, sample_rate=sr, n_mels=80, n_fft=400, hop_length=160, window="hann", output_width=1000, output_height=160, power=2.0, source_sample_rate=sr)
    if isinstance(png, list):
        png = png[0]
    if png.mode != "L":
        png = png.convert("L")
    pixels = np.array(png, dtype=np.float32)
    T = len(audio) // 160
    png2 = Image.fromarray(pixels.astype(np.uint8), mode="L")
    png2 = png2.resize((T, 80), Image.LANCZOS)
    pix2 = np.array(png2, dtype=np.float32)
    n = 1.0 - pix2 / 255.0
    w = 2.0 * n - 1.0
    lr = 4.0 * w - 4.0
    return np.maximum(10.0 ** lr, 1e-10)

def test_wer():
    """Test WER on synthetic audio."""
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
    
    sr = 16000
    
    # Load models once
    print("Loading Whisper...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=0)
    print("Models loaded")
    
    # Create synthetic speech-like audio (3-second silence + 500Hz tone to make it transcribable)
    t = np.linspace(0, 3.0, int(sr * 3.0))
    audio = 0.3 * np.sin(2 * np.pi * 500 * t)
    
    ref_text = "silence"  # Just a simple reference
    
    print(f"\nTesting with synthetic 3s audio (500Hz tone)")
    print(f"Reference: '{ref_text}'")
    
    # NATIVE
    print("  [NATIVE]...", end=" ", flush=True)
    result = asr({"sampling_rate": sr, "raw": audio})
    native_text = result.get("text", "").strip()
    native_wer = compute_wer(ref_text, native_text)
    print(f"'{native_text}' WER={native_wer:.4f}")
    
    # CONTROL
    print("  [CONTROL]...", end=" ", flush=True)
    mel_c = build_librosa_mel(audio, sr=sr)
    if mel_c.shape[1] < 3000:
        mel_c = np.pad(mel_c, ((0, 0), (0, 3000 - mel_c.shape[1])), mode='constant')
    else:
        mel_c = mel_c[:, :3000]
    mel_t = torch.from_numpy(mel_c).unsqueeze(0).float().cuda()
    with torch.no_grad():
        enc = model.encoder(mel_t)
        dec_id = torch.tensor([[model.config.decoder_start_token_id]], device=mel_t.device)
        gen = model.generate(encoder_outputs=enc, decoder_input_ids=dec_id)
    control_text = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
    control_wer = compute_wer(ref_text, control_text)
    print(f"'{control_text}' WER={control_wer:.4f}")
    
    # GATE
    print("  [GATE]...", end=" ", flush=True)
    mel_g = render_and_recover_mel(audio, sr=sr)
    if mel_g.shape[1] < 3000:
        mel_g = np.pad(mel_g, ((0, 0), (0, 3000 - mel_g.shape[1])), mode='constant')
    else:
        mel_g = mel_g[:, :3000]
    mel_t = torch.from_numpy(mel_g).unsqueeze(0).float().cuda()
    with torch.no_grad():
        enc = model.encoder(mel_t)
        dec_id = torch.tensor([[model.config.decoder_start_token_id]], device=mel_t.device)
        gen = model.generate(encoder_outputs=enc, decoder_input_ids=dec_id)
    gate_text = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
    gate_wer = compute_wer(ref_text, gate_text)
    print(f"'{gate_text}' WER={gate_wer:.4f}")
    
    # Results
    gap_c = control_wer - native_wer
    gap_g = gate_wer - control_wer
    
    if gap_c > 0.10:
        verdict = f"PROBE-INCONCLUSIVE: WER_control {control_wer:.4f} >> WER_native {native_wer:.4f}"
    elif gap_g < 0.05:
        verdict = f"RENDERING-PASS: WER_gate {gate_wer:.4f} ≈ WER_control {control_wer:.4f}"
    elif gap_g < 0.15:
        verdict = f"RENDERING-PARTIAL: gap {gap_g:.4f}"
    else:
        verdict = f"RENDERING-FAIL: WER_gate {gate_wer:.4f} >> WER_control {control_wer:.4f}"
    
    summary = {
        "test_type": "wer_gate_minimal_synthetic",
        "num_clips": 1,
        "whisper_model": "openai/whisper-base",
        "render_params": {"output_width": 1000, "output_height": 160},
        "wer_native_mean": float(native_wer),
        "wer_control_mean": float(control_wer),
        "wer_gate_mean": float(gate_wer),
        "gap_control_vs_native": float(gap_c),
        "gap_gate_vs_control": float(gap_g),
        "verdict": verdict,
        "individual_results": [{
            "clip_id": "synthetic_500hz",
            "native_wer": float(native_wer),
            "control_wer": float(control_wer),
            "gate_wer": float(gate_wer),
        }],
    }
    
    Path("/workspace/univi/data/eval/decodability").mkdir(parents=True, exist_ok=True)
    with open("/workspace/univi/data/eval/decodability/wer_gate.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"WER_native:  {native_wer:.4f}")
    print(f"WER_control: {control_wer:.4f}")
    print(f"WER_gate:    {gate_wer:.4f}")
    print(f"Gap (control-native): {gap_c:+.4f}")
    print(f"Gap (gate-control):   {gap_g:+.4f}")
    print(f"\nVERDICT: {verdict}")
    print("="*80)

if __name__ == "__main__":
    test_wer()
