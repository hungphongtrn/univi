"""Decodability gate (mel-direct Whisper WER) for Univi's log-mel PNG render.

Three conditions on the SAME real librispeech clips, all decoded by the SAME
Whisper model. All feature paths use the repo's whisper_log_mel normalization,
which is byte-identical to transformers WhisperFeatureExtractor's, so a recovered
feature array feeds straight in as `input_features`.

  NATIVE  : raw audio -> Whisper's own feature extractor -> generate
  CONTROL : raw audio -> librosa mel + repo norm -> feature-direct -> generate
  GATE    : raw audio -> repo render_log_mel_spectrogram -> PNG -> recover mel
            -> feature-direct -> generate

Gaps:
  GATE - CONTROL  = information lost by the PNG render/resize (the real gate)
  CONTROL - NATIVE = librosa-vs-Whisper filterbank/norm artifact (probe overhead)
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # Xet client hangs on this box; use classic HTTPS
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root for `data` pkg

import numpy as np
import torch
from PIL import Image

from data.preprocessing.render_utils import render_log_mel_spectrogram

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
N_FRAMES = 3000  # Whisper 30s context
MODEL_ID = str(Path(__file__).resolve().parent / "whisper-tiny.en")  # local, hub-bypass
N_CLIPS = 30
MAX_DUR = 10.0  # single-page clips only (avoids multi-page stitching)
LIBRI_DIR = Path(__file__).resolve().parent / "LibriSpeech" / "test-clean"
OUT = Path(__file__).resolve().parent / "wer_gate.json"


def repo_whisper_values(mel_power: np.ndarray) -> np.ndarray:
    """Repo's whisper_log_mel normalization -> Whisper encoder-input scale."""
    log_mel = np.log10(np.maximum(mel_power, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    return (log_mel + 4.0) / 4.0


def librosa_features(audio: np.ndarray) -> np.ndarray:
    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP,
        window="hann", power=2.0,
    )
    return repo_whisper_values(mel)  # (80, T)


def recover_from_png(img: Image.Image) -> np.ndarray:
    """Invert the grayscale render back to (80, W) whisper-values."""
    arr = np.array(img.convert("L")).astype(np.float32)  # (H=160, W)
    normed01 = 1.0 - arr / 255.0
    wv = normed01 * 2.0 - 1.0                             # whisper-values scale
    h = wv.shape[0]
    if h != N_MELS:  # vertical 160 -> 80 (exact 2:1 average)
        assert h % N_MELS == 0, h
        wv = wv.reshape(N_MELS, h // N_MELS, wv.shape[1]).mean(axis=1)
    return wv  # (80, W)


def pad_or_trim(feat: np.ndarray) -> np.ndarray:
    t = feat.shape[1]
    if t < N_FRAMES:
        pad = np.full((N_MELS, N_FRAMES - t), feat.min(), dtype=feat.dtype)
        return np.concatenate([feat, pad], axis=1)
    return feat[:, :N_FRAMES]


_NORM = re.compile(r"[^a-z0-9 ]+")


def norm(text: str) -> str:
    return _NORM.sub(" ", text.lower()).strip()


def wer_pair(ref: str, hyp: str) -> tuple[int, int]:
    r, h = norm(ref).split(), norm(hyp).split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return int(d[len(r), len(h)]), len(r)


def load_local_clips() -> list[tuple[np.ndarray, str]]:
    """Read LibriSpeech test-clean FLACs + transcripts from local disk."""
    import soundfile as sf

    # transcripts: <spk>-<chap>.trans.txt, lines "<uttid> TEXT"
    transcripts: dict[str, str] = {}
    for tf in sorted(LIBRI_DIR.rglob("*.trans.txt")):
        for line in tf.read_text().splitlines():
            uid, _, text = line.partition(" ")
            transcripts[uid] = text
    clips: list[tuple[np.ndarray, str]] = []
    for flac in sorted(LIBRI_DIR.rglob("*.flac")):
        audio, sr = sf.read(str(flac), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        if len(audio) / SR > MAX_DUR:
            continue
        ref = transcripts.get(flac.stem)
        if not ref:
            continue
        clips.append((audio, ref))
        if len(clips) >= N_CLIPS:
            break
    return clips


def main() -> None:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device} model={MODEL_ID}", flush=True)
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device).eval()

    print(f"[data] reading local librispeech test-clean from {LIBRI_DIR} ...", flush=True)
    clips = load_local_clips()
    print(f"[data] collected {len(clips)} clips (<= {MAX_DUR}s)", flush=True)
    if not clips:
        raise SystemExit("no clips found — is test-clean extracted?")

    def generate(input_features: np.ndarray) -> str:
        feat = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            ids = model.generate(input_features=feat, max_new_tokens=200)
        return processor.batch_decode(ids, skip_special_tokens=True)[0]

    agg = {k: [0, 0] for k in ("native", "control", "gate")}  # [edits, ref_words]
    durs = []
    for i, (audio, ref) in enumerate(clips):
        durs.append(len(audio) / SR)
        # NATIVE
        nat_feat = processor(audio, sampling_rate=SR, return_tensors="np").input_features[0]
        hyp_nat = generate(nat_feat)
        # CONTROL
        hyp_ctl = generate(pad_or_trim(librosa_features(audio)))
        # GATE
        pages = render_log_mel_spectrogram(
            audio, sample_rate=SR, source_sample_rate=SR, n_mels=N_MELS,
            n_fft=N_FFT, hop_length=HOP, window="hann", power=2.0,
            page_duration_sec=10.0, max_pages=4, output_width=1000,
            output_height=160, normalization="whisper_log_mel",
        )
        page = pages[0] if isinstance(pages, list) else pages
        hyp_gate = generate(pad_or_trim(recover_from_png(page)))

        for key, hyp in (("native", hyp_nat), ("control", hyp_ctl), ("gate", hyp_gate)):
            e, n = wer_pair(ref, hyp)
            agg[key][0] += e
            agg[key][1] += n
        if i < 3:
            print(f"[ex{i}] REF : {norm(ref)[:90]}", flush=True)
            print(f"[ex{i}] NAT : {norm(hyp_nat)[:90]}", flush=True)
            print(f"[ex{i}] CTL : {norm(hyp_ctl)[:90]}", flush=True)
            print(f"[ex{i}] GATE: {norm(hyp_gate)[:90]}", flush=True)
        print(f"[{i+1}/{len(clips)}] done", flush=True)

    result = {
        "model": MODEL_ID,
        "n_clips": len(clips),
        "max_dur_s": MAX_DUR,
        "render": {"output_width": 1000, "output_height": 160, "page_duration_sec": 10.0},
        "mean_dur_s": round(float(np.mean(durs)), 2),
        "wer": {k: round(agg[k][0] / max(1, agg[k][1]), 4) for k in agg},
        "ref_words": {k: agg[k][1] for k in agg},
    }
    wn, wc, wg = result["wer"]["native"], result["wer"]["control"], result["wer"]["gate"]
    if wc > 0.5 and wc > 3 * wn:
        verdict = "PROBE-INCONCLUSIVE: control WER too high; mel-direct path broken (filterbank/norm)."
    elif wg <= wc + 0.05:
        verdict = "LEARNABILITY-LIMITED: render preserves phonetics (gate ~= control); failure is training-side."
    elif wg > wc + 0.15:
        verdict = "RENDERING-LIMITED: PNG render destroys signal (gate >> control); rebuild renderer."
    else:
        verdict = "AMBIGUOUS: gate modestly worse than control; render degrades but does not destroy."
    result["verdict"] = verdict
    OUT.write_text(json.dumps(result, indent=2))
    print("\n=== RESULT ===", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    print(f"\n{verdict}", flush=True)


if __name__ == "__main__":
    main()
