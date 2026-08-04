"""Materialize a PRIOR-PROOF spoken-digit AUDIO lane (audio analog of the H3
random-string OCR diagnostic).

Motivation (user's concern): librispeech transcripts are natural English, so a
pretrained decoder can lower loss via its LANGUAGE PRIOR without reading the
spectrogram at all — and worse, the prior removes the *incentive* to learn to read
(the "audio drowned" trap). A target with ZERO language structure fixes this: the
only way below the guessing floor is to READ the spectrogram.

Each row concatenates a random sequence of spoken digits (from FSDD — real human
speech, digits 0–9) into one audio clip, renders it to a log-mel spectrogram with
the SAME production geometry librispeech uses (16 kHz, 80 mels, hop 160, 10 ms/px,
1000×160 page), and asks the model to transcribe the digit string. Digits are
uniform i.i.d., so the language prior gives NO shortcut — exactly like the
random-letter OCR lane, but for audio.

No-reading floor: for uniform i.i.d. digits over V=10, target entropy is
(n_digits * ln 10) nats; a non-reader can't beat n_digits*ln(10) / n_target_tokens.

Usage (CPU-only):
  uv run python -m data.preprocessing.spoken_digits \
      --out data/materialized/spoken-digits-v0 --train-rows 30000 --val-rows 1000
  # smoke:
  uv run python -m data.preprocessing.spoken_digits \
      --out data/materialized/spoken-digits-v0-smoke --train-rows 300 --val-rows 64
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset, Features, Image, Sequence
from datasets import Value as V

from data.preprocessing.render_utils import render_log_mel_spectrogram

logger = logging.getLogger(__name__)

FSDD_DIR = "data/raw/fsdd/recordings"
FSDD_SR = 8000              # FSDD native sample rate
N_DIGITS = 8               # digits per sequence (max ~8s audio → one 10s page)
GAP_SEC = 0.08             # silence between digit clips
PROMPT = "Transcribe the digits spoken in this spectrogram image."
SOURCE_ID = "synthetic/spoken-digits-fsdd"
PREP_VERSION = "0.1.0-spoken-digits"

# Production audio render config (matches librispeech_asr._render_defaults).
RENDER = dict(
    sample_rate=16000, n_mels=80, hop_length=160,
    page_duration_sec=10.0, output_width=1000, output_height=160,
)

FEATURES = Features(
    {
        "images": Sequence(Image()),
        "messages": [
            {"role": V("string"), "content": [{"type": V("string"), "text": V("string")}]}
        ],
        "source_dataset_id": V("string"),
        "split": V("string"),
        "row_id": V("string"),
        "render_config": V("string"),
        "modality_label": V("string"),
        "preprocessing_version": V("string"),
        "original_token_length": V("int64"),
        "source_split": V("string"),
    }
)


def _load_fsdd_pool() -> dict[int, list[np.ndarray]]:
    """Load every FSDD clip (8 kHz mono float32), grouped by digit 0–9."""
    import librosa

    pool: dict[int, list[np.ndarray]] = defaultdict(list)
    wavs = sorted(Path(FSDD_DIR).glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(
            f"No FSDD recordings under {FSDD_DIR} — download the dataset first."
        )
    for wav in wavs:
        digit = int(wav.name.split("_", 1)[0])  # "{digit}_{speaker}_{idx}.wav"
        samples, _ = librosa.load(str(wav), sr=FSDD_SR, mono=True)
        pool[digit].append(samples.astype(np.float32))
    logger.info("Loaded FSDD: %d clips across digits %s",
                sum(len(v) for v in pool.values()), sorted(pool))
    return pool


def _make_row_audio(rng: random.Random, pool: dict[int, list[np.ndarray]]):
    """Sample N random digits, concatenate a random clip of each with silence gaps.
    Returns (audio_8k, digit_string)."""
    gap = np.zeros(int(GAP_SEC * FSDD_SR), dtype=np.float32)
    digits = [rng.randrange(10) for _ in range(N_DIGITS)]
    parts: list[np.ndarray] = []
    for j, d in enumerate(digits):
        clip = rng.choice(pool[d])
        # peak-normalize each clip so volume differences don't dominate the mel.
        peak = float(np.max(np.abs(clip))) or 1.0
        parts.append(clip / peak)
        if j != len(digits) - 1:
            parts.append(gap)
    audio = np.concatenate(parts)
    target = " ".join(str(d) for d in digits)
    return audio, target


def _render_config_json() -> str:
    return json.dumps(
        {**RENDER, "source_sample_rate": FSDD_SR, "n_digits": N_DIGITS,
         "render_method": "log_mel_spectrogram", "digit_vocab": 10},
        sort_keys=True,
    )


def _build_split(split: str, n_rows: int, seed: int, pool, tokenizer):
    rng = random.Random(seed)
    rcfg = _render_config_json()
    rows = {k: [] for k in FEATURES}
    tok_total = 0
    for i in range(n_rows):
        audio, target = _make_row_audio(rng, pool)
        rendered = render_log_mel_spectrogram(
            audio, source_sample_rate=FSDD_SR, **RENDER
        )
        pages = rendered if isinstance(rendered, list) else [rendered]
        img_records = []
        for page in pages:
            buf = io.BytesIO()
            page.convert("L").save(buf, format="PNG")
            img_records.append({"bytes": buf.getvalue(), "path": None})

        otl = len(tokenizer(target, add_special_tokens=False)["input_ids"])
        tok_total += otl
        user_content = [{"type": "image", "text": None} for _ in pages]
        user_content.append({"type": "text", "text": PROMPT})
        rows["images"].append(img_records)
        rows["messages"].append(
            [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": target}]},
            ]
        )
        rows["source_dataset_id"].append(SOURCE_ID)
        rows["split"].append(split)
        rows["row_id"].append(f"spdigit-{split}-{i:07d}")
        rows["render_config"].append(rcfg)
        rows["modality_label"].append("audio")
        rows["preprocessing_version"].append(PREP_VERSION)
        rows["original_token_length"].append(otl)
        rows["source_split"].append(split)
        if (i + 1) % 5000 == 0:
            logger.info("  %s: rendered %d/%d", split, i + 1, n_rows)
    return Dataset.from_dict(rows, features=FEATURES), tok_total


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--train-rows", type=int, default=30_000)
    p.add_argument("--val-rows", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--tokenizer", default="unsloth/Qwen3-1.7B",
                   help="Decoder tokenizer for target token counts + floor.")
    args = p.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    pool = _load_fsdd_pool()

    out = Path(args.out)
    (out / "spoken-digits").mkdir(parents=True, exist_ok=True)

    logger.info("Building train split (%d rows)…", args.train_rows)
    train_ds, train_tok = _build_split("train", args.train_rows, args.seed, pool, tokenizer)
    logger.info("Building validation split (%d rows)…", args.val_rows)
    val_ds, val_tok = _build_split("validation", args.val_rows, args.seed + 10**6, pool, tokenizer)

    train_ds.save_to_disk(str(out / "spoken-digits" / "train"))
    val_ds.save_to_disk(str(out / "spoken-digits" / "validation"))

    manifest = {
        "artifact": out.name,
        "subsets": [
            {"config_name": "spoken-digits", "path": "spoken-digits/train",
             "split": "train", "is_training_split": True},
            {"config_name": "spoken-digits", "path": "spoken-digits/validation",
             "split": "validation", "is_training_split": False},
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # --- analytic no-reading floor ---
    h_digit = math.log(10.0)  # nats/digit, uniform i.i.d.
    val_digits_per_tok = (N_DIGITS * args.val_rows) / max(val_tok, 1)
    floor_per_token = h_digit * val_digits_per_tok
    floor = {
        "digit_vocab": 10,
        "n_random_digits_per_string": N_DIGITS,
        "H_digit_nats": h_digit,
        "val_avg_tokens_per_string": val_tok / max(args.val_rows, 1),
        "val_random_digits_per_token": val_digits_per_tok,
        "no_reading_floor_nats_per_token": floor_per_token,
        "note": (
            f"Eval CE (nats/tok) cannot go below ~{floor_per_token:.4f} without READING "
            "the spectrogram; a trained reader → ~0. Uniform i.i.d. digits ⇒ the language "
            "prior gives NO shortcut (unlike librispeech transcripts)."
        ),
    }
    (out / "floor.json").write_text(json.dumps(floor, indent=2))
    logger.info("Wrote %s (train=%d val=%d)", out, len(train_ds), len(val_ds))
    logger.info("NO-READING FLOOR ≈ %.4f nats/tok (H_digit=%.4f, %.3f digits/token).",
                floor_per_token, h_digit, val_digits_per_tok)


if __name__ == "__main__":
    main()
