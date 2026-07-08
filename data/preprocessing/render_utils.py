from __future__ import annotations

import math

import librosa
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _find_font(font_path: str | None = None) -> str | None:
    if font_path is not None:
        return font_path
    candidates = ["DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, 14).path
        except (OSError, IOError):
            continue
    return None


def render_text_page(
    text: str,
    canvas_width: int = 1024,
    font_size: int = 14,
    font_path: str | None = None,
    background_color: str = "white",
    text_color: str = "black",
) -> Image.Image:
    font_file = _find_font(font_path)
    if font_file is not None:
        font = ImageFont.truetype(font_file, font_size)
    else:
        font = ImageFont.load_default()

    try:
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
    except AttributeError:
        line_height = font_size + 4

    left_margin = 20
    right_margin = 20
    top_margin = 20
    bottom_margin = 20
    usable_width = canvas_width - left_margin - right_margin

    wrapped_lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        line = ""
        for word in words:
            test_line = f"{line} {word}".strip()
            bbox = font.getbbox(test_line)
            text_width = bbox[2] - bbox[0]
            if text_width > usable_width and line:
                wrapped_lines.append(line)
                line = word
            else:
                line = test_line
        if line:
            wrapped_lines.append(line)
        else:
            wrapped_lines.append("")

    canvas_height = max(
        top_margin + bottom_margin + len(wrapped_lines) * line_height,
        64,
    )

    img = Image.new("RGB", (canvas_width, canvas_height), color=background_color)
    draw = ImageDraw.Draw(img)

    y = top_margin
    for line in wrapped_lines:
        draw.text((left_margin, y), line, font=font, fill=text_color)
        y += line_height

    return img


def render_log_mel_spectrogram(
    audio_path: str,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    window: str = "hann",
    central_duration: float = 10.0,
    power: float = 2.0,
) -> Image.Image:
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    total_samples = len(y)
    target_samples = int(central_duration * sr)

    if total_samples > target_samples:
        start = (total_samples - target_samples) // 2
        y = y[start : start + target_samples]
    elif total_samples < target_samples:
        pad = target_samples - total_samples
        y = np.pad(y, (pad // 2, pad - pad // 2), mode="constant")

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        power=power,
    )
    log_S = librosa.power_to_db(S, ref=np.max)

    log_min = log_S.min()
    log_max = log_S.max()
    if log_max - log_min > 0:
        normalized = (log_S - log_min) / (log_max - log_min)
    else:
        normalized = np.zeros_like(log_S)

    normalized = (255 * (1.0 - normalized)).astype(np.uint8)

    height, width = normalized.shape
    img = Image.fromarray(normalized, mode="L").resize(
        (width * 4, height * 4), Image.LANCZOS
    )

    return img
