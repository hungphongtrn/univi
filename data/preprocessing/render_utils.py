from __future__ import annotations

import librosa
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_SPECTROGRAM_UPSCALE = 4
_PIXEL_MAX = 255


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
    font_size = max(font_size, 14)
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


def render_text_pages(
    text: str,
    canvas_width: int = 1024,
    canvas_height: int = 1024,
    font_size: int = 14,
    font_path: str | None = None,
    background_color: str = "white",
    text_color: str = "black",
) -> list[Image.Image]:
    font_size = max(font_size, 14)
    font_file = _find_font(font_path)
    font = (
        ImageFont.truetype(font_file, font_size)
        if font_file is not None
        else ImageFont.load_default()
    )

    try:
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
    except AttributeError:
        line_height = font_size + 4

    margin = 20
    usable_width = canvas_width - 2 * margin
    lines_per_page = max(1, (canvas_height - 2 * margin) // line_height)
    wrapped_lines: list[str] = []

    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        line = ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if font.getlength(candidate) <= usable_width:
                line = candidate
                continue
            if line:
                wrapped_lines.append(line)
                line = ""
            while font.getlength(word) > usable_width:
                low, high = 1, len(word)
                while low < high:
                    middle = (low + high + 1) // 2
                    if font.getlength(word[:middle]) <= usable_width:
                        low = middle
                    else:
                        high = middle - 1
                wrapped_lines.append(word[:low])
                word = word[low:]
            line = word
        wrapped_lines.append(line)

    if not wrapped_lines:
        wrapped_lines = [""]

    pages = []
    for start in range(0, len(wrapped_lines), lines_per_page):
        page = Image.new(
            "RGB", (canvas_width, canvas_height), color=background_color
        )
        draw = ImageDraw.Draw(page)
        for line_index, line in enumerate(
            wrapped_lines[start : start + lines_per_page]
        ):
            draw.text(
                (margin, margin + line_index * line_height),
                line,
                font=font,
                fill=text_color,
            )
        pages.append(page)
    return pages


def _load_audio_samples(
    audio_source: str | np.ndarray,
    *,
    sample_rate: int,
    source_sample_rate: int | None,
) -> np.ndarray:
    if isinstance(audio_source, str):
        samples, _ = librosa.load(audio_source, sr=sample_rate, mono=True)
        return np.asarray(samples, dtype=np.float32)

    samples = np.asarray(audio_source, dtype=np.float32)
    if samples.ndim == 2:
        if samples.shape[0] <= 8:
            samples = samples.mean(axis=0)
        elif samples.shape[1] <= 8:
            samples = samples.mean(axis=1)
        else:
            raise ValueError(f"Cannot infer channel axis from audio shape {samples.shape}")
    elif samples.ndim != 1:
        raise ValueError(f"Expected mono or multi-channel audio, got shape {samples.shape}")

    if source_sample_rate is None:
        raise ValueError("source_sample_rate is required when rendering an audio array")
    if source_sample_rate != sample_rate:
        samples = librosa.resample(
            samples,
            orig_sr=source_sample_rate,
            target_sr=sample_rate,
        )
    return np.asarray(samples, dtype=np.float32)


def _whisper_log_mel(
    samples: np.ndarray,
    *,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    window: str,
    power: float,
) -> np.ndarray:
    spectrogram = librosa.feature.melspectrogram(
        y=samples,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        power=power,
    )
    expected_frames = max(1, len(samples) // hop_length)
    spectrogram = spectrogram[:, :expected_frames]
    log_mel = np.log10(np.maximum(spectrogram, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    whisper_values = (log_mel + 4.0) / 4.0
    return np.clip((whisper_values + 1.0) / 2.0, 0.0, 1.0)


def _render_log_mel_image(
    log_mel: np.ndarray,
    *,
    output_width: int,
    output_height: int,
) -> Image.Image:
    pixels = np.rint(_PIXEL_MAX * (1.0 - log_mel)).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    image = image.resize((output_width, output_height), Image.LANCZOS)
    return image.convert("RGB")


def render_log_mel_spectrogram(
    audio_source: str | np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    window: str = "hann",
    output_size: int = 512,
    power: float = 2.0,
    *,
    source_sample_rate: int | None = None,
    central_duration: float | None = None,
    page_duration_sec: float | None = None,
    max_pages: int | None = None,
    output_width: int | None = None,
    output_height: int | None = None,
    normalization: str = "whisper_log_mel",
) -> Image.Image | list[Image.Image]:
    if normalization != "whisper_log_mel":
        raise ValueError(f"Unsupported log-mel normalization: {normalization}")

    samples = _load_audio_samples(
        audio_source,
        sample_rate=sample_rate,
        source_sample_rate=source_sample_rate,
    )
    if samples.size == 0:
        raise ValueError("Cannot render empty audio")

    if central_duration is not None:
        if central_duration <= 0:
            raise ValueError("central_duration must be positive")
        if page_duration_sec is not None:
            raise ValueError(
                "central_duration and page_duration_sec are mutually exclusive"
            )
        target_samples = int(round(central_duration * sample_rate))
        if len(samples) > target_samples:
            start = (len(samples) - target_samples) // 2
            samples = samples[start : start + target_samples]
        elif len(samples) < target_samples:
            padding = target_samples - len(samples)
            samples = np.pad(samples, (padding // 2, padding - padding // 2))

    width = output_width or output_size
    height = output_height or output_size
    if page_duration_sec is None:
        log_mel = _whisper_log_mel(
            samples,
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            power=power,
        )
        return _render_log_mel_image(
            log_mel,
            output_width=width,
            output_height=height,
        )

    if page_duration_sec <= 0:
        raise ValueError("page_duration_sec must be positive")
    page_samples = int(round(page_duration_sec * sample_rate))
    page_count = max(1, int(np.ceil(len(samples) / page_samples)))
    if max_pages is not None and page_count > max_pages:
        raise ValueError(
            f"Audio requires {page_count} pages, exceeding max_pages={max_pages}"
        )

    padded_samples = np.pad(samples, (0, page_count * page_samples - len(samples)))
    log_mel = _whisper_log_mel(
        padded_samples,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        power=power,
    )
    page_frames = page_samples // hop_length
    return [
        _render_log_mel_image(
            log_mel[:, page_index * page_frames : (page_index + 1) * page_frames],
            output_width=width,
            output_height=height,
        )
        for page_index in range(page_count)
    ]


def tile_spectrogram_image(
    image: Image.Image,
    tile_size: int | None = None,
    pad_color: int = 255,
) -> list[Image.Image]:
    width, height = image.size
    if tile_size is None:
        tile_size = height
    tiles = []
    for x in range(0, width, tile_size):
        right = min(x + tile_size, width)
        tile = image.crop((x, 0, right, height))
        if tile.width < tile_size:
            padded = Image.new("RGB", (tile_size, tile_size), color=pad_color)
            padded.paste(tile, (0, 0))
            tile = padded
        tiles.append(tile)
    return tiles
