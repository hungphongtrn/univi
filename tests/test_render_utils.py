from PIL import Image
from data.preprocessing.render_utils import (
    render_text_page,
    render_log_mel_spectrogram,
)


def test_render_text_page_creates_image():
    img = render_text_page("Hello world")
    assert isinstance(img, Image.Image)
    assert img.width == 1024
    assert img.height > 0


def test_render_text_page_uses_minimum_font_size():
    img = render_text_page("Test", font_size=14)
    assert isinstance(img, Image.Image)


def test_render_log_mel_spectrogram_creates_image():
    img = render_log_mel_spectrogram("tests/fixtures/test_tone.wav")
    assert isinstance(img, Image.Image)
    assert img.width > 0 and img.height > 0


def test_render_log_mel_spectrogram_central_window():
    img = render_log_mel_spectrogram("tests/fixtures/test_15s_tone.wav")
    assert isinstance(img, Image.Image)
