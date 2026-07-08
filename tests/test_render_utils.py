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
    text = (
        "This is a slightly longer sentence that wraps to test"
        " that sub-14 font sizes are clamped to 14."
    )
    img_14 = render_text_page(text, font_size=14)
    img_8 = render_text_page(text, font_size=8)
    assert isinstance(img_14, Image.Image)
    assert isinstance(img_8, Image.Image)
    assert (
        img_14.width,
        img_14.height,
    ) == (
        img_8.width,
        img_8.height,
    ), "font_size=8 should produce identical layout to font_size=14"


def test_render_log_mel_spectrogram_creates_image():
    img = render_log_mel_spectrogram("tests/fixtures/test_tone.wav")
    assert isinstance(img, Image.Image)
    assert img.width > 0 and img.height > 0


def test_render_log_mel_spectrogram_central_window():
    img = render_log_mel_spectrogram("tests/fixtures/test_15s_tone.wav")
    assert isinstance(img, Image.Image)
