from PIL import Image
from data.preprocessing.render_utils import (
    render_text_page,
    render_text_pages,
    render_log_mel_spectrogram,
    tile_spectrogram_image,
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


def test_render_text_pages_preserves_readable_size_across_fixed_pages():
    pages = render_text_pages(
        "A short line\n" * 40,
        canvas_width=256,
        canvas_height=128,
        font_size=14,
    )

    assert len(pages) > 1
    assert all(page.size == (256, 128) for page in pages)


def test_render_log_mel_spectrogram_creates_image():
    img = render_log_mel_spectrogram("tests/fixtures/test_tone.wav")
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.width > 0 and img.height > 0


def test_render_log_mel_spectrogram_central_window():
    img_5s = render_log_mel_spectrogram("tests/fixtures/test_tone.wav")
    img_15s = render_log_mel_spectrogram("tests/fixtures/test_15s_tone.wav")
    assert isinstance(img_5s, Image.Image)
    assert isinstance(img_15s, Image.Image)
    assert (
        img_5s.width,
        img_5s.height,
    ) == (
        img_15s.width,
        img_15s.height,
    ), "5s and 15s clips should produce same spectrogram dimensions under 10s central window"


def test_tile_spectrogram_square_image_returns_one_tile():
    img = Image.new("RGB", (64, 64))
    tiles = tile_spectrogram_image(img)
    assert len(tiles) == 1
    assert tiles[0].size == (64, 64)


def test_tile_spectrogram_splits_wide_image():
    img = Image.new("RGB", (320, 64))
    tiles = tile_spectrogram_image(img)
    assert len(tiles) == 5
    for tile in tiles:
        assert tile.width == tile.height == 64


def test_tile_spectrogram_final_tile_padded():
    img = Image.new("RGB", (100, 64))
    tiles = tile_spectrogram_image(img)
    assert len(tiles) == 2
    assert tiles[0].size == (64, 64)
    assert tiles[1].size == (64, 64)
    assert tiles[0].getpixel((63, 0)) != tiles[1].getpixel((63, 0))


def test_tile_spectrogram_chronological_order():
    width, height = 128, 64
    img = Image.new("RGB", (width, height), color=0)
    for y in range(height):
        img.putpixel((1, y), (255, 0, 0))
    tiles = tile_spectrogram_image(img)
    assert len(tiles) == 2
    assert tiles[0].getpixel((1, 0)) == (255, 0, 0)
    assert tiles[1].getpixel((0, 0)) == (0, 0, 0)


def test_tile_spectrogram_custom_tile_size():
    img = Image.new("RGB", (200, 50))
    tiles = tile_spectrogram_image(img, tile_size=50)
    assert len(tiles) == 4
    assert tiles[0].size == (50, 50)
    assert tiles[3].size == (50, 50)
