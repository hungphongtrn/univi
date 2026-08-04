import numpy as np

from PIL import Image
from data.preprocessing.render_utils import (
    TEXT_MARGIN,
    fill_boxes,
    render_text_page,
    render_text_pages,
    render_text_pages_with_boxes,
    render_log_mel_spectrogram,
    tile_spectrogram_image,
    word_boxes_for_pages,
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


def _randstr_text(n_words: int = 80, seed: int = 0) -> str:
    """The H14/H13 staged geometry: ``n_words`` groups of 5 lowercase letters."""
    import random

    rng = random.Random(seed)
    return " ".join(
        "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        for _ in range(n_words)
    )


def test_render_text_pages_with_boxes_matches_render_text_pages():
    """The paired helper must not perturb a single pixel of the plain renderer."""
    text = _randstr_text()
    pages_only = render_text_pages(text)
    pages, boxes = render_text_pages_with_boxes(text)
    assert [p.tobytes() for p in pages] == [p.tobytes() for p in pages_only]
    assert [b.word for b in boxes] == text.split(" ")


def test_word_boxes_cover_every_glyph():
    """Boxes and pixels must not drift: blanking every box must erase all ink.

    This is the invariant the H14 masked-region lane depends on — an occlusion
    that leaves part of a word visible would silently make the target wrong.
    Boxes are *advance* boxes, so a 1 px dilation (``LaneSpec.mask_pad``, whose
    default is 1) is required to swallow the antialiasing on the ``x1`` column.
    """
    for seed in range(5):
        pages, boxes = render_text_pages_with_boxes(_randstr_text(seed=seed))
        assert len(pages) == 1
        page = pages[0].convert("L")
        array = np.array(page)
        assert (array < 250).sum() > 0, "nothing was drawn"

        padded = [(b.x0 - 1, b.y0 - 1, b.x1 + 1, b.y1 + 1) for b in boxes]
        blanked = fill_boxes(page, padded, fill_color=255)
        assert int((np.array(blanked) < 250).sum()) == 0

        # ...and every box individually contains ink (no phantom/empty boxes).
        assert all((array[b.y0 : b.y1, b.x0 : b.x1] < 128).sum() > 0 for b in boxes)


def test_word_boxes_page_coordinates_and_paging():
    """Boxes are page-local, ordered, and start at the drawing margin."""
    text = "\n".join(f"line{i} word{i}" for i in range(120))
    pages, boxes = render_text_pages_with_boxes(text, canvas_height=256)
    assert len(pages) > 1
    assert [b.index for b in boxes] == list(range(len(boxes)))
    assert {b.page for b in boxes} == set(range(len(pages)))
    for box in boxes:
        assert box.y0 >= TEXT_MARGIN and box.y1 <= pages[0].height
        assert box.x0 >= TEXT_MARGIN and box.x1 <= pages[0].width
    # the first word of every line starts exactly at the left margin
    assert all(b.x0 == TEXT_MARGIN for b in boxes if b.word.startswith("line"))


def test_fill_boxes_changes_only_inside_rects():
    pages, boxes = render_text_pages_with_boxes(_randstr_text(seed=2))
    clean = np.array(pages[0].convert("L"))
    rects = [b.rect for b in boxes[3:9]]
    masked = np.array(fill_boxes(pages[0], rects, fill_color="black").convert("L"))

    covered = np.zeros(clean.shape, dtype=bool)
    for x0, y0, x1, y1 in rects:
        covered[y0:y1, x0:x1] = True
    assert (masked[covered] == 0).all(), "occluded rects are not solid fill"
    assert (masked[~covered] == clean[~covered]).all(), "pixels changed outside the rects"


def test_render_log_mel_spectrogram_creates_image():
    img = render_log_mel_spectrogram("tests/fixtures/test_tone.wav")
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (512, 512), f"Expected 512x512, got {img.size}"


def test_render_log_mel_spectrogram_uses_complete_audio():
    """5s and 15s clips both produce 512x512 images (no central cropping/padding)."""
    img_5s = render_log_mel_spectrogram("tests/fixtures/test_tone.wav")
    img_15s = render_log_mel_spectrogram("tests/fixtures/test_15s_tone.wav")
    assert isinstance(img_5s, Image.Image)
    assert isinstance(img_15s, Image.Image)
    # Both are resized to 512x512 regardless of audio duration
    assert img_5s.size == (512, 512)
    assert img_15s.size == (512, 512)
    # Content differs (different audio lengths produce different spectrograms)
    assert img_5s.tobytes() != img_15s.tobytes(), (
        "5s and 15s clips should produce different pixel content"
    )


def test_render_log_mel_spectrogram_creates_fixed_rectangular_pages():
    audio = np.zeros(15 * 16000, dtype=np.float32)
    pages = render_log_mel_spectrogram(
        audio,
        source_sample_rate=16000,
        page_duration_sec=10.0,
        max_pages=4,
        output_width=1000,
        output_height=160,
    )
    assert isinstance(pages, list)
    assert len(pages) == 2
    assert all(page.size == (1000, 160) for page in pages)
    assert all(page.mode == "RGB" for page in pages)

def test_render_log_mel_spectrogram_can_select_central_window():
    sample_rate = 16000
    core = np.sin(
        2 * np.pi * 440 * np.arange(sample_rate, dtype=np.float32) / sample_rate
    )
    quiet_edges = np.concatenate([np.zeros(sample_rate), core, np.zeros(sample_rate)])
    loud_edges = np.concatenate([np.ones(sample_rate), core, np.ones(sample_rate)])

    quiet_image = render_log_mel_spectrogram(
        quiet_edges,
        source_sample_rate=sample_rate,
        central_duration=1.0,
    )
    loud_image = render_log_mel_spectrogram(
        loud_edges,
        source_sample_rate=sample_rate,
        central_duration=1.0,
    )

    assert quiet_image.tobytes() == loud_image.tobytes()


def test_whisper_dynamic_range_maps_to_full_grayscale(monkeypatch):
    monkeypatch.setattr(
        "data.preprocessing.render_utils.librosa.feature.melspectrogram",
        lambda **kwargs: np.array([[1.0, 1e-4, 1e-8]], dtype=np.float32),
    )
    image = render_log_mel_spectrogram(
        np.ones(480, dtype=np.float32),
        source_sample_rate=16000,
        n_mels=1,
        hop_length=160,
        output_width=3,
        output_height=1,
    )
    assert [image.getpixel((x, 0))[0] for x in range(3)] == [0, 128, 255]



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
