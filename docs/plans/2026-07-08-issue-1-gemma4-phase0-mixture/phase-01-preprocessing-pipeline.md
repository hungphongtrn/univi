# Phase 1: Preprocessing Pipeline

## Phase Goal
Materialize all five source datasets as Hugging Face datasets in Gemma 4 multimodal `messages` format, verify the schema is correct for each source, and produce a concatenated-and-shuffled training mixture ready for `FastVisionModel`.

## Project Structure After Phase 1

```
univi/
├── .python-version              # Python 3.12.0 for uv-managed environment
├── pyproject.toml               # uv-managed project dependencies
├── uv.lock                      # Locked dependency graph
├── data/
│   └── preprocessing/
│       ├── __init__.py
│       ├── librispeech_asr.py       # Audio → log-mel spectrogram image
│       ├── densefusion.py           # Text-image description pairs
│       ├── fineweb_edu.py           # Raw text → rendered page image
│       ├── smoltalk.py              # Instruction → rendered page image
│       ├── valor32k.py              # Tri-modal visual bundle
│       ├── merge_mixture.py         # Concatenate + shuffle all sources
│       └── render_utils.py          # Shared rendering helpers (text, spectrogram)
├── tests/
│   ├── test_librispeech_asr.py
│   ├── test_densefusion.py
│   ├── test_fineweb_edu.py
│   ├── test_smoltalk.py
│   ├── test_valor32k.py
│   ├── test_merge_mixture.py
│   └── test_render_utils.py
```

## Tasks

### Task 0: Initialize uv Python project

**Files:**
- Create: `pyproject.toml`
- Create: `uv.lock`
- Create: `.python-version`

- [x] **Step 1: Initialize the project with uv**

Run: `uv init --python 3.12.0 --no-readme`

Expected: `pyproject.toml` and `.python-version` exist, with `requires-python = ">=3.12.0"`.

- [x] **Step 2: Add preprocessing dependencies with uv**

Run:

```bash
uv add 'Pillow>=10.0.0' 'librosa>=0.10.0' 'numpy>=1.24.0' 'matplotlib>=3.7.0' 'datasets>=2.14.0' 'soundfile>=0.12.0'
```

Expected: dependencies are recorded in `pyproject.toml` and locked in `uv.lock`.

- [x] **Step 3: Add pytest as a development dependency**

Run: `uv add --dev pytest`

Expected: `pytest` is recorded in the `dev` dependency group so later tasks can run `uv run pytest ...`.

- [x] **Step 4: Remove default uv application scaffold**

Remove the generated `main.py`, because Phase 1 defines preprocessing modules and CLIs rather than an application entry point.

### Task 1: Shared rendering utilities

**Files:**
- Create: `data/preprocessing/render_utils.py`
- Create: `data/preprocessing/__init__.py`
- Test: `tests/test_render_utils.py`

- [ ] **Step 1: Write text rendering utility**

```python
def render_text_page(
    text: str,
    canvas_width: int = 1024,
    font_size: int = 14,
    font_path: str | None = None,
    background_color: str = "white",
    text_color: str = "black",
) -> Image.Image:
    """Render text as a DeepSeek-OCR-style page image.

    Args:
        text: Text content to render (may include newlines).
        canvas_width: Fixed canvas width in pixels (default 1024).
        font_size: Font size in points (default 14, minimum 14 per conservative settings).
        font_path: Path to a .ttf file (default: search for DejaVu Sans Mono,
            Liberation Mono, or fall back to Pillow default).
        background_color: Canvas background color.
        text_color: Text color.

    Returns:
        Pillow Image with rendered text.
    """
    # TODO: implement using Pillow ImageDraw; auto-wrap text to fit canvas_width;
    #   use conservative compression (no high-density packing).
```

- [ ] **Step 2: Write spectrogram rendering utility**

```python
def render_log_mel_spectrogram(
    audio_path: str,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_fft: int = 400,        # 25 ms at 16 kHz → 400 samples
    hop_length: int = 160,   # 10 ms at 16 kHz → 160 samples
    window: str = "hann",
    central_duration: float = 10.0,
    power: float = 2.0,
) -> Image.Image:
    """Render audio as a Whisper-style log-mel spectrogram image.

    Args:
        audio_path: Path to audio file.
        sample_rate: Target sample rate in Hz (default 16000).
        n_mels: Number of mel bins (default 80).
        n_fft: FFT window size (default 400 = 25 ms).
        hop_length: STFT hop length (default 160 = 10 ms).
        window: STFT window type (default "hann").
        central_duration: Central audio window in seconds (default 10.0).
        power: Exponent for power spectrogram (default 2.0).

    Returns:
        Pillow Image of the log-mel spectrogram.
    """
    # TODO: implement using librosa; load audio, trim to central window,
    #   compute mel spectrogram, convert to dB, normalize, render as image.
```

- [ ] **Step 3: Create deterministic test fixtures**

    ```python
    # Generate synthetic audio tones for spectrogram tests.
    # Run before Step 4 (test writing).
    import numpy as np
    import soundfile as sf
    import os

    os.makedirs("tests/fixtures", exist_ok=True)

    sr = 16000
    t_5s = np.linspace(0, 5.0, int(sr * 5.0), endpoint=False)
    tone_5s = 0.5 * np.sin(2 * np.pi * 440 * t_5s)
    sf.write("tests/fixtures/test_tone.wav", tone_5s, sr)

    t_15s = np.linspace(0, 15.0, int(sr * 15.0), endpoint=False)
    tone_15s = 0.5 * np.sin(2 * np.pi * 440 * t_15s)
    sf.write("tests/fixtures/test_15s_tone.wav", tone_15s, sr)
    ```

    Also create `tests/fixtures/__init__.py` (empty) and ensure
    `data/preprocessing/__init__.py` exists (empty, per project structure).

- [ ] **Step 4: Write tests for render utilities**

```python
def test_render_text_page_creates_image():
    img = render_text_page("Hello world")
    assert isinstance(img, Image.Image)
    assert img.width == 1024
    assert img.height > 0

def test_render_text_page_uses_minimum_font_size():
    img = render_text_page("Test", font_size=14)
    assert isinstance(img, Image.Image)

def test_render_log_mel_spectrogram_creates_image():
    # Use a synthetic tone or a small bundled test audio fixture
    img = render_log_mel_spectrogram("tests/fixtures/test_tone.wav")
    assert isinstance(img, Image.Image)
    assert img.width > 0 and img.height > 0

def test_render_log_mel_spectrogram_central_window():
    # Verify that a >10s audio clip is trimmed to ~10s window
    img = render_log_mel_spectrogram("tests/fixtures/test_15s_tone.wav")
    assert isinstance(img, Image.Image)
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `python -m pytest tests/test_render_utils.py -v`
Expected: FAIL with import errors (render_text_page etc. not yet implemented)

- [ ] **Step 7: Implement both rendering functions**

Implement `render_text_page` using `PIL.ImageDraw` with text wrapping. Implement `render_log_mel_spectrogram` using `librosa` to compute the mel spectrogram, convert to dB, normalize to 0-255, and render with `matplotlib` or raw PIL.

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_render_utils.py -v`
Expected: PASS

- [ ] **Step 9: Verify uv-managed dependencies are available**

Run: `uv run python -c "import PIL, librosa, numpy, matplotlib, datasets, soundfile"`
Expected: PASS with no import errors.

### Task 2: LibriSpeech ASR preprocessing

**Files:**
- Create: `data/preprocessing/librispeech_asr.py`
- Test: `tests/test_librispeech_asr.py`

- [ ] **Step 1: Write the failing test**

```python
def test_librispeech_asr_preprocess_generates_messages():
    dataset = preprocess_librispeech_asr(
        subset="clean-100",
        max_samples=2,
        sample_rate=16000,
        n_mels=80,
    )
    assert len(dataset) == 2
    row = dataset[0]
    assert "messages" in row
    assert len(row["messages"]) == 2  # user + assistant
    user_content = row["messages"][0]["content"]
    assert user_content[0]["type"] == "image"  # spectrogram first
    assert user_content[1]["type"] == "text"
    assert "transcribe" in user_content[1]["text"].lower()
    assert row["messages"][1]["role"] == "assistant"
    assert len(row["messages"][1]["content"][0]["text"]) > 0

def test_librispeech_asr_metadata_present():
    dataset = preprocess_librispeech_asr(subset="clean-100", max_samples=1)
    row = dataset[0]
    assert "source_dataset_id" in row
    assert row["source_dataset_id"] == "openslr/librispeech_asr"
    assert "split" in row
    assert "row_id" in row
    assert "render_config" in row
    assert "modality_label" in row
    assert "preprocessing_version" in row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_librispeech_asr.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `preprocess_librispeech_asr` using `datasets.Dataset.map`**

Load `openslr/librispeech_asr` via `datasets.load_dataset`. Transform rows with `source.map(...)` — do **not** accumulate rows in a Python list and call `Dataset.from_list`, because that holds the full materialized source in memory and risks OOM on large splits.

For each row, load audio, call `render_log_mel_spectrogram`, construct `messages` with spectrogram image + `Transcribe the speech represented by this spectrogram image.` as user, transcript as assistant. Use `with_indices=True` so `row_id` has a stable `str(index)` fallback when the source `id` field is absent. Attach all 6 metadata fields: `source_dataset_id` (`"openslr/librispeech_asr"`), `split`, `row_id` (unique per split), `render_config` (spectrogram params used), `modality_label` (`"audio"`), `preprocessing_version` (semantic version string).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_librispeech_asr.py -v`
Expected: PASS

### Task 3: DenseFusion text-image description preprocessing

**Files:**
- Create: `data/preprocessing/densefusion.py`
- Test: `tests/test_densefusion.py`

- [ ] **Step 1: Write the failing test**

```python
def test_densefusion_preprocess_generates_messages():
    dataset = preprocess_densefusion(
        subset="default", max_samples=2
    )
    assert len(dataset) == 2
    row = dataset[0]
    assert "messages" in row
    user_content = row["messages"][0]["content"]
    assert user_content[0]["type"] == "image"  # source image first
    assert user_content[1]["type"] == "text"
    assert "describe" in user_content[1]["text"].lower()

def test_densefusion_metadata_present():
    dataset = preprocess_densefusion(subset="default", max_samples=1)
    row = dataset[0]
    assert row["source_dataset_id"] == "BAAI/DenseFusion-1M"
    assert "split" in row
    assert "row_id" in row
    assert "render_config" in row
    assert row["modality_label"] == "image-text"
    assert "preprocessing_version" in row
```

- [ ] **Step 2**

Run: `python -m pytest tests/test_densefusion.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `preprocess_densefusion`**

Load a `BAAI/DenseFusion-1M` subset, preserve source image, construct `messages` with image + `Describe this image.` as user, description as assistant. Attach all 6 metadata fields: `source_dataset_id` (`"BAAI/DenseFusion-1M"`), `split`, `row_id`, `render_config`, `modality_label` (`"image-text"`), `preprocessing_version`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_densefusion.py -v`
Expected: PASS

### Task 4: FineWeb-Edu raw text preprocessing

**Files:**
- Create: `data/preprocessing/fineweb_edu.py`
- Test: `tests/test_fineweb_edu.py`

- [ ] **Step 1: Write the failing test**

```python
def test_fineweb_edu_preprocess_generates_messages():
    dataset = preprocess_fineweb_edu(max_samples=2, max_chars=2000)
    assert len(dataset) == 2
    row = dataset[0]
    assert "messages" in row
    user_content = row["messages"][0]["content"]
    assert user_content[0]["type"] == "image"  # rendered text page
    assert "transcribe" in user_content[1]["text"].lower()
    # assistant text should match the raw text that was rendered
    assistant_text = row["messages"][1]["content"][0]["text"]
    assert len(assistant_text) > 0

def test_fineweb_edu_metadata_present():
    dataset = preprocess_fineweb_edu(max_samples=1, max_chars=2000)
    row = dataset[0]
    assert row["source_dataset_id"] == "HuggingFaceFW/fineweb-edu"
    assert "split" in row
    assert "row_id" in row
    assert "render_config" in row
    assert row["modality_label"] == "text"
    assert "preprocessing_version" in row
```

- [ ] **Step 2**

Run: `python -m pytest tests/test_fineweb_edu.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `preprocess_fineweb_edu`**

Load `HuggingFaceFW/fineweb-edu`, chunk text rows, call `render_text_page`, construct `messages` with rendered image + `Transcribe the text shown in the image.` as user, raw text as assistant. Attach all 6 metadata fields: `source_dataset_id` (`"HuggingFaceFW/fineweb-edu"`), `split`, `row_id`, `render_config`, `modality_label` (`"text"`), `preprocessing_version`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fineweb_edu.py -v`
Expected: PASS

### Task 5: SmolTalk instruction-following preprocessing

**Files:**
- Create: `data/preprocessing/smoltalk.py`
- Test: `tests/test_smoltalk.py`

- [ ] **Step 1: Write the failing test**

```python
def test_smoltalk_preprocess_generates_messages():
    dataset = preprocess_smoltalk(max_samples=2)
    assert len(dataset) == 2
    row = dataset[0]
    assert "messages" in row
    user_content = row["messages"][0]["content"]
    assert user_content[0]["type"] == "image"
    assert "follow" in user_content[1]["text"].lower()
    assert row["messages"][1]["role"] == "assistant"

def test_smoltalk_metadata_present():
    dataset = preprocess_smoltalk(max_samples=1)
    row = dataset[0]
    assert row["source_dataset_id"] == "HuggingFaceTB/smoltalk"
    assert "split" in row
    assert "row_id" in row
    assert "render_config" in row
    assert row["modality_label"] == "text"
    assert "preprocessing_version" in row
```

- [ ] **Step 2**

Run: `python -m pytest tests/test_smoltalk.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `preprocess_smoltalk`**

Load `HuggingFaceTB/smoltalk` with `config="all"` (default) via `load_dataset("HuggingFaceTB/smoltalk", config, split=split)`. Extract instruction/user turn, truncate to `max_chars=2000` (default) before calling `render_text_page`, construct `messages` with rendered instruction image + `Follow the instruction shown in the image.` as user, response as assistant. Attach all 6 metadata fields: `source_dataset_id` (`"HuggingFaceTB/smoltalk"`), `split`, `row_id`, `render_config` (includes `config` and `max_chars`), `modality_label` (`"text"`), `preprocessing_version`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_smoltalk.py -v`
Expected: PASS

### Task 6: Valor32k tri-modal visual bundle preprocessing

**Files:**
- Create: `data/preprocessing/valor32k.py`
- Test: `tests/test_valor32k.py`

- [ ] **Step 1: Write the failing test**

```python
def test_valor32k_preprocess_generates_messages():
    dataset = preprocess_valor32k(
        max_samples=3,  # 1 visual, 1 audio, 1 audio-visual
    )
    assert len(dataset) == 3
    for row in dataset:
        assert "messages" in row
        user_content = row["messages"][0]["content"]
        # First image is always the rendered question/options
        assert user_content[0]["type"] == "image"
        assert user_content[-1]["type"] == "text"  # last is instruction
        assert "answer" in user_content[-1]["text"].lower()

def test_valor32k_image_count_by_modality():
    dataset = preprocess_valor32k(max_samples=10)
    for row in dataset:
        modality = row.get("modality_label")
        images = [c for c in row["messages"][0]["content"] if c["type"] == "image"]
        if modality == "visual":
            assert len(images) == 5  # 1 question + 4 frames
        elif modality == "audio":
            assert len(images) == 2  # 1 question + 1 spectrogram
        elif modality == "audio-visual":
            assert len(images) == 6  # 1 question + 1 spectrogram + 4 frames

def test_valor32k_metadata_present():
    dataset = preprocess_valor32k(max_samples=1)
    row = dataset[0]
    assert row["source_dataset_id"] == "inesriahi/valor32k-avqa-v2"
    assert "split" in row
    assert "row_id" in row
    assert "render_config" in row
    assert row["modality_label"] in ("visual", "audio", "audio-visual")
    assert "preprocessing_version" in row
```

- [ ] **Step 2**

Run: `python -m pytest tests/test_valor32k.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `preprocess_valor32k`**

Load `inesriahi/valor32k-avqa-v2`. For each row, render question + options via `render_text_page`. For `audio`/`audio-visual`, extract audio and render spectrogram. For `visual`/`audio-visual`, sample 4 video frames. Construct `messages` in fixed order: question image, spectrogram (if present), frames (if present), then `Answer the multiple-choice question shown in the images. Reply with only A, B, C, or D.` Assistant: single letter answer. Use train split for training sets, validation/test for eval sets. Attach all 6 metadata fields: `source_dataset_id` (`"inesriahi/valor32k-avqa-v2"`), `split`, `row_id`, `render_config`, `modality_label` (`"visual"`/`"audio"`/`"audio-visual"`), `preprocessing_version`.

**Attached-media requirement & failure criterion:** Real Valor32k HF rows expose only QA metadata and `video_id` — they do **not** include decoded `audio` arrays or `frames`/`images`/`video_frames`/`image` columns. The preprocessor requires these media columns to be attached before calling. If required media is absent:
  - `audio` / `audio-visual` rows without an `audio` dict → `RuntimeError` mentioning `video_id`.
  - `visual` / `audio-visual` rows without frame columns → `RuntimeError` mentioning `video_id`.
  - See `decisions.md` for the full rationale.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_valor32k.py -v`
Expected: PASS

### Task 7: Merge mixture script

**Files:**
- Create: `data/preprocessing/merge_mixture.py`
- Test: `tests/test_merge_mixture.py`

- [ ] **Step 1: Write the failing test**

```python
def test_merge_mixture_concatenates_and_shuffles():
    sources = {
        "librispeech": preprocess_librispeech_asr(max_samples=5),
        "densefusion": preprocess_densefusion(max_samples=5),
        "fineweb": preprocess_fineweb_edu(max_samples=5),
        "smoltalk": preprocess_smoltalk(max_samples=5),
        "valor32k": preprocess_valor32k(max_samples=5),
    }
    merged = merge_and_shuffle(sources, seed=42)
    assert len(merged) == 25
    assert "messages" in merged[0]

def test_merge_mixture_preserves_metadata():
    sources = {
        "librispeech": preprocess_librispeech_asr(max_samples=2),
    }
    merged = merge_and_shuffle(sources, seed=42)
    row = merged[0]
    assert "source_dataset_id" in row
    assert "split" in row
    assert "row_id" in row
    assert "render_config" in row
    assert "modality_label" in row
    assert "preprocessing_version" in row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_merge_mixture.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `merge_and_shuffle`**

Concatenate all source datasets, shuffle with a fixed seed, return the merged dataset. Include source metadata column to track provenance. (Pushing to HF Hub is deferred — the CLI entry point in Step 7 writes locally via `save_to_disk`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_merge_mixture.py -v`
Expected: PASS

- [ ] **Step 5: Write CLI entry-point test**

```python
import subprocess
import tempfile

def test_merge_mixture_cli():
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [
                "python", "-m", "data.preprocessing.merge_mixture",
                "--librispeech-samples", "2",
                "--densefusion-samples", "2",
                "--fineweb-samples", "2",
                "--smoltalk-samples", "2",
                "--valor32k-samples", "2",
                "--output", tmp,
                "--seed", "42",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # Verify output dataset exists and is loadable
        from datasets import load_from_disk
        ds = load_from_disk(tmp)
        assert len(ds) == 10
        assert "messages" in ds[0]
        assert "source_dataset_id" in ds[0]
        assert "modality_label" in ds[0]
```

- [ ] **Step 6: Run CLI test to verify it fails**

Run: `python -m pytest tests/test_merge_mixture.py::test_merge_mixture_cli -v`
Expected: FAIL (no `__main__` block or argparse)

- [ ] **Step 7: Implement CLI entry point**

Add `if __name__ == "__main__"` block to `merge_mixture.py` using `argparse`. Accept per-source `--*-samples N` flags, `--output PATH`, and `--seed INT` (default 42). Load each source preprocessor with the given sample count, call `merge_and_shuffle`, write the merged dataset to `--output` via `datasets.Dataset.save_to_disk`.

```python
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech-samples", type=int, default=0)
    parser.add_argument("--densefusion-samples", type=int, default=0)
    parser.add_argument("--fineweb-samples", type=int, default=0)
    parser.add_argument("--smoltalk-samples", type=int, default=0)
    parser.add_argument("--valor32k-samples", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # ... build sources dict, call merge_and_shuffle, save
```

- [ ] **Step 8: Run CLI test to verify it passes**

Run: `python -m pytest tests/test_merge_mixture.py::test_merge_mixture_cli -v`
Expected: PASS

### Task 8: Integration validation

- [ ] **Step 1: Run all preprocessing tests**

```bash
python -m pytest tests/ -v
```

Expected: ALL PASS

- [ ] **Step 2: Run end-to-end materialization with small counts**

```bash
python -m data.preprocessing.merge_mixture \
    --librispeech-samples 10 \
    --densefusion-samples 10 \
    --fineweb-samples 10 \
    --smoltalk-samples 10 \
    --valor32k-samples 10 \
    --output ./data/materialized/smoke-v0
```

Expected: Script completes, 50 examples in output dataset, correct schema.

- [ ] **Step 3: Verify schema of output dataset**

```python
from datasets import load_from_disk
ds = load_from_disk("./data/materialized/smoke-v0")
```

Confirm each row has `messages` list with valid image objects and text content. Verify no row contains native answer-bearing text in the user instruction.

## Implementation Patterns

### Source preprocessors must use `datasets.Dataset.map`

All source preprocessors (`librispeech_asr`, `densefusion`, `fineweb_edu`, `smoltalk`, `valor32k`) must transform rows via `source.map(...)` rather than accumulating rows in a Python list and calling `Dataset.from_list`. The list-accumulation pattern materialises the entire resulting dataset in Python heap memory, which does not scale to splits with hundreds of thousands of examples.

Acceptable patterns:

- **Row-wise `map`**: The map function writes temporary files (e.g., audio → WAV for spectrogram rendering) and returns the new row dict. This is fine because rendering is already side-effecty and I/O-bound.
- **Batched `map`**: If the renderer supports it, use small bounded batches (e.g., `batched=True, batch_size=128`). Never collect all batches into an unbounded Python list.
- **Streaming**: Future phases may load source datasets with `streaming=True` for even lower memory overhead; `map` remains compatible with streaming datasets.
- **`with_indices=True`**: Always pass this when the `row_id` fallback needs the row index.

### `render_config` must be a stable JSON string

All preprocessors must serialize `render_config` with `json.dumps(config, sort_keys=True)` rather than storing a native dict. This ensures that future `datasets.concatenate_datasets` does not fail due to incompatible Arrow feature schemas for nested dict columns when different sources have different config keys. Consumers must parse the field with `json.loads` before inspecting.

### `row_id` must be a string

Coerce `row_id` to `str` in all preprocessors, even when the source `id` field is numeric: `str(row.get("id", index))`. This guarantees a consistent string feature type across concatenated sources.

## Phase Completion Criteria
- [ ] All 7 task test suites pass
- [ ] End-to-end materialization script produces a valid dataset with 5+ source examples
- [ ] Every user message has images before text instruction
- [ ] No user message contains native answer-bearing content
- [ ] Source metadata is present for every row with all 6 required fields: `source_dataset_id`, `split`, `row_id`, `render_config`, `modality_label`, `preprocessing_version`
- [ ] Dataset can be loaded by `datasets.load_from_disk` and iterated without errors

## Handoff Notes
- The smoke training script in Phase 2 expects the dataset to be available at `data/materialized/smoke-v0` or as a HF Hub dataset.
- The `render_text_page` and `render_log_mel_spectrogram` utilities in `render_utils.py` are shared; do not change their signatures without updating all consumers.
- Font and colormap choices in `render_utils.py` become experimental hyperparameters — document them in the metadata.
- The materialized dataset should be pushed to HF Hub (`hungphongtrn/univi-phase0-smoke-v0` for the smoke version) so the training script can load it from anywhere.
- If any source dataset's media files (`valor32k` video/audio for train split) are not directly accessible via HF dataset, document the resolution in decisions.md and use the available subset or fallback.
