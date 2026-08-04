"""Materialize the H14 "masked-region targets" lane.

Rationale (H14): occlude random regions of the rendered page and make the
**target depend on the mask** — the model must transcribe what is visible and
emit a sentinel where something is hidden.  Which spans are occluded is drawn
i.i.d. per row, so:

- **(B) prior** — no language prior can predict the target, because the target
  is a function of a random mask (and, on the staged ``randstr`` lane, of random
  letters as well).
- **(C) scan** — to know a span is hidden the model must *localize* it on the
  page.  That is the cursor mechanism H11 says is missing.

Target format is the doc's recommended variant (b), one sentinel per masked
**run** (not per word)::

    words   : the cat sat on the mat          (run = ["sat"])
    target  : the cat <mask> on the mat

**Staging.** This is instantiated on the ``randstr`` generator geometry first
(groups of lowercase letters, font 14, 1024x1024 canvas) with long targets
(``--n-groups 80`` ~ 400 letters, i.e. the H13 ``d3`` rung), because that lane is
already prior-proof and already reads at +115%.  ``--text-source`` is the seam
for the real-text (fineweb) variant; it is deliberately **not implemented yet**
— port only after the mask mechanism works here.

Word length is **jittered** (``--group-len-min``/``--group-len-max``, default
3-7) rather than fixed at the randstr lane's 5.  A same-line run is merged into
one solid rectangle, so a fixed length would make the rectangle's *width* an
exact readout of the run length — leaking variant (c)'s ``<mask:k>`` through the
pixels and inflating span-boundary accuracy.  ``min == max`` restores the
fixed-length generator.

Guarding the doc's named risk (degenerate shortcut)
---------------------------------------------------
The model could learn to detect black boxes and emit sentinels *without reading
the surrounding text*.  Three stored columns make that separable at eval time:

- ``visible_words`` — the ordered visible-word sequence.  Score exact-match
  accuracy over these to get **visible-text accuracy**, independent of whether
  the sentinels landed correctly.  If this sits at the H09 floor while span
  boundaries are right, the model learned box detection only.
- ``sentinel_char_starts`` / ``sentinel_visible_index`` — where each sentinel
  begins in the target string, and how many visible words precede it.  Score
  **span-boundary accuracy** at exactly those positions (and the token after).
- ``words`` + ``mask_word_indices`` — the ground-truth occluded strings, so the
  **hallucinated-content rate** (fraction of masked words the model emits
  anyway) is computable.  A non-reader cannot know them: a masked word is
  ``L ~ U[3, 7]`` uniform letters, i.e. an ``E_L[26^-L]`` ~ 1.2e-5 guess (see
  ``floor.json["masked_word_guess_accuracy"]``).

Mask-permutation control
------------------------
The sharpest metric in the design is *same image, different mask*.  Each row
therefore also carries an **independent second mask** (``mask_b_*``) and its
ready-made target (``mask_b_target``), so an eval can render the B-masked page
with :func:`render_masked_pages` (deterministic, CPU-only, no re-derivation) and
score "did changing only the mask change the output?".

Note the extra columns (``words``, ``visible_words``, ``mask_*``, ``mask_b_*``)
make this lane's Features **incompatible with ``concatenate_datasets``** against
the other lanes — same caveat as ``poisoned_text.py``.  H14 is single-lane, so
this is fine; a future mixture must ``remove_columns`` them first.

Usage (CPU-only, no GPU needed):
  HF_HUB_OFFLINE=1 uv run python -m data.preprocessing.masked_regions \
      --out data/materialized/h14-masked-v0 --n-groups 80 \
      --train-rows 20000 --val-rows 500
  # tiny smoke set:
  HF_HUB_OFFLINE=1 uv run python -m data.preprocessing.masked_regions \
      --out /tmp/h14-smoke --train-rows 200 --val-rows 32
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, Features, Image, Sequence
from datasets import Value as V

from data.preprocessing.render_utils import (
    fill_boxes,
    render_text_pages_with_boxes,
)

logger = logging.getLogger(__name__)

CHARSET = "abcdefghijklmnopqrstuvwxyz"  # V=26, monospace-legible, no glyph ambiguity
# Letters per word, drawn uniformly from [GROUP_LEN_MIN, GROUP_LEN_MAX] per word.
#
# WHY JITTERED (H14 "OPEN — block width leaks run length"): same-line words of a
# masked run are merged into ONE solid rectangle, so with a FIXED length the
# rectangle's width is an exact function of the run length (5/11/17 monospace
# characters for a 1/2/3-word run — perfectly separable, MI = H(run length)).
# That hands the model roughly what target variant (c) `<mask:1>` would supply,
# destroying variant (b)'s length-blindness and making the degenerate
# "detect the box, emit a sentinel, never read" shortcut much easier.  With
# lengths jittered over 3..7 the per-run character widths overlap
# (1 word: 3-7, 2 words: 7-15, 3 words: 11-23) and width stops determining run
# length.  Set min == max to reproduce the old fixed-length behaviour.
GROUP_LEN_MIN = 3
GROUP_LEN_MAX = 7
N_GROUPS = 80      # words per page → ~400 random letters on average (the H13 d3 rung)
SENTINEL = "<mask>"
PROMPT = (
    "Transcribe the text shown in the image. "
    "Write {sentinel} in place of each blacked-out region."
)
CANVAS = 1024      # match fineweb-edu / random-strings render geometry
FONT_SIZE = 14
SOURCE_ID = "synthetic/masked-regions"
PREP_VERSION = "0.1.0-h14"

# Mask sampling defaults (the doc: "contiguous runs of 1-3 words, ~15-30% of
# words total").  MIN_GAP keeps at least one *visible* word between two runs:
# without it two adjacent runs would render as one uninterrupted black block
# while the target carried two sentinels, i.e. an unanswerable row.
RATE_MIN, RATE_MAX = 0.15, 0.30
RUN_MIN, RUN_MAX = 1, 3
MIN_GAP = 1


@dataclass(frozen=True)
class LaneSpec:
    """One configuration of the masked-region lane.

    ``group_len_min``/``group_len_max``/``n_groups`` come from
    ``random_strings.py`` (the staged generator), except that the letters-per-word
    is **jittered** here rather than fixed — see ``GROUP_LEN_MIN`` for why (a
    fixed length makes the occlusion rectangle's width a lossless readout of the
    run length).  ``group_len_min == group_len_max`` restores the fixed-length
    generator exactly.  ``rate_min``/``rate_max``/``run_min``/``run_max`` are the
    mask knobs H14 turns.

    ``mask_pad_x``/``mask_pad_y`` dilate each occlusion rectangle.  ``WordBox``
    is the *advance* box, and glyph antialiasing bleeds ~1 px past it — measured
    over 100 randstr pages, an undilated fill leaves 41 stray subpixels and
    every one of them is on the ``x1`` column.  Hence the defaults **(1, 0)**:
    measured to leave exactly zero ink, while keeping each rectangle strictly
    inside its own text line so an occlusion can never eat a pixel of the line
    above or below (``y0..y1`` already spans the full ascent+descent).  A
    leftover sliver would be a real leak — it tells a sharp reader something
    about a word it is not supposed to see.  Real text with accented capitals
    overshoots the ascent, so the fineweb port will need ``mask_pad_y>=2`` and
    must re-measure.
    """

    group_len_min: int = GROUP_LEN_MIN
    group_len_max: int = GROUP_LEN_MAX
    n_groups: int = N_GROUPS
    font_size: int = FONT_SIZE
    rate_min: float = RATE_MIN
    rate_max: float = RATE_MAX
    run_min: int = RUN_MIN
    run_max: int = RUN_MAX
    mask_pad_x: int = 1
    mask_pad_y: int = 0
    fill_color: str = "black"
    sentinel: str = SENTINEL
    max_pages: int = 1
    subset_name: str = "masked-randstr"

    @property
    def n_group_lengths(self) -> int:
        """How many distinct word lengths the generator can draw."""
        return self.group_len_max - self.group_len_min + 1

    @property
    def mean_group_len(self) -> float:
        return 0.5 * (self.group_len_min + self.group_len_max)

    @property
    def group_len_entropy_nats(self) -> float:
        """``H(L)`` — the entropy of a single word's LENGTH, in nats.

        Zero when the length is fixed.  A non-reader must predict where each
        visible word ends as well as which letters it contains, so this is real
        additional entropy that the no-reading floor may credit (see
        ``no_reading_floor_with_length_nats_per_token``).
        """
        return math.log(self.n_group_lengths)

    @property
    def mean_n_random_letters(self) -> float:
        return self.mean_group_len * self.n_groups

    @property
    def prompt(self) -> str:
        return PROMPT.format(sentinel=self.sentinel)


# messages/images schema identical to the other lanes (univi/trainer +
# univi.hybrid.data.HybridCollator); everything below `--- H14 additions` is
# additive and must be dropped before concatenating with another lane.
FEATURES = Features(
    {
        "images": Sequence(Image()),
        "messages": [
            {
                "role": V("string"),
                "content": [{"type": V("string"), "text": V("string")}],
            }
        ],
        "source_dataset_id": V("string"),
        "split": V("string"),
        "row_id": V("string"),
        "render_config": V("string"),
        "modality_label": V("string"),
        "preprocessing_version": V("string"),
        "original_token_length": V("int64"),
        "source_split": V("string"),
        # --- H14 additions ---------------------------------------------------
        # Ground truth for the page, mask-independent.
        "words": Sequence(V("string")),
        # Primary mask (the one that was rendered and supervised).
        "mask_rate": V("float32"),
        "mask_word_indices": Sequence(V("int32")),
        "mask_run_starts": Sequence(V("int32")),
        "mask_run_lengths": Sequence(V("int32")),
        "visible_words": Sequence(V("string")),
        "sentinel_char_starts": Sequence(V("int32")),
        "sentinel_visible_index": Sequence(V("int32")),
        # Independent second mask for the mask-permutation control: same image
        # content, different occlusion ⇒ a reading model must change its output.
        "mask_b_word_indices": Sequence(V("int32")),
        "mask_b_run_starts": Sequence(V("int32")),
        "mask_b_run_lengths": Sequence(V("int32")),
        "mask_b_target": V("string"),
    }
)


# ---------------------------------------------------------------------------
# Mask sampling / target construction  (public: an eval reuses these verbatim)
# ---------------------------------------------------------------------------


def sample_mask_runs(rng: random.Random, n_words: int, spec: LaneSpec) -> list[tuple[int, int]]:
    """Sample occlusion runs as ``[(start_word_index, run_length), ...]``.

    Runs are ``run_min..run_max`` words long, separated by at least ``MIN_GAP``
    visible words, and cover a per-row rate drawn uniformly from
    ``[rate_min, rate_max]``.  Runs may cross a rendered line break — that is
    still one contiguous span *in reading order*, and the gap rule guarantees a
    visible word separates any two distinct runs, so "one sentinel per
    uninterrupted occlusion" stays unambiguous.
    """
    if n_words <= 0:
        return []
    rate = rng.uniform(spec.rate_min, spec.rate_max)
    want = max(1, min(n_words, round(rate * n_words)))

    blocked: set[int] = set()   # words that are masked or adjacent to a mask
    masked: set[int] = set()
    runs: list[tuple[int, int]] = []
    total = 0
    # Bounded rejection sampling: the placement gets harder as the page fills,
    # so cap attempts rather than spin.  A short row simply ends up nearer
    # rate_min, which the measured-rate report at the end will show.
    for _ in range(200 * max(1, want)):
        if total >= want:
            break
        length = rng.randint(spec.run_min, spec.run_max)
        length = max(1, min(length, want - total, n_words))
        start = rng.randrange(0, n_words - length + 1)
        span = range(start - MIN_GAP, start + length + MIN_GAP)
        if any(i in blocked for i in span):
            continue
        runs.append((start, length))
        for i in range(start, start + length):
            masked.add(i)
        blocked.update(span)
        total += length
    runs.sort()
    return runs


def build_target(
    words: list[str], runs: list[tuple[int, int]], sentinel: str = SENTINEL
) -> dict:
    """Render variant (b): visible words verbatim, one *sentinel per run*.

    Returns the supervised ``target`` plus everything an eval needs to score the
    two halves of the task separately (see the module docstring):
    ``visible_words``, ``sentinel_char_starts`` (character offsets into
    ``target``), ``sentinel_visible_index`` (how many visible words precede each
    sentinel) and ``masked_indices``.
    """
    run_at = {start: length for start, length in runs}
    masked_indices = [i for start, length in runs for i in range(start, start + length)]

    pieces: list[str] = []
    visible_words: list[str] = []
    sentinel_visible_index: list[int] = []
    sentinel_piece_index: list[int] = []

    i = 0
    while i < len(words):
        if i in run_at:
            sentinel_piece_index.append(len(pieces))
            sentinel_visible_index.append(len(visible_words))
            pieces.append(sentinel)
            i += run_at[i]
        else:
            visible_words.append(words[i])
            pieces.append(words[i])
            i += 1

    target = " ".join(pieces)
    # Character offsets: pieces are joined by exactly one space, so the start of
    # piece k is sum(len(p)+1 for p in pieces[:k]).
    offsets: list[int] = []
    running = 0
    for k, piece in enumerate(pieces):
        offsets.append(running)
        running += len(piece) + 1
    sentinel_char_starts = [offsets[k] for k in sentinel_piece_index]

    return {
        "target": target,
        "visible_words": visible_words,
        "masked_indices": sorted(masked_indices),
        "sentinel_char_starts": sentinel_char_starts,
        "sentinel_visible_index": sentinel_visible_index,
    }


def mask_rects(
    boxes, runs: list[tuple[int, int]], pad_x: int = 1, pad_y: int = 0
) -> list[list[tuple]]:
    """Occlusion rectangles per page: ``rects_by_page[page] = [(x0,y0,x1,y1), ...]``.

    Words of one run that share a rendered line are merged into a single solid
    rectangle (the inter-word space is filled too, so the run reads as one
    block).  A run that wraps produces one rectangle per line it touches — still
    one sentinel, because it is one contiguous span in reading order.
    """
    n_pages = max((b.page for b in boxes), default=0) + 1
    rects: list[list[tuple]] = [[] for _ in range(n_pages)]
    for start, length in runs:
        group: list = []
        for index in range(start, start + length):
            box = boxes[index]
            if group and (box.page, box.line) == (group[-1].page, group[-1].line):
                group.append(box)
                continue
            if group:
                rects[group[0].page].append(_merge(group, pad_x, pad_y))
            group = [box]
        if group:
            rects[group[0].page].append(_merge(group, pad_x, pad_y))
    return rects


def _merge(group: list, pad_x: int, pad_y: int) -> tuple[int, int, int, int]:
    return (
        min(b.x0 for b in group) - pad_x,
        min(b.y0 for b in group) - pad_y,
        max(b.x1 for b in group) + pad_x,
        max(b.y1 for b in group) + pad_y,
    )


def render_masked_pages(
    words: list[str], runs: list[tuple[int, int]], spec: LaneSpec
) -> tuple[list, list]:
    """``(masked_pages, word_boxes)`` for *words* occluded at *runs*.

    Deterministic and CPU-only, so an eval can build the mask-permutation
    control image from the stored ``words`` + ``mask_b_run_*`` without
    re-deriving any geometry.  The clean page is just ``runs=[]``.
    """
    text = " ".join(words)
    pages, boxes = render_text_pages_with_boxes(
        text, canvas_width=CANVAS, canvas_height=CANVAS, font_size=spec.font_size
    )
    if [b.word for b in boxes] != words:
        # wrap_text_pages hard-wraps a word wider than the usable width, which
        # would desynchronise word indices from the rendered boxes and silently
        # mask the wrong glyphs.
        raise ValueError(
            "rendered word sequence does not match the source words "
            f"({len(boxes)} boxes vs {len(words)} words) — a word was hard-wrapped."
        )
    if len(pages) > spec.max_pages:
        raise ValueError(
            f"{spec.subset_name}: {len(words)} words at font {spec.font_size} need "
            f"{len(pages)} pages on a {CANVAS}px canvas (max_pages={spec.max_pages}). "
            "Lower --n-groups/--font-size, or raise --max-pages."
        )
    rects = mask_rects(boxes, runs, spec.mask_pad_x, spec.mask_pad_y)
    masked = [
        fill_boxes(page, rects[i] if i < len(rects) else [], spec.fill_color)
        for i, page in enumerate(pages)
    ]
    return masked, boxes


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def _make_words(rng: random.Random, spec: LaneSpec) -> list[str]:
    """``n_groups`` random words, each ``U[group_len_min, group_len_max]`` letters.

    The LENGTH is drawn before the letters so that a fixed spec
    (``group_len_min == group_len_max``) consumes the RNG identically to a plain
    fixed-length generator apart from one extra draw per word.
    """
    words: list[str] = []
    for _ in range(spec.n_groups):
        length = rng.randint(spec.group_len_min, spec.group_len_max)
        words.append("".join(rng.choice(CHARSET) for _ in range(length)))
    return words


def _render_config_json(spec: LaneSpec) -> str:
    return json.dumps(
        {
            "canvas_height": CANVAS,
            "canvas_width": CANVAS,
            "font_size": spec.font_size,
            "image_mode": "L",
            "render_method": "text_page",
            "charset_size": len(CHARSET),
            "group_len_min": spec.group_len_min,
            "group_len_max": spec.group_len_max,
            "group_len_jittered": spec.group_len_min != spec.group_len_max,
            "mean_group_len": spec.mean_group_len,
            "n_groups": spec.n_groups,
            "mean_n_random_letters": spec.mean_n_random_letters,
            "mask_rate_min": spec.rate_min,
            "mask_rate_max": spec.rate_max,
            "mask_run_min": spec.run_min,
            "mask_run_max": spec.run_max,
            "mask_min_gap": MIN_GAP,
            "mask_pad_x": spec.mask_pad_x,
            "mask_pad_y": spec.mask_pad_y,
            "mask_fill_color": spec.fill_color,
            "sentinel": spec.sentinel,
            "target_format": "sentinel_per_run",
            "max_pages": spec.max_pages,
        },
        sort_keys=True,
    )


def _make_row(index: int, split: str, seed: int, tokenizer, spec: LaneSpec) -> dict:
    """Build one row: words, primary mask A (rendered + supervised), control mask B.

    The three RNG streams are seeded from ``(seed, subset, split, index, tag)``
    rather than drawn from one sequential stream, so a row is reproducible on
    its own and ``mask_b`` is genuinely independent of ``mask_a``.
    """
    stem = f"{seed}|{spec.subset_name}|{split}|{index}"
    words = _make_words(random.Random(f"{stem}|words"), spec)

    rng_a = random.Random(f"{stem}|mask_a")
    rng_b = random.Random(f"{stem}|mask_b")
    runs_a = sample_mask_runs(rng_a, len(words), spec)
    built_a = build_target(words, runs_a, spec.sentinel)

    # The control is only informative if it actually changes the target.
    runs_b = sample_mask_runs(rng_b, len(words), spec)
    built_b = build_target(words, runs_b, spec.sentinel)
    for _ in range(50):
        if built_b["target"] != built_a["target"]:
            break
        runs_b = sample_mask_runs(rng_b, len(words), spec)
        built_b = build_target(words, runs_b, spec.sentinel)
    else:
        raise ValueError(f"row {index}: could not draw a mask_b distinct from mask_a")

    pages, _ = render_masked_pages(words, runs_a, spec)
    image_bytes = []
    for page in pages:
        buf = io.BytesIO()
        page.convert("L").save(buf, format="PNG", compress_level=1)
        image_bytes.append(buf.getvalue())

    target = built_a["target"]
    otl = len(tokenizer(target, add_special_tokens=False)["input_ids"])

    return {
        "images": image_bytes,
        "messages": [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "text": None} for _ in image_bytes],
                    {"type": "text", "text": spec.prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ],
        "source_dataset_id": f"synthetic/{spec.subset_name}",
        "split": split,
        "row_id": f"{spec.subset_name}-{split}-{index:07d}",
        "render_config": _render_config_json(spec),
        "modality_label": "text",
        "preprocessing_version": PREP_VERSION,
        "original_token_length": otl,
        "source_split": split,
        "words": words,
        "mask_rate": len(built_a["masked_indices"]) / max(len(words), 1),
        "mask_word_indices": built_a["masked_indices"],
        "mask_run_starts": [s for s, _ in runs_a],
        "mask_run_lengths": [n for _, n in runs_a],
        "visible_words": built_a["visible_words"],
        "sentinel_char_starts": built_a["sentinel_char_starts"],
        "sentinel_visible_index": built_a["sentinel_visible_index"],
        "mask_b_word_indices": built_b["masked_indices"],
        "mask_b_run_starts": [s for s, _ in runs_b],
        "mask_b_run_lengths": [n for _, n in runs_b],
        "mask_b_target": built_b["target"],
    }


@dataclass
class SplitStats:
    """Accounting the analytic floor is derived from.

    ``visible_letters`` is summed from the ACTUAL word strings, not from
    ``len(visible_words) * group_len`` — with a jittered length the latter is
    simply wrong, and it was the derivation the fixed-length version used.
    """

    target_tokens: int = 0
    visible_letters: int = 0
    visible_words: int = 0
    masked_letters: int = 0
    masked_words: int = 0
    word_length_histogram: dict[int, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.word_length_histogram is None:
            self.word_length_histogram = {}


def _build_split(split: str, n_rows: int, seed: int, tokenizer, spec: LaneSpec):
    rows = {key: [] for key in FEATURES}
    stats = SplitStats()
    for i in range(n_rows):
        row = _make_row(i, split, seed, tokenizer, spec)
        for key, value in row.items():
            rows[key].append(value)
        stats.target_tokens += row["original_token_length"]
        stats.visible_words += len(row["visible_words"])
        stats.visible_letters += sum(len(w) for w in row["visible_words"])
        masked = [row["words"][j] for j in row["mask_word_indices"]]
        stats.masked_words += len(masked)
        stats.masked_letters += sum(len(w) for w in masked)
        for word in row["words"]:
            stats.word_length_histogram[len(word)] = (
                stats.word_length_histogram.get(len(word), 0) + 1
            )
        if (i + 1) % 5000 == 0:
            logger.info("  %s: rendered %d/%d", split, i + 1, n_rows)
    ds = Dataset.from_dict(rows, features=FEATURES)
    return ds, stats


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--train-rows", type=int, default=20_000)
    p.add_argument("--val-rows", type=int, default=500)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument(
        "--tokenizer",
        default="data/checkpoints/encoder-free-v0/best",
        help="Local dir with tokenizer.json (for real target token counts + floor).",
    )
    p.add_argument(
        "--text-source",
        default="random",
        choices=["random", "fineweb"],
        help=(
            "Where the page text comes from. 'random' = the prior-proof randstr "
            "generator H14 stages on. 'fineweb' is the real-text port and is NOT "
            "built yet (see docs/hypothesis/todo/H14-masked-region-targets.md: "
            "port only after the mask mechanism works on randstr)."
        ),
    )
    # --- generator geometry (mirrors random_strings.py) --------------------
    p.add_argument(
        "--group-len-min",
        type=int,
        default=GROUP_LEN_MIN,
        help=(
            "Minimum letters per word. Word length is drawn uniformly from "
            "[--group-len-min, --group-len-max]. The jitter is what stops the "
            "occlusion rectangle's WIDTH from revealing how many words a run hid "
            "(with a fixed length the widths are perfectly separable). Set "
            "--group-len-min == --group-len-max for the old fixed-length lane."
        ),
    )
    p.add_argument(
        "--group-len-max", type=int, default=GROUP_LEN_MAX, help="Maximum letters per word."
    )
    p.add_argument("--n-groups", type=int, default=N_GROUPS, help="Words per page.")
    p.add_argument("--font-size", type=int, default=FONT_SIZE)
    p.add_argument("--max-pages", type=int, default=1)
    # --- mask knobs ---------------------------------------------------------
    p.add_argument("--rate-min", type=float, default=RATE_MIN)
    p.add_argument("--rate-max", type=float, default=RATE_MAX)
    p.add_argument("--run-min", type=int, default=RUN_MIN)
    p.add_argument("--run-max", type=int, default=RUN_MAX)
    p.add_argument(
        "--mask-pad-x",
        type=int,
        default=1,
        help=(
            "Horizontal dilation of each occlusion rect (px). 1 erases every glyph "
            "on the a-z charset (0 leaves ~0.4 antialiased subpixels/page, all on "
            "the box's right edge)."
        ),
    )
    p.add_argument(
        "--mask-pad-y",
        type=int,
        default=0,
        help=(
            "Vertical dilation (px). 0 keeps each rect inside its own text line, so "
            "an occlusion cannot touch the line above/below. Real text with accented "
            "capitals needs >=2 (re-measure before using)."
        ),
    )
    p.add_argument("--fill-color", default="black")
    p.add_argument("--sentinel", default=SENTINEL)
    p.add_argument(
        "--subset-name",
        default="masked-randstr",
        help="Subset dir + config_name. Must be listed in univi.trainer.VALID_SUBSETS.",
    )
    args = p.parse_args(argv)

    if args.text_source != "random":
        raise NotImplementedError(
            "--text-source=fineweb (the real-text H14 variant) is not built yet. "
            "H14 stages on the prior-proof randstr lane first; port to real text "
            "only after the mask mechanism works there."
        )
    if not 0.0 < args.rate_min <= args.rate_max < 1.0:
        raise ValueError("require 0 < --rate-min <= --rate-max < 1")
    if not 1 <= args.run_min <= args.run_max:
        raise ValueError("require 1 <= --run-min <= --run-max")
    if not 1 <= args.group_len_min <= args.group_len_max:
        raise ValueError("require 1 <= --group-len-min <= --group-len-max")

    spec = LaneSpec(
        group_len_min=args.group_len_min,
        group_len_max=args.group_len_max,
        n_groups=args.n_groups,
        font_size=args.font_size,
        rate_min=args.rate_min,
        rate_max=args.rate_max,
        run_min=args.run_min,
        run_max=args.run_max,
        mask_pad_x=args.mask_pad_x,
        mask_pad_y=args.mask_pad_y,
        fill_color=args.fill_color,
        sentinel=args.sentinel,
        max_pages=args.max_pages,
        subset_name=args.subset_name,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    out = Path(args.out)
    (out / spec.subset_name).mkdir(parents=True, exist_ok=True)

    logger.info(
        "Lane %s: %d words x %d-%d letters/page, font %d, mask %.0f-%.0f%% in runs of %d-%d",
        spec.subset_name, spec.n_groups, spec.group_len_min, spec.group_len_max,
        spec.font_size,
        100 * spec.rate_min, 100 * spec.rate_max, spec.run_min, spec.run_max,
    )
    logger.info("Building train split (%d rows)…", args.train_rows)
    train_ds, _ = _build_split("train", args.train_rows, args.seed, tokenizer, spec)
    logger.info("Building validation split (%d rows)…", args.val_rows)
    # disjoint seed stream for validation
    val_ds, val_stats = _build_split(
        "validation", args.val_rows, args.seed + 10**6, tokenizer, spec
    )

    train_ds.save_to_disk(str(out / spec.subset_name / "train"))
    val_ds.save_to_disk(str(out / spec.subset_name / "validation"))

    # Merge into an existing manifest rather than clobbering it, so several mask
    # rates can share ONE root (train on the mixture, eval each rate).
    manifest_path = out / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"artifact": out.name, "canvas": {"height": CANVAS, "width": CANVAS}, "subsets": []}
    )
    manifest.setdefault("lane_specs", {})[spec.subset_name] = {
        "group_len_min": spec.group_len_min,
        "group_len_max": spec.group_len_max,
        "group_len_jittered": spec.group_len_min != spec.group_len_max,
        "mean_group_len": spec.mean_group_len,
        "n_groups": spec.n_groups,
        "font_size": spec.font_size,
        "mean_n_random_letters": spec.mean_n_random_letters,
        "rate_min": spec.rate_min,
        "rate_max": spec.rate_max,
        "run_min": spec.run_min,
        "run_max": spec.run_max,
        "mask_pad_x": spec.mask_pad_x,
        "mask_pad_y": spec.mask_pad_y,
        "sentinel": spec.sentinel,
        "target_format": "sentinel_per_run",
    }
    manifest["subsets"] = [
        s for s in manifest["subsets"] if s["config_name"] != spec.subset_name
    ] + [
        {
            "config_name": spec.subset_name,
            "path": f"{spec.subset_name}/train",
            "split": "train",
            "is_training_split": True,
        },
        {
            "config_name": spec.subset_name,
            "path": f"{spec.subset_name}/validation",
            "split": "validation",
            "is_training_split": False,
        },
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # --- measured mask statistics + analytic floors ------------------------
    import pyarrow.compute as pc

    n_val = max(len(val_ds), 1)
    rates = val_ds.with_format("arrow")["mask_rate"].to_pylist()
    run_lengths = [
        n for row in val_ds.with_format("arrow")["mask_run_lengths"].to_pylist() for n in row
    ]
    n_runs = pc.list_value_length(val_ds.data.column("mask_run_starts")).to_pylist()
    n_visible = pc.list_value_length(val_ds.data.column("visible_words")).to_pylist()
    mean_rate = sum(rates) / n_val
    var_rate = sum((r - mean_rate) ** 2 for r in rates) / n_val

    val_tok = val_stats.target_tokens
    h_char = math.log(len(CHARSET))  # nats/letter, uniform i.i.d.
    h_len = spec.group_len_entropy_nats  # nats/word for the LENGTH; 0 if fixed
    # Letters are summed from the ACTUAL strings.  The fixed-length version
    # derived this as `len(visible_words) * group_len`, which is only correct
    # when the length cannot vary — with jitter it must be measured.
    visible_letters_per_tok = val_stats.visible_letters / max(val_tok, 1)
    visible_words_per_tok = val_stats.visible_words / max(val_tok, 1)
    floor_per_token = h_char * visible_letters_per_tok
    floor_with_length = floor_per_token + h_len * visible_words_per_tok

    # A masked word is L letters with L drawn uniformly, so the chance of a
    # non-reader emitting it exactly is E_L[26^-L] (it must also guess L; this
    # ignores that, so it is an UPPER bound on a guesser's success rate).
    guess_accuracies = {
        L: len(CHARSET) ** (-L)
        for L in range(spec.group_len_min, spec.group_len_max + 1)
    }
    guess_accuracy = sum(guess_accuracies.values()) / len(guess_accuracies)

    floor = {
        "lane": spec.subset_name,
        "tokenizer": args.tokenizer,
        "charset": CHARSET,
        "charset_size": len(CHARSET),
        "group_len_min": spec.group_len_min,
        "group_len_max": spec.group_len_max,
        "group_len_jittered": spec.group_len_min != spec.group_len_max,
        "mean_group_len": spec.mean_group_len,
        "sentinel": spec.sentinel,
        "target_format": "sentinel_per_run",
        "H_char_nats": h_char,
        "H_word_length_nats": h_len,
        "masked_word_guess_accuracy": guess_accuracy,
        "masked_word_guess_accuracy_worst_case": guess_accuracies[spec.group_len_min],
        "val_word_length_histogram": {
            str(k): v for k, v in sorted(val_stats.word_length_histogram.items())
        },
        "val_mean_word_length": (
            (val_stats.visible_letters + val_stats.masked_letters)
            / max(val_stats.visible_words + val_stats.masked_words, 1)
        ),
        "val_mean_mask_rate": mean_rate,
        "val_std_mask_rate": math.sqrt(var_rate),
        "val_min_mask_rate": min(rates),
        "val_max_mask_rate": max(rates),
        "val_mean_runs_per_row": sum(n_runs) / n_val,
        "val_mean_run_length": (sum(run_lengths) / len(run_lengths)) if run_lengths else 0.0,
        "val_run_length_histogram": {
            str(k): run_lengths.count(k) for k in range(spec.run_min, spec.run_max + 1)
        },
        "val_mean_visible_words": sum(n_visible) / n_val,
        "val_mean_target_tokens": val_tok / n_val,
        "val_visible_letters_per_token": visible_letters_per_tok,
        "val_visible_words_per_token": visible_words_per_tok,
        "no_reading_floor_nats_per_token": floor_per_token,
        "no_reading_floor_with_length_nats_per_token": floor_with_length,
        "note": (
            "FOUR numbers, do not confuse them.\n"
            f"(1) `no_reading_floor_nats_per_token` = {floor_per_token:.4f} is a strict "
            "LOWER BOUND on a non-reader's CE: it amortises ONLY the entropy of the "
            "VISIBLE letters (uniform i.i.d. over 26) over all target tokens. The mask "
            "pattern itself — which of the words are replaced by a sentinel, and how "
            "the runs are grouped — carries ADDITIONAL entropy that a non-reader also "
            "cannot predict, so a true non-reader sits ABOVE this line. Crossing it is "
            "sufficient evidence of reading; staying above it is not evidence of "
            "non-reading (same caveat as the poisoned-text lane).\n"
            f"(1b) `no_reading_floor_with_length_nats_per_token` = {floor_with_length:.4f} "
            "is a TIGHTER (still strict) lower bound that additionally credits the "
            f"entropy of each visible word's LENGTH, H(L) = ln({spec.n_group_lengths}) = "
            f"{h_len:.4f} nats/word — a non-reader must also predict where each word "
            "ends. It is 0 and the two floors coincide when the length is fixed "
            "(`group_len_min == group_len_max`). NOTE the letters term is measured from "
            "the ACTUAL strings, not `n_visible * group_len`: with a jittered length the "
            "latter derivation is simply wrong.\n"
            "(2) `masked_word_guess_accuracy` = "
            f"{guess_accuracy:.3g} is the per-masked-word floor for the "
            "hallucinated-content metric: an occluded word is L letters drawn i.i.d. "
            f"uniform over {len(CHARSET)} with L ~ U[{spec.group_len_min}, "
            f"{spec.group_len_max}], so the chance of emitting it exactly is "
            f"E_L[26^-L] = {guess_accuracy:.3g} (loosest single case, the shortest word: "
            f"{guess_accuracies[spec.group_len_min]:.3g}). Any masked word the model "
            "emits correctly at a rate above this came from pixels it cannot see — i.e. "
            "it did not come from anywhere, and the metric is measuring fabrication, "
            "not knowledge.\n"
            "(3) The DEGENERATE-SHORTCUT guard is not a floor but a split: score "
            "`visible_words` accuracy (reading) SEPARATELY from sentinel placement at "
            "`sentinel_char_starts` / `sentinel_visible_index` (box detection). High "
            "span-boundary accuracy with floor-level visible-text accuracy means the "
            "model learned to see black rectangles, not to read, and does NOT support "
            "H14. The mask-permutation control (`mask_b_*`, rendered with "
            "`render_masked_pages`) is the sharpest test: same image content, different "
            "occlusion must change the output.\n"
            "(4) WORD-LENGTH JITTER closes the run-length leak. Same-line words of a run "
            "are merged into one solid rectangle, so at a FIXED length the rectangle's "
            "WIDTH is an exact function of how many words it hid (5/11/17 monospace "
            "characters for 1/2/3 words) — i.e. the image silently supplied target "
            f"variant (c) `<mask:k>`. Lengths are drawn U[{spec.group_len_min}, "
            f"{spec.group_len_max}] here so the per-k width distributions OVERLAP. The "
            "leak is reduced, NOT eliminated: a very wide box is still more likely to be "
            "3 words. Measure it (mutual information between rect width and words-in-rect) "
            "rather than assuming, and read span-boundary accuracy against the "
            "width-only MAP baseline, not against chance."
        ),
    }
    (out / spec.subset_name / "floor.json").write_text(json.dumps(floor, indent=2))
    if spec.subset_name == "masked-randstr":
        (out / "floor.json").write_text(json.dumps(floor, indent=2))

    logger.info("Wrote %s (train=%d val=%d)", out, len(train_ds), len(val_ds))
    logger.info(
        "MASK RATE measured %.4f ± %.4f (range %.4f–%.4f); %.2f runs/row, mean run %.2f words, "
        "%.1f visible words/row, %.1f target tokens/row",
        mean_rate, floor["val_std_mask_rate"], min(rates), max(rates),
        floor["val_mean_runs_per_row"], floor["val_mean_run_length"],
        floor["val_mean_visible_words"], floor["val_mean_target_tokens"],
    )
    logger.info(
        "WORD LENGTH %d-%d (jittered=%s), measured mean %.3f letters; histogram %s",
        spec.group_len_min, spec.group_len_max,
        spec.group_len_min != spec.group_len_max,
        floor["val_mean_word_length"], floor["val_word_length_histogram"],
    )
    logger.info(
        "NO-READING FLOOR (visible letters only, LOWER BOUND) ≈ %.4f nats/tok "
        "[+ word-length entropy: %.4f nats/tok]; masked-word guess accuracy = %.3g",
        floor_per_token, floor_with_length, floor["masked_word_guess_accuracy"],
    )


if __name__ == "__main__":
    main()
