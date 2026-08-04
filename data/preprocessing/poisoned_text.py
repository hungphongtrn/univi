"""Materialize the H15 "prior-poisoned text" lane.

Rationale (H15): H10 showed the model buys reading only where the language prior
fails, and on real text the prior only fails at position 0 — so reading stays
shallow and concentrated at the start.  Corrupt the *source text before
rendering* and the prior is wrong at roughly every 6th–7th character, at **all**
positions, so reading is the only way to lower loss anywhere in the sequence.

Each row takes a real fineweb-edu document, substitutes ``p`` (default 0.15) of
its **alphabetic** characters with a uniformly random lowercase letter (drawn
independently per character), renders the **poisoned** text to the fineweb-edu
geometry (1024x1024 canvas, font 14, grayscale "L"), and makes the **poisoned**
text the transcription target.  Image and target agree exactly — the task stays
solvable, it just stops being predictable.

**The poison mask is stored per row** (``poison_mask``: character indices into
the target string that were resampled).  That is the point of the design: those
positions are drawn i.i.d. uniform over 26 letters independently of everything
else, so *no* model can exceed 1/26 accuracy there without reading pixels.
Accuracy at masked positions, aligned vs blank, is therefore a **pure-reading**
metric — which fixes the H09 problem that eval CE is orthogonal to grounding.

Target/image agreement is exact by construction: the source is first pushed
through the renderer's own layout pass (``render_utils.wrap_text_pages``), the
first ``max_pages`` pages' worth of wrapped lines are joined with newlines, and
*that* string is what gets poisoned, rendered and supervised.  A final
``render_text_pages`` round-trip asserts the page budget actually holds.

Note the extra columns (``poison_mask``, ``poison_rate``, ``source_row_id``)
make this lane's Features **incompatible with ``concatenate_datasets``** against
the other lanes.  H15 is single-lane, so this is fine; a future mixture must
drop them first.

Usage (CPU-only, no GPU needed — rasterising real fineweb pages costs ~230 ms
of CPU per row, so use --num-proc):
  HF_HUB_OFFLINE=1 uv run python -m data.preprocessing.poisoned_text \
      --out data/materialized/h15-poisoned-v0 \
      --train-rows 60000 --val-rows 1000 --num-proc 8
  # tiny smoke set:
  HF_HUB_OFFLINE=1 uv run python -m data.preprocessing.poisoned_text \
      --out /tmp/h15-smoke --train-rows 200 --val-rows 32

``--max-target-tokens`` additionally trims each target (at a rendered-line
boundary, re-rendering so the image still matches) so rows survive the
trainer's ``max_length`` filter — see the "expanded token budget" warning the
script prints at the end.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

from datasets import Features, Image, Sequence, load_from_disk
from datasets import Value as V

from data.preprocessing.render_utils import render_text_pages, wrap_text_pages

logger = logging.getLogger(__name__)

CHARSET = "abcdefghijklmnopqrstuvwxyz"  # V=26 — the poison alphabet
PROMPT = "Transcribe the text shown in the image."
CANVAS = 1024      # match fineweb-edu render geometry
FONT_SIZE = 14
SOURCE_ID = "HuggingFaceFW/fineweb-edu"
PREP_VERSION = "0.1.0-h15"
DEFAULT_SOURCE = "data/materialized/univi-3M-v0-split/fineweb-edu"

# Only the first few pages are ever kept, so wrapping the whole (sometimes
# 100k-char) document is wasted work.  At font 14 the monospace face fits ~117
# chars/line and 57 lines/page, i.e. ~6.7k chars/page; 60k chars covers 8+ pages.
SOURCE_CHAR_BUDGET = 60_000

# Trainer-side constants (univi.trainer._IMAGE_TOKEN_BUDGET /
# _TEMPLATE_TOKEN_OVERHEAD) — replicated here only to *report* how many rows
# would survive `training.max_length`, never to filter.
_IMAGE_TOKEN_BUDGET = 282
_TEMPLATE_TOKEN_OVERHEAD = 256


@dataclass(frozen=True)
class LaneSpec:
    """One configuration of the poisoned-text lane.

    ``poison_rate`` is the per-alphabetic-character substitution probability —
    the single knob H15 turns.  ``max_pages`` caps images/row (the trainer's
    ``dataset.max_train_images``); ``max_target_tokens`` optionally caps the
    supervised target so rows survive ``training.max_length``.
    """

    poison_rate: float = 0.15
    max_pages: int = 4
    max_target_tokens: int | None = None
    font_size: int = FONT_SIZE
    subset_name: str = "poisoned-text"


# messages/images schema identical to the other lanes (univi/trainer +
# univi.hybrid.data.HybridCollator); poison_* / source_row_id are additive.
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
        # --- H15 additions --------------------------------------------------
        "poison_mask": Sequence(V("int32")),
        "poison_rate": V("float32"),
        "source_row_id": V("string"),
    }
)


def _visible_lines(text: str, spec: LaneSpec) -> list[str]:
    """Return the wrapped lines that fit on the first ``max_pages`` pages.

    Uses the renderer's own layout pass, so the returned lines are exactly the
    lines the rasteriser would draw.  Trailing blank lines are dropped: they
    carry no glyphs, so supervising the newlines that produce them would ask
    the model to predict something it cannot see.
    """
    truncated = text[:SOURCE_CHAR_BUDGET]
    lines, lines_per_page, _ = wrap_text_pages(
        truncated,
        canvas_width=CANVAS,
        canvas_height=CANVAS,
        font_size=spec.font_size,
    )
    needed = lines_per_page * spec.max_pages
    if len(lines) < needed and len(truncated) < len(text):
        # Pathological layout (e.g. a document that is mostly newlines) where
        # SOURCE_CHAR_BUDGET cut a document that had not yet filled the pages.
        lines, lines_per_page, _ = wrap_text_pages(
            text,
            canvas_width=CANVAS,
            canvas_height=CANVAS,
            font_size=spec.font_size,
        )
        needed = lines_per_page * spec.max_pages
    visible = lines[:needed]
    while visible and not visible[-1]:
        visible.pop()
    return visible


def _poison(text: str, rate: float, rng: random.Random) -> tuple[str, list[int]]:
    """Resample ``rate`` of the alphabetic characters, uniformly over a-z.

    The replacement is drawn from the full 26-letter alphabet *independently of
    the original character*.  That costs ~1/26 of masked positions landing back
    on the same letter, and it is deliberate: it makes the masked character
    exactly uniform, so the ln(26) / (1/26) floors below are exact.  Rejection
    sampling to force a change would make the masked character depend on the
    original, handing a prior-driven model a sliver of free information.
    """
    chars = list(text)
    mask: list[int] = []
    for i, ch in enumerate(chars):
        if ch.isalpha() and rng.random() < rate:
            chars[i] = rng.choice(CHARSET)
            mask.append(i)
    return "".join(chars), mask


def _cut_lines(lines: list[str], mask: list[int], n_lines: int) -> tuple[str, list[int]]:
    """Keep the first ``n_lines`` rendered lines; drop mask entries past the cut."""
    kept = lines[:n_lines]
    while kept and not kept[-1]:
        kept.pop()
    text = "\n".join(kept)
    return text, [i for i in mask if i < len(text)]


def _fit_token_budget(
    lines: list[str], mask: list[int], tokenizer, budget: int
) -> tuple[str, list[int], int]:
    """Trim whole rendered lines until the target fits ``budget`` tokens."""
    text = "\n".join(lines)
    n_tok = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    if n_tok <= budget:
        return text, mask, n_tok
    # One-shot estimate from the token→char offsets, then a short linear
    # shrink to absorb re-tokenisation at the new boundary.
    if getattr(tokenizer, "is_fast", False):
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        char_limit = encoded["offset_mapping"][budget - 1][1]
    else:  # slow tokenizer: proportional guess, the shrink loop finishes the job
        char_limit = int(len(text) * budget / n_tok)
    n_lines, consumed = 0, 0
    for line in lines:
        nxt = consumed + len(line) + (1 if n_lines else 0)
        if nxt > char_limit:
            break
        consumed, n_lines = nxt, n_lines + 1
    n_lines = max(n_lines, 1)
    while n_lines > 0:
        text, cut_mask = _cut_lines(lines, mask, n_lines)
        n_tok = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n_tok <= budget:
            return text, cut_mask, n_tok
        n_lines -= 1
    return "", [], 0


def _render_config_json(spec: LaneSpec) -> str:
    return json.dumps(
        {
            "canvas_height": CANVAS,
            "canvas_width": CANVAS,
            "font_size": spec.font_size,
            "image_mode": "L",
            "png_compression_level": 1,
            "render_method": "text_page",
            "source_dataset_id": SOURCE_ID,
            "poison_charset_size": len(CHARSET),
            "poison_rate": spec.poison_rate,
            "poison_unit": "alphabetic_character",
            "max_pages": spec.max_pages,
            "max_target_tokens": spec.max_target_tokens,
        },
        sort_keys=True,
    )


def _make_row(raw: str, src_id: str, index: int, split: str, seed: int, tokenizer, spec: LaneSpec):
    """Poison, render and package one source document.

    The RNG is seeded from ``(seed, subset, split, index)`` rather than drawn
    from one sequential stream, so the poison draw for a given row is identical
    regardless of ``--num-proc`` (and of how many rows precede it).
    """
    rng = random.Random(f"{seed}|{spec.subset_name}|{split}|{index}")
    lines = _visible_lines(raw, spec)
    if not lines:
        return None
    poisoned, mask = _poison("\n".join(lines), spec.poison_rate, rng)
    plines = poisoned.split("\n")
    if spec.max_target_tokens is not None:
        target, mask, otl = _fit_token_budget(
            plines, mask, tokenizer, spec.max_target_tokens
        )
        if not target:
            return None
        plines = target.split("\n")
    else:
        target = poisoned
        otl = len(tokenizer(target, add_special_tokens=False)["input_ids"])

    pages = render_text_pages(
        target, canvas_width=CANVAS, canvas_height=CANVAS, font_size=spec.font_size
    )
    # Guard: the target is built from the *clean* layout, so a width change
    # under poisoning (only possible with a non-monospace fallback font) could
    # push glyphs off the last page.  Shrink until the budget holds — a target
    # containing text the model cannot see would silently make the task
    # impossible rather than merely prior-proof.
    while len(pages) > spec.max_pages and len(plines) > 1:
        target, mask = _cut_lines(plines, mask, len(plines) - 1)
        plines = target.split("\n")
        otl = len(tokenizer(target, add_special_tokens=False)["input_ids"])
        pages = render_text_pages(
            target, canvas_width=CANVAS, canvas_height=CANVAS, font_size=spec.font_size,
        )
    if len(pages) > spec.max_pages:
        raise ValueError(
            f"row {src_id}: {len(pages)} pages > max_pages={spec.max_pages} "
            "even after shrinking to one line."
        )

    image_bytes = []
    for page in pages:
        buf = io.BytesIO()
        page.convert("L").save(buf, format="PNG", compress_level=1)
        image_bytes.append(buf.getvalue())

    return {
        "images": image_bytes,
        "messages": [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "text": None} for _ in image_bytes],
                    {"type": "text", "text": PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target}],
            },
        ],
        "source_dataset_id": SOURCE_ID,
        "split": split,
        "row_id": f"{spec.subset_name}-{split}-{index:07d}",
        "render_config": _render_config_json(spec),
        "modality_label": "text",
        "preprocessing_version": PREP_VERSION,
        "original_token_length": otl,
        "source_split": split,
        "poison_mask": mask,
        "poison_rate": spec.poison_rate,
        "source_row_id": str(src_id),
    }


# Placeholder emitted for a source document that renders to nothing; `.map`
# cannot drop rows, so these are flagged and filtered out afterwards.
_EMPTY_ROW = {
    "images": [],
    "messages": [{"role": "assistant", "content": [{"type": "text", "text": ""}]}],
    "source_dataset_id": "", "split": "", "row_id": "", "render_config": "",
    "modality_label": "", "preprocessing_version": "", "original_token_length": 0,
    "source_split": "", "poison_mask": [], "poison_rate": 0.0, "source_row_id": "",
}


def _build_split(
    split: str, n_rows: int, seed: int, tokenizer, spec: LaneSpec,
    source_dir: Path, num_proc: int = 1,
):
    """Build one split from the materialized fineweb-edu lane.

    That lane's assistant turn holds the full transcription target, i.e.
    exactly the text that was rendered — that is our clean source.  Only the
    two needed columns are selected, so the (huge) image payloads are never
    read.
    """
    src = load_from_disk(str(source_dir / split))
    if n_rows > len(src):
        raise ValueError(
            f"Requested {n_rows} rows from {source_dir / split} which has {len(src)}."
        )
    src = src.select(range(n_rows)).select_columns(["messages", "row_id"])

    def _process(row, index: int):
        built = _make_row(
            row["messages"][1]["content"][0]["text"], row["row_id"],
            index, split, seed, tokenizer, spec,
        )
        if built is None:
            return {**_EMPTY_ROW, "_keep": False}
        return {**built, "_keep": True}

    ds = src.map(
        _process,
        with_indices=True,
        remove_columns=src.column_names,
        num_proc=num_proc if num_proc > 1 else None,
        desc=f"Poisoning + rendering {split}",
    )
    n_built = len(ds)
    ds = ds.filter(
        bool, input_columns=["_keep"], num_proc=num_proc if num_proc > 1 else None,
        desc=f"Dropping empty {split} rows",
    ).remove_columns("_keep")
    skipped = n_built - len(ds)
    if skipped:
        logger.warning("%s: dropped %d source rows that rendered to nothing", split, skipped)
    return ds.cast(FEATURES), skipped


def _split_stats(ds, full: bool) -> dict:
    """Summarise a built split.  ``full`` also scans target text (validation only)."""
    import pyarrow.compute as pc

    tokens = ds.with_format("arrow")["original_token_length"].to_pylist()
    pages = pc.list_value_length(ds.data.column("images")).to_pylist()
    n_poisoned = pc.list_value_length(ds.data.column("poison_mask")).to_pylist()
    stats = {
        "rows": len(ds),
        "tokens": sum(tokens),
        "pages": sum(pages),
        "poisoned": sum(n_poisoned),
        "expanded_tokens": [
            t + p * _IMAGE_TOKEN_BUDGET + _TEMPLATE_TOKEN_OVERHEAD
            for t, p in zip(tokens, pages)
        ],
        "chars": 0,
        "alpha": 0,
    }
    if full:
        for row in ds.select_columns(["messages"]):
            target = row["messages"][1]["content"][0]["text"]
            stats["chars"] += len(target)
            stats["alpha"] += sum(c.isalpha() for c in target)
    return stats


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--train-rows", type=int, default=60_000)
    p.add_argument("--val-rows", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument(
        "--tokenizer",
        default="data/checkpoints/encoder-free-v0/best",
        help="Local dir with tokenizer.json (for real target token counts + floor).",
    )
    p.add_argument(
        "--poison-rate",
        type=float,
        default=0.15,
        help="P(substitute) per alphabetic character, uniform over a-z.",
    )
    p.add_argument(
        "--subset-name",
        default="poisoned-text",
        help="Subset dir + config_name. Must be listed in univi.trainer.VALID_SUBSETS.",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=4,
        help="Images per row (keep <= dataset.max_train_images).",
    )
    p.add_argument(
        "--max-target-tokens",
        type=int,
        default=None,
        help=(
            "Optional cap on supervised target tokens (trimmed at a rendered-line "
            "boundary, image re-rendered to match). Needed to survive the trainer's "
            "training.max_length filter; see the budget report printed at the end."
        ),
    )
    p.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Materialized fineweb-edu lane dir containing train/ and validation/.",
    )
    p.add_argument("--font-size", type=int, default=FONT_SIZE)
    p.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help=(
            "Worker processes for poisoning/rendering (~230 ms/row single-core; "
            "rasterising real fineweb pages dominates). Does NOT change the output: "
            "each row's poison draw is seeded from its own index."
        ),
    )
    args = p.parse_args(argv)

    if not 0.0 <= args.poison_rate <= 1.0:
        raise ValueError("--poison-rate must be in [0, 1]")

    spec = LaneSpec(
        poison_rate=args.poison_rate,
        max_pages=args.max_pages,
        max_target_tokens=args.max_target_tokens,
        font_size=args.font_size,
        subset_name=args.subset_name,
    )

    # The tokenizer is pickled into `.map` workers; Rust-side threading on top
    # of process forking is both slower and noisy.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    source_dir = Path(args.source)

    out = Path(args.out)
    (out / spec.subset_name).mkdir(parents=True, exist_ok=True)

    logger.info(
        "Lane %s: poison_rate=%.3f, max_pages=%d, max_target_tokens=%s, font %d, source %s",
        spec.subset_name, spec.poison_rate, spec.max_pages,
        spec.max_target_tokens, spec.font_size, source_dir,
    )
    logger.info("Building train split (%d rows)…", args.train_rows)
    train_ds, _ = _build_split(
        "train", args.train_rows, args.seed, tokenizer, spec, source_dir, args.num_proc
    )
    logger.info("Building validation split (%d rows)…", args.val_rows)
    # disjoint seed stream (and a disjoint source split) for validation
    val_ds, _ = _build_split(
        "validation", args.val_rows, args.seed + 10**6, tokenizer, spec,
        source_dir, args.num_proc,
    )
    train_stats = _split_stats(train_ds, full=False)
    val_stats = _split_stats(val_ds, full=True)

    train_ds.save_to_disk(str(out / spec.subset_name / "train"))
    val_ds.save_to_disk(str(out / spec.subset_name / "validation"))

    # Merge into an existing manifest rather than clobbering it, so several
    # poison rates can share ONE root (train on the mixture, eval each rate).
    manifest_path = out / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"artifact": out.name, "canvas": {"height": CANVAS, "width": CANVAS}, "subsets": []}
    )
    manifest.setdefault("lane_specs", {})[spec.subset_name] = {
        "poison_rate": spec.poison_rate,
        "max_pages": spec.max_pages,
        "max_target_tokens": spec.max_target_tokens,
        "font_size": spec.font_size,
        "source": str(source_dir),
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

    # --- analytic floor ---------------------------------------------------
    n_val = max(len(val_ds), 1)
    h_char = math.log(len(CHARSET))  # nats per poisoned character, uniform i.i.d.
    poisoned_per_row = val_stats["poisoned"] / n_val
    tokens_per_row = val_stats["tokens"] / n_val
    poisoned_per_token = val_stats["poisoned"] / max(val_stats["tokens"], 1)
    floor_per_token = h_char * poisoned_per_token
    floor = {
        "lane": spec.subset_name,
        "source_dataset_id": SOURCE_ID,
        "tokenizer": args.tokenizer,
        "poison_charset": CHARSET,
        "poison_charset_size": len(CHARSET),
        "poison_rate": spec.poison_rate,
        "H_poisoned_char_nats": h_char,
        "poisoned_position_accuracy_floor": 1.0 / len(CHARSET),
        "poisoned_position_nats_floor": h_char,
        "val_measured_poison_rate_of_alpha": val_stats["poisoned"] / max(val_stats["alpha"], 1),
        "val_avg_poisoned_chars_per_row": poisoned_per_row,
        "val_avg_target_chars_per_row": val_stats["chars"] / n_val,
        "val_avg_target_tokens_per_row": tokens_per_row,
        "val_avg_pages_per_row": val_stats["pages"] / n_val,
        "val_poisoned_chars_per_token": poisoned_per_token,
        "no_reading_floor_nats_per_token": floor_per_token,
        "note": (
            "TWO different numbers, do not confuse them.\n"
            "(1) TIGHT, per-poisoned-position: each character at an index listed in "
            f"`poison_mask` was drawn i.i.d. uniform over {len(CHARSET)} letters, "
            "independently of the surrounding text and of the character it replaced. "
            f"So WITHOUT reading the image no model can beat {h_char:.4f} nats "
            f"(= ln {len(CHARSET)}; a Jensen bound that holds for ANY predictive "
            f"distribution) or {1.0/len(CHARSET):.4f} argmax accuracy at those "
            "positions. This is the pure-reading metric H15 is built around: measure "
            "masked-position accuracy aligned vs blank/permuted.\n"
            "(2) LOOSE, per-token whole-sequence: "
            f"`no_reading_floor_nats_per_token` = {floor_per_token:.4f} amortises ONLY "
            "the poisoned characters' entropy over ALL target tokens. It is a LOWER "
            "BOUND on a non-reader's CE, not an estimate of it: the other "
            f"{100*(1-spec.poison_rate):.0f}% of alphabetic characters plus every "
            "digit, punctuation mark and whitespace still carry English structure "
            "with unknown positive entropy, so a real non-reader sits WELL ABOVE this "
            "line. Unlike the `random-strings` lane -- where the floor is tight "
            "because the whole target is random -- crossing this line here is "
            "sufficient evidence of reading, but staying above it is NOT evidence of "
            "non-reading. Do not use it as a pass/fail gate; use metric (1).\n"
            "(3) Caveats: `poison_mask` is indexed in CHARACTERS of the assistant "
            "target string, so a token-level probe must map tokens to character spans "
            "(e.g. `return_offsets_mapping=True`) before scoring; a token straddling a "
            "masked and an unmasked character is not purely prior-proof. The "
            "per-token number is tokenizer-specific (measured at materialization "
            f"time with `{args.tokenizer}`) because poisoning fragments BPE merges "
            "and raises tokens/char."
        ),
    }
    (out / spec.subset_name / "floor.json").write_text(json.dumps(floor, indent=2))
    if spec.subset_name == "poisoned-text":
        (out / "floor.json").write_text(json.dumps(floor, indent=2))

    logger.info("Wrote %s (train=%d val=%d)", out, len(train_ds), len(val_ds))
    logger.info(
        "Measured poison rate = %.4f of alphabetic chars (target %.4f); "
        "%.1f poisoned chars/row, %.1f target tokens/row, %.2f pages/row",
        floor["val_measured_poison_rate_of_alpha"], spec.poison_rate,
        poisoned_per_row, tokens_per_row, floor["val_avg_pages_per_row"],
    )
    logger.info(
        "POISONED-POSITION FLOOR (tight) = %.4f nats / %.2f%% accuracy. "
        "Per-token lower bound (LOOSE) = %.4f nats/tok.",
        h_char, 100.0 / len(CHARSET), floor_per_token,
    )

    # Rows the trainer would silently drop (univi.trainer._filter_training_tokens).
    expanded = sorted(train_stats["expanded_tokens"])
    if expanded:
        median = expanded[len(expanded) // 2]
        for max_length in (2048, 4096, 8192, 16384):
            survive = sum(e <= max_length for e in expanded) / len(expanded)
            level = logging.WARNING if survive < 0.9 else logging.INFO
            logger.log(
                level,
                "training.max_length=%d would retain %.1f%% of train rows "
                "(estimated expanded tokens: median %d, max %d)",
                max_length, 100 * survive, median, expanded[-1],
            )
        logger.info(
            "If retention is low, re-run with --max-target-tokens "
            "(<= max_length - max_pages*%d - %d) or a smaller --max-pages.",
            _IMAGE_TOKEN_BUDGET, _TEMPLATE_TOKEN_OVERHEAD,
        )


if __name__ == "__main__":
    main()
