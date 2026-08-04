"""Materialize the H3 "prior-proof" random-string OCR lane.

Rationale (H3): every other lane lets the pretrained Qwen3 LM prior lower loss
via a language shortcut, so a content-blind steering prefix (see H2) or a shallow
reader both look similar. Random character strings have ZERO language structure —
the ONLY way to lower loss below the guessing floor is to READ the pixels. This
isolates the embedder's reading capacity from mixture/prior rewards.

Each row renders a random lowercase-letter string (grouped like words) to the SAME
geometry the fineweb-edu lane uses (1024x1024 canvas, font 14, grayscale "L"), so a
model warm-started from an existing checkpoint sees an in-distribution image. The
target transcript is the string itself; it appears NOWHERE in the prompt.

Also computes the analytic no-reading floor: for uniform i.i.d. letters over a
charset of size V, the target's entropy is (n_random_letters * ln V) nats, and a
non-reading model cannot beat floor_per_token = n_letters*ln(V) / n_target_tokens.
A trained reader should drive eval loss well BELOW this line (toward ~0).

Usage (CPU-only, no GPU needed):
  uv run python -m data.preprocessing.random_strings \
      --out data/materialized/h3-randstr-v0 --train-rows 100000 --val-rows 2000
  # tiny smoke set:
  uv run python -m data.preprocessing.random_strings \
      --out data/materialized/h3-randstr-v0-smoke --train-rows 500 --val-rows 64

H13 density ladder: --group-len/--n-groups/--font-size vary chars-per-page and
target length independently, and --subset-name writes several rungs into ONE root
(the manifest merges) so a single config trains on the mixture and evals each rung
separately. Rungs must be listed in ``univi.trainer.VALID_SUBSETS``:
  uv run python -m data.preprocessing.random_strings \
      --out data/materialized/h13-density-v0 --subset-name randstr-d3 \
      --n-groups 80 --train-rows 20000 --val-rows 500

H17 render resolution: --canvas raises the page above the 1024 default (scale
--font-size with it to hold characters-per-page roughly constant). This exists
because ``Gemma4UnifiedImageProcessor`` resizes a SQUARE page to a side that
depends ONLY on ``max_soft_tokens`` — 768 at 280, 1104 at 560, 1584 at 1120 — so a
1024 render is DOWNSAMPLED 0.75x at 280 and UPSAMPLED 1.55x at 1120: past 560 the
extra soft tokens subdivide interpolated pixels. A canvas of >=1584 is what makes
budget 1120 supply real optical detail. Measured in
``scratchpad/h17_render_resolution.py`` -> ``data/eval/h17-render-resolution.json``.
DEFAULT IS UNCHANGED (1024) and byte-identical to the pre-``--canvas`` renderer:
  uv run python -m data.preprocessing.random_strings \
      --out data/materialized/h17-canvas-v0 --subset-name randstr-d1c \
      --canvas 1584 --font-size 22
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

from datasets import Dataset, Features, Image, Sequence, Value
from datasets import Value as V

from data.preprocessing.render_utils import render_text_pages

logger = logging.getLogger(__name__)

CHARSET = "abcdefghijklmnopqrstuvwxyz"  # V=26, monospace-legible, no glyph ambiguity
GROUP_LEN = 5      # letters per group
N_GROUPS = 5       # groups per string  → 25 random letters, 29 chars with spaces
PROMPT = "Transcribe the text shown in the image."
CANVAS = 1024      # match fineweb-edu render geometry
FONT_SIZE = 14
SOURCE_ID = "synthetic/random-strings"
PREP_VERSION = "0.3.0-h3"


@dataclass(frozen=True)
class LaneSpec:
    """One rung of the density ladder (H13).

    ``group_len``/``n_groups`` set characters-per-page (and hence target length);
    ``font_size`` sets how many text lines fall inside one 48px vision patch. H13
    varies them independently to separate (A) bandwidth from (C) scan.

    ``canvas`` (H17) sets the render resolution. It does NOT change how many soft
    tokens the page costs — the processor resizes a square page to a
    budget-determined side regardless — it changes only whether that resize is a
    down- or an up-sample, i.e. how much real optical detail reaches the model.
    """

    group_len: int = GROUP_LEN
    n_groups: int = N_GROUPS
    font_size: int = FONT_SIZE
    subset_name: str = "random-strings"
    canvas: int = CANVAS

    @property
    def n_random_letters(self) -> int:
        return self.group_len * self.n_groups

# messages schema identical to the other lanes (univi/trainer + PackedIterableDataset)
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
    }
)


def _make_target(rng: random.Random, spec: LaneSpec) -> str:
    groups = [
        "".join(rng.choice(CHARSET) for _ in range(spec.group_len))
        for _ in range(spec.n_groups)
    ]
    return " ".join(groups)


def _render_config_json(spec: LaneSpec) -> str:
    return json.dumps(
        {
            "canvas_height": spec.canvas,
            "canvas_width": spec.canvas,
            "font_size": spec.font_size,
            "image_mode": "L",
            "render_method": "text_page",
            "charset_size": len(CHARSET),
            "n_random_letters": spec.n_random_letters,
        },
        sort_keys=True,
    )


def _build_split(split: str, n_rows: int, seed: int, tokenizer, spec: LaneSpec):
    rng = random.Random(seed)
    rcfg = _render_config_json(spec)
    rows = {k: [] for k in FEATURES}
    tok_total = 0
    for i in range(n_rows):
        target = _make_target(rng, spec)
        pages = render_text_pages(
            target, canvas_width=spec.canvas, canvas_height=spec.canvas, font_size=spec.font_size
        )
        if len(pages) != 1:
            # Taking pages[0] would silently truncate the image while keeping the
            # full target — the model would be asked to transcribe text it cannot
            # see, and the rung's "reading gain" would be meaningless. Fail loudly.
            raise ValueError(
                f"{spec.subset_name}: {spec.n_random_letters} letters at font "
                f"{spec.font_size} need {len(pages)} pages on a {spec.canvas}px canvas. "
                "Lower --n-groups/--font-size, or raise --canvas."
            )
        page = pages[0].convert("L")  # grayscale, matching fineweb-edu
        buf = io.BytesIO()
        page.save(buf, format="PNG")
        # Store as an encoded-bytes record so the Image feature skips the
        # PIL-object nbytes estimate (which mis-handles in-memory images).
        otl = len(tokenizer(target, add_special_tokens=False)["input_ids"])
        tok_total += otl
        rows["images"].append([{"bytes": buf.getvalue(), "path": None}])
        rows["messages"].append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "text": None},
                        {"type": "text", "text": PROMPT},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": target}],
                },
            ]
        )
        rows["source_dataset_id"].append(f"synthetic/{spec.subset_name}")
        rows["split"].append(split)
        rows["row_id"].append(f"{spec.subset_name}-{split}-{i:07d}")
        rows["render_config"].append(rcfg)
        rows["modality_label"].append("text")
        rows["preprocessing_version"].append(PREP_VERSION)
        rows["original_token_length"].append(otl)
        rows["source_split"].append(split)
        if (i + 1) % 10000 == 0:
            logger.info("  %s: rendered %d/%d", split, i + 1, n_rows)
    ds = Dataset.from_dict(rows, features=FEATURES)
    return ds, tok_total


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--train-rows", type=int, default=100_000)
    p.add_argument("--val-rows", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument(
        "--tokenizer",
        default="data/checkpoints/encoder-free-v0/best",
        help="Local dir with tokenizer.json (for real target token counts + floor).",
    )
    # --- density-ladder knobs (H13) --------------------------------------
    p.add_argument("--group-len", type=int, default=GROUP_LEN, help="Letters per group.")
    p.add_argument("--n-groups", type=int, default=N_GROUPS, help="Groups per string.")
    p.add_argument(
        "--font-size",
        type=int,
        default=FONT_SIZE,
        help="Render font size (clamped to >=14 by render_text_pages); sets lines/patch.",
    )
    p.add_argument(
        "--subset-name",
        default="random-strings",
        help="Subset dir + config_name. Must be listed in univi.trainer.VALID_SUBSETS.",
    )
    p.add_argument(
        "--canvas",
        type=int,
        default=CANVAS,
        help=(
            "Square render canvas in px (H17). Does NOT change soft-token cost; "
            "changes only how much real optical detail survives the processor's "
            "budget-determined resize (768/1104/1584 px at 280/560/1120). Scale "
            "--font-size with it to hold characters-per-page constant."
        ),
    )
    args = p.parse_args(argv)

    spec = LaneSpec(
        group_len=args.group_len,
        n_groups=args.n_groups,
        font_size=args.font_size,
        subset_name=args.subset_name,
        canvas=args.canvas,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    out = Path(args.out)
    (out / spec.subset_name).mkdir(parents=True, exist_ok=True)

    logger.info(
        "Lane %s: %d letters/page (%dx%d groups), font %d, canvas %d",
        spec.subset_name, spec.n_random_letters, spec.n_groups, spec.group_len,
        spec.font_size, spec.canvas,
    )
    logger.info("Building train split (%d rows)…", args.train_rows)
    train_ds, train_tok = _build_split("train", args.train_rows, args.seed, tokenizer, spec)
    logger.info("Building validation split (%d rows)…", args.val_rows)
    # disjoint seed stream for validation
    val_ds, val_tok = _build_split(
        "validation", args.val_rows, args.seed + 10**6, tokenizer, spec
    )

    train_ds.save_to_disk(str(out / spec.subset_name / "train"))
    val_ds.save_to_disk(str(out / spec.subset_name / "validation"))

    # Merge into an existing manifest rather than clobbering it: an H13 ladder
    # materializes several rungs (subsets) into ONE root so a single config can
    # train on the mixture and eval each rung separately.
    manifest_path = out / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {
            "artifact": out.name,
            "canvas": {"height": spec.canvas, "width": spec.canvas},
            "subsets": [],
        }
    )
    # NOTE: the root-level "canvas" is only written when the manifest is created.
    # A root holding lanes at DIFFERENT canvases must be read per-lane, from
    # ``lane_specs[<subset>]["canvas"]`` below — which is why it is recorded there.
    manifest.setdefault("lane_specs", {})[spec.subset_name] = {
        "group_len": spec.group_len,
        "n_groups": spec.n_groups,
        "font_size": spec.font_size,
        "n_random_letters": spec.n_random_letters,
        "canvas": spec.canvas,
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

    # --- analytic no-reading floor ---------------------------------------
    n_letters = spec.n_random_letters
    h_char = math.log(len(CHARSET))  # nats/letter, uniform i.i.d.
    val_chars_per_tok = (n_letters * args.val_rows) / max(val_tok, 1)
    floor_per_token = h_char * val_chars_per_tok  # nats/token a non-reader can't beat
    floor = {
        "charset": CHARSET,
        "charset_size": len(CHARSET),
        "n_random_letters_per_string": n_letters,
        "H_char_nats": h_char,
        "val_avg_tokens_per_string": val_tok / max(args.val_rows, 1),
        "val_random_letters_per_token": val_chars_per_tok,
        "no_reading_floor_nats_per_token": floor_per_token,
        "note": (
            "Eval CE loss (nats/tok) on this lane cannot go below "
            f"~{floor_per_token:.4f} without READING pixels; a trained reader → ~0. "
            "This is the per-token recoding of the string entropy; formatting spaces "
            "carry ~0 entropy and are excluded from the letter count."
        ),
    }
    # Per-subset floor (each ladder rung has its own), plus the legacy
    # single-lane `<out>/floor.json` path existing scripts/configs already read.
    (out / spec.subset_name / "floor.json").write_text(json.dumps(floor, indent=2))
    if spec.subset_name == "random-strings":
        (out / "floor.json").write_text(json.dumps(floor, indent=2))
    logger.info("Wrote %s (train=%d val=%d)", out, len(train_ds), len(val_ds))
    logger.info(
        "NO-READING FLOOR ≈ %.4f nats/tok (H_char=%.4f, %.3f letters/token). "
        "Trained reader should drive eval well below this.",
        floor_per_token, h_char, val_chars_per_tok,
    )


if __name__ == "__main__":
    main()
