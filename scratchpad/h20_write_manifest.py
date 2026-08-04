"""Write the merged manifest for the H20 wide-render root.

``univi.trainer._load_local`` takes ONE ``dataset.path`` and reads
``manifest.json`` from it: entries with ``is_training_split: true`` are
concatenated into the train mixture, and ``config_name`` must be in
``univi.trainer.VALID_SUBSETS``.  Row counts are read from the datasets on disk,
never hard-coded, so a short render cannot silently pass as a full one.

CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# (config_name, path, split, is_training_split)
# EXACTLY ONE spoken-digits `train` entry may carry is_training_split=True --
# _load_local concatenates every True entry whose config_name matches, so two
# would double-count the lane. The full 30k split is materialized and listed
# with False so switching the anchor share is a one-line edit, not a re-render.
# An entry with split "train" and is_training_split False is inert: the eval
# loader only ever looks up split == "validation".
ENTRIES = [
    ("librispeech", "librispeech/train", "train", True),
    ("librispeech", "librispeech/validation", "validation", False),
    ("spoken-digits", "spoken-digits-15pct/train", "train", True),
    ("spoken-digits", "spoken-digits/train", "train", False),
    ("spoken-digits", "spoken-digits/validation", "validation", False),
]

NOTE = (
    "H20 WIDE-RENDER mixture root: both audio lanes re-materialized at 2000x160 "
    "(from 1000x160) by data/preprocessing/widen_audio_pages.py, PIL LANCZOS, in "
    "image space. At the production 10 ms hop the page is already 1 render pixel "
    "per mel frame, so the extra width is interpolation either way: widening the "
    "stored page and re-rendering the mel at output_width 2000 differ by 0.727% of "
    "the page's own sd (widen_audio_pages --self-test, 7 real LibriSpeech pages; "
    "H20 reports 0.72% by an independent path). WHY WIDE: the image processor "
    "targets constant AREA, so grid columns = sqrt(budget * W/H) and the token "
    "count is ~budget whatever the aspect. 2000x160 @1120 = 1062 soft tokens at "
    "84.7 ms/column -- FEWER tokens than the 1079 the 1000x160 page costs at the "
    "same budget, and 1.42x finer in time. spoken-digits-15pct/train is a "
    "deterministic 18,355-row subsample (shuffle seed 42) taken BEFORE widening; "
    "the trainer's max_train_rows_per_subset is a single GLOBAL cap and cannot "
    "express a per-lane share. 18355/(18355+104014) = 15.00% OF ROWS -- but only "
    "4.59% of supervised TOKENS, because a digit target is 15 tokens against "
    "librispeech's 55.0 (H13's lesson: balancing rows does not balance gradient). "
    "The full 30,000-row wide split is also materialized at spoken-digits/train "
    "with is_training_split=false; swapping the two manifest flags raises the "
    "anchor to 22.4% of rows / 7.29% of tokens, which is the MAXIMUM this lane "
    "can supply -- a 15%-of-tokens anchor would need ~67,300 digit rows."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    from datasets import load_from_disk

    root = Path(args.root)
    subsets = []
    for config_name, rel, split, is_train in ENTRIES:
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(f"missing split: {path}")
        ds = load_from_disk(str(path))
        subsets.append(
            {
                "config_name": config_name,
                "path": rel,
                "split": split,
                "is_training_split": is_train,
                "rows": len(ds),
            }
        )
        print(f"  {rel:34s} rows={len(ds)}")

    manifest = {
        "artifact": "h20-audio-wide-v0",
        "note": NOTE,
        "render": {"output_width": 2000, "output_height": 160,
                   "hop_length": 160, "page_duration_sec": 10.0, "n_mels": 80},
        "sources": {
            "librispeech": "data/materialized/univi-3M-v0-split/librispeech",
            "spoken-digits": "data/materialized/spoken-digits-v0/spoken-digits",
        },
        "subsets": subsets,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
