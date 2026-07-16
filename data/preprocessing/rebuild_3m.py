from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from data.preprocessing.densefusion import preprocess_densefusion
from data.preprocessing.fineweb_edu import preprocess_fineweb_edu
from data.preprocessing.librispeech_asr import preprocess_librispeech_asr
from data.preprocessing.smoltalk import preprocess_smoltalk
from data.preprocessing.text_token_utils import compute_token_length

_SOURCE_IDS = {
    "librispeech": "openslr/librispeech_asr",
    "densefusion": "HuggingFaceM4/FineVision",
    "fineweb-edu": "HuggingFaceFW/fineweb-edu",
    "smoltalk": "HuggingFaceTB/smoltalk",
}

VALIDATION_SPLITS = {
    "librispeech": ("validation",),
    "densefusion": ("validation",),
    "fineweb-edu": ("validation",),
    "smoltalk": ("validation",),
}

TRAINING_SPLITS = {
    "librispeech": ("train",),
    "densefusion": ("train",),
    "fineweb-edu": ("train",),
    "smoltalk": ("train",),
}


def _row_has_required_fields(row: dict) -> bool:
    required = {"messages", "source_dataset_id", "split", "row_id", "render_config", "modality_label", "preprocessing_version"}
    return required.issubset(row.keys())


def _check_row_images(row: dict) -> None:
    placeholders = sum(
        item.get("type") == "image"
        for message in row["messages"]
        for item in message["content"]
    )
    if placeholders != len(row.get("images", [])):
        raise ValueError(
            f"Row {row.get('row_id', '?')}: {placeholders} image placeholders "
            f"but {len(row.get('images', []))} images"
        )




def _add_token_length(row: dict, tokenizer, text_key: str = "target_text") -> dict:
    if "original_token_length" in row and row["original_token_length"] != -1:
        return row
    source = row.get(text_key, "")
    if not source:
        for message in row.get("messages", []):
            if message.get("role") == "assistant":
                source = "".join(
                    item.get("text", "")
                    for item in message.get("content", [])
                    if item.get("type") == "text"
                )
                break
    if tokenizer is not None and source:
        row["original_token_length"] = compute_token_length(source, tokenizer)
    else:
        row["original_token_length"] = -1
    return row


def _select_source_from_fullv0(
    original: Dataset, source_dataset_id: str, num_proc: int = 4
) -> Dataset:
    return original.filter(
        lambda source: source == source_dataset_id,
        input_columns=["source_dataset_id"],
        num_proc=num_proc,
        desc=f"Selecting {source_dataset_id}",
    )


def _deterministic_split(
    dataset: Dataset, train_ratio: float = 0.9
) -> tuple[Dataset, Dataset]:
    split = dataset.train_test_split(
        test_size=1.0 - train_ratio, seed=42, shuffle=True
    )
    return split["train"], split["test"]


def _set_target_split(dataset: Dataset, split_name: str, num_proc: int) -> Dataset:
    def set_split(row):
        row["source_split"] = row.get("source_split", row.get("split"))
        row["split"] = split_name
        return row

    return dataset.map(set_split, num_proc=num_proc, desc=f"Setting target split {split_name}")


def _save_split(
    dataset: Dataset,
    output_root: Path,
    config_name: str,
    split_name: str,
    expected_source: str,
    force: bool = False,
) -> dict:
    _check_row_images(dataset[0])

    subset_path = output_root / config_name / split_name
    if subset_path.exists():
        if force:
            shutil.rmtree(subset_path)
        else:
            raise FileExistsError(
                f"Refusing to replace existing subset: {subset_path}. "
                f"Pass --force to overwrite."
            )
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = subset_path.parent / f".{split_name}.incomplete"
    if staging_path.exists():
        shutil.rmtree(staging_path)
    dataset.save_to_disk(str(staging_path), max_shard_size="500MB")
    staging_path.rename(subset_path)
    return {
        "config_name": config_name,
        "split": split_name,
        "path": str(Path(config_name) / split_name),
        "rows": len(dataset),
        "source_dataset_id": expected_source,
        "is_training_split": split_name in TRAINING_SPLITS[config_name],
    }


def _reuse_split(
    output_root: Path,
    config_name: str,
    split_name: str,
    expected_source: str,
) -> tuple[Dataset, dict] | None:
    subset_path = output_root / config_name / split_name
    if not subset_path.exists():
        return None
    dataset = load_from_disk(str(subset_path))
    if len(dataset) > 0:
        _check_row_images(dataset[0])
    return dataset, {
        "config_name": config_name,
        "split": split_name,
        "path": str(Path(config_name) / split_name),
        "rows": len(dataset),
        "source_dataset_id": expected_source,
        "is_training_split": split_name in TRAINING_SPLITS[config_name],
    }


def _materialize_split(
    dataset: Dataset,
    output_root: Path,
    config_name: str,
    split_name: str,
    expected_source: str,
) -> dict:
    reused = _reuse_split(output_root, config_name, split_name, expected_source)
    if reused is not None:
        return reused[1]
    return _save_split(
        dataset,
        output_root,
        config_name,
        split_name,
        expected_source,
    )


def _write_manifest(output_root: Path, manifest: dict) -> None:
    with (output_root / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _reject_legacy_output(output_root: Path) -> None:
    for config_name in _SOURCE_IDS:
        config_path = output_root / config_name
        if not config_path.exists():
            continue
        has_new_layout = any(
            (config_path / split_name).exists()
            for split_name in ("train", "validation")
        )
        if not has_new_layout and (config_path / "state.json").exists():
            raise FileExistsError(
                f"Legacy flat dataset found at {config_path}. Move or remove the "
                "existing output before rebuilding the train/validation layout."
            )


def _build_librispeech(
    original: Dataset | None,
    output_root: Path,
    tokenizer,
    args,
) -> list[dict]:
    entries = []
    config_name = "librispeech"
    expected_source = _SOURCE_IDS["librispeech"]

    # Train: load fresh from the selected LibriSpeech clean 360-hour split.
    force = getattr(args, "force", False) or getattr(
        args, "force_librispeech", False
    )
    reuse_train = _reuse_split(output_root, config_name, "train", expected_source)
    if reuse_train is not None and not force:
        entries.append(reuse_train[1])
    else:
        if reuse_train is not None and force:
            print(
                "  Rebuilding LibriSpeech train from "
                "openslr/librispeech_asr clean/train.360"
            )
        else:
            print(
                "  Loading LibriSpeech train from "
                "openslr/librispeech_asr clean/train.360 ..."
            )
        train_ds = preprocess_librispeech_asr(
            subset="clean",
            split="train.360",
            max_samples=getattr(args, "librispeech_train_samples", None),
            tokenizer=tokenizer,
            tokenizer_name=args.tokenizer,
            num_proc=args.num_proc,
        )
        train_ds = _set_target_split(train_ds, "train", args.num_proc)
        entry = _save_split(
            train_ds, output_root, config_name, "train", expected_source,
            force=force,
        )
        entries.append(entry)

    # Validation: use the selected 2,703-row clean validation split.
    reuse_val = _reuse_split(output_root, config_name, "validation", expected_source)
    if reuse_val is not None and not force:
        entries.append(reuse_val[1])
    else:
        if reuse_val is not None and force:
            print(
                "  Rebuilding LibriSpeech validation from "
                "openslr/librispeech_asr clean/validation"
            )
        else:
            print(
                "  Loading LibriSpeech validation from "
                "openslr/librispeech_asr clean/validation ..."
            )
        val_ds = preprocess_librispeech_asr(
            subset="clean",
            split="validation",
            max_samples=args.librispeech_val_samples,
            tokenizer=tokenizer,
            tokenizer_name=args.tokenizer,
            num_proc=args.num_proc,
        )
        val_ds = _set_target_split(val_ds, "validation", args.num_proc)

        entry = _save_split(
            val_ds, output_root, config_name, "validation", expected_source,
            force=force,
        )
        entries.append(entry)

    return entries


def _build_densefusion(
    original: Dataset | None,
    output_root: Path,
    tokenizer,
    args,
) -> list[dict]:
    config_name = "densefusion"
    expected_source = _SOURCE_IDS["densefusion"]
    entries = []

    reused_train = _reuse_split(output_root, config_name, "train", expected_source)
    reused_val = _reuse_split(output_root, config_name, "validation", expected_source)
    if reused_train is not None:
        entries.append(reused_train[1])
    if reused_val is not None:
        entries.append(reused_val[1])

    if reused_train is not None and reused_val is not None:
        return entries

    if original is None:
        raise ValueError(
            "Need --input (full-v0) to build missing densefusion split(s)."
        )
        print("  Splitting densefusion rows from full-v0 90/10...")
        ds = _select_source_from_fullv0(original, expected_source, args.num_proc)
        ds = ds.sort("row_id")
        ds = ds.map(
            lambda row: _add_token_length(row, tokenizer),
            num_proc=args.num_proc,
            desc="Adding token length to densefusion",
        )
        train_ds, val_ds = _deterministic_split(ds, 0.9)

        if reused_train is None:
            train_split = _set_target_split(train_ds, "train", args.num_proc)
            entry = _save_split(
                train_split, output_root, config_name, "train", expected_source
            )
            entries.append(entry)

        if reused_val is None:
            val_split = val_ds
            if args.densefusion_val_samples is not None and len(val_ds) > args.densefusion_val_samples:
                val_split = val_ds.select(range(args.densefusion_val_samples))
            val_split = _set_target_split(val_split, "validation", args.num_proc)
            entry = _save_split(
                val_split, output_root, config_name, "validation", expected_source
            )
            entries.append(entry)

    return entries


def _build_fineweb_edu(
    output_root: Path,
    tokenizer,
    args,
) -> list[dict]:
    config_name = "fineweb-edu"
    expected_source = _SOURCE_IDS["fineweb-edu"]
    entries = []

    reused_train = _reuse_split(output_root, config_name, "train", expected_source)
    reused_val = _reuse_split(output_root, config_name, "validation", expected_source)
    if reused_train is not None:
        entries.append(reused_train[1])
    if reused_val is not None:
        entries.append(reused_val[1])

    if reused_train is None or reused_val is None:
        print(f"  Preprocessing {args.text_samples} fineweb-edu rows, 90/10 split...")
        full_ds = preprocess_fineweb_edu(
            split="train",
            max_samples=args.text_samples,
            max_chars=None,
            canvas_width=1024,
            canvas_height=1024,
            font_size=14,
            max_output_tokens=None,
            tokenizer=tokenizer,
            tokenizer_name=args.tokenizer,
            num_proc=args.num_proc,
            source_path=args.fineweb_source_path,
        )
        train_ds, val_ds = _deterministic_split(full_ds, 0.9)

        if reused_train is None:
            train_split = _set_target_split(train_ds, "train", args.num_proc)
            entry = _save_split(
                train_split, output_root, config_name, "train", expected_source
            )
            entries.append(entry)

        if reused_val is None:
            val_split = _set_target_split(val_ds, "validation", args.num_proc)
            entry = _save_split(
                val_split, output_root, config_name, "validation", expected_source
            )
            entries.append(entry)

    return entries


def _build_smoltalk(
    output_root: Path,
    tokenizer,
    args,
) -> list[dict]:
    entries = []
    config_name = "smoltalk"
    expected_source = _SOURCE_IDS["smoltalk"]

    # source train -> target train
    reused_train = _reuse_split(output_root, config_name, "train", expected_source)
    if reused_train is not None:
        _, entry = reused_train
        entries.append(entry)
    else:
        print(f"  Preprocessing smoltalk train ({args.text_samples} rows)...")
        train_ds = preprocess_smoltalk(
            split="train",
            max_samples=args.text_samples,
            max_chars=None,
            canvas_width=1024,
            canvas_height=1024,
            font_size=14,
            max_output_tokens=None,
            tokenizer=tokenizer,
            tokenizer_name=args.tokenizer,
            num_proc=args.num_proc,
            source_path=args.smoltalk_source_path,
        )
        train_ds = _set_target_split(train_ds, "train", args.num_proc)
        entry = _save_split(
            train_ds, output_root, config_name, "train", expected_source
        )
        entries.append(entry)

    # source test -> target validation
    reused_val = _reuse_split(output_root, config_name, "validation", expected_source)
    if reused_val is not None:
        _, entry = reused_val
        entries.append(entry)
    else:
        print("  Preprocessing smoltalk test -> validation...")
        val_ds = preprocess_smoltalk(
            split="test",
            max_samples=args.smoltalk_val_samples,
            max_chars=None,
            canvas_width=1024,
            canvas_height=1024,
            font_size=14,
            max_output_tokens=None,
            tokenizer=tokenizer,
            tokenizer_name=args.tokenizer,
            num_proc=args.num_proc,
            source_path=args.smoltalk_source_path,
        )
        val_ds = _set_target_split(val_ds, "validation", args.num_proc)
        entry = _save_split(
            val_ds, output_root, config_name, "validation", expected_source
        )
        entries.append(entry)
    return entries


def _upload_subsets(
    output_root: Path,
    subsets: list[dict],
    upload_repo: str,
    num_proc: int,
) -> None:
    from datasets import load_from_disk

    for entry in subsets:
        config_name = entry["config_name"]
        split_name = entry["split"]
        subset_path = output_root / config_name / split_name
        print(f"  Uploading {config_name}/{split_name} -> {upload_repo} ...")
        ds = load_from_disk(str(subset_path))
        ds.push_to_hub(upload_repo, config_name=config_name, split=split_name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild univi-3M-v0 as four configs (librispeech, densefusion, fineweb-edu, smoltalk) with train/validation splits."
    )
    parser.add_argument("--input", help="Existing full-v0 path (required for densefusion reuse).")
    parser.add_argument("--output", required=True, help="New artifact root path.")
    parser.add_argument("--text-samples", type=int, default=1_000_000, help="FineWeb-Edu and SmolTalk training sample count.")
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing LibriSpeech (and other) splits instead of reusing them.",
    )
    parser.add_argument(
        "--force-librispeech",
        action="store_true",
        help="Rebuild only LibriSpeech splits while reusing other configurations.",
    )
    parser.add_argument(
        "--librispeech-train-samples", type=int, default=None,
        help="Cap for LibriSpeech training split (applied to clean/train.360).",
    )
    parser.add_argument(
        "--librispeech-val-samples", type=int, default=None,
        help="Cap for the clean LibriSpeech validation split.",
    )
    parser.add_argument(
        "--densefusion-val-samples", type=int, default=None,
        help="Cap for densefusion validation split.",
    )
    parser.add_argument(
        "--smoltalk-val-samples", type=int, default=None,
        help="Cap for smoltalk validation split.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--tokenizer", default="unsloth/gemma-4-E2B-it")
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument(
        "--offline", action="store_true",
        help="Use complete FineWeb-Edu and SmolTalk snapshots already in the Hugging Face cache.",
    )
    parser.add_argument(
        "--push-repo", "--upload-repo", default=None, dest="upload_repo",
        help="Push all subsets to this Hub repo after local processing completes.",
    )
    args = parser.parse_args(argv)

    args.fineweb_source_path = None
    args.smoltalk_source_path = None
    if args.offline:
        from huggingface_hub import snapshot_download

        args.fineweb_source_path = snapshot_download(
            _SOURCE_IDS["fineweb-edu"], repo_type="dataset", local_files_only=True
        )
        args.smoltalk_source_path = snapshot_download(
            _SOURCE_IDS["smoltalk"], repo_type="dataset", local_files_only=True
        )

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    _reject_legacy_output(output_root)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=args.offline
    )
    manifest = {
        "artifact": "univi-3M-v0",
        "tokenizer": args.tokenizer,
        "max_output_tokens": args.max_output_tokens,
        "canvas": [1024, 1024],
        "subsets": [],
    }

    original = None
    if args.input:
        original = load_from_disk(args.input)
        manifest["input_artifact"] = args.input
    # librispeech no longer requires --input; it loads from source.
    # Densefusion still needs original for initial build (handled in _build_densefusion).

    # 1. librispeech
    print("Building librispeech...")
    manifest["subsets"].extend(
        _build_librispeech(original, output_root, tokenizer, args)
    )
    _write_manifest(output_root, manifest)

    # 2. densefusion. Existing splits can be reused without loading full-v0.
    print("Building densefusion...")
    manifest["subsets"].extend(
        _build_densefusion(original, output_root, tokenizer, args)
    )
    _write_manifest(output_root, manifest)

    # 3. fineweb-edu
    print("Building fineweb-edu...")
    manifest["subsets"].extend(
        _build_fineweb_edu(output_root, tokenizer, args)
    )
    _write_manifest(output_root, manifest)

    # 4. smoltalk
    print("Building smoltalk...")
    manifest["subsets"].extend(
        _build_smoltalk(output_root, tokenizer, args)
    )
    _write_manifest(output_root, manifest)

    print(f"\nManifest written to {output_root / 'manifest.json'}")
    print(f"Total subsets: {len(manifest['subsets'])}")

    if args.upload_repo:
        print(f"\nUploading all subsets to {args.upload_repo} ...")
        _upload_subsets(output_root, manifest["subsets"], args.upload_repo, args.num_proc)
        print("Upload complete.")


if __name__ == "__main__":
    main()
