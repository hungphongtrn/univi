from __future__ import annotations

import argparse
import sys

from datasets import Dataset, concatenate_datasets


def merge_and_shuffle(sources: dict[str, Dataset], seed: int = 42) -> Dataset:
    non_empty = {name: ds for name, ds in sources.items() if len(ds) > 0}
    if not non_empty:
        raise ValueError("No non-empty sources provided to merge.")
    datasets_to_merge = []
    for ds in non_empty.values():
        datasets_to_merge.append(ds)
    merged = concatenate_datasets(datasets_to_merge)
    shuffled = merged.shuffle(seed=seed)
    return shuffled


def _load_source(source_name: str, samples: int) -> Dataset:
    if source_name == "librispeech":
        from data.preprocessing.librispeech_asr import preprocess_librispeech_asr

        return preprocess_librispeech_asr(max_samples=samples)
    if source_name == "densefusion":
        from data.preprocessing.densefusion import preprocess_densefusion

        return preprocess_densefusion(max_samples=samples)
    if source_name == "fineweb":
        from data.preprocessing.fineweb_edu import preprocess_fineweb_edu

        return preprocess_fineweb_edu(max_samples=samples)
    if source_name == "smoltalk":
        from data.preprocessing.smoltalk import preprocess_smoltalk

        return preprocess_smoltalk(max_samples=samples)
    msg = f"Unknown source: {source_name}"
    raise ValueError(msg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple preprocessed datasets into one shuffled dataset."
    )
    parser.add_argument(
        "--librispeech-samples",
        type=int,
        default=0,
        help="Number of LibriSpeech samples (0 to skip).",
    )
    parser.add_argument(
        "--densefusion-samples",
        type=int,
        default=0,
        help="Number of DenseFusion samples (0 to skip).",
    )
    parser.add_argument(
        "--fineweb-samples",
        type=int,
        default=0,
        help="Number of FineWeb-Edu samples (0 to skip).",
    )
    parser.add_argument(
        "--smoltalk-samples",
        type=int,
        default=0,
        help="Number of SmolTalk samples (0 to skip).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the merged dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42).",
    )

    args = parser.parse_args(argv)

    source_configs = {
        "librispeech": args.librispeech_samples,
        "densefusion": args.densefusion_samples,
        "fineweb": args.fineweb_samples,
        "smoltalk": args.smoltalk_samples,
    }

    sources: dict[str, Dataset] = {}
    for name, samples in source_configs.items():
        if samples > 0:
            sources[name] = _load_source(name, samples)

    merged = merge_and_shuffle(sources, seed=args.seed)
    merged.save_to_disk(args.output)
    print(
        f"Merged dataset saved to {args.output} with {len(merged)} rows.",
        flush=True,
    )


if __name__ == "__main__":
    main()
