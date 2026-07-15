from __future__ import annotations

import argparse

from datasets import Dataset, concatenate_datasets


def merge_and_shuffle(sources: dict[str, Dataset], seed: int = 42) -> Dataset:
    non_empty = {name: ds for name, ds in sources.items() if len(ds) > 0}
    if not non_empty:
        raise ValueError("No non-empty sources provided to merge.")
    merged = concatenate_datasets(list(non_empty.values()))
    shuffled = merged.shuffle(seed=seed)
    return shuffled


def _load_source(source_name: str, samples: int, offset: int = 0, num_proc: int = 4) -> Dataset:
    kw_samples = None if samples == -1 else samples
    if source_name == "librispeech":
        from data.preprocessing.librispeech_asr import preprocess_librispeech_asr

        return preprocess_librispeech_asr(max_samples=kw_samples, offset=offset, num_proc=num_proc)
    if source_name == "densefusion":
        from data.preprocessing.densefusion import preprocess_densefusion

        return preprocess_densefusion(max_samples=kw_samples, offset=offset, num_proc=num_proc)
    if source_name == "fineweb":
        from data.preprocessing.fineweb_edu import preprocess_fineweb_edu

        return preprocess_fineweb_edu(max_samples=kw_samples, offset=offset, num_proc=num_proc)
    if source_name == "smoltalk":
        from data.preprocessing.smoltalk import preprocess_smoltalk

        return preprocess_smoltalk(max_samples=kw_samples, offset=offset, num_proc=num_proc)
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
        help="Number of LibriSpeech samples (0 to skip, -1 for all).",
    )
    parser.add_argument(
        "--librispeech-offset",
        type=int,
        default=0,
        help="Row offset for LibriSpeech selection.",
    )
    parser.add_argument(
        "--densefusion-samples",
        type=int,
        default=0,
        help="Number of DenseFusion samples (0 to skip, -1 for all).",
    )
    parser.add_argument(
        "--densefusion-offset",
        type=int,
        default=0,
        help="Row offset for DenseFusion selection.",
    )
    parser.add_argument(
        "--fineweb-samples",
        type=int,
        default=0,
        help="Number of FineWeb-Edu samples (0 to skip, -1 for all).",
    )
    parser.add_argument(
        "--fineweb-offset",
        type=int,
        default=0,
        help="Row offset for FineWeb-Edu selection.",
    )
    parser.add_argument(
        "--smoltalk-samples",
        type=int,
        default=0,
        help="Number of SmolTalk samples (0 to skip, -1 for all).",
    )
    parser.add_argument(
        "--smoltalk-offset",
        type=int,
        default=0,
        help="Row offset for SmolTalk selection.",
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
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help="Number of parallel processes for dataset loading and .map() (default: 4).",
    )

    args = parser.parse_args(argv)

    source_configs = {
        "librispeech": args.librispeech_samples,
        "densefusion": args.densefusion_samples,
        "fineweb": args.fineweb_samples,
        "smoltalk": args.smoltalk_samples,
    }
    source_offsets = {
        "librispeech": args.librispeech_offset,
        "densefusion": args.densefusion_offset,
        "fineweb": args.fineweb_offset,
        "smoltalk": args.smoltalk_offset,
    }

    configs_to_load = [
        (name, samples, source_offsets[name])
        for name, samples in source_configs.items()
        if samples != 0
    ]

    sources: dict[str, Dataset] = {}
    for name, samples, offset in configs_to_load:
        sources[name] = _load_source(name, samples, offset, num_proc=args.num_proc)

    merged = merge_and_shuffle(sources, seed=args.seed)
    merged.save_to_disk(args.output)
    print(
        f"Merged dataset saved to {args.output} with {len(merged)} rows.",
        flush=True,
    )


if __name__ == "__main__":
    main()
