from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, load_from_disk


def prepare_source_eval(
    source_name: str,
    max_samples: int,
    offset: int = 0,
    **preprocessor_kwargs,
) -> Dataset:
    if source_name == "librispeech":
        from data.preprocessing.librispeech_asr import preprocess_librispeech_asr

        return preprocess_librispeech_asr(
            max_samples=max_samples,
            offset=offset,
            include_native=True,
            **preprocessor_kwargs,
        )
    if source_name == "densefusion":
        from data.preprocessing.densefusion import preprocess_densefusion

        return preprocess_densefusion(
            max_samples=max_samples,
            offset=offset,
            include_native=True,
            **preprocessor_kwargs,
        )
    if source_name == "fineweb":
        from data.preprocessing.fineweb_edu import preprocess_fineweb_edu

        return preprocess_fineweb_edu(
            max_samples=max_samples,
            offset=offset,
            include_native=True,
            **preprocessor_kwargs,
        )
    if source_name == "smoltalk":
        from data.preprocessing.smoltalk import preprocess_smoltalk

        return preprocess_smoltalk(
            max_samples=max_samples,
            offset=offset,
            include_native=True,
            **preprocessor_kwargs,
        )
    msg = f"Unknown source: {source_name}"
    raise ValueError(msg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare per-source eval datasets with native fields preserved."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output root directory for eval datasets.",
    )
    parser.add_argument(
        "--librispeech-samples",
        type=int,
        default=100,
        help="Number of LibriSpeech eval samples.",
    )
    parser.add_argument(
        "--librispeech-offset",
        type=int,
        default=0,
        help="Row offset for LibriSpeech eval selection.",
    )
    parser.add_argument(
        "--densefusion-samples",
        type=int,
        default=100,
        help="Number of DenseFusion eval samples.",
    )
    parser.add_argument(
        "--densefusion-offset",
        type=int,
        default=0,
        help="Row offset for DenseFusion eval selection.",
    )
    parser.add_argument(
        "--fineweb-samples",
        type=int,
        default=100,
        help="Number of FineWeb-Edu eval samples.",
    )
    parser.add_argument(
        "--fineweb-offset",
        type=int,
        default=0,
        help="Row offset for FineWeb-Edu eval selection.",
    )
    parser.add_argument(
        "--smoltalk-samples",
        type=int,
        default=100,
        help="Number of SmolTalk eval samples.",
    )
    parser.add_argument(
        "--smoltalk-offset",
        type=int,
        default=0,
        help="Row offset for SmolTalk eval selection.",
    )
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    configs = {
        "librispeech": (args.librispeech_samples, args.librispeech_offset),
        "densefusion": (args.densefusion_samples, args.densefusion_offset),
        "fineweb": (args.fineweb_samples, args.fineweb_offset),
        "smoltalk": (args.smoltalk_samples, args.smoltalk_offset),
    }

    results: dict[str, Dataset] = {}
    for name, (samples, offset) in configs.items():
        if samples > 0:
            ds = prepare_source_eval(name, samples, offset=offset)
            out_path = output / name
            ds.save_to_disk(str(out_path))
            results[name] = ds
            print(
                f"Eval dataset '{name}' saved to {out_path} with {len(ds)} rows.",
                flush=True,
            )

    if not results:
        print("No eval datasets requested.", flush=True)


if __name__ == "__main__":
    main()
