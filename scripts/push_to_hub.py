from datasets import DatasetDict, load_dataset
from pathlib import Path
import json

BASE = Path("data/materialized/univi-3M-v0-split")
REPO = "hungphongtrn/univi-3M-v0"
SUBSETS = ["librispeech", "densefusion", "fineweb-edu", "smoltalk"]

manifest = json.loads((BASE / "manifest.json").read_text())
canvas = manifest.get("canvas", [1024, 1024])
max_output_tokens = manifest.get("max_output_tokens", 1024)
tokenizer = manifest.get("tokenizer", "")

for subset in SUBSETS:
    print(f"\n=== Loading {subset} ===")
    dataset = DatasetDict({
        "train": load_dataset(str(BASE / subset), split="train"),
        "validation": load_dataset(str(BASE / subset), split="validation"),
    })

    print(f"  train: {len(dataset['train'])} rows")
    print(f"  validation: {len(dataset['validation'])} rows")

    print(f"  Pushing to {REPO} (config={subset}) ...")
    dataset.push_to_hub(REPO, config_name=subset, private=False)
    print(f"  Done.")

print("\nAll subsets pushed.")
