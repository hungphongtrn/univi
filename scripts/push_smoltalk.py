from datasets import DatasetDict, load_dataset
from pathlib import Path

BASE = Path("data/materialized/univi-3M-v0-split")
REPO = "hungphongtrn/univi-3M-v0"
subset = "smoltalk"

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
