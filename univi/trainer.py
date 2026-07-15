"""
Shared training helpers for visual modality unification experiments.

Loss masking:
    UnslothVisionDataCollator applies response-only loss masking automatically
    via the chat template processing — user-turn tokens receive -100 labels
    so they do not contribute to the loss. No explicit masking step is needed.
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import torch
import yaml
from datasets import concatenate_datasets, load_dataset as hf_load
from datasets import load_from_disk

# Unsloth must be imported before trl/transformers/peft so its
# monkey-patches are applied before those libraries are initialised.
from unsloth import FastVisionModel, UnslothVisionDataCollator, is_bfloat16_supported

from trl import SFTConfig, SFTTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(config: dict):
    model_cfg = config["model"]
    kwargs = dict(
        model_name=model_cfg["name"],
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        use_gradient_checkpointing=model_cfg.get(
            "use_gradient_checkpointing", True
        ),
    )
    for opt_key in (
        "max_seq_length",
        "gpu_memory_utilization",
        "float8_kv_cache",
        "unsloth_tiled_mlp",
    ):
        if opt_key in model_cfg:
            kwargs[opt_key] = model_cfg[opt_key]
    model, tokenizer = FastVisionModel.from_pretrained(**kwargs)
    return model, tokenizer


def apply_lora(model, lora_cfg: dict):
    model = FastVisionModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        finetune_vision_layers=lora_cfg.get("finetune_vision_layers", True),
    )
    return model


def load_dataset(config: dict):
    path = config["dataset"]["path"]
    dataset_cfg = config["dataset"]
    local_path = Path(path)
    subsets = dataset_cfg.get("subsets", [])
    seed = dataset_cfg.get("shuffle_seed", 42)
    if local_path.exists():
        manifest_path = local_path / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open() as handle:
                manifest = json.load(handle)
            entries = [
                item
                for item in manifest["subsets"]
                if item.get("is_training_split", True)
                and (not subsets or item["config_name"] in subsets)
            ]
            datasets = [load_from_disk(local_path / item["path"]) for item in entries]
            return concatenate_datasets(datasets).shuffle(seed=seed)
        return load_from_disk(path)
    hub_repo = config["dataset"].get("hf_hub_repo_id")
    if hub_repo:
        if subsets:
            split_map = dataset_cfg.get("train_splits", {})
            datasets = []
            for name in subsets:
                for split in split_map.get(name, ["train"]):
                    datasets.append(hf_load(hub_repo, name=name, split=split))
            return concatenate_datasets(datasets).shuffle(seed=seed)
        return hf_load(hub_repo, split="train")
    raise FileNotFoundError(
        f"Dataset not found at {path} and no hf_hub_repo_id configured."
    )


def train(config: dict):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, tokenizer = build_model(config)
    model = apply_lora(model, config["lora"])

    model = FastVisionModel.for_training(model)

    tokenizer = tokenizer.get_chat_template()

    dataset = load_dataset(config)
    train_cfg = config["training"]

    training_args = SFTConfig(
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        max_steps=train_cfg.get("max_steps", -1),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=train_cfg.get("warmup_steps", 0),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "linear"),
        optim=train_cfg.get("optim", "adamw_8bit"),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 25),
        output_dir=train_cfg["output_dir"],
        report_to=train_cfg.get("report_to", "none"),
        remove_unused_columns=train_cfg.get("remove_unused_columns", False),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=train_cfg["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()

    trainer.save_model(train_cfg["output_dir"] + "/final")
    tokenizer.save_pretrained(train_cfg["output_dir"] + "/final")

    hub_cfg = config.get("hub", {})
    model_repo_id = hub_cfg.get("model_repo_id", "")
    hf_token = os.environ.get("HF_TOKEN")
    if model_repo_id and hf_token:
        model.push_to_hub(model_repo_id, token=True)
        tokenizer.push_to_hub(model_repo_id, token=True)

    if torch.cuda.is_available():
        allocated = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(
            "CUDA peak memory: "
            f"allocated={allocated:.3f} GiB, reserved={reserved:.3f} GiB"
        )

    return trainer
