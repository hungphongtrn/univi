"""Range-download ONLY the 12B unified vision embedder (~100MB) from the 24GB
single-file safetensors, and save a remapped state_dict for Gemma4UnifiedVisionEmbedder.

Avoids the full 24GB pull (and the Xet client hang) — plain curl range requests to
the resolve URL work on this box. Header offsets come from /tmp/g12b_header.json.
"""
import json, subprocess, struct
from pathlib import Path
import torch

URL = "https://huggingface.co/google/gemma-4-12B-it/resolve/main/model.safetensors"
HEADER_LEN = 88952
BASE = 8 + HEADER_LEN  # data section start
OUT = Path("data/hybrid/gemma12b-vision")
OUT.mkdir(parents=True, exist_ok=True)

hdr = json.load(open("/tmp/g12b_header.json"))
hdr.pop("__metadata__", None)

# checkpoint key -> module param name on Gemma4UnifiedVisionEmbedder
KEYMAP = {
    "model.vision_embedder.patch_ln1.weight": "patch_ln1.weight",
    "model.vision_embedder.patch_ln1.bias": "patch_ln1.bias",
    "model.vision_embedder.patch_dense.weight": "patch_dense.weight",
    "model.vision_embedder.patch_dense.bias": "patch_dense.bias",
    "model.vision_embedder.patch_ln2.weight": "patch_ln2.weight",
    "model.vision_embedder.patch_ln2.bias": "patch_ln2.bias",
    "model.vision_embedder.pos_embedding": "pos_embedding",
    "model.vision_embedder.pos_norm.weight": "pos_norm.weight",
    "model.vision_embedder.pos_norm.bias": "pos_norm.bias",
    "model.embed_vision.embedding_projection.weight": "multimodal_embedder.embedding_projection.weight",
}

DTYPE = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}

state = {}
for ckpt_key, mod_key in KEYMAP.items():
    meta = hdr[ckpt_key]
    a, b = meta["data_offsets"]
    abs_a, abs_b = BASE + a, BASE + b - 1  # inclusive
    raw = subprocess.check_output(
        ["curl", "-sL", "-r", f"{abs_a}-{abs_b}", URL]
    )
    assert len(raw) == b - a, f"{ckpt_key}: got {len(raw)} want {b-a}"
    t = torch.frombuffer(bytearray(raw), dtype=DTYPE[meta["dtype"]]).reshape(meta["shape"]).clone()
    state[mod_key] = t
    print(f"  {mod_key:52s} {tuple(t.shape)!s:22s} {t.dtype}")

torch.save(state, OUT / "vision_embedder.pt")
print(f"\nSaved {len(state)} tensors -> {OUT/'vision_embedder.pt'}")
