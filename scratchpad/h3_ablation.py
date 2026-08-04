"""
H3 random-string OCR ablation (encoder-free VLM).

Reconstructs the visual-grounding diagnostic for the H3 "prior-proof" OCR run
(``train_encoder_free.py`` + ``configs/h3_randstr.yaml``). Because the H3 lane
is transcribe-random-letters, there is NO language shortcut: any loss below the
guessing floor, and any token-accuracy above the blank-image baseline, can only
come from READING pixels. This script measures exactly that, per row:

  * answer-only CE (nats/tok) under three image conditions --
      aligned  : the row's real spectrogram... no, its real rendered-string image
      permuted : a donor image from a *different* row (Modality-Permutation Control)
      blank    : a mid-gray image (no pixel information at all)
  * teacher-forced next-token accuracy on the answer tokens under each condition.

Signals (pre-registered in CLAUDE.md):
  Δperm% = (loss_permuted - loss_aligned) / loss_aligned   -- grounding magnitude
  Δblank% = (loss_blank   - loss_aligned) / loss_aligned   -- is the real image used
  tok-acc(aligned) vs tok-acc(blank)  -- reading gain over the self-calibrated
                                         guessing baseline for THIS tokenization.

Reuses the training seams directly (no monkeypatching): ``build_encoder_free_univi``
builds the model AND warm-loads ``embedder.pt`` when ``model_name`` is a checkpoint
dir, so ``--checkpoint <dir>`` loads the trained decoder + embedder together.

Usage:
  HF_HUB_OFFLINE=1 uv run python scratchpad/h3_ablation.py \
      --checkpoint data/checkpoints/h3-randstr-v0/best --tag h3
  HF_HUB_OFFLINE=1 uv run python scratchpad/h3_ablation.py \
      --checkpoint data/checkpoints/encoder-free-v0/best --tag h1-baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# This script lives in scratchpad/; put the repo root on the path so the
# training module (at repo root) is importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("h3_ablation")

# Reuse the exact training seams (CPU-safe imports; heavy CUDA is deferred inside
# build_encoder_free_univi).
from train_encoder_free import (  # noqa: E402
    EFConfig,
    _image_transform,
    _message_to_string,
    build_encoder_free_univi,
)

IGNORE_INDEX = -100
FLOOR_PATH = "data/materialized/h3-randstr-v0/floor.json"


def _ids_for(messages, tokenizer, image_token: str, num_patches: int, add_gen: bool):
    """Tokenize a message list exactly as ``tokenize_sample`` does, with control
    over ``add_generation_prompt`` so we can split prompt vs answer."""
    block = image_token * num_patches
    chat = [
        {"role": m["role"], "content": _message_to_string(m["content"], block)}
        for m in messages
    ]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=add_gen)
    return tokenizer.encode(text, add_special_tokens=False)


def _answer_span(messages, tokenizer, image_token: str, num_patches: int):
    """Return (full_ids, answer_start) where answer_start is the index of the
    first supervised answer token. Falls back to a suffix search if the
    prompt-only encoding is not a clean prefix of the full encoding."""
    full = _ids_for(messages, tokenizer, image_token, num_patches, add_gen=False)
    prompt = _ids_for(messages[:-1], tokenizer, image_token, num_patches, add_gen=True)
    if full[: len(prompt)] == prompt:
        return full, len(prompt)
    # Fallback: locate the answer text tokens inside the full sequence.
    ans_text = _message_to_string(messages[-1]["content"], image_token * num_patches)
    ans_ids = tokenizer.encode(ans_text, add_special_tokens=False)
    for start in range(len(full) - len(ans_ids), -1, -1):
        if full[start : start + len(ans_ids)] == ans_ids:
            return full, start
    # Last resort: supervise the trailing half.
    return full, len(prompt)


def _row_eval(vlm, tokenizer, image_token_id, image_token, num_patches, transform,
              messages, image, device):
    """One forward pass; return (answer_ce_nats, answer_tok_acc, n_answer_tokens)."""
    import torch
    import torch.nn.functional as F

    full_ids, answer_start = _answer_span(messages, tokenizer, image_token, num_patches)
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    # Answer-only labels: mask everything before the answer, plus any image /
    # placeholder tokens (they live in the prompt, but mask defensively).
    labels = input_ids.clone()
    labels[:, :answer_start] = IGNORE_INDEX
    labels[input_ids == image_token_id] = IGNORE_INDEX

    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

    pixel_values = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                          enabled=(device == "cuda")):
        hidden = vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            position_ids=position_ids,
        )
    lm_w = vlm.get_lm_head_weight().float()
    logits = hidden[0].float() @ lm_w.t()          # [S, V]
    shift_logits = logits[:-1]                      # predict token t+1 from t
    shift_labels = labels[0, 1:]
    valid = shift_labels != IGNORE_INDEX
    if int(valid.sum()) == 0:
        return float("nan"), float("nan"), 0
    sl = shift_logits[valid]
    tgt = shift_labels[valid]
    ce = F.cross_entropy(sl, tgt, reduction="mean").item()
    acc = (sl.argmax(dim=-1) == tgt).float().mean().item()
    return ce, acc, int(valid.sum())


def _mean(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")


def run(checkpoint: str, tag: str, max_samples: int, seed: int, output: str | None,
        connector: str = "linear", mlp_ratio: int = 4):
    import torch
    from datasets import load_from_disk

    cfg = EFConfig()
    cfg.model_name = checkpoint  # → warm-loads decoder (from_pretrained) + embedder.pt
    cfg.connector = connector    # MUST match the checkpoint's embedder architecture
    cfg.mlp_ratio = mlp_ratio
    logger.info("Building encoder-free model from checkpoint: %s", checkpoint)
    vlm, tokenizer, image_token_id = build_encoder_free_univi(cfg)
    vlm.eval()
    vlm.model.config.use_cache = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_patches = vlm.num_patches
    image_token = cfg.image_token
    transform = _image_transform(cfg.image_size)

    ds = load_from_disk("data/materialized/h3-randstr-v0/random-strings/validation")
    ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))
    n = len(ds)
    logger.info("Evaluating %d rows (num_patches=%d)", n, num_patches)

    # Deterministic donor permutation (derangement-ish: shift by 1).
    donor_idx = [(i + 1) % n for i in range(n)]
    from PIL import Image
    blank = Image.new("RGB", (cfg.image_size, cfg.image_size), (128, 128, 128))

    rows = {"aligned_ce": [], "permuted_ce": [], "blank_ce": [],
            "aligned_acc": [], "permuted_acc": [], "blank_acc": [], "n_tok": []}

    for i in range(n):
        msgs = ds[i]["messages"]
        img = ds[i]["images"][0]
        donor = ds[donor_idx[i]]["images"][0]

        a_ce, a_acc, ntok = _row_eval(vlm, tokenizer, image_token_id, image_token,
                                      num_patches, transform, msgs, img, device)
        p_ce, p_acc, _ = _row_eval(vlm, tokenizer, image_token_id, image_token,
                                   num_patches, transform, msgs, donor, device)
        b_ce, b_acc, _ = _row_eval(vlm, tokenizer, image_token_id, image_token,
                                   num_patches, transform, msgs, blank, device)
        rows["aligned_ce"].append(a_ce); rows["permuted_ce"].append(p_ce); rows["blank_ce"].append(b_ce)
        rows["aligned_acc"].append(a_acc); rows["permuted_acc"].append(p_acc); rows["blank_acc"].append(b_acc)
        rows["n_tok"].append(ntok)
        if (i + 1) % 25 == 0:
            logger.info("  %d/%d | aligned ce %.3f acc %.3f", i + 1, n,
                        _mean(rows["aligned_ce"]), _mean(rows["aligned_acc"]))

    aligned_ce = _mean(rows["aligned_ce"])
    permuted_ce = _mean(rows["permuted_ce"])
    blank_ce = _mean(rows["blank_ce"])
    d_perm = (permuted_ce - aligned_ce) / aligned_ce if aligned_ce else float("nan")
    d_blank = (blank_ce - aligned_ce) / aligned_ce if aligned_ce else float("nan")

    floor = None
    try:
        floor = json.loads(Path(FLOOR_PATH).read_text()).get("no_reading_floor_nats_per_token")
    except Exception:
        pass

    result = {
        "tag": tag,
        "checkpoint": checkpoint,
        "n_samples": n,
        "answer_only_ce_nats": {
            "aligned": aligned_ce, "permuted": permuted_ce, "blank": blank_ce,
        },
        "teacher_forced_tok_acc": {
            "aligned": _mean(rows["aligned_acc"]),
            "permuted": _mean(rows["permuted_acc"]),
            "blank": _mean(rows["blank_acc"]),
        },
        "delta_perm_rel": d_perm,
        "delta_blank_rel": d_blank,
        "no_reading_floor_nats": floor,
        "mean_answer_tokens": _mean([float(t) for t in rows["n_tok"]]),
    }

    print("\n================ H3 ABLATION (%s) ================" % tag)
    print(f"checkpoint         : {checkpoint}")
    print(f"samples            : {n}")
    print(f"no-reading floor   : {floor:.3f} nats/tok (answer letters only)" if floor else "")
    print("--- answer-only CE (nats/tok) ---")
    print(f"  aligned   : {aligned_ce:.3f}")
    print(f"  permuted  : {permuted_ce:.3f}   Δperm = {d_perm*100:+.2f}%")
    print(f"  blank     : {blank_ce:.3f}   Δblank = {d_blank*100:+.2f}%")
    print("--- teacher-forced answer token-accuracy ---")
    print(f"  aligned   : {result['teacher_forced_tok_acc']['aligned']*100:.2f}%")
    print(f"  permuted  : {result['teacher_forced_tok_acc']['permuted']*100:.2f}%")
    print(f"  blank     : {result['teacher_forced_tok_acc']['blank']*100:.2f}%  (guessing baseline)")
    reading_gain = (result['teacher_forced_tok_acc']['aligned']
                    - result['teacher_forced_tok_acc']['blank']) * 100
    print(f"  reading gain (aligned - blank): {reading_gain:+.2f} pts")
    print("=" * 56)

    out_path = Path(output) if output else Path(f"data/eval/h3-ablation-{tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %s", out_path)
    return result


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Encoder-free checkpoint dir (decoder + embedder.pt).")
    p.add_argument("--tag", default="h3", help="Label for output + report.")
    p.add_argument("--max-samples", type=int, default=200, help="Rows from the validation split.")
    p.add_argument("--seed", type=int, default=3407, help="Shuffle seed for row sampling.")
    p.add_argument("--output", default=None, help="Override output JSON path.")
    p.add_argument("--connector", default="linear", choices=["linear", "mlp"],
                   help="Embedder connector — MUST match the checkpoint's architecture.")
    p.add_argument("--mlp-ratio", type=int, default=4, help="MLP connector hidden ratio.")
    a = p.parse_args(argv)
    run(a.checkpoint, a.tag, a.max_samples, a.seed, a.output, a.connector, a.mlp_ratio)


if __name__ == "__main__":
    main()
