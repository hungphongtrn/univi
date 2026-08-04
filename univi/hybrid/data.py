"""
Data collator for the issue-#6 hybrid model (:mod:`univi.hybrid.model`).

The materialized Univi datasets store each row as ``messages`` (a chat list whose
content parts are ``{"type": "image"}`` / ``{"type": "text", "text": ...}``) plus
``images`` (a list of PIL images, one per image content part, in order).

This collator turns a batch of such rows into the tensors
:class:`~univi.hybrid.model.UniViHybridForConditionalGeneration` consumes:

- image pixels via the **real Gemma 4 image processor** (so the patch layout,
  position ids and per-image soft-token count exactly match the vision tower);
- text via the **Qwen 3 tokenizer**, with each image content part expanded into
  ``num_soft_tokens_per_image`` copies of the image-placeholder token so the
  count of placeholders equals the count of projected soft tokens;
- response-only label masking (Qwen ``<|im_start|>assistant`` .. ``<|im_end|>``),
  matching the training regime used elsewhere in the repo.

All images across the batch are processed in a single call so their patch tensors
are padded to a common length; the vision tower returns the valid soft tokens
per image, flattened in batch-then-image order, which lines up with the
row-major order ``masked_scatter`` fills placeholder positions.

Copyright (c) 2026, hungphongtrn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


IGNORE_INDEX = -100

#: Fill value of the "blank" (uninformative) image used by every grounding probe
#: in this repo — mid-gray, so it is neither black nor white and carries no
#: content. See ``scratchpad/hybrid_pretrained_ablation.py``.
BLANK_FILL = 128

#: Prefix of the extra batch keys carrying the blank-image tensors (H16). The
#: consumer strips the prefix and swaps them in for the aligned image tensors.
BLANK_PREFIX = "blank_"


def blank_like(image: Any) -> Any:
    """A content-free mid-gray image with ``image``'s geometry.

    Matches the probes for single-band (``L``) sources — every rendered-text and
    spectrogram lane — which is where the blank branch is used. NOTE: the probes
    write ``Image.new("RGB", size, 128)`` for RGB sources, and PIL reads a bare
    int in a multi-band mode as *the first band only* → (128, 0, 0), i.e. dark
    red rather than gray. Here an RGB source gets a true gray (128, 128, 128).
    """
    from PIL import Image

    mode = image.mode if image.mode in ("L", "RGB") else "RGB"
    fill = BLANK_FILL if mode == "L" else (BLANK_FILL, BLANK_FILL, BLANK_FILL)
    return Image.new(mode, image.size, fill)


@dataclass
class HybridCollator:
    """Collate materialized rows into hybrid-model inputs.

    Args:
        tokenizer: a Qwen 3 tokenizer (fast). Must already contain the
            image-placeholder special token; ``image_token_id`` is its id.
        image_processor: a ``Gemma4ImageProcessor``.
        image_token_id: placeholder token id (must equal the model config's).
        max_length: truncation budget for the assembled token sequence.
        response_only: when True, mask everything but assistant responses.
        emit_blank_images: when True, additionally emit ``blank_pixel_values`` /
            ``blank_image_position_ids`` / ``blank_num_soft_tokens_per_image``:
            the same batch with every image replaced by a mid-gray one of
            identical geometry, pushed through the *same* image processor so the
            soft-token accounting is identical and the blank tensors can be
            swapped straight in. Off by default — the extra processing is only
            paid for by the blind-branch consumers (H16).
    """

    tokenizer: Any
    image_processor: Any
    image_token_id: int
    max_length: int = 4096
    response_only: bool = True
    emit_blank_images: bool = False

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _build_row(
        self, messages: list[dict], soft_counts: list[int]
    ) -> tuple[list[int], list[int]]:
        """Return ``(input_ids, labels)`` for one row.

        ``soft_counts`` holds the soft-token count for each image in the row, in
        order of appearance; images are consumed left to right as ``{"type":
        "image"}`` content parts are encountered.
        """
        ids: list[int] = []
        labels: list[int] = []
        img_cursor = 0

        for msg in messages:
            role = msg["role"]
            header = self._encode(f"<|im_start|>{role}\n")
            ids += header
            labels += [IGNORE_INDEX] * len(header)

            is_response = role == "assistant"
            content = msg["content"]
            parts = content if isinstance(content, list) else [
                {"type": "text", "text": content}
            ]
            for part in parts:
                if part.get("type") == "image":
                    n = soft_counts[img_cursor]
                    img_cursor += 1
                    ids += [self.image_token_id] * n
                    labels += [IGNORE_INDEX] * n
                else:
                    tok = self._encode(part.get("text") or "")
                    ids += tok
                    # Response text is supervised; prompt text is masked.
                    if is_response and self.response_only:
                        labels += tok
                    else:
                        labels += [IGNORE_INDEX] * len(tok)

            trailer = self._encode("<|im_end|>\n")
            ids += trailer
            if is_response and self.response_only:
                labels += trailer  # supervise the end-of-turn marker too
            else:
                labels += [IGNORE_INDEX] * len(trailer)

        if not self.response_only:
            labels = list(ids)

        return ids, labels

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        # 1) process every image in the batch in one call (uniform padding).
        all_images: list[Any] = []
        per_row_counts: list[list[int]] = []

        for row in features:
            imgs = row.get("images") or []
            imgs = [im.convert("RGB") for im in imgs]
            all_images.extend(imgs)

        image_inputs: dict[str, Any] = {}
        soft_flat: list[int] = []
        if all_images:
            image_inputs = self.image_processor(
                images=all_images, return_tensors="pt"
            )
            soft_flat = image_inputs["num_soft_tokens_per_image"].tolist()

        # distribute the flat soft-token counts back to rows in order.
        cursor = 0
        for row in features:
            k = len(row.get("images") or [])
            per_row_counts.append(soft_flat[cursor : cursor + k])
            cursor += k

        # 2) build token sequences and labels.
        batch_ids: list[list[int]] = []
        batch_labels: list[list[int]] = []
        for row, counts in zip(features, per_row_counts):
            ids, labels = self._build_row(row["messages"], counts)
            ids = ids[: self.max_length]
            labels = labels[: self.max_length]
            batch_ids.append(ids)
            batch_labels.append(labels)

        # 3) left-to-right pad to the batch max.
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        max_len = max(len(x) for x in batch_ids)

        input_ids = torch.full((len(batch_ids), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch_ids), max_len), dtype=torch.long)
        labels = torch.full(
            (len(batch_ids), max_len), IGNORE_INDEX, dtype=torch.long
        )
        for i, (ids, lbl) in enumerate(zip(batch_ids, batch_labels)):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
            labels[i, : len(lbl)] = torch.tensor(lbl, dtype=torch.long)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if all_images:
            batch["pixel_values"] = image_inputs["pixel_values"]
            batch["image_position_ids"] = image_inputs["image_position_ids"]
            batch["num_soft_tokens_per_image"] = image_inputs[
                "num_soft_tokens_per_image"
            ]
            if self.emit_blank_images:
                blank_inputs = self.image_processor(
                    images=[blank_like(im) for im in all_images],
                    return_tensors="pt",
                )
                # The token sequence was built from the ALIGNED soft-token counts,
                # so a blank whose counts differ would silently desync the
                # placeholder accounting. Same geometry ⇒ same counts; assert it.
                if not torch.equal(
                    blank_inputs["num_soft_tokens_per_image"],
                    image_inputs["num_soft_tokens_per_image"],
                ):
                    raise ValueError(
                        "blank images produced a different soft-token count than "
                        "the aligned images; the blind branch cannot reuse the "
                        "same input_ids."
                    )
                batch[BLANK_PREFIX + "pixel_values"] = blank_inputs["pixel_values"]
                batch[BLANK_PREFIX + "image_position_ids"] = blank_inputs[
                    "image_position_ids"
                ]
                batch[BLANK_PREFIX + "num_soft_tokens_per_image"] = blank_inputs[
                    "num_soft_tokens_per_image"
                ]
        return batch
