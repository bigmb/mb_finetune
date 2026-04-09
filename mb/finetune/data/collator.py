"""Smart data collator for mb_finetune.

Handles mixed tensor shapes (e.g. pixel_values + input_ids) and pads
appropriately for different model architectures.
"""

from __future__ import annotations

from typing import Any, Dict, List
import torch

__all__ = ["SmartCollator"]


class SmartCollator:
    """
    Collator that stacks tensors and pads where necessary.

    Works transparently with both text-only and multimodal batches.
    """

    def __init__(self, pad_token_id: int = 0, label_pad_token_id: int = -100):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not features:
            return {}

        batch: Dict[str, Any] = {}
        # Intersect keys across all samples so a field absent from any one
        # sample (e.g. pixel_values on a text-only row) doesn't cause a KeyError.
        keys = set(features[0].keys())
        for f in features[1:]:
            keys &= set(f.keys())

        for key in keys:
            values = [f[key] for f in features]

            if isinstance(values[0], torch.Tensor):
                # If all shapes match, stack directly
                if all(v.shape == values[0].shape for v in values):
                    batch[key] = torch.stack(values)
                else:
                    # Pad to max length in the last dimension
                    max_len = max(v.shape[-1] for v in values)
                    pad_id = self.label_pad_token_id if key == "labels" else self.pad_token_id

                    padded = []
                    for v in values:
                        pad_size = max_len - v.shape[-1]
                        if pad_size > 0:
                            padding = torch.full((*v.shape[:-1], pad_size), pad_id, dtype=v.dtype)
                            v = torch.cat([v, padding], dim=-1)
                        padded.append(v)
                    batch[key] = torch.stack(padded)
            else:
                batch[key] = values

        return batch
