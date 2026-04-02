"""Multimodal (image + text) dataset for finetuning.

Wraps 'BaseDataset' and resolves image paths before handing
the sample to the model adapter's 'format_input()'.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union
from mb.finetune.data.base import BaseDataset
from pathlib import Path

__all__ = ["MultimodalDataset"]


class MultimodalDataset(BaseDataset):
    """Dataset for image + text input -> text output tasks."""

    def __init__(
        self,
        data_path: str,
        format_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        image_dir: str = "",
        image_column: str = "image",
        text_column: str = "text",
        target_column: str = "output",
        image_size: int = 224,
        split: str = "train",
    ) -> None:
        self.image_dir = image_dir
        self.image_column = image_column
        self.text_column = text_column
        self.target_column = target_column
        self.image_size = image_size
        super().__init__(data_path=data_path, format_fn=format_fn, split=split)

    def _resolve_image(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve the image field to a full path or PIL Image.
        """
        img = sample.get(self.image_column)
        if img is None:
            return sample

        if isinstance(img, str):
            img_path = Path(img)
            if not img_path.is_absolute() and self.image_dir:
                img_path = Path(self.image_dir) / img
            sample = {**sample, self.image_column: str(img_path)}

        return sample

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        sample = self._resolve_image(sample)
        return self.format_fn(sample) if self.format_fn else sample
