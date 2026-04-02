"""Text-only dataset for finetuning.

Wraps 'BaseDataset' with text-specific defaults and validation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional
from mb.finetune.data.base import BaseDataset

__all__ = ["TextDataset"]


class TextDataset(BaseDataset):
    """
    Dataset for text-only input → text output tasks.
    """

    def __init__(
        self,
        data_path: str,
        format_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        text_column: str = "text",
        target_column: str = "output",
        split: str = "train",
    ) -> None:
        self.text_column = text_column
        self.target_column = target_column
        super().__init__(data_path=data_path, format_fn=format_fn, split=split)

    def _validate_sample(self, sample: Dict[str, Any]) -> None:
        if self.text_column not in sample:
            raise KeyError(
                f"Text column '{self.text_column}' not found in sample. "
                f"Available keys: {list(sample.keys())}"
            )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        self._validate_sample(sample)
        return self.format_fn(sample) if self.format_fn else sample
