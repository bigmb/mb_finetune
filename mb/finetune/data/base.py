"""Base dataset class for mb_finetune.

Handles loading data from JSON, CSV, Parquet, and JSONL files.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union
from mb.pandas.dfload import load_any_df
from torch.utils.data import Dataset

__all__ = ["BaseDataset"]


class BaseDataset(Dataset):
    """
    Base dataset that loads structured data and delegates formatting to
    a model adapter's ``format_input()`` method.
    """

    def __init__(
        self,
        data_path: str,
        format_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        split: str = "train",
    ) -> None:
        self.data_path = data_path
        self.format_fn = format_fn
        self.split = split
        self.samples: List[Dict[str, Any]] = []
        self._load_data()
        
    def _load_data(self) -> None:
        """
        Loading data using mb.pandas.dfload
        """
        df = load_any_df(self.data_path)
        self.samples = df.to_dict(orient="records")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return self.format_fn(sample) if self.format_fn else sample
