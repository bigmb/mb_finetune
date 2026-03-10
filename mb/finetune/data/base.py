"""Base dataset class for mb_finetune.

Handles loading data from JSON, CSV, Parquet, and JSONL files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from mb.pandas.dfload import load_any_df
from torch.utils.data import Dataset

__all__ = ["BaseDataset"]


class BaseDataset(Dataset):
    """Base dataset that loads structured data and delegates formatting to
    a model adapter's ``format_input()`` method.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        format_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
        split: str = "train",
    ) -> None:
        self.data_path = Path(data_path)
        self.format_fn = format_fn
        self.split = split
        self.samples: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load data from disk based on file extension."""
        suffix = self.data_path.suffix.lower()

        if suffix in {".json", ".jsonl"}:
            self.samples = self._load_json(self.data_path)
        elif suffix in {".csv", ".parquet"}:
            self.samples = load_any_df(self.data_path).to_dict(orient="records")
        elif self.data_path.is_dir():
            # Try loading a file named {split}.json / {split}.jsonl inside the directory
            for ext in [".json", ".jsonl", ".csv", ".parquet"]:
                candidate = self.data_path / f"{self.split}{ext}"
                if candidate.exists():
                    return self._load_from_file(candidate)
            raise FileNotFoundError(
                f"No data file found for split '{self.split}' in {self.data_path}"
            )
        else:
            raise ValueError(f"Unsupported data format: {suffix}")

    def _load_from_file(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            self.samples = self._load_json(path)
        elif suffix in {".csv", ".parquet"}:
            self.samples = load_any_df(path).to_dict(orient="records")

    @staticmethod
    def _load_json(path: Path) -> List[Dict[str, Any]]:
        text = path.read_text(encoding="utf-8")
        # Try JSON Lines first
        lines = text.strip().splitlines()
        if len(lines) > 1:
            try:
                return [json.loads(line) for line in lines if line.strip()]
            except json.JSONDecodeError:
                pass
        # Regular JSON
        data = json.loads(text)
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected a JSON array or JSONL file, got {type(data).__name__}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return self.format_fn(sample)
