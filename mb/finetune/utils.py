"""Utility functions for the mb_finetune package."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import numpy as np

logger = logging.getLogger("mb.finetune")

__all__ = [
    "set_seed",
    "get_device",
    "count_parameters",
    "print_gpu_memory",
]


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def print_gpu_memory() -> None:
    """Log current GPU memory usage."""
    if not torch.cuda.is_available():
        logger.info("No GPU available.")
        return
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
        logger.info(f"GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

