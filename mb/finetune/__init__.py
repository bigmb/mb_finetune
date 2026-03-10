"""mb_finetune – Multi-model finetuning package.

Supports finetuning vision-language and text models (Qwen, BLIP, CLIP, etc.)
using HuggingFace Transformers.

Input:  image + text  OR  text-only
Output: text  OR  text + description
"""

from mb.finetune.version import version as __version__
from mb.finetune.config import FinetuneConfig
from mb.finetune.trainer import FinetuneTrainer
from mb.finetune.models.registry import ModelRegistry

__all__ = [
    "__version__",
    "FinetuneConfig",
    "FinetuneTrainer",
    "ModelRegistry",
]
