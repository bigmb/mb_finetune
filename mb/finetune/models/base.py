"""
Base class for model adapters.

Every model adapter (Qwen, BLIP, CLIP, …) must subclass 'ModelBaseAdapter'
and implement the required methods so the 'FinetuneTrainer' can work with
any supported model in a uniform way.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin

from mb.finetune.config import FinetuneConfig

__all__ = ["ModelBaseAdapter"]


class ModelBaseAdapter(abc.ABC):
    """Uniform interface around a HuggingFace model + processor/tokenizer."""

    def __init__(self, config: FinetuneConfig) -> None:
        self.config = config
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._processor: Optional[ProcessorMixin] = None

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            raise RuntimeError("Model not loaded call load_model() first.")
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded call load_model() first.")
        return self._tokenizer

    @property
    def processor(self) -> Optional[ProcessorMixin]:
        return self._processor

    ## all models to have their own implementation of these methods, since loading and input formatting
    ## can be very different across model types (e.g. CLIP needs special handling for images)
    ## (loading model , preparing for training, formatting input samples)
    @abc.abstractmethod
    def load_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """
        Load the model + tokenizer (and optional processor) from HuggingFace.

        Must set 'self._model', 'self._tokenizer', and optionally
        'self._processor'.

        Returns the '(model, tokenizer)' tuple.
        """
        ...

    @abc.abstractmethod
    def prepare_for_training(self) -> PreTrainedModel:
        """
        Apply LoRA / quantisation / gradient-checkpointing and return the
        model ready for finetuning.
        """
        ...

    @abc.abstractmethod
    def format_input(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a raw dataset sample into the tokenized/processed tensor
        dict expected by the model's 'forward()' method.
        """
        ...
        
    def _resolve_dtype(self, default=torch.float32) -> torch.dtype:
        """
        Map the string dtype from config to a 'torch.dtype'.
        """
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(self.config.model.torch_dtype, default)

    def _apply_lora(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        Wrap the model with PEFT / LoRA if enabled in config.
        """
        lora = self.config.model.lora
        if lora is None or not lora.enabled:
            return model

        from peft import LoraConfig, get_peft_model, TaskType

        task_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
            "SEQ_CLS": TaskType.SEQ_CLS,
            "FEATURE_EXTRACTION": TaskType.FEATURE_EXTRACTION,
        }

        peft_config = LoraConfig(
            r=lora.r,
            lora_alpha=lora.lora_alpha,
            lora_dropout=lora.lora_dropout,
            target_modules=lora.target_modules,
            bias=lora.bias,
            task_type=task_map.get(lora.task_type, TaskType.CAUSAL_LM),
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model

    def _enable_gradient_checkpointing(self, model: PreTrainedModel) -> None:
        """
        Enable gradient checkpointing if configured.
        """
        if self.config.train.gradient_checkpointing:
            model.gradient_checkpointing_enable()

    def save_model(self, output_dir: str) -> None:
        """
        Save model + tokenizer to disk.
        """
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        if self._processor is not None:
            self._processor.save_pretrained(output_dir)
