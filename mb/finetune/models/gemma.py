"""Gemma model adapter for finetuning.

Supports text generation finetuning with HuggingFace Gemma models.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from mb.finetune.models.base import ModelBaseAdapter
from mb.finetune.models.registry import ModelRegistry

__all__ = ["GemmaAdapter"]


@ModelRegistry.register("gemma")
class GemmaAdapter(ModelBaseAdapter):
    """
    Adapter for Gemma text-generation models.
    """

    _DEFAULT_MODEL = "google/gemma-2b"

    def load_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        model_id = self.config.model.model_id or self._DEFAULT_MODEL
        dtype = self._resolve_dtype()

        quant_kwargs: Dict[str, Any] = {}
        if self.config.model.load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.model.load_in_8bit:
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=self.config.model.device_map,
            trust_remote_code=self.config.model.trust_remote_code,
            **quant_kwargs,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=self.config.model.trust_remote_code,
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        return self._model, self._tokenizer

    def prepare_for_training(self) -> PreTrainedModel:
        model = self.model
        self._enable_gradient_checkpointing(model)
        model = self._apply_lora(model)
        self._model = model
        return model

    def format_input(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config.data
        text = str(sample.get(cfg.text_column, "") or "")
        target = str(sample.get(cfg.target_column, "") or "")

        if cfg.output_type == "text_description":
            description = str(sample.get(cfg.description_column, "") or "")
            full_target = f"{target}\n{description}" if description else target
        else:
            full_target = target

        prompt = cfg.prompt_template.format(text=text) if cfg.prompt_template else text
        full_text = f"{prompt}\n{full_target}" if full_target else prompt

        encodings = self._tokenizer(
            full_text,
            max_length=cfg.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        inputs = {k: v.squeeze(0) for k, v in encodings.items()}
        labels = inputs["input_ids"].clone()
        labels[labels == self._tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs
