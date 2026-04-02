"""BLIP-2 model adapter for finetuning.

Supports multimodal (image + text → text) finetuning using
Salesforce's BLIP-2 architecture via HuggingFace Transformers.

Recommended model IDs:
    - Salesforce/blip2-opt-2.7b
    - Salesforce/blip2-opt-6.7b
    - Salesforce/blip2-flan-t5-xl
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from mb.finetune.config import FinetuneConfig
from mb.finetune.models.base import ModelBaseAdapter
from mb.finetune.models.registry import ModelRegistry

__all__ = ["BlipAdapter"]


@ModelRegistry.register("blip")
class BlipAdapter(ModelBaseAdapter):
    """Adapter for BLIP-2 image-captioning / VQA models."""

    _DEFAULT_MODEL = "Salesforce/blip2-opt-2.7b"

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

        self._model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=self.config.model.device_map,
            trust_remote_code=self.config.model.trust_remote_code,
            **quant_kwargs,
        )

        self._processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=self.config.model.trust_remote_code
        )
        self._tokenizer = self._processor.tokenizer

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        return self._model, self._tokenizer

    def prepare_for_training(self) -> PreTrainedModel:
        model = self.model

        # Freeze the vision encoder – we only train the language model + Q-Former.
        for param in model.vision_model.parameters():
            param.requires_grad = False

        self._enable_gradient_checkpointing(model)
        model = self._apply_lora(model)
        self._model = model
        return model

    def format_input(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config.data
        text = sample.get(cfg.text_column, "")
        target = sample.get(cfg.target_column, "")

        if cfg.output_type == "text_description":
            description = sample.get(cfg.description_column, "")
            full_target = f"{target}\n{description}" if description else target
        else:
            full_target = target

        prompt = cfg.prompt_template.format(text=text) if cfg.prompt_template else text

        from PIL import Image

        image = sample.get(cfg.image_column)
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        inputs = self._processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=cfg.max_length,
            truncation=True,
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Build labels from the target text
        label_encoding = self._tokenizer(
            full_target,
            max_length=cfg.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs["labels"] = label_encoding["input_ids"].squeeze(0)

        return inputs
