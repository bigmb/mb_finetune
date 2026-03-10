"""Qwen-VL / Qwen2.5-VL model adapter for finetuning.

Supports both text-only and multimodal (image + text) inputs.
Outputs text or text + description.

Recommended model IDs:
    - Qwen/Qwen2.5-VL-7B-Instruct   (multimodal)
    - Qwen/Qwen2.5-7B-Instruct       (text-only)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from mb.finetune.config import FinetuneConfig
from mb.finetune.models.base import BaseModelAdapter
from mb.finetune.models.registry import ModelRegistry

__all__ = ["QwenAdapter"]


@ModelRegistry.register("qwen")
class QwenAdapter(BaseModelAdapter):
    """Adapter for the Qwen / Qwen-VL family."""

    # Default model IDs if the user does not specify one.
    _DEFAULT_MULTIMODAL = "Qwen/Qwen2.5-VL-7B-Instruct"
    _DEFAULT_TEXT = "Qwen/Qwen2.5-7B-Instruct"

    def load_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        model_id = self.config.model.model_id
        if not model_id:
            model_id = (
                self._DEFAULT_MULTIMODAL
                if self.config.data.input_type == "multimodal"
                else self._DEFAULT_TEXT
            )

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

        attn_impl = "flash_attention_2" if self.config.model.use_flash_attention else "eager"

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=self.config.model.device_map,
            trust_remote_code=self.config.model.trust_remote_code,
            attn_implementation=attn_impl,
            **quant_kwargs,
        )

        # For VL models, use AutoProcessor; text-only → AutoTokenizer
        if self.config.data.input_type == "multimodal":
            self._processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=self.config.model.trust_remote_code
            )
            self._tokenizer = self._processor.tokenizer
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=self.config.model.trust_remote_code
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
        """Tokenize / process a single sample.

        For multimodal inputs the sample is expected to contain an image
        (PIL Image or path) and text.  For text-only, just the text.
        """
        cfg = self.config.data
        text = sample.get(cfg.text_column, "")
        target = sample.get(cfg.target_column, "")

        # Build the target string
        if cfg.output_type == "text_description":
            description = sample.get(cfg.description_column, "")
            full_target = f"{target}\n{description}" if description else target
        else:
            full_target = target

        # Apply prompt template if provided
        if cfg.prompt_template:
            prompt = cfg.prompt_template.format(text=text)
        else:
            prompt = text

        full_text = f"{prompt}\n{full_target}"

        if self.config.data.input_type == "multimodal" and self._processor is not None:
            from PIL import Image

            image = sample.get(cfg.image_column)
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            proc_text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(
                text=proc_text + full_target,
                images=image,
                return_tensors="pt",
                padding="max_length",
                max_length=cfg.max_length,
                truncation=True,
            )
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs["labels"] = inputs["input_ids"].clone()
            return inputs

        # Text-only path
        encodings = self._tokenizer(
            full_text,
            max_length=cfg.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encodings = {k: v.squeeze(0) for k, v in encodings.items()}
        encodings["labels"] = encodings["input_ids"].clone()
        return encodings
