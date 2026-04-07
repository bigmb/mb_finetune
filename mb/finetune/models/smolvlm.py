"""SmolVLM2 model adapter for finetuning.

Supports multimodal (image + text -> text) finetuning using
HuggingFace SmolVLM2 models (Idefics3-based architecture).

Recommended model IDs:
    - HuggingFaceTB/SmolVLM2-256M-Video-Instruct
    - HuggingFaceTB/SmolVLM2-500M-Video-Instruct
    - HuggingFaceTB/SmolVLM2-2.2B-Instruct
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from mb.finetune.config import FinetuneConfig
from mb.finetune.models.base import ModelBaseAdapter
from mb.finetune.models.registry import ModelRegistry

__all__ = ["SmolVLMAdapter"]


@ModelRegistry.register("smolvlm")
class SmolVLMAdapter(ModelBaseAdapter):
    """Adapter for HuggingFace SmolVLM2 vision-language models."""

    _DEFAULT_MODEL = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

    def load_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        model_id = self.config.model.model_id or self._DEFAULT_MODEL
        dtype = self._resolve_dtype(default=torch.bfloat16)

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

        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=self.config.model.device_map,
            trust_remote_code=self.config.model.trust_remote_code,
            _attn_implementation=attn_impl,
            **quant_kwargs,
        )

        self._processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        self._tokenizer = self._processor.tokenizer

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        return self._model, self._tokenizer

    def prepare_for_training(self) -> PreTrainedModel:
        model = self.model

        # Freeze vision encoder, train language model
        if hasattr(model, "model") and hasattr(model.model, "vision_model"):
            for param in model.model.vision_model.parameters():
                param.requires_grad = False

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

        # Build prompt from template or raw text
        if cfg.prompt_template:
            template_vars = {k: str(v) if v is not None else "" for k, v in sample.items()}
            template_vars["text"] = text
            prompt_text = cfg.prompt_template.format_map(template_vars)
        else:
            prompt_text = text

        # Build message content list
        content = []

        if cfg.input_type == "multimodal":
            image_val = sample.get(cfg.image_column)
            if isinstance(image_val, str) and image_val.strip():
                image_path = Path(image_val.strip())
                if image_path.exists():
                    content.append({"type": "image", "path": str(image_path)})

        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]

        # Use chat template to produce input_ids + pixel_values etc.
        proc_text = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        full_text = proc_text + full_target

        # Load image for processor if multimodal
        images = None
        if cfg.input_type == "multimodal":
            image_val = sample.get(cfg.image_column)
            if isinstance(image_val, str) and image_val.strip():
                image_path = Path(image_val.strip())
                if image_path.exists():
                    from PIL import Image

                    images = [Image.open(image_path).convert("RGB")]

        inputs = self._processor(
            text=full_text,
            images=images,
            return_tensors="pt",
            padding="max_length",
            max_length=cfg.max_length,
            truncation=True,
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Build labels: mask everything before the target
        labels = inputs["input_ids"].clone()
        labels[labels == self._tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return inputs
