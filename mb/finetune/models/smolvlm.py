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

        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": full_target}]},
        ]

        # Use apply_chat_template with tokenize=True so image token
        # expansion and tokenization happen in one pass (avoids the
        # mismatch between text-level image placeholders and token-level
        # image tokens that truncation would cause).
        # Images are loaded automatically from the "path" key in messages.
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            processor_kwargs={"return_tensors": "pt"},
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Truncate or pad to max_length after image tokens are resolved
        max_len = cfg.max_length
        seq_len = inputs["input_ids"].shape[0]

        if seq_len > max_len:
            for k in inputs:
                if isinstance(inputs[k], torch.Tensor) and inputs[k].dim() >= 1 and inputs[k].shape[0] == seq_len:
                    inputs[k] = inputs[k][:max_len]
        elif seq_len < max_len:
            pad_len = max_len - seq_len
            pad_id = self._tokenizer.pad_token_id or 0
            for k in inputs:
                if isinstance(inputs[k], torch.Tensor) and inputs[k].dim() >= 1 and inputs[k].shape[0] == seq_len:
                    pad_val = -100 if k == "labels" else pad_id if "ids" in k else 0
                    padding = torch.full((pad_len,), pad_val, dtype=inputs[k].dtype)
                    inputs[k] = torch.cat([inputs[k], padding])

        # Build labels: mask pad tokens
        labels = inputs["input_ids"].clone()
        labels[labels == self._tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return inputs
