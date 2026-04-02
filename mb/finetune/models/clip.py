"""CLIP model adapter for finetuning.

CLIP is primarily a contrastive model. This adapter supports finetuning CLIP
for image-text matching / embedding alignment. Since CLIP doesn't natively
generate text, this adapter is best used when the task is contrastive
learning or feature-extraction based finetuning, followed by a lightweight
text decoder head.

Recommended model IDs:
    - openai/clip-vit-base-patch32
    - openai/clip-vit-large-patch14
    - openai/clip-vit-large-patch14-336
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoProcessor,
    CLIPModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from mb.finetune.config import FinetuneConfig
from mb.finetune.models.base import ModelBaseAdapter
from mb.finetune.models.registry import ModelRegistry

__all__ = ["CLIPAdapter"]


class CLIPWithTextHead(nn.Module):
    """Thin wrapper: CLIP encoder + a small text-generation head.

    Used to turn CLIP into a model that can produce text output during
    finetuning (e.g. captioning). The head is a small transformer decoder
    on top of the CLIP text/image embeddings.
    """

    def __init__(self, clip_model: CLIPModel, vocab_size: int, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.clip = clip_model
        proj_dim = clip_model.config.projection_dim

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.embed_proj = nn.Linear(proj_dim, hidden_dim)
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.clip, "gradient_checkpointing_enable"):
            return self.clip.gradient_checkpointing_enable(**kwargs)
        return None

    def gradient_checkpointing_disable(self):
        if hasattr(self.clip, "gradient_checkpointing_disable"):
            return self.clip.gradient_checkpointing_disable()
        return None

    @staticmethod
    def _extract_tensor(embeds):
        if torch.is_tensor(embeds):
            return embeds

        if hasattr(embeds, "pooler_output") and embeds.pooler_output is not None:
            return embeds.pooler_output

        if hasattr(embeds, "last_hidden_state") and embeds.last_hidden_state is not None:
            return embeds.last_hidden_state[:, 0]

        raise TypeError(f"Unsupported embedding output type: {type(embeds)!r}")

    def forward(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        # Encode image if present
        if pixel_values is not None:
            image_embeds = self.clip.get_image_features(pixel_values=pixel_values)
            image_embeds = self._extract_tensor(image_embeds)
            memory = self.embed_proj(image_embeds).unsqueeze(1)  # (B, 1, H)
        else:
            text_embeds = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = self._extract_tensor(text_embeds)
            memory = self.embed_proj(text_embeds).unsqueeze(1)

        if labels is not None:
            tgt = self.token_embed(labels)
            decoded = self.decoder(tgt, memory)
            logits = self.output_proj(decoded)

            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {"loss": loss, "logits": logits}

        return {"logits": memory}


@ModelRegistry.register("clip")
class CLIPAdapter(ModelBaseAdapter):
    """Adapter for OpenAI CLIP models + optional text generation head."""

    _DEFAULT_MODEL = "openai/clip-vit-base-patch32"

    def load_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        model_id = self.config.model.model_id or self._DEFAULT_MODEL
        dtype = self._resolve_dtype()

        clip_model = CLIPModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=self.config.model.device_map,
            trust_remote_code=self.config.model.trust_remote_code,
        )

        self._processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=self.config.model.trust_remote_code
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=self.config.model.trust_remote_code
        )

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Wrap CLIP with a text generation head for text output
        self._model = CLIPWithTextHead(
            clip_model,
            vocab_size=len(self._tokenizer),
        )

        return self._model, self._tokenizer

    def prepare_for_training(self) -> PreTrainedModel:
        model = self._model

        # Freeze CLIP vision/text encoders, only train the decoder head
        for param in model.clip.parameters():
            param.requires_grad = False

        # Optionally unfreeze some CLIP layers for finetuning
        if self.config.model.lora and self.config.model.lora.enabled:
            # For CLIP we won't apply LoRA (custom head is small enough),
            # but we can unfreeze the last few layers of the text encoder.
            for param in model.clip.text_model.encoder.layers[-2:].parameters():
                param.requires_grad = True

        return model

    @staticmethod
    def _to_text(value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, float) and value != value:
            return ""

        if isinstance(value, list):
            return ", ".join(str(item) for item in value)

        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return ""

            if stripped.startswith("[") or stripped.startswith("{"):
                try:
                    parsed = json.loads(stripped)
                    return CLIPAdapter._to_text(parsed)
                except (json.JSONDecodeError, TypeError):
                    return stripped

            return stripped

        return str(value)

    def format_input(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.config.data
        text = self._to_text(sample.get(cfg.text_column, ""))
        target = self._to_text(sample.get(cfg.target_column, ""))

        clip_text_limit = getattr(self._tokenizer, "model_max_length", cfg.max_length)
        clip_cfg = getattr(getattr(self._model, "clip", None), "config", None)
        if clip_cfg is not None:
            text_cfg = getattr(clip_cfg, "text_config", None)
            if text_cfg is not None and hasattr(text_cfg, "max_position_embeddings"):
                clip_text_limit = min(clip_text_limit, int(text_cfg.max_position_embeddings))

        text_max_length = min(cfg.max_length, int(clip_text_limit))

        if cfg.output_type == "text_description":
            description = self._to_text(sample.get(cfg.description_column, ""))
            full_target = f"{target}\n{description}" if description else target
        else:
            full_target = target

        result: Dict[str, Any] = {}

        if cfg.input_type == "multimodal":
            from PIL import Image

            image = sample.get(cfg.image_column)
            if isinstance(image, float) and image != image:
                image = None

            if isinstance(image, str):
                image = image.strip()
                if image:
                    image_path = Path(image)
                    image = Image.open(image_path).convert("RGB") if image_path.exists() else None
                else:
                    image = None

            if image is not None:
                img_inputs = self._processor(
                    images=image,
                    return_tensors="pt",
                )
                result["pixel_values"] = img_inputs["pixel_values"].squeeze(0)

        # Always tokenize the text
        text_inputs = self._tokenizer(
            text,
            max_length=text_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        result["input_ids"] = text_inputs["input_ids"].squeeze(0)
        result["attention_mask"] = text_inputs["attention_mask"].squeeze(0)

        # Labels
        label_inputs = self._tokenizer(
            full_target,
            max_length=cfg.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        result["labels"] = label_inputs["input_ids"].squeeze(0)

        return result
