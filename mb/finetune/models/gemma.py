
from mb.finetune.config import FinetuneConfig
from mb.finetune.models.base import ModelBaseAdapter
from mb.finetune.models.registry import ModelRegistry

__all__ = ["GEMMAAdapter"]


class GEMMAWithTextHead(nn.Module):
    """Thin wrapper: GEMMA encoder + a small text-generation head.

    Used to turn GEMMA into a model that can produce text output during
    finetuning (e.g. captioning). The head is a small transformer decoder
    on top of the GEMMA text/image embeddings.
    """

    def __init__(self, gemma_model: GEMMAModel, vocab_size: int, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.gemma = gemma_model
        proj_dim = gemma_model.config.projection_dim

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.embed_proj = nn.Linear(proj_dim, hidden_dim)
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

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
            image_embeds = self.gemma.get_image_features(pixel_values=pixel_values)
            memory = self.embed_proj(image_embeds).unsqueeze(1)  # (B, 1, H)
        else:
            text_embeds = self.gemma.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
            memory = self.embed_proj(text_embeds).unsqueeze(1)

        if labels is not None:
            tgt = self.token_embed(labels)
            decoded = self.decoder(tgt, memory)
            logits = self.output_proj(decoded)

            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {"loss": loss, "logits": logits}

        return {"logits": memory}


@ModelRegistry.register("gemma")
class GEMMAAdapter(ModelBaseAdapter):
    """Adapter for Gemma models + optional text generation head."""

    _DEFAULT_MODEL = "huggingface/gemma-270M"
