# mb_finetune

Multi-model finetuning package using HuggingFace Transformers.

## Supported Models

| Model | Input | Output | Model ID (default) |
|-------|-------|--------|---------------------|
| **Qwen-VL** | image + text | text / text+description | `Qwen/Qwen2.5-VL-7B-Instruct` |
| **Qwen** | text | text / text+description | `Qwen/Qwen2.5-7B-Instruct` |
| **BLIP-2** | image + text | text / text+description | `Salesforce/blip2-opt-2.7b` |
| **CLIP** | image + text | text / text+description | `openai/clip-vit-base-patch32` |

## Installation

```bash
pip install -e .
```

## Quick Start

### From YAML config
```bash
python -m mb.finetune.run --config configs/example_multimodal.yaml
```

### From Python
```python
from mb.finetune import FinetuneConfig, FinetuneTrainer

config = FinetuneConfig.from_yaml("configs/example_text.yaml")
trainer = FinetuneTrainer(config)
trainer.train()
trainer.save()
```

### Programmatic config
```python
from mb.finetune.config import FinetuneConfig, ModelConfig, DataConfig, TrainConfig

config = FinetuneConfig(
    model=ModelConfig(model_name="qwen", model_id="Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True),
    data=DataConfig(input_type="text", train_path="data/train.json"),
    train=TrainConfig(num_epochs=3, learning_rate=2e-5),
)

trainer = FinetuneTrainer(config)
trainer.train()
```

## Data Format

Training data should be JSON/JSONL/CSV/Parquet with columns matching the config:

**Text-only:**
```json
[
  {"instruction": "Summarize this.", "response": "Summary here."},
  {"instruction": "Translate to French.", "response": "Traduction ici."}
]
```

**Multimodal (image + text):**
```json
[
  {"image": "img_001.jpg", "question": "What is in this image?", "answer": "A cat.", "description": "A fluffy orange cat."},
  {"image": "img_002.jpg", "question": "Describe the scene.", "answer": "A park.", "description": "A sunny park with trees."}
]
```

## Package Structure

```
mb/finetune/
├── __init__.py          # Package entry point
├── config.py            # Configuration dataclasses (YAML / dict)
├── trainer.py           # Main FinetuneTrainer engine
├── run.py               # CLI entry point
├── utils.py             # Utility functions
├── version.py           # Version info
├── models/
│   ├── base.py          # Abstract BaseModelAdapter
│   ├── registry.py      # ModelRegistry (maps names → adapters)
│   ├── qwen.py          # Qwen / Qwen-VL adapter
│   ├── blip.py          # BLIP-2 adapter
│   └── clip.py          # CLIP + text-head adapter
├── data/
│   ├── base.py          # BaseDataset (JSON/CSV/Parquet loader)
│   ├── text_dataset.py  # Text-only dataset
│   ├── multimodal_dataset.py  # Image+text dataset
│   └── collator.py      # SmartCollator
└── callbacks/
    ├── logging.py       # Training logging callback
    └── checkpoint.py    # Checkpoint management callback
```

## Adding a New Model

```python
from mb.finetune.models.base import BaseModelAdapter
from mb.finetune.models.registry import ModelRegistry

@ModelRegistry.register("my_model")
class MyModelAdapter(BaseModelAdapter):
    def load_model(self):
        ...
    def prepare_for_training(self):
        ...
    def format_input(self, sample):
        ...
```
