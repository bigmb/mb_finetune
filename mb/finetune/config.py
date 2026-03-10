"""Configuration dataclasses for mb_finetune.

Supports loading from YAML files or constructing programmatically.

Example:
    from mb.finetune.config import FinetuneConfig
    cfg = FinetuneConfig.from_yaml("config.yaml")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = [
    "ModelConfig",
    "DataConfig",
    "TrainConfig",
    "OutputConfig",
    "FinetuneConfig",
]

PathLike = Union[str, Path]


@dataclass
class ModelConfig:
    """Model-related configuration."""

    model_name: str = "qwen"                      # key in ModelRegistry
    model_id: str = ""                              # HuggingFace model id / path
    task_type: str = "text_generation"              # text_generation | image_text_to_text | feature_extraction
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    use_flash_attention: bool = True
    torch_dtype: str = "bfloat16"                   # float16 | bfloat16 | float32
    device_map: str = "auto"
    trust_remote_code: bool = True
    lora: Optional[LoRAConfig] = None

    def __post_init__(self):
        if self.lora is None:
            self.lora = LoRAConfig()


@dataclass
class LoRAConfig:
    """LoRA / PEFT configuration."""

    enabled: bool = True
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class DataConfig:
    """Dataset-related configuration."""

    input_type: str = "text"                        # text | multimodal
    output_type: str = "text"                       # text | text_description
    train_path: str = ""                            # path to training data (json / csv / parquet / folder)
    val_path: str = ""                              # path to validation data
    image_dir: str = ""                             # base directory for images (multimodal)
    text_column: str = "text"
    image_column: str = "image"
    target_column: str = "output"
    description_column: str = "description"         # used when output_type == text_description
    max_length: int = 512
    image_size: int = 224
    prompt_template: str = ""                       # optional prompt template with {text} / {image} placeholders


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    seed: int = 42
    dataloader_num_workers: int = 4
    optim: str = "adamw_torch"
    deepspeed: Optional[str] = None                 # path to deepspeed config json


@dataclass
class OutputConfig:
    """Output / checkpoint configuration."""

    output_dir: str = "./output"
    save_total_limit: int = 3
    push_to_hub: bool = False
    hub_model_id: str = ""
    logging_dir: str = "./logs"
    report_to: str = "tensorboard"                  # tensorboard | wandb | none
    resume_from_checkpoint: Optional[str] = None


@dataclass
class FinetuneConfig:
    """Top-level configuration that bundles all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: PathLike, encoding: str = "utf-8") -> "FinetuneConfig":
        """Load a ``FinetuneConfig`` from a YAML file."""
        import yaml  # type: ignore

        raw = Path(path).read_text(encoding=encoding)
        data: Dict[str, Any] = yaml.safe_load(raw) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FinetuneConfig":
        """Construct from a plain dictionary (e.g. parsed YAML / JSON)."""

        lora_data = data.get("model", {}).pop("lora", None)
        lora_cfg = LoRAConfig(**lora_data) if isinstance(lora_data, dict) else LoRAConfig()

        model_cfg = ModelConfig(**data.get("model", {}))
        model_cfg.lora = lora_cfg

        return cls(
            model=model_cfg,
            data=DataConfig(**data.get("data", {})),
            train=TrainConfig(**data.get("train", {})),
            output=OutputConfig(**data.get("output", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise back to a plain dict."""
        from dataclasses import asdict
        return asdict(self)
