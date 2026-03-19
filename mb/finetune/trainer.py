"""FinetuneTrainer – main training engine for mb_finetune.

Orchestrates model loading, dataset creation, and HuggingFace Trainer
execution.  Supports text-only and multimodal finetuning.

Usage:
    from mb.finetune import FinetuneConfig, FinetuneTrainer

    config = FinetuneConfig.from_yaml("config.yaml")
    trainer = FinetuneTrainer(config)
    trainer.train()
    trainer.save()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from transformers import Trainer, TrainingArguments
from mb.finetune.config import FinetuneConfig
from mb.finetune.data.collator import SmartCollator
from mb.finetune.data.multimodal_dataset import MultimodalDataset
from mb.finetune.data.text_dataset import TextDataset
from mb.finetune.models.registry import ModelRegistry
from mb.finetune.models.base import BaseModelAdapter
from mb.finetune.callbacks.logging import LoggingCallback
from mb.finetune.callbacks.checkpoint import CheckpointCallback
from mb.utils.logging import logg

__all__ = ["FinetuneTrainer"]

class FinetuneTrainer:
    """High-level finetuning orchestrator.

    Parameters
    ----------
    config : FinetuneConfig
        Full configuration (model, data, training, output).
    """

    def __init__(self, config: FinetuneConfig, logger=None) -> None:
        self.config = config
        self.adapter: Optional[BaseModelAdapter] = None
        self.hf_trainer: Optional[Trainer] = None
        self.logger = logger 

    def train(self) -> None:
        """
        Run the full finetuning pipeline: load → prepare → train.
        """
        logg.info("Initialising finetuning pipeline",logger=self.logger)

        # 1. Load model adapter
        self.adapter = self._load_adapter()

        # 2. Load model + tokenizer
        logg.info(f"Loading model '{self.config.model.model_name}'", logger=self.logger)
        model, tokenizer = self.adapter.load_model()
        model = self.adapter.prepare_for_training()

        # 3. Build datasets
        train_dataset = self._build_dataset("train")
        eval_dataset = self._build_dataset("eval") if self.config.data.val_path else None

        # 4. Data collator
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        collator = SmartCollator(pad_token_id=pad_id)

        # 5. Training arguments
        training_args = self._build_training_args()

        # 6. Build HF Trainer
        self.hf_trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            callbacks=[LoggingCallback(), CheckpointCallback()],
        )

        # 7. Train!
        logg.info("Starting training", logger=self.logger)
        resume = self.config.output.resume_from_checkpoint
        self.hf_trainer.train(resume_from_checkpoint=resume)

        logg.info("Training complete.", logger=self.logger)

    def save(self, output_dir: Optional[str] = None) -> None:
        """
        Save the finetuned model + tokenizer to disk.
        """
        save_dir = output_dir or self.config.output.output_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if self.adapter is not None:
            self.adapter.save_model(save_dir)
            logg.info(f"Model saved to {save_dir}", logger=self.logger)
        elif self.hf_trainer is not None:
            self.hf_trainer.save_model(save_dir)
            logg.info(f"Model saved to {save_dir}", logger=self.logger)

    def evaluate(self):
        """
        Run evaluation and return metrics.
        """
        if self.hf_trainer is None:
            raise RuntimeError("Trainer not initialised – call train() first.")
        return self.hf_trainer.evaluate()

    def _load_adapter(self) -> BaseModelAdapter:
        """
        Instantiate the correct model adapter from the registry.
        """
        adapter_cls = ModelRegistry.get(self.config.model.model_name)
        return adapter_cls(self.config)

    def _build_dataset(self, split: str):
        """
        Build a train or eval dataset based on config.
        """
        data_cfg = self.config.data
        data_path = data_cfg.train_path if split == "train" else data_cfg.val_path

        if not data_path:
            return None

        format_fn = self.adapter.format_input

        if data_cfg.input_type == "multimodal":
            return MultimodalDataset(
                data_path=data_path,
                format_fn=format_fn,
                image_dir=data_cfg.image_dir,
                image_column=data_cfg.image_column,
                text_column=data_cfg.text_column,
                target_column=data_cfg.target_column,
                image_size=data_cfg.image_size,
                split=split,
            )
        else:
            return TextDataset(
                data_path=data_path,
                format_fn=format_fn,
                text_column=data_cfg.text_column,
                target_column=data_cfg.target_column,
                split=split,
            )

    def _build_training_args(self) -> TrainingArguments:
        """
        Map ``TrainConfig`` + ``OutputConfig`` → HuggingFace ``TrainingArguments``.
        """
        t = self.config.train
        o = self.config.output

        return TrainingArguments(
            output_dir=o.output_dir,
            num_train_epochs=t.num_epochs,
            per_device_train_batch_size=t.per_device_train_batch_size,
            per_device_eval_batch_size=t.per_device_eval_batch_size,
            gradient_accumulation_steps=t.gradient_accumulation_steps,
            learning_rate=t.learning_rate,
            weight_decay=t.weight_decay,
            warmup_ratio=t.warmup_ratio,
            lr_scheduler_type=t.lr_scheduler_type,
            max_grad_norm=t.max_grad_norm,
            fp16=t.fp16,
            bf16=t.bf16,
            gradient_checkpointing=t.gradient_checkpointing,
            save_steps=t.save_steps,
            eval_steps=t.eval_steps if self.config.data.val_path else None,
            eval_strategy="steps" if self.config.data.val_path else "no",
            logging_steps=t.logging_steps,
            logging_dir=o.logging_dir,
            seed=t.seed,
            dataloader_num_workers=t.dataloader_num_workers,
            save_total_limit=o.save_total_limit,
            report_to=o.report_to,
            push_to_hub=o.push_to_hub,
            hub_model_id=o.hub_model_id if o.push_to_hub else None,
            optim=t.optim,
            deepspeed=t.deepspeed,
            remove_unused_columns=False,  # important for multimodal
        )