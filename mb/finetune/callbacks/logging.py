"""Logging callback for finetuning."""

from __future__ import annotations

import logging
from typing import Any, Dict

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

__all__ = ["LoggingCallback"]

logger = logging.getLogger("mb.finetune")


class LoggingCallback(TrainerCallback):
    """Custom logging callback that writes structured training metrics."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, Any] = None,
        **kwargs,
    ):
        if logs is None:
            return

        step = state.global_step
        epoch = state.epoch or 0.0
        parts = [f"step={step}", f"epoch={epoch:.2f}"]

        for key in ("loss", "eval_loss", "learning_rate"):
            if key in logs:
                parts.append(f"{key}={logs[key]:.6f}")

        logger.info(" | ".join(parts))

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("=== Finetuning started ===")
        logger.info(f"  Output dir: {args.output_dir}")
        logger.info(f"  Epochs:     {args.num_train_epochs}")
        logger.info(f"  Batch size: {args.per_device_train_batch_size}")
        logger.info(f"  LR:         {args.learning_rate}")

    def on_train_end(self, args, state, control, **kwargs):
        logger.info(f"=== Finetuning complete — {state.global_step} steps ===")
