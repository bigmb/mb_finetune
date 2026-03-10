"""Checkpoint management callback for finetuning."""

from __future__ import annotations

import logging
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

__all__ = ["CheckpointCallback"]

logger = logging.getLogger("mb.finetune")


class CheckpointCallback(TrainerCallback):
    """
    Callback that logs checkpoint saves and can do custom cleanup.
    """

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        logger.info(f"Checkpoint saved: {checkpoint_dir}")

    def on_train_end(self, args, state, control, **kwargs):
        logger.info(f"Final model saved to: {args.output_dir}")
