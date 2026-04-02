"""
Checkpoint management callback for finetuning. Using Transformer utils for checkpointing
"""

from __future__ import annotations
from mb.utils.logging import logg
from pathlib import Path

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

__all__ = ["CheckpointCallback"]

class CheckpointCallback(TrainerCallback):
    """
    Callback that logs checkpoint saves and can do custom cleanup.
    """
    def __init__(self, logger=None):
        self.logger = logger

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        logg.info(f"Checkpoint saved: {checkpoint_dir}", logger=self.logger)

    def on_train_end(self, args, state, control, **kwargs):
        logg.info(f"Final model saved to: {args.output_dir}", logger=self.logger)
