"""
Logging callback for finetuning. Using Transformer utils for logging structured training metrics.
"""

from __future__ import annotations
from typing import Any, Dict
from mb.utils.logging import logg
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

__all__ = ["LoggingCallback"]


class LoggingCallback(TrainerCallback):
    """Custom logging callback that writes structured training metrics."""
    def __init__(self, logger=None):
        self.logger = logger

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

        logg.info(" | ".join(parts), logger=self.logger)

    def on_train_begin(self, args, state, control, **kwargs):
        logg.info("=== Finetuning started ===", logger=self.logger)
        logg.info(f"  Output dir: {args.output_dir}", logger=self.logger)
        logg.info(f"  Epochs:     {args.num_train_epochs}", logger=self.logger)
        logg.info(f"  Batch size: {args.per_device_train_batch_size}", logger=self.logger)
        logg.info(f"  LR:         {args.learning_rate}", logger=self.logger)

    def on_train_end(self, args, state, control, **kwargs):
        logg.info(f"=== Finetuning complete — {state.global_step} steps ===", logger=self.logger)
