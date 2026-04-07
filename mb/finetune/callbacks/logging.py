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
    def __init__(self, logger=None, train_dataset=None):
        self.logger = logger
        self._train_dataset = train_dataset

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
        effective_batch = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * max(args.world_size, 1)
        )
        logg.info("=== Finetuning started ===", logger=self.logger)
        logg.info(f"  Output dir:           {args.output_dir}", logger=self.logger)
        logg.info(f"  Epochs:               {args.num_train_epochs}", logger=self.logger)
        logg.info(f"  Per-device batch:     {args.per_device_train_batch_size}", logger=self.logger)
        logg.info(f"  Gradient accum steps: {args.gradient_accumulation_steps}", logger=self.logger)
        logg.info(f"  World size:           {max(args.world_size, 1)}", logger=self.logger)
        logg.info(f"  Effective batch size: {effective_batch}", logger=self.logger)
        logg.info(f"  LR:                   {args.learning_rate}", logger=self.logger)

        # Log sample data to TensorBoard
        self._log_data_samples_to_tb(args, self._train_dataset)

        # Log model graph to TensorBoard
        model = kwargs.get("model")
        if model is not None:
            self._log_model_graph_to_tb(args, model, self._train_dataset)

    def on_train_end(self, args, state, control, **kwargs):
        logg.info(f"=== Finetuning complete — {state.global_step} steps ===", logger=self.logger)

    @staticmethod
    def _log_model_graph_to_tb(args, model, train_dataset):
        """Log model computation graph to TensorBoard."""
        try:
            import torch
            from torch.utils.tensorboard import SummaryWriter

            if train_dataset is None or len(train_dataset) == 0:
                return

            # Get a single sample from the dataset
            sample = train_dataset[0]
            if not isinstance(sample, dict):
                return

            # Build a dummy batch on the model's device
            device = next(model.parameters()).device
            dummy_inputs = {}
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    dummy_inputs[k] = v.unsqueeze(0).to(device)

            if "input_ids" not in dummy_inputs:
                return

            writer = SummaryWriter(log_dir=args.logging_dir)

            # Trace the model with torch.no_grad to avoid grad issues
            model.eval()
            with torch.no_grad():
                writer.add_graph(model, input_to_model=dummy_inputs, use_strict_trace=False)

            writer.close()
            model.train()
            logg.info("Model graph logged to TensorBoard")
        except Exception as e:
            logg.warning(f"Could not log model graph to TensorBoard: {e}")

    @staticmethod
    def _log_data_samples_to_tb(args, train_dataset):
        """Log a few dataset samples as text to TensorBoard."""
        if train_dataset is None:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=args.logging_dir)
            num_samples = min(10, len(train_dataset))

            writer.add_text(
                "data/info",
                f"**Dataset size:** {len(train_dataset)} samples  \n"
                f"**Showing first {num_samples} raw samples**",
                global_step=0,
            )

            for i in range(num_samples):
                sample = train_dataset.samples[i] if hasattr(train_dataset, "samples") else {}
                if not sample:
                    continue
                lines = [f"**Sample {i}**\n"]
                for k, v in sample.items():
                    v_str = str(v)
                    if len(v_str) > 500:
                        v_str = v_str[:500] + "…"
                    lines.append(f"- **{k}**: {v_str}")
                writer.add_text(f"data/sample_{i}", "  \n".join(lines), global_step=0)

            writer.close()
        except Exception as e:
            logg.warning(f"Could not log data samples to TensorBoard: {e}")
