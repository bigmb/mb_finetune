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
        """Log model architecture summary to TensorBoard.

        torch.jit.trace (used by add_graph) reliably fails on modern HuggingFace
        transformer models due to dynamic control flow, ModelOutput return types,
        and PEFT/quantisation wrappers.  We log the architecture as structured text
        instead, which always works and is just as useful for auditing the model.
        """
        try:
            from torch.utils.tensorboard import SummaryWriter

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            summary_lines = [
                f"**Model type:** `{type(model).__name__}`",
                f"**Total parameters:** {total_params:,}",
                f"**Trainable parameters:** {trainable_params:,}",
                f"**Frozen parameters:** {total_params - trainable_params:,}",
                "",
                "**Layer breakdown:**",
                "```",
            ]
            for name, module in model.named_modules():
                if name == "":
                    continue
                depth = name.count(".")
                if depth > 2:
                    continue
                num_params = sum(p.numel() for p in module.parameters(recurse=False))
                indent = "  " * depth
                summary_lines.append(
                    f"{indent}{name}: {type(module).__name__}"
                    + (f"  [{num_params:,} params]" if num_params else "")
                )
            summary_lines.append("```")

            writer = SummaryWriter(log_dir=args.logging_dir)
            try:
                writer.add_text("model/architecture", "  \n".join(summary_lines), global_step=0)
            finally:
                writer.close()

            logg.info(
                f"Model architecture logged to TensorBoard "
                f"({trainable_params:,} / {total_params:,} params trainable)"
            )
        except Exception as e:
            logg.warning(f"Could not log model architecture to TensorBoard: {e}")

    @staticmethod
    def _log_data_samples_to_tb(args, train_dataset):
        """Log a few dataset samples as text and images to TensorBoard."""
        if train_dataset is None:
            return
        try:
            import numpy as np
            from PIL import Image
            from torch.utils.tensorboard import SummaryWriter

            num_samples = min(10, len(train_dataset))
            image_column = getattr(train_dataset, "image_column", None)

            writer = SummaryWriter(log_dir=args.logging_dir)
            try:
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

                    if hasattr(train_dataset, "_resolve_image"):
                        sample = train_dataset._resolve_image(sample)

                    lines = [f"**Sample {i}**\n"]
                    for k, v in sample.items():
                        v_str = str(v)
                        if len(v_str) > 500:
                            v_str = v_str[:500] + "…"
                        lines.append(f"- **{k}**: {v_str}")
                    writer.add_text(f"data/sample_{i}", "  \n".join(lines), global_step=0)

                    if image_column and image_column in sample:
                        img_path = sample[image_column]
                        try:
                            img = Image.open(img_path).convert("RGB")
                            img_array = np.array(img)
                            img_chw = img_array.transpose(2, 0, 1).astype(np.float32) / 255.0
                            writer.add_image(f"data/sample_{i}", img_chw, global_step=0)
                        except Exception as img_err:
                            logg.warning(f"Could not log image for sample {i}: {img_err}")
            finally:
                writer.close()
        except Exception as e:
            logg.warning(f"Could not log data samples to TensorBoard: {e}")
