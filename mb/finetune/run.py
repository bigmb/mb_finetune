"""CLI entry point for mb_finetune.

Usage:
    python -m mb.finetune.run --config config.yaml
"""

import argparse
from mb.finetune.config import FinetuneConfig
from mb.finetune.trainer import FinetuneTrainer
from mb.utils.logging import logg, logger as mb_logger

def main():
    parser = argparse.ArgumentParser(
        description="mb_finetune – Finetune vision-language and text models"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only (requires resume_from_checkpoint in config)",
    )
    args = parser.parse_args()

    config = FinetuneConfig.from_yaml(args.config)

    trainer = FinetuneTrainer(config)
    logger = trainer.logger or mb_logger

    if args.eval_only:
        trainer.train()  # will resume from checkpoint
        metrics = trainer.evaluate()
        logg.info(f"Evaluation metrics: {metrics}", logger=logger)
    else:
        trainer.train()
        trainer.save()
        logg.info("Done.", logger=logger)


if __name__ == "__main__":
    main()
