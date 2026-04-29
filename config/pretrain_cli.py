"""CLI builders for the pretraining entry script."""

import argparse
from typing import Dict

from src.training_common import str2bool


def build_config_from_args(args: argparse.Namespace) -> Dict:
    """Build the nested training config from CLI arguments."""
    return {
        "model": {
            "model_name": args.model_name,
            "max_seq_length": args.max_seq_length,
        },
        "data": {
            "csv_paths": args.csv_paths,
            "validation_split": args.validation_split,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "deduplicate": args.deduplicate,
        },
        "training": {
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_grad_norm": args.max_grad_norm,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "logging_steps": args.logging_steps,
            "seed": args.seed,
        },
        "optimization": {
            "mixed_precision": args.mixed_precision,
            "gradient_checkpointing": args.gradient_checkpointing,
            "use_amp": args.use_amp,
            "use_grad_scaler": args.use_grad_scaler,
        },
        "paths": {
            "output_dir": args.output_dir,
            "logs_dir": args.logs_dir,
            "tokenizer_path": args.tokenizer_path,
            "model_cache_dir": args.model_cache_dir,
        },
        "runtime": {
            "gpus": args.gpus,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for pretraining."""
    parser = argparse.ArgumentParser(
        description="Train Qwen-1.8B for SMILES modeling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to checkpoint directory to resume from",
    )

    parser.add_argument(
        "--model-name",
        default="qwen/Qwen-1_8B",
        help="Base model name used by ModelScope download",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=256,
        help="Maximum SMILES sequence length",
    )

    parser.add_argument(
        "--csv-paths",
        nargs="+",
        default=["data/test_lipids.csv"],
        help="One or more pretraining CSV paths",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.15,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--deduplicate",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to deduplicate SMILES samples before training",
    )

    parser.add_argument("--num-epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Scheduler warmup steps",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Run validation every N optimizer steps",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N optimizer steps",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=50,
        help="Log training metrics every N optimizer steps",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument(
        "--mixed-precision",
        choices=["none", "fp16", "bf16"],
        default="bf16",
        help="Mixed precision mode used during training",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--use-amp",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Compatibility flag kept for old configs; runtime precision is still derived from --mixed-precision",
    )
    parser.add_argument(
        "--use-grad-scaler",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Compatibility flag kept for old configs; scaler usage is still derived from --mixed-precision",
    )

    parser.add_argument(
        "--output-dir",
        default="models/qwen_1.8b_smiles_pretrained",
        help="Directory for checkpoints and final model",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory for training logs and metrics",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="models/qwen_1.8b_smiles_pretrained/tokenizer.json",
        help="Path for saving the main tokenizer artifact",
    )
    parser.add_argument(
        "--model-cache-dir",
        default="models/cache/qwen-1.8b",
        help="ModelScope cache directory for downloaded base weights",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="GPU selection: '0' for single GPU, '0,1' for multi-GPU, 'all' for all visible GPUs, or 'cpu'",
    )

    return parser
