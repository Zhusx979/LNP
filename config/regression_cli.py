"""CLI builders for the regression entry script."""

import argparse
from typing import Dict

from src.training_common import str2bool


DEFAULT_REGRESSION_CSV = None
DEFAULT_AGILE_CELL_LINE = "Hela"
DEFAULT_AGILE_SPLIT = "scaffold"


def build_config_from_args(args: argparse.Namespace) -> Dict:
    """Build the nested training config from CLI arguments."""
    return {
        "model": {
            "mixed_precision": args.mixed_precision,
            "full_finetune": args.full_finetune,
            "train_embeddings": args.train_embeddings,
            "use_amp": args.use_amp,
            "use_grad_scaler": args.use_grad_scaler,
        },
        "data": {
            "csv_path": args.csv,
            "batch_size": args.batch_size,
            "auto_discover_agile": not args.no_auto_discover,
            "agile_cell_line": args.agile_cell_line,
            "agile_split": args.agile_split,
        },
        "training": {
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_grad_norm": args.max_grad_norm,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "paths": {
            "pretrained_model_path": args.pretrained_model_path,
            "tokenizer_path": args.tokenizer_path,
            "output_dir": args.output_dir,
            "logs_dir": args.logs_dir,
        },
        "runtime": {
            "gpus": args.gpus,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for regression fine-tuning."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen for transfection efficiency regression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with the dataset defaults defined in this script
  python train_regression.py

  # Train only on AGILE/Hela/cliff
  python train_regression.py --agile-cell-line Hela --agile-split cliff

  # Use manual CSV
  python train_regression.py --csv path/to/labels.csv

  # Custom settings
  python train_regression.py --num-epochs 20 --learning-rate 5e-6
        """,
    )
    parser.add_argument(
        "--pretrained-model-path",
        "--pretrained_model_path",
        dest="pretrained_model_path",
        default="models/qwen_1.8b_smiles_pretrained/final_model",
        help="Path to pretrained model from Stage 1",
    )
    parser.add_argument(
        "--tokenizer-path",
        "--tokenizer_path",
        dest="tokenizer_path",
        default="models/qwen_1.8b_smiles_pretrained/tokenizer.json",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_REGRESSION_CSV,
        help="Path to regression CSV (optional, auto-discovers AGILE if not provided)",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        default="models/qwen_1.8b_smiles_regression",
        help="Output directory",
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory for regression logs and metrics",
    )
    parser.add_argument(
        "--no-auto-discover",
        action="store_true",
        help="Disable AGILE auto-discovery",
    )
    parser.add_argument(
        "--agile-cell-line",
        default=DEFAULT_AGILE_CELL_LINE,
        help="Filter AGILE by cell line: Hela or RaW",
    )
    parser.add_argument(
        "--agile-split",
        default=DEFAULT_AGILE_SPLIT,
        help="Filter AGILE by split type: cliff or scaffold",
    )

    parser.add_argument(
        "--num-epochs",
        "--epochs",
        dest="num_epochs",
        type=int,
        default=1,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        "--batch",
        dest="batch_size",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        dest="learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        "--gradient_accumulation_steps",
        dest="gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max-grad-norm",
        "--max_grad_norm",
        dest="max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.001,
        help="AdamW weight decay",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument(
        "--mixed-precision",
        "--mixed_precision",
        dest="mixed_precision",
        choices=["none", "fp16", "bf16"],
        default="bf16",
        help="Mixed precision mode used during training",
    )
    parser.add_argument(
        "--use-amp",
        "--use_amp",
        dest="use_amp",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Compatibility flag kept for old configs; runtime precision is still derived from --mixed-precision",
    )
    parser.add_argument(
        "--use-grad-scaler",
        "--use_grad_scaler",
        dest="use_grad_scaler",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Compatibility flag kept for old configs; scaler usage is still derived from --mixed-precision",
    )
    parser.add_argument(
        "--full-finetune",
        "--full_finetune",
        dest="full_finetune",
        action="store_true",
        help="Train the entire backbone instead of freezing it",
    )
    parser.add_argument(
        "--train-embeddings",
        dest="train_embeddings",
        action="store_true",
        default=True,
        help="Keep resized input embeddings trainable when backbone is frozen",
    )
    parser.add_argument(
        "--no-train-embeddings",
        dest="train_embeddings",
        action="store_false",
        help="Freeze resized input embeddings together with the backbone",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="GPU selection: '0' for single GPU, '0,1' for multi-GPU, 'all' for all visible GPUs, or 'cpu'",
    )

    return parser
