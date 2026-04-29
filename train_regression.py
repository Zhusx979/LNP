"""
Stage 2: Training Script for Regression Task (Transfection Efficiency)

Supports:
1. Auto-discovery of AGILE datasets (AGILE/*/*/{train,test}.csv)
2. Manual CSV path: CSV with columns [SMILES, TARGET]

This script will:
1. Load pretrained Qwen + SMILES tokenizer from Stage 1
2. Initialize regression head
3. Fine-tune on transfection efficiency labels
4. Evaluate with RMSE, MAE, Pearson correlation
"""

import logging
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import explained_variance_score, median_absolute_error, r2_score

# Import custom modules
PROJECT_ROOT = Path(__file__).parent
PROJECT_ROOT.joinpath("logs").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from src.model_regression import QwenRegressionModel, RegressionDataModule
from src.tokenizer import SMILESTokenizer

try:
    from transformers import AutoModelForCausalLM
except ImportError:
    AutoModelForCausalLM = None

logger = logging.getLogger(__name__)




def str2bool(value):
    """Parse flexible boolean CLI values."""
    if isinstance(value, bool):
        return value

    lowered = value.lower()
    if lowered in {'true', '1', 'yes', 'y', 'on'}:
        return True
    if lowered in {'false', '0', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_path(path_value: str) -> Path:
    """Resolve a project-relative path."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def configure_logging(logs_dir: Path, dataset_name: Optional[str] = None):
    """Configure file and console logging once CLI args are known."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_filename = (
        f"regression_training_{dataset_name}.log"
        if dataset_name
        else "regression_training.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / log_filename),
            logging.StreamHandler()
        ],
        force=True,
    )


def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_artifact_name(value: str) -> str:
    """Create a filesystem-friendly dataset identifier."""
    normalized = re.sub(r'[^A-Za-z0-9._-]+', '_', value.strip())
    normalized = normalized.strip('._-')
    return normalized or "dataset"


def infer_regression_dataset_name(
    csv_path: Optional[str],
    auto_discover_agile: bool,
    agile_cell_line: Optional[str],
    agile_split: Optional[str],
) -> str:
    """Infer a stable dataset name for run artifacts."""
    if csv_path:
        return sanitize_artifact_name(Path(csv_path).stem)

    if auto_discover_agile:
        return sanitize_artifact_name(
            f"AGILE_{agile_cell_line or 'ALL'}_{agile_split or 'ALL'}"
        )

    return "regression_run"


def parse_gpu_ids(gpus: Optional[str]) -> List[int]:
    """Parse GPU selection like '0', '0,1', 'all', or 'cpu'."""
    if gpus is None:
        return [0] if torch.cuda.is_available() else []

    normalized = gpus.strip().lower()
    if normalized in {'cpu', 'none'}:
        return []
    if normalized == 'all':
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))

    gpu_ids = []
    for token in gpus.split(','):
        token = token.strip()
        if not token:
            continue
        try:
            gpu_ids.append(int(token))
        except ValueError as exc:
            raise ValueError(
                f"Invalid GPU selection '{gpus}'. Use formats like '0', '0,1', 'all', or 'cpu'."
            ) from exc

    if not gpu_ids:
        raise ValueError(
            f"Invalid GPU selection '{gpus}'. Use formats like '0', '0,1', 'all', or 'cpu'."
        )

    deduplicated = []
    for gpu_id in gpu_ids:
        if gpu_id not in deduplicated:
            deduplicated.append(gpu_id)
    return deduplicated


def resolve_device_config(gpus: Optional[str]) -> Tuple[torch.device, List[int]]:
    """Resolve the runtime device and validated GPU ids."""
    gpu_ids = parse_gpu_ids(gpus)
    if not gpu_ids:
        return torch.device('cpu'), []

    if not torch.cuda.is_available():
        logger.warning("CUDA is unavailable, falling back to CPU.")
        return torch.device('cpu'), []

    available_gpu_count = torch.cuda.device_count()
    invalid_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= available_gpu_count]
    if invalid_gpu_ids:
        raise ValueError(
            f"Requested GPU ids {invalid_gpu_ids} but only {available_gpu_count} GPU(s) are available."
        )

    return torch.device(f'cuda:{gpu_ids[0]}'), gpu_ids


def ensure_transformers():
    """Fail fast when training starts without required dependencies."""
    if AutoModelForCausalLM is None:
        print("ERROR: transformers not installed. Run: pip install -r requirements.txt")
        sys.exit(1)


def ensure_local_qwen_code(model_dir: Path):
    """Copy missing Qwen trust_remote_code files into a saved local model directory."""
    config_path = model_dir / 'config.json'
    if not config_path.exists():
        return

    required_files = [
        'modeling_qwen.py',
        'configuration_qwen.py',
        'qwen_generation_utils.py',
        'tokenization_qwen.py',
        'cpp_kernels.py',
    ]
    missing_files = [name for name in required_files if not (model_dir / name).exists()]
    if not missing_files:
        return

    with open(config_path, 'r', encoding='utf-8') as handle:
        config = json.load(handle)

    source_dir = config.get('_name_or_path')
    if not source_dir:
        return

    source_path = Path(source_dir)
    if not source_path.exists():
        return

    copied_files = []
    for filename in missing_files:
        source_file = source_path / filename
        destination_file = model_dir / filename
        if source_file.exists():
            shutil.copy2(source_file, destination_file)
            copied_files.append(filename)

    if copied_files:
        logger.info(
            "Copied Qwen support files into %s: %s",
            model_dir,
            ", ".join(copied_files),
        )


def build_config_from_args(args: argparse.Namespace) -> Dict:
    """Build the nested training config from CLI arguments."""
    return {
        'model': {
            'mixed_precision': args.mixed_precision,
            'full_finetune': args.full_finetune,
            'train_embeddings': args.train_embeddings,
            'use_amp': args.use_amp,
            'use_grad_scaler': args.use_grad_scaler,
        },
        'data': {
            'csv_path': args.csv,
            'batch_size': args.batch_size,
            'auto_discover_agile': not args.no_auto_discover,
            'agile_cell_line': args.agile_cell_line,
            'agile_split': args.agile_split,
        },
        'training': {
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'max_grad_norm': args.max_grad_norm,
            'weight_decay': args.weight_decay,
            'seed': args.seed,
        },
        'paths': {
            'pretrained_model_path': args.pretrained_model_path,
            'tokenizer_path': args.tokenizer_path,
            'output_dir': args.output_dir,
            'logs_dir': args.logs_dir,
        },
        'runtime': {
            'gpus': args.gpus,
        },
    }

# Edit these defaults if you want to choose an AGILE subset
# without passing command-line arguments.
DEFAULT_REGRESSION_CSV = None
DEFAULT_AGILE_CELL_LINE = 'Hela'
DEFAULT_AGILE_SPLIT = 'scaffold'
def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for regression fine-tuning."""
    parser = argparse.ArgumentParser(
        description='Fine-tune Qwen for transfection efficiency regression',
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
    parser.add_argument('--pretrained-model-path', '--pretrained_model_path',
                        dest='pretrained_model_path',
                        default='models/qwen_1.8b_smiles_pretrained/final_model',
                        help='Path to pretrained model from Stage 1')
    parser.add_argument('--tokenizer-path', '--tokenizer_path',
                        dest='tokenizer_path',
                        default='models/qwen_1.8b_smiles_pretrained/tokenizer.json',
                        help='Path to tokenizer')
    parser.add_argument('--csv', default=DEFAULT_REGRESSION_CSV,
                        help='Path to regression CSV (optional, auto-discovers AGILE if not provided)')
    parser.add_argument('--output-dir', '--output',
                        dest='output_dir',
                        default='models/qwen_1.8b_smiles_regression',
                        help='Output directory')
    parser.add_argument('--logs-dir', default='logs',
                        help='Directory for regression logs and metrics')
    parser.add_argument('--no-auto-discover', action='store_true',
                        help='Disable AGILE auto-discovery')
    parser.add_argument('--agile-cell-line', default=DEFAULT_AGILE_CELL_LINE,
                        help='Filter AGILE by cell line: Hela or RaW')
    parser.add_argument('--agile-split', default=DEFAULT_AGILE_SPLIT,
                        help='Filter AGILE by split type: cliff or scaffold')

    parser.add_argument('--num-epochs', '--epochs',
                        dest='num_epochs',
                        type=int,
                        default=1,
                        help='Number of epochs')
    parser.add_argument('--batch-size', '--batch',
                        dest='batch_size',
                        type=int,
                        default=1,
                        help='Batch size')
    parser.add_argument('--learning-rate', '--lr',
                        dest='learning_rate',
                        type=float,
                        default=1e-5,
                        help='Learning rate')
    parser.add_argument('--gradient-accumulation-steps', '--gradient_accumulation_steps',
                        dest='gradient_accumulation_steps',
                        type=int,
                        default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--max-grad-norm', '--max_grad_norm',
                        dest='max_grad_norm',
                        type=float,
                        default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                        help='AdamW weight decay')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    parser.add_argument('--mixed-precision', '--mixed_precision',
                        dest='mixed_precision',
                        choices=['none', 'fp16', 'bf16'], default='bf16',
                        help='Mixed precision mode used during training')
    parser.add_argument('--use-amp', '--use_amp',
                        dest='use_amp',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True,
                        help='Compatibility flag kept for old configs; runtime precision is still derived from --mixed-precision')
    parser.add_argument('--use-grad-scaler', '--use_grad_scaler',
                        dest='use_grad_scaler',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True,
                        help='Compatibility flag kept for old configs; scaler usage is still derived from --mixed-precision')
    parser.add_argument('--full-finetune', '--full_finetune',
                        dest='full_finetune',
                        action='store_true',
                        help='Train the entire backbone instead of freezing it')
    parser.add_argument('--train-embeddings',
                        dest='train_embeddings',
                        action='store_true',
                        default=True,
                        help='Keep resized input embeddings trainable when backbone is frozen')
    parser.add_argument('--no-train-embeddings',
                        dest='train_embeddings',
                        action='store_false',
                        help='Freeze resized input embeddings together with the backbone')
    parser.add_argument('--gpus', default=None,
                        help="GPU selection: '0' for single GPU, '0,1' for multi-GPU, 'all' for all visible GPUs, or 'cpu'")

    return parser


class QwenRegressionTrainer:
    """Training class for transfection efficiency regression."""
    
    def __init__(
        self,
        config: Dict,
    ):
        """
        Initialize trainer.
        
        Args:
            pretrained_model_path: Path to pretrained Qwen model from Stage 1
            tokenizer_path: Path to saved SMILES tokenizer
            regression_csv: Path to CSV with SMILES + TARGET (optional, uses AGILE if None)
            output_dir: Output directory for fine-tuned model
            auto_discover_agile: Auto-discover AGILE datasets if csv is None
        """
        # Check for required files
        self.config = config
        self.paths_config = config['paths']
        self.data_config = config['data']
        self.training_config = config['training']
        self.model_config = config['model']
        self.runtime_config = config.get('runtime', {})
        self.dataset_name = infer_regression_dataset_name(
            csv_path=self.data_config.get('csv_path'),
            auto_discover_agile=self.data_config.get('auto_discover_agile', False),
            agile_cell_line=self.data_config.get('agile_cell_line'),
            agile_split=self.data_config.get('agile_split'),
        )

        self.pretrained_model_path = resolve_path(self.paths_config['pretrained_model_path'])
        self.tokenizer_path = resolve_path(self.paths_config['tokenizer_path'])
        self.output_root_dir = resolve_path(self.paths_config['output_dir'])
        self.output_dir = self.output_root_dir / self.dataset_name
        self.logs_dir = resolve_path(self.paths_config['logs_dir'])

        self._validate_inputs(self.pretrained_model_path, self.tokenizer_path)
        self.regression_csv = (
            str(resolve_path(self.data_config['csv_path']))
            if self.data_config['csv_path'] is not None
            else None
        )
        self.batch_size = self.data_config['batch_size']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.auto_discover_agile = self.data_config['auto_discover_agile']
        self.agile_cell_line = self.data_config.get('agile_cell_line')
        self.agile_split = self.data_config.get('agile_split')
        
        self.device, self.gpu_ids = resolve_device_config(self.runtime_config.get('gpus'))
        logger.info(f"Device: {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"GPU ids: {self.gpu_ids if self.gpu_ids else 'CPU only'}")
        logger.info(f"Training seed: {self.training_config['seed']}")
        if self.regression_csv is None and self.auto_discover_agile:
            logger.info(
                "Selected AGILE subset -> cell line: %s, split: %s",
                self.agile_cell_line or "ALL",
                self.agile_split or "ALL",
            )
        set_seed(self.training_config['seed'])
        
        # Components
        self.tokenizer = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.data_module = None
        self.amp_dtype = None
        self.use_amp = self.model_config['use_amp']
        self.use_grad_scaler = self.model_config['use_grad_scaler']
        self.metrics_history = []
        self.best_epoch = None
        self.best_model_metrics = None
        
        # Training state
        self.best_val_rmse = float('inf')

    def _artifact_path(self, stem: str, suffix: str) -> Path:
        """Build a dataset-scoped artifact path inside the logs directory."""
        return self.logs_dir / f"{stem}_{self.dataset_name}{suffix}"

    @staticmethod
    def _safe_correlation(fn, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """Compute correlation safely for short or constant arrays."""
        if len(y_true) < 2:
            return float('nan'), float('nan')
        if np.isclose(np.std(y_true), 0.0) or np.isclose(np.std(y_pred), 0.0):
            return float('nan'), float('nan')

        try:
            corr, p_value = fn(y_true, y_pred)
        except Exception:
            return float('nan'), float('nan')
        return float(corr), float(p_value)

    @classmethod
    def _compute_regression_metrics(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute a comprehensive set of regression metrics."""
        if len(y_true) == 0:
            return {
                'mse': float('nan'),
                'rmse': float('nan'),
                'mae': float('nan'),
                'median_ae': float('nan'),
                'r2': float('nan'),
                'explained_variance': float('nan'),
                'pearson_r': float('nan'),
                'pearson_pvalue': float('nan'),
                'spearman_r': float('nan'),
                'spearman_pvalue': float('nan'),
                'mean_error': float('nan'),
                'std_error': float('nan'),
                'max_abs_error': float('nan'),
                'smape': float('nan'),
                'mape_nonzero': float('nan'),
                'nrmse_range': float('nan'),
                'nrmse_std': float('nan'),
            }

        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        mse = float(np.mean(np.square(errors)))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(abs_errors))
        median_ae = float(median_absolute_error(y_true, y_pred))
        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors))
        max_abs_error = float(np.max(abs_errors))

        pearson_r, pearson_pvalue = cls._safe_correlation(pearsonr, y_true, y_pred)
        spearman_r, spearman_pvalue = cls._safe_correlation(spearmanr, y_true, y_pred)

        y_range = float(np.max(y_true) - np.min(y_true)) if len(y_true) else float('nan')
        y_std = float(np.std(y_true)) if len(y_true) else float('nan')
        nrmse_range = float(rmse / y_range) if y_range and not np.isclose(y_range, 0.0) else float('nan')
        nrmse_std = float(rmse / y_std) if y_std and not np.isclose(y_std, 0.0) else float('nan')

        denominator = np.abs(y_true) + np.abs(y_pred) + 1e-12
        smape = float(np.mean(2.0 * abs_errors / denominator) * 100.0)
        non_zero_mask = np.abs(y_true) > 1e-12
        if np.any(non_zero_mask):
            mape_nonzero = float(
                np.mean(abs_errors[non_zero_mask] / np.abs(y_true[non_zero_mask])) * 100.0
            )
        else:
            mape_nonzero = float('nan')

        if len(y_true) >= 2:
            r2 = float(r2_score(y_true, y_pred))
            explained_variance = float(explained_variance_score(y_true, y_pred))
        else:
            r2 = float('nan')
            explained_variance = float('nan')

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'median_ae': median_ae,
            'r2': r2,
            'explained_variance': explained_variance,
            'pearson_r': pearson_r,
            'pearson_pvalue': pearson_pvalue,
            'spearman_r': spearman_r,
            'spearman_pvalue': spearman_pvalue,
            'mean_error': mean_error,
            'std_error': std_error,
            'max_abs_error': max_abs_error,
            'smape': smape,
            'mape_nonzero': mape_nonzero,
            'nrmse_range': nrmse_range,
            'nrmse_std': nrmse_std,
        }

    @staticmethod
    def _to_builtin(value):
        """Convert numpy/pandas scalars to JSON-serializable Python values."""
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    def _json_ready_dict(self, data: Dict) -> Dict:
        """Recursively convert metrics dictionaries for JSON serialization."""
        json_ready = {}
        for key, value in data.items():
            if isinstance(value, dict):
                json_ready[key] = self._json_ready_dict(value)
            elif isinstance(value, list):
                json_ready[key] = [
                    self._json_ready_dict(item) if isinstance(item, dict) else self._to_builtin(item)
                    for item in value
                ]
            else:
                json_ready[key] = self._to_builtin(value)
        return json_ready

    def _unwrap_model(self):
        """Return the underlying model when wrapped by DataParallel."""
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module
        return self.model

    def _maybe_enable_data_parallel(self):
        """Wrap the model for multi-GPU training when requested."""
        if len(self.gpu_ids) <= 1:
            logger.info("  GPU mode: single device")
            return

        self.model = torch.nn.DataParallel(
            self.model,
            device_ids=self.gpu_ids,
            output_device=self.gpu_ids[0],
        )
        logger.info(f"  GPU mode: DataParallel on GPUs {self.gpu_ids}")
    
    @staticmethod
    def _validate_inputs(pretrained_path: Path, tokenizer_path: Path):
        """Validate that model and tokenizer files exist."""
        if not pretrained_path.exists():
            raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    def setup(self):
        """Initialize tokenizer, model, and data."""
        ensure_transformers()
        logger.info("="*60)
        logger.info("SETUP PHASE (REGRESSION)")
        logger.info("="*60)
        
        # Load tokenizer
        logger.info("[1/3] Loading SMILES tokenizer...")
        self.tokenizer = SMILESTokenizer()
        self.tokenizer.load(self.tokenizer_path)
        logger.info(f"  Initial vocabulary size: {len(self.tokenizer)}")

        data_module = RegressionDataModule(
            csv_path=self.regression_csv,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            auto_discover_agile=self.auto_discover_agile,
            agile_cell_line=self.agile_cell_line,
            agile_split=self.agile_split,
        )
        self.data_module = data_module

        smiles_list, _ = data_module.load_data()
        original_vocab_size = len(self.tokenizer)
        self.tokenizer.build_vocab(smiles_list)
        expanded_vocab_size = len(self.tokenizer)
        logger.info(
            f"  Regression vocabulary size: {expanded_vocab_size} "
            f"(added {expanded_vocab_size - original_vocab_size} tokens)"
        )

        # Load pretrained model
        logger.info("[2/3] Loading pretrained Qwen model...")
        ensure_local_qwen_code(self.pretrained_model_path)
        mixed_precision = self.model_config['mixed_precision']
        if torch.cuda.is_available() and mixed_precision == 'fp16':
            self.amp_dtype = torch.float16
        elif torch.cuda.is_available() and mixed_precision == 'bf16':
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = None
        self.use_amp = self.amp_dtype is not None
        self.use_grad_scaler = self.amp_dtype == torch.float16

        dtype = self.amp_dtype or torch.float32

        base_model = AutoModelForCausalLM.from_pretrained(
            self.pretrained_model_path,
            torch_dtype=dtype,
            device_map=None,
            trust_remote_code=True,
        )
        base_model.resize_token_embeddings(len(self.tokenizer))

        # Create regression model
        self.model = QwenRegressionModel(base_model)
        self.model = self.model.to(self.device, dtype=dtype)
        self._configure_trainable_parameters()
        self._maybe_enable_data_parallel()
        logger.info(f"  Mixed precision mode: {mixed_precision}")
        logger.info(f"  AMP enabled: {self.use_amp}")
        logger.info(f"  GradScaler enabled: {self.use_grad_scaler}")
        logger.info(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"  Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        # Load data
        logger.info("[3/3] Loading regression dataset...")
        data_module.setup()
        self.train_loader, self.val_loader, self.test_loader = data_module.create_loaders()
        
        logger.info(f"  Train batches: {len(self.train_loader)}")
        logger.info(f"  Val batches: {len(self.val_loader)}")
        logger.info(f"  Test batches: {len(self.test_loader)}")
        logger.info(f"  Dataset artifact name: {self.dataset_name}")
        logger.info(f"  Output directory: {self.output_dir}")
        
        logger.info("="*60)
        logger.info("Setup complete\n")

    def _configure_trainable_parameters(self):
        """Freeze the backbone by default to keep single-GPU fine-tuning tractable."""
        trainable_model = self._unwrap_model()

        if self.model_config['full_finetune']:
            for parameter in trainable_model.parameters():
                parameter.requires_grad = True
            logger.info("  Full fine-tuning enabled")
            return

        for parameter in trainable_model.model.parameters():
            parameter.requires_grad = False

        if self.model_config['train_embeddings']:
            input_embeddings = trainable_model.model.get_input_embeddings()
            for parameter in input_embeddings.parameters():
                parameter.requires_grad = True

        for parameter in trainable_model.head.parameters():
            parameter.requires_grad = True

        logger.info(
            f"  Backbone frozen: True | Train embeddings: {self.model_config['train_embeddings']}"
        )
    
    def train(self):
        """Train regression model."""
        logger.info("="*60)
        logger.info("TRAINING PHASE (REGRESSION)")
        logger.info("="*60 + "\n")
        
        # Setup optimizer
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay'],
        )

        criterion = nn.MSELoss()
        scaler = GradScaler(enabled=self.use_grad_scaler)
        
        try:
            for epoch in range(self.training_config['num_epochs']):
                # Training
                train_loss = self._train_epoch(epoch, optimizer, criterion, scaler)
                
                # Validation
                val_metrics, _ = self._evaluate_loader(
                    self.val_loader,
                    criterion,
                    split_name='val',
                    save_predictions=False,
                )
                val_loss = val_metrics['loss']
                val_rmse = val_metrics['rmse']
                val_mae = val_metrics['mae']
                
                logger.info(f"Epoch {epoch+1}/{self.training_config['num_epochs']}")
                logger.info(f"  Train Loss: {train_loss:.4f}")
                logger.info(f"  Val Loss: {val_loss:.4f}")
                logger.info(f"  Val RMSE: {val_rmse:.4f}")
                logger.info(f"  Val MAE: {val_mae:.4f}")
                logger.info(f"  Val R2: {val_metrics['r2']:.4f}")
                logger.info(f"  Val Pearson: {val_metrics['pearson_r']:.4f}")
                logger.info(f"  Val Spearman: {val_metrics['spearman_r']:.4f}")
                
                # Save best model
                if np.isfinite(val_rmse) and val_rmse < self.best_val_rmse:
                    self.best_val_rmse = val_rmse
                    self._save_model(is_best=True)
                    self.best_epoch = epoch + 1
                    self.best_model_metrics = dict(val_metrics)
                    logger.info(f"New best RMSE: {val_rmse:.4f}")
                
                self.metrics_history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    **val_metrics,
                })
        
        except KeyboardInterrupt:
            logger.info("Training interrupted")
        
        finally:
            # Save final model and metrics
            self._save_model(is_best=False)
            
            metrics_df = pd.DataFrame(self.metrics_history)
            metrics_path = self._artifact_path('regression_metrics', '.csv')
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Metrics saved: {metrics_path}")

            final_val_metrics, final_val_predictions = self._evaluate_saved_model(
                'final_model',
                self.val_loader,
                criterion,
                split_name='val',
            )
            final_test_metrics, final_test_predictions = self._evaluate_saved_model(
                'final_model',
                self.test_loader,
                criterion,
                split_name='test',
            )
            best_val_metrics, best_val_predictions = self._evaluate_saved_model(
                'best_model',
                self.val_loader,
                criterion,
                split_name='val',
            )
            best_test_metrics, best_test_predictions = self._evaluate_saved_model(
                'best_model',
                self.test_loader,
                criterion,
                split_name='test',
            )

            summary = {
                'dataset_name': self.dataset_name,
                'output_dir': str(self.output_dir),
                'logs_dir': str(self.logs_dir),
                'best_epoch': self.best_epoch,
                'best_val_rmse': self.best_val_rmse,
                'loaded_csv_files': getattr(self.data_module, 'loaded_csv_files', []),
                'split_sizes': {
                    'train': len(self.train_loader.dataset) if self.train_loader is not None else 0,
                    'val': len(self.val_loader.dataset) if self.val_loader is not None else 0,
                    'test': len(self.test_loader.dataset) if self.test_loader is not None else 0,
                },
                'best_model_validation': best_val_metrics,
                'best_model_test': best_test_metrics,
                'final_model_validation': final_val_metrics,
                'final_model_test': final_test_metrics,
            }
            summary_path = self._artifact_path('regression_summary', '.json')
            with open(summary_path, 'w', encoding='utf-8') as handle:
                json.dump(self._json_ready_dict(summary), handle, ensure_ascii=False, indent=2)
            logger.info(f"Summary saved: {summary_path}")

            if final_val_predictions is not None:
                final_val_predictions.to_csv(
                    self._artifact_path('final_model_val_predictions', '.csv'),
                    index=False,
                )
            if final_test_predictions is not None:
                final_test_predictions.to_csv(
                    self._artifact_path('final_model_test_predictions', '.csv'),
                    index=False,
                )
            if best_val_predictions is not None:
                best_val_predictions.to_csv(
                    self._artifact_path('best_model_val_predictions', '.csv'),
                    index=False,
                )
            if best_test_predictions is not None:
                best_test_predictions.to_csv(
                    self._artifact_path('best_model_test_predictions', '.csv'),
                    index=False,
                )
            
            logger.info("="*60)
            logger.info("Training complete")
            logger.info("="*60)
    
    def _train_epoch(self, epoch: int, optimizer, criterion, scaler):
        """Train one epoch."""
        self.model.train()

        total_loss = 0.0
        gradient_accumulation_steps = int(self.training_config['gradient_accumulation_steps'])
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch + 1}",
        ):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch.pop('label').view(-1)
            
            # Forward pass
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                predictions = self.model(**batch).view(-1)
                raw_loss = criterion(predictions, labels)
            loss = raw_loss / gradient_accumulation_steps
            # Accumulate metrics
            total_loss += raw_loss.item()

            # Backward pass
            if self.use_grad_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = (
                (batch_idx + 1) % gradient_accumulation_steps == 0
                or (batch_idx + 1) == len(self.train_loader)
            )
            if should_step:
                if self.use_grad_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    float(self.training_config['max_grad_norm']),
                )
                # Optimizer step
                if self.use_grad_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def _evaluate_loader(
        self,
        loader,
        criterion,
        split_name: str,
        save_predictions: bool = True,
    ):
        """Run evaluation on one split and optionally collect predictions."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_smiles = []
        total_samples = 0

        if loader is None:
            return {'split': split_name, 'loss': float('nan'), 'num_samples': 0}, None
        
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch.pop('label').view(-1)
            
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                predictions = self.model(**batch).view(-1)
                loss = criterion(predictions, labels)
            
            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_preds.extend(predictions.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())
        
        dataset = getattr(loader, 'dataset', None)
        if save_predictions and dataset is not None and hasattr(dataset, 'smiles_list'):
            all_smiles = list(dataset.smiles_list)
        
        all_preds = np.array(all_preds, dtype=float)
        all_labels = np.array(all_labels, dtype=float)
        label_scaler = getattr(loader.dataset, "label_scaler", None)
        if label_scaler is not None:
            all_preds = label_scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
            all_labels = label_scaler.inverse_transform(all_labels.reshape(-1, 1)).flatten()

        metrics = self._compute_regression_metrics(all_labels, all_preds)
        metrics['split'] = split_name
        metrics['loss'] = float(total_loss / total_samples) if total_samples else float('nan')
        metrics['num_samples'] = int(len(all_labels))

        predictions_df = None
        if save_predictions:
            prediction_rows = {
                'smiles': all_smiles[:len(all_labels)] if all_smiles else [''] * len(all_labels),
                'y_true': all_labels,
                'y_pred': all_preds,
                'error': all_preds - all_labels,
                'abs_error': np.abs(all_preds - all_labels),
                'squared_error': np.square(all_preds - all_labels),
            }
            predictions_df = pd.DataFrame(prediction_rows)

        return metrics, predictions_df

    def _load_saved_model_weights(self, model_subdir: str) -> bool:
        """Load a saved checkpoint into the current model for evaluation."""
        model_path = self.output_dir / model_subdir / "model.pt"
        if not model_path.exists():
            logger.warning(f"Saved model not found for evaluation: {model_path}")
            return False

        state_dict = torch.load(model_path, map_location='cpu')
        self._unwrap_model().load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        return True

    def _evaluate_saved_model(self, model_subdir: str, loader, criterion, split_name: str):
        """Evaluate a persisted model checkpoint on a split."""
        if loader is None or not self._load_saved_model_weights(model_subdir):
            return None, None

        return self._evaluate_loader(
            loader,
            criterion,
            split_name=f"{model_subdir}_{split_name}",
            save_predictions=True,
        )
    
    def _save_model(self, is_best: bool = False):
        """Save model checkpoint."""
        save_dir = self.output_dir / ("best_model" if is_best else "final_model")
        save_dir.mkdir(parents=True, exist_ok=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cpu_state_dict = {
            key: value.detach().cpu()
            for key, value in self._unwrap_model().state_dict().items()
        }
        torch.save(cpu_state_dict, save_dir / "model.pt")
        if self.tokenizer is not None:
            self.tokenizer.save(save_dir / "tokenizer.json")
        logger.info(f"Model saved: {save_dir}")


def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    config = build_config_from_args(args)
    dataset_name = infer_regression_dataset_name(
        csv_path=config['data']['csv_path'],
        auto_discover_agile=config['data']['auto_discover_agile'],
        agile_cell_line=config['data'].get('agile_cell_line'),
        agile_split=config['data'].get('agile_split'),
    )
    configure_logging(resolve_path(config['paths']['logs_dir']), dataset_name=dataset_name)
    
    logger.info("="*70)
    logger.info("STAGE 2: TRANSFECTION EFFICIENCY REGRESSION")
    logger.info("="*70)
    
    trainer = QwenRegressionTrainer(
        config=config
    )
    
    
    # Setup and train
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
