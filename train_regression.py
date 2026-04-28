"""
Stage 2: Training Script for Regression Task (Transfection Efficiency)

Supports:
1. Auto-discovery of AGILE datasets (AGILE/*/*/test.csv)
2. Manual CSV path: CSV with columns [SMILES, TARGET]

This script will:
1. Load pretrained Qwen + SMILES tokenizer from Stage 1
2. Initialize regression head
3. Fine-tune on transfection efficiency labels
4. Evaluate with RMSE, MAE, Pearson correlation
"""

import logging
import random
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


def configure_logging(logs_dir: Path):
    """Configure file and console logging once CLI args are known."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / 'regression_training.log'),
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
        print("ERROR: transformers not installed. Run: pip install transformers torch")
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


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for regression fine-tuning."""
    parser = argparse.ArgumentParser(
        description='Fine-tune Qwen for transfection efficiency regression',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover AGILE datasets (default)
  python train_regression.py
  
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
    parser.add_argument('--csv', default=None,
                        help='Path to regression CSV (optional, auto-discovers AGILE if not provided)')
    parser.add_argument('--output-dir', '--output',
                        dest='output_dir',
                        default='models/qwen_1.8b_smiles_regression',
                        help='Output directory')
    parser.add_argument('--logs-dir', default='logs',
                        help='Directory for regression logs and metrics')
    parser.add_argument('--no-auto-discover', action='store_true',
                        help='Disable AGILE auto-discovery')

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

        self.pretrained_model_path = resolve_path(self.paths_config['pretrained_model_path'])
        self.tokenizer_path = resolve_path(self.paths_config['tokenizer_path'])
        self.output_dir = resolve_path(self.paths_config['output_dir'])
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
        
        self.device, self.gpu_ids = resolve_device_config(self.runtime_config.get('gpus'))
        logger.info(f"Device: {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"GPU ids: {self.gpu_ids if self.gpu_ids else 'CPU only'}")
        logger.info(f"Training seed: {self.training_config['seed']}")
        set_seed(self.training_config['seed'])
        
        # Components
        self.tokenizer = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.amp_dtype = None
        self.use_amp = self.model_config['use_amp']
        self.use_grad_scaler = self.model_config['use_grad_scaler']
        
        # Training state
        self.best_val_rmse = float('inf')

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
        )

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
        
        metrics_history = []
        
        try:
            for epoch in range(self.training_config['num_epochs']):
                # Training
                train_loss = self._train_epoch(epoch, optimizer, criterion, scaler)
                
                # Validation
                val_loss, val_rmse, val_mae = self._validate(criterion)
                
                logger.info(f"Epoch {epoch+1}/{self.training_config['num_epochs']}")
                logger.info(f"  Train Loss: {train_loss:.4f}")
                logger.info(f"  Val Loss: {val_loss:.4f}")
                logger.info(f"  Val RMSE: {val_rmse:.4f}")
                logger.info(f"  Val MAE: {val_mae:.4f}")
                
                # Save best model
                if val_rmse < self.best_val_rmse:
                    self.best_val_rmse = val_rmse
                    self._save_model(is_best=True)
                    logger.info(f"New best RMSE: {val_rmse:.4f}")
                
                metrics_history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_rmse': val_rmse,
                    'val_mae': val_mae,
                })
        
        except KeyboardInterrupt:
            logger.info("Training interrupted")
        
        finally:
            # Save final model and metrics
            self._save_model(is_best=False)
            
            metrics_df = pd.DataFrame(metrics_history)
            metrics_path = self.logs_dir / 'regression_metrics.csv'
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Metrics saved: {metrics_path}")
            
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
    def _validate(self, criterion):
        """Validate and compute metrics."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch.pop('label').view(-1)
            
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                predictions = self.model(**batch).view(-1)
                loss = criterion(predictions, labels)
            
            total_loss += loss.item()
            all_preds.extend(predictions.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        label_scaler = getattr(self.val_loader.dataset, "label_scaler", None)
        if label_scaler is not None:
            all_preds = label_scaler.inverse_transform(all_preds.reshape(-1, 1)).flatten()
            all_labels = label_scaler.inverse_transform(all_labels.reshape(-1, 1)).flatten()
        
        rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
        mae = np.mean(np.abs(all_preds - all_labels))
        
        return total_loss / len(self.val_loader), rmse, mae
    
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
    configure_logging(resolve_path(config['paths']['logs_dir']))
    
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
