"""
Training Script for SMILES Causal Language Modeling with Qwen-1.8B

This script implements a complete training pipeline with:
- Mixed precision training (float16)
- Gradient accumulation for single GPU
- Checkpointing and resume capability
- Comprehensive logging
- Validation and perplexity tracking
"""

import logging
import random
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import pandas as pd

try:
    from modelscope import snapshot_download
except ImportError:
    snapshot_download = None

# Import custom modules
PROJECT_ROOT = Path(__file__).parent
PROJECT_ROOT.joinpath("logs").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT))
from src.tokenizer import SMILESTokenizer
from src.dataset import SMILESDataModule

# Try to import transformers
try:
    from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup
except ImportError:
    AutoModelForCausalLM = None
    get_linear_schedule_with_warmup = None

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
    """Configure file and console logging once the CLI args are known."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / 'training.log'),
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


def _compute_token_accuracy_stats(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[int, int, int]:
    """Count top-1/top-5 correct next-token predictions on non-padding positions."""
    valid_mask = attention_mask.bool()
    token_count = int(valid_mask.sum().item())
    if token_count == 0:
        return 0, 0, 0

    predictions = logits.argmax(dim=-1)
    correct_top1 = int(((predictions == target_ids) & valid_mask).sum().item())

    topk = min(5, logits.size(-1))
    topk_predictions = torch.topk(logits, k=topk, dim=-1).indices
    correct_top5 = int(
        (topk_predictions.eq(target_ids.unsqueeze(-1)).any(dim=-1) & valid_mask).sum().item()
    )

    return correct_top1, correct_top5, token_count


def _to_serializable(value):
    """Convert numpy scalars inside nested structures into builtin Python values."""
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


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


def build_config_from_args(args: argparse.Namespace) -> Dict:
    """Build the nested training config from CLI arguments."""
    return {
        'model': {
            'model_name': args.model_name,
            'max_seq_length': args.max_seq_length,
        },
        'data': {
            'csv_paths': args.csv_paths,
            'validation_split': args.validation_split,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'deduplicate': args.deduplicate,
        },
        'training': {
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'warmup_steps': args.warmup_steps,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'max_grad_norm': args.max_grad_norm,
            'eval_steps': args.eval_steps,
            'save_steps': args.save_steps,
            'logging_steps': args.logging_steps,
            'seed': args.seed,
        },
        'optimization': {
            'mixed_precision': args.mixed_precision,
            'gradient_checkpointing': args.gradient_checkpointing,
            'use_amp': args.use_amp,
            'use_grad_scaler': args.use_grad_scaler,
        },
        'paths': {
            'output_dir': args.output_dir,
            'logs_dir': args.logs_dir,
            'tokenizer_path': args.tokenizer_path,
            'model_cache_dir': args.model_cache_dir,
        },
        'runtime': {
            'gpus': args.gpus,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for pretraining."""
    parser = argparse.ArgumentParser(
        description='Train Qwen-1.8B for SMILES modeling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--resume-from', default=None,
                        help='Path to checkpoint directory to resume from')

    parser.add_argument('--model-name', default='qwen/Qwen-1_8B',
                        help='Base model name used by ModelScope download')
    parser.add_argument('--max-seq-length', type=int, default=256,
                        help='Maximum SMILES sequence length')

    parser.add_argument('--csv-paths', nargs='+', default=['data/test_lipids.csv'],
                        help='One or more pretraining CSV paths')
    parser.add_argument('--validation-split', type=float, default=0.15,
                        help='Validation split ratio')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Training batch size')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of DataLoader workers')
    parser.add_argument('--deduplicate', type=str2bool, nargs='?', const=True, default=False,
                        help='Whether to deduplicate SMILES samples before training')

    parser.add_argument('--num-epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                        help='Optimizer learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='AdamW weight decay')
    parser.add_argument('--warmup-steps', type=int, default=100,
                        help='Scheduler warmup steps')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--eval-steps', type=int, default=500,
                        help='Run validation every N optimizer steps')
    parser.add_argument('--save-steps', type=int, default=500,
                        help='Save checkpoint every N optimizer steps')
    parser.add_argument('--logging-steps', type=int, default=50,
                        help='Log training metrics every N optimizer steps')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    parser.add_argument('--mixed-precision', choices=['none', 'fp16', 'bf16'], default='bf16',
                        help='Mixed precision mode used during training')
    parser.add_argument('--gradient-checkpointing', type=str2bool, nargs='?', const=True, default=True,
                        help='Enable gradient checkpointing')
    parser.add_argument('--use-amp', type=str2bool, nargs='?', const=True, default=True,
                        help='Compatibility flag kept for old configs; runtime precision is still derived from --mixed-precision')
    parser.add_argument('--use-grad-scaler', type=str2bool, nargs='?', const=True, default=True,
                        help='Compatibility flag kept for old configs; scaler usage is still derived from --mixed-precision')

    parser.add_argument('--output-dir', default='models/qwen_1.8b_smiles_pretrained',
                        help='Directory for checkpoints and final model')
    parser.add_argument('--logs-dir', default='logs',
                        help='Directory for training logs and metrics')
    parser.add_argument('--tokenizer-path', default='models/qwen_1.8b_smiles_pretrained/tokenizer.json',
                        help='Path for saving the main tokenizer artifact')
    parser.add_argument('--model-cache-dir', default='models/cache/qwen-1.8b',
                        help='ModelScope cache directory for downloaded base weights')
    parser.add_argument('--gpus', default=None,
                        help="GPU selection: '0' for single GPU, '0,1' for multi-GPU, 'all' for all visible GPUs, or 'cpu'")

    return parser


def ensure_transformers():
    """Fail fast when training starts without required dependencies."""
    if AutoModelForCausalLM is None or get_linear_schedule_with_warmup is None:
        print("ERROR: transformers not installed. Run: pip install transformers torch")
        sys.exit(1)


class QwenSMILESPretrainer:
    """Main training class for SMILES causal language modeling."""
    
    def __init__(self, config: Dict, resume_from: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dictionary
            resume_from: Path to checkpoint to resume from
        """
        self.config = config
        self.device, self.gpu_ids = resolve_device_config(
            self.config.get('runtime', {}).get('gpus')
        )
        self.resume_from = resume_from
        
        logger.info(f"Device: {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"GPU ids: {self.gpu_ids if self.gpu_ids else 'CPU only'}")
        logger.info(f"Training seed: {self.config['training']['seed']}")
        set_seed(self.config['training']['seed'])
        
        # Initialize paths
        self.output_dir = resolve_path(self.config['paths']['output_dir'])
        self.logs_dir = resolve_path(self.config['paths']['logs_dir'])
        self.tokenizer_path = resolve_path(self.config['paths']['tokenizer_path'])
        self.model_cache_dir = resolve_path(self.config['paths']['model_cache_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.data_module = None
        self.amp_dtype = None
        self.model_source_dir = None
        self.use_amp = bool(self.config['optimization'].get('use_amp', False))
        self.use_grad_scaler = bool(self.config['optimization'].get('use_grad_scaler', False))
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.metrics_df = None
        self.validation_history = []
        self.best_val_metrics = None

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
    
    def setup(self):
        """Initialize tokenizer, data module, and model."""
        ensure_transformers()
        logger.info("="*60)
        logger.info("SETUP PHASE")
        logger.info("="*60)
        
        # Initialize SMILES tokenizer
        logger.info("\n[1/4] Initializing SMILES tokenizer...")
        self.tokenizer = SMILESTokenizer(max_length=self.config['model']['max_seq_length'])
        
        # Initialize data module
        logger.info("[2/4] Initializing data module...")
        self.data_module = SMILESDataModule(
            csv_paths=self._get_csv_paths(),
            tokenizer=self.tokenizer,
            max_length=self.config['model']['max_seq_length'],
            batch_size=self.config['data']['batch_size'],
            validation_split=self.config['data']['validation_split'],
            deduplicate=self.config['data']['deduplicate'],
            num_workers=self.config['data']['num_workers'],
        )
        
        # Setup data module (loads CSV and builds vocab)
        self.data_module.setup()
        
        # Create data loaders
        self.train_loader, self.val_loader = self.data_module.create_loaders()
        
        logger.info(f"  Train batches: {len(self.train_loader)}")
        logger.info(f"  Val batches: {len(self.val_loader)}")
        
        # Save tokenizer
        self.tokenizer.save(str(self.tokenizer_path))
        logger.info(f"  Tokenizer saved to {self.tokenizer_path}")
        
        # Load Qwen model
        logger.info("[3/4] Loading Qwen model...")
        model_name = self.config['model']['model_name']
        logger.info(f"  Model: {model_name}")
        mixed_precision = self.config['optimization'].get('mixed_precision', 'none')

        if torch.cuda.is_available() and mixed_precision == 'fp16':
            self.amp_dtype = torch.float16
        elif torch.cuda.is_available() and mixed_precision == 'bf16':
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = None

        self.use_amp = self.amp_dtype is not None
        self.use_grad_scaler = self.amp_dtype == torch.float16
        
        try:
            if self.resume_from:
                model_dir = str(Path(self.resume_from))
                logger.info(f"  Resuming model weights from: {model_dir}")
            else:
                if snapshot_download is None:
                    raise ImportError(
                        "modelscope is required to download the base model. "
                        "Install it with: pip install modelscope"
                    )
                model_dir = snapshot_download(
                    model_name,
                    revision="v1.0.0",
                    cache_dir=str(self.model_cache_dir)
                )

            if self.use_grad_scaler:
                dtype = torch.float32
            else:
                dtype = self.amp_dtype or torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=dtype,
                device_map=None,
                trust_remote_code=True
            )
            self.model_source_dir = Path(model_dir)

            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model = self.model.to(device=self.device, dtype=dtype)
            self.model.config.pad_token_id = self.tokenizer.token2id['[PAD]']
            self.model.config.eos_token_id = self.tokenizer.token2id['[EOS]']
            self.model.config.bos_token_id = self.tokenizer.token2id['[CLS]']

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error("Try: pip install modelscope transformers torch")
            sys.exit(1)
        
        if self.config['optimization']['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable()
            logger.info("  Gradient checkpointing enabled")

        self._maybe_enable_data_parallel()

        model_dtypes = sorted({str(param.dtype) for param in self.model.parameters()})
        logger.info(f"  Mixed precision mode: {mixed_precision}")
        logger.info(f"  AMP enabled: {self.use_amp}")
        logger.info(f"  GradScaler enabled: {self.use_grad_scaler}")
        logger.info(f"  Model parameter dtypes: {', '.join(model_dtypes)}")
        
        logger.info(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"  Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        # Setup optimizer
        logger.info("[4/4] Setting up optimizer and scheduler...")
        self._setup_optimizer()

        # Initialize mixed precision scaler
        self.scaler = GradScaler(enabled=self.use_grad_scaler)

        if self.resume_from:
            self._restore_training_state(self.resume_from)
        
        logger.info("="*60)
        logger.info("Setup complete\n")
    
    def _get_csv_paths(self) -> List[str]:
        """Get absolute paths to CSV files."""
        csv_paths = []
        base_dir = Path(__file__).parent # Go to root directory
        for csv_name in self.config['data']['csv_paths']:
            csv_path = base_dir / csv_name
            
            if not csv_path.exists():
                logger.warning(f"CSV not found: {csv_path}")
            csv_paths.append(str(csv_path))
        
        return csv_paths
    
    def _setup_optimizer(self):
        """Initialize optimizer and scheduler."""
        # Filter parameters that require gradients
        optimizer_grouped_parameters = [
            {
                'params': [p for p in self.model.parameters() if p.requires_grad],
                'weight_decay': self.config['training']['weight_decay'],
            }
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config['training']['learning_rate'],
        )
        
        # Calculate total training steps
        num_epochs = self.config['training']['num_epochs']
        gradient_accumulation_steps = self.config['training']['gradient_accumulation_steps']
        steps_per_epoch = max(1, (len(self.train_loader) + gradient_accumulation_steps - 1) // gradient_accumulation_steps)
        num_training_steps = max(1, steps_per_epoch * num_epochs)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=num_training_steps,
        )
        
        logger.info(f"  Total training steps: {num_training_steps}")
        logger.info(f"  Learning rate: {self.config['training']['learning_rate']}")
        logger.info(f"  Warmup steps: {self.config['training']['warmup_steps']}")

    def _restore_training_state(self, checkpoint_dir: str):
        """Restore optimizer/scheduler/scaler state from a checkpoint directory."""
        state_path = Path(checkpoint_dir) / 'training_state.pt'
        if not state_path.exists():
            logger.warning(f"Training state not found for resume: {state_path}")
            return

        state = torch.load(state_path, map_location='cpu')
        self.current_step = state.get('step', 0)
        self.current_epoch = state.get('epoch', 0)
        self.best_val_loss = state.get('best_val_loss', float('inf'))
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.scheduler.load_state_dict(state['scheduler_state'])
        scaler_state = state.get('scaler_state')
        if scaler_state and self.scaler is not None:
            self.scaler.load_state_dict(scaler_state)
        logger.info(
            f"  Restored training state from {state_path} "
            f"(epoch={self.current_epoch}, step={self.current_step})"
        )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        total_loss = 0.0
        total_top1_correct = 0
        total_top5_correct = 0
        total_tokens = 0
        total_sequences = 0
        gradient_accumulation_steps = self.config['training']['gradient_accumulation_steps']
        logging_steps = self.config['training']['logging_steps']
        
        logger.info(f"\nEpoch {epoch + 1}/{self.config['training']['num_epochs']}")

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(tqdm(self.train_loader)):
            if not isinstance(batch, dict):
                raise TypeError(
                    f"Expected DataLoader to yield a dict batch, got {type(batch).__name__}"
                )
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['target_ids'],
                )
                raw_loss = outputs.loss

            # Scale loss for gradient accumulation
            loss = raw_loss / gradient_accumulation_steps
            
            # Backward pass
            if self.use_grad_scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Accumulate metrics
            total_loss += raw_loss.item()
            batch_top1_correct, batch_top5_correct, batch_token_count = _compute_token_accuracy_stats(
                outputs.logits.detach(),
                batch['target_ids'],
                batch['attention_mask'],
            )
            total_top1_correct += batch_top1_correct
            total_top5_correct += batch_top5_correct
            total_tokens += batch_token_count
            total_sequences += batch['input_ids'].size(0)
            
            # Update weights every accumulation step
            should_step = (
                (batch_idx + 1) % gradient_accumulation_steps == 0
                or (batch_idx + 1) == len(self.train_loader)
            )
            if should_step:
                # Clip gradients
                if self.use_grad_scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )
                
                # Optimizer step
                if self.use_grad_scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                
                self.current_step += 1

                # Logging / validation / checkpoint are step-based so they only fire once.
                if self.current_step % logging_steps == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    perplexity = np.exp(avg_loss)
                    logger.info(
                        f"  Step {self.current_step} | Loss: {avg_loss:.4f} | "
                        f"  Perplexity: {perplexity:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                    )

                if self.current_step % self.config['training']['eval_steps'] == 0:
                    val_metrics = self.validate(record_history=True, context='step_eval')
                    self.model.train()  # Resume training mode

                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.best_val_metrics = {
                            'epoch': self.current_epoch + 1,
                            'step': self.current_step,
                            **val_metrics,
                        }
                        self.save_checkpoint(is_best=True)
                        logger.info(f"New best validation loss: {val_metrics['val_loss']:.4f}")

                if self.current_step % self.config['training']['save_steps'] == 0:
                    self.save_checkpoint()
        
        # Epoch metrics
        epoch_loss = total_loss / len(self.train_loader)
        epoch_perplexity = np.exp(epoch_loss)
        
        return {
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'train_perplexity': epoch_perplexity,
            'train_token_accuracy': (
                total_top1_correct / total_tokens if total_tokens else float('nan')
            ),
            'train_top5_token_accuracy': (
                total_top5_correct / total_tokens if total_tokens else float('nan')
            ),
            'train_token_count': total_tokens,
            'train_sequence_count': total_sequences,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'optimizer_steps': self.current_step,
        }
    
    @torch.no_grad()
    def validate(
        self,
        record_history: bool = True,
        context: str = "epoch_end",
        epoch: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        total_top1_correct = 0
        total_top5_correct = 0
        total_tokens = 0
        
        for batch in tqdm(self.val_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['target_ids'],
                )
                loss = outputs.loss
            
            total_loss += loss.item() * batch['input_ids'].size(0)
            total_samples += batch['input_ids'].size(0)
            batch_top1_correct, batch_top5_correct, batch_token_count = _compute_token_accuracy_stats(
                outputs.logits.detach(),
                batch['target_ids'],
                batch['attention_mask'],
            )
            total_top1_correct += batch_top1_correct
            total_top5_correct += batch_top5_correct
            total_tokens += batch_token_count
        
        avg_val_loss = total_loss / total_samples if total_samples else float('nan')
        val_perplexity = np.exp(avg_val_loss) if total_samples else float('nan')
        val_metrics = {
            'val_loss': avg_val_loss,
            'val_perplexity': val_perplexity,
            'val_token_accuracy': (
                total_top1_correct / total_tokens if total_tokens else float('nan')
            ),
            'val_top5_token_accuracy': (
                total_top5_correct / total_tokens if total_tokens else float('nan')
            ),
            'val_token_count': total_tokens,
            'val_sequence_count': total_samples,
        }
        
        logger.info(
            "  Validation | Loss: %.4f | Perplexity: %.4f | Token Acc: %.4f | Top5 Acc: %.4f",
            val_metrics['val_loss'],
            val_metrics['val_perplexity'],
            val_metrics['val_token_accuracy'],
            val_metrics['val_top5_token_accuracy'],
        )

        if record_history:
            self.validation_history.append(
                {
                    'context': context,
                    'epoch': (epoch + 1) if epoch is not None else (self.current_epoch + 1),
                    'step': self.current_step,
                    **val_metrics,
                }
            )
        
        return val_metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        save_dir = self.output_dir / f"checkpoint-{self.current_step}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self._unwrap_model().save_pretrained(str(save_dir))
        self._copy_qwen_support_files(save_dir)
        if self.tokenizer is not None:
            self.tokenizer.save(str(save_dir / "tokenizer.json"))
        
        # Save training state
        state = {
            'step': self.current_step,
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'scaler_state': self.scaler.state_dict() if self.scaler is not None else None,
        }
        torch.save(state, str(save_dir / 'training_state.pt'))
        
        logger.info(f"Checkpoint saved: {save_dir}")
        
        # Copy as best if applicable
        if is_best:
            best_dir = self.output_dir / "best_model"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(save_dir, best_dir)
            logger.info(f"Best model updated: {best_dir}")

    def _copy_qwen_support_files(self, target_dir: Path):
        """Copy trust_remote_code support files so local checkpoints are reloadable."""
        if self.model_source_dir is None or not self.model_source_dir.exists():
            return

        support_files = [
            "modeling_qwen.py",
            "configuration_qwen.py",
            "qwen_generation_utils.py",
            "tokenization_qwen.py",
            "cpp_kernels.py",
        ]

        for filename in support_files:
            source = self.model_source_dir / filename
            destination = target_dir / filename
            if source.exists() and not destination.exists():
                shutil.copy2(source, destination)
    
    def train(self):
        """Main training loop."""
        logger.info("="*60)
        logger.info("TRAINING PHASE")
        logger.info("="*60)
        
        metrics_history = []
        
        try:
            start_epoch = self.current_epoch
            for epoch in range(start_epoch, self.config['training']['num_epochs']):
                self.current_epoch = epoch
                
                # Train epoch
                epoch_metrics = self.train_epoch(epoch)
                metrics_history.append(epoch_metrics)
                
                # Validate at end of epoch
                val_metrics = self.validate(record_history=True, context='epoch_end', epoch=epoch)
                epoch_metrics.update(val_metrics)
                
                # Check for new best
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.best_val_metrics = {
                        'epoch': epoch + 1,
                        'step': self.current_step,
                        **val_metrics,
                    }
                    self.save_checkpoint(is_best=True)
                
                # Log epoch summary
                logger.info(f"\nEpoch {epoch + 1} Summary:")
                logger.info(f"  Train Loss: {epoch_metrics['train_loss']:.4f}")
                logger.info(f"  Train Perplexity: {epoch_metrics['train_perplexity']:.4f}")
                logger.info(f"  Train Token Acc: {epoch_metrics['train_token_accuracy']:.4f}")
                logger.info(f"  Val Loss: {epoch_metrics['val_loss']:.4f}")
                logger.info(f"  Val Perplexity: {epoch_metrics['val_perplexity']:.4f}")
                logger.info(f"  Val Token Acc: {epoch_metrics['val_token_accuracy']:.4f}")
                logger.info(f"  Val Top5 Token Acc: {epoch_metrics['val_top5_token_accuracy']:.4f}")
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            # Save final model
            final_dir = self.output_dir / "final_model"
            if final_dir.exists():
                shutil.rmtree(final_dir)
            self._unwrap_model().save_pretrained(str(final_dir))
            self._copy_qwen_support_files(final_dir)
            if self.tokenizer is not None:
                self.tokenizer.save(str(final_dir / "tokenizer.json"))
            logger.info(f"Final model saved: {final_dir}")
            
            # Save metrics
            self.metrics_df = pd.DataFrame(metrics_history)
            metrics_path = self.logs_dir / "pretrain_metrics.csv"
            self.metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Metrics saved: {metrics_path}")

            validation_metrics_path = self.logs_dir / "pretrain_validation_metrics.csv"
            pd.DataFrame(self.validation_history).to_csv(validation_metrics_path, index=False)
            logger.info(f"Validation metrics saved: {validation_metrics_path}")

            summary = {
                'best_val_loss': self.best_val_loss,
                'best_val_metrics': self.best_val_metrics,
                'completed_epochs': len(metrics_history),
                'completed_steps': self.current_step,
                'train_csv_paths': self.config['data']['csv_paths'],
                'metrics_path': str(metrics_path),
                'validation_metrics_path': str(validation_metrics_path),
            }
            summary_path = self.logs_dir / "pretrain_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as handle:
                json.dump(_to_serializable(summary), handle, ensure_ascii=False, indent=2)
            logger.info(f"Summary saved: {summary_path}")
            
            logger.info("="*60)
            logger.info("Training complete")
            logger.info("="*60)


def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    config = build_config_from_args(args)
    configure_logging(resolve_path(config['paths']['logs_dir']))
    
    # Create trainer
    trainer = QwenSMILESPretrainer(config, args.resume_from)
    
    # Setup and train
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
