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
import json
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
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
from src.logging_utils import configure_logging
from config.pretrain_cli import build_config_from_args, build_parser
from src.pretrain_utils import compute_token_accuracy_stats, to_serializable
from src.qwen_utils import copy_qwen_support_files, ensure_required_dependencies
from src.training_common import (
    cleanup_distributed_training,
    distributed_barrier,
    is_main_process,
    maybe_wrap_model_for_multi_gpu,
    resolve_device_config,
    resolve_path,
    set_seed,
    setup_distributed_training,
    unwrap_data_parallel,
)

# Try to import transformers
try:
    from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup
except ImportError:
    AutoModelForCausalLM = None
    get_linear_schedule_with_warmup = None

logger = logging.getLogger(__name__)


def ensure_transformers():
    """Fail fast when training starts without required dependencies."""
    ensure_required_dependencies(
        {
            "transformers.AutoModelForCausalLM": AutoModelForCausalLM,
            "transformers.get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
        },
        "pip install transformers torch",
    )


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
            self.config.get('runtime', {}).get('gpus'),
            logger=logger,
        )
        self.distributed = setup_distributed_training(self.device, logger=logger)
        self.resume_from = resume_from
        
        logger.info(f"Device: {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"GPU ids: {self.gpu_ids if self.gpu_ids else 'CPU only'}")
        logger.info(f"Training seed: {self.config['training']['seed']}")
        set_seed(self.config['training']['seed'])
        
        # Initialize paths
        self.output_dir = resolve_path(self.config['paths']['output_dir'], PROJECT_ROOT)
        self.logs_dir = resolve_path(self.config['paths']['logs_dir'], PROJECT_ROOT)
        self.tokenizer_path = resolve_path(self.config['paths']['tokenizer_path'], PROJECT_ROOT)
        self.model_cache_dir = resolve_path(self.config['paths']['model_cache_dir'], PROJECT_ROOT)
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
        return unwrap_data_parallel(self.model)

    def _maybe_wrap_model_for_multi_gpu(self):
        """Wrap the model for multi-GPU training when requested."""
        self.model = maybe_wrap_model_for_multi_gpu(
            self.model,
            self.device,
            self.gpu_ids,
            distributed=self.distributed,
            logger=logger,
        )

    @property
    def is_main_process(self) -> bool:
        """Whether this process is responsible for logging and checkpoints."""
        return is_main_process()

    def _reduce_stats(self, stats: Dict[str, float]) -> Dict[str, float]:
        """Sum numeric statistics across ranks for distributed evaluation."""
        if not self.distributed.enabled:
            return stats

        keys = list(stats.keys())
        values = torch.tensor(
            [float(stats[key]) for key in keys],
            device=self.device,
            dtype=torch.float64,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        reduced = {}
        for key, value in zip(keys, values.tolist()):
            if isinstance(stats[key], int):
                reduced[key] = int(round(value))
            else:
                reduced[key] = value
        return reduced
    
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
        self.train_loader, self.val_loader = self.data_module.create_loaders(
            distributed=self.distributed.enabled,
            world_size=self.distributed.world_size,
            rank=self.distributed.rank,
        )
        
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
            self.model.config.use_cache = False
            gradient_checkpointing_kwargs = {'use_reentrant': False}
            try:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )
            except TypeError:
                self.model.gradient_checkpointing_enable()
            logger.info("  Gradient checkpointing enabled")

        self._maybe_wrap_model_for_multi_gpu()

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

        train_sampler = getattr(self.train_loader, 'sampler', None)
        if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(tqdm(self.train_loader)):
            if not isinstance(batch, dict):
                raise TypeError(
                    f"Expected DataLoader to yield a dict batch, got {type(batch).__name__}"
                )
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            should_step = (
                (batch_idx + 1) % gradient_accumulation_steps == 0
                or (batch_idx + 1) == len(self.train_loader)
            )
            sync_context = (
                self.model.no_sync
                if self.distributed.enabled and not should_step
                else nullcontext
            )

            with sync_context():
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
            batch_top1_correct, batch_top5_correct, batch_token_count = compute_token_accuracy_stats(
                outputs.logits.detach(),
                batch['target_ids'],
                batch['attention_mask'],
            )
            total_top1_correct += batch_top1_correct
            total_top5_correct += batch_top5_correct
            total_tokens += batch_token_count
            total_sequences += batch['input_ids'].size(0)

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
                if self.current_step % logging_steps == 0 and self.is_main_process:
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
                        if self.is_main_process:
                            logger.info(f"New best validation loss: {val_metrics['val_loss']:.4f}")

                if self.current_step % self.config['training']['save_steps'] == 0:
                    self.save_checkpoint()
        
        # Epoch metrics
        reduced_stats = self._reduce_stats(
            {
                'total_loss': total_loss,
                'total_top1_correct': total_top1_correct,
                'total_top5_correct': total_top5_correct,
                'total_tokens': total_tokens,
                'total_sequences': total_sequences,
                'num_batches': len(self.train_loader),
            }
        )
        epoch_loss = reduced_stats['total_loss'] / reduced_stats['num_batches']
        epoch_perplexity = np.exp(epoch_loss)
        
        return {
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'train_perplexity': epoch_perplexity,
            'train_token_accuracy': (
                reduced_stats['total_top1_correct'] / reduced_stats['total_tokens']
                if reduced_stats['total_tokens']
                else float('nan')
            ),
            'train_top5_token_accuracy': (
                reduced_stats['total_top5_correct'] / reduced_stats['total_tokens']
                if reduced_stats['total_tokens']
                else float('nan')
            ),
            'train_token_count': reduced_stats['total_tokens'],
            'train_sequence_count': reduced_stats['total_sequences'],
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
            batch_top1_correct, batch_top5_correct, batch_token_count = compute_token_accuracy_stats(
                outputs.logits.detach(),
                batch['target_ids'],
                batch['attention_mask'],
            )
            total_top1_correct += batch_top1_correct
            total_top5_correct += batch_top5_correct
            total_tokens += batch_token_count
        
        reduced_stats = self._reduce_stats(
            {
                'total_loss': total_loss,
                'total_samples': total_samples,
                'total_top1_correct': total_top1_correct,
                'total_top5_correct': total_top5_correct,
                'total_tokens': total_tokens,
            }
        )

        avg_val_loss = (
            reduced_stats['total_loss'] / reduced_stats['total_samples']
            if reduced_stats['total_samples']
            else float('nan')
        )
        val_perplexity = (
            np.exp(avg_val_loss) if reduced_stats['total_samples'] else float('nan')
        )
        val_metrics = {
            'val_loss': avg_val_loss,
            'val_perplexity': val_perplexity,
            'val_token_accuracy': (
                reduced_stats['total_top1_correct'] / reduced_stats['total_tokens']
                if reduced_stats['total_tokens']
                else float('nan')
            ),
            'val_top5_token_accuracy': (
                reduced_stats['total_top5_correct'] / reduced_stats['total_tokens']
                if reduced_stats['total_tokens']
                else float('nan')
            ),
            'val_token_count': reduced_stats['total_tokens'],
            'val_sequence_count': reduced_stats['total_samples'],
        }
        
        if self.is_main_process:
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
        if not self.is_main_process:
            distributed_barrier()
            return

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
        distributed_barrier()

    def _copy_qwen_support_files(self, target_dir: Path):
        """Copy trust_remote_code support files so local checkpoints are reloadable."""
        copy_qwen_support_files(self.model_source_dir, target_dir)
    
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
                if self.is_main_process:
                    logger.info(f"\nEpoch {epoch + 1} Summary:")
                    logger.info(f"  Train Loss: {epoch_metrics['train_loss']:.4f}")
                    logger.info(f"  Train Perplexity: {epoch_metrics['train_perplexity']:.4f}")
                    logger.info(f"  Train Token Acc: {epoch_metrics['train_token_accuracy']:.4f}")
                    logger.info(f"  Val Loss: {epoch_metrics['val_loss']:.4f}")
                    logger.info(f"  Val Perplexity: {epoch_metrics['val_perplexity']:.4f}")
                    logger.info(f"  Val Token Acc: {epoch_metrics['val_token_accuracy']:.4f}")
                    logger.info(f"  Val Top5 Token Acc: {epoch_metrics['val_top5_token_accuracy']:.4f}")
        
        except KeyboardInterrupt:
            if self.is_main_process:
                logger.info("Training interrupted by user")
        
        finally:
            if self.is_main_process:
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
                    json.dump(to_serializable(summary), handle, ensure_ascii=False, indent=2)
                logger.info(f"Summary saved: {summary_path}")

                logger.info("="*60)
                logger.info("Training complete")
                logger.info("="*60)
            distributed_barrier()
            cleanup_distributed_training()


def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()
    config = build_config_from_args(args)
    configure_logging(resolve_path(config['paths']['logs_dir'], PROJECT_ROOT))
    
    # Create trainer
    trainer = QwenSMILESPretrainer(config, args.resume_from)
    
    # Setup and train
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
