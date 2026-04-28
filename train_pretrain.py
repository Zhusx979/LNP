"""
Training Script for SMILES Causal Language Modeling with Qwen-1.8B

This script implements a complete training pipeline with:
- Mixed precision training (float16)
- Gradient accumulation for single GPU
- Checkpointing and resume capability
- Comprehensive logging
- Validation and perplexity tracking
"""

import json
import logging
import os
from modelscope import snapshot_download
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

# Import custom modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src.tokenizer import SMILESTokenizer
from src.dataset import SMILESDataModule

# Try to import transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
except ImportError:
    print("ERROR: transformers not installed. Run: pip install transformers torch")
    sys.exit(1)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class QwenSMILESPretrainer:
    """Main training class for SMILES causal language modeling."""
    
    def __init__(self, config_path: str, resume_from: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to config JSON file
            resume_from: Path to checkpoint to resume from
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resume_from = resume_from
        
        logger.info(f"Device: {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        # Initialize paths
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.logs_dir = Path(self.config['paths']['logs_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.data_module = None
        self.amp_dtype = None
        self.use_amp = Path(self.config['optimization']['use_amp'])
        self.use_grad_scaler = Path(self.config['optimization']['outuse_grad_scalerput_dir'])
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.metrics_df = None
    
    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    
    def setup(self):
        """Initialize tokenizer, data module, and model."""
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
        tokenizer_path = self.output_dir / "tokenizer.json"
        self.tokenizer.save(str(tokenizer_path))
        logger.info(f"  Tokenizer saved to {tokenizer_path}")
        
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
            model_dir = snapshot_download(
                model_name,
                revision="v1.0.0", 
                cache_dir=self.config['paths']['model_cache_dir']
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
        num_training_steps = len(self.train_loader) * num_epochs // gradient_accumulation_steps
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=num_training_steps,
        )
        
        logger.info(f"  Total training steps: {num_training_steps}")
        logger.info(f"  Learning rate: {self.config['training']['learning_rate']}")
        logger.info(f"  Warmup steps: {self.config['training']['warmup_steps']}")
    
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
        gradient_accumulation_steps = self.config['training']['gradient_accumulation_steps']
        logging_steps = self.config['training']['logging_steps']
        
        logger.info(f"\nEpoch {epoch + 1}/{self.config['training']['num_epochs']}")
        
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
                loss = outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            if self.use_grad_scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Accumulate metrics
            total_loss += loss.item()
            
            # Update weights every accumulation step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
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
            
            # Logging
            if (batch_idx + 1) % logging_steps == 0:
                avg_loss = total_loss / ((batch_idx + 1) / gradient_accumulation_steps)
                perplexity = np.exp(avg_loss)
                logger.info(
                    f"  Step {self.current_step} | Loss: {avg_loss:.4f} | "
                    f"  Perplexity: {perplexity:.4f} | LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )
            
            # Validation
            if self.current_step > 0 and self.current_step % self.config['training']['eval_steps'] == 0:
                val_loss = self.validate()
                self.model.train()  # Resume training mode
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)
                    logger.info(f"New best validation loss: {val_loss:.4f}")
            
            # Checkpoint
            if self.current_step > 0 and self.current_step % self.config['training']['save_steps'] == 0:
                self.save_checkpoint()
        
        # Epoch metrics
        epoch_loss = total_loss / len(self.train_loader)
        epoch_perplexity = np.exp(epoch_loss)
        
        return {
            'epoch': epoch,
            'train_loss': epoch_loss,
            'train_perplexity': epoch_perplexity,
        }
    
    @torch.no_grad()
    def validate(self) -> float:
        """
        Run validation.
        
        Returns:
            Validation loss
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
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
        
        avg_val_loss = total_loss / total_samples
        val_perplexity = np.exp(avg_val_loss)
        
        logger.info(f"  Validation | Loss: {avg_val_loss:.4f} | Perplexity: {val_perplexity:.4f}")
        
        return avg_val_loss
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        save_dir = self.output_dir / f"checkpoint-{self.current_step}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(save_dir))
        
        # Save training state
        state = {
            'step': self.current_step,
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'scaler_state': self.scaler.state_dict(),
        }
        torch.save(state, str(save_dir / 'training_state.pt'))
        
        logger.info(f"Checkpoint saved: {save_dir}")
        
        # Copy as best if applicable
        if is_best:
            best_dir = self.output_dir / "best_model"
            if best_dir.exists():
                import shutil
                shutil.rmtree(best_dir)
            import shutil
            shutil.copytree(save_dir, best_dir)
            logger.info(f"Best model updated: {best_dir}")
    
    def train(self):
        """Main training loop."""
        logger.info("="*60)
        logger.info("TRAINING PHASE")
        logger.info("="*60)
        
        metrics_history = []
        
        try:
            for epoch in range(self.config['training']['num_epochs']):
                self.current_epoch = epoch
                
                # Train epoch
                epoch_metrics = self.train_epoch(epoch)
                metrics_history.append(epoch_metrics)
                
                # Validate at end of epoch
                val_loss = self.validate()
                epoch_metrics['val_loss'] = val_loss
                epoch_metrics['val_perplexity'] = np.exp(val_loss)
                
                # Check for new best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)
                
                # Log epoch summary
                logger.info(f"\nEpoch {epoch + 1} Summary:")
                logger.info(f"  Train Loss: {epoch_metrics['train_loss']:.4f}")
                logger.info(f"  Train Perplexity: {epoch_metrics['train_perplexity']:.4f}")
                logger.info(f"  Val Loss: {val_loss:.4f}")
                logger.info(f"  Val Perplexity: {np.exp(val_loss):.4f}")
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        finally:
            # Save final model
            final_dir = self.output_dir / "final_model"
            if final_dir.exists():
                import shutil
                shutil.rmtree(final_dir)
            self.model.save_pretrained(str(final_dir))
            logger.info(f"Final model saved: {final_dir}")
            
            # Save metrics
            self.metrics_df = pd.DataFrame(metrics_history)
            metrics_path = self.logs_dir / "pretrain_metrics.csv"
            self.metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Metrics saved: {metrics_path}")
            
            logger.info("="*60)
            logger.info("Training complete")
            logger.info("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train Qwen-1.8B for SMILES modeling')
    parser.add_argument('--config', default='configs/config_pretrain.json',
                       help='Path to config file')
    parser.add_argument('--resume-from', default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Create trainer
    trainer = QwenSMILESPretrainer(args.config, args.resume_from)
    
    # Setup and train
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
