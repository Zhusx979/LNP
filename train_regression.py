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

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Import custom modules
sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.model_regression import QwenRegressionModel, RegressionDataModule
from src.tokenizer import SMILESTokenizer

try:
    from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup
except ImportError:
    print("ERROR: transformers not installed")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/regression_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class QwenRegressionTrainer:
    """Training class for transfection efficiency regression."""
    
    def __init__(
        self,
        config,
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
        self._validate_inputs(config.pretrained_model_path, config.tokenizer_path)
        self.config = config
        self.pretrained_model_path = config.pretrained_model_path
        self.tokenizer_path = config.tokenizer_path
        self.regression_csv = config.csv
        self.batch_size = config.batch
        self.output_dir = Path(config.output)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.auto_discover_agile = not config.no_auto_discover
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Components
        self.tokenizer = None
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.amp_dtype = None
        self.use_amp = config.use_amp
        self.use_grad_scaler = config.use_grad_scaler
        
        # Training state
        self.best_val_rmse = float('inf')
    
    @staticmethod
    def _validate_inputs(pretrained_path: str, tokenizer_path: str):
        """Validate that model and tokenizer files exist."""
        if not Path(pretrained_path).exists():
            raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")
        if not Path(tokenizer_path).exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    def setup(self):
        """Initialize tokenizer, model, and data."""
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
        mixed_precision = self.config.mixed_precision
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
        if self.config.full_finetune:
            for parameter in self.model.parameters():
                parameter.requires_grad = True
            logger.info("  Full fine-tuning enabled")
            return

        for parameter in self.model.model.parameters():
            parameter.requires_grad = False

        if self.config.train_embeddings:
            input_embeddings = self.model.model.get_input_embeddings()
            for parameter in input_embeddings.parameters():
                parameter.requires_grad = True

        for parameter in self.model.head.parameters():
            parameter.requires_grad = True

        logger.info(
            f"  Backbone frozen: True | Train embeddings: {self.config.train_embeddings}"
        )
    
    def train(self, num_epochs: int = 10, learning_rate: float = 1e-5):
        """Train regression model."""
        logger.info("="*60)
        logger.info("TRAINING PHASE (REGRESSION)")
        logger.info("="*60 + "\n")
        
        # Setup optimizer
        optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=0.001,
        )

        criterion = nn.MSELoss()
        scaler = GradScaler(enabled=self.use_grad_scaler)
        
        metrics_history = []
        
        try:
            for epoch in range(num_epochs):
                # Training
                train_loss = self._train_epoch(epoch, optimizer, criterion, scaler)
                
                # Validation
                val_loss, val_rmse, val_mae = self._validate(criterion)
                
                logger.info(f"Epoch {epoch+1}/{num_epochs}")
                logger.info(f"  Train Loss: {train_loss:.4f}")
                logger.info(f"  Val Loss: {val_loss:.4f}")
                logger.info(f"  Val RMSE: {val_rmse:.4f}")
                logger.info(f"  Val MAE: {val_mae:.4f}")
                
                # Save best model
                if val_rmse < self.best_val_rmse:
                    self.best_val_rmse = val_rmse
                    self._save_model(is_best=True)
                    logger.info(f"  ? New best RMSE: {val_rmse:.4f}")
                
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
            metrics_path = Path('logs') / 'regression_metrics.csv'
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Metrics saved: {metrics_path}")
            
            logger.info("="*60)
            logger.info("Training complete")
            logger.info("="*60)
    
    def _train_epoch(self, epoch: int, optimizer, criterion, scaler):
        """Train one epoch."""
        self.model.train()

        total_loss = 0.0
        gradient_accumulation_steps = int(self.config.gradient_accumulation_steps)
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
                    float(self.config.max_grad_norm),
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
            for key, value in self.model.state_dict().items()
        }
        torch.save(cpu_state_dict, save_dir / "model.pt")
        if self.tokenizer is not None:
            self.tokenizer.save(save_dir / "tokenizer.json")
        logger.info(f"Model saved: {save_dir}")


def main():
    """Main entry point."""
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
  python train_regression.py --epochs 20 --lr 5e-6
        """)
    parser.add_argument('--pretrained_model_path', default='models/qwen_1.8b_smiles_pretrained/final_model',
                       help='Path to pretrained model from Stage 1')
    parser.add_argument('--tokenizer_path', default='models/qwen_1.8b_smiles_pretrained/tokenizer.json',
                       help='Path to tokenizer')
    parser.add_argument('--csv', default=None,
                       help='Path to regression CSV (optional, auto-discovers AGILE if not provided)')
    parser.add_argument('--output', default='models/qwen_1.8b_smiles_regression',
                       help='Output directory')
    
    parser.add_argument('--no-auto-discover', action='store_true',
                       help='Disable AGILE auto-discovery')
    
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of epochs')
    parser.add_argument('--batch', type=int, default=1,
                       help='Number of batches')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--gradient_accumulation_steps', default=4,
                       type=int,
                       help='gradient_accumulation_steps')
    parser.add_argument('--max_grad_norm', default=1.0,
                       type=float,
                       help='max_grad_norm')
    
    parser.add_argument('--use_amp', default=False,
                       help='use_amp')
    parser.add_argument('--use_grad_scaler', default=False,
                       help='use_grad_scaler')
    
    parser.add_argument('--mixed_precision', default="none",
                       help='mixed_precision')
    parser.add_argument('--full_finetune', action='store_true',
                       help='Train the entire backbone instead of freezing it')
    parser.add_argument('--train_embeddings', action='store_true', default=True,
                       help='Keep resized input embeddings trainable when backbone is frozen')
    parser.add_argument('--no-train-embeddings', dest='train_embeddings', action='store_false',
                       help='Freeze resized input embeddings together with the backbone')
    
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("STAGE 2: TRANSFECTION EFFICIENCY REGRESSION")
    logger.info("="*70)
    
    trainer = QwenRegressionTrainer(
        config=args
    )
    
    
    # Setup and train
    trainer.setup()
    trainer.train(num_epochs=args.epochs, learning_rate=args.lr)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
