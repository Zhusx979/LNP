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
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple
import json
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
sys.path.insert(0, str(PROJECT_ROOT))
from src.model_regression import QwenRegressionModel, RegressionDataModule
from src.tokenizer import SMILESTokenizer
from src.logging_utils import configure_logging
from src.qwen_utils import ensure_local_qwen_code, ensure_required_dependencies
from config.regression_cli import build_config_from_args, build_parser
from src.regression_utils import (
    compute_regression_metrics,
    infer_regression_dataset_name,
    json_ready_dict,
    safe_correlation,
    to_builtin,
)
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

try:
    from transformers import AutoModelForCausalLM
except ImportError:
    AutoModelForCausalLM = None

logger = logging.getLogger(__name__)


def ensure_transformers():
    """Fail fast when training starts without required dependencies."""
    ensure_required_dependencies(
        {"transformers.AutoModelForCausalLM": AutoModelForCausalLM},
        "pip install -r requirements.txt",
    )


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

        self.pretrained_model_path = resolve_path(self.paths_config['pretrained_model_path'], PROJECT_ROOT)
        self.tokenizer_path = resolve_path(self.paths_config['tokenizer_path'], PROJECT_ROOT)
        self.output_root_dir = resolve_path(self.paths_config['output_dir'], PROJECT_ROOT)
        self.output_dir = self.output_root_dir / self.dataset_name
        self.logs_dir = resolve_path(self.paths_config['logs_dir'], PROJECT_ROOT)

        self._validate_inputs(self.pretrained_model_path, self.tokenizer_path)
        self.regression_csv = (
            str(resolve_path(self.data_config['csv_path'], PROJECT_ROOT))
            if self.data_config['csv_path'] is not None
            else None
        )
        self.batch_size = self.data_config['batch_size']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.auto_discover_agile = self.data_config['auto_discover_agile']
        self.agile_cell_line = self.data_config.get('agile_cell_line')
        self.agile_split = self.data_config.get('agile_split')
        
        self.device, self.gpu_ids = resolve_device_config(
            self.runtime_config.get('gpus'),
            logger=logger,
        )
        self.distributed = setup_distributed_training(self.device, logger=logger)
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
        return safe_correlation(fn, y_true, y_pred)

    @classmethod
    def _compute_regression_metrics(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute a comprehensive set of regression metrics."""
        return compute_regression_metrics(y_true, y_pred)

    @staticmethod
    def _to_builtin(value):
        """Convert numpy/pandas scalars to JSON-serializable Python values."""
        return to_builtin(value)

    def _json_ready_dict(self, data: Dict) -> Dict:
        """Recursively convert metrics dictionaries for JSON serialization."""
        return json_ready_dict(data)

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
        """Whether this process is responsible for logging and artifacts."""
        return is_main_process()
    
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
        copied_qwen_files = ensure_local_qwen_code(self.pretrained_model_path)
        if copied_qwen_files:
            logger.info(
                "Copied Qwen support files into %s: %s",
                self.pretrained_model_path,
                ", ".join(copied_qwen_files),
            )
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
        self._maybe_wrap_model_for_multi_gpu()
        logger.info(f"  Mixed precision mode: {mixed_precision}")
        logger.info(f"  AMP enabled: {self.use_amp}")
        logger.info(f"  GradScaler enabled: {self.use_grad_scaler}")
        logger.info(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"  Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        # Load data
        logger.info("[3/3] Loading regression dataset...")
        data_module.setup()
        self.train_loader, self.val_loader, self.test_loader = data_module.create_loaders(
            distributed=self.distributed.enabled,
            world_size=self.distributed.world_size,
            rank=self.distributed.rank,
        )
        
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
                if self.is_main_process:
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
                distributed_barrier()
        
        except KeyboardInterrupt:
            if self.is_main_process:
                logger.info("Training interrupted")
        
        finally:
            if self.is_main_process:
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
            distributed_barrier()
            cleanup_distributed_training()
    
    def _train_epoch(self, epoch: int, optimizer, criterion, scaler):
        """Train one epoch."""
        self.model.train()

        total_loss = 0.0
        gradient_accumulation_steps = int(self.training_config['gradient_accumulation_steps'])
        optimizer.zero_grad(set_to_none=True)
        train_sampler = getattr(self.train_loader, 'sampler', None)
        if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        for batch_idx, batch in tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch + 1}",
        ):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch.pop('label').view(-1)
            
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
                # Forward pass
                with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    predictions = self.model(**batch).view(-1)
                    raw_loss = criterion(predictions, labels)
                loss = raw_loss / gradient_accumulation_steps
                # Backward pass
                if self.use_grad_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            # Accumulate metrics
            total_loss += raw_loss.item()

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

        total_loss_tensor = torch.tensor(total_loss, device=self.device, dtype=torch.float64)
        if self.distributed.enabled:
            torch.distributed.all_reduce(total_loss_tensor, op=torch.distributed.ReduceOp.SUM)
            mean_loss = total_loss_tensor.item() / (
                len(self.train_loader) * self.distributed.world_size
            )
        else:
            mean_loss = total_loss / len(self.train_loader)

        return mean_loss
    
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
    configure_logging(
        resolve_path(config['paths']['logs_dir'], PROJECT_ROOT),
        log_filename=f"regression_training_{dataset_name}.log",
    )
    
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
