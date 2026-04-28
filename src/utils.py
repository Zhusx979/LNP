"""
Utility functions for training pipeline.
"""

import torch
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_dir: str, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        
    Returns:
        Dictionary with training state (step, epoch, etc.)
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Load model
    model_path = checkpoint_dir / "pytorch_model.bin"
    if model_path.exists():
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        logger.info(f"Loaded model from {checkpoint_dir}")
    
    # Load training state
    state_path = checkpoint_dir / "training_state.pt"
    training_state = {}
    if state_path.exists():
        training_state = torch.load(state_path, map_location='cpu')
        
        if optimizer and 'optimizer_state' in training_state:
            optimizer.load_state_dict(training_state['optimizer_state'])
        
        if scheduler and 'scheduler_state' in training_state:
            scheduler.load_state_dict(training_state['scheduler_state'])
        
        logger.info(f"Restored training state (step={training_state.get('step', 0)})")
    
    return training_state


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity (exp(loss))
    """
    import math
    return math.exp(loss)


def save_config(config: Dict, path: str):
    """Save configuration to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {path}")


def load_config(path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        config = json.load(f)
    return config


def count_parameters(model) -> Tuple[int, int]:
    """
    Count total and trainable parameters.
    
    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
