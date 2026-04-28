"""
SMILES-Based Qwen Pretraining & Regression Pipeline

A complete training pipeline for SMILES-based molecular modeling:
- Stage 1: Self-supervised causal language modeling (SMILES pretraining)
- Stage 2: Downstream task fine-tuning (transfection efficiency regression)
"""

__version__ = "1.0.0"
__author__ = "Deep Learning Research Team"

from .tokenizer import SMILESTokenizer
from .dataset import SMILESDataset, SMILESDataModule
from .model_regression import QwenRegressionModel, RegressionDataset, RegressionDataModule

__all__ = [
    'SMILESTokenizer',
    'SMILESDataset',
    'SMILESDataModule',
    'QwenRegressionModel',
    'RegressionDataset',
    'RegressionDataModule',
]
