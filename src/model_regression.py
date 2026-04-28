

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class QwenRegressionModel(nn.Module):
    """
    Qwen-based regression model for transfection efficiency prediction.
    
    Architecture:
    - Qwen-1.8B backbone (from pretraining)
    - Mean pooling over sequence
    - Linear regression head (hidden_dim �� 1)
    """
    
    def __init__(self, model, hidden_size: int = 2048, dropout: float = 0.1):
        """
        Initialize regression model.
        
        Args:
            model: Pretrained Qwen model
            hidden_size: Hidden dimension from Qwen
            dropout: Dropout rate for head
        """
        super().__init__()
        self.model = model
        self.hidden_size = hidden_size
        
        # Regression head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),  # Output: continuous value
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Regression predictions [batch_size, 1]
        """
        # Get hidden states from Qwen
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Use mean pooling over non-padding tokens
        hidden_states = outputs.hidden_states[-1]  # Last layer
        
        # Apply attention mask and compute mean
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        masked_hidden = hidden_states * mask_expanded
        sum_hidden = masked_hidden.sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        
        # Avoid division by zero
        mean_pooled = sum_hidden / (sum_mask + 1e-9)
        
        # Regression head
        predictions = self.head(mean_pooled)
        
        return predictions


class RegressionDataset(Dataset):
    """
    Dataset for regression task: SMILES �� transfection efficiency.
    
    Expected CSV format:
    - SMILES: SMILES string
    - transfection_efficiency: float value (0-1 or unbounded)
    """
    
    def __init__(
        self,
        smiles_list: List[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int = 256,
        normalize: bool = True,
    ):
        """
        Initialize regression dataset.
        
        Args:
            smiles_list: List of SMILES strings
            labels: Array of regression targets
            tokenizer: SMILES tokenizer
            max_length: Max sequence length
            normalize: Normalize labels (zero mean, unit std)
        """
        self.smiles_list = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Normalize labels
        self.normalize = normalize
        self.label_scaler = None
        if normalize:
            self.label_scaler = StandardScaler()
            self.labels = self.label_scaler.fit_transform(labels.reshape(-1, 1)).flatten()
    
    def __len__(self) -> int:
        return len(self.smiles_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        smiles = self.smiles_list[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoded = self.tokenizer.encode(smiles)
        
        return {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float32),
        }


class RegressionDataModule:
    """
    Data loading module for regression task.
    
    Supports both:
    1. Auto-discovery of AGILE dataset (AGILE/*/*/test.csv)
    2. Manual CSV path specification
    """
    
    def __init__(
        self,
        csv_path: Optional[str] = None,
        tokenizer=None,
        max_length: int = 256,
        batch_size: int = 32,
        validation_split: float = 0.2,
        test_split: float = 0.1,
        normalize_labels: bool = True,
        auto_discover_agile: bool = True,
    ):
        """
        Initialize data module.
        
        Args:
            csv_path: Path to single regression CSV (optional)
            tokenizer: SMILES tokenizer
            max_length: Max sequence length
            batch_size: Batch size
            validation_split: Validation set fraction
            test_split: Test set fraction
            normalize_labels: Normalize regression targets
            auto_discover_agile: Auto-discover AGILE datasets if csv_path is None
        """
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.normalize_labels = normalize_labels
        self.auto_discover_agile = auto_discover_agile
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.loaded_csv_files = []
    
    @staticmethod
    def discover_agile_csvs(root_dir: str = "AGILE") -> List[str]:
        """
        Auto-discover all test.csv files in AGILE directory structure.
        
        Expected structure:
            AGILE/
            ������ RaW/
            ��   ������ scaffold/
            ��   ��   ������ test.csv
            ��   ��   ������ train.csv
            ��   ������ cliff/
            ��       ������ test.csv
            ��       ������ train.csv
            ������ Hela/
                ������ scaffold/
                ��   ������ test.csv
                ��   ������ train.csv
                ������ cliff/
                    ������ test.csv
                    ������ train.csv
        
        Args:
            root_dir: Root directory to search
            
        Returns:
            List of paths to test.csv files
        """
        agile_root = Path(root_dir)
        
        if not agile_root.exists():
            logger.warning(f"AGILE directory not found: {agile_root}")
            return []
        
        csv_files = sorted(agile_root.glob("**/test.csv"))
        logger.info(f"Discovered {len(csv_files)} AGILE test.csv files")
        
        for csv_file in csv_files:
            logger.info(f"  - {csv_file.relative_to(agile_root)}")
        
        return [str(f) for f in csv_files]
    
    def load_data(self) -> Tuple[List[str], np.ndarray]:
        """
        Load SMILES and labels from CSV(s).
        
        Supports both:
        1. Single CSV with columns: SMILES, TARGET
        2. Multiple AGILE CSVs (auto-discovered)
        
        Returns:
            (smiles_list, labels_array)
        """
        all_smiles = []
        all_labels = []
        
        # Determine which CSVs to load
        csv_files = []
        if self.csv_path:
            # Manual CSV path
            csv_files = [self.csv_path]
        elif self.auto_discover_agile:
            # Auto-discover AGILE
            csv_files = self.discover_agile_csvs()
            if not csv_files:
                raise FileNotFoundError(
                    "No CSV files found. Either:\n"
                    "1. Provide --csv path/to/labels.csv\n"
                    "2. Ensure AGILE directory exists with test.csv files"
                )
        else:
            raise ValueError("Must provide csv_path or enable auto_discover_agile")
        
        self.loaded_csv_files = csv_files
        
        # Load all CSVs
        for csv_path in csv_files:
            csv_file = Path(csv_path)
            
            if not csv_file.exists():
                logger.warning(f"CSV not found: {csv_file}")
                continue
            
            try:
                df = pd.read_csv(csv_file)
                
                # Handle both column names: 'TARGET' and 'transfection_efficiency'
                if 'TARGET' in df.columns:
                    label_col = 'TARGET'
                elif 'transfection_efficiency' in df.columns:
                    label_col = 'transfection_efficiency'
                else:
                    logger.warning(f"No target column found in {csv_file}")
                    continue
                
                # Validate SMILES column
                if 'SMILES' not in df.columns:
                    logger.warning(f"No SMILES column in {csv_file}")
                    continue
                
                smiles = df['SMILES'].dropna().tolist()
                labels = df[label_col].dropna().values
                
                # Ensure same length
                min_len = min(len(smiles), len(labels))
                smiles = smiles[:min_len]
                labels = labels[:min_len]
                
                all_smiles.extend(smiles)
                all_labels.extend(labels)
                
                logger.info(f"  Loaded {len(smiles)} samples from {csv_file.name}")
            
            except Exception as e:
                logger.warning(f"Error loading {csv_file}: {e}")
                continue
        
        if not all_smiles:
            raise ValueError("No valid SMILES/label pairs found in any CSV")
        
        all_labels = np.array(all_labels)
        
        logger.info(f"? Total samples loaded: {len(all_smiles)}")
        logger.info(f"Label range: [{all_labels.min():.4f}, {all_labels.max():.4f}]")
        logger.info(f"Label mean: {all_labels.mean():.4f}, std: {all_labels.std():.4f}")
        
        return all_smiles, all_labels
    
    def setup(self):
        """Load data and split into train/val/test."""
        smiles_list, labels = self.load_data()
        
        # Stratify split by label quantiles
        n_total = len(smiles_list)
        n_test = int(n_total * self.test_split)
        n_val = int(n_total * self.validation_split)
        n_train = n_total - n_test - n_val
        
        # Random split
        indices = np.arange(n_total)
        np.random.shuffle(indices)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        
        # Create datasets
        self.train_dataset = RegressionDataset(
            [smiles_list[i] for i in train_idx],
            labels[train_idx],
            self.tokenizer,
            self.max_length,
            normalize=self.normalize_labels,
        )
        
        self.val_dataset = RegressionDataset(
            [smiles_list[i] for i in val_idx],
            labels[val_idx],
            self.tokenizer,
            self.max_length,
            normalize=self.normalize_labels,
        )
        
        self.test_dataset = RegressionDataset(
            [smiles_list[i] for i in test_idx],
            labels[test_idx],
            self.tokenizer,
            self.max_length,
            normalize=self.normalize_labels,
        )
        
        logger.info(f"Train/Val/Test split: {n_train}/{n_val}/{n_test}")
    
    def create_loaders(self):
        """Create DataLoaders."""
        if self.train_dataset is None:
            self.setup()
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )
        
        return self.train_loader, self.val_loader, self.test_loader
