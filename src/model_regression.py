

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
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
        label_scaler: Optional[StandardScaler] = None,
        fit_label_scaler: bool = False,
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
        self.label_scaler = label_scaler
        if normalize:
            if self.label_scaler is None:
                self.label_scaler = StandardScaler()
                fit_label_scaler = True
            if fit_label_scaler:
                self.labels = self.label_scaler.fit_transform(labels.reshape(-1, 1)).flatten()
            else:
                self.labels = self.label_scaler.transform(labels.reshape(-1, 1)).flatten()
    
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
    1. Auto-discovery of AGILE datasets (AGILE/*/*/{train,test}.csv)
    2. Manual CSV path specification
    """

    AGILE_CELL_LINE_MAP = {
        'hela': 'Hela',
        'raw': 'RaW',
    }
    AGILE_SPLIT_MAP = {
        'cliff': 'cliff',
        'scaffold': 'scaffold',
    }
    
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
        agile_root_dir: str = "AGILE",
        agile_cell_line: Optional[str] = None,
        agile_split: Optional[str] = None,
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
            agile_root_dir: Root AGILE directory
            agile_cell_line: Optional AGILE cell line filter (Hela or RaW)
            agile_split: Optional AGILE split filter (cliff or scaffold)
        """
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.normalize_labels = normalize_labels
        self.auto_discover_agile = auto_discover_agile
        self.agile_root_dir = agile_root_dir
        self.agile_cell_line = self._normalize_agile_option(
            agile_cell_line,
            self.AGILE_CELL_LINE_MAP,
            "agile_cell_line",
        )
        self.agile_split = self._normalize_agile_option(
            agile_split,
            self.AGILE_SPLIT_MAP,
            "agile_split",
        )
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.loaded_csv_files = []
        self.label_scaler = None
        self.agile_train_data: Optional[Tuple[List[str], np.ndarray]] = None
        self.agile_test_data: Optional[Tuple[List[str], np.ndarray]] = None

    @classmethod
    def _normalize_agile_option(
        cls,
        value: Optional[str],
        mapping: Dict[str, str],
        option_name: str,
    ) -> Optional[str]:
        """Normalize AGILE CLI/programmatic options to directory names."""
        if value is None:
            return None

        normalized = value.strip().lower()
        if normalized not in mapping:
            valid_values = ", ".join(mapping.values())
            raise ValueError(f"Invalid {option_name}: {value}. Expected one of: {valid_values}")
        return mapping[normalized]
    
    @staticmethod
    def discover_agile_csvs(
        root_dir: str = "AGILE",
        cell_line: Optional[str] = None,
        split_name: Optional[str] = None,
        csv_name: str = "test.csv",
    ) -> List[str]:
        """
        Auto-discover matching CSV files in AGILE directory structure.
        
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
            cell_line: Optional AGILE cell line filter
            split_name: Optional AGILE split filter
            csv_name: CSV filename to discover (train.csv or test.csv)
            
        Returns:
            List of paths to matching CSV files
        """
        agile_root = Path(root_dir)
        normalized_cell_line = RegressionDataModule._normalize_agile_option(
            cell_line,
            RegressionDataModule.AGILE_CELL_LINE_MAP,
            "cell_line",
        )
        normalized_split = RegressionDataModule._normalize_agile_option(
            split_name,
            RegressionDataModule.AGILE_SPLIT_MAP,
            "split_name",
        )
        
        if not agile_root.exists():
            logger.warning(f"AGILE directory not found: {agile_root}")
            return []

        if normalized_cell_line is not None:
            cell_line_dirs = [normalized_cell_line]
        else:
            cell_line_dirs = list(RegressionDataModule.AGILE_CELL_LINE_MAP.values())

        if normalized_split is not None:
            split_dirs = [normalized_split]
        else:
            split_dirs = list(RegressionDataModule.AGILE_SPLIT_MAP.values())

        csv_files = []
        for cell_line_dir in cell_line_dirs:
            for split_dir in split_dirs:
                csv_file = agile_root / cell_line_dir / split_dir / csv_name
                if csv_file.exists():
                    csv_files.append(csv_file)

        csv_files = sorted(csv_files)
        logger.info(f"Discovered {len(csv_files)} AGILE {csv_name} files")
        
        for csv_file in csv_files:
            logger.info(f"  - {csv_file.relative_to(agile_root)}")
        
        return [str(f) for f in csv_files]

    @staticmethod
    def _load_csv_file(csv_file: Path) -> Tuple[List[str], np.ndarray]:
        """Load one CSV and return aligned SMILES/labels."""
        df = pd.read_csv(csv_file)

        if 'TARGET' in df.columns:
            label_col = 'TARGET'
        elif 'transfection_efficiency' in df.columns:
            label_col = 'transfection_efficiency'
        else:
            raise ValueError(f"No target column found in {csv_file}")

        if 'SMILES' not in df.columns:
            raise ValueError(f"No SMILES column in {csv_file}")

        subset = df[['SMILES', label_col]].dropna()
        smiles = subset['SMILES'].astype(str).tolist()
        labels = subset[label_col].astype(float).to_numpy()

        return smiles, labels

    def _load_csv_files(
        self,
        csv_files: List[str],
        split_label: str,
    ) -> Tuple[List[str], np.ndarray]:
        """Load and concatenate multiple CSV files."""
        all_smiles: List[str] = []
        all_labels: List[np.ndarray] = []

        for csv_path in csv_files:
            csv_file = Path(csv_path)

            if not csv_file.exists():
                logger.warning(f"CSV not found: {csv_file}")
                continue

            try:
                smiles, labels = self._load_csv_file(csv_file)
                all_smiles.extend(smiles)
                all_labels.append(labels)
                logger.info(f"  Loaded {len(smiles)} {split_label} samples from {csv_file}")
            except Exception as e:
                logger.warning(f"Error loading {csv_file}: {e}")

        if not all_smiles:
            return [], np.array([], dtype=float)

        return all_smiles, np.concatenate(all_labels)
    
    def load_data(self) -> Tuple[List[str], np.ndarray]:
        """
        Load SMILES and labels from CSV(s).
        
        Supports both:
        1. Single CSV with columns: SMILES, TARGET
        2. Multiple AGILE CSVs (auto-discovered)
        
        Returns:
            (smiles_list, labels_array)
        """
        self.agile_train_data = None
        self.agile_test_data = None
        
        if self.csv_path:
            csv_files = [self.csv_path]
            self.loaded_csv_files = csv_files
            smiles, labels = self._load_csv_files(csv_files, "manual")
            if not smiles:
                raise ValueError("No valid SMILES/label pairs found in the provided CSV")
            all_smiles = smiles
            all_labels = labels
        elif self.auto_discover_agile:
            train_csv_files = self.discover_agile_csvs(
                root_dir=self.agile_root_dir,
                cell_line=self.agile_cell_line,
                split_name=self.agile_split,
                csv_name="train.csv",
            )
            test_csv_files = self.discover_agile_csvs(
                root_dir=self.agile_root_dir,
                cell_line=self.agile_cell_line,
                split_name=self.agile_split,
                csv_name="test.csv",
            )

            if not train_csv_files and not test_csv_files:
                raise FileNotFoundError(
                    "No CSV files found. Either:\n"
                    "1. Provide --csv path/to/labels.csv\n"
                    "2. Ensure AGILE directory exists with train.csv/test.csv files"
                )

            train_smiles, train_labels = self._load_csv_files(train_csv_files, "train")
            test_smiles, test_labels = self._load_csv_files(test_csv_files, "test")

            if not train_smiles and not test_smiles:
                raise ValueError("No valid SMILES/label pairs found in selected AGILE dataset(s)")

            self.loaded_csv_files = train_csv_files + test_csv_files
            self.agile_train_data = (train_smiles, train_labels)
            self.agile_test_data = (test_smiles, test_labels)

            all_smiles = train_smiles + test_smiles
            non_empty_label_arrays = [arr for arr in [train_labels, test_labels] if len(arr) > 0]
            all_labels = (
                np.concatenate(non_empty_label_arrays)
                if non_empty_label_arrays
                else np.array([], dtype=float)
            )
        else:
            raise ValueError("Must provide csv_path or enable auto_discover_agile")

        if not all_smiles or len(all_labels) == 0:
            raise ValueError("No valid SMILES/label pairs found in any CSV")
        
        logger.info(f"? Total samples loaded: {len(all_smiles)}")
        logger.info(f"Label range: [{all_labels.min():.4f}, {all_labels.max():.4f}]")
        logger.info(f"Label mean: {all_labels.mean():.4f}, std: {all_labels.std():.4f}")
        
        return all_smiles, all_labels

    def _setup_from_manual_csv(self, smiles_list: List[str], labels: np.ndarray):
        """Split a single CSV into train/val/test partitions."""
        n_total = len(smiles_list)
        n_test = int(n_total * self.test_split)
        n_val = int(n_total * self.validation_split)
        n_train = n_total - n_test - n_val

        indices = np.arange(n_total)
        np.random.shuffle(indices)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]

        train_labels = labels[train_idx]
        self.label_scaler = StandardScaler() if self.normalize_labels else None

        self.train_dataset = RegressionDataset(
            [smiles_list[i] for i in train_idx],
            train_labels,
            self.tokenizer,
            self.max_length,
            normalize=self.normalize_labels,
            label_scaler=self.label_scaler,
            fit_label_scaler=self.normalize_labels,
        )

        self.val_dataset = RegressionDataset(
            [smiles_list[i] for i in val_idx],
            labels[val_idx],
            self.tokenizer,
            self.max_length,
            normalize=self.normalize_labels,
            label_scaler=self.label_scaler,
        )

        self.test_dataset = RegressionDataset(
            [smiles_list[i] for i in test_idx],
            labels[test_idx],
            self.tokenizer,
            self.max_length,
            normalize=self.normalize_labels,
            label_scaler=self.label_scaler,
        )

        logger.info(f"Train/Val/Test split: {n_train}/{n_val}/{n_test}")

    def _setup_from_agile_splits(self):
        """Use AGILE train.csv/test.csv files directly and split train into train/val."""
        if self.agile_train_data is None or self.agile_test_data is None:
            raise RuntimeError("AGILE split data has not been loaded")

        train_smiles, train_labels = self.agile_train_data
        test_smiles, test_labels = self.agile_test_data

        if not train_smiles or len(train_labels) == 0:
            raise ValueError("Selected AGILE dataset(s) do not contain valid train.csv samples")
        if not test_smiles or len(test_labels) == 0:
            raise ValueError("Selected AGILE dataset(s) do not contain valid test.csv samples")

        n_train_total = len(train_smiles)
        n_val = int(n_train_total * self.validation_split)
        if self.validation_split > 0 and n_val == 0 and n_train_total > 1:
            n_val = 1
        if n_val >= n_train_total:
            n_val = max(0, n_train_total - 1)

        indices = np.arange(n_train_total)
        np.random.shuffle(indices)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        self.label_scaler = StandardScaler() if self.normalize_labels else None

        self.train_dataset = RegressionDataset(
            [train_smiles[i] for i in train_idx],
            train_labels[train_idx],
            self.tokenizer,
            self.max_length,
            normalize=self.normalize_labels,
            label_scaler=self.label_scaler,
            fit_label_scaler=self.normalize_labels,
        )

        self.val_dataset = RegressionDataset(
            [train_smiles[i] for i in val_idx],
            train_labels[val_idx],
            self.tokenizer,
            self.max_length,
            normalize=self.normalize_labels,
            label_scaler=self.label_scaler,
        )

        self.test_dataset = RegressionDataset(
            test_smiles,
            test_labels,
            self.tokenizer,
            self.max_length,
            normalize=self.normalize_labels,
            label_scaler=self.label_scaler,
        )

        logger.info(
            "AGILE split sizes -> train: %d, val: %d, test: %d",
            len(train_idx),
            len(val_idx),
            len(test_smiles),
        )
    
    def setup(self):
        """Load data and split into train/val/test."""
        smiles_list, labels = self.load_data()
        if self.agile_train_data is not None and self.agile_test_data is not None:
            self._setup_from_agile_splits()
        else:
            self._setup_from_manual_csv(smiles_list, labels)
    
    def create_loaders(
        self,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
    ):
        """Create DataLoaders."""
        if self.train_dataset is None:
            self.setup()

        train_sampler = None
        if distributed:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False,
            )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            pin_memory=torch.cuda.is_available(),
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        
        return self.train_loader, self.val_loader, self.test_loader
