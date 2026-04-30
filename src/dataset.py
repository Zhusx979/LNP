"""
SMILES Dataset for Causal Language Modeling

Handles:
- CSV loading and merging
- SMILES validation
- Tokenization
- Next-token prediction pairs
- PyTorch DataLoader
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from .tokenizer import SMILESTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SMILESDataset(Dataset):
    """
    PyTorch Dataset for SMILES causal language modeling.
    
    For each SMILES string, creates next-token prediction pairs:
    - Input: tokens[0:n-1]
    - Target: tokens[1:n]
    """
    
    def __init__(
        self,
        smiles_list: List[str],
        tokenizer: SMILESTokenizer,
        max_length: int = 256,
    ):
        """
        Initialize dataset.
        
        Args:
            smiles_list: List of SMILES strings
            tokenizer: Initialized SMILESTokenizer with vocab
            max_length: Maximum sequence length for padding/truncation
        """
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Preprocess all SMILES into token pairs
        self._preprocess_smiles()
    
    def _preprocess_smiles(self):
        """Convert SMILES to next-token pairs and store."""
        logger.info(f"Preprocessing {len(self.smiles_list)} SMILES strings...")
        
        for idx, smiles in enumerate(self.smiles_list):
            if idx % 100000 == 0:
                logger.info(f"  Processed {idx}/{len(self.smiles_list)} SMILES")
            
            try:
                # Tokenize SMILES (without special tokens for raw tokenization)
                tokens = self.tokenizer.tokenize(smiles)
                
                # Skip very short SMILES (less than 2 tokens)
                if len(tokens) < 2:
                    continue
                
                # Truncate to max_length - 2 (for [CLS] and [EOS])
                if len(tokens) > self.max_length - 2:
                    tokens = tokens[:self.max_length - 2]
                
                # Convert tokens to IDs
                input_ids = []
                for token in tokens:
                    if token in self.tokenizer.token2id:
                        input_ids.append(self.tokenizer.token2id[token])
                    else:
                        input_ids.append(self.tokenizer.token2id['[UNK]'])
                
                # Create next-token pairs
                # Input: [CLS] + tokens[:-1], padded to max_length
                # Target: tokens[1:] + [EOS], padded to max_length
                input_ids_with_cls = [self.tokenizer.token2id['[CLS]']] + input_ids[:-1]
                target_ids = input_ids[1:] + [self.tokenizer.token2id['[EOS]']]
                
                # Pad both to max_length
                input_ids_padded = input_ids_with_cls + [self.tokenizer.token2id['[PAD]']] * (self.max_length - len(input_ids_with_cls))
                target_ids_padded = target_ids + [self.tokenizer.token2id['[PAD]']] * (self.max_length - len(target_ids))
                
                # Create attention mask (1 for real tokens, 0 for padding)
                attention_mask = [1] * len(input_ids_with_cls) + [0] * (self.max_length - len(input_ids_with_cls))
                
                self.data.append({
                    'input_ids': input_ids_padded[:self.max_length],
                    'target_ids': target_ids_padded[:self.max_length],
                    'attention_mask': attention_mask,
                    'smiles': smiles,
                })
            
            except Exception as e:
                logger.warning(f"Error processing SMILES {idx}: {smiles}")
                logger.warning(f"  Error: {str(e)}")
                continue
        
        logger.info(f"Preprocessed {len(self.data)} SMILES successfully")
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single sample as tensors."""
        sample = self.data[idx]
        return {
            'input_ids': torch.tensor(sample['input_ids'], dtype=torch.long),
            'target_ids': torch.tensor(sample['target_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(sample['attention_mask'], dtype=torch.long),
        }


class SMILESDataModule:
    """
    Data loading module for SMILES training.
    Handles CSV loading, splitting, and DataLoader creation.
    """
    
    def __init__(
        self,
        csv_paths: List[str],
        tokenizer: SMILESTokenizer,
        max_length: int = 256,
        validation_split: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = True,
        deduplicate: bool = True,
    ):
        """
        Initialize data module.
        
        Args:
            csv_paths: List of CSV file paths
            tokenizer: SMILESTokenizer instance
            max_length: Max sequence length
            validation_split: Fraction for validation (0.0-1.0)
            batch_size: Batch size for DataLoader
            num_workers: Number of workers for DataLoader
            shuffle: Shuffle training data
            deduplicate: Remove duplicate SMILES strings
        """
        self.csv_paths = csv_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.deduplicate = deduplicate
        
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
    
    
    def load_csv_files(self) -> List[str]:
        """
        Load and merge SMILES from all CSV files.
        
        Returns:
            List of SMILES strings
        """
        all_smiles = []
        
        for csv_path in self.csv_paths:
            logger.info(f"Loading CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Check for SMILES column
            if 'SMILES' not in df.columns:
                raise ValueError(f"CSV {csv_path} does not have 'SMILES' column")
            
            smiles = df['SMILES'].dropna().tolist()
            logger.info(f"  Loaded {len(smiles)} SMILES from {csv_path}")
            all_smiles.extend(smiles)
        
        def deduplicate_preserve_order(smiles_list):
            seen = set()
            result = []
            for s in smiles_list:
                if s not in seen:
                    seen.add(s)
                    result.append(s)
            return result

        # Deduplicate if requested
        if self.deduplicate:
            unique_smiles = deduplicate_preserve_order(all_smiles)
            logger.info(
                f"After deduplication: {len(unique_smiles)} unique SMILES "
                f"(removed {len(all_smiles) - len(unique_smiles)})"
            )
            return unique_smiles
        return all_smiles
    


    def setup(self):
        """Load data and build vocabulary."""
        # Load SMILES from CSV
        smiles_list = self.load_csv_files()
        
        # Build tokenizer vocabulary
        logger.info("Building tokenizer vocabulary...")
        self.tokenizer.build_vocab(smiles_list)
        logger.info(f"Vocabulary size: {len(self.tokenizer)}")
        
        # Split into train/val
        n_val = int(len(smiles_list) * self.validation_split)
        indices = np.arange(len(smiles_list))
        if self.shuffle:
            np.random.shuffle(indices)
        
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        train_smiles = [smiles_list[i] for i in train_indices]
        val_smiles = [smiles_list[i] for i in val_indices]
        
        logger.info(f"Train/Val split: {len(train_smiles)} / {len(val_smiles)}")
        
        # Create datasets
        self.train_dataset = SMILESDataset(
            train_smiles,
            self.tokenizer,
            self.max_length,
        )
        
        self.val_dataset = SMILESDataset(
            val_smiles,
            self.tokenizer,
            self.max_length,
        )
    
    def create_loaders(
        self,
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
    ):
        """Create PyTorch DataLoaders."""
        if self.train_dataset is None:
            self.setup()

        train_sampler = None
        val_sampler = None
        if distributed:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=self.shuffle,
                drop_last=False,
            )
            val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle if train_sampler is None else False,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        logger.info(f"? Created DataLoaders")
        logger.info(f"  Train batches: {len(self.train_loader)}")
        logger.info(f"  Val batches: {len(self.val_loader)}")
        
        return self.train_loader, self.val_loader
    
    def get_train_loader(self) -> DataLoader:
        """Get training DataLoader."""
        if self.train_loader is None:
            self.create_loaders()
        return self.train_loader
    
    def get_val_loader(self) -> DataLoader:
        """Get validation DataLoader."""
        if self.val_loader is None:
            self.create_loaders()
        return self.val_loader


# ============================================================================
# Testing
# ============================================================================

def test_dataset_pipeline():
    """Test complete data pipeline."""
    print("\n" + "="*60)
    print("DATASET PIPELINE TEST")
    print("="*60 + "\n")
    
    # Create test CSV files
    test_dir = Path("data")
    test_dir.mkdir(exist_ok=True)
    
    # Sample SMILES data
    test_data = {
        'Combo': ['A1B1C1D1', 'A2B2C2D2'],
        'SMILES': [
            'CCCCCCCCCCOC(CCCN(C(=O)CCCCC(=O)OC(CCCCCC)CCCCCC)C(C(=O)NCCCCN1CCCCC1CC)C(CCC)CCC)OCCCCCCCCCC',
            'CCCCCCCCCCOC(CCCN(C(=O)CCOC(=O)C(CCCCCCCC)CCCCCCCC)C(C(=O)NCCCN(CC)CCC)C(CCC)CCC)OCCCCCCCCCC',
        ],
        'Status': ['Valid lipid structure', 'Valid lipid structure'],
    }
    
    df = pd.DataFrame(test_data)
    csv_path = test_dir / "test_lipids.csv"
    df.to_csv(csv_path, index=False)
    
    # Initialize tokenizer and data module
    tokenizer = SMILESTokenizer(max_length=256)
    data_module = SMILESDataModule(
        csv_paths=[str(csv_path)],
        tokenizer=tokenizer,
        max_length=256,
        batch_size=2,
        validation_split=0.5,
    )
    
    # Setup and create loaders
    data_module.setup()
    train_loader, val_loader = data_module.create_loaders()
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Target IDs shape: {batch['target_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"\nFirst sample input_ids (first 30): {batch['input_ids'][0][:30]}")
    print(f"First sample target_ids (first 30): {batch['target_ids'][0][:30]}")
    print(f"First sample attention_mask (first 30): {batch['attention_mask'][0][:30]}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    test_dataset_pipeline()
