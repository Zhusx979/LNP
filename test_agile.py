"""
Quick test for AGILE dataset auto-discovery.

Run this to verify that all AGILE datasets are discovered correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.model_regression import RegressionDataModule
from src.tokenizer import SMILESTokenizer
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_agile_discovery():
    """Test auto-discovery of AGILE datasets."""
    print("\n" + "="*70)
    print("AGILE DATASET AUTO-DISCOVERY TEST")
    print("="*70 + "\n")
    
    # Discover CSV files
    csv_files = RegressionDataModule.discover_agile_csvs("AGILE")
    
    if not csv_files:
        print("FAILED: No AGILE CSV files discovered")
        return False
    
    print(f"Discovered {len(csv_files)} CSV files\n")
    
    # Test loading each file
    total_rows = 0
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            rows = len(df)
            total_rows += rows
            
            # Check columns
            has_smiles = 'SMILES' in df.columns
            has_target = 'TARGET' in df.columns
            
            status = "FINDING" if (has_smiles and has_target) else "MISSING"
            print(f"{status} {Path(csv_file).relative_to('AGILE')}: {rows} rows")
            
            if not has_smiles:
                print(f"  WARNING: Missing 'SMILES' column")
            if not has_target:
                print(f"  WARNING: Missing 'TARGET' column")
        
        except Exception as e:
            print(f"{Path(csv_file).name}: Error - {e}")
            return False
    
    print(f"\nTotal samples: {total_rows}")
    print("\n" + "="*70)
    return True


def test_data_loading():
    """Test full data loading pipeline."""
    print("\n" + "="*70)
    print("DATA LOADING TEST")
    print("="*70 + "\n")
    
    try:
        # Initialize tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = SMILESTokenizer()
        
        # Check if tokenizer is already saved
        tokenizer_path = Path("models/qwen_1.8b_smiles_pretrained/tokenizer.json")
        if tokenizer_path.exists():
            tokenizer.load(str(tokenizer_path))
            logger.info(f"Loaded tokenizer with {len(tokenizer)} tokens")
        else:
            logger.warning(f"Tokenizer not found at {tokenizer_path}")
            logger.info("Using fresh tokenizer (will be populated during training)")
        
        # Initialize data module
        logger.info("\nInitializing data module...")
        data_module = RegressionDataModule(
            csv_path=None,
            tokenizer=tokenizer,
            auto_discover_agile=True,
            batch_size=32,
        )
        
        # Load data
        logger.info("Loading AGILE datasets...")
        smiles_list, labels = data_module.load_data()
        
        print(f"\n Loaded {len(smiles_list)} SMILES-label pairs")
        print(f"  Label range: [{labels.min():.4f}, {labels.max():.4f}]")
        print(f"  Label mean: {labels.mean():.4f}, Label std: {labels.std():.4f}")
        
        # Show sample
        print(f"\nSample data:")
        for i in range(min(3, len(smiles_list))):
            smiles = smiles_list[i]
            label = labels[i]
            print(f"  {i+1}. SMILES: {smiles[:50]} TARGET: {label:.4f}")
        
        print("\n" + "="*70)
        return True
    
    except Exception as e:
        print(f"? FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success1 = test_agile_discovery()
    success2 = test_data_loading()
    
    if success1 and success2:
        print("\nALL TESTS PASSED\n")
        print("You can now run Stage 2 training with:")
        print("  python train_regression.py\n")
        sys.exit(0)
    else:
        print("\n? SOME TESTS FAILED\n")
        sys.exit(1)
