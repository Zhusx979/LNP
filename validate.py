"""
Quick Start & Validation Script

Run this to verify the pipeline is working correctly before full training.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.tokenizer import SMILESTokenizer, test_tokenizer, test_vocab_building, test_encode_decode
from src.dataset import test_dataset_pipeline


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_tokenizer_module():
    """Test SMILES tokenizer."""
    print_header("TEST 1: SMILES TOKENIZER")
    test_tokenizer()
    test_vocab_building()
    test_encode_decode()
    print("Tokenizer tests passed")


def test_dataset_module():
    """Test dataset pipeline."""
    print_header("TEST 2: DATASET PIPELINE")
    test_dataset_pipeline()
    print("Dataset tests passed")


def test_csv_availability():
    """Check if CSV files are available."""
    print_header("TEST 3: CSV FILE AVAILABILITY")
    root = Path(__file__).parent
    data_dir = root / "data"
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("WARNING: No CSV files found in root directory")
        return False
    
    print(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        size_mb = csv_file.stat().st_size / (1024*1024)
        print(f"  - {csv_file.name} ({size_mb:.1f} MB)")
    
    return True


def test_dependencies():
    """Check if all dependencies are installed."""
    print_header("TEST 4: DEPENDENCY CHECK")
    
    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'HuggingFace Transformers',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
    }
    
    missing = []
    for module_name, display_name in dependencies.items():
        try:
            __import__(module_name)
            print(f"{display_name} installed")
        except ImportError:
            print(f"{display_name} NOT installed")
            missing.append(module_name)
    
    if missing:
        print(f"\nMissing dependencies. Install with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True


def test_gpu():
    """Check GPU availability."""
    print_header("TEST 5: GPU AVAILABILITY")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("GPU not available - training will use CPU (slower)")
            return False
    except Exception as e:
        print(f"GPU check failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("SMILES-BASED QWEN PIPELINE - VALIDATION SUITE")
    print("="*70)
    
    results = {}
    
    # Test 1: Dependencies
    results['dependencies'] = test_dependencies()
    
    if not results['dependencies']:
        print("\n" + "="*70)
        print("VALIDATION FAILED - Missing dependencies")
        print("="*70)
        return False
    
    # Test 2: GPU
    results['gpu'] = test_gpu()
    
    # Test 3: CSV files
    results['csv'] = test_csv_availability()
    
    # Test 4: Tokenizer
    try:
        test_tokenizer_module()
        results['tokenizer'] = True
    except Exception as e:
        print(f"Tokenizer test failed: {e}")
        results['tokenizer'] = False
    
    # Test 5: Dataset
    try:
        test_dataset_module()
        results['dataset'] = True
    except Exception as e:
        print(f"Dataset test failed: {e}")
        results['dataset'] = False
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    all_passed = all([
        results['dependencies'],
        results['tokenizer'],
        results['dataset'],
        results['csv'],
    ])
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "? FAIL"
        print(f"  {status}: {test_name.capitalize()}")
    
    print("\n" + "="*70)
    
    if all_passed:
        print("ALL TESTS PASSED - Ready to train!")
        print("\n  Next steps:")
        print("  1. Review CLI args: python train_pretrain.py --help")
        print("  2. Start training: python train_pretrain.py")
        print("  3. Monitor logs: tail -f logs/training.log")
        print("="*70)
        return True
    else:
        print("SOME TESTS FAILED - Please fix above issues")
        print("="*70)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
