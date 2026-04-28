# SMILES-Based Qwen Pretraining & Regression Pipeline

## Overview

This project implements a complete two-stage training pipeline for SMILES-based molecular modeling using the Qwen-1.8B language model:

1. **Stage 1: Self-Supervised Pretraining** (Causal Language Modeling)
   - Train Qwen to predict next tokens in SMILES sequences
   - Uses a custom SMILES-aware tokenizer (preserves chemical structure)
   - Input: 200+ LNP lipid SMILES strings

2. **Stage 2: Downstream Regression** (Transfection Efficiency Prediction)
   - Fine-tune pretrained model with regression head
   - Status: Awaiting transfection efficiency labels

---

## Project Structure

```
.
©À©¤©¤ src/
©¦   ©À©¤©¤ tokenizer.py          # SMILES-aware tokenizer
©¦   ©À©¤©¤ dataset.py            # Data loading & preprocessing
©¦   ©À©¤©¤ model_regression.py   # Stage 2: Regression model
©¦   ©¸©¤©¤ utils.py              # Utilities
©À©¤©¤ configs/
©¦   ©¸©¤©¤ config_pretrain.json  # Training configuration
©À©¤©¤ models/                   # Saved models directory
©À©¤©¤ logs/                     # Training logs
©À©¤©¤ data/                     # Data directory
©À©¤©¤ train_pretrain.py         # Stage 1: Training script
©À©¤©¤ train_regression.py       # Stage 2: Training script (awaiting labels)
©À©¤©¤ requirements.txt          # Python dependencies
©¸©¤©¤ README.md                 # This file
```

---

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `torch>=2.0.0` ¡ª PyTorch with CUDA support
- `transformers>=4.30.0` ¡ª HuggingFace Transformers
- `pandas>=1.5.0` ¡ª Data manipulation
- `scikit-learn>=1.2.0` ¡ª ML utilities (normalization, metrics)

### 2. Verify GPU (Optional)

```bash
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

---

## Stage 1: SMILES Pretraining

### Quick Start

```bash
# Option 1: Default configuration (recommended for testing)
python train_pretrain.py

# Option 2: Custom config
python train_pretrain.py --config configs/config_pretrain.json

# Option 3: Resume from checkpoint
python train_pretrain.py --resume-from models/qwen_1.8b_smiles_pretrained/checkpoint-1000
```

### What Happens

1. **Tokenization**: Loads your CSV files and builds SMILES vocabulary
   - Treats Cl, Br as single tokens (chemistry-aware)
   - Preserves ring numbers, brackets, operators
   - Vocab size: ~100-150 tokens

2. **Dataset Creation**: Converts SMILES to next-token prediction pairs
   - Input sequence: [CLS] + token[0:n-1]
   - Target sequence: token[1:n] + [EOS]
   - Pads to max_length=256

3. **Model Training**: Fine-tunes Qwen-1.8B
   - Mixed precision (float16) for efficiency
   - Gradient accumulation for single GPU
   - Validates every 500 steps

4. **Outputs**:
   - `models/qwen_1.8b_smiles_pretrained/best_model/` ¡ª Best checkpoint
   - `models/qwen_1.8b_smiles_pretrained/final_model/` ¡ª Final model
   - `models/qwen_1.8b_smiles_pretrained/tokenizer.json` ¡ª Saved tokenizer
   - `logs/pretrain_metrics.csv` ¡ª Training metrics

### Configuration

Edit `configs/config_pretrain.json` to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 16 | Batch size per step |
| `learning_rate` | 5e-5 | Initial learning rate |
| `num_epochs` | 5 | Training epochs |
| `gradient_accumulation_steps` | 4 | Steps for gradient accumulation |
| `max_seq_length` | 256 | Max SMILES token length |
| `eval_steps` | 500 | Validation frequency |

### Monitoring Training

```bash
# Watch training metrics
tail -f logs/training.log

# View metrics CSV
cat logs/pretrain_metrics.csv
```

---

## Stage 2: Transfection Efficiency Regression

### Status: AWAITING LABELS

To run Stage 2, you need a CSV file with:
- **Column 1**: `SMILES` (SMILES strings)
- **Column 2**: `transfection_efficiency` (float values, e.g., 0.0-1.0)

### Example CSV Format

```
SMILES,transfection_efficiency
CCO,0.75
CC(=O)O,0.82
CCCCCCCCCCOC(...),0.91
```

### How to Run (Once Labels Available)

```bash
python train_regression.py \
    --csv path/to/labels.csv \
    --pretrained models/qwen_1.8b_smiles_pretrained/best_model \
    --tokenizer models/qwen_1.8b_smiles_pretrained/tokenizer.json \
    --epochs 10 \
    --lr 1e-5
```

### Model Architecture

- **Backbone**: Pretrained Qwen-1.8B from Stage 1
- **Head**: Linear regression (hidden_dim ¡ú 1)
- **Pooling**: Mean pooling over sequence (masked)
- **Loss**: MSELoss
- **Metrics**: RMSE, MAE, Pearson correlation

---

## Testing & Validation

### 1. Test Tokenizer

```bash
python src/tokenizer.py
```

Expected output:
```
============================================================
SMILES TOKENIZER TEST
============================================================

? PASS: CCO
? PASS: c1ccccc1
? PASS: CCl
? PASS: CBr
...
```

### 2. Test Dataset Pipeline

```bash
python src/dataset.py
```

Verifies:
- CSV loading
- SMILES tokenization
- DataLoader creation

### 3. End-to-End Training Test (Quick)

```python
# In Python
import sys
sys.path.insert(0, 'src')
from train_pretrain import QwenSMILESPretrainer

trainer = QwenSMILESPretrainer('configs/config_pretrain.json')
trainer.setup()

# Test one batch
batch = next(iter(trainer.train_loader))
print(f"Batch shapes: {batch['input_ids'].shape}")
```

---

## Key Design Decisions

### 1. SMILES Tokenizer (Pattern-Based, Not BPE)

**Why?** SMILES has strict chemical syntax. Standard tokenizers break it:
- BPE might split "Cl" ¡ú "C" + "l" (invalid chemistry)
- Pattern-based preserves "Cl" as single token (correct)

### 2. Qwen-1.8B (Not Larger Models)

**Why?** 
- Sufficient capacity for SMILES (~100-150 token vocabulary)
- Fits on single GPU (~7GB with float16)
- Faster training and inference

### 3. Causal LM (Not Masked LM)

**Why?**
- Standard for LLMs (GPT-style)
- Cleaner for next-token prediction
- Enables SMILES generation from seed tokens

### 4. Gradient Accumulation (Single GPU)

**Why?**
- Simulate larger batches without OOM
- Accumulate gradients over 4-8 steps
- Standard practice for single-GPU training

---

## Troubleshooting

### Issue: `CUDA out of memory`

**Solutions:**
1. Reduce `batch_size` in config (8 ¡ú 4)
2. Increase `gradient_accumulation_steps` (4 ¡ú 8)
3. Set `torch_dtype` to `float16` in config

### Issue: `CSV file not found`

**Solution:** Ensure CSV files are in the root directory:
```bash
ls -la *.csv  # Should show both generated1 and generated2
```

### Issue: `Model loading fails`

**Solution:** Install transformers:
```bash
pip install -U transformers torch
```

### Issue: Very slow training

**Solution:** Enable GPU:
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

---

## Next Steps

### After Stage 1 Completes

1. **Evaluate pretraining**:
   - Check `logs/pretrain_metrics.csv`
   - Verify perplexity decreased
   - Optionally generate SMILES from seed tokens

2. **Prepare for Stage 2**:
   - Obtain transfection efficiency labels
   - Format as CSV: [SMILES, transfection_efficiency]
   - Place in `data/` directory

3. **Run Stage 2**:
   - Execute `train_regression.py` with label CSV
   - Fine-tune regression head
   - Evaluate on test set

---

## Citation & References

This pipeline is built on:
- **Qwen Models**: [alibaba/Qwen](https://github.com/QwenLM/Qwen)
- **HuggingFace Transformers**: [huggingface/transformers](https://github.com/huggingface/transformers)
- **SMILES Notation**: [Weininger, JCICS 1988](https://doi.org/10.1021/ci00062a008)

---

## License

MIT License

---

## Contact

For questions or issues, please check:
1. Training logs: `logs/training.log`
2. Configuration: `configs/config_pretrain.json`
3. Source code comments in `src/`

---

**Status**: ? Stage 1 ready to train | ? Stage 2 awaiting transfection efficiency labels
