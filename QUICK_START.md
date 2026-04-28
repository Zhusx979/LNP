# QUICK REFERENCE GUIDE

## ? Project Overview

**Complete pipeline for SMILES-based molecular modeling with Qwen-1.8B**

```
CSV Files (SMILES)
        ¡ý
    [Stage 1: Pretraining]
    - SMILES Tokenizer
    - Causal Language Modeling
    - Qwen-1.8B Fine-tuning
        ¡ý
  Pretrained Model + Tokenizer
        ¡ý
    [Stage 2: Regression] (Awaiting labels)
    - Regression Head
    - Transfection Efficiency Prediction
        ¡ý
  Fine-tuned Model + Metrics
```

---

## ? Quick Start (5 Minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Validate Setup
```bash
python validate.py
```

Expected output:
```
? ALL TESTS PASSED - Ready to train!
```

### 3. Start Training
```bash
python train_pretrain.py
```

Training will:
- Load your CSV files (automatic CSV discovery in root)
- Build SMILES vocabulary (~100-150 tokens)
- Train Qwen-1.8B with mixed precision
- Save checkpoints every 500 steps
- Validate every 500 steps

### 4. Monitor Progress
```bash
# Watch logs in real-time
tail -f logs/training.log

# View metrics
head logs/pretrain_metrics.csv
```

---

## ? File Structure

| File | Purpose |
|------|---------|
| `train_pretrain.py` | **Main training script** (Stage 1) |
| `train_regression.py` | Stage 2 fine-tuning (awaiting labels) |
| `src/tokenizer.py` | SMILES-aware tokenizer (pattern-based) |
| `src/dataset.py` | Data loading & preprocessing |
| `src/model_regression.py` | Regression model (Stage 2) |
| `configs/config_pretrain.json` | Training hyperparameters |
| `validate.py` | Setup validation script |
| `README.md` | Full documentation |

---

## ?? Configuration

### Edit `configs/config_pretrain.json`:

```json
{
  "data": {
    "batch_size": 16,           // ¡û Reduce if OOM (out of memory)
    "validation_split": 0.15,   // ¡û 15% validation set
    "deduplicate": true         // ¡û Remove duplicate SMILES
  },
  "training": {
    "num_epochs": 5,            // ¡û Number of training epochs
    "learning_rate": 5e-5,      // ¡û Learning rate
    "gradient_accumulation_steps": 4, // ¡û Gradient accumulation (for single GPU)
    "eval_steps": 500           // ¡û Validation frequency
  }
}
```

### Quick Adjustments:

| Problem | Solution |
|---------|----------|
| GPU out of memory | Reduce `batch_size` (16¡ú8) or increase `gradient_accumulation_steps` (4¡ú8) |
| Training too slow | Keep current settings (gradient accumulation is standard) |
| Want more validation | Reduce `eval_steps` (500¡ú250) |

---

## ? Monitoring & Outputs

### Training Outputs

```
models/qwen_1.8b_smiles_pretrained/
©À©¤©¤ best_model/
©¦   ©À©¤©¤ pytorch_model.bin      ¡û Best checkpoint
©¦   ©À©¤©¤ config.json
©¦   ©¸©¤©¤ ...
©À©¤©¤ final_model/
©¦   ©À©¤©¤ pytorch_model.bin      ¡û Final model
©¦   ©¸©¤©¤ ...
©À©¤©¤ tokenizer.json              ¡û SMILES tokenizer vocab
©¸©¤©¤ checkpoint-500/
    ©À©¤©¤ pytorch_model.bin      ¡û Intermediate checkpoints
    ©¸©¤©¤ training_state.pt

logs/
©À©¤©¤ training.log               ¡û Training log file
©¸©¤©¤ pretrain_metrics.csv       ¡û Training metrics (loss, perplexity)
```

### Viewing Metrics

```bash
# View CSV in terminal
cat logs/pretrain_metrics.csv

# Or in Python
import pandas as pd
df = pd.read_csv('logs/pretrain_metrics.csv')
print(df)

# Plot with pandas
df.plot(x='epoch', y=['train_loss', 'val_loss'])
```

### Expected Training Curves

- **Loss**: Should decrease smoothly (not jump around)
- **Perplexity**: Should decrease (lower is better)
- **Validation Loss**: Should be close to training loss (no severe overfitting)

---

## ? Testing

### Run All Tests
```bash
python validate.py
```

### Test Individual Components

```bash
# Test tokenizer
python src/tokenizer.py
# Expected: ? PASS for all SMILES examples

# Test dataset
python src/dataset.py
# Expected: Loads CSV, creates batches, shows shapes
```

### Quick Training Test (1 batch)
```python
import sys
sys.path.insert(0, 'src')
from train_pretrain import QwenSMILESPretrainer

trainer = QwenSMILESPretrainer('configs/config_pretrain.json')
trainer.setup()

# Get one batch
batch = next(iter(trainer.train_loader))
print(f"? Batch shape: {batch['input_ids'].shape}")
```

---

## ? Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'transformers'`
**Fix:**
```bash
pip install transformers torch
```

### Issue: `CUDA out of memory`
**Fix:** Edit `configs/config_pretrain.json`:
```json
{
  "data": {"batch_size": 8},
  "training": {"gradient_accumulation_steps": 8}
}
```

### Issue: CSV files not found
**Fix:** Ensure CSV files are in root directory:
```bash
ls -la *.csv
# Should show: LNP_virtual_lipid_library_generated1.csv
#             LNP_virtual_lipid_library_generated2.csv
```

### Issue: GPU not detected
**Check:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show GPU name
```

### Issue: Training very slow
**Likely:** Using CPU instead of GPU. Check:
```bash
nvidia-smi  # Should show GPU usage during training
```

---

## ? Stage 2: When Labels Arrive

### Expected Label Format
```csv
SMILES,transfection_efficiency
CCCCCCCCCCOC(CCCN(C(=O)CCCCC(=O)OC(CCCCCC)CCCCCC)C(C(=O)NCCCCN1CCCCC1CC)C(CCC)CCC)OCCCCCCCCCC,0.85
CCO,0.45
CC(=O)O,0.62
```

### Run Stage 2
```bash
python train_regression.py \
    --csv path/to/labels.csv \
    --pretrained models/qwen_1.8b_smiles_pretrained/best_model \
    --epochs 10
```

---

## ? Key Metrics

### Stage 1 (Pretraining) - What to Expect
- **Initial Loss**: ~8-10 (random predictions)
- **Final Loss**: ~2-4 (after training)
- **Perplexity**: Exp(loss) - should decrease from ~3000 to ~10-50
- **Training Time**: ~1-5 hours on single GPU (depending on data size & GPU)

### Stage 2 (Regression) - What to Expect
- **RMSE**: Depends on label range (normalized values)
- **MAE**: Mean absolute error
- **Correlation**: Pearson R? with actual values

---

## ? Understanding the Code

### Tokenizer Flow
```
SMILES String
    ¡ý
Pattern Matching (Regex)
    ¡ý
Token List (e.g., ['C', 'C', 'O'])
    ¡ý
Vocabulary Lookup
    ¡ý
Token IDs (e.g., [5, 5, 10])
    ¡ý
Padding to max_length=256
    ¡ý
Attention Mask
```

### Dataset Flow
```
CSV Files
    ¡ý
Load & Merge SMILES
    ¡ý
Deduplicate (optional)
    ¡ý
Tokenize Each SMILES
    ¡ý
Create Next-Token Pairs
    ¡ý
(Input: tokens[0:n-1], Target: tokens[1:n])
    ¡ý
PyTorch Dataset
    ¡ý
DataLoader (batching)
```

### Training Flow
```
Load Batch
    ¡ý
Forward Pass (Qwen model)
    ¡ý
Compute Loss (next-token prediction)
    ¡ý
Backward Pass (gradients)
    ¡ý
Gradient Accumulation (4-8 steps)
    ¡ý
Optimizer Step (update weights)
    ¡ý
Log Metrics (loss, perplexity)
    ¡ý
Validation Every 500 Steps
    ¡ý
Save Best Checkpoint
```

---

## ? Tips & Tricks

### 1. Resume Training from Checkpoint
```bash
python train_pretrain.py --resume-from models/qwen_1.8b_smiles_pretrained/checkpoint-1000
```

### 2. Use Different Model Size
Edit `configs/config_pretrain.json`:
```json
{
  "model": {
    "model_name": "Qwen/Qwen-7B"  // Or Qwen-1.8B (default)
  }
}
```

### 3. Adjust Batch Size for Your Hardware
```json
{
  "data": {
    "batch_size": 32  // For GPU with 24GB+ VRAM
  }
}
```

### 4. Save Memory: Gradient Checkpointing
Already enabled in config:
```json
{
  "optimization": {
    "gradient_checkpointing": true  // Trades speed for memory
  }
}
```

### 5. Monitor GPU Usage
```bash
# Watch GPU in real-time
watch -n 0.5 nvidia-smi
```

---

## ? References

- **SMILES Notation**: [Weininger 1988](https://doi.org/10.1021/ci00062a008)
- **Qwen Models**: [Alibaba Qwen](https://github.com/QwenLM/Qwen)
- **HuggingFace**: [transformers library](https://huggingface.co/transformers/)
- **PyTorch**: [pytorch.org](https://pytorch.org)

---

## ? FAQ

**Q: What's the difference between best_model and final_model?**
A: `best_model` has the lowest validation loss. `final_model` is the last checkpoint after all epochs. Usually `best_model` is better.

**Q: Why is Cl treated as one token?**
A: In SMILES, "Cl" is chlorine (one atom). Breaking it to "C" + "l" would corrupt the chemistry.

**Q: Can I use a different model instead of Qwen?**
A: Yes! Edit config: `"model_name": "gpt2"` or any HuggingFace model. But Stage 2 regression might need adjustment.

**Q: What's next after pretraining?**
A: Obtain transfection efficiency labels, then run Stage 2 to fine-tune for your regression task.

**Q: How do I use the trained model after training?**
A: See [README.md](README.md) for detailed inference examples.

---

## ? Status

| Stage | Status | Notes |
|-------|--------|-------|
| **Stage 1: Pretraining** | ? Ready | Run `train_pretrain.py` |
| **Stage 2: Regression** | ? Awaiting | Requires transfection_efficiency labels |

---

**Questions?** Check [README.md](README.md) or review source code comments.
