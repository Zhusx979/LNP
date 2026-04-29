# Virtual LNP

SMILES-driven lipid nanoparticle (LNP) modeling pipeline: molecular language pretraining followed by downstream regression.

## Overview

This repository provides a two-stage workflow for learning LNP-related representations from SMILES and transferring them to property prediction tasks:

1. `train_pretrain.py`
   Self-supervised causal language modeling on SMILES sequences
2. `train_regression.py`
   Regression fine-tuning on labeled data (`transfection_efficiency` / `TARGET`)

Instead of a generic text tokenizer, the project includes a rule-based SMILES tokenizer that preserves chemically meaningful patterns such as ring indices, brackets, and bond operators.

## Highlights

- Chemistry-aware SMILES tokenization
- Molecular sequence pretraining on Qwen-1.8B
- Downstream regression fine-tuning and evaluation (RMSE / MAE / Pearson)
- Support for custom CSVs and AGILE auto-discovery (`AGILE/*/*/{train,test}.csv`)
- Lightweight layout for reproducibility and extension

## Repository Structure

```text

train_pretrain.py            # Stage 1: SMILES pretraining entry point
train_regression.py          # Stage 2: regression fine-tuning entry point
validate_enviroment.py       # Environment and pipeline quick check
requirements.txt
config/
   pretrain_cli.py           # Pretraining CLI arguments
   regression_cli.py         # Regression CLI arguments
src/
   tokenizer.py              # SMILES tokenizer
   dataset.py                # Pretraining data module
   model_regression.py       # Regression model and data module
   pretrain_utils.py
   regression_utils.py
   training_common.py
data/                        # Example pretraining data
AGILE/                       # Downstream evaluation data
```

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
pip install modelscope
```

### 2) Run environment check

```bash
python validate_enviroment.py
```

### 3) Stage 1: pretraining

```bash
python train_pretrain.py
```

Common customizations:

```bash
# Specify data and training epochs
python train_pretrain.py --csv-paths data/test_lipids.csv --num-epochs 5 --batch-size 4

# GPU selection: single GPU / multi-GPU / all visible GPUs / CPU
python train_pretrain.py --gpus 0
python train_pretrain.py --gpus 0,1
python train_pretrain.py --gpus all
python train_pretrain.py --gpus cpu
```

### 4) Stage 2: regression fine-tuning

```bash
# Use default AGILE subset (defined in regression_cli.py)
python train_regression.py

# Select a specific AGILE subset
python train_regression.py --agile-cell-line Hela --agile-split cliff

# Use a custom regression CSV
python train_regression.py --csv path/to/your_labels.csv
```

```bash
# GPU selection
python train_regression.py --gpus 0
python train_regression.py --gpus 0,1
python train_regression.py --gpus all
python train_regression.py --gpus cpu
```

## Data Format

Pretraining CSV must include:

```text
SMILES
```

Regression CSV must include:

```text
SMILES,<TARGET or transfection_efficiency>
```

Supported label columns:

- `TARGET`
- `transfection_efficiency`

If `--csv` is not provided, the regression script automatically tries to load `AGILE/*/*/{train,test}.csv`.

## Outputs

Stage 1 artifacts are saved by default to:

```text
models/qwen_1.8b_smiles_pretrained/
```

Typical artifacts:

- `best_model/`
- `final_model/`
- `tokenizer.json`

Stage 2 artifacts are saved by default to:

```text
models/qwen_1.8b_smiles_regression/<dataset_name>/
```

Logs and metrics are written to:

```text
logs/
```

## Key Files for Reproduction

- `train_pretrain.py`: pretraining pipeline and training loop
- `train_regression.py`: regression fine-tuning and evaluation
- `src/tokenizer.py`: SMILES tokenization and encoding logic
- `src/dataset.py`: pretraining sample construction
- `src/model_regression.py`: regression head and forward logic

## Use Cases

- LNP molecular representation learning
- Virtual lipid library modeling
- SMILES-based property prediction
- Adapting general language models to chemical sequence tasks
