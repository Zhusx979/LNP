# Virtual LNP

SMILES-based pretraining and regression pipeline for lipid nanoparticle (LNP) modeling.

## Overview

The project is organized as a two-stage pipeline:

1. `train_pretrain.py`
   Self-supervised causal language modeling on SMILES sequences.
2. `train_regression.py`
   Regression fine-tuning on labeled downstream datasets.

Instead of using a generic text tokenizer, the codebase includes a rule-based SMILES tokenizer designed to preserve chemically meaningful tokens, ring indices, brackets, and bond operators.

## Highlights

- SMILES-aware tokenizer for chemistry-preserving sequence encoding
- Causal language modeling for molecular representation pretraining
- Regression fine-tuning for downstream LNP property prediction
- Support for custom CSV datasets and auto-discovered `AGILE/**/test.csv` files
- Lightweight project layout for reproduction and method extension

## Repository Layout

```text
src/
  tokenizer.py          SMILES tokenizer
  dataset.py            pretraining dataset pipeline
  model_regression.py   regression model and data module

configs/
  config_pretrain.json  pretraining configuration

data/                   example pretraining data
AGILE/                  downstream evaluation data
train_pretrain.py       stage 1 entry point
train_regression.py     stage 2 entry point
validate.py             environment and pipeline check
```

## Data Format

### Pretraining

The pretraining CSV specified in `configs/config_pretrain.json` must contain:

```text
SMILES
```

Current default:

```text
data/test_lipids.csv
```

### Regression

Regression CSV files must contain:

```text
SMILES,<TARGET or transfection_efficiency>
```

Supported label columns:

- `TARGET`
- `transfection_efficiency`

If `--csv` is not provided, the regression script attempts to load `AGILE/**/test.csv`.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
pip install modelscope
```

Validate the environment:

```bash
python validate.py
```

Run stage 1 pretraining:

```bash
python train_pretrain.py
```

Run stage 2 regression:

```bash
python train_regression.py
```

Use a custom regression dataset:

```bash
python train_regression.py --csv path/to/your_labels.csv
```

## Outputs

Pretraining outputs are saved to:

```text
models/qwen_1.8b_smiles_pretrained/
```

Typical artifacts:

- `best_model/`
- `final_model/`
- `tokenizer.json`

Regression outputs are saved to:

```text
models/qwen_1.8b_smiles_regression/
```

Logs and metrics are written to:

```text
logs/
```

## Files Most Relevant for Reproduction

- `configs/config_pretrain.json` for hyperparameters and data paths
- `src/tokenizer.py` for SMILES tokenization logic
- `src/dataset.py` for pretraining sample construction
- `src/model_regression.py` for regression head design and label normalization

## Use Case

This repository is intended for researchers working on:

- LNP molecular representation learning
- virtual lipid library modeling
- downstream property prediction from SMILES
- adaptation of language models to chemical sequence tasks
