# Virtual LNP

SMILES-driven lipid nanoparticle (LNP) modeling pipeline with two stages:

1. SMILES causal language-model pretraining on Qwen-1.8B
2. Regression fine-tuning for downstream property prediction

This repository is organized for local experimentation and Linux server training. The instructions below match the current project structure and CLI behavior.



## Repository Layout

```text
.
|-- train_pretrain.py              # Stage 1 pretraining entry point
|-- train_regression.py            # Stage 2 regression entry point
|-- validate_enviroment.py         # Dependency / tokenizer / dataset quick check
|-- requirements.txt
|-- README.md
|-- config/
|   |-- pretrain_cli.py            # Pretraining CLI definitions
|   `-- regression_cli.py          # Regression CLI definitions
|-- src/
|   |-- dataset.py                 # Pretraining dataset and dataloaders
|   |-- logging_utils.py
|   |-- model_regression.py        # Regression model and dataloaders
|   |-- pretrain_utils.py
|   |-- qwen_utils.py
|   |-- regression_utils.py
|   |-- tokenizer.py               # SMILES tokenizer
|   `-- training_common.py         # Device / distributed training helpers
|-- data/                          # Example pretraining CSVs
`-- AGILE/                         # Bundled regression train/test splits
```

Runtime artifacts are created automatically:

- `models/qwen_1.8b_smiles_pretrained/`
- `models/qwen_1.8b_smiles_regression/`
- `logs/`

## Requirements

Recommended environment:

- Linux for actual training
- Python 3.10
- CUDA-enabled PyTorch for GPU runs

Python packages are pinned in `requirements.txt` and currently include:

- `torch>=2.0.0`
- `transformers==4.33.3`
- `transformers_stream_generator==0.0.5`
- `accelerate>=0.20.0`
- `modelscope==1.9.5`
- `pandas`, `numpy`, `scikit-learn`, `scipy`, `tqdm`, `tensorboard`

## Deployment and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Zhusx979/LNP.git
cd virtual_LNP
```

### 2. Create a Python environment

Using `conda`:

```bash
conda create -n virtual_lnp python=3.10 -y
conda activate virtual_lnp
```

Or using `venv`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you are training on GPU, make sure the installed PyTorch build matches your CUDA runtime. If needed, reinstall PyTorch from the official wheel index for your CUDA version before running training.

### 4. Run the validation script

```bash
python validate_enviroment.py
```

This checks:

- required Python packages
- GPU visibility
- tokenizer logic
- dataset pipeline
- local CSV availability

## Data Requirements

### Stage 1 pretraining CSV

Pretraining CSV files must contain a `SMILES` column.

Example:

```csv
SMILES
CCO
CCCCN(CC)CC
```

### Stage 2 regression CSV

Regression CSV files must contain:

- `SMILES`
- one label column: `TARGET` or `transfection_efficiency`

Example:

```csv
SMILES,TARGET
CCO,0.42
CCCCN(CC)CC,0.67
```

If `--csv` is not provided for regression, the code automatically loads AGILE data from:

```text
AGILE/*/*/{train,test}.csv
```

## Stage 1: Pretraining

Default command:

```bash
python train_pretrain.py
```

Useful examples:

```bash
# Custom data and epochs
python train_pretrain.py \
  --csv-paths data/test_lipids.csv \
  --num-epochs 5 \
  --batch-size 1

# Single GPU
python train_pretrain.py --gpus 0

# CPU
python train_pretrain.py --gpus cpu
```

Useful flags:

- `--model-name`: base Qwen model name for ModelScope download
- `--csv-paths`: one or more pretraining CSVs
- `--mixed-precision {none,fp16,bf16}`
- `--gradient-checkpointing`
- `--gradient-accumulation-steps`
- `--resume-from <checkpoint_dir>`

Get the full CLI:

```bash
python train_pretrain.py --help
```

## Stage 2: Regression Fine-Tuning

This stage expects Stage 1 artifacts to exist first. By default it loads:

```text
models/qwen_1.8b_smiles_pretrained/final_model/
models/qwen_1.8b_smiles_pretrained/tokenizer.json
```

Default command:

```bash
python train_regression.py
```

Useful examples:

```bash
# Specific AGILE subset
python train_regression.py --agile-cell-line Hela --agile-split cliff

# Manual regression CSV
python train_regression.py --csv path/to/labels.csv

# Single GPU
python train_regression.py --gpus 0

# Use a custom pretrained checkpoint
python train_regression.py \
  --pretrained-model-path models/qwen_1.8b_smiles_pretrained/final_model \
  --tokenizer-path models/qwen_1.8b_smiles_pretrained/tokenizer.json
```

Get the full CLI:

```bash
python train_regression.py --help
```

## Multi-GPU Training on Linux

Multi-GPU training now uses DistributedDataParallel and must be launched with `torchrun`.


```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train_pretrain.py --gpus all
```

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train_regression.py --gpus all
```

For a first stability check, start small:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  train_pretrain.py \
  --gpus all \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --mixed-precision bf16
```

## Outputs

### Stage 1 outputs

Default directory:

```text
models/qwen_1.8b_smiles_pretrained/
```

Typical contents:

- `checkpoint-*/`
- `best_model/`
- `final_model/`
- `tokenizer.json`

### Stage 2 outputs

Default directory:

```text
models/qwen_1.8b_smiles_regression/<dataset_name>/
```

Typical contents:

- `best_model/model.pt`
- `final_model/model.pt`
- saved tokenizer copy
- prediction CSVs

### Logs

Metrics and summaries are written under:

```text
logs/
```

Examples:

- `pretrain_metrics.csv`
- `pretrain_validation_metrics.csv`
- `pretrain_summary.json`
- `regression_metrics_<dataset>.csv`
- `regression_summary_<dataset>.json`

## Common Deployment Notes

### 1. Base model download

Stage 1 downloads the base Qwen model through ModelScope unless you resume from an existing checkpoint. The training server must be able to access that model source, or you must prepare the model cache in advance.

### 2. Trust-remote-code support files

The code copies Qwen support files into saved checkpoints so local reloads continue to work. Keep checkpoint directories intact after training.

### 3. GPU memory

Qwen-1.8B is large. If you hit OOM:

- reduce `--batch-size`
- increase `--gradient-accumulation-steps`
- prefer `--mixed-precision bf16` on supported GPUs
- use multi-GPU `torchrun` on Linux
