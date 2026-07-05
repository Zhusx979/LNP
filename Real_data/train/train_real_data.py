from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
HF_CACHE_ROOT = SCRIPT_DIR / ".hf_cache"
HF_MODULES_CACHE = HF_CACHE_ROOT / "modules"
HF_HUB_CACHE = HF_CACHE_ROOT / "hub"

HF_MODULES_CACHE.mkdir(parents=True, exist_ok=True)
HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_ROOT))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("HF_MODULES_CACHE", str(HF_MODULES_CACHE))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_regression import QwenRegressionModel
from src.qwen_utils import ensure_local_qwen_code
from src.regression_utils import compute_regression_metrics, json_ready_dict
from src.tokenizer import SMILESTokenizer


TARGET_COLUMN_CANDIDATES = ("target", "TARGET", "transfection_efficiency")
SMILES_COLUMN_CANDIDATES = ("smiles", "SMILES")
COMBO_COLUMN_CANDIDATES = ("combo", "Combo", "COMBO")
LORA_TARGET_MODULES = ["c_attn", "c_proj", "w1", "w2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the pretrained Qwen SMILES model on Real_data Excel labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        default=str(PROJECT_ROOT / "Real_data" / "real_data_smiles_target.xlsx"),
        help="Path to the real-data Excel file (.xlsx preferred, .xls also supported if pandas engine is available).",
    )
    parser.add_argument(
        "--pretrained-model-path",
        default=str(PROJECT_ROOT / "models" / "qwen_1.8b_smiles_pretrained" / "final_model"),
        help="Stage-1 pretrained Qwen checkpoint directory.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=str(PROJECT_ROOT / "models" / "qwen_1.8b_smiles_pretrained" / "tokenizer.json"),
        help="Tokenizer JSON produced by pretraining.",
    )
    parser.add_argument(
        "--target-transform",
        choices=["log2", "log10", "normalize", "none"],
        default="log2",
        help="Target preprocessing applied before regression training.",
    )
    parser.add_argument(
        "--finetune-method",
        choices=["head", "lora"],
        default="head",
        help="Use the default prediction head or LoRA adapters.",
    )
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Mini-batch size.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Number of steps to accumulate gradients before optimizer step.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Validation ratio for the single Excel dataset.",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Test ratio for the single Excel dataset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Tokenizer sequence length.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Keep 0 on Windows unless needed.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Target device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=["auto", "none", "fp16", "bf16"],
        default="auto",
        help="AMP mode used during training when CUDA is available.",
    )
    parser.add_argument(
        "--train-embeddings",
        action="store_true",
        default=False,
        help="When using head fine-tuning, also train resized input embeddings.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout used by the regression head.",
    )
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only clean/split/export the dataset and config. Skip model loading and training.",
    )
    return parser.parse_args()


def import_transformers():
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: transformers. Run `pip install -r requirements.txt openpyxl` first."
        ) from exc
    return AutoModelForCausalLM


def import_peft():
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "LoRA mode requires `peft`. Run `pip install peft` before using --finetune-method lora."
        ) from exc
    return LoraConfig, TaskType, get_peft_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_amp_dtype(device: torch.device, mode: str) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    if mode == "none":
        return None
    if mode == "fp16":
        return torch.float16
    if mode == "bf16":
        return torch.bfloat16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def resolve_required_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    raise ValueError(f"Missing required column. Expected one of: {list(candidates)}")


def load_real_data_frame(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    smiles_col = resolve_required_column(df, SMILES_COLUMN_CANDIDATES)
    target_col = resolve_required_column(df, TARGET_COLUMN_CANDIDATES)

    combo_col = None
    try:
        combo_col = resolve_required_column(df, COMBO_COLUMN_CANDIDATES)
    except ValueError:
        combo_col = None

    subset = pd.DataFrame(
        {
            "combo": (
                df[combo_col].astype(str).str.strip()
                if combo_col is not None
                else [f"sample_{idx:05d}" for idx in range(len(df))]
            ),
            "smiles": df[smiles_col].astype(str).str.strip(),
            "target": pd.to_numeric(df[target_col], errors="coerce"),
        }
    )
    subset = subset.replace({"": np.nan}).dropna(subset=["smiles", "target"]).reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"No usable rows found in {path}")
    return subset


def transform_targets(values: np.ndarray, method: str) -> Tuple[np.ndarray, Dict[str, float]]:
    values = np.asarray(values, dtype=np.float64)
    metadata: Dict[str, float] = {"method": method}

    if method == "log2":
        if np.any(values <= 0):
            raise ValueError("log2 transform requires all target values to be > 0.")
        return np.log2(values), metadata

    if method == "log10":
        if np.any(values <= 0):
            raise ValueError("log10 transform requires all target values to be > 0.")
        return np.log10(values), metadata

    if method == "normalize":
        mean = float(values.mean())
        std = float(values.std())
        if math.isclose(std, 0.0):
            raise ValueError("normalize transform requires non-constant target values.")
        metadata["mean"] = mean
        metadata["std"] = std
        return (values - mean) / std, metadata

    if method == "none":
        return values.copy(), metadata

    raise ValueError(f"Unsupported target transform: {method}")


def inverse_transform_targets(values: np.ndarray, metadata: Dict[str, float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    method = metadata["method"]
    if method == "log2":
        return np.power(2.0, values)
    if method == "log10":
        return np.power(10.0, values)
    if method == "normalize":
        return values * float(metadata["std"]) + float(metadata["mean"])
    if method == "none":
        return values
    raise ValueError(f"Unsupported target transform: {method}")


def compute_split_sizes(total: int, validation_split: float, test_split: float) -> Tuple[int, int, int]:
    if total < 3:
        raise ValueError("At least 3 rows are required for train/val/test splitting.")
    if validation_split < 0 or test_split < 0 or validation_split + test_split >= 1:
        raise ValueError("validation_split + test_split must be >= 0 and < 1.")

    n_test = max(1, int(round(total * test_split)))
    n_val = max(1, int(round(total * validation_split)))
    n_train = total - n_val - n_test

    if n_train < 1:
        deficit = 1 - n_train
        while deficit > 0 and n_val > 1:
            n_val -= 1
            deficit -= 1
        while deficit > 0 and n_test > 1:
            n_test -= 1
            deficit -= 1
        n_train = total - n_val - n_test

    if min(n_train, n_val, n_test) < 1:
        raise ValueError("Unable to create non-empty train/val/test splits. Adjust split ratios.")

    return n_train, n_val, n_test


def split_dataframe(
    df: pd.DataFrame,
    validation_split: float,
    test_split: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_train, n_val, n_test = compute_split_sizes(len(df), validation_split, test_split)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:n_train + n_val + n_test]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df


class SmilesRegressionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: SMILESTokenizer, max_length: int):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        encoded = self.tokenizer.encode(
            row["smiles"],
            add_special_tokens=True,
            padding=True,
            truncation=True,
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "label": torch.tensor(float(row["target_transformed"]), dtype=torch.float32),
        }


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype]):
    if device.type == "cuda" and amp_dtype is not None:
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


@dataclass
class RunLayout:
    run_name: str
    run_dir: Path
    prepared_dir: Path
    reports_dir: Path
    checkpoints_dir: Path
    best_checkpoint_dir: Path
    final_checkpoint_dir: Path
    config_path: Path
    metadata_path: Path


def build_run_layout(run_name: str) -> RunLayout:
    run_dir = SCRIPT_DIR / "runs" / run_name
    prepared_dir = SCRIPT_DIR / "prepared_data" / run_name
    reports_dir = run_dir / "reports"
    checkpoints_dir = run_dir / "checkpoints"
    best_checkpoint_dir = checkpoints_dir / "best"
    final_checkpoint_dir = checkpoints_dir / "final"
    config_path = SCRIPT_DIR / "configs" / f"{run_name}.json"
    metadata_path = run_dir / "run_metadata.json"

    for path in (
        run_dir,
        prepared_dir,
        reports_dir,
        checkpoints_dir,
        best_checkpoint_dir,
        final_checkpoint_dir,
        config_path.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)

    return RunLayout(
        run_name=run_name,
        run_dir=run_dir,
        prepared_dir=prepared_dir,
        reports_dir=reports_dir,
        checkpoints_dir=checkpoints_dir,
        best_checkpoint_dir=best_checkpoint_dir,
        final_checkpoint_dir=final_checkpoint_dir,
        config_path=config_path,
        metadata_path=metadata_path,
    )


def prepare_split_exports(layout: RunLayout, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_df.to_excel(layout.prepared_dir / "train.xlsx", index=False)
    val_df.to_excel(layout.prepared_dir / "val.xlsx", index=False)
    test_df.to_excel(layout.prepared_dir / "test.xlsx", index=False)


def make_run_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"real_data_{args.finetune_method}_{args.target_transform}_{timestamp}"


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_ready_dict(payload), handle, ensure_ascii=False, indent=2)


def configure_trainable_parameters(model: QwenRegressionModel, method: str, train_embeddings: bool) -> None:
    if method == "lora":
        for parameter in model.head.parameters():
            parameter.requires_grad = True
        return

    for parameter in model.model.parameters():
        parameter.requires_grad = False

    if train_embeddings:
        for parameter in model.model.get_input_embeddings().parameters():
            parameter.requires_grad = True

    for parameter in model.head.parameters():
        parameter.requires_grad = True


def create_regression_model(
    args: argparse.Namespace,
    tokenizer: SMILESTokenizer,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
) -> QwenRegressionModel:
    AutoModelForCausalLM = import_transformers()
    ensure_local_qwen_code(Path(args.pretrained_model_path))
    dtype = amp_dtype or torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    base_model.resize_token_embeddings(len(tokenizer))

    if args.finetune_method == "lora":
        LoraConfig, TaskType, get_peft_model = import_peft()
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=LORA_TARGET_MODULES,
        )
        base_model = get_peft_model(base_model, lora_config)

    model = QwenRegressionModel(base_model, dropout=args.dropout)
    configure_trainable_parameters(model, args.finetune_method, args.train_embeddings)
    model = model.to(device=device, dtype=dtype)
    return model


def evaluate_model(
    model: QwenRegressionModel,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    amp_dtype: Optional[torch.dtype],
    transform_metadata: Dict[str, float],
    split_name: str,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    transformed_predictions: List[float] = []
    transformed_labels: List[float] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            labels = batch.pop("label").view(-1)
            with autocast_context(device, amp_dtype):
                predictions = model(**batch).view(-1)
                loss = criterion(predictions, labels)

            total_loss += float(loss.item())
            transformed_predictions.extend(predictions.detach().cpu().tolist())
            transformed_labels.extend(labels.detach().cpu().tolist())

    transformed_predictions_np = np.asarray(transformed_predictions, dtype=np.float64)
    transformed_labels_np = np.asarray(transformed_labels, dtype=np.float64)
    original_predictions = inverse_transform_targets(transformed_predictions_np, transform_metadata)
    original_labels = inverse_transform_targets(transformed_labels_np, transform_metadata)

    metrics = compute_regression_metrics(original_labels, original_predictions)
    metrics["split"] = split_name
    metrics["loss_transformed"] = (
        total_loss / max(1, len(loader))
        if loader is not None
        else float("nan")
    )
    metrics["num_samples"] = int(len(original_labels))

    base_frame = loader.dataset.frame.reset_index(drop=True).copy()
    prediction_frame = pd.DataFrame(
        {
            "combo": base_frame["combo"],
            "smiles": base_frame["smiles"],
            "target_original": original_labels,
            "prediction_original": original_predictions,
            "target_transformed": transformed_labels_np,
            "prediction_transformed": transformed_predictions_np,
            "error_original": original_predictions - original_labels,
            "abs_error_original": np.abs(original_predictions - original_labels),
        }
    )
    return metrics, prediction_frame


def save_checkpoint(
    model: QwenRegressionModel,
    tokenizer: SMILESTokenizer,
    checkpoint_dir: Path,
    finetune_method: str,
    metadata: Dict,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(checkpoint_dir / "tokenizer.json")
    write_json(checkpoint_dir / "run_metadata.json", metadata)

    if finetune_method == "head":
        state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        torch.save(state_dict, checkpoint_dir / "model.pt")
        return

    model.model.save_pretrained(checkpoint_dir / "adapter")
    head_state = {key: value.detach().cpu() for key, value in model.head.state_dict().items()}
    torch.save(head_state, checkpoint_dir / "head.pt")


def run_training(args: argparse.Namespace) -> Dict:
    set_seed(args.seed)

    data_path = Path(args.data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Training data file not found: {data_path}")

    frame = load_real_data_frame(data_path)
    transformed_targets, transform_metadata = transform_targets(frame["target"].to_numpy(dtype=np.float64), args.target_transform)
    frame = frame.copy()
    frame["target_original"] = frame["target"].astype(float)
    frame["target_transformed"] = transformed_targets

    run_name = make_run_name(args)
    layout = build_run_layout(run_name)

    config_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "script": str(Path(__file__).resolve()),
        "data_path": str(data_path),
        "pretrained_model_path": str(Path(args.pretrained_model_path).resolve()),
        "tokenizer_path": str(Path(args.tokenizer_path).resolve()),
        "arguments": vars(args),
        "target_transform": transform_metadata,
    }
    write_json(layout.config_path, config_payload)

    train_df, val_df, test_df = split_dataframe(
        frame,
        validation_split=args.validation_split,
        test_split=args.test_split,
        seed=args.seed,
    )
    prepare_split_exports(layout, train_df, val_df, test_df)

    metadata = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "prepared",
        "finetune_method": args.finetune_method,
        "target_transform": transform_metadata,
        "paths": {
            "data_path": str(data_path),
            "pretrained_model_path": str(Path(args.pretrained_model_path).resolve()),
            "tokenizer_path": str(Path(args.tokenizer_path).resolve()),
            "run_dir": str(layout.run_dir.resolve()),
            "prepared_dir": str(layout.prepared_dir.resolve()),
            "reports_dir": str(layout.reports_dir.resolve()),
            "best_checkpoint_dir": str(layout.best_checkpoint_dir.resolve()),
            "final_checkpoint_dir": str(layout.final_checkpoint_dir.resolve()),
            "config_path": str(layout.config_path.resolve()),
        },
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "train_config": {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_grad_norm": args.max_grad_norm,
            "max_length": args.max_length,
            "seed": args.seed,
            "device": args.device,
            "mixed_precision": args.mixed_precision,
            "dropout": args.dropout,
            "train_embeddings": args.train_embeddings,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": deepcopy(LORA_TARGET_MODULES),
        },
        "dataset_summary": {
            "num_rows": int(len(frame)),
            "target_min": float(frame["target_original"].min()),
            "target_max": float(frame["target_original"].max()),
            "target_mean": float(frame["target_original"].mean()),
            "target_std": float(frame["target_original"].std()),
        },
    }
    write_json(layout.metadata_path, metadata)

    if args.prepare_only:
        metadata["status"] = "prepare_only_complete"
        write_json(layout.metadata_path, metadata)
        return metadata

    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(device, args.mixed_precision)

    tokenizer = SMILESTokenizer(max_length=args.max_length)
    tokenizer.load(args.tokenizer_path)
    tokenizer.build_vocab(frame["smiles"].tolist())

    train_dataset = SmilesRegressionDataset(train_df, tokenizer, args.max_length)
    val_dataset = SmilesRegressionDataset(val_df, tokenizer, args.max_length)
    test_dataset = SmilesRegressionDataset(test_df, tokenizer, args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_regression_model(args, tokenizer, device, amp_dtype)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=(amp_dtype == torch.float16 and device.type == "cuda"))

    metrics_history: List[Dict[str, float]] = []
    best_val_rmse = float("inf")
    best_val_metrics: Optional[Dict[str, float]] = None
    best_test_metrics: Optional[Dict[str, float]] = None
    best_val_predictions: Optional[pd.DataFrame] = None
    best_test_predictions: Optional[pd.DataFrame] = None

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            labels = batch.pop("label").view(-1)
            should_step = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or (batch_index + 1) == len(train_loader)
            )
            with autocast_context(device, amp_dtype):
                predictions = model(**batch).view(-1)
                raw_loss = criterion(predictions, labels)
                loss = raw_loss / args.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += float(raw_loss.item())

            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics, val_predictions = evaluate_model(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            amp_dtype=amp_dtype,
            transform_metadata=transform_metadata,
            split_name="val",
        )
        test_metrics, test_predictions = evaluate_model(
            model=model,
            loader=test_loader,
            device=device,
            criterion=criterion,
            amp_dtype=amp_dtype,
            transform_metadata=transform_metadata,
            split_name="test",
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss_transformed": train_loss,
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_r2": test_metrics["r2"],
        }
        metrics_history.append(epoch_record)

        if np.isfinite(val_metrics["rmse"]) and val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse"])
            best_val_metrics = val_metrics
            best_test_metrics = test_metrics
            best_val_predictions = val_predictions
            best_test_predictions = test_predictions

            metadata["status"] = "training_in_progress"
            metadata["best_epoch"] = epoch + 1
            metadata["best_val_rmse"] = best_val_rmse
            write_json(layout.metadata_path, metadata)
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                checkpoint_dir=layout.best_checkpoint_dir,
                finetune_method=args.finetune_method,
                metadata=metadata,
            )

    save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        checkpoint_dir=layout.final_checkpoint_dir,
        finetune_method=args.finetune_method,
        metadata=metadata,
    )

    final_val_metrics, final_val_predictions = evaluate_model(
        model=model,
        loader=val_loader,
        device=device,
        criterion=criterion,
        amp_dtype=amp_dtype,
        transform_metadata=transform_metadata,
        split_name="final_val",
    )
    final_test_metrics, final_test_predictions = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        criterion=criterion,
        amp_dtype=amp_dtype,
        transform_metadata=transform_metadata,
        split_name="final_test",
    )

    pd.DataFrame(metrics_history).to_csv(layout.reports_dir / "metrics_history.csv", index=False)
    if best_val_predictions is not None:
        best_val_predictions.to_excel(layout.reports_dir / "best_val_predictions.xlsx", index=False)
    if best_test_predictions is not None:
        best_test_predictions.to_excel(layout.reports_dir / "best_test_predictions.xlsx", index=False)
    final_val_predictions.to_excel(layout.reports_dir / "final_val_predictions.xlsx", index=False)
    final_test_predictions.to_excel(layout.reports_dir / "final_test_predictions.xlsx", index=False)

    metadata["status"] = "training_complete"
    metadata["device_resolved"] = str(device)
    metadata["amp_dtype"] = str(amp_dtype) if amp_dtype is not None else "none"
    metadata["trainable_parameter_count"] = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    metadata["total_parameter_count"] = int(sum(parameter.numel() for parameter in model.parameters()))
    metadata["best_val_rmse"] = best_val_rmse
    metadata["best_val_metrics"] = best_val_metrics
    metadata["best_test_metrics"] = best_test_metrics
    metadata["final_val_metrics"] = final_val_metrics
    metadata["final_test_metrics"] = final_test_metrics
    write_json(layout.metadata_path, metadata)
    write_json(layout.reports_dir / "summary.json", metadata)
    write_json(
        SCRIPT_DIR / "latest_run.json",
        {
            "run_name": run_name,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(layout.run_dir.resolve()),
            "best_checkpoint_dir": str(layout.best_checkpoint_dir.resolve()),
            "final_checkpoint_dir": str(layout.final_checkpoint_dir.resolve()),
            "metadata_path": str(layout.metadata_path.resolve()),
        },
    )
    return metadata


def print_summary(metadata: Dict) -> None:
    print("=" * 72)
    print(f"run_name          : {metadata['run_name']}")
    print(f"status            : {metadata['status']}")
    print(f"finetune_method   : {metadata['finetune_method']}")
    print(f"target_transform  : {metadata['target_transform']['method']}")
    print(f"prepared_dir      : {metadata['paths']['prepared_dir']}")
    print(f"run_dir           : {metadata['paths']['run_dir']}")
    print(f"best_checkpoint   : {metadata['paths']['best_checkpoint_dir']}")
    print(f"final_checkpoint  : {metadata['paths']['final_checkpoint_dir']}")
    print(f"split_sizes       : {metadata['split_sizes']}")
    if "best_val_rmse" in metadata:
        print(f"best_val_rmse     : {metadata['best_val_rmse']}")
    print("=" * 72)


def main() -> None:
    args = parse_args()
    metadata = run_training(args)
    print_summary(metadata)


if __name__ == "__main__":
    main()
