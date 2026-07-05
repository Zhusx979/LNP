from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
TRAIN_DIR = PROJECT_ROOT / "Real_data" / "train"
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
from src.tokenizer import SMILESTokenizer


SMILES_COLUMN_CANDIDATES = ("SMILES", "smiles")
COMBO_COLUMN_CANDIDATES = ("Combo", "combo", "COMBO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on the waiting-hub candidate Excel with a fine-tuned Real_data model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidate-path",
        default=str(PROJECT_ROOT / "data" / "waiting hub" / "候选库4万.xlsx"),
        help="Path to the candidate Excel file containing Combo and SMILES columns.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Specific fine-tuning run directory under Real_data/train/runs. If omitted, use latest_run.json.",
    )
    parser.add_argument(
        "--checkpoint",
        choices=["best", "final"],
        default="best",
        help="Which saved checkpoint from the training run to use.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Target device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override tokenizer sequence length. Leave unset to use the saved tokenizer value.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Only validate candidate columns and selected training metadata. Skip model loading and prediction.",
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
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError(
            "LoRA inference requires `peft`. Run `pip install peft` before loading a LoRA checkpoint."
        ) from exc
    return PeftModel


def resolve_required_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    raise ValueError(f"Missing required column. Expected one of: {list(candidates)}")


def load_candidate_frame(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    combo_col = resolve_required_column(df, COMBO_COLUMN_CANDIDATES)
    smiles_col = resolve_required_column(df, SMILES_COLUMN_CANDIDATES)
    subset = pd.DataFrame(
        {
            "Combo": df[combo_col].astype(str).str.strip(),
            "SMILES": df[smiles_col].astype(str).str.strip(),
        }
    )
    subset = subset.replace({"": np.nan}).dropna(subset=["Combo", "SMILES"]).reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"No usable candidate rows found in {path}")
    return subset


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_model_dtype(metadata: Dict, device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None

    dtype_name = str(metadata.get("amp_dtype", "none"))
    if dtype_name == "torch.bfloat16":
        return torch.bfloat16
    if dtype_name == "torch.float16":
        return torch.float16
    return None


def read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_inference_progress_stages(batch_count: int) -> Dict[str, Dict[str, int]]:
    """Describe the visible stages in waiting-hub inference."""
    return {
        "load_candidates": {"total": 1},
        "resolve_run": {"total": 1},
        "load_checkpoint": {"total": 1},
        "predict_batches": {"total": batch_count},
    }


def summarize_predictions(predictions: np.ndarray, top_k: int = 10) -> Dict[str, object]:
    """Summarize a vector of unlabeled predictions."""
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.size == 0:
        return {
            "prediction_count": 0,
            "prediction_min": float("nan"),
            "prediction_max": float("nan"),
            "prediction_mean": float("nan"),
            "prediction_median": float("nan"),
            "prediction_std": float("nan"),
            "prediction_p05": float("nan"),
            "prediction_p25": float("nan"),
            "prediction_p75": float("nan"),
            "prediction_p95": float("nan"),
            "top_predictions": [],
        }

    top_k = max(1, min(int(top_k), predictions.size))
    ranked = np.sort(predictions)[::-1]
    return {
        "prediction_count": int(predictions.size),
        "prediction_min": float(np.min(predictions)),
        "prediction_max": float(np.max(predictions)),
        "prediction_mean": float(np.mean(predictions)),
        "prediction_median": float(np.median(predictions)),
        "prediction_std": float(np.std(predictions)),
        "prediction_p05": float(np.percentile(predictions, 5)),
        "prediction_p25": float(np.percentile(predictions, 25)),
        "prediction_p75": float(np.percentile(predictions, 75)),
        "prediction_p95": float(np.percentile(predictions, 95)),
        "top_predictions": [float(value) for value in ranked[:top_k]],
    }


def resolve_run_directory(run_dir_arg: Optional[str]) -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Specified run directory does not exist: {run_dir}")
        return run_dir

    latest_run_path = TRAIN_DIR / "latest_run.json"
    if not latest_run_path.exists():
        raise FileNotFoundError(
            f"latest_run.json not found at {latest_run_path}. Train a model first or pass --run-dir explicitly."
        )

    latest = read_json(latest_run_path)
    run_dir = Path(latest["run_dir"]).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Saved latest run directory does not exist: {run_dir}")
    return run_dir


def resolve_checkpoint_paths(run_dir: Path, checkpoint_name: str) -> Tuple[Path, Dict]:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Run metadata not found: {metadata_path}")

    metadata = read_json(metadata_path)
    checkpoint_dir = run_dir / "checkpoints" / checkpoint_name
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    return checkpoint_dir, metadata


def inverse_transform_targets(values: np.ndarray, metadata: Dict[str, float]) -> np.ndarray:
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


class CandidateDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: SMILESTokenizer):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        encoded = self.tokenizer.encode(
            row["SMILES"],
            add_special_tokens=True,
            padding=True,
            truncation=True,
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
        }


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def load_regression_model(
    checkpoint_dir: Path,
    metadata: Dict,
    device: torch.device,
    max_length_override: Optional[int],
) -> Tuple[QwenRegressionModel, SMILESTokenizer]:
    AutoModelForCausalLM = import_transformers()

    tokenizer = SMILESTokenizer()
    tokenizer.load(checkpoint_dir / "tokenizer.json")
    if max_length_override is not None:
        tokenizer.max_length = max_length_override

    pretrained_model_path = Path(metadata["paths"]["pretrained_model_path"]).resolve()
    ensure_local_qwen_code(pretrained_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    base_model.resize_token_embeddings(len(tokenizer))

    finetune_method = metadata["finetune_method"]
    if finetune_method == "lora":
        PeftModel = import_peft()
        adapter_dir = checkpoint_dir / "adapter"
        if not adapter_dir.exists():
            raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_dir}")
        base_model = PeftModel.from_pretrained(base_model, adapter_dir)

    dropout = float(metadata["train_config"]["dropout"])
    model = QwenRegressionModel(base_model, dropout=dropout)

    if finetune_method == "head":
        state_path = checkpoint_dir / "model.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Regression checkpoint not found: {state_path}")
        try:
            state_dict = torch.load(state_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        head_path = checkpoint_dir / "head.pt"
        if not head_path.exists():
            raise FileNotFoundError(f"Regression head checkpoint not found: {head_path}")
        try:
            head_state = torch.load(head_path, map_location="cpu", weights_only=True)
        except TypeError:
            head_state = torch.load(head_path, map_location="cpu")
        model.head.load_state_dict(head_state)

    model_dtype = resolve_model_dtype(metadata, device)
    if model_dtype is not None:
        model = model.to(device=device, dtype=model_dtype)
    else:
        model = model.to(device=device)
    model.eval()
    return model, tokenizer


def run_preview(
    candidate_path: Path,
    run_dir: Optional[Path],
    checkpoint_dir: Optional[Path],
    metadata: Optional[Dict],
    candidate_frame: pd.DataFrame,
    warning: Optional[str] = None,
) -> Dict:
    preview = {
        "preview_time": datetime.now().isoformat(timespec="seconds"),
        "candidate_path": str(candidate_path.resolve()),
        "candidate_rows": int(len(candidate_frame)),
        "candidate_columns": ["Combo", "SMILES"],
        "run_dir": str(run_dir.resolve()) if run_dir is not None else None,
        "checkpoint_dir": str(checkpoint_dir.resolve()) if checkpoint_dir is not None else None,
        "training_status": metadata.get("status") if metadata else None,
        "finetune_method": metadata.get("finetune_method") if metadata else None,
        "target_transform": metadata.get("target_transform", {}).get("method") if metadata else None,
        "warning": warning,
    }
    reports_path = SCRIPT_DIR / "reports" / "latest_preview.json"
    reports_path.parent.mkdir(parents=True, exist_ok=True)
    with open(reports_path, "w", encoding="utf-8") as handle:
        json.dump(preview, handle, ensure_ascii=False, indent=2)
    return preview


def predict(args: argparse.Namespace) -> Dict:
    candidate_path = Path(args.candidate_path).resolve()
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate Excel file not found: {candidate_path}")

    setup_bar = tqdm(total=4, desc="Waiting hub setup", unit="stage", dynamic_ncols=True)
    setup_bar.set_postfix_str("load candidates")
    candidate_frame = load_candidate_frame(candidate_path)
    setup_bar.update(1)
    run_dir: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None
    metadata: Optional[Dict] = None

    try:
        setup_bar.set_postfix_str("resolve run")
        run_dir = resolve_run_directory(args.run_dir)
        checkpoint_dir, metadata = resolve_checkpoint_paths(run_dir, args.checkpoint)
        setup_bar.update(1)
    except FileNotFoundError as exc:
        if args.preview_only:
            setup_bar.close()
            return run_preview(
                candidate_path=candidate_path,
                run_dir=None,
                checkpoint_dir=None,
                metadata=None,
                candidate_frame=candidate_frame,
                warning=str(exc),
            )
        raise

    if args.preview_only:
        setup_bar.close()
        return run_preview(candidate_path, run_dir, checkpoint_dir, metadata, candidate_frame)

    assert run_dir is not None and checkpoint_dir is not None and metadata is not None
    device = resolve_device(args.device)
    setup_bar.set_postfix_str("load model")
    model, tokenizer = load_regression_model(
        checkpoint_dir=checkpoint_dir,
        metadata=metadata,
        device=device,
        max_length_override=args.max_length,
    )
    setup_bar.update(1)

    dataset = CandidateDataset(candidate_frame, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    transformed_predictions = []
    inference_bar = tqdm(
        loader,
        total=len(loader),
        desc="Predict batches",
        unit="batch",
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for batch_index, batch in enumerate(inference_bar):
            batch = move_batch_to_device(batch, device)
            predictions = model(**batch).view(-1)
            transformed_predictions.extend(predictions.detach().cpu().tolist())
            inference_bar.set_postfix(
                batch=batch_index + 1,
                count=len(transformed_predictions),
            )
    inference_bar.close()
    setup_bar.update(1)
    setup_bar.close()

    transformed_predictions_np = np.asarray(transformed_predictions, dtype=np.float64)
    original_predictions = inverse_transform_targets(
        transformed_predictions_np,
        metadata["target_transform"],
    )
    prediction_summary = summarize_predictions(original_predictions, top_k=10)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inference_name = f"{run_dir.name}_{args.checkpoint}_{timestamp}"
    prediction_dir = SCRIPT_DIR / "predictions" / inference_name
    report_dir = SCRIPT_DIR / "reports" / inference_name
    prediction_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    output_frame = pd.DataFrame(
        {
            "Combo": candidate_frame["Combo"],
            "SMILES": candidate_frame["SMILES"],
            "Prediction": original_predictions,
        }
    )
    sorted_output_frame = output_frame.sort_values("Prediction", ascending=False).reset_index(drop=True)
    output_frame.to_excel(prediction_dir / "waiting_hub_predictions.xlsx", index=False)
    output_frame.to_csv(prediction_dir / "waiting_hub_predictions.csv", index=False, encoding="utf-8-sig")
    sorted_output_frame.to_excel(
        prediction_dir / "waiting_hub_predictions_sorted.xlsx",
        index=False,
    )
    sorted_output_frame.to_csv(
        prediction_dir / "waiting_hub_predictions_sorted.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "inference_name": inference_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_path": str(candidate_path),
        "candidate_rows": int(len(candidate_frame)),
        "run_dir": str(run_dir.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "prediction_dir": str(prediction_dir.resolve()),
        "report_dir": str(report_dir.resolve()),
        "finetune_method": metadata["finetune_method"],
        "target_transform": metadata["target_transform"],
        **prediction_summary,
        "top_prediction_rows": sorted_output_frame.head(10).to_dict(orient="records"),
    }
    with open(report_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(SCRIPT_DIR / "latest_inference.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def print_summary(summary: Dict) -> None:
    print("=" * 72)
    for key, value in summary.items():
        print(f"{key:18}: {value}")
    print("=" * 72)


def main() -> None:
    args = parse_args()
    summary = predict(args)
    print_summary(summary)


if __name__ == "__main__":
    main()
