"""Helpers specific to regression fine-tuning and reporting."""

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import explained_variance_score, median_absolute_error, r2_score


def sanitize_artifact_name(value: str) -> str:
    """Create a filesystem-friendly dataset identifier."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    normalized = normalized.strip("._-")
    return normalized or "dataset"


def infer_regression_dataset_name(
    csv_path: Optional[str],
    auto_discover_agile: bool,
    agile_cell_line: Optional[str],
    agile_split: Optional[str],
) -> str:
    """Infer a stable dataset name for run artifacts."""
    if csv_path:
        return sanitize_artifact_name(Path(csv_path).stem)

    if auto_discover_agile:
        return sanitize_artifact_name(
            f"AGILE_{agile_cell_line or 'ALL'}_{agile_split or 'ALL'}"
        )

    return "regression_run"


def safe_correlation(fn, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Compute correlation safely for short or constant arrays."""
    if len(y_true) < 2:
        return float("nan"), float("nan")
    if np.isclose(np.std(y_true), 0.0) or np.isclose(np.std(y_pred), 0.0):
        return float("nan"), float("nan")

    try:
        corr, p_value = fn(y_true, y_pred)
    except Exception:
        return float("nan"), float("nan")
    return float(corr), float(p_value)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute a comprehensive set of regression metrics."""
    if len(y_true) == 0:
        return {
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "median_ae": float("nan"),
            "r2": float("nan"),
            "explained_variance": float("nan"),
            "pearson_r": float("nan"),
            "pearson_pvalue": float("nan"),
            "spearman_r": float("nan"),
            "spearman_pvalue": float("nan"),
            "mean_error": float("nan"),
            "std_error": float("nan"),
            "max_abs_error": float("nan"),
            "smape": float("nan"),
            "mape_nonzero": float("nan"),
            "nrmse_range": float("nan"),
            "nrmse_std": float("nan"),
        }

    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    mse = float(np.mean(np.square(errors)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_errors))
    median_ae = float(median_absolute_error(y_true, y_pred))
    mean_error = float(np.mean(errors))
    std_error = float(np.std(errors))
    max_abs_error = float(np.max(abs_errors))

    pearson_r, pearson_pvalue = safe_correlation(pearsonr, y_true, y_pred)
    spearman_r, spearman_pvalue = safe_correlation(spearmanr, y_true, y_pred)

    y_range = float(np.max(y_true) - np.min(y_true)) if len(y_true) else float("nan")
    y_std = float(np.std(y_true)) if len(y_true) else float("nan")
    nrmse_range = float(rmse / y_range) if y_range and not np.isclose(y_range, 0.0) else float("nan")
    nrmse_std = float(rmse / y_std) if y_std and not np.isclose(y_std, 0.0) else float("nan")

    denominator = np.abs(y_true) + np.abs(y_pred) + 1e-12
    smape = float(np.mean(2.0 * abs_errors / denominator) * 100.0)
    non_zero_mask = np.abs(y_true) > 1e-12
    if np.any(non_zero_mask):
        mape_nonzero = float(
            np.mean(abs_errors[non_zero_mask] / np.abs(y_true[non_zero_mask])) * 100.0
        )
    else:
        mape_nonzero = float("nan")

    if len(y_true) >= 2:
        r2 = float(r2_score(y_true, y_pred))
        explained_variance = float(explained_variance_score(y_true, y_pred))
    else:
        r2 = float("nan")
        explained_variance = float("nan")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "median_ae": median_ae,
        "r2": r2,
        "explained_variance": explained_variance,
        "pearson_r": pearson_r,
        "pearson_pvalue": pearson_pvalue,
        "spearman_r": spearman_r,
        "spearman_pvalue": spearman_pvalue,
        "mean_error": mean_error,
        "std_error": std_error,
        "max_abs_error": max_abs_error,
        "smape": smape,
        "mape_nonzero": mape_nonzero,
        "nrmse_range": nrmse_range,
        "nrmse_std": nrmse_std,
    }


def to_builtin(value):
    """Convert numpy/pandas scalars to JSON-serializable Python values."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def json_ready_dict(data: Dict) -> Dict:
    """Recursively convert metrics dictionaries for JSON serialization."""
    json_ready = {}
    for key, value in data.items():
        if isinstance(value, dict):
            json_ready[key] = json_ready_dict(value)
        elif isinstance(value, list):
            json_ready[key] = [
                json_ready_dict(item) if isinstance(item, dict) else to_builtin(item)
                for item in value
            ]
        else:
            json_ready[key] = to_builtin(value)
    return json_ready
