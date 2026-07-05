import importlib.util
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.regression_utils import compute_regression_metrics


def load_module(module_name: str, relative_path: str):
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


train_real_data = load_module("train_real_data_module", "Real_data/train/train_real_data.py")
predict_waiting_hub = load_module(
    "predict_waiting_hub_module",
    "data/waiting hub/test/predict_waiting_hub.py",
)


def test_compute_regression_metrics_includes_quantiles_and_bias_breakdown():
    y_true = np.array([1.0, 2.0, 4.0, 8.0], dtype=float)
    y_pred = np.array([1.5, 1.0, 5.0, 7.0], dtype=float)

    metrics = compute_regression_metrics(y_true, y_pred)

    assert "p90_abs_error" in metrics
    assert "p95_abs_error" in metrics
    assert "under_prediction_rate" in metrics
    assert "over_prediction_rate" in metrics
    assert "target_mean" in metrics
    assert "prediction_mean" in metrics
    assert metrics["prediction_count"] == 4


def test_summarize_split_metrics_flattens_val_and_test_metrics():
    summary = train_real_data.summarize_split_metrics(
        epoch=2,
        train_loss=0.25,
        val_metrics={"rmse": 1.2, "mae": 0.9, "p90_abs_error": 2.0},
        test_metrics={"rmse": 1.4, "mae": 1.1, "p90_abs_error": 2.5},
    )

    assert summary["epoch"] == 2
    assert summary["train_loss_transformed"] == 0.25
    assert summary["val_rmse"] == 1.2
    assert summary["test_p90_abs_error"] == 2.5


def test_build_training_progress_stages_mentions_model_loading_and_epochs():
    stages = train_real_data.build_training_progress_stages(num_epochs=3)

    assert "load_pretrained_model" in stages
    assert "train_epochs" in stages
    assert stages["train_epochs"]["total"] == 3


def test_summarize_predictions_reports_distribution_and_top_scores():
    summary = predict_waiting_hub.summarize_predictions(
        np.array([0.5, 1.5, 2.5, 4.0], dtype=float),
        top_k=2,
    )

    assert summary["prediction_count"] == 4
    assert summary["prediction_median"] == 2.0
    assert "prediction_p95" in summary
    assert len(summary["top_predictions"]) == 2


def test_build_inference_progress_stages_mentions_checkpoint_and_batches():
    stages = predict_waiting_hub.build_inference_progress_stages(batch_count=5)

    assert "load_checkpoint" in stages
    assert stages["predict_batches"]["total"] == 5
