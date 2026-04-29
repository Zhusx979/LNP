"""Shared training helpers used by the project entry scripts."""

import argparse
import logging
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


def str2bool(value):
    """Parse flexible boolean CLI values."""
    if isinstance(value, bool):
        return value

    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_path(path_value: str, project_root: Path) -> Path:
    """Resolve a project-relative path."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_gpu_ids(gpus: Optional[str]) -> List[int]:
    """Parse GPU selection like '0', '0,1', 'all', or 'cpu'."""
    if gpus is None:
        return [0] if torch.cuda.is_available() else []

    normalized = gpus.strip().lower()
    if normalized in {"cpu", "none"}:
        return []
    if normalized == "all":
        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))

    gpu_ids = []
    for token in gpus.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            gpu_ids.append(int(token))
        except ValueError as exc:
            raise ValueError(
                f"Invalid GPU selection '{gpus}'. Use formats like '0', '0,1', 'all', or 'cpu'."
            ) from exc

    if not gpu_ids:
        raise ValueError(
            f"Invalid GPU selection '{gpus}'. Use formats like '0', '0,1', 'all', or 'cpu'."
        )

    deduplicated = []
    for gpu_id in gpu_ids:
        if gpu_id not in deduplicated:
            deduplicated.append(gpu_id)
    return deduplicated


def resolve_device_config(
    gpus: Optional[str],
    logger: Optional[logging.Logger] = None,
) -> Tuple[torch.device, List[int]]:
    """Resolve the runtime device and validated GPU ids."""
    gpu_ids = parse_gpu_ids(gpus)
    if not gpu_ids:
        return torch.device("cpu"), []

    if not torch.cuda.is_available():
        if logger is not None:
            logger.warning("CUDA is unavailable, falling back to CPU.")
        return torch.device("cpu"), []

    available_gpu_count = torch.cuda.device_count()
    invalid_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < 0 or gpu_id >= available_gpu_count]
    if invalid_gpu_ids:
        raise ValueError(
            f"Requested GPU ids {invalid_gpu_ids} but only {available_gpu_count} GPU(s) are available."
        )

    return torch.device(f"cuda:{gpu_ids[0]}"), gpu_ids


def unwrap_data_parallel(model):
    """Return the underlying model when wrapped by DataParallel."""
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def maybe_enable_data_parallel(
    model,
    gpu_ids: Sequence[int],
    logger: Optional[logging.Logger] = None,
):
    """Wrap the model for multi-GPU training when requested."""
    if len(gpu_ids) <= 1:
        if logger is not None:
            logger.info("  GPU mode: single device")
        return model

    wrapped_model = torch.nn.DataParallel(
        model,
        device_ids=list(gpu_ids),
        output_device=gpu_ids[0],
    )
    if logger is not None:
        logger.info("  GPU mode: DataParallel on GPUs %s", list(gpu_ids))
    return wrapped_model
