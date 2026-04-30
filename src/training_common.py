"""Shared training helpers used by the project entry scripts."""

import argparse
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


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
    if is_distributed_run():
        local_rank = get_local_rank()
        if torch.cuda.is_available():
            available_gpu_count = torch.cuda.device_count()
            if local_rank < 0 or local_rank >= available_gpu_count:
                raise ValueError(
                    f"LOCAL_RANK={local_rank} is invalid for {available_gpu_count} visible GPU(s)."
                )
            return torch.device(f"cuda:{local_rank}"), [local_rank]
        return torch.device("cpu"), []

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


@dataclass(frozen=True)
class DistributedConfig:
    """Metadata about the current distributed runtime."""

    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: Optional[str] = None


def get_world_size() -> int:
    """Return WORLD_SIZE from the environment, defaulting to single-process."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank() -> int:
    """Return distributed rank from the environment."""
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    """Return local process rank from the environment."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed_run() -> bool:
    """Whether the current process was launched for distributed training."""
    return get_world_size() > 1


def is_main_process() -> bool:
    """Whether the current process is rank 0."""
    return get_rank() == 0


def setup_distributed_training(
    device: torch.device,
    logger: Optional[logging.Logger] = None,
) -> DistributedConfig:
    """Initialize distributed training when launched with torchrun."""
    if not is_distributed_run():
        return DistributedConfig(enabled=False)

    backend = "nccl" if device.type == "cuda" else "gloo"
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    config = DistributedConfig(
        enabled=True,
        rank=dist.get_rank(),
        local_rank=get_local_rank(),
        world_size=dist.get_world_size(),
        backend=backend,
    )
    if logger is not None:
        logger.info(
            "  Distributed mode: rank %d/%d | local_rank=%d | backend=%s",
            config.rank,
            config.world_size,
            config.local_rank,
            config.backend,
        )
    return config


def cleanup_distributed_training():
    """Tear down the process group if one is active."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier():
    """Synchronize all ranks when distributed training is enabled."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def unwrap_data_parallel(model):
    """Return the underlying model when wrapped by DataParallel or DDP."""
    if isinstance(model, (torch.nn.DataParallel, DistributedDataParallel)):
        return model.module
    return model


def maybe_wrap_model_for_multi_gpu(
    model,
    device: torch.device,
    gpu_ids: Sequence[int],
    distributed: Optional[DistributedConfig] = None,
    logger: Optional[logging.Logger] = None,
):
    """Wrap the model for multi-GPU training when requested."""
    if distributed is not None and distributed.enabled:
        if device.type != "cuda":
            raise ValueError("Distributed training currently expects CUDA devices.")
        wrapped_model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
        )
        if logger is not None:
            logger.info("  GPU mode: DistributedDataParallel")
        return wrapped_model

    if len(gpu_ids) <= 1:
        if logger is not None:
            logger.info("  GPU mode: single device")
        return model

    raise RuntimeError(
        "Multi-GPU training now requires torchrun DistributedDataParallel. "
        "Launch with a command such as: "
        "`torchrun --nproc_per_node=<num_gpus> train_pretrain.py --gpus all`."
    )
