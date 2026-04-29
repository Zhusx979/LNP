"""Helpers for working with locally saved Qwen checkpoints."""

import json
import shutil
import sys
from pathlib import Path
from typing import Mapping, Optional


QWEN_SUPPORT_FILES = [
    "modeling_qwen.py",
    "configuration_qwen.py",
    "qwen_generation_utils.py",
    "tokenization_qwen.py",
    "cpp_kernels.py",
]


def ensure_required_dependencies(dependencies: Mapping[str, object], install_hint: str):
    """Fail fast when required optional dependencies are missing."""
    missing = [name for name, value in dependencies.items() if value is None]
    if missing:
        print(
            f"ERROR: Missing required dependencies ({', '.join(missing)}). "
            f"Run: {install_hint}"
        )
        sys.exit(1)


def copy_qwen_support_files(source_dir: Optional[Path], target_dir: Path):
    """Copy trust_remote_code support files so local checkpoints are reloadable."""
    if source_dir is None:
        return []

    source_dir = Path(source_dir)
    if not source_dir.exists():
        return []

    copied_files = []
    for filename in QWEN_SUPPORT_FILES:
        source = source_dir / filename
        destination = target_dir / filename
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)
            copied_files.append(filename)
    return copied_files


def ensure_local_qwen_code(model_dir: Path):
    """Copy missing Qwen trust_remote_code files into a saved local model directory."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return []

    missing_files = [name for name in QWEN_SUPPORT_FILES if not (model_dir / name).exists()]
    if not missing_files:
        return []

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    source_dir = config.get("_name_or_path")
    if not source_dir:
        return []

    source_path = Path(source_dir)
    if not source_path.exists():
        return []

    copied_files = []
    for filename in missing_files:
        source_file = source_path / filename
        destination_file = model_dir / filename
        if source_file.exists():
            shutil.copy2(source_file, destination_file)
            copied_files.append(filename)
    return copied_files
