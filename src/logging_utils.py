"""Logging helpers for training scripts."""

import logging
from pathlib import Path


def configure_logging(logs_dir: Path, log_filename: str = "training.log"):
    """Configure file and console logging once CLI args are known."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / log_filename),
            logging.StreamHandler(),
        ],
        force=True,
    )
