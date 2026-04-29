"""Helpers specific to SMILES pretraining."""

from typing import Tuple

import numpy as np
import torch


def compute_token_accuracy_stats(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[int, int, int]:
    """Count top-1/top-5 correct next-token predictions on non-padding positions."""
    valid_mask = attention_mask.bool()
    token_count = int(valid_mask.sum().item())
    if token_count == 0:
        return 0, 0, 0

    predictions = logits.argmax(dim=-1)
    correct_top1 = int(((predictions == target_ids) & valid_mask).sum().item())

    topk = min(5, logits.size(-1))
    topk_predictions = torch.topk(logits, k=topk, dim=-1).indices
    correct_top5 = int(
        (topk_predictions.eq(target_ids.unsqueeze(-1)).any(dim=-1) & valid_mask).sum().item()
    )

    return correct_top1, correct_top5, token_count


def to_serializable(value):
    """Convert numpy scalars inside nested structures into builtin Python values."""
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
