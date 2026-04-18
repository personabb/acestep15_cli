"""Torch RNG helpers for ``infer_sft`` script-level reproducibility control."""

from __future__ import annotations

from typing import List

import torch


def set_torch_seed(seed: int) -> None:
    """Seed CPU/CUDA/MPS torch RNGs with the same integer."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def build_track_seeds(seed: int, batch_size: int) -> List[int]:
    """Expand a fixed seed into the same ``seed+i`` track list as ``infer_sft``."""
    if seed == -1:
        raise ValueError("Fixed track seeds are required for scripted Phase 2 mode.")
    return [seed + i for i in range(batch_size)]
