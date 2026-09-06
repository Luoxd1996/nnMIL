"""Utilities for excluding padded patches from MIL attention."""

import torch


def valid_mask_from_bag_sizes(features: torch.Tensor, bag_sizes: torch.Tensor) -> torch.Tensor:
    """Build a [B, N] mask from padded feature bags and their true lengths."""
    if features.ndim != 3:
        raise ValueError(f"Expected features with shape [B, N, D], got {tuple(features.shape)}")
    bag_sizes = torch.as_tensor(bag_sizes, device=features.device, dtype=torch.long).reshape(-1)
    if bag_sizes.numel() != features.shape[0]:
        raise ValueError("bag_sizes must contain one length per batch element")
    if torch.any(bag_sizes < 1) or torch.any(bag_sizes > features.shape[1]):
        raise ValueError("bag_sizes must be in [1, padded_sequence_length]")
    positions = torch.arange(features.shape[1], device=features.device).unsqueeze(0)
    return positions < bag_sizes.unsqueeze(1)
