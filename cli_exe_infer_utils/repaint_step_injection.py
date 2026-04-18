"""Step-level repaint injection and boundary blending helpers."""

from __future__ import annotations

import torch


def apply_repaint_step_injection(
    xt: torch.Tensor,
    clean_src_latents: torch.Tensor,
    repaint_mask: torch.Tensor,
    t_next: float,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Replace non-repaint regions with the matching noised source latents."""
    zt_src = t_next * noise + (1.0 - t_next) * clean_src_latents
    mask_expanded = repaint_mask.unsqueeze(-1).expand_as(xt)
    return torch.where(mask_expanded, xt, zt_src)


def build_soft_repaint_mask(
    repaint_mask: torch.Tensor,
    crossfade_frames: int,
) -> torch.Tensor:
    """Build a soft repaint mask with linear ramps at the edit boundaries."""
    soft_mask = repaint_mask.float().clone()
    if crossfade_frames <= 0:
        return soft_mask

    batch_size, target_length = repaint_mask.shape
    for batch_index in range(batch_size):
        row = repaint_mask[batch_index]
        if row.all() or not row.any():
            continue

        indices = torch.nonzero(row, as_tuple=False).squeeze(-1)
        if indices.numel() == 0:
            continue

        left = indices[0].item()
        right = indices[-1].item() + 1

        fade_start = max(left - crossfade_frames, 0)
        left_ramp_length = left - fade_start
        if left_ramp_length > 0:
            ramp = torch.linspace(
                0.0,
                1.0,
                left_ramp_length + 2,
                device=soft_mask.device,
            )[1:-1]
            soft_mask[batch_index, fade_start:left] = ramp

        fade_end = min(right + crossfade_frames, target_length)
        right_ramp_length = fade_end - right
        if right_ramp_length > 0:
            ramp = torch.linspace(
                1.0,
                0.0,
                right_ramp_length + 2,
                device=soft_mask.device,
            )[1:-1]
            soft_mask[batch_index, right:fade_end] = ramp

    return soft_mask


def apply_repaint_boundary_blend(
    x_gen: torch.Tensor,
    clean_src_latents: torch.Tensor,
    repaint_mask: torch.Tensor,
    crossfade_frames: int = 10,
) -> torch.Tensor:
    """Blend generated and preserved latents at repaint boundaries."""
    soft_mask = build_soft_repaint_mask(repaint_mask, crossfade_frames)
    expanded_mask = soft_mask.unsqueeze(-1).expand_as(x_gen)
    return expanded_mask * x_gen + (1.0 - expanded_mask) * clean_src_latents
