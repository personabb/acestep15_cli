"""Reference-style latent masking, splice, and analysis helpers for RETAKE."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from cli_exe_infer_utils.repaint_step_injection import (
    apply_repaint_boundary_blend,
)


def normalize_repainting_regions(
    *,
    repainting_start: float | None = None,
    repainting_end: float | None = None,
    repainting_regions: Optional[list[dict[str, Any]]] = None,
) -> list[tuple[float, float]]:
    """Normalize and merge repaint regions expressed in seconds."""
    raw_regions = repainting_regions or []
    if raw_regions:
        normalized_regions: list[tuple[float, float]] = []
        for region in raw_regions:
            if not isinstance(region, dict):
                continue
            start = float(region.get("start", 0.0))
            end = float(region.get("end", 0.0))
            if end <= start:
                raise ValueError("Each repainting region must satisfy end > start")
            normalized_regions.append((start, end))
        if not normalized_regions:
            raise ValueError("repainting_regions must contain at least one valid region")
        normalized_regions.sort(key=lambda region: (region[0], region[1]))
        merged_regions: list[tuple[float, float]] = []
        for start, end in normalized_regions:
            if not merged_regions or start > merged_regions[-1][1]:
                merged_regions.append((start, end))
                continue
            merged_regions[-1] = (merged_regions[-1][0], max(merged_regions[-1][1], end))
        return merged_regions

    if repainting_start is None or repainting_end is None:
        raise ValueError("A repainting range is required")
    if repainting_end <= repainting_start:
        raise ValueError("repainting_end must be greater than repainting_start")
    return [(float(repainting_start), float(repainting_end))]


def get_repainting_region_envelope(
    *,
    repainting_start: float | None = None,
    repainting_end: float | None = None,
    repainting_regions: Optional[list[dict[str, Any]]] = None,
) -> tuple[float, float]:
    """Return the inclusive time envelope that covers all repaint regions."""
    normalized_regions = normalize_repainting_regions(
        repainting_start=repainting_start,
        repainting_end=repainting_end,
        repainting_regions=repainting_regions,
    )
    return normalized_regions[0][0], normalized_regions[-1][1]


def build_repaint_mask(
    *,
    target_length: int,
    sample_rate: int,
    repainting_start: float | None = None,
    repainting_end: float | None = None,
    repainting_regions: Optional[list[dict[str, Any]]] = None,
) -> torch.Tensor:
    """Build a boolean ``[1, T]`` edit mask in latent-frame space."""
    if target_length <= 0:
        raise ValueError("target_length must be greater than zero")

    repaint_mask = torch.zeros((1, target_length), dtype=torch.bool)
    for start_sec, end_sec in normalize_repainting_regions(
        repainting_start=repainting_start,
        repainting_end=repainting_end,
        repainting_regions=repainting_regions,
    ):
        start_latent = int(start_sec * sample_rate // 1920)
        end_latent = int(end_sec * sample_rate // 1920)
        start_latent = max(0, min(start_latent, target_length - 1))
        end_latent = max(start_latent + 1, min(end_latent, target_length))
        repaint_mask[0, start_latent:end_latent] = True
    return repaint_mask


def mix_source_latents_into_noise(
    *,
    noise: torch.Tensor,
    source_latents: torch.Tensor,
    mix_ratio: float,
) -> torch.Tensor:
    """Blend source latents across the full latent tensor."""
    ratio = float(mix_ratio)
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError("source_latent_mix_ratio must be between 0.0 and 1.0")
    if ratio == 0.0:
        return noise
    source_like = source_latents.to(device=noise.device, dtype=noise.dtype)
    return noise * (1.0 - ratio) + source_like * ratio


def apply_source_latent_bias_to_noise(
    *,
    noise: torch.Tensor,
    source_latents: torch.Tensor,
    repaint_mask: torch.Tensor,
    bias_ratio: float,
) -> torch.Tensor:
    """Bias edit-region noise toward source latents while preserving outside noise."""
    ratio = float(bias_ratio)
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError("source_latent_mix_ratio must be between 0.0 and 1.0")
    if ratio == 0.0:
        return noise
    mask = repaint_mask.to(device=noise.device, dtype=torch.bool).unsqueeze(-1).expand_as(noise)
    source_like = source_latents.to(device=noise.device, dtype=noise.dtype)
    biased_edit_noise = noise * (1.0 - ratio) + source_like * ratio
    return noise.where(~mask, biased_edit_noise)


def apply_experimental_latent_splice(
    *,
    pred_latents: torch.Tensor,
    source_latents: torch.Tensor,
    repaint_mask: torch.Tensor,
    crossfade_frames: int,
) -> torch.Tensor:
    """Restore outside-edit regions from source latents with optional crossfade."""
    if pred_latents.ndim != 3 or source_latents.ndim != 3:
        raise ValueError("pred_latents and source_latents must be 3D tensors")
    mask = repaint_mask if repaint_mask.ndim == 2 else repaint_mask.any(dim=-1)
    if mask.ndim != 2:
        raise ValueError("repaint_mask must be a 2D or 3D tensor")
    aligned_source = source_latents.to(device=pred_latents.device, dtype=pred_latents.dtype)
    mask = mask.to(device=pred_latents.device, dtype=torch.bool)
    if crossfade_frames > 0:
        return apply_repaint_boundary_blend(
            pred_latents,
            aligned_source,
            mask,
            crossfade_frames,
        )
    return torch.where(mask.unsqueeze(-1), pred_latents, aligned_source)


def build_step_skip_timestep_schedule(
    *,
    infer_steps: int,
    skip_ratio: float,
    shift: float = 1.0,
) -> dict[str, Any]:
    """Build the reference step-skip schedule by truncating the timestep prefix."""
    ratio = float(skip_ratio)
    if infer_steps <= 0:
        raise ValueError("infer_steps must be greater than zero")
    if ratio < 0.0 or ratio >= 1.0:
        raise ValueError("source_latent_mix_ratio must be between 0.0 and 1.0")
    full_timesteps = [1.0 - (index / infer_steps) for index in range(infer_steps + 1)]
    if shift != 1.0:
        full_timesteps = [
            float(shift * value / (1 + (shift - 1) * value))
            for value in full_timesteps
        ]
    target_noise_level = 1.0 - ratio
    candidate_timesteps = full_timesteps[:-1]
    nearest_t = min(candidate_timesteps, key=lambda value: abs(value - target_noise_level))
    start_index = candidate_timesteps.index(nearest_t)
    truncated_timesteps = full_timesteps[start_index:]
    return {
        "requested_skip_ratio": ratio,
        "target_noise_level": target_noise_level,
        "start_t": float(nearest_t),
        "start_index": start_index,
        "remaining_steps": len(truncated_timesteps) - 1,
        "timesteps": [float(value) for value in truncated_timesteps],
    }


def _expand_repaint_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Expand a repaint mask by ``radius`` frames on both sides."""
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D tensor")
    if radius <= 0:
        return mask
    return F.max_pool1d(
        mask.float().unsqueeze(1),
        kernel_size=radius * 2 + 1,
        stride=1,
        padding=radius,
    ).squeeze(1).bool()


def _summarize_latent_tensor_pair(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Summarize similarity metrics for two latent tensors."""
    if lhs.shape != rhs.shape:
        raise ValueError("Latent tensors must share the same shape")
    lhs_values = lhs.detach().cpu().float()
    rhs_values = rhs.detach().cpu().float()
    if mask is not None:
        expanded_mask = mask.unsqueeze(-1).expand_as(lhs_values) if mask.ndim == 2 else mask.expand_as(lhs_values)
        selected = expanded_mask.detach().cpu().bool()
        lhs_values = lhs_values[selected]
        rhs_values = rhs_values[selected]
    else:
        lhs_values = lhs_values.reshape(-1)
        rhs_values = rhs_values.reshape(-1)
    if lhs_values.numel() == 0:
        return {"count": 0, "cosine": None, "mae": None, "rmse": None, "max_abs": None}
    lhs_values = lhs_values.reshape(-1)
    rhs_values = rhs_values.reshape(-1)
    diff = lhs_values - rhs_values
    lhs_norm = torch.linalg.vector_norm(lhs_values)
    rhs_norm = torch.linalg.vector_norm(rhs_values)
    cosine = None
    if float(lhs_norm) > 0.0 and float(rhs_norm) > 0.0:
        cosine = float(torch.dot(lhs_values, rhs_values) / (lhs_norm * rhs_norm))
    return {
        "count": int(lhs_values.numel()),
        "cosine": cosine,
        "mae": float(diff.abs().mean()),
        "rmse": float(torch.sqrt((diff * diff).mean())),
        "max_abs": float(diff.abs().max()),
    }


def build_text2audio_latent_analysis(
    *,
    pred_latents: Optional[torch.Tensor],
    target_latents: Optional[torch.Tensor],
    source_latents: Optional[torch.Tensor] = None,
    repaint_mask: Optional[torch.Tensor] = None,
    crossfade_frames: int = 0,
    spliced_latents: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Build reference-style latent diagnostics for RETAKE outputs."""
    analysis: dict[str, Any] = {}
    if pred_latents is not None and target_latents is not None:
        analysis["pred_vs_target_global"] = _summarize_latent_tensor_pair(pred_latents, target_latents)
    strict_outside_mask = None
    if repaint_mask is not None:
        strict_outside_mask = ~_expand_repaint_mask(repaint_mask, crossfade_frames)
        analysis["mask"] = {
            "edit_frames": int(repaint_mask.sum().item()),
            "strict_outside_frames": int(strict_outside_mask.sum().item()),
            "crossfade_frames": int(crossfade_frames),
        }
        if pred_latents is not None and target_latents is not None:
            analysis["pred_vs_target_edit"] = _summarize_latent_tensor_pair(pred_latents, target_latents, mask=repaint_mask)
            analysis["pred_vs_target_outside_strict"] = _summarize_latent_tensor_pair(
                pred_latents,
                target_latents,
                mask=strict_outside_mask,
            )
    if source_latents is not None and target_latents is not None:
        analysis["source_vae_vs_target_global"] = _summarize_latent_tensor_pair(source_latents, target_latents)
        if strict_outside_mask is not None:
            analysis["source_vae_vs_target_edit"] = _summarize_latent_tensor_pair(source_latents, target_latents, mask=repaint_mask)
            analysis["source_vae_vs_target_outside_strict"] = _summarize_latent_tensor_pair(
                source_latents,
                target_latents,
                mask=strict_outside_mask,
            )
    if source_latents is not None and pred_latents is not None:
        analysis["source_vae_vs_pred_global"] = _summarize_latent_tensor_pair(source_latents, pred_latents)
    if spliced_latents is not None and source_latents is not None and strict_outside_mask is not None:
        analysis["spliced_vs_source_vae_outside_strict"] = _summarize_latent_tensor_pair(
            spliced_latents,
            source_latents,
            mask=strict_outside_mask,
        )
        if pred_latents is not None:
            analysis["spliced_vs_pred_edit"] = _summarize_latent_tensor_pair(
                spliced_latents,
                pred_latents,
                mask=repaint_mask,
            )
    return analysis
