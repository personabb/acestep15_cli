"""Latent-region composition helpers for reference-style RETAKE compose."""

from __future__ import annotations

from typing import Any

import torch

from cli_exe_infer_utils.retake_latents import apply_experimental_latent_splice, build_repaint_mask


def compose_retake_latent_regions(
    *,
    source_final_latents: torch.Tensor,
    region_selections: list[dict[str, Any]],
    sample_rate: int,
    latent_crossfade_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Compose selected RETAKE regions back into source final latents."""
    if source_final_latents.ndim != 3 or source_final_latents.shape[0] != 1:
        raise ValueError("source_final_latents must be shaped [1, T, C]")
    if not region_selections:
        raise ValueError("region_selections must not be empty")

    target_length = int(source_final_latents.shape[1])
    composed_latents = source_final_latents.clone()
    combined_mask = torch.zeros((1, target_length), dtype=torch.bool)
    composition_regions: list[dict[str, Any]] = []

    ordered_selections = sorted(
        region_selections,
        key=lambda selection: (
            float(selection.get("start", 0.0)),
            float(selection.get("end", 0.0)),
            int(selection.get("region_index", 0)),
        ),
    )
    for selection in ordered_selections:
        selected_latents = selection.get("final_latents")
        if not torch.is_tensor(selected_latents):
            raise ValueError("Each region selection must include tensor final_latents")
        if selected_latents.shape != source_final_latents.shape:
            raise ValueError("All selected latents must already align to the source shape")

        start = float(selection.get("start", 0.0))
        end = float(selection.get("end", 0.0))
        if end <= start:
            raise ValueError("Each region selection must satisfy end > start")

        region_mask = build_repaint_mask(
            target_length=target_length,
            sample_rate=sample_rate,
            repainting_regions=[{"start": start, "end": end}],
        )
        combined_mask = torch.logical_or(combined_mask, region_mask)
        composed_latents = apply_experimental_latent_splice(
            pred_latents=selected_latents,
            source_latents=composed_latents,
            repaint_mask=region_mask,
            crossfade_frames=latent_crossfade_frames,
        )
        composition_regions.append(
            {
                "region_index": int(selection.get("region_index", len(composition_regions))),
                "start": start,
                "end": end,
                "selected_session_dir": str(selection.get("session_dir") or ""),
                "selected_track_index": int(selection.get("track_index", 0)),
            }
        )

    return composed_latents, combined_mask, composition_regions
