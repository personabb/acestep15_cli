"""Runtime helpers for RETAKE latent loading, alignment, and decode."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch


def load_source_final_latents(
    dit_handler: Any,
    source: dict[str, Any],
) -> tuple[torch.Tensor, str]:
    """Load saved source final latents, or VAE-encode the source audio as fallback."""
    source_latents = source.get("latents")
    source_origin = str(source.get("source_latent_origin") or "")
    if source_latents is not None:
        return _to_track_batch(source_latents), source_origin or "legacy_latents_npy"
    return encode_audio_to_final_latents(dit_handler, source["wav_path"]), "vae_fallback"


def align_source_latents(
    dit_handler: Any,
    source_latents: Any,
    target_length: int,
    *,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Trim or silence-pad source latents to the requested ``[B, T, C]`` shape."""
    if target_length <= 0:
        raise ValueError("target_length must be greater than zero")

    latents = _to_track_batch(source_latents)
    if latents.shape[1] > target_length:
        latents = latents[:, :target_length, :]
    elif latents.shape[1] < target_length:
        dit_handler._ensure_silence_latent_on_device()
        pad_frames = target_length - latents.shape[1]
        silence_pad = dit_handler._get_silence_latent_slice(pad_frames).to(latents.device).to(latents.dtype)
        latents = torch.cat([latents, silence_pad.unsqueeze(0)], dim=1)

    if latents.shape[0] == 1 and batch_size > 1:
        latents = latents.expand(batch_size, -1, -1).clone()
    elif latents.shape[0] != batch_size:
        raise ValueError(f"Expected latent batch size {batch_size}, got {latents.shape[0]}")

    if device is not None or dtype is not None:
        latents = latents.to(device=device or latents.device, dtype=dtype or latents.dtype)
    return latents


def encode_audio_to_final_latents(
    dit_handler: Any,
    audio_path: str,
) -> torch.Tensor:
    """Encode a saved waveform into final decode latents ``[1, T, C]``."""
    audio_np, sample_rate = sf.read(audio_path, always_2d=True)
    if sample_rate != dit_handler.sample_rate:
        raise ValueError(
            f"Expected source audio at {dit_handler.sample_rate} Hz, got {sample_rate} Hz: {audio_path}",
        )
    audio_tensor = torch.from_numpy(audio_np.T).float()
    if audio_tensor.shape[0] == 1:
        audio_tensor = torch.cat([audio_tensor, audio_tensor], dim=0)
    with torch.inference_mode():
        with dit_handler._load_model_context("vae"):
            latents = dit_handler.tiled_encode(audio_tensor, offload_latent_to_cpu=True)
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)
    return latents.transpose(1, 2).contiguous().cpu()


def decode_latents_to_audio_tensors(
    dit_handler: Any,
    pred_latents: torch.Tensor,
) -> torch.Tensor:
    """Decode ``[B, T, C]`` latents into normalized CPU waveform tensors."""
    if pred_latents.ndim != 3:
        raise ValueError("pred_latents must be a 3D tensor")

    with torch.inference_mode():
        with dit_handler._load_model_context("vae"):
            latents_for_decode = pred_latents.transpose(1, 2).contiguous()
            if not (dit_handler.use_mlx_vae and dit_handler.mlx_vae is not None):
                vae_param = next(dit_handler.vae.parameters())
                latents_for_decode = latents_for_decode.to(
                    device=vae_param.device,
                    dtype=vae_param.dtype,
                )
            pred_wavs = dit_handler.tiled_decode(latents_for_decode)
    if pred_wavs.dtype != torch.float32:
        pred_wavs = pred_wavs.float()
    peak = pred_wavs.abs().amax(dim=[1, 2], keepdim=True)
    if torch.any(peak > 1.0):
        pred_wavs = pred_wavs / peak.clamp(min=1.0)
    return pred_wavs.cpu()


def _to_track_batch(value: Any) -> torch.Tensor:
    """Normalize a latent value into ``[1, T, C]`` torch form."""
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value).float()
    elif torch.is_tensor(value):
        tensor = value.detach().cpu().float()
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected source latents with 2 or 3 dims, got {tensor.ndim}")
    if tensor.shape[0] != 1:
        raise ValueError("Expected a single-source latent batch")
    return tensor.contiguous()
