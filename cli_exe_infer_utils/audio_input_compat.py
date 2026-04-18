"""Wrapper-side audio input compatibility helpers.

Avoid ``torchaudio.load() -> torchcodec`` on local wrapper entrypoints so
source/reference audio can still be decoded when shared FFmpeg libraries are
missing in the host environment.
"""

from __future__ import annotations

import math
import random
from types import MethodType
from typing import Any, Optional

from loguru import logger
import soundfile as sf
import torch


def apply_soundfile_audio_input_compat(
    dit_handler: Any,
    *,
    patch_src_audio: bool = True,
    patch_reference_audio: bool = False,
) -> None:
    """Patch instance-level ACE-Step audio loaders only where the wrapper needs them."""
    if patch_src_audio and not getattr(dit_handler, "_soundfile_src_audio_compat_applied", False):
        dit_handler.process_src_audio = MethodType(_process_src_audio_with_soundfile, dit_handler)
        dit_handler._soundfile_src_audio_compat_applied = True

    if patch_reference_audio and not getattr(dit_handler, "_soundfile_reference_audio_compat_applied", False):
        dit_handler.process_reference_audio = MethodType(_process_reference_audio_with_soundfile, dit_handler)
        dit_handler._soundfile_reference_audio_compat_applied = True


def _load_audio_tensor(audio_file: str) -> tuple[torch.Tensor, int]:
    """Decode audio with soundfile first, then fall back to torchaudio when needed."""
    try:
        audio_np, sample_rate = sf.read(audio_file, dtype="float32", always_2d=True)
        if audio_np.size == 0:
            raise ValueError(f"Empty audio file: {audio_file}")
        return torch.from_numpy(audio_np.T).float().contiguous(), int(sample_rate)
    except (OSError, RuntimeError, ValueError) as soundfile_error:
        logger.debug(
            f"[audio_input_compat] soundfile failed for {audio_file}: {soundfile_error}; "
            "falling back to torchaudio.load()"
        )
        import torchaudio

        audio, sample_rate = torchaudio.load(audio_file)
        if audio.numel() == 0:
            raise ValueError(f"Empty audio file: {audio_file}")
        return audio, int(sample_rate)


def _process_src_audio_with_soundfile(self: Any, audio_file: Optional[str]) -> Optional[torch.Tensor]:
    if audio_file is None:
        return None

    try:
        audio, sample_rate = _load_audio_tensor(audio_file)
        return self._normalize_audio_to_stereo_48k(audio, sample_rate)
    except (OSError, RuntimeError, ValueError):
        logger.exception("[process_src_audio] Error processing source audio")
        return None


def _process_reference_audio_with_soundfile(self: Any, audio_file: Optional[str]) -> Optional[torch.Tensor]:
    if audio_file is None:
        return None

    try:
        audio, sample_rate = _load_audio_tensor(audio_file)
        logger.debug(f"[process_reference_audio] Reference audio shape: {audio.shape}")
        logger.debug(f"[process_reference_audio] Reference audio sample rate: {sample_rate}")
        logger.debug(
            f"[process_reference_audio] Reference audio duration: {audio.shape[-1] / sample_rate:.6f} seconds"
        )

        audio = self._normalize_audio_to_stereo_48k(audio, sample_rate)
        if self.is_silence(audio):
            return None

        target_sample_rate = int(getattr(self, "sample_rate", 48000))
        target_frames = 30 * target_sample_rate
        segment_frames = 10 * target_sample_rate

        if audio.shape[-1] < target_frames:
            repeat_times = math.ceil(target_frames / audio.shape[-1])
            audio = audio.repeat(1, repeat_times)

        total_frames = audio.shape[-1]
        segment_size = total_frames // 3

        front_start = random.randint(0, max(0, segment_size - segment_frames))
        front_audio = audio[:, front_start : front_start + segment_frames]

        middle_start = segment_size + random.randint(0, max(0, segment_size - segment_frames))
        middle_audio = audio[:, middle_start : middle_start + segment_frames]

        back_start = 2 * segment_size + random.randint(
            0,
            max(0, (total_frames - 2 * segment_size) - segment_frames),
        )
        back_audio = audio[:, back_start : back_start + segment_frames]

        return torch.cat([front_audio, middle_audio, back_audio], dim=-1)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning(f"[process_reference_audio] Invalid or unsupported reference audio: {exc}")
        return None
