"""LM-side helpers for ``infer_sft`` script-level Phase 2 control."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from acestep.inference import GenerationParams
from acestep.llm_inference import LLMHandler

from cli_exe_infer_utils.sft_rng import set_torch_seed


# Keep ``genres:`` enabled in Phase 2 so fixed-seed single/multi runs use the
# same constrained-decoding contract.
SHARED_PHASE2_SKIP_GENRES = False


def should_use_scripted_phase2_lm(
    batch_size: int,
    params: GenerationParams,
    llm_handler: Optional[LLMHandler],
) -> bool:
    """Return whether ``infer_sft`` should bypass core LM batching.

    The script path is used when the LM actually samples audio codes
    (``thinking=True``) with a fixed seed so single- and multi-track runs share
    the same Phase 2 entrypoint and ``skip_genres=False`` behavior.
    """
    if batch_size <= 0 or params.seed == -1 or params.task_type != "text2music":
        return False
    if not params.thinking:
        return False
    if llm_handler is None or not llm_handler.llm_initialized:
        return False
    return True


def run_shared_lm_phase_1(
    llm_handler: LLMHandler,
    params: GenerationParams,
    seed: int,
) -> Dict[str, Any]:
    """Run LM Phase 1 once and return the shared metadata."""
    set_torch_seed(seed)
    phase1 = llm_handler.generate_with_stop_condition(
        caption=params.caption or "",
        lyrics=params.lyrics or "",
        infer_type="dit",
        temperature=params.lm_temperature,
        cfg_scale=params.lm_cfg_scale,
        negative_prompt=params.lm_negative_prompt,
        top_k=_resolve_top_k(params),
        top_p=_resolve_top_p(params),
        target_duration=params.duration,
        user_metadata=_build_user_metadata(params),
        use_cot_caption=params.use_cot_caption,
        use_cot_language=params.use_cot_language,
        use_cot_metas=params.use_cot_metas,
        use_constrained_decoding=params.use_constrained_decoding,
        constrained_decoding_debug=False,
        batch_size=1,
    )
    if not phase1.get("success", False):
        raise RuntimeError(f"LM Phase 1 に失敗しました: {phase1.get('error')}")
    return phase1.get("metadata") or {}


def generate_track_audio_codes(
    llm_handler: LLMHandler,
    params: GenerationParams,
    lm_metadata: Dict[str, Any],
    track_seeds: Sequence[int],
) -> List[str]:
    """Generate per-track audio codes from the beginning of each track seed."""
    cot_text = llm_handler._format_metadata_as_cot(lm_metadata)
    formatted_prompt = llm_handler.build_formatted_prompt_with_cot(
        params.caption or "",
        params.lyrics or "",
        cot_text,
    )

    return [
        _generate_single_track_audio_codes(
            llm_handler=llm_handler,
            formatted_prompt=formatted_prompt,
            params=params,
            cot_text=cot_text,
            seed=track_seed,
        )
        for track_seed in track_seeds
    ]


def _generate_single_track_audio_codes(
    llm_handler: LLMHandler,
    formatted_prompt: str,
    params: GenerationParams,
    cot_text: str,
    seed: int,
) -> str:
    """Generate one track's audio codes starting from ``track_seed``.

    The shared script-side Phase 2 contract reseeds from the beginning for
    every track and keeps the ``genres:`` field enabled.
    """
    set_torch_seed(seed)

    output_text, status = llm_handler.generate_from_formatted_prompt(
        formatted_prompt=formatted_prompt,
        cfg={
            "temperature": params.lm_temperature,
            "cfg_scale": params.lm_cfg_scale,
            "negative_prompt": params.lm_negative_prompt,
            "top_k": _resolve_top_k(params),
            "top_p": _resolve_top_p(params),
            "repetition_penalty": 1.0,
            "target_duration": params.duration,
            "user_metadata": None,
            "skip_genres": SHARED_PHASE2_SKIP_GENRES,
            "skip_caption": True,
            "skip_language": True,
            "generation_phase": "codes",
            "caption": params.caption or "",
            "lyrics": params.lyrics or "",
            "cot_text": cot_text,
        },
        use_constrained_decoding=params.use_constrained_decoding,
        constrained_decoding_debug=False,
        stop_at_reasoning=False,
    )
    if not output_text:
        raise RuntimeError(f"LM Phase 2 に失敗しました: {status}")

    _, audio_codes = llm_handler.parse_lm_output(output_text)
    if not audio_codes:
        raise RuntimeError("LM Phase 2 が空の audio_codes を返しました。")
    return audio_codes


def _build_user_metadata(params: GenerationParams) -> Optional[Dict[str, Any]]:
    """Build the same user metadata payload that core inference sends to the LM."""
    user_metadata: Dict[str, Any] = {}

    if params.bpm is not None:
        try:
            bpm_value = float(params.bpm)
        except (TypeError, ValueError):
            bpm_value = 0.0
        if bpm_value > 0:
            user_metadata["bpm"] = int(bpm_value)

    if params.keyscale and params.keyscale.strip():
        key_scale_clean = params.keyscale.strip()
        if key_scale_clean.lower() not in ["n/a", ""]:
            user_metadata["keyscale"] = key_scale_clean

    if params.timesignature and params.timesignature.strip():
        time_sig_clean = params.timesignature.strip()
        if time_sig_clean.lower() not in ["n/a", ""]:
            user_metadata["timesignature"] = time_sig_clean

    if params.duration is not None:
        try:
            duration_value = float(params.duration)
        except (TypeError, ValueError):
            duration_value = 0.0
        if duration_value > 0:
            user_metadata["duration"] = int(duration_value)

    return user_metadata or None


def _resolve_top_k(params: GenerationParams) -> Optional[int]:
    """Convert ``GenerationParams.lm_top_k`` to the LM API contract."""
    if not params.lm_top_k or params.lm_top_k == 0:
        return None
    return int(params.lm_top_k)


def _resolve_top_p(params: GenerationParams) -> Optional[float]:
    """Convert ``GenerationParams.lm_top_p`` to the LM API contract."""
    if not params.lm_top_p or params.lm_top_p >= 1.0:
        return None
    return params.lm_top_p
