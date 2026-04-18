"""Script-level Phase 1/2 orchestration for ``infer_sft.py``.

Fixed-seed LM audio-code generation is driven here so single- and multi-track
runs share the same Phase 2 reseed-from-start behavior.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

import torch

from acestep.inference import (
    GenerationConfig,
    GenerationParams,
    GenerationResult,
    _update_metadata_from_lm,
    generate_music,
)
from acestep.llm_inference import LLMHandler

from cli_exe_infer_utils.sft_lm import generate_track_audio_codes, run_shared_lm_phase_1
from cli_exe_infer_utils.sft_rng import build_track_seeds, set_torch_seed
from cli_exe_infer_utils.task_type_fix import force_text2music_task_type


def rebuild_track_output_params(
    request_params: GenerationParams,
    generated_params: Dict[str, Any],
    track_seed: int,
    audio_codes: str,
) -> Dict[str, Any]:
    """Restore user-facing request fields while preserving track-specific outputs."""
    rebuilt = request_params.to_dict()
    rebuilt["seed"] = track_seed
    rebuilt["audio_codes"] = audio_codes
    for key in ("lora_loaded", "use_lora", "lora_scale", "lora_weights_hash"):
        if key in generated_params:
            rebuilt[key] = generated_params[key]
    return rebuilt


def aggregate_generation_results(
    track_results: Sequence[GenerationResult],
    lm_metadata: Optional[Dict[str, Any]],
) -> GenerationResult:
    """Combine sequential single-track results into one session-level result."""
    audios: List[Dict[str, Any]] = []
    pred_latents: List[torch.Tensor] = []
    time_costs: Dict[str, float] = {}
    status_messages: List[str] = []

    for result in track_results:
        audios.extend(result.audios)
        if result.status_message:
            status_messages.append(result.status_message)

        track_latents = result.extra_outputs.get("pred_latents")
        if isinstance(track_latents, torch.Tensor):
            pred_latents.append(track_latents)

        for key, value in (result.extra_outputs.get("time_costs") or {}).items():
            if isinstance(value, (int, float)):
                time_costs[key] = time_costs.get(key, 0.0) + float(value)

    extra_outputs: Dict[str, Any] = {"lm_metadata": lm_metadata or {}}
    if pred_latents:
        extra_outputs["pred_latents"] = torch.cat(pred_latents, dim=0)
    if time_costs:
        extra_outputs["time_costs"] = time_costs

    return GenerationResult(
        audios=audios,
        status_message="\n".join(status_messages),
        extra_outputs=extra_outputs,
        success=True,
        error=None,
    )


def run_scripted_phase2_generation(
    dit_handler,
    llm_handler: LLMHandler,
    params: GenerationParams,
    batch_size: int,
    audio_format: str,
    cot_seed: Optional[int] = None,
) -> GenerationResult:
    """Run fixed-seed LM audio-code generation via script-level Phase 1/2 orchestration.

    cot_seed: LM Phase 1（CoT）専用シード。None のとき params.seed を使う。
              params.seed（= SEED）は LM Phase 2 / DiT のトラックシードとして独立して使われる。
    """
    track_seeds = build_track_seeds(params.seed, batch_size)
    phase1_seed = cot_seed if cot_seed is not None else track_seeds[0]
    lm_metadata = run_shared_lm_phase_1(
        llm_handler=llm_handler,
        params=params,
        seed=phase1_seed,
    )
    audio_codes_list = generate_track_audio_codes(
        llm_handler=llm_handler,
        params=params,
        lm_metadata=lm_metadata,
        track_seeds=track_seeds,
    )

    batch_result = _run_dit_batch_generation(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        request_params=params,
        lm_metadata=lm_metadata,
        track_seeds=track_seeds,
        audio_codes_list=audio_codes_list,
        audio_format=audio_format,
        cot_seed=phase1_seed,
    )
    return aggregate_generation_results([batch_result], lm_metadata)


def _run_dit_batch_generation(
    dit_handler,
    llm_handler: LLMHandler,
    request_params: GenerationParams,
    lm_metadata: Dict[str, Any],
    track_seeds: List[int],
    audio_codes_list: List[str],
    audio_format: str,
    cot_seed: Optional[int] = None,
) -> GenerationResult:
    """Generate all tracks in a single DiT batch call.

    DiT の generate_audio はバッチ内アイテムごとに独立した
    torch.Generator でノイズを生成するため、固定シードでも
    batch_size > 1 のバッチ呼び出しが可能。
    audio_code_string に List[str] を渡すと generate_music_request が
    バッチごとに展開してくれる。
    """
    resolved_inputs = _resolve_dit_inputs(request_params, lm_metadata)
    dit_params = replace(
        request_params,
        caption=resolved_inputs["caption"],
        lyrics=resolved_inputs["lyrics"],
        vocal_language=resolved_inputs["vocal_language"],
        bpm=resolved_inputs["bpm"],
        keyscale=resolved_inputs["keyscale"],
        timesignature=resolved_inputs["timesignature"],
        duration=resolved_inputs["duration"],
        seed=track_seeds[0],
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_language=False,
        audio_codes=audio_codes_list,  # type: ignore[arg-type]
    )
    set_torch_seed(track_seeds[0])
    with force_text2music_task_type(dit_handler):
        result = generate_music(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            params=dit_params,
            config=GenerationConfig(
                batch_size=len(track_seeds),
                audio_format=audio_format,
                use_random_seed=False,
                seeds=track_seeds,
            ),
            save_dir=None,
        )
    if not result.success:
        raise RuntimeError(f"DiT バッチ生成に失敗しました: {result.error}")

    for idx, audio_dict in enumerate(result.audios):
        rebuilt = rebuild_track_output_params(
            request_params=request_params,
            generated_params=audio_dict.get("params", {}),
            track_seed=track_seeds[idx] if idx < len(track_seeds) else track_seeds[-1],
            audio_codes=audio_codes_list[idx] if idx < len(audio_codes_list) else audio_codes_list[-1],
        )
        if cot_seed is not None:
            rebuilt["cot_seed"] = cot_seed
        audio_dict["params"] = rebuilt
    return result


def _resolve_dit_inputs(
    request_params: GenerationParams,
    lm_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve the actual DiT conditioning values using the same rules as core."""
    bpm, key_scale, time_signature, duration, vocal_language, caption, lyrics = (
        _update_metadata_from_lm(
            metadata=lm_metadata,
            bpm=request_params.bpm,
            key_scale=request_params.keyscale,
            time_signature=request_params.timesignature,
            audio_duration=request_params.duration,
            vocal_language=request_params.vocal_language,
            caption=request_params.caption,
            lyrics=request_params.lyrics,
        )
    )
    if request_params.use_cot_caption:
        caption = lm_metadata.get("caption", caption)
    if request_params.use_cot_language:
        vocal_language = lm_metadata.get("vocal_language", vocal_language)

    return {
        "caption": caption,
        "lyrics": lyrics,
        "vocal_language": vocal_language,
        "bpm": bpm,
        "keyscale": key_scale,
        "timesignature": time_signature,
        "duration": duration,
    }
