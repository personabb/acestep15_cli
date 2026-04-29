"""
infer_retake.py  –  ACE-Step RETAKE スクリプト

infer_sft.py で生成・保存したセッションをベースに、バリエーションを生成する。

使い方:
    1. SOURCE_SESSION_DIR と SOURCE_TRACK_INDEX を設定する
    2. python infer_retake.py

元の LM 音楽コード（5Hz）を再利用して LLM をスキップし、DiT だけ実行する。
RETAKE_SEED を変えることで、同じコード構造から別の音色・バリエーションを生成できる。
内部では 5Hz コードを 25Hz 潜在表現にアップスケールしてから DiT の
audio_code_hints として渡す（_decode_audio_codes_to_latents 経由）。

設計ノート:
  - audio_codes (5Hz): {n}_params.json の "audio_codes" フィールドに保存済み
  - 25Hz 潜在:  {n}_latents.npy に保存済み（VAE エンコード済み）
  - src_latent として使うのは必ず 25Hz にアップスケール/エンコードされたもの
  - 生成結果は OUTPUT_DIR/YYYYMMDD_HHMMSS/ に同形式で保存され、再度 RETAKE に使える
"""

import datetime
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
from loguru import logger

sys.path.insert(0, ".")
sys.path.insert(0, "./ACE-Step-1.5")

from acestep.audio_utils import generate_uuid_from_params, get_lora_weights_hash, normalize_audio
from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, GenerationResult, generate_music
from acestep.llm_inference import LLMHandler
from cli_exe_infer_utils.audio_input_compat import apply_soundfile_audio_input_compat
from cli_exe_infer_utils.retake_latents import (
    apply_experimental_latent_splice,
    build_repaint_mask,
    build_step_skip_timestep_schedule,
    build_text2audio_latent_analysis,
    get_repainting_region_envelope,
    mix_source_latents_into_noise,
)
from cli_exe_infer_utils.retake_runtime import (
    align_source_latents,
    decode_latents_to_audio_tensors,
    load_source_final_latents,
)
from cli_exe_infer_utils.retake_seed_utils import format_seed_setting, resolve_retake_seeds, serialize_seed_setting
from cli_exe_infer_utils.session import (
    LATENT_BUNDLE_FINAL_KEY,
    LATENT_BUNDLE_VERSION,
    load_source_track,
    make_session_dir,
    save_session_artifacts,
)
from cli_exe_infer_utils.task_type_fix import force_text2music_task_type

# ============================================================
# RETAKE ソース設定
# ============================================================

# infer_sft.py / infer_repaint.py で生成したセッションフォルダ
# 例: './output/20260405_120000'
SOURCE_SESSION_DIR = "./output/20260413_004941"

# 元トラックのインデックス（1-origin）
SOURCE_TRACK_INDEX = 1

# ============================================================
# RETAKE 設定
# ============================================================
APPLY_LATENT_SPLICE = True
# worker.py / music_generator.py と同様に、front の RETAKE 強度をそのまま使う。
SOURCE_LATENT_MIX_RATIO = 0.3
USE_STEP_SKIP = True
LATENT_CROSSFADE_FRAMES = 25
REFERENCE_AUDIO_PATH = None   # str or None

# ============================================================
# audio_codes メソッド固有設定
# ============================================================

# caption / lyrics を変更するとスタイルを変えつつコードを再利用できる
# None の場合は元の session から読み込む（cot_* 優先で解決済みの値を使用）
RETAKE_CAPTION = None   # str or None
RETAKE_LYRICS  = None   # str or None

# 乱数シード設定:
#   - int >= 0            : [seed, seed+1, ...] に展開（従来互換）
#   - -1                  : 各バッチ要素をランダム seed にする
#   - list / tuple[int]   : BATCH_SIZE 個の seed を明示指定。要素 -1 はランダム化
RETAKE_SEED: int | list[int] | tuple[int, ...] = 2392007026

# 再生成区間（秒）。None で全体、リストで区間指定
REPAINTING_REGIONS = [{"start": 107.0, "end": 111.0}]

# ============================================================
# 共通モデル設定
# ============================================================

DIT_MODEL = "acestep-v15-sft"

# LMモデル（audio_codes メソッドでは使わないが、ハンドラ初期化に必要）
LM_MODEL = "acestep-5Hz-lm-1.7B"

CHECKPOINT_DIR     = "./ACE-Step-1.5/checkpoints"
DIT_OFFLOAD_TO_CPU = True
DIT_QUANTIZATION   = None
LLM_OFFLOAD_TO_CPU = True

# ============================================================
# 共通生成設定
# ============================================================

INFERENCE_STEPS = 64      # turbo モデルは 8 推奨
GUIDANCE_SCALE  = 7.0
BATCH_SIZE      = 4
AUDIO_FORMAT    = "wav"
OUTPUT_DIR      = "./output"
DEBUG_MODE      = False


# ============================================================
# 関数定義
# ============================================================

def setup_checkpoint_dir() -> None:
    """チェックポイントディレクトリを作成する。"""
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"チェックポイントディレクトリ: {CHECKPOINT_DIR}")


def detect_device() -> None:
    """GPU/CPU を判定し、利用可能なデバイス情報を表示する。"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        bf16_ok  = torch.cuda.is_bf16_supported()
        print(f"GPU: {gpu_name}  (bfloat16: {bf16_ok})")
    else:
        print("GPU: 利用不可（CPU 使用）")


def initialize_handlers(load_llm: bool = True) -> tuple:
    """DiT ハンドラと LLM ハンドラを初期化して返す。

    Returns:
        (dit_handler, llm_handler)
    """
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler() if load_llm else None

    print(f"\nDiT モデル初期化中... ({DIT_MODEL})")
    status_msg, success = dit_handler.initialize_service(
        project_root="./ACE-Step-1.5",
        config_path=DIT_MODEL,
        device="auto",
        use_mlx_dit=False,
        offload_to_cpu=DIT_OFFLOAD_TO_CPU,
        quantization=DIT_QUANTIZATION,
    )
    if not success:
        raise RuntimeError(f"DiT 初期化失敗: {status_msg}")
    print(f"DiT 初期化完了: {status_msg}")

    if load_llm:
        # audio_codes メソッドでは LLM を使わないが、ハンドラの初期化は必要
        print(f"\nLLM 初期化中... ({LM_MODEL})")
        llm_status_msg, llm_success = llm_handler.initialize(
            checkpoint_dir=CHECKPOINT_DIR,
            lm_model_path=LM_MODEL,
            backend="pt",
            device="auto",
            offload_to_cpu=LLM_OFFLOAD_TO_CPU,
        )
        if not llm_success:
            print(f"LLM 初期化失敗: {llm_status_msg}（LLM なしで続行します）")
        else:
            print(f"LLM 初期化完了: {llm_status_msg}")

    return dit_handler, llm_handler


def _resolve_param(source_params: dict, cot_key: str, raw_key: str):
    """CoT で解決済みの値を優先し、なければ元の値を返す。

    LLM が自動推定したメタデータ（bpm, key 等）は cot_* フィールドに入っている。
    ユーザが明示指定した場合は raw_key フィールドに入っている。
    RETAKE では解決済みの値を使うことで、元の楽曲と同じ音楽的特性を継承する。
    """
    cot_val = source_params.get(cot_key)
    raw_val = source_params.get(raw_key)
    # 0 は有効な値（BPM=0 は無効だが念のため確認）
    if cot_val is not None and cot_val != "" and cot_val != 0:
        return cot_val
    return raw_val


def _build_step_skip_metadata(
    step_skip_schedule: Optional[dict[str, Any]],
    *,
    requested_inference_steps: int,
) -> Optional[dict[str, Any]]:
    """Serialize the step-skip schedule for logs and saved artifacts."""
    if step_skip_schedule is None:
        return None
    return {
        "requested_skip_ratio": step_skip_schedule["requested_skip_ratio"],
        "target_noise_level": step_skip_schedule["target_noise_level"],
        "start_t": step_skip_schedule["start_t"],
        "start_index": step_skip_schedule["start_index"],
        "remaining_steps": step_skip_schedule["remaining_steps"],
        "requested_inference_steps": requested_inference_steps,
        "timesteps": [float(value) for value in step_skip_schedule["timesteps"]],
    }


def _resolve_audio_codes_step_skip_schedule(
    *,
    infer_steps: int,
    shift: float = 1.0,
) -> tuple[float, Optional[dict[str, Any]]]:
    """Resolve the audio_codes RETAKE strength and optional step-skip schedule."""
    mix_ratio = float(SOURCE_LATENT_MIX_RATIO)
    if mix_ratio < 0.0 or mix_ratio >= 1.0:
        raise ValueError("SOURCE_LATENT_MIX_RATIO must be between 0.0 and 1.0")
    step_skip_schedule = None
    if USE_STEP_SKIP and mix_ratio > 0.0:
        step_skip_schedule = build_step_skip_timestep_schedule(
            infer_steps=infer_steps,
            skip_ratio=mix_ratio,
            shift=shift,
        )
    return mix_ratio, step_skip_schedule


def build_params_audio_codes(source: dict, seed: int) -> GenerationParams:
    """audio_codes メソッド用の GenerationParams を構築する。

    5Hz 音楽コード（LM 出力）を audio_codes に渡し、LLM を完全スキップする。
    完全再現のため、元の生成時と同じ DiT 入力条件（拡張キャプション + 解決済みメタデータ）を使う:
      - lm_metadata["caption"] → CoT 拡張キャプション（DiT テキスト Encoder への入力）
      - lm_metadata["bpm"] / "keyscale" / "timesignature" → 解決済みメタデータ
      - use_cot_caption=False, use_cot_language=False, use_cot_metas=False → LLM 完全スキップ

    内部パイプラインでは以下の流れで 25Hz 潜在にアップスケールされる:
      audio_codes (5Hz str) → _decode_audio_codes_to_latents()
        → quantizer.get_output_from_indices() + detokenizer()
        → 25Hz 潜在（audio_code_hints として DiT に渡る）

    Args:
        source: load_source_track() の戻り値
        seed: 代表 seed。params 保存と Phase 1 の決定論的化に使う。

    Returns:
        GenerationParams
    """
    params = source["params"]
    audio_codes = params.get("audio_codes", "")

    if not audio_codes or not str(audio_codes).strip():
        raise ValueError(
            f"ソーストラック {SOURCE_TRACK_INDEX} に audio_codes が保存されていません。\n"
            "infer_sft.py を THINKING=True で実行して生成してください。"
        )

    # lm_metadata.json から CoT 拡張キャプションと解決済みメタデータを読む。
    # params.json の cot_* は保存時に lm_metadata を反映するが、
    # ここでは正本として lm_metadata.json を優先する。
    lm_meta = source.get("lm_metadata") or {}

    # BPM / keyscale / timesignature / language:
    # lm_metadata → params.json cot_* → params.json 生値 の順で優先
    bpm = (
        lm_meta.get("bpm")
        or _resolve_param(params, "cot_bpm", "bpm")
    )
    keyscale = (
        lm_meta.get("keyscale")
        or _resolve_param(params, "cot_keyscale", "keyscale")
        or ""
    )
    timesignature = (
        str(lm_meta.get("timesignature") or "")
        or _resolve_param(params, "cot_timesignature", "timesignature")
        or ""
    )
    vocal_language = (
        lm_meta.get("language")
        or lm_meta.get("vocal_language")
        or _resolve_param(params, "cot_vocal_language", "vocal_language")
        or "unknown"
    )
    duration = _resolve_param(params, "cot_duration", "duration")

    # caption: RETAKE 設定で上書きするか、lm_metadata の拡張キャプションを使う
    # ※ 参照実装では source.dit_caption = cot_metadata["caption"] を使っている
    caption = RETAKE_CAPTION
    if caption is None:
        # lm_metadata["caption"] は CoT が生成した拡張キャプション（優先）。
        # params.json の cot_caption にも保存されるが、ここでは lm_metadata を正本として扱う。
        caption = (
            lm_meta.get("caption")
            or _resolve_param(params, "cot_caption", "caption")
            or ""
        )

    lyrics = RETAKE_LYRICS
    if lyrics is None:
        lyrics = _resolve_param(params, "cot_lyrics", "lyrics") or ""

    return GenerationParams(
        task_type="text2music",
        caption=caption,
        lyrics=lyrics,
        instrumental=params.get("instrumental", False),
        bpm=bpm,
        keyscale=keyscale,
        timesignature=timesignature,
        vocal_language=vocal_language,
        duration=duration if (duration is not None and float(duration) > 0) else -1.0,
        seed=seed,
        inference_steps=INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        # LLM を完全スキップ: audio_codes + 解決済みメタデータを直接使用
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,     # LLM による caption 拡張をスキップ
        use_cot_language=False,    # LLM による language 解決をスキップ
        audio_codes=str(audio_codes),
    )



def _resolve_repainting_window(
    source: Optional[dict] = None,
    *,
    sample_rate: int = 48000,
    target_length: Optional[int] = None,
) -> tuple[Optional[list[dict[str, Any]]], float, float]:
    """Resolve the effective repaint window from REPAINTING_REGIONS."""
    if REPAINTING_REGIONS:
        start, end = get_repainting_region_envelope(repainting_regions=REPAINTING_REGIONS)
        return REPAINTING_REGIONS, start, end

    # REPAINTING_REGIONS が None/空 → 全体を対象にする
    if target_length is not None:
        end = (target_length * 1920) / sample_rate
    elif source is not None:
        duration = _resolve_param(source["params"], "cot_duration", "duration")
        if duration is None or float(duration) <= 0:
            raise ValueError("REPAINTING_REGIONS が空のとき source duration が必要です。")
        end = float(duration)
    else:
        raise ValueError("REPAINTING_REGIONS が空のとき source か target_length が必要です。")
    return None, 0.0, float(end)


def _build_latent_bundle_payload(
    *,
    final_decode_latents: torch.Tensor,
    pred_latents: Optional[torch.Tensor] = None,
    target_latents: Optional[torch.Tensor] = None,
    src_latents: Optional[torch.Tensor] = None,
    chunk_masks: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Build a local latent bundle matching the reference payload layout."""
    payload: dict[str, Any] = {
        "version": LATENT_BUNDLE_VERSION,
        "final_key": LATENT_BUNDLE_FINAL_KEY,
        LATENT_BUNDLE_FINAL_KEY: final_decode_latents.detach().cpu(),
    }
    if pred_latents is not None:
        payload["pred_latents"] = pred_latents.detach().cpu()
    if target_latents is not None:
        payload["target_latents"] = target_latents.detach().cpu()
    if src_latents is not None:
        payload["src_latents"] = src_latents.detach().cpu()
    if chunk_masks is not None:
        payload["chunk_masks"] = chunk_masks.detach().cpu()
    return payload


def _build_audio_codes_track_artifacts(
    *,
    source_latent_origin: str,
    source_latent_mix_ratio: float,
    requested_inference_steps: int,
    repaint_mask: Optional[torch.Tensor],
    raw_pred_latents: torch.Tensor,
    spliced_latents: torch.Tensor,
    target_latents: Optional[torch.Tensor],
    source_latents: torch.Tensor,
    repainting_regions: Optional[list[dict[str, Any]]],
    repainting_start: float,
    repainting_end: float,
    step_skip_schedule: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build per-track bundle, analysis, and param updates for RETAKE saves."""
    track_artifacts: list[dict[str, Any]] = []
    retake_regions = repainting_regions or [{"start": repainting_start, "end": repainting_end}]
    step_skip_metadata = _build_step_skip_metadata(
        step_skip_schedule,
        requested_inference_steps=requested_inference_steps,
    )
    effective_inference_steps = (
        step_skip_schedule["remaining_steps"]
        if step_skip_schedule is not None
        else requested_inference_steps
    )
    for index in range(spliced_latents.shape[0]):
        track_mask = repaint_mask[index:index + 1] if repaint_mask is not None else None
        track_pred_latents = raw_pred_latents[index:index + 1]
        track_spliced_latents = spliced_latents[index:index + 1]
        track_target_latents = target_latents[index:index + 1] if target_latents is not None else None
        track_source_latents = source_latents[index:index + 1]
        analysis = build_text2audio_latent_analysis(
            pred_latents=track_pred_latents,
            target_latents=track_target_latents,
            source_latents=track_source_latents,
            repaint_mask=track_mask,
            crossfade_frames=LATENT_CROSSFADE_FRAMES,
            spliced_latents=track_spliced_latents,
        )
        analysis["source_latent_mix_ratio"] = source_latent_mix_ratio
        analysis["requested_inference_steps"] = requested_inference_steps
        analysis["effective_inference_steps"] = effective_inference_steps
        if step_skip_metadata is not None:
            analysis["step_skip_schedule"] = step_skip_metadata
        analysis["source_latent_origin"] = source_latent_origin
        latent_bundle = None
        if DEBUG_MODE:
            latent_bundle = _build_latent_bundle_payload(
                final_decode_latents=track_spliced_latents,
                pred_latents=track_pred_latents,
                target_latents=track_target_latents,
                src_latents=track_source_latents,
                chunk_masks=track_mask,
            )
        track_artifacts.append(
            {
                "analysis": analysis,
                "latent_bundle": latent_bundle,
                "params_updates": {
                    "source_latent_origin": source_latent_origin,
                    "retake_source_latent_mix_ratio": source_latent_mix_ratio,
                    "retake_requested_inference_steps": requested_inference_steps,
                    "retake_effective_inference_steps": effective_inference_steps,
                    "retake_regions": retake_regions,
                    "retake_variant": {
                        "method": "audio_codes",
                        "apply_latent_splice": APPLY_LATENT_SPLICE,
                        "source_latent_mix_ratio": source_latent_mix_ratio,
                        "use_step_skip": bool(step_skip_schedule is not None),
                        "step_skip_ratio": source_latent_mix_ratio if step_skip_schedule is not None else 0.0,
                        "requested_inference_steps": requested_inference_steps,
                        "effective_inference_steps": effective_inference_steps,
                        "latent_crossfade_frames": LATENT_CROSSFADE_FRAMES,
                    },
                },
            }
        )
        if step_skip_metadata is not None:
            track_artifacts[-1]["params_updates"]["retake_step_skip_schedule"] = step_skip_metadata
    return track_artifacts


def _serialize_generation_seed_input(seed_config: Any) -> Optional[Any]:
    """Convert ``GenerationConfig.seeds`` into the handler runtime seed input format."""
    if seed_config is None:
        return None
    if isinstance(seed_config, list):
        if not seed_config:
            return None
        return ",".join(str(seed) for seed in seed_config)
    if isinstance(seed_config, int):
        return str(seed_config)
    return seed_config


def _normalize_retake_audio_tensor(
    audio_tensor: torch.Tensor,
    params: GenerationParams,
    index: int,
) -> torch.Tensor:
    """Apply the same post-decode normalization policy as ``generate_music``."""
    if params.enable_normalization and params.normalization_db <= 0.0:
        peak_before = torch.max(torch.abs(audio_tensor)).item()
        logger.info(
            f"[Normalization] Audio {index} BEFORE: Peak={peak_before:.4f}, "
            f"Target={params.normalization_db}dB"
        )
        audio_tensor = normalize_audio(audio_tensor, params.normalization_db)
        peak_after = torch.max(torch.abs(audio_tensor)).item()
        logger.info(f"[Normalization] Audio {index} AFTER: Peak={peak_after:.4f}")
    return audio_tensor


def _wrap_dit_time_costs(time_costs: dict[str, Any]) -> dict[str, Any]:
    """Mirror ``acestep.inference.generate_music`` DiT-only time-cost packaging."""
    unified_time_costs: dict[str, Any] = {}
    for key, value in time_costs.items():
        unified_time_costs[f"dit_{key}"] = value
    if unified_time_costs:
        unified_time_costs["pipeline_total_time"] = unified_time_costs.get("dit_total_time_cost", 0.0)
    return unified_time_costs


def _prepare_audio_codes_retake_latent_state(
    dit_handler: AceStepHandler,
    raw_pred_latents: torch.Tensor,
    source_final_latents: torch.Tensor,
    source_latent_origin: str,
    source_latent_mix_ratio: float,
    requested_inference_steps: int,
    step_skip_schedule: Optional[dict[str, Any]],
    repainting_regions: Optional[list[dict[str, Any]]],
    repainting_start: float,
    repainting_end: float,
    target_latents: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Prepare final decode latents and RETAKE artifacts before waveform decode."""
    batch_size = raw_pred_latents.shape[0]
    aligned_source_latents = align_source_latents(
        dit_handler,
        source_final_latents,
        raw_pred_latents.shape[1],
        batch_size=batch_size,
        device=raw_pred_latents.device,
        dtype=raw_pred_latents.dtype,
    )
    repaint_mask: Optional[torch.Tensor] = None
    spliced_latents = raw_pred_latents
    if APPLY_LATENT_SPLICE:
        repaint_mask = build_repaint_mask(
            target_length=raw_pred_latents.shape[1],
            sample_rate=dit_handler.sample_rate,
            repainting_start=repainting_start,
            repainting_end=repainting_end,
            repainting_regions=repainting_regions,
        ).to(device=raw_pred_latents.device)
        if batch_size > 1:
            repaint_mask = repaint_mask.expand(batch_size, -1).clone()
        spliced_latents = apply_experimental_latent_splice(
            pred_latents=raw_pred_latents,
            source_latents=aligned_source_latents,
            repaint_mask=repaint_mask,
            crossfade_frames=LATENT_CROSSFADE_FRAMES,
        )

    raw_pred_latents_cpu = raw_pred_latents.detach().cpu()
    spliced_latents_cpu = spliced_latents.detach().cpu()
    aligned_source_latents_cpu = aligned_source_latents.detach().cpu()
    repaint_mask_cpu = repaint_mask.detach().cpu() if repaint_mask is not None else None
    target_latents_cpu = target_latents.detach().cpu() if target_latents is not None else None

    return {
        "spliced_latents": spliced_latents,
        "spliced_latents_cpu": spliced_latents_cpu,
        "raw_pred_latents_cpu": raw_pred_latents_cpu,
        "aligned_source_latents_cpu": aligned_source_latents_cpu,
        "repaint_mask_cpu": repaint_mask_cpu,
        "track_artifacts": _build_audio_codes_track_artifacts(
            source_latent_origin=source_latent_origin,
            source_latent_mix_ratio=source_latent_mix_ratio,
            requested_inference_steps=requested_inference_steps,
            repaint_mask=repaint_mask_cpu,
            raw_pred_latents=raw_pred_latents_cpu,
            spliced_latents=spliced_latents_cpu,
            target_latents=target_latents_cpu,
            source_latents=aligned_source_latents_cpu,
            repainting_regions=repainting_regions,
            repainting_start=repainting_start,
            repainting_end=repainting_end,
            step_skip_schedule=step_skip_schedule,
        ),
    }


def _run_audio_codes_retake_single_decode(
    dit_handler: AceStepHandler,
    params: GenerationParams,
    config: GenerationConfig,
    source_final_latents: torch.Tensor,
    source_latent_origin: str,
    source_latent_mix_ratio: float,
    requested_inference_steps: int,
    step_skip_schedule: Optional[dict[str, Any]],
    repainting_regions: Optional[list[dict[str, Any]]],
    repainting_start: float,
    repainting_end: float,
) -> GenerationResult:
    """Run audio-codes RETAKE so latent splice happens before the only VAE decode."""
    with force_text2music_task_type(dit_handler):
        return _run_audio_codes_retake_single_decode_inner(
            dit_handler=dit_handler,
            params=params,
            config=config,
            source_final_latents=source_final_latents,
            source_latent_origin=source_latent_origin,
            source_latent_mix_ratio=source_latent_mix_ratio,
            requested_inference_steps=requested_inference_steps,
            step_skip_schedule=step_skip_schedule,
            repainting_regions=repainting_regions,
            repainting_start=repainting_start,
            repainting_end=repainting_end,
        )


def _run_audio_codes_retake_single_decode_inner(
    dit_handler: AceStepHandler,
    params: GenerationParams,
    config: GenerationConfig,
    source_final_latents: torch.Tensor,
    source_latent_origin: str,
    source_latent_mix_ratio: float,
    requested_inference_steps: int,
    step_skip_schedule: Optional[dict[str, Any]],
    repainting_regions: Optional[list[dict[str, Any]]],
    repainting_start: float,
    repainting_end: float,
) -> GenerationResult:
    """Inner implementation of audio-codes RETAKE (called inside force_text2music_task_type)."""
    try:
        progress = dit_handler._resolve_generate_music_progress(None)
        readiness_error = dit_handler._validate_generate_music_readiness()
        if readiness_error is not None:
            return GenerationResult(**readiness_error)

        task_type, instruction = dit_handler._resolve_generate_music_task(
            task_type=params.task_type,
            audio_code_string=params.audio_codes,
            instruction=params.instruction,
        )

        runtime = dit_handler._prepare_generate_music_runtime(
            batch_size=config.batch_size,
            audio_duration=params.duration,
            repainting_end=params.repainting_end,
            seed=_serialize_generation_seed_input(config.seeds),
            use_random_seed=config.use_random_seed,
        )
        actual_batch_size = runtime["actual_batch_size"]
        actual_seed_list = runtime["actual_seed_list"]
        seed_value_for_ui = runtime["seed_value_for_ui"]
        audio_duration = runtime["audio_duration"]
        repainting_end = runtime["repainting_end"]

        refer_audios, processed_src_audio, audio_error = dit_handler._prepare_reference_and_source_audio(
            reference_audio=params.reference_audio,
            src_audio=None if params.task_type == "text2music" else params.src_audio,
            audio_code_string=params.audio_codes,
            actual_batch_size=actual_batch_size,
            task_type=task_type,
        )
        if audio_error is not None:
            return GenerationResult(**audio_error)

        service_inputs = dit_handler._prepare_generate_music_service_inputs(
            actual_batch_size=actual_batch_size,
            processed_src_audio=processed_src_audio,
            audio_duration=audio_duration,
            captions=params.caption,
            lyrics=params.lyrics,
            vocal_language=params.vocal_language,
            instruction=instruction,
            bpm=params.bpm,
            key_scale=params.keyscale,
            time_signature=params.timesignature,
            task_type=task_type,
            audio_code_string=params.audio_codes,
            repainting_start=params.repainting_start,
            repainting_end=repainting_end,
        )
        service_run = dit_handler._run_generate_music_service_with_progress(
            progress=progress,
            actual_batch_size=actual_batch_size,
            audio_duration=audio_duration,
            inference_steps=params.inference_steps,
            timesteps=params.timesteps,
            service_inputs=service_inputs,
            refer_audios=refer_audios,
            guidance_scale=params.guidance_scale,
            actual_seed_list=actual_seed_list,
            audio_cover_strength=params.audio_cover_strength,
            cover_noise_strength=params.cover_noise_strength,
            use_adg=params.use_adg,
            cfg_interval_start=params.cfg_interval_start,
            cfg_interval_end=params.cfg_interval_end,
            shift=params.shift,
            infer_method=params.infer_method,
        )
        outputs = service_run["outputs"]
        infer_steps_for_progress = service_run["infer_steps_for_progress"]

        raw_pred_latents, time_costs = dit_handler._prepare_generate_music_decode_state(
            outputs=outputs,
            infer_steps_for_progress=infer_steps_for_progress,
            actual_batch_size=actual_batch_size,
            audio_duration=audio_duration,
            latent_shift=params.latent_shift,
            latent_rescale=params.latent_rescale,
        )
        latent_state = _prepare_audio_codes_retake_latent_state(
            dit_handler=dit_handler,
            raw_pred_latents=raw_pred_latents,
            source_final_latents=source_final_latents,
            source_latent_origin=source_latent_origin,
            source_latent_mix_ratio=source_latent_mix_ratio,
            requested_inference_steps=requested_inference_steps,
            step_skip_schedule=step_skip_schedule,
            repainting_regions=repainting_regions,
            repainting_start=repainting_start,
            repainting_end=repainting_end,
            target_latents=outputs.get("target_latents_input"),
        )

        decode_start = time.time()
        decoded_wavs = decode_latents_to_audio_tensors(dit_handler, latent_state["spliced_latents"])
        time_costs = dict(time_costs)
        time_costs["vae_decode_time_cost"] = time.time() - decode_start
        time_costs["total_time_cost"] = time_costs.get("total_time_cost", 0.0) + time_costs["vae_decode_time_cost"]
        time_costs["offload_time_cost"] = dit_handler.current_offload_cost

        base_params = params.to_dict()
        audios: list[dict[str, Any]] = []
        for index in range(actual_batch_size):
            audio_params = base_params.copy()
            audio_params["seed"] = actual_seed_list[index] if index < len(actual_seed_list) else None
            audio_params["lora_loaded"] = dit_handler.lora_loaded
            audio_params["use_lora"] = dit_handler.use_lora
            audio_params["lora_scale"] = dit_handler.lora_scale
            audio_params["lora_weights_hash"] = get_lora_weights_hash(dit_handler)

            audio_tensor = _normalize_retake_audio_tensor(decoded_wavs[index], params, index)
            audios.append(
                {
                    "path": "",
                    "tensor": audio_tensor,
                    "key": generate_uuid_from_params(audio_params),
                    "sample_rate": dit_handler.sample_rate,
                    "params": audio_params,
                }
            )

        extra_outputs = {
            "pred_latents": latent_state["spliced_latents_cpu"],
            "target_latents": (
                outputs["target_latents_input"].detach().cpu()
                if outputs.get("target_latents_input") is not None
                else None
            ),
            "src_latents": outputs["src_latents"].detach().cpu() if outputs.get("src_latents") is not None else None,
            "chunk_masks": outputs["chunk_masks"].detach().cpu() if outputs.get("chunk_masks") is not None else None,
            "latent_masks": outputs["latent_masks"].detach().cpu() if outputs.get("latent_masks") is not None else None,
            "spans": outputs.get("spans", []),
            "time_costs": _wrap_dit_time_costs(time_costs),
            "seed_value": seed_value_for_ui,
            "encoder_hidden_states": (
                outputs["encoder_hidden_states"].detach().cpu()
                if outputs.get("encoder_hidden_states") is not None
                else None
            ),
            "encoder_attention_mask": (
                outputs["encoder_attention_mask"].detach().cpu()
                if outputs.get("encoder_attention_mask") is not None
                else None
            ),
            "context_latents": (
                outputs["context_latents"].detach().cpu()
                if outputs.get("context_latents") is not None
                else None
            ),
            "lyric_token_idss": (
                outputs["lyric_token_idss"].detach().cpu()
                if outputs.get("lyric_token_idss") is not None
                else None
            ),
            "lm_metadata": None,
            "retake_raw_pred_latents": latent_state["raw_pred_latents_cpu"],
            "retake_source_final_latents": latent_state["aligned_source_latents_cpu"],
            "retake_repaint_mask": latent_state["repaint_mask_cpu"],
            "track_artifacts": latent_state["track_artifacts"],
        }
        return GenerationResult(
            audios=audios,
            status_message="Generation completed successfully!",
            extra_outputs=extra_outputs,
            success=True,
            error=None,
        )
    except Exception as exc:
        logger.exception("[retake] Audio-codes single-decode generation failed")
        return GenerationResult(
            audios=[],
            status_message=f"Error: {exc}",
            extra_outputs={},
            success=False,
            error=str(exc),
        )



def run_retake(
    dit_handler: AceStepHandler,
    llm_handler: LLMHandler,
    source: dict,
    resolved_seeds: list[int],
) -> GenerationResult:
    """RETAKE を実行して GenerationResult を返す。

    Args:
        dit_handler: 初期化済み AceStepHandler
        llm_handler: 初期化済み LLMHandler
        source:      load_source_track() の戻り値
        resolved_seeds: generate_music() に渡す確定済みの per-item seed 配列

    Returns:
        GenerationResult
    """
    if not resolved_seeds:
        raise ValueError("resolved_seeds must not be empty.")

    primary_seed = resolved_seeds[0]
    source_final_latents: Optional[torch.Tensor] = None
    source_latent_origin: Optional[str] = None
    resolved_mix_ratio = 0.0
    step_skip_schedule: Optional[dict[str, Any]] = None
    repainting_regions: Optional[list[dict[str, Any]]] = None

    params = build_params_audio_codes(source, seed=primary_seed)
    repainting_regions, repainting_start, repainting_end = _resolve_repainting_window(source)
    params.repainting_start = repainting_start
    params.repainting_end = repainting_end
    if REFERENCE_AUDIO_PATH:
        params.reference_audio = REFERENCE_AUDIO_PATH
    source_final_latents, source_latent_origin = load_source_final_latents(dit_handler, source)
    resolved_mix_ratio, step_skip_schedule = _resolve_audio_codes_step_skip_schedule(
        infer_steps=params.inference_steps,
        shift=params.shift,
    )
    if step_skip_schedule is not None:
        params.timesteps = step_skip_schedule["timesteps"]

    apply_soundfile_audio_input_compat(
        dit_handler,
        patch_src_audio=False,
        patch_reference_audio=bool(getattr(params, "reference_audio", None)),
    )

    config = GenerationConfig(
        batch_size=BATCH_SIZE,
        audio_format=AUDIO_FORMAT,
        use_random_seed=False,
        seeds=resolved_seeds,
    )

    # Phase 1 (CoT) 決定論的化: generate_music() 直前に torch シードをセット
    torch.manual_seed(primary_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(primary_seed)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(primary_seed)

    print("\nRETAKE 実行中...")
    original_prepare_noise = None
    if source_final_latents is not None:
        original_prepare_noise = dit_handler.model.prepare_noise

        def patched_prepare_noise(context_latents, seed_value=None):
            noise = original_prepare_noise(context_latents, seed_value)
            aligned_source_latents = align_source_latents(
                dit_handler,
                source_final_latents,
                context_latents.shape[1],
                batch_size=context_latents.shape[0],
                device=context_latents.device,
                dtype=context_latents.dtype,
            )
            if step_skip_schedule is not None:
                start_t = step_skip_schedule["start_t"]
                return start_t * noise + (1.0 - start_t) * aligned_source_latents
            if resolved_mix_ratio <= 0.0:
                return noise
            return mix_source_latents_into_noise(
                noise=noise,
                source_latents=aligned_source_latents,
                mix_ratio=resolved_mix_ratio,
            )

        dit_handler.model.prepare_noise = patched_prepare_noise

    try:
        # audio_codes RETAKE は latent splice 後に 1 回だけ decode する。
        if source_final_latents is not None and source_latent_origin is not None:
            result = _run_audio_codes_retake_single_decode(
                dit_handler=dit_handler,
                params=params,
                config=config,
                source_final_latents=source_final_latents,
                source_latent_origin=source_latent_origin,
                source_latent_mix_ratio=resolved_mix_ratio,
                requested_inference_steps=params.inference_steps,
                step_skip_schedule=step_skip_schedule,
                repainting_regions=repainting_regions,
                repainting_start=repainting_start,
                repainting_end=repainting_end,
            )
        else:
            # source latent が取得できない場合のフォールバック
            result = generate_music(dit_handler, llm_handler, params, config, save_dir=None)
    finally:
        if original_prepare_noise is not None:
            dit_handler.model.prepare_noise = original_prepare_noise

    if result.success:
        print(f"\nRETAKE に成功しました！ ({len(result.audios)} トラック)")
    else:
        print(f"\nRETAKE に失敗しました: {result.error}")

    return result



def main() -> None:
    """全処理を順に実行するエントリーポイント。"""
    raw_retake_seed = serialize_seed_setting(RETAKE_SEED)
    resolved_retake_seeds = resolve_retake_seeds(RETAKE_SEED, BATCH_SIZE)
    mix_ratio, step_skip_preview = _resolve_audio_codes_step_skip_schedule(
        infer_steps=INFERENCE_STEPS,
        shift=1.0,
    )
    step_skip_metadata = _build_step_skip_metadata(step_skip_preview, requested_inference_steps=INFERENCE_STEPS)
    effective_steps = step_skip_preview["remaining_steps"] if step_skip_preview is not None else INFERENCE_STEPS
    repaint_region_text = (
        ", ".join(f"{r['start']}–{r['end']}s" for r in REPAINTING_REGIONS)
        if REPAINTING_REGIONS
        else "全体"
    )

    print("=" * 60)
    print(f"ソースセッション   : {SOURCE_SESSION_DIR}")
    print(f"ソーストラック     : #{SOURCE_TRACK_INDEX}")
    print(f"DiT モデル        : {DIT_MODEL}")
    print(f"LM  モデル        : {LM_MODEL}")
    print(f"DiT CPU オフロード : {DIT_OFFLOAD_TO_CPU}  (量子化: {DIT_QUANTIZATION})")
    print(f"LLM CPU オフロード : {LLM_OFFLOAD_TO_CPU}")
    print(f"Seed request      : {format_seed_setting(RETAKE_SEED)}")
    print(f"Resolved seeds    : {resolved_retake_seeds}")
    print(f"Caption 上書き    : {RETAKE_CAPTION or '（元の値を継承）'}")
    print(f"Lyrics  上書き    : {'あり' if RETAKE_LYRICS else '（元の値を継承）'}")
    print(f"Latent splice     : {APPLY_LATENT_SPLICE}")
    print(f"Latent mix ratio  : {mix_ratio}")
    print(f"Step skip         : {step_skip_preview is not None}")
    if step_skip_preview is not None:
        print(
            "Step skip start t : "
            f"{step_skip_preview['start_t']:.6f}  "
            f"({step_skip_preview['remaining_steps']}/{INFERENCE_STEPS} steps)"
        )
    print(f"Crossfade frames  : {LATENT_CROSSFADE_FRAMES}")
    print(f"Repaint 区間      : {repaint_region_text}")
    print("=" * 60)

    setup_checkpoint_dir()
    detect_device()

    # ── ソース読み込み ───────────────────────────────────────────────────
    print(f"\nソーストラックを読み込み中: {SOURCE_SESSION_DIR} / #{SOURCE_TRACK_INDEX}")
    source = load_source_track(SOURCE_SESSION_DIR, SOURCE_TRACK_INDEX)
    session_info = source["session"]

    print(f"  task_type      : {session_info.get('task_type', '?')}")
    print(f"  dit_model      : {session_info.get('dit_model', '?')}")
    params = source["params"]
    has_codes   = bool(str(params.get("audio_codes", "")).strip())
    has_latents = source["latents"] is not None
    bpm_info    = _resolve_param(params, "cot_bpm", "bpm")
    key_info    = _resolve_param(params, "cot_keyscale", "keyscale") or "(auto)"
    dur_info    = _resolve_param(params, "cot_duration", "duration")
    seed_info   = params.get("seed")
    print(f"  audio_codes    : {'✓' if has_codes else '✗'}")
    print(f"  latents.npy    : {'✓' if has_latents else '✗'}")
    if source.get("source_latent_origin"):
        print(f"  latent origin  : {source['source_latent_origin']}")
    print(f"  BPM / Key      : {bpm_info} / {key_info}")
    print(f"  Duration       : {dur_info}s")
    print(f"  Source seed    : {seed_info}")

    # ── モデル初期化・RETAKE 実行 ────────────────────────────────────────
    dit_handler, llm_handler = initialize_handlers()
    result = run_retake(dit_handler, llm_handler, source, resolved_retake_seeds)

    if result.success:
        session_dir = make_session_dir(OUTPUT_DIR)
        session_meta = {
            "timestamp":                 datetime.datetime.now().isoformat(),
            "task_type":                 "retake_audio_codes",
            "dit_model":                 DIT_MODEL,
            "lm_model":                  LM_MODEL,
            "inference_steps":           effective_steps,
            "requested_inference_steps": INFERENCE_STEPS,
            "effective_inference_steps": effective_steps,
            "guidance_scale":            GUIDANCE_SCALE,
            "batch_size":                BATCH_SIZE,
            "retake_method":             "audio_codes",
            "source_session_dir":        os.path.abspath(SOURCE_SESSION_DIR),
            "source_track_index":        SOURCE_TRACK_INDEX,
            "source_seed":               params.get("seed"),
            "retake_seed":               raw_retake_seed,
            "retake_seeds_resolved":     resolved_retake_seeds,
            "apply_latent_splice":       APPLY_LATENT_SPLICE,
            "source_latent_mix_ratio":   mix_ratio,
            "use_step_skip":             step_skip_preview is not None,
            "retake_step_skip_schedule": step_skip_metadata,
            "latent_crossfade_frames":   LATENT_CROSSFADE_FRAMES,
            "repainting_regions":        REPAINTING_REGIONS,
            "source_latent_origin":      source.get("source_latent_origin"),
            "debug_mode":                DEBUG_MODE,
        }
        save_session_artifacts(
            result,
            session_dir,
            session_meta,
            track_artifacts=result.extra_outputs.get("track_artifacts"),
            save_debug_artifacts=DEBUG_MODE,
        )


if __name__ == "__main__":
    main()
