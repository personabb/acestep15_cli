"""
session.py  –  セッション保存・読み込みユーティリティ

infer_sft.py / infer_repaint.py / infer_retake.py から共通で使う。

生成結果を以下のフォルダ構造で保存する:

    outputs/YYYYMMDD_HHMMSS/
        session.json          # セッションメタデータ
        lm_metadata.json      # LM CoT メタデータ（全トラック共通・1セッション1個）
        01.wav                # 生成音声 #1
        01_params.json        # 生成パラメータ #1（audio_codes 含む）
        01_latents.npy        # 25Hz 潜在表現 #1  shape=[T, 64] float32  T=25Hz×秒数
        02.wav
        02_params.json
        02_latents.npy
        ...

重要な設計注記:
  - 5Hz 音楽コード（LM 出力）のトラック差分は {n}_params.json の "audio_codes" に保存する
  - 25Hz 潜在表現は result.extra_outputs["pred_latents"] から直接取得して保存する
    shape: [batch, T, 64] → トラックごとに [T, 64] を保存
    ※ WAV を VAE encoder で再エンコードすると劣化するため NG
  - audio_codes を DiT に渡す際は内部で 25Hz にアップスケールされる
    （_decode_audio_codes_to_latents: FSQ quantizer.get_output_from_indices + detokenizer）
  - torchaudio は torchcodec エラーで動かない。必ず soundfile を使うこと
"""

import datetime
import json
import os
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch

from acestep.inference import GenerationResult

LATENT_BUNDLE_VERSION = 1
LATENT_BUNDLE_FINAL_KEY = "final_decode_latents"
DEBUG_MODE = False


# ---------------------------------------------------------------------------
# セッションディレクトリ管理
# ---------------------------------------------------------------------------

def make_session_dir(output_base: str) -> str:
    """output_base/YYYYMMDD_HHMMSS/ を作成してパスを返す。"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(output_base, timestamp)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def _normalize_lm_text(value: Any) -> Optional[str]:
    """LM メタデータの文字列値を正規化し、有効なときだけ返す。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    return text


def _normalize_lm_bpm(value: Any) -> Optional[int]:
    """LM メタデータから有効な BPM を整数で取り出す。"""
    text = _normalize_lm_text(value)
    if text is None:
        return None
    try:
        bpm = int(float(text))
    except (TypeError, ValueError):
        return None
    return bpm if bpm > 0 else None


def _normalize_lm_duration(value: Any) -> Optional[float]:
    """LM メタデータから有効な duration を取り出す。"""
    text = _normalize_lm_text(value)
    if text is None:
        return None
    try:
        duration = float(text)
    except (TypeError, ValueError):
        return None
    if duration <= 0:
        return None
    return int(duration) if duration.is_integer() else duration


def _normalize_resolved_language(value: Any) -> Optional[str]:
    """有効な vocal language を正規化して返す。"""
    text = _normalize_lm_text(value)
    if text is None or text.lower() == "unknown":
        return None
    return text


def _normalize_existing_cot_text(value: Any) -> Optional[str]:
    """既存の cot_* 文字列値を有効なときだけ返す。"""
    return _normalize_lm_text(value)


def _normalize_existing_cot_duration(value: Any) -> Optional[float]:
    """既存の cot_duration を正規化して返す。"""
    return _normalize_lm_duration(value)


def _enrich_params_with_lm_metadata(params: dict, lm_metadata: Optional[dict]) -> dict:
    """params.json 用の ``cot_*`` に解決済み値を反映する。

    優先順位:
      1. lm_metadata.json にある LM/CoT の解決済み値
      2. 既存の cot_* 値
      3. 実際に生成に使った元パラメータ値
    """
    enriched = dict(params)

    lm_metadata = lm_metadata if isinstance(lm_metadata, dict) else {}

    bpm = (
        _normalize_lm_bpm(lm_metadata.get("bpm"))
        or _normalize_lm_bpm(enriched.get("cot_bpm"))
        or _normalize_lm_bpm(enriched.get("bpm"))
    )
    if bpm is not None:
        enriched["cot_bpm"] = bpm

    keyscale = (
        _normalize_lm_text(lm_metadata.get("keyscale"))
        or _normalize_existing_cot_text(enriched.get("cot_keyscale"))
        or _normalize_lm_text(enriched.get("keyscale"))
    )
    if keyscale is not None:
        enriched["cot_keyscale"] = keyscale

    timesignature = (
        _normalize_lm_text(lm_metadata.get("timesignature"))
        or _normalize_existing_cot_text(enriched.get("cot_timesignature"))
        or _normalize_lm_text(enriched.get("timesignature"))
    )
    if timesignature is not None:
        enriched["cot_timesignature"] = timesignature

    duration = (
        _normalize_lm_duration(lm_metadata.get("duration"))
        or _normalize_existing_cot_duration(enriched.get("cot_duration"))
        or _normalize_lm_duration(enriched.get("duration"))
    )
    if duration is not None:
        enriched["cot_duration"] = duration

    vocal_language = (
        _normalize_resolved_language(lm_metadata.get("language"))
        or _normalize_resolved_language(lm_metadata.get("vocal_language"))
        or _normalize_resolved_language(enriched.get("cot_vocal_language"))
        or _normalize_resolved_language(enriched.get("vocal_language"))
    )
    if vocal_language is not None:
        enriched["cot_vocal_language"] = vocal_language

    caption = (
        _normalize_lm_text(lm_metadata.get("caption"))
        or _normalize_existing_cot_text(enriched.get("cot_caption"))
        or _normalize_lm_text(enriched.get("caption"))
    )
    if caption is not None:
        enriched["cot_caption"] = caption

    lyrics = (
        _normalize_lm_text(lm_metadata.get("lyrics"))
        or _normalize_existing_cot_text(enriched.get("cot_lyrics"))
        or _normalize_lm_text(enriched.get("lyrics"))
    )
    if lyrics is not None:
        enriched["cot_lyrics"] = lyrics

    return enriched


# ---------------------------------------------------------------------------
# アーティファクト保存
# ---------------------------------------------------------------------------

def save_session_artifacts(
    result: GenerationResult,
    session_dir: str,
    session_meta: dict,
    track_artifacts: Optional[list[dict[str, Any]]] = None,
    save_debug_artifacts: Optional[bool] = None,
) -> None:
    """GenerationResult の全アーティファクトを session_dir に保存する。

    保存内容:
      - lm_metadata.json      (extra_outputs["lm_metadata"])
      - {n:02d}.wav           (audio tensor → soundfile.write, 48kHz stereo PCM_16)
      - {n:02d}_params.json   (audio_dict["params"] に audio_codes と解決済み cot_* を含む)
      - {n:02d}_latents.npy   (pred_latents[n-1] → [T, 64] float32, T=25Hz×秒数)
      - {n:02d}_latent_bundle.pt (DEBUG_MODE 有効時のみ。final_decode_latents 等を含む torch bundle)
      - {n:02d}_analysis.json (RETAKE 分析メタデータ。存在する場合のみ)
      - session.json          (session_meta + tracks リスト)

    Args:
        result:       generate_music() の戻り値
        session_dir:  make_session_dir() で作成したフォルダ
        session_meta: session.json に埋め込む追加情報 dict
        track_artifacts: トラックごとの追加保存物。``latent_bundle`` /
            ``analysis`` / ``params_updates`` を任意で指定できる。
        save_debug_artifacts: ``True`` のときだけ latent bundle を保存する。
            ``None`` の場合は module-level ``DEBUG_MODE`` を使う。
    """
    os.makedirs(session_dir, exist_ok=True)
    save_debug_artifacts = DEBUG_MODE if save_debug_artifacts is None else bool(save_debug_artifacts)

    # ── lm_metadata.json ──────────────────────────────────────────────────
    lm_metadata = result.extra_outputs.get("lm_metadata")
    if lm_metadata is not None:
        _write_json(os.path.join(session_dir, "lm_metadata.json"), lm_metadata)

    # ── DiT 出力 pred_latents（VAE decoder への入力）を取得 ───────────────
    # shape: [batch, T, 64]  T=25Hz×秒数（例: 30秒 → T=750）
    pred_latents = result.extra_outputs.get("pred_latents")
    target_latents = result.extra_outputs.get("target_latents")
    src_latents = result.extra_outputs.get("src_latents")
    chunk_masks = result.extra_outputs.get("chunk_masks")
    track_artifacts = track_artifacts or []

    # ── 各トラック ────────────────────────────────────────────────────────
    tracks = []
    for idx, audio_dict in enumerate(result.audios):
        n = idx + 1
        wav_filename     = f"{n:02d}.wav"
        params_filename  = f"{n:02d}_params.json"
        latents_filename = f"{n:02d}_latents.npy"
        latent_bundle_filename = f"{n:02d}_latent_bundle.pt"
        analysis_filename = f"{n:02d}_analysis.json"

        wav_path     = os.path.join(session_dir, wav_filename)
        params_path  = os.path.join(session_dir, params_filename)
        latents_path = os.path.join(session_dir, latents_filename)
        latent_bundle_path = os.path.join(session_dir, latent_bundle_filename)
        analysis_path = os.path.join(session_dir, analysis_filename)

        audio_tensor = audio_dict.get("tensor")      # [channels, samples], float32, CPU
        sample_rate  = audio_dict.get("sample_rate", 48000)
        duration_sec: Optional[float] = None
        has_latents  = False
        has_latent_bundle = False
        has_analysis = False
        artifact = track_artifacts[idx] if idx < len(track_artifacts) else {}
        if artifact is None:
            artifact = {}

        # WAV 保存
        if audio_tensor is not None:
            audio_tensor = audio_tensor.cpu().float()
            wav_np = audio_tensor.numpy().T   # [channels, samples] → [samples, channels]
            sf.write(wav_path, wav_np, sample_rate, subtype="PCM_16")
            duration_sec = wav_np.shape[0] / sample_rate
        else:
            existing_path = audio_dict.get("path", "")
            if existing_path and os.path.exists(existing_path):
                wav_np, sample_rate = sf.read(existing_path, always_2d=True)
                sf.write(wav_path, wav_np, sample_rate, subtype="PCM_16")
                duration_sec = wav_np.shape[0] / sample_rate

        # 25Hz 潜在表現を pred_latents から直接保存（VAE encoder 再エンコード不要）
        # pred_latents shape: [batch, T, 64] → トラックごとに [T, 64] を保存
        if pred_latents is not None and idx < pred_latents.shape[0]:
            track_latent = pred_latents[idx].float().cpu().numpy()  # [T, 64]
            np.save(latents_path, track_latent)
            has_latents = True

        if save_debug_artifacts:
            bundle_payload = artifact.get("latent_bundle")
            if bundle_payload is None:
                track_pred_latents = _slice_track_latent_value(pred_latents, idx)
                bundle_payload = _build_latent_bundle_payload(
                    final_decode_latents=track_pred_latents,
                    pred_latents=track_pred_latents,
                    target_latents=_slice_track_latent_value(target_latents, idx),
                    src_latents=_slice_track_latent_value(src_latents, idx),
                    chunk_masks=_slice_track_latent_value(chunk_masks, idx),
                )
            if bundle_payload is not None:
                _save_latent_bundle(latent_bundle_path, bundle_payload)
                has_latent_bundle = True

        # params.json
        params = _enrich_params_with_lm_metadata(
            audio_dict.get("params", {}),
            lm_metadata,
        )
        params.update(artifact.get("params_updates") or {})
        if has_latent_bundle:
            params["latent_25hz_bundle_file"] = latent_bundle_filename
            params["latent_25hz_bundle_version"] = LATENT_BUNDLE_VERSION
            params["latent_25hz_final_key"] = LATENT_BUNDLE_FINAL_KEY
        analysis = artifact.get("analysis")
        if analysis is not None:
            _write_json(analysis_path, analysis)
            has_analysis = True
            params["retake_analysis"] = analysis
            params["retake_analysis_file"] = analysis_filename
        _write_json(params_path, params)

        has_audio_codes = bool(str(params.get("audio_codes", "")).strip())

        tracks.append({
            "index":          n,
            "wav_file":       wav_filename,
            "params_file":    params_filename,
            "latents_file":   latents_filename,
            "seed":           params.get("seed"),
            "duration_sec":   duration_sec,
            "has_audio_codes": has_audio_codes,
            "has_latents":    has_latents,
            "latent_bundle_file": latent_bundle_filename if has_latent_bundle else None,
            "analysis_file": analysis_filename if has_analysis else None,
            "has_latent_bundle": has_latent_bundle,
            "has_analysis":   has_analysis,
        })

    # ── session.json ─────────────────────────────────────────────────────
    session_data = {**session_meta, "tracks": tracks}
    _write_json(os.path.join(session_dir, "session.json"), session_data)

    # ── 結果表示 ──────────────────────────────────────────────────────────
    print(f"\nセッションを保存しました: {session_dir}")
    for t in tracks:
        marks = f"codes:{'✓' if t['has_audio_codes'] else '✗'}  latents:{'✓' if t['has_latents'] else '✗'}"
        dur   = f"  {t['duration_sec']:.1f}s" if t["duration_sec"] else ""
        print(f"  [{t['index']:02d}] {t['wav_file']}{dur}  {marks}")


# ---------------------------------------------------------------------------
# アーティファクト読み込み（RETAKE 用）
# ---------------------------------------------------------------------------

def load_source_track(session_dir: str, track_index: int) -> dict:
    """session_dir から track_index 番目のアーティファクトを読み込む。

    Args:
        session_dir:  save_session_artifacts() で作成したセッションフォルダ
        track_index:  読み込むトラックのインデックス (1-origin)

    Returns:
        {
            "session":  session.json の内容 (dict),
            "params":   {n:02d}_params.json の内容 (dict),  ← audio_codes (5Hz) を含む
            "wav_path": {n:02d}.wav の絶対パス (str),
            "latents":  np.ndarray [T, 64] float32 または None,  T=25Hz×秒数（例: 30秒 → T=750）
        }

    Raises:
        FileNotFoundError: session.json や params.json が見つからない場合
    """
    session_dir = os.path.abspath(session_dir)
    n = track_index

    session_path = os.path.join(session_dir, "session.json")
    if not os.path.exists(session_path):
        raise FileNotFoundError(f"session.json が見つかりません: {session_path}")

    with open(session_path, encoding="utf-8") as f:
        session = json.load(f)

    params_path  = os.path.join(session_dir, f"{n:02d}_params.json")
    wav_path     = os.path.join(session_dir, f"{n:02d}.wav")
    latents_path = os.path.join(session_dir, f"{n:02d}_latents.npy")

    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.json が見つかりません: {params_path}")

    with open(params_path, encoding="utf-8") as f:
        params = json.load(f)

    latent_bundle: Optional[dict] = None
    latent_bundle_file = params.get("latent_25hz_bundle_file") or f"{n:02d}_latent_bundle.pt"
    latent_bundle_path = os.path.join(session_dir, latent_bundle_file)
    if os.path.exists(latent_bundle_path):
        latent_bundle = _load_latent_bundle(latent_bundle_path)

    analysis: Optional[dict] = None
    analysis_file = params.get("retake_analysis_file") or f"{n:02d}_analysis.json"
    analysis_path = os.path.join(session_dir, analysis_file)
    if os.path.exists(analysis_path):
        with open(analysis_path, encoding="utf-8") as f:
            analysis = json.load(f)

    # latents: [T, 64] float32  T=25Hz×秒数（pred_latents から直接保存済み）
    latents: Optional[np.ndarray] = None
    source_latent_origin: Optional[str] = None
    if latent_bundle is not None:
        final_key = str(latent_bundle.get("final_key") or LATENT_BUNDLE_FINAL_KEY)
        latents = _latent_value_to_numpy(latent_bundle.get(final_key))
        source_latent_origin = "saved_bundle"
    if latents is None and os.path.exists(latents_path):
        latents = np.load(latents_path).astype(np.float32)   # [T, 64]
        source_latent_origin = "legacy_latents_npy"

    # lm_metadata.json: LM CoT が生成した拡張キャプション・解決済みメタデータ。
    # save_session_artifacts() はこれを params.json の cot_* にも反映する。
    lm_metadata: Optional[dict] = None
    lm_metadata_path = os.path.join(session_dir, "lm_metadata.json")
    if os.path.exists(lm_metadata_path):
        with open(lm_metadata_path, encoding="utf-8") as f:
            lm_metadata = json.load(f)

    return {
        "session":     session,
        "params":      params,
        "wav_path":    wav_path,
        "latents":     latents,
        "lm_metadata": lm_metadata,   # LM CoT メタデータ（RETAKE で拡張キャプション等を再利用するため）
        "latent_bundle": latent_bundle,
        "analysis":    analysis,
        "source_latent_origin": source_latent_origin,
    }


# ---------------------------------------------------------------------------
# 内部ヘルパー
# ---------------------------------------------------------------------------

def _write_json(path: str, data: dict) -> None:
    """dict を JSON ファイルに書き出す（非シリアライザブルな値は str 変換）。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _detach_cpu_latent_value(value: Any) -> Any:
    """Detach a tensor-like value to a contiguous CPU representation."""
    if value is None:
        return None
    detached = value.detach() if hasattr(value, "detach") else value
    cpu_value = detached.cpu() if hasattr(detached, "cpu") else detached
    return cpu_value.contiguous() if hasattr(cpu_value, "contiguous") else cpu_value


def _slice_track_latent_value(value: Any, track_index: int) -> Any:
    """Slice a batch latent tensor down to one track when possible."""
    if value is None:
        return None
    try:
        return value[track_index:track_index + 1]
    except Exception:
        return value


def _build_latent_bundle_payload(
    *,
    final_decode_latents: Any = None,
    pred_latents: Any = None,
    target_latents: Any = None,
    src_latents: Any = None,
    chunk_masks: Any = None,
) -> Optional[dict[str, Any]]:
    """Build the local latent bundle payload saved beside session artifacts."""
    payload: dict[str, Any] = {
        "version": LATENT_BUNDLE_VERSION,
        "final_key": LATENT_BUNDLE_FINAL_KEY,
    }
    values = {
        LATENT_BUNDLE_FINAL_KEY: final_decode_latents,
        "pred_latents": pred_latents,
        "target_latents": target_latents,
        "src_latents": src_latents,
        "chunk_masks": chunk_masks,
    }
    for key, value in values.items():
        detached = _detach_cpu_latent_value(value)
        if detached is not None:
            payload[key] = detached
    return payload if len(payload) > 2 else None


def _save_latent_bundle(path: str, payload: dict[str, Any]) -> None:
    """Persist a latent bundle to disk using torch serialization."""
    torch.save(payload, path)


def _load_latent_bundle(path: str) -> dict[str, Any]:
    """Load a latent bundle payload from disk."""
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Latent bundle payload must be a dictionary")
    return payload


def _latent_value_to_numpy(value: Any) -> Optional[np.ndarray]:
    """Normalize a latent tensor or array to ``[T, C]`` float32 numpy form."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        array = value
    elif torch.is_tensor(value):
        array = value.detach().cpu().float().numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected latent array with 2 dims, got {array.ndim}")
    return array.astype(np.float32)
