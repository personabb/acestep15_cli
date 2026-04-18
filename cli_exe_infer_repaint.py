"""
infer_repaint.py  –  ACE-Step Repaint スクリプト

元音声の指定区間を再生成する。

生成結果は OUTPUT_DIR/YYYYMMDD_HHMMSS/ フォルダに保存される:
  01.wav, 01_params.json, 01_latents.npy (25Hz VAE 潜在),
  lm_metadata.json, session.json
これらは infer_retake.py から再利用できる。
"""

import os
import sys
import datetime
from pathlib import Path

import torch

sys.path.insert(0, '.')
sys.path.insert(0, './ACE-Step-1.5')

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, GenerationResult, generate_music
from acestep.llm_inference import LLMHandler
from cli_exe_infer_utils.audio_input_compat import apply_soundfile_audio_input_compat
from cli_exe_infer_utils.session import make_session_dir, save_session_artifacts

# ============================================================
# モデル設定
# ============================================================

# DiTモデル（repaint 対応: sft / turbo のみ、base は不可）
#   'acestep-v15-sft'    : 50ステップ、高品質（SFTファインチューン）
#   'acestep-v15-turbo'  : 8ステップ、高速（INFERENCE_STEPS を 8 に変更推奨）
DIT_MODEL = 'acestep-v15-sft'

# LMモデル（repaint では内部でスキップされる。初期化のみ）
#   'acestep-5Hz-lm-0.6B' : 軽量
#   'acestep-5Hz-lm-1.7B' : 標準（mainパッケージに含まれる）
#   'acestep-5Hz-lm-4B'   : 高品質, VRAM多く必要
LM_MODEL = 'acestep-5Hz-lm-1.7B'

# チェックポイント保存先
CHECKPOINT_DIR = './ACE-Step-1.5/checkpoints'

# ============================================================
# CPUオフロード設定（VRAMが少ない場合は True 推奨）
# ============================================================

DIT_OFFLOAD_TO_CPU = True   # True: DiT全体をCPUオフロード（VRAM節約・速度低下）
DIT_QUANTIZATION   = None    # 'int8': さらにVRAM削減（Noneで無効）
LLM_OFFLOAD_TO_CPU = True   # True: 推論時のみGPUへ移動（backend='pt'のみ有効）

# ============================================================
# 入力ファイル（caption / lyrics を外部テキストで管理）
# ============================================================
#CAPTION_FILE = './inputs/repaint_caption.txt'   # repaint 専用（区間のスタイル記述）
#LYRICS_FILE  = './inputs/repaint_lyrics.txt'    # repaint 専用（区間の歌詞のみ）
CAPTION_FILE = './inputs/caption_siokaze.txt'
LYRICS_FILE  = './inputs/lyrics_siokaze.txt'

# ============================================================
# Repaint 固有パラメータ
# ============================================================

# 元音声ファイルパス（再生成の対象となる音声）
SRC_AUDIO = 'output/20260406_032504/01.wav'

# 再生成する区間（秒）
#REPAINTING_START = 140.0    # 再生成開始時刻（秒）
#REPAINTING_END   = 166.0   # 再生成終了時刻（秒）、-1 でファイル末尾まで
REPAINTING_START = 63.0    # 再生成開始時刻（秒）
REPAINTING_END   = 66.0   # 再生成終了時刻（秒）、-1 でファイル末尾まで

# 元音声構造の保持強度（repaintタスクの場合は必ず 1.0 に設定）
AUDIO_COVER_STRENGTH = 1.0

# ============================================================
# 生成パラメータ
# ============================================================
TASK_TYPE       = 'repaint'
SEED            = 54904495
INFERENCE_STEPS = 64      # turboモデルは 8 推奨
GUIDANCE_SCALE  = 7.0
VOCAL_LANGUAGE = "ja"    # 歌詞の言語（例: 'ja', 'en'）。"unknown" で自動推定

BATCH_SIZE   = 4
AUDIO_FORMAT = 'wav'      # 'wav' or 'flac'
OUTPUT_DIR   = './output'
DEBUG_MODE   = False


# ============================================================
# 関数定義
# ============================================================

def setup_checkpoint_dir() -> None:
    """チェックポイントディレクトリを作成する。"""
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    print(f"チェックポイントディレクトリ: {CHECKPOINT_DIR}")


def download_lm_if_needed() -> None:
    """1.7B以外のLMモデルを事前ダウンロードする（llm_handler は自動DLしないため）。"""
    if LM_MODEL == 'acestep-5Hz-lm-1.7B':
        print(f"[LM: {LM_MODEL}] mainモデルに含まれるため自動取得されます")
        return

    from acestep.model_downloader import ensure_lm_model
    print(f"[LM: {LM_MODEL}] llm_handler用に事前ダウンロード中...")
    ok, msg = ensure_lm_model(LM_MODEL, checkpoints_dir=Path(CHECKPOINT_DIR))
    print(f"  {msg}")
    if not ok:
        raise RuntimeError(f"LMモデルのダウンロードに失敗: {msg}")


def detect_device() -> None:
    """GPU/CPUを判定し、利用可能なデバイス情報を表示する。"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        bf16_ok  = torch.cuda.is_bf16_supported()
        dtype    = torch.bfloat16 if bf16_ok else torch.float16
        device   = 'cuda'
        print(f"GPU: {gpu_name}  (bfloat16: {bf16_ok})")
    else:
        dtype  = torch.float32
        device = 'cpu'
        print("GPU: 利用不可（CPU使用）")

    print(f"dtype: {dtype}  device: {device}")


def load_inputs() -> tuple:
    """CAPTION_FILE / LYRICS_FILE からテキストを読み込む。

    Returns:
        (caption, lyrics): str, str
    """
    with open(CAPTION_FILE, encoding='utf-8') as f:
        caption = f.read().strip()
    with open(LYRICS_FILE, encoding='utf-8') as f:
        lyrics = f.read()

    print(f"Caption ({len(caption)} chars): {caption[:80]}...")
    print(f"Lyrics  ({len(lyrics)} chars): 読み込み完了")
    return caption, lyrics


def initialize_handlers() -> tuple:
    """DiTハンドラとLLMハンドラを初期化して返す。

    Returns:
        (dit_handler, llm_handler)
    """
    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()

    # DiT 初期化
    print(f"\nDiTモデル初期化中... ({DIT_MODEL})")
    status_msg, success = dit_handler.initialize_service(
        project_root='./ACE-Step-1.5',
        config_path=DIT_MODEL,
        device='auto',
        use_mlx_dit=False,
        offload_to_cpu=DIT_OFFLOAD_TO_CPU,
        quantization=DIT_QUANTIZATION,
    )
    if not success:
        raise RuntimeError(f"DiT初期化失敗: {status_msg}")
    print(f"DiT初期化完了: {status_msg}")

    # LLM 初期化（repaint では内部でスキップされるが、ハンドラは必要）
    print(f"\nLLM初期化中... ({LM_MODEL})")
    llm_status_msg, llm_success = llm_handler.initialize(
        checkpoint_dir=CHECKPOINT_DIR,
        lm_model_path=LM_MODEL,
        backend='pt',
        device='auto',
        offload_to_cpu=LLM_OFFLOAD_TO_CPU,
    )
    if not llm_success:
        print(f"LLM初期化失敗: {llm_status_msg}（LLMなしで続行します）")
    else:
        print(f"LLM初期化完了: {llm_status_msg}")

    return dit_handler, llm_handler


def run_generation(dit_handler: AceStepHandler, llm_handler: LLMHandler,
                   caption: str, lyrics: str) -> GenerationResult:
    """Repaint を実行して GenerationResult を返す。

    ファイルへの保存は呼び出し元 (main) が save_session_artifacts() で行う。

    Args:
        dit_handler: 初期化済み AceStepHandler
        llm_handler: 初期化済み LLMHandler
        caption:     再生成区間のスタイル記述テキスト
        lyrics:      再生成区間の歌詞テキスト

    Returns:
        GenerationResult（audios に tensor と params が含まれる）
    """
    params = GenerationParams(
        task_type=TASK_TYPE,
        src_audio=SRC_AUDIO,
        repainting_start=REPAINTING_START,
        repainting_end=REPAINTING_END,
        vocal_language=VOCAL_LANGUAGE,
        #audio_cover_strength=AUDIO_COVER_STRENGTH,
        caption=caption,
        lyrics=lyrics,
        seed=SEED,
        inference_steps=INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    )
    # SEED が指定されている場合は seeds を直接渡す（params.seed は使われないため）
    seeds_for_config = [SEED + i for i in range(BATCH_SIZE)] if SEED != -1 else None
    config = GenerationConfig(
        batch_size=BATCH_SIZE,
        audio_format=AUDIO_FORMAT,
        use_random_seed=(SEED == -1),
        seeds=seeds_for_config,
    )

    apply_soundfile_audio_input_compat(
        dit_handler,
        patch_src_audio=bool(getattr(params, "src_audio", None)),
        patch_reference_audio=bool(getattr(params, "reference_audio", None)),
    )

    print("\nRepaint 実行中...")
    # save_dir=None: generate_music はファイル保存しない（tensor のみ返す）
    # アーティファクト保存は save_session_artifacts() が担当
    result = generate_music(dit_handler, llm_handler, params, config, save_dir=None)

    if result.success:
        print(f"\nRepaint に成功しました！ ({len(result.audios)} トラック)")
    else:
        print(f"\nRepaint に失敗しました: {result.error}")

    return result


def main() -> None:
    """全処理を順に実行するエントリーポイント。"""
    print("=" * 60)
    print(f"タスク             : {TASK_TYPE}")
    print(f"DiT モデル        : {DIT_MODEL}")
    print(f"LM  モデル        : {LM_MODEL}")
    print(f"チェックポイント  : {CHECKPOINT_DIR}")
    print(f"DiT CPUオフロード : {DIT_OFFLOAD_TO_CPU}  (量子化: {DIT_QUANTIZATION})")
    print(f"LLM CPUオフロード : {LLM_OFFLOAD_TO_CPU}")
    print(f"Caption ファイル  : {CAPTION_FILE}")
    print(f"Lyrics  ファイル  : {LYRICS_FILE}")
    print("-" * 60)
    print(f"元音声ファイル    : {SRC_AUDIO}")
    print(f"再生成区間        : {REPAINTING_START}秒 〜 "
          f"{'末尾' if REPAINTING_END < 0 else f'{REPAINTING_END}秒'}")
    print(f"音声保持強度      : {AUDIO_COVER_STRENGTH}")
    print("=" * 60)

    setup_checkpoint_dir()
    download_lm_if_needed()
    detect_device()

    caption, lyrics = load_inputs()
    dit_handler, llm_handler = initialize_handlers()
    result = run_generation(dit_handler, llm_handler, caption, lyrics)

    if result.success:
        # セッションフォルダを作成してアーティファクトを保存
        session_dir = make_session_dir(OUTPUT_DIR)
        session_meta = {
            "timestamp":        datetime.datetime.now().isoformat(),
            "task_type":        TASK_TYPE,
            "dit_model":        DIT_MODEL,
            "lm_model":         LM_MODEL,
            "inference_steps":  INFERENCE_STEPS,
            "guidance_scale":   GUIDANCE_SCALE,
            "batch_size":       BATCH_SIZE,
            "caption_file":     CAPTION_FILE,
            "lyrics_file":      LYRICS_FILE,
            "src_audio":        SRC_AUDIO,
            "repainting_start": REPAINTING_START,
            "repainting_end":   REPAINTING_END,
            "audio_cover_strength": AUDIO_COVER_STRENGTH,
            "debug_mode":       DEBUG_MODE,
        }
        save_session_artifacts(
            result,
            session_dir,
            session_meta,
            save_debug_artifacts=DEBUG_MODE,
        )


if __name__ == '__main__':
    main()
