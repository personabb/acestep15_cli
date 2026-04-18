"""
infer_sft.py  –  ACE-Step 音楽生成スクリプト（SFT / Turbo / Base モデル対応）

使い方:
    python infer_sft.py

設定は下の「グローバル設定」セクションを編集してください。
caption / lyrics は inputs/ フォルダ内のテキストファイルで管理します。

生成結果は OUTPUT_DIR/YYYYMMDD_HHMMSS/ フォルダに保存される:
  01.wav, 01_params.json (audio_codes 含む), 01_latents.npy (25Hz VAE 潜在),
  lm_metadata.json, session.json
これらは infer_retake.py から再利用できる。
"""

import sys
import datetime
from pathlib import Path

import torch

sys.path.insert(0, '.')
sys.path.insert(0, './ACE-Step-1.5')

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, GenerationResult, generate_music
from acestep.llm_inference import LLMHandler
from cli_exe_infer_utils.sft_lm import should_use_scripted_phase2_lm
from cli_exe_infer_utils.sft_rng import set_torch_seed
from cli_exe_infer_utils.sft_workflow import run_scripted_phase2_generation
from cli_exe_infer_utils.session import make_session_dir, save_session_artifacts

# ============================================================
# モデル設定
# ============================================================

# DiTモデル（音楽生成の主要モデル）
#   'acestep-v15-turbo'  : 8ステップ, 高速, CFGなし
#   'acestep-v15-sft'    : 50ステップ, 高品質（SFTファインチューン）
#   'acestep-v15-base'   : 50ステップ, extract/lego/complete タスク対応
DIT_MODEL = 'acestep-v15-sft'

# LMモデル（歌詞・楽曲構造生成用の言語モデル）
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
#CAPTION_FILE = './inputs/caption_bgm.txt'
#LYRICS_FILE  = './inputs/lyrics_bgm.txt'
CAPTION_FILE = './inputs/caption_mirai.txt'
LYRICS_FILE  = './inputs/lyrics_mirai.txt'
#CAPTION_FILE = './inputs/caption.txt'
#LYRICS_FILE  = './inputs/lyrics.txt'

# ============================================================
# 生成パラメータ
# ============================================================
TASK_TYPE       = 'text2music'
BPM             = None #167    # 曲のテンポ（BPM）。None で自動推定
KEYSCALE        = "" #"" #"D major"    # 曲のキー（例: 'C', 'G#'）。"" で自動推定
TIMESIGNATURE    = "4"    # 曲の拍子（例: '4/4'）。"" で自動推定
VOCAL_LANGUAGE   = "ja"    # 歌詞の言語（例: 'ja', 'en'）。"unknown" で自動推定
INSTRUMENTAL    =  False #True   # True: インスト曲（歌詞なし）
DURATION        =  -1 #240 #-1     # 生成時間（秒）。-1 で自動
SEED            = 2392007029 #86689620 #54904495 #-1 #4276084885      # 乱数シード。-1 でランダム（同じシードで同じ結果になります）
COT_SEED        = 254843255 #86689620                               # CoT（LM Phase 1）専用シード。SEED と独立して指定できる
INFERENCE_STEPS = 64      # turboモデルは 8 推奨
GUIDANCE_SCALE  = 7.0
THINKING = True            # True: 生成前に思考過程を表示（LMがある場合のみ）
USE_COT_METAS = True     # True: CoTプロンプトで複数の思考過程を生成（LLMがある場合のみ）
LM_NEGATIVE_PROMPT = 'low quality, dissonant harmony, unstable rhythm, broken rhythm, monotonous rhythm, muddy mix, clipped audio, weak melody, chaotic arrangement, unnatural transitions'


BATCH_SIZE   = 1
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

    # LLM 初期化
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
    """音楽生成を実行して GenerationResult を返す。

    固定 seed で LM が audio_codes を生成する場合は、single / multi ともに
    LM Phase 2 を track seed の先頭から開始し、genres フィールドも残す。
    ファイルへの保存は呼び出し元 (main) が save_session_artifacts() で行う。

    Args:
        dit_handler: 初期化済み AceStepHandler
        llm_handler: 初期化済み LLMHandler
        caption:     音楽スタイル記述テキスト
        lyrics:      歌詞テキスト

    Returns:
        GenerationResult（audios に tensor と params が含まれる）
    """
    params = GenerationParams(
        task_type=TASK_TYPE,
        caption=caption,
        lyrics=lyrics,
        instrumental=INSTRUMENTAL,
        bpm=BPM,
        keyscale=KEYSCALE,
        timesignature=TIMESIGNATURE,
        vocal_language=VOCAL_LANGUAGE,
        duration=DURATION,
        seed=SEED,
        inference_steps=INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        thinking=THINKING,
        use_cot_metas=USE_COT_METAS,
        lm_negative_prompt=LM_NEGATIVE_PROMPT,
    )
    if should_use_scripted_phase2_lm(BATCH_SIZE, params, llm_handler):
        print("\n音楽生成中... (LM Phase 2 を track seed から実行)")
        result = run_scripted_phase2_generation(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            params=params,
            batch_size=BATCH_SIZE,
            audio_format=AUDIO_FORMAT,
            cot_seed=COT_SEED,
        )
        print(f"\n生成に成功しました！ ({len(result.audios)} トラック)")
        return result

    # SEED が指定されている場合は seeds を直接渡す（params.seed は使われないため）
    seeds_for_config = [SEED + i for i in range(BATCH_SIZE)] if SEED != -1 else None
    config = GenerationConfig(
        batch_size=BATCH_SIZE,
        audio_format=AUDIO_FORMAT,
        use_random_seed=(SEED == -1),
        seeds=seeds_for_config,
    )

    # ACE-Step コアを修正せずに Phase 1 (CoT) を決定論的にする。
    # generate_music() 呼び出し直前に torch シードをセットしておくと、
    # Phase 1 がこの状態から実行され、Phase 2 は _run_pt() の per-item seed で上書きされる。
    # バッチの場合: seeds_for_config[0] = ユーザー指定シード、[1..] = SEED+i で固定。
    if seeds_for_config:
        actual_cot_seed = seeds_for_config[0]
        set_torch_seed(actual_cot_seed)
    else:
        # SEED=-1 のとき: 明示的なランダムシードを生成して CoT Phase 1 の再現性を確保する
        actual_cot_seed = int(torch.randint(0, 2**32, (1,)).item())
        set_torch_seed(actual_cot_seed)

    print("\n音楽生成中...")
    # save_dir=None: generate_music はファイル保存しない（tensor のみ返す）
    # アーティファクト保存は save_session_artifacts() が担当
    result = generate_music(dit_handler, llm_handler, params, config, save_dir=None)

    if result.success:
        for audio_dict in result.audios:
            audio_dict.setdefault("params", {})["cot_seed"] = actual_cot_seed
        print(f"\n生成に成功しました！ ({len(result.audios)} トラック)")
        print(f"cot_seed (実際に使用): {actual_cot_seed}")
    else:
        print(f"\n生成に失敗しました: {result.error}")

    return result


def main() -> None:
    """全処理を順に実行するエントリーポイント。"""
    print("=" * 60)
    print(f"DiT モデル        : {DIT_MODEL}")
    print(f"LM  モデル        : {LM_MODEL}")
    print(f"チェックポイント  : {CHECKPOINT_DIR}")
    print(f"DiT CPUオフロード : {DIT_OFFLOAD_TO_CPU}  (量子化: {DIT_QUANTIZATION})")
    print(f"LLM CPUオフロード : {LLM_OFFLOAD_TO_CPU}")
    print(f"Caption ファイル  : {CAPTION_FILE}")
    print(f"Lyrics  ファイル  : {LYRICS_FILE}")
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
            "timestamp":       datetime.datetime.now().isoformat(),
            "task_type":       TASK_TYPE,
            "dit_model":       DIT_MODEL,
            "lm_model":        LM_MODEL,
            "inference_steps": INFERENCE_STEPS,
            "guidance_scale":  GUIDANCE_SCALE,
            "batch_size":      BATCH_SIZE,
            "seed":            SEED,
            "cot_seed":        COT_SEED,
            "caption_file":    CAPTION_FILE,
            "lyrics_file":     LYRICS_FILE,
            "debug_mode":      DEBUG_MODE,
        }
        save_session_artifacts(
            result,
            session_dir,
            session_meta,
            save_debug_artifacts=DEBUG_MODE,
        )


if __name__ == '__main__':
    main()
