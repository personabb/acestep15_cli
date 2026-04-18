"""CLI 向け text2music task 固定ユーティリティ。

audio_codes が提供されると ACE-Step 内部で text2music → cover に解決されるが、
CLI では resolved task 名だけを text2music に保ちたい。

ただし前回の回帰調査で分かった通り、instruction や cover-like conditioning を
一緒に潰すと音が壊れるため、この helper は vendor の instruction 解決や
mode flag には触らない。
"""

from __future__ import annotations

import contextlib
from typing import Any


@contextlib.contextmanager
def force_text2music_task_type(dit_handler: Any):
    """Keep the resolved task name as ``text2music`` without muting conditioning."""
    original_resolve = dit_handler._resolve_generate_music_task

    def fixed_resolve(task_type, audio_code_string, instruction=None):
        if task_type == "text2music":
            _, resolved_instruction = original_resolve(task_type, audio_code_string, instruction)
            return "text2music", resolved_instruction
        return original_resolve(task_type, audio_code_string, instruction)

    dit_handler._resolve_generate_music_task = fixed_resolve

    try:
        yield
    finally:
        dit_handler._resolve_generate_music_task = original_resolve
