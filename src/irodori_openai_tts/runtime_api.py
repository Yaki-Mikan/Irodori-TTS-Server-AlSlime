"""AlSlimeフォークのランタイム管理ヘルパー。

フォーク追加分: ランタイムプロファイル（通常／省メモリ）のファイル永続化と、
サーバー自身の再起動ヘルパーを提供する。プロファイルごとの設定値はここで
一元管理し（複数箇所へ固定値を書かない）、起動時に Settings へ適用する。
保存したプロファイルは次回（再）起動から有効になる。
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .config import Settings

logger = logging.getLogger(__name__)

PROFILE_FILE = Path("runtime_profile.json")

# ランタイムプロファイル。プロファイル別の設定値の管理はここだけで行う。
# 省メモリモードは codec を CPU へ逃がして VRAM 使用量を抑える。
RUNTIME_PROFILES: list[dict[str, Any]] = [
    {
        "id": "normal",
        "label": "通常モード",
        "settings": {
            "model_device": "auto",
            "codec_device": "auto",
        },
    },
    {
        "id": "low_vram",
        "label": "省メモリモード",
        "settings": {
            "model_device": "auto",
            "codec_device": "cpu",
        },
    },
]

DEFAULT_PROFILE_ID = "normal"

# 起動時に Settings へ実際に適用したプロファイル
# （保存済みの選択との差分から restart_required を導く）。
_active_profile_id: str = DEFAULT_PROFILE_ID


def profile_by_id(profile_id: str) -> dict[str, Any] | None:
    for profile in RUNTIME_PROFILES:
        if profile["id"] == profile_id:
            return profile
    return None


def load_selected_profile_id() -> str:
    try:
        with PROFILE_FILE.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        profile_id = str(payload.get("profile", "")).strip()
    except (OSError, ValueError):
        return DEFAULT_PROFILE_ID
    if profile_by_id(profile_id) is None:
        return DEFAULT_PROFILE_ID
    return profile_id


def save_selected_profile_id(profile_id: str) -> None:
    if profile_by_id(profile_id) is None:
        raise ValueError(f"Unknown runtime profile {profile_id!r}.")
    with PROFILE_FILE.open("w", encoding="utf-8") as handle:
        json.dump({"profile": profile_id}, handle, ensure_ascii=False, indent=2)


def apply_saved_profile(settings: Settings) -> None:
    """保存済みプロファイルを Settings へ適用する（起動時に1回呼ぶ）。"""
    global _active_profile_id
    profile_id = load_selected_profile_id()
    profile = profile_by_id(profile_id)
    if profile is None:  # 保存値は load_selected_profile_id で検証済みのため通常来ない。
        return
    for key, value in profile["settings"].items():
        setattr(settings, key, value)
    _active_profile_id = profile_id
    logger.info("runtime profile applied: %s", profile_id)


def active_profile_id() -> str:
    return _active_profile_id


def profiles_response() -> dict[str, Any]:
    selected = load_selected_profile_id()
    return {
        "profiles": RUNTIME_PROFILES,
        "selected_profile": selected,
        "active_profile": _active_profile_id,
        "restart_required": selected != _active_profile_id,
    }


def restart_server() -> None:
    """現在のサーバープロセスを再起動する。

    応答を返した後、少し遅らせて ``python -m irodori_openai_tts`` と元の引数で
    プロセスを再実行する。再起動ポリシー付きのコンテナ運用では、プロセスの
    終了を監視側の再起動に任せる形でも成立する。
    """

    def _restart() -> None:
        time.sleep(0.5)
        args = [sys.executable, "-m", "irodori_openai_tts", *sys.argv[1:]]
        logger.info("restarting server: %s", args)
        if os.name == "nt":
            # Windows の execv はソケット等の引き継ぎが不完全なため、
            # 新プロセスを起動してから自分は即終了する。
            subprocess.Popen(
                args,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            os._exit(0)
        os.execv(sys.executable, args)

    threading.Thread(target=_restart, daemon=True).start()
