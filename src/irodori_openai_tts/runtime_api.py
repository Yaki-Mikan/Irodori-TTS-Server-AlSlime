"""AlSlimeフォークのランタイム管理ヘルパー。

フォーク追加分: ランタイムプロファイル（通常／省メモリ）・計算精度・
最後にロードしたチェックポイントのファイル永続化と、サーバー自身の
再起動ヘルパーを提供する。プロファイル
ごとの設定値はここで一元管理し（複数箇所へ固定値を書かない）、起動時に
Settings へ適用する。保存した選択は次回（再）起動から有効になる。
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

# 本体（音声を生成するモデル）の計算精度。コーデックは対象外で常に既定のまま。
# bf16 は CUDA / XPU でしか動かないため、使えない環境では選択肢から外す。
MODEL_PRECISIONS: list[dict[str, str]] = [
    {"id": "fp32", "label": "fp32（高精度・重い）"},
    {"id": "bf16", "label": "bf16（省メモリ・高速）"},
]

# 起動時に Settings へ実際に適用したプロファイルと計算精度
# （保存済みの選択との差分から restart_required を導く）。
_active_profile_id: str = DEFAULT_PROFILE_ID
_active_precision: str = "fp32"
_available_precisions: list[str] = ["fp32"]


def profile_by_id(profile_id: str) -> dict[str, Any] | None:
    for profile in RUNTIME_PROFILES:
        if profile["id"] == profile_id:
            return profile
    return None


def _detect_available_precisions(settings: Settings) -> list[str]:
    if str(settings.model_device).strip().lower() == "cpu":
        return ["fp32"]
    try:
        import torch
    except ImportError:
        return ["fp32"]
    accelerated = torch.cuda.is_available()
    if not accelerated and hasattr(torch, "xpu"):
        accelerated = bool(torch.xpu.is_available())
    return ["fp32", "bf16"] if accelerated else ["fp32"]


def _load_profile_file() -> dict[str, Any]:
    try:
        with PROFILE_FILE.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _save_profile_file(payload: dict[str, Any]) -> None:
    with PROFILE_FILE.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_selected_profile_id() -> str:
    profile_id = str(_load_profile_file().get("profile", "")).strip()
    if profile_by_id(profile_id) is None:
        return DEFAULT_PROFILE_ID
    return profile_id


def load_selected_precision() -> str:
    """保存済みの計算精度。未保存・この環境で使えない値なら起動時の値を返す。"""
    precision = str(_load_profile_file().get("model_precision", "")).strip().lower()
    if precision not in _available_precisions:
        return _active_precision
    return precision


def save_selected_profile_id(profile_id: str) -> None:
    if profile_by_id(profile_id) is None:
        raise ValueError(f"Unknown runtime profile {profile_id!r}.")
    payload = _load_profile_file()
    payload["profile"] = profile_id
    _save_profile_file(payload)


def save_selected_precision(precision: str) -> None:
    precision = str(precision).strip().lower()
    if precision not in _available_precisions:
        raise ValueError(f"Model precision {precision!r} is not available on this server.")
    payload = _load_profile_file()
    payload["model_precision"] = precision
    _save_profile_file(payload)


def save_last_checkpoint(settings: Settings) -> None:
    """ロードに成功したチェックポイントを記憶する（次回起動時の選択になる）。"""
    local = settings.checkpoint is not None and str(settings.checkpoint).strip() != ""
    value = str(settings.checkpoint if local else settings.hf_checkpoint).strip()
    payload = _load_profile_file()
    payload["checkpoint"] = value
    payload["checkpoint_local"] = local
    _save_profile_file(payload)


def forget_checkpoint(checkpoint: str) -> None:
    """記憶しているチェックポイントが削除されたら記憶も消す（次回は既定のモデル）。"""
    payload = _load_profile_file()
    if str(payload.get("checkpoint", "")).strip() != str(checkpoint).strip():
        return
    payload.pop("checkpoint", None)
    payload.pop("checkpoint_local", None)
    _save_profile_file(payload)


def _apply_saved_checkpoint(settings: Settings) -> None:
    payload = _load_profile_file()
    value = str(payload.get("checkpoint", "")).strip()
    if value == "":
        return
    if payload.get("checkpoint_local"):
        if not Path(value).expanduser().is_file():
            logger.warning("saved checkpoint file not found; using the default model: %s", value)
            return
        settings.checkpoint = value
    else:
        settings.checkpoint = None
        settings.hf_checkpoint = value
    logger.info("saved checkpoint applied: %s", value)


def apply_saved_profile(settings: Settings) -> None:
    """保存済みプロファイル・計算精度・最後にロードしたチェックポイントを
    Settings へ適用する（起動時に1回呼ぶ）。"""
    _apply_saved_checkpoint(settings)
    global _active_profile_id, _active_precision, _available_precisions
    profile_id = load_selected_profile_id()
    profile = profile_by_id(profile_id)
    if profile is not None:  # 保存値は load_selected_profile_id で検証済みのため通常は真。
        for key, value in profile["settings"].items():
            setattr(settings, key, value)
        _active_profile_id = profile_id
        logger.info("runtime profile applied: %s", profile_id)

    _available_precisions = _detect_available_precisions(settings)
    saved = str(_load_profile_file().get("model_precision", "")).strip().lower()
    if saved:
        if saved in _available_precisions:
            settings.model_precision = saved
        else:
            logger.warning(
                "saved model precision %r is not available on this server; using fp32", saved
            )
            settings.model_precision = "fp32"
    elif str(settings.model_precision).strip().lower() not in _available_precisions:
        settings.model_precision = "fp32"
    _active_precision = str(settings.model_precision).strip().lower()
    logger.info("model precision applied: %s", _active_precision)


def active_profile_id() -> str:
    return _active_profile_id


def profiles_response() -> dict[str, Any]:
    selected = load_selected_profile_id()
    selected_precision = load_selected_precision()
    return {
        "profiles": RUNTIME_PROFILES,
        "selected_profile": selected,
        "active_profile": _active_profile_id,
        "precisions": [p for p in MODEL_PRECISIONS if p["id"] in _available_precisions],
        "selected_precision": selected_precision,
        "active_precision": _active_precision,
        "restart_required": (
            selected != _active_profile_id or selected_precision != _active_precision
        ),
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
