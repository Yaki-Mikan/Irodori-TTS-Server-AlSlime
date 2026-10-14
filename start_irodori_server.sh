#!/usr/bin/env bash
# Linux 用の起動スクリプト（start_irodori_server.bat と同じ手順）。
# 動作未確認: Linux の GPU 環境で実際に起動・停止を確かめていない。
set -euo pipefail
cd "$(dirname "$0")"
HERE="$PWD"
export UV_PYTHON_INSTALL_DIR="$HERE/.python"
export UV_CACHE_DIR="$HERE/.uv-cache"
export UV_PROJECT_ENVIRONMENT="$HERE/.venv"
export HF_HOME="$HERE/.huggingface"
export TORCH_HOME="$HERE/.torch-cache"
UV="$HERE/.tools/uv"
LOCK_STAMP="$HERE/.venv/.alslime-lock-sha256"

if [ ! -x "$UV" ]; then
    echo "[setup] Downloading uv..."
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$HERE/.tools" UV_NO_MODIFY_PATH=1 sh
fi
if [ ! -x "$UV" ]; then
    echo "[setup] Failed to download uv. Install uv manually and retry."
    exit 1
fi

# uv.lock のハッシュを .venv 内の控えと比べ、環境が無いか依存が変わったときだけ sync する。
lock_hash="$(sha256sum "$HERE/uv.lock" | cut -d' ' -f1)"
stamp_hash=""
if [ -f "$LOCK_STAMP" ] && [ -x "$HERE/.venv/bin/python" ]; then
    stamp_hash="$(cat "$LOCK_STAMP")"
fi
if [ "$stamp_hash" != "$lock_hash" ]; then
    echo "[setup] Building Python environment: uv sync --extra cu128"
    "$UV" sync --extra cu128
    printf '%s\n' "$lock_hash" > "$LOCK_STAMP"
fi

# exec で置き換え、シェル → uv → python の段を一つ減らす（停止時に木ごと止めやすくする）。
# AlSlime が渡した接続先を使う。単独起動時の公開先はこの PC のみにする。
export IRODORI_HOST="${IRODORI_HOST:-127.0.0.1}"
exec "$UV" run --no-sync python -m irodori_openai_tts
