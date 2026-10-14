"""AlSlimeフォークのモデル保管庫。

フォーク追加分: モデルの保存場所の固定、ダウンロード済み判定、進捗付きの
バックグラウンドダウンロード、削除をまとめる。

huggingface_hub はモジュール読み込み時に環境変数から保存先を決めるため、
``apply_models_dir`` は huggingface_hub を import する前に呼ぶ必要がある。
その制約を守るため、このモジュールのトップレベルでは huggingface_hub を
import せず、各関数の中で import する。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .model_catalog import CODEC_APPROX_SIZE_GB, catalog_entry

logger = logging.getLogger(__name__)

CODEC_FILENAME = "weights.pth"


def resolve_models_dir(settings: Settings) -> Path:
    return Path(settings.models_dir).expanduser().resolve()


def apply_models_dir(settings: Settings) -> Path:
    """モデルの保存先を huggingface_hub の hub キャッシュに固定する。

    ``HF_HOME`` はトークンや xet の一時データの置き場としてそのまま残し、
    ``hub`` 配下だけを ``models_dir`` へ向ける。ディレクトリの作成は
    huggingface_hub に任せる（テスト実行時にカレントへ作らないため）。
    """
    models_dir = resolve_models_dir(settings)
    os.environ["HF_HUB_CACHE"] = str(models_dir)
    return models_dir


@dataclass(frozen=True)
class CheckpointSource:
    repo_id: str
    relative_path: Path  # snapshot からの model.safetensors の相対パス
    allow_patterns: list[str]


def checkpoint_source(checkpoint: str) -> CheckpointSource:
    """本家 download_hf_checkpoint と同じ分解（owner/repo または owner/repo/subfolder）。"""
    from irodori_tts.inference_runtime import _split_hf_checkpoint_source

    repo_id, subfolder = _split_hf_checkpoint_source(checkpoint)
    if subfolder is None:
        return CheckpointSource(repo_id, Path("model.safetensors"), ["model.safetensors", "tokenizer/*"])
    relative = Path(subfolder) / "model.safetensors"
    return CheckpointSource(
        repo_id, relative, [relative.as_posix(), f"{subfolder}/tokenizer/*", "tokenizer/*"]
    )


def is_checkpoint_downloaded(checkpoint: str) -> bool:
    from huggingface_hub import snapshot_download

    try:
        source = checkpoint_source(checkpoint)
        snapshot_dir = Path(
            snapshot_download(
                repo_id=source.repo_id, allow_patterns=source.allow_patterns, local_files_only=True
            )
        )
    except Exception:
        return False
    return (snapshot_dir / source.relative_path).is_file()


def is_codec_downloaded(codec_repo: str) -> bool:
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo_id=codec_repo, filename=CODEC_FILENAME, local_files_only=True)
    except Exception:
        return False
    return Path(path).is_file()


# ---- 進捗の集計 ----


class ProgressAggregator:
    """huggingface_hub が作る進捗バーから、書き込み済みバイト数と合計を集計する。

    バーが無効化されていると tqdm 自身の ``n`` は増えないため、``update`` で
    渡された値を自前で数える。合計はファイルの大きさが分かった時点でバーの
    ``total`` へ属性代入されるため、読む時にバーから取り直す。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bars: list[tuple[Any, list[int]]] = []

    def register(self, bar: Any, initial: int) -> list[int]:
        counter = [int(initial or 0)]
        with self._lock:
            self._bars.append((bar, counter))
        return counter

    def snapshot(self) -> tuple[int, int]:
        with self._lock:
            downloaded = sum(counter[0] for _, counter in self._bars)
            total = sum(int(getattr(bar, "total", 0) or 0) for bar, _ in self._bars)
        return downloaded, total


def make_progress_tqdm(aggregator: ProgressAggregator) -> type:
    from huggingface_hub.utils import tqdm as hf_tqdm

    class _ProgressTqdm(hf_tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            unit = kwargs.get("unit")
            desc = str(kwargs.get("desc") or "")
            name = str(kwargs.get("name") or "")
            # 対象はバイト単位のバーのうち、ネットワーク転送量ではなく
            # ディスクへの書き込み量を表すもの（転送量は重複排除で前後する）。
            self._counter: list[int] | None = None
            super().__init__(*args, **kwargs)
            if unit == "B" and not name.endswith(".transfer") and not desc.startswith("Downloading bytes"):
                self._counter = aggregator.register(self, int(kwargs.get("initial") or 0))

        def update(self, n: Any = 1) -> Any:
            if self._counter is not None and n:
                self._counter[0] += int(n)
            return super().update(n)

    return _ProgressTqdm


# ---- ダウンロード ----


class DownloadBusyError(RuntimeError):
    pass


@dataclass
class DownloadState:
    active: bool = False
    checkpoint: str | None = None
    phase: str = "idle"  # idle / codec / model / done / error
    downloaded_bytes: int = 0
    total_bytes: int = 0
    error: str | None = None
    finished_at: float | None = None
    aggregator: ProgressAggregator | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, Any]:
        downloaded, total = self.downloaded_bytes, self.total_bytes
        if self.aggregator is not None:
            downloaded, total = self.aggregator.snapshot()
        percent = round(downloaded * 100.0 / total, 1) if total > 0 else None
        if self.phase == "done":
            percent = 100.0
        return {
            "active": self.active,
            "checkpoint": self.checkpoint,
            "phase": self.phase,
            "downloaded_bytes": downloaded,
            "total_bytes": total,
            "percent": percent,
            "error": self.error,
            "finished_at": self.finished_at,
        }


class ModelDownloader:
    """カタログのチェックポイントを別スレッドで取得する（同時1件）。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._state = DownloadState()
        self._thread: threading.Thread | None = None

    def state(self) -> dict[str, Any]:
        with self._lock:
            return self._state.as_dict()

    def is_downloading(self, checkpoint: str | None = None) -> bool:
        with self._lock:
            if not self._state.active:
                return False
            return checkpoint is None or self._state.checkpoint == str(checkpoint).strip()

    def start(self, checkpoint: str) -> str:
        """取得を開始する。戻り値は ``started`` か ``already_downloaded``。"""
        entry = catalog_entry(checkpoint)
        if entry is None:
            raise ValueError(f"Unknown checkpoint: {checkpoint}")
        codec_repo = str(self.settings.codec_repo)
        with self._lock:
            if self._state.active:
                raise DownloadBusyError(
                    f"Another download is in progress: {self._state.checkpoint}"
                )
            if is_checkpoint_downloaded(entry.checkpoint) and is_codec_downloaded(codec_repo):
                return "already_downloaded"
            aggregator = ProgressAggregator()
            self._state = DownloadState(
                active=True, checkpoint=entry.checkpoint, phase="codec", aggregator=aggregator
            )
            self._thread = threading.Thread(
                target=self._run, args=(entry.checkpoint, codec_repo, aggregator), daemon=True
            )
            try:
                self._thread.start()
            except Exception as exc:
                self._state.active = False
                self._state.phase = "error"
                self._state.error = str(exc)
                self._state.finished_at = time.time()
                raise
        return "started"

    def delete(self, checkpoint: str) -> bool:
        """使用中の確認から削除完了まで、新しい取得の登録を止める。"""
        with self._lock:
            if self._state.active and self._state.checkpoint == checkpoint:
                raise DownloadBusyError("Checkpoint is still downloading.")
            return delete_checkpoint(self.settings, checkpoint)

    def _run(self, checkpoint: str, codec_repo: str, aggregator: ProgressAggregator) -> None:
        try:
            from huggingface_hub import hf_hub_download, snapshot_download

            tqdm_class = make_progress_tqdm(aggregator)
            if not is_codec_downloaded(codec_repo):
                logger.info("downloading codec from hf://%s", codec_repo)
                hf_hub_download(repo_id=codec_repo, filename=CODEC_FILENAME, tqdm_class=tqdm_class)
            self._set_phase("model")
            source = checkpoint_source(checkpoint)
            logger.info("downloading checkpoint from hf://%s", checkpoint)
            snapshot_download(
                repo_id=source.repo_id, allow_patterns=source.allow_patterns, tqdm_class=tqdm_class
            )
            if not is_checkpoint_downloaded(checkpoint):
                raise FileNotFoundError(f"model.safetensors not found after download: {checkpoint}")
            self._finish("done", None)
            logger.info("download completed: %s", checkpoint)
        except Exception as exc:  # 失敗理由は状態として画面へ返す。
            logger.exception("download failed: %s", checkpoint)
            self._finish("error", str(exc))

    def _set_phase(self, phase: str) -> None:
        with self._lock:
            self._state.phase = phase

    def _finish(self, phase: str, error: str | None) -> None:
        with self._lock:
            downloaded, total = (0, 0)
            if self._state.aggregator is not None:
                downloaded, total = self._state.aggregator.snapshot()
            self._state.downloaded_bytes = downloaded
            self._state.total_bytes = total
            self._state.aggregator = None
            self._state.active = False
            self._state.phase = phase
            self._state.error = error
            self._state.finished_at = time.time()


# ---- 削除 ----


def delete_checkpoint(settings: Settings, checkpoint: str) -> bool:
    """カタログのチェックポイントを保存場所から消す。消した場合は真。"""
    from huggingface_hub import scan_cache_dir

    repo_id = checkpoint_source(checkpoint).repo_id
    models_dir = resolve_models_dir(settings)
    if not models_dir.is_dir():
        return False
    cache_info = scan_cache_dir(models_dir)
    for repo in cache_info.repos:
        if repo.repo_type == "model" and repo.repo_id == repo_id:
            hashes = [revision.commit_hash for revision in repo.revisions]
            if not hashes:
                return False
            cache_info.delete_revisions(*hashes).execute()
            logger.info("deleted checkpoint %s (%d revisions)", repo_id, len(hashes))
            return True
    return False


def codec_summary(settings: Settings) -> dict[str, Any]:
    codec_repo = str(settings.codec_repo)
    return {
        "repo": codec_repo,
        "approx_size_gb": CODEC_APPROX_SIZE_GB,
        "downloaded": is_codec_downloaded(codec_repo),
    }
