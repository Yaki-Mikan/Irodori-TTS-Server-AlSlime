from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    default_runtime_device,
    download_hf_checkpoint,
)

from .config import Settings

logger = logging.getLogger(__name__)


class RuntimeLoadTimeoutError(RuntimeError):
    pass


class RuntimeManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._runtime: InferenceRuntime | None = None
        self._checkpoint_path: str | None = None

    def get(self) -> InferenceRuntime:
        if self._runtime is not None:
            return self._runtime

        timeout = float(self.settings.model_load_timeout)
        acquired = self._lock.acquire(timeout=max(timeout, 0.0))
        if not acquired:
            raise RuntimeLoadTimeoutError(
                f"Model is still loading. Retry after a moment. timeout={timeout:.1f}s"
            )

        try:
            if self._runtime is None:
                logger.info("loading runtime")
                t0 = time.perf_counter()
                self._checkpoint_path = self._resolve_checkpoint_path()
                logger.info("checkpoint resolved: %s", self._checkpoint_path)
                self._runtime = InferenceRuntime.from_key(
                    RuntimeKey(
                        checkpoint=self._checkpoint_path,
                        model_device=self._resolve_device(self.settings.model_device),
                        codec_repo=str(self.settings.codec_repo),
                        model_precision=str(self.settings.model_precision),
                        codec_device=self._resolve_device(self.settings.codec_device),
                        codec_precision=str(self.settings.codec_precision),
                        codec_deterministic_encode=bool(self.settings.codec_deterministic_encode),
                        codec_deterministic_decode=bool(self.settings.codec_deterministic_decode),
                        compile_model=bool(self.settings.compile_model),
                        compile_dynamic=bool(self.settings.compile_dynamic),
                    )
                )
                elapsed = time.perf_counter() - t0
                logger.info("runtime loaded in %.2fs", elapsed)
            return self._runtime
        finally:
            self._lock.release()

    @property
    def checkpoint_path(self) -> str | None:
        return self._checkpoint_path

    @property
    def is_loaded(self) -> bool:
        return self._runtime is not None

    @property
    def is_loading(self) -> bool:
        return self._runtime is None and self._lock.locked()

    # ---- フォーク追加分: 解放・切替（AlSlime のエンジン管理用） ----

    @property
    def selected_checkpoint(self) -> str:
        """現在設定中のチェックポイント（ローカルパスまたは HF リポジトリID）。"""
        if self.settings.checkpoint is not None and str(self.settings.checkpoint).strip() != "":
            return str(self.settings.checkpoint)
        return str(self.settings.hf_checkpoint)

    def unload(self) -> None:
        """ロード済みランタイムを解放し、GPU メモリのキャッシュも解放する。"""
        with self._lock:
            unloaded = self._runtime is not None
            self._runtime = None
            self._checkpoint_path = None
        if not unloaded:
            return
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # torch 無し・デバイス無しでも解放自体は成立させる。
            pass
        logger.info("runtime unloaded")

    def switch(self, checkpoint: str, load: bool) -> None:
        """チェックポイントを切り替える（要件どおり旧モデルの解放を先に行う）。

        checkpoint がローカルファイルとして存在すればローカル指定、
        そうでなければ HF リポジトリID として扱う。load が真なら
        切替後にそのままロードまで行う。
        """
        value = str(checkpoint).strip()
        if value == "":
            raise ValueError("checkpoint must not be empty.")
        self.unload()
        path = Path(value).expanduser()
        if path.is_file():
            self.settings.checkpoint = value
        else:
            self.settings.checkpoint = None
            self.settings.hf_checkpoint = value
        if load:
            self.get()

    def _resolve_checkpoint_path(self) -> str:
        if self.settings.checkpoint is not None and str(self.settings.checkpoint).strip() != "":
            path = Path(str(self.settings.checkpoint)).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            return str(path)

        repo_id = str(self.settings.hf_checkpoint).strip()
        if repo_id == "":
            raise ValueError("Set IRODORI_CHECKPOINT or IRODORI_HF_CHECKPOINT.")
        logger.info("downloading checkpoint assets from hf://%s", repo_id)
        t0 = time.perf_counter()
        path = download_hf_checkpoint(repo_id)
        elapsed = time.perf_counter() - t0
        logger.info("checkpoint download/cache lookup completed in %.2fs", elapsed)
        return path

    @staticmethod
    def _resolve_device(value: str) -> str:
        raw = str(value).strip().lower()
        if raw in {"", "auto"}:
            return default_runtime_device()
        return str(value)
