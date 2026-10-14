from __future__ import annotations

import os
import threading
from pathlib import Path

import huggingface_hub
import pytest

from irodori_openai_tts import model_store
from irodori_openai_tts.config import Settings
from irodori_openai_tts.model_store import (
    DownloadBusyError,
    ModelDownloader,
    ProgressAggregator,
    checkpoint_source,
    make_progress_tqdm,
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(models_dir=tmp_path / "models", _env_file=None)


def test_apply_models_dir_points_hub_cache_at_models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", "placeholder")
    settings = _settings(tmp_path)

    resolved = model_store.apply_models_dir(settings)

    assert resolved == (tmp_path / "models").resolve()
    assert os.environ["HF_HUB_CACHE"] == str(resolved)
    # ディレクトリの作成は huggingface_hub に任せる。
    assert not resolved.exists()


def test_checkpoint_source_handles_subfolder():
    plain = checkpoint_source("owner/repo")
    assert plain.repo_id == "owner/repo"
    assert plain.relative_path == Path("model.safetensors")
    assert plain.allow_patterns == ["model.safetensors", "tokenizer/*"]

    sub = checkpoint_source("owner/repo/int8")
    assert sub.repo_id == "owner/repo"
    assert sub.relative_path == Path("int8") / "model.safetensors"
    assert sub.allow_patterns == ["int8/model.safetensors", "int8/tokenizer/*", "tokenizer/*"]


def test_is_checkpoint_downloaded_requires_local_snapshot(tmp_path, monkeypatch):
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    calls: list[dict] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    assert model_store.is_checkpoint_downloaded("owner/repo") is False
    (snapshot / "model.safetensors").write_bytes(b"x")
    assert model_store.is_checkpoint_downloaded("owner/repo") is True
    assert calls[-1]["local_files_only"] is True
    assert calls[-1]["repo_id"] == "owner/repo"

    def failing_snapshot_download(**kwargs):
        raise OSError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", failing_snapshot_download)
    assert model_store.is_checkpoint_downloaded("owner/repo") is False


def test_is_codec_downloaded(tmp_path, monkeypatch):
    weights = tmp_path / "weights.pth"
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(weights))
    assert model_store.is_codec_downloaded("owner/codec") is False
    weights.write_bytes(b"w")
    assert model_store.is_codec_downloaded("owner/codec") is True


def test_progress_tqdm_counts_only_written_bytes():
    aggregator = ProgressAggregator()
    cls = make_progress_tqdm(aggregator)

    reconstruct = cls(
        disable=True, name="huggingface_hub.snapshot_download", desc="Reconstructing", unit="B",
        unit_scale=True, total=0, initial=0,
    )
    transfer = cls(
        disable=True, name="huggingface_hub.snapshot_download.transfer",
        desc="Downloading bytes", unit="B", unit_scale=True, total=0, initial=0,
    )
    files = cls(disable=True, name="huggingface_hub.snapshot_download", desc="Fetching 2 files", total=2)

    # ファイルの大きさが分かった時点で total は属性代入される。
    reconstruct.total = 100
    transfer.total = 100
    reconstruct.update(30)
    transfer.update(80)
    files.update(1)

    assert aggregator.snapshot() == (30, 100)

    resumed = cls(disable=True, name="huggingface_hub.http_get", desc="weights.pth", unit="B", total=50, initial=20)
    resumed.update(5)
    assert aggregator.snapshot() == (55, 150)


class _DownloadHarness:
    def __init__(self, monkeypatch, *, checkpoint_downloaded=False, codec_downloaded=False):
        self.checkpoint_downloaded = checkpoint_downloaded
        self.codec_downloaded = codec_downloaded
        self.release = threading.Event()
        self.entered = threading.Event()
        self.snapshot_calls: list[dict] = []
        self.codec_calls: list[dict] = []
        self.fail_with: Exception | None = None
        monkeypatch.setattr(
            model_store, "is_checkpoint_downloaded", lambda checkpoint: self.checkpoint_downloaded
        )
        monkeypatch.setattr(model_store, "is_codec_downloaded", lambda repo: self.codec_downloaded)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", self._hf_hub_download)
        monkeypatch.setattr(huggingface_hub, "snapshot_download", self._snapshot_download)

    def _hf_hub_download(self, **kwargs):
        self.codec_calls.append(kwargs)
        bar = kwargs["tqdm_class"](disable=True, unit="B", total=10, initial=0, desc="weights.pth")
        bar.update(10)
        self.codec_downloaded = True
        return "/cache/weights.pth"

    def _snapshot_download(self, **kwargs):
        self.snapshot_calls.append(kwargs)
        self.entered.set()
        self.release.wait(timeout=2)
        if self.fail_with is not None:
            raise self.fail_with
        bar = kwargs["tqdm_class"](disable=True, unit="B", total=0, initial=0, desc="Reconstructing")
        bar.total = 90
        bar.update(90)
        self.checkpoint_downloaded = True
        return "/cache/snap"


def test_model_downloader_runs_codec_then_model(tmp_path, monkeypatch):
    harness = _DownloadHarness(monkeypatch)
    downloader = ModelDownloader(_settings(tmp_path))

    assert downloader.start("Aratako/Irodori-TTS-v4-Small") == "started"
    assert harness.entered.wait(timeout=2)
    mid = downloader.state()
    assert mid["active"] is True
    assert mid["phase"] == "model"
    assert mid["checkpoint"] == "Aratako/Irodori-TTS-v4-Small"
    assert downloader.is_downloading("Aratako/Irodori-TTS-v4-Small")
    assert not downloader.is_downloading("Aratako/Irodori-TTS-v4-Large")

    with pytest.raises(DownloadBusyError):
        downloader.start("Aratako/Irodori-TTS-v4-Large")

    harness.release.set()
    downloader._thread.join(timeout=2)
    done = downloader.state()
    assert done["active"] is False
    assert done["phase"] == "done"
    assert done["percent"] == 100.0
    assert done["downloaded_bytes"] == 100
    assert done["total_bytes"] == 100
    assert harness.codec_calls[0]["repo_id"] == "Aratako/Semantic-DACVAE-Japanese-32dim"
    assert harness.snapshot_calls[0]["repo_id"] == "Aratako/Irodori-TTS-v4-Small"
    assert harness.snapshot_calls[0]["allow_patterns"] == ["model.safetensors", "tokenizer/*"]


def test_model_downloader_reports_already_downloaded_and_unknown(tmp_path, monkeypatch):
    _DownloadHarness(monkeypatch, checkpoint_downloaded=True, codec_downloaded=True)
    downloader = ModelDownloader(_settings(tmp_path))

    assert downloader.start("Aratako/Irodori-TTS-v4-Small") == "already_downloaded"
    with pytest.raises(ValueError):
        downloader.start("owner/unknown")


def test_model_downloader_records_error(tmp_path, monkeypatch):
    harness = _DownloadHarness(monkeypatch, codec_downloaded=True)
    harness.fail_with = RuntimeError("network down")
    harness.release.set()
    downloader = ModelDownloader(_settings(tmp_path))

    assert downloader.start("Aratako/Irodori-TTS-v4-Large") == "started"
    downloader._thread.join(timeout=2)
    state = downloader.state()
    assert state["active"] is False
    assert state["phase"] == "error"
    assert "network down" in state["error"]
    assert harness.codec_calls == []


def test_downloader_setup_failure_clears_active_and_allows_retry(tmp_path, monkeypatch):
    harness = _DownloadHarness(monkeypatch)
    harness.release.set()
    downloader = ModelDownloader(_settings(tmp_path))
    original = model_store.make_progress_tqdm

    def fail_setup(aggregator):
        raise RuntimeError("progress setup failed")

    monkeypatch.setattr(model_store, "make_progress_tqdm", fail_setup)
    downloader.start("Aratako/Irodori-TTS-v4-Small")
    downloader._thread.join(timeout=2)
    assert downloader.state()["active"] is False
    assert downloader.state()["error"] == "progress setup failed"
    monkeypatch.setattr(model_store, "make_progress_tqdm", original)
    assert downloader.start("Aratako/Irodori-TTS-v4-Small") == "started"
    downloader._thread.join(timeout=2)
    assert downloader.state()["phase"] == "done"


def test_downloader_thread_start_failure_clears_active(tmp_path, monkeypatch):
    _DownloadHarness(monkeypatch)
    downloader = ModelDownloader(_settings(tmp_path))

    def fail_start(thread):
        raise RuntimeError("cannot start thread")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="cannot start thread"):
        downloader.start("Aratako/Irodori-TTS-v4-Small")
    assert downloader.state()["active"] is False
    assert downloader.state()["phase"] == "error"


def test_delete_rejects_active_download(tmp_path, monkeypatch):
    harness = _DownloadHarness(monkeypatch)
    downloader = ModelDownloader(_settings(tmp_path))
    checkpoint = "Aratako/Irodori-TTS-v4-Small"
    monkeypatch.setattr(model_store, "delete_checkpoint", lambda *args: pytest.fail("active model deleted"))
    downloader.start(checkpoint)
    try:
        assert harness.entered.wait(timeout=2)
        with pytest.raises(DownloadBusyError):
            downloader.delete(checkpoint)
    finally:
        harness.release.set()
        downloader._thread.join(timeout=2)


def test_download_checks_cache_after_concurrent_delete(tmp_path, monkeypatch):
    harness = _DownloadHarness(monkeypatch, checkpoint_downloaded=True, codec_downloaded=True)
    downloader = ModelDownloader(_settings(tmp_path))
    checkpoint = "Aratako/Irodori-TTS-v4-Small"
    deleting = threading.Event()
    release = threading.Event()
    starting = threading.Event()
    results = []

    def delete(*args):
        deleting.set()
        assert release.wait(timeout=2)
        harness.checkpoint_downloaded = False
        return True

    def start():
        starting.set()
        results.append(downloader.start(checkpoint))

    monkeypatch.setattr(model_store, "delete_checkpoint", delete)
    deletion = threading.Thread(target=lambda: downloader.delete(checkpoint))
    download = threading.Thread(target=start)
    deletion.start()
    try:
        assert deleting.wait(timeout=2)
        download.start()
        assert starting.wait(timeout=2)
    finally:
        release.set()
        deletion.join(timeout=2)
        download.join(timeout=2)
        harness.release.set()
        if downloader._thread is not None:
            downloader._thread.join(timeout=2)
    assert results == ["started"]


class _FakeRevision:
    def __init__(self, commit_hash: str) -> None:
        self.commit_hash = commit_hash


class _FakeRepo:
    def __init__(self, repo_id: str, repo_type: str = "model") -> None:
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.revisions = frozenset({_FakeRevision("aaa"), _FakeRevision("bbb")})


class _FakeStrategy:
    def __init__(self, executed: list[tuple[str, ...]], hashes: tuple[str, ...]) -> None:
        self.executed = executed
        self.hashes = hashes

    def execute(self) -> None:
        self.executed.append(tuple(sorted(self.hashes)))


class _FakeCacheInfo:
    def __init__(self, repos, executed) -> None:
        self.repos = frozenset(repos)
        self.executed = executed

    def delete_revisions(self, *hashes):
        return _FakeStrategy(self.executed, hashes)


def test_delete_checkpoint_removes_all_revisions_of_repo(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    executed: list[tuple[str, ...]] = []
    scanned: list[Path] = []

    def fake_scan(cache_dir):
        scanned.append(Path(cache_dir))
        return _FakeCacheInfo(
            [_FakeRepo("Aratako/Irodori-TTS-v4-Small"), _FakeRepo("Aratako/Semantic-DACVAE-Japanese-32dim")],
            executed,
        )

    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", fake_scan)

    # 保存先が無ければ何もしない。
    assert model_store.delete_checkpoint(settings, "Aratako/Irodori-TTS-v4-Small") is False
    assert scanned == []

    (tmp_path / "models").mkdir()
    assert model_store.delete_checkpoint(settings, "Aratako/Irodori-TTS-v4-Small") is True
    assert executed == [("aaa", "bbb")]
    assert scanned == [(tmp_path / "models").resolve()]

    assert model_store.delete_checkpoint(settings, "Aratako/Irodori-TTS-v4-Large") is False
