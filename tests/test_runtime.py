from __future__ import annotations

import threading

import pytest

from irodori_openai_tts import runtime as runtime_module
from irodori_openai_tts.config import Settings
from irodori_openai_tts.runtime import RuntimeLoadTimeoutError, RuntimeManager


def test_runtime_load_timeout_while_another_thread_is_loading(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"test")
    settings = Settings(
        checkpoint=str(checkpoint),
        model_device="cpu",
        codec_device="cpu",
        model_load_timeout=0.05,
        _env_file=None,
    )
    manager = RuntimeManager(settings)
    started = threading.Event()
    release = threading.Event()
    loaded_runtime = object()
    errors: list[BaseException] = []

    def fake_from_key(_key):
        started.set()
        release.wait(timeout=2)
        return loaded_runtime

    monkeypatch.setattr(
        runtime_module.InferenceRuntime,
        "from_key",
        staticmethod(fake_from_key),
    )

    def load_runtime():
        try:
            assert manager.get() is loaded_runtime
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    thread = threading.Thread(target=load_runtime)
    thread.start()
    assert started.wait(timeout=1)

    with pytest.raises(RuntimeLoadTimeoutError):
        manager.get()

    release.set()
    thread.join(timeout=2)
    assert errors == []
    assert manager.is_loaded
    assert not manager.is_loading


def test_runtime_resolves_local_checkpoint_path(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"test")
    manager = RuntimeManager(Settings(checkpoint=str(checkpoint), _env_file=None))

    assert manager._resolve_checkpoint_path() == str(checkpoint)


def test_runtime_rejects_missing_local_checkpoint(tmp_path):
    manager = RuntimeManager(
        Settings(checkpoint=str(tmp_path / "missing.safetensors"), _env_file=None)
    )

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        manager._resolve_checkpoint_path()


def test_runtime_downloads_hf_checkpoint_when_local_checkpoint_is_unset(monkeypatch):
    manager = RuntimeManager(Settings(hf_checkpoint="owner/repo", _env_file=None))

    def fake_download(repo_id):
        assert repo_id == "owner/repo"
        return "/cache/model.safetensors"

    monkeypatch.setattr(runtime_module, "download_hf_checkpoint", fake_download)

    assert manager._resolve_checkpoint_path() == "/cache/model.safetensors"


def test_runtime_switch_waits_for_loading_and_times_out(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"test")
    settings = Settings(
        checkpoint=str(checkpoint),
        model_device="cpu",
        codec_device="cpu",
        model_load_timeout=0.05,
        _env_file=None,
    )
    manager = RuntimeManager(settings)
    started = threading.Event()
    release = threading.Event()

    def fake_from_key(_key):
        started.set()
        release.wait(timeout=2)
        return object()

    monkeypatch.setattr(runtime_module.InferenceRuntime, "from_key", staticmethod(fake_from_key))
    thread = threading.Thread(target=manager.get)
    thread.start()
    assert started.wait(timeout=1)

    # ロード中は設定を書き換えず、ロード待ちと同じタイムアウトで失敗する。
    with pytest.raises(RuntimeLoadTimeoutError):
        manager.switch("owner/other", load=False)
    assert settings.checkpoint == str(checkpoint)

    release.set()
    thread.join(timeout=2)
    assert manager.is_loaded


def test_runtime_switch_unloads_and_rewrites_settings(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"test")
    settings = Settings(checkpoint=str(checkpoint), model_device="cpu", codec_device="cpu", _env_file=None)
    manager = RuntimeManager(settings)
    monkeypatch.setattr(
        runtime_module.InferenceRuntime, "from_key", staticmethod(lambda _key: object())
    )
    manager.get()
    assert manager.is_loaded

    manager.switch("owner/other", load=False)

    assert not manager.is_loaded
    assert manager.checkpoint_path is None
    assert settings.checkpoint is None
    assert settings.hf_checkpoint == "owner/other"
    assert manager.selected_checkpoint == "owner/other"

    manager.switch(str(checkpoint), load=False)
    assert settings.checkpoint == str(checkpoint)


def test_delete_holds_runtime_guard_until_files_are_removed(monkeypatch):
    from irodori_openai_tts.runtime import RuntimeInUseError

    manager = RuntimeManager(Settings(hf_checkpoint="owner/repo", model_load_timeout=0, _env_file=None))
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    def delete():
        entered.set()
        assert release.wait(timeout=2)
        completed.set()
        return True

    worker = threading.Thread(target=lambda: manager.delete_if_unused("owner/repo", delete))
    worker.start()
    try:
        assert entered.wait(timeout=2)
        with pytest.raises(RuntimeLoadTimeoutError):
            manager.get()
        with pytest.raises(RuntimeLoadTimeoutError):
            manager.switch("owner/other", load=False)
    finally:
        release.set()
        worker.join(timeout=2)
    assert completed.is_set()
    manager._runtime = object()
    with pytest.raises(RuntimeInUseError):
        manager.delete_if_unused("owner/repo", lambda: pytest.fail("loaded model deleted"))
