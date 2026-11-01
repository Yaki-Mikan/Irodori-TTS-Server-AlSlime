from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from irodori_openai_tts import app as main
from irodori_openai_tts import runtime_api
from irodori_openai_tts.config import Settings


@pytest.fixture
def profile_file(tmp_path, monkeypatch):
    path = tmp_path / "runtime_profile.json"
    monkeypatch.setattr(runtime_api, "PROFILE_FILE", path)
    monkeypatch.setattr(runtime_api, "_active_profile_id", runtime_api.DEFAULT_PROFILE_ID)
    monkeypatch.setattr(runtime_api, "_active_precision", "fp32")
    monkeypatch.setattr(runtime_api, "_available_precisions", ["fp32"])
    return path


def _settings() -> Settings:
    return Settings(model_device="auto", codec_device="auto", _env_file=None)


def test_apply_saved_precision_when_available(profile_file, monkeypatch):
    profile_file.write_text(
        json.dumps({"profile": "low_vram", "model_precision": "bf16"}), encoding="utf-8"
    )
    monkeypatch.setattr(runtime_api, "_detect_available_precisions", lambda _s: ["fp32", "bf16"])
    settings = _settings()

    runtime_api.apply_saved_profile(settings)

    assert settings.model_precision == "bf16"
    assert settings.codec_device == "cpu"
    response = runtime_api.profiles_response()
    assert response["active_precision"] == "bf16"
    assert response["selected_precision"] == "bf16"
    assert [p["id"] for p in response["precisions"]] == ["fp32", "bf16"]
    assert response["restart_required"] is False


def test_apply_saved_bf16_falls_back_without_accelerator(profile_file, monkeypatch):
    profile_file.write_text(json.dumps({"model_precision": "bf16"}), encoding="utf-8")
    monkeypatch.setattr(runtime_api, "_detect_available_precisions", lambda _s: ["fp32"])
    settings = _settings()

    runtime_api.apply_saved_profile(settings)

    assert settings.model_precision == "fp32"
    response = runtime_api.profiles_response()
    assert [p["id"] for p in response["precisions"]] == ["fp32"]
    assert response["selected_precision"] == "fp32"
    assert response["restart_required"] is False


def test_saving_precision_keeps_profile_and_requires_restart(profile_file, monkeypatch):
    monkeypatch.setattr(runtime_api, "_available_precisions", ["fp32", "bf16"])
    runtime_api.save_selected_profile_id("low_vram")

    runtime_api.save_selected_precision("bf16")

    saved = json.loads(profile_file.read_text(encoding="utf-8"))
    assert saved == {"profile": "low_vram", "model_precision": "bf16"}
    assert runtime_api.profiles_response()["restart_required"] is True


def test_save_unavailable_precision_is_rejected(profile_file):
    with pytest.raises(ValueError):
        runtime_api.save_selected_precision("bf16")
    assert not profile_file.exists()


def test_profile_endpoint_accepts_precision_only(profile_file, monkeypatch):
    monkeypatch.setattr(runtime_api, "_available_precisions", ["fp32", "bf16"])
    client = TestClient(main.app)

    response = client.post("/v1/runtime/profile", json={"model_precision": "bf16"})

    assert response.status_code == 200
    body = response.json()
    assert body["selected_precision"] == "bf16"
    assert body["selected_profile"] == runtime_api.DEFAULT_PROFILE_ID
    assert body["restart_required"] is True


def test_profile_endpoint_rejects_unavailable_precision(profile_file):
    client = TestClient(main.app)

    response = client.post("/v1/runtime/profile", json={"model_precision": "bf16"})

    assert response.status_code == 400


def test_saved_hf_checkpoint_is_applied_on_startup(profile_file):
    settings = _settings()
    settings.hf_checkpoint = "Aratako/Irodori-TTS-v4-Large"
    runtime_api.save_last_checkpoint(settings)

    restarted = _settings()
    runtime_api.apply_saved_profile(restarted)

    assert restarted.checkpoint is None
    assert restarted.hf_checkpoint == "Aratako/Irodori-TTS-v4-Large"


def test_saved_local_checkpoint_is_applied_only_while_the_file_exists(profile_file, tmp_path):
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"test")
    settings = _settings()
    settings.checkpoint = str(model)
    runtime_api.save_last_checkpoint(settings)

    restarted = _settings()
    runtime_api.apply_saved_profile(restarted)
    assert restarted.checkpoint == str(model)

    model.unlink()
    missing = _settings()
    default_hf = missing.hf_checkpoint
    runtime_api.apply_saved_profile(missing)
    assert missing.checkpoint is None
    assert missing.hf_checkpoint == default_hf


def test_forget_checkpoint_clears_only_the_remembered_one(profile_file):
    settings = _settings()
    settings.hf_checkpoint = "Aratako/Irodori-TTS-v4-Large"
    runtime_api.save_last_checkpoint(settings)
    runtime_api.save_selected_profile_id("low_vram")

    runtime_api.forget_checkpoint("Aratako/Irodori-TTS-v4-Small")
    assert json.loads(profile_file.read_text(encoding="utf-8"))["checkpoint"] == (
        "Aratako/Irodori-TTS-v4-Large"
    )

    runtime_api.forget_checkpoint("Aratako/Irodori-TTS-v4-Large")
    saved = json.loads(profile_file.read_text(encoding="utf-8"))
    assert "checkpoint" not in saved
    assert saved["profile"] == "low_vram"


class _SwitchingManager:
    """switch で設定だけを書き換える偽マネージャー（ロードは成功扱い）。"""

    def switch(self, checkpoint: str, load: bool) -> None:
        main.settings.checkpoint = None
        main.settings.hf_checkpoint = checkpoint

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def is_loading(self) -> bool:
        return False

    @property
    def selected_checkpoint(self) -> str:
        return str(main.settings.hf_checkpoint)

    @property
    def checkpoint_path(self):
        return None


def test_model_endpoint_remembers_only_loaded_checkpoint(profile_file, monkeypatch):
    monkeypatch.setattr(main.settings, "checkpoint", None)
    monkeypatch.setattr(main.settings, "hf_checkpoint", "Aratako/Irodori-TTS-v4-Small")
    monkeypatch.setattr(main, "runtime_manager", _SwitchingManager())
    client = TestClient(main.app)

    switched = client.post(
        "/v1/runtime/model", json={"checkpoint": "Aratako/Irodori-TTS-v4-Large", "load": False}
    )
    assert switched.status_code == 200
    assert not profile_file.exists()

    loaded = client.post(
        "/v1/runtime/model", json={"checkpoint": "Aratako/Irodori-TTS-v4-Large", "load": True}
    )
    assert loaded.status_code == 200
    saved = json.loads(profile_file.read_text(encoding="utf-8"))
    assert saved["checkpoint"] == "Aratako/Irodori-TTS-v4-Large"
    assert saved["checkpoint_local"] is False
