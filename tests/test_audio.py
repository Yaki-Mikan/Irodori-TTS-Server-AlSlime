from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
import soundfile as sf
import torch

from irodori_openai_tts import audio as audio_module
from irodori_openai_tts.audio import decode_audio_bytes, encode_audio, normalize_response_format


def test_decode_audio_bytes_roundtrip():
    buffer = BytesIO()
    sf.write(buffer, torch.zeros(1600).numpy(), 16000, format="WAV")

    wav, sample_rate = decode_audio_bytes(buffer.getvalue(), "ref.wav")

    assert sample_rate == 16000
    assert wav.shape == (1, 1600)
    assert wav.dtype == torch.float32


def test_decode_audio_bytes_rejects_undecodable_data():
    with pytest.raises(ValueError, match="Could not decode audio file"):
        decode_audio_bytes(b"this is not audio", "broken.wav")


def test_normalize_response_format_uses_default_and_lowercases():
    assert normalize_response_format(None, default="wav") == "wav"
    assert normalize_response_format(" FLAC ", default="wav") == "flac"


def test_normalize_response_format_rejects_unknown_format():
    with pytest.raises(ValueError, match="Unsupported response_format"):
        normalize_response_format("xyz", default="wav")


def test_encode_wav_from_1d_audio():
    audio = torch.zeros(100)

    payload = encode_audio(audio, sample_rate=1000, response_format="wav")

    assert payload.startswith(b"RIFF")
    assert b"WAVE" in payload[:16]


def test_encode_mp3_uses_soundfile_when_torchaudio_is_unavailable(monkeypatch):
    torchaudio_calls = 0

    def fail_torchaudio_save(*args, **kwargs):
        nonlocal torchaudio_calls
        torchaudio_calls += 1
        raise RuntimeError("torchaudio encoder unavailable")

    monkeypatch.setattr(audio_module.torchaudio, "save", fail_torchaudio_save)
    audio = torch.zeros(48000)

    payload = encode_audio(audio, sample_rate=48000, response_format="mp3")

    assert torchaudio_calls == 0
    assert payload.startswith(b"ID3") or payload.startswith(b"\xff")


def test_encode_opus_uses_soundfile_when_torchaudio_is_unavailable(monkeypatch):
    torchaudio_calls = 0

    def fail_torchaudio_save(*args, **kwargs):
        nonlocal torchaudio_calls
        torchaudio_calls += 1
        raise RuntimeError("torchaudio encoder unavailable")

    monkeypatch.setattr(audio_module.torchaudio, "save", fail_torchaudio_save)
    audio = torch.zeros(48000)

    payload = encode_audio(audio, sample_rate=48000, response_format="opus")

    assert torchaudio_calls == 0
    assert payload.startswith(b"OggS")


def test_encode_aac_uses_torchaudio_before_ffmpeg(monkeypatch):
    def fake_torchaudio_save(path, *args, **kwargs):
        Path(path).write_bytes(b"torchaudio-aac")

    def fail_ffmpeg_run(*args, **kwargs):
        raise AssertionError("ffmpeg should not be called when torchaudio succeeds")

    monkeypatch.setattr(audio_module.torchaudio, "save", fake_torchaudio_save)
    monkeypatch.setattr(audio_module.subprocess, "run", fail_ffmpeg_run)
    audio = torch.zeros(48000)

    payload = encode_audio(audio, sample_rate=48000, response_format="aac")

    assert payload == b"torchaudio-aac"


def test_encode_aac_falls_back_to_ffmpeg(monkeypatch):
    def fail_torchaudio_save(*args, **kwargs):
        raise RuntimeError("torchaudio encoder unavailable")

    def fake_run(command, *, check, capture_output):
        assert check is True
        assert capture_output is True
        assert "-codec:a" in command
        assert command[command.index("-codec:a") + 1] == "aac"
        assert "-f" in command
        assert command[command.index("-f") + 1] == "adts"
        Path(command[-1]).write_bytes(b"fake-aac")

    monkeypatch.setattr(audio_module.torchaudio, "save", fail_torchaudio_save)
    monkeypatch.setattr(audio_module.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)
    audio = torch.zeros(48000)

    payload = encode_audio(audio, sample_rate=48000, response_format="aac")

    assert payload == b"fake-aac"


def test_encode_pcm_clamps_and_returns_int16_bytes():
    audio = torch.tensor([-2.0, 0.0, 2.0])

    payload = encode_audio(audio, sample_rate=1000, response_format="pcm")

    assert len(payload) == 6


def test_encode_audio_rejects_invalid_shape():
    audio = torch.zeros(1, 1, 10)

    with pytest.raises(ValueError, match="Expected audio shape"):
        encode_audio(audio, sample_rate=1000, response_format="wav")
