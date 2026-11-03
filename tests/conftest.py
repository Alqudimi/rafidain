"""Shared fixtures: a fully wired app with the mock engine and tmp storage."""

from __future__ import annotations

import numpy as np
import pytest

from rafidain.core.config import Settings
from rafidain.main import create_app


def make_settings(tmp_path, **overrides) -> Settings:
    engine = {"impl": "mock", "eager_load": True, "warmup": False}
    engine.update(overrides.pop("engine", {}))
    kwargs = {
        "_env_file": None,
        "engine": engine,
        "data_dir": tmp_path / "data",
        "api_keys": [],
        "session_ttl_s": 60.0,
        "session_cleanup_interval_s": 3600.0,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)


@pytest.fixture()
def app(tmp_path):
    return create_app(make_settings(tmp_path))


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


def synth_wav_bytes(
    seconds: float = 2.0, sr: int = 16_000, f0: float = 150.0
) -> bytes:
    """A small in-memory WAV with a tone — valid 'reference voice' for tests."""
    import io
    import wave

    t = np.arange(int(seconds * sr)) / sr
    sweep = f0 + 30 * np.sin(2 * np.pi * 2.0 * t)
    x = np.sin(2 * np.pi * np.cumsum(sweep) / sr).astype(np.float32) * 0.6
    pcm = (x * 32768).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(pcm)
    return buf.getvalue()


@pytest.fixture()
def voice_id(client) -> str:
    resp = client.post(
        "/api/v1/voices",
        files={"file": ("ref.wav", synth_wav_bytes(), "audio/wav")},
        data={"name": "test-voice"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["voice_id"]


@pytest.fixture()
def session_info(client, voice_id) -> dict:
    resp = client.post(
        "/api/v1/sessions", json={"voice_id": voice_id, "latency_profile": "balanced"}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()
