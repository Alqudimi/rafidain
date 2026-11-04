"""Integration tests for the file-based conversion endpoint + web UI."""

from __future__ import annotations

import io
import wave

from tests.conftest import synth_wav_bytes


def _parse_wav(data: bytes) -> tuple[int, int]:
    """Return (sample_rate, num_frames) of a mono 16-bit WAV."""
    with wave.open(io.BytesIO(data), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        return wav.getframerate(), wav.getnframes()


def test_convert_with_registered_voice(client, voice_id):
    resp = client.post(
        "/api/v1/convert",
        files={"source": ("me.wav", synth_wav_bytes(seconds=1.5), "audio/wav")},
        data={"voice_id": voice_id, "latency_profile": "balanced"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "audio/wav"
    rate, frames = _parse_wav(resp.content)
    assert rate == 16_000
    assert frames > 16_000  # ~1.5 s of audio came back
    assert resp.headers["X-Conversion-Profile"] == "balanced"
    assert float(resp.headers["X-Conversion-RTF"]) >= 0.0


def test_convert_with_inline_reference(client):
    resp = client.post(
        "/api/v1/convert",
        files={
            "source": ("me.wav", synth_wav_bytes(seconds=1.0), "audio/wav"),
            "reference": ("target.wav", synth_wav_bytes(seconds=2.0, f0=220.0), "audio/wav"),
        },
        data={"latency_profile": "ultra_low"},
    )
    assert resp.status_code == 200, resp.text
    rate, frames = _parse_wav(resp.content)
    assert rate == 16_000
    assert frames >= 16_000  # passthrough mock: exactly 1.0 s of audio


def test_convert_rejects_both_and_neither(client, voice_id):
    # both target sources at once
    resp = client.post(
        "/api/v1/convert",
        files={
            "source": ("me.wav", synth_wav_bytes(), "audio/wav"),
            "reference": ("target.wav", synth_wav_bytes(), "audio/wav"),
        },
        data={"voice_id": voice_id},
    )
    assert resp.status_code == 422
    # neither
    resp = client.post(
        "/api/v1/convert",
        files={"source": ("me.wav", synth_wav_bytes(), "audio/wav")},
    )
    assert resp.status_code == 422


def test_convert_rejects_garbage_audio(client, voice_id):
    resp = client.post(
        "/api/v1/convert",
        files={"source": ("junk.bin", b"not audio at all", "application/octet-stream")},
        data={"voice_id": voice_id},
    )
    assert resp.status_code == 422


def test_convert_requires_api_key(tmp_path):
    from rafidain.core.config import Settings
    from rafidain.main import create_app

    settings = Settings(
        _env_file=None,
        engine={"impl": "mock", "eager_load": True, "warmup": False},
        data_dir=tmp_path / "data",
        api_keys=["secret123"],
    )
    from fastapi.testclient import TestClient

    with TestClient(create_app(settings)) as c:
        resp = c.post(
            "/api/v1/convert",
            files={"source": ("me.wav", synth_wav_bytes(), "audio/wav")},
            data={"voice_id": "whatever"},
        )
        assert resp.status_code == 401
        resp = c.post(
            "/api/v1/convert",
            files={"source": ("me.wav", synth_wav_bytes(), "audio/wav")},
            data={"voice_id": "whatever"},
            headers={"X-API-Key": "secret123"},
        )
        # unknown voice id → 404 (auth passed)
        assert resp.status_code == 404


def test_web_ui_is_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "رافدَين" in resp.text
    for asset in ("/style.css", "/app.js", "/pcm-worklet.js"):
        a = client.get(asset)
        assert a.status_code == 200, asset
        assert len(a.content) > 500


def test_api_docs_still_win_over_static_root(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/api/v1/voices").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/does-not-exist.css").status_code == 404
