"""Full-stack integration: REST lifecycle + WebSocket streaming (mock engine)."""

from __future__ import annotations

import json

from tests.conftest import synth_wav_bytes


def test_health_and_ready(client):
    assert client.get("/healthz").json()["status"] == "ok"
    ready = client.get("/readyz").json()
    assert ready["ready"] is True
    assert ready["engine"]["impl"] == "mock"
    assert client.get("/metrics").status_code == 200


def test_voice_crud(client):
    # register
    resp = client.post(
        "/api/v1/voices",
        files={"file": ("a.wav", synth_wav_bytes(), "audio/wav")},
        data={"name": "kareem"},
    )
    assert resp.status_code == 201, resp.text
    voice = resp.json()
    assert voice["name"] == "kareem"
    assert voice["duration_s"] >= 1.0

    # list + get
    listing = client.get("/api/v1/voices").json()["voices"]
    assert any(v["voice_id"] == voice["voice_id"] for v in listing)
    got = client.get(f"/api/v1/voices/{voice['voice_id']}")
    assert got.status_code == 200 and got.json()["voice_id"] == voice["voice_id"]

    # delete
    assert client.delete(f"/api/v1/voices/{voice['voice_id']}").json()["deleted"]
    assert client.get(f"/api/v1/voices/{voice['voice_id']}").status_code == 404


def test_register_rejects_garbage(client):
    resp = client.post(
        "/api/v1/voices",
        files={"file": ("x.wav", b"not audio at all", "audio/wav")},
        data={"name": "bad"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "voice_validation_failed"


def test_session_requires_known_voice(client):
    resp = client.post("/api/v1/sessions", json={"voice_id": "ghost"})
    assert resp.status_code == 404


def test_full_streaming_flow(client, session_info):
    session_id = session_info["session_id"]
    with client.websocket_connect(f"/api/v1/stream/{session_id}") as ws:
        ready = json.loads(ws.receive_text())
        assert ready["type"] == "session.ready"
        assert ready["profile"]["key"] == "balanced"
        assert ready["audio_in"]["sample_rate"] == 16000
        assert ready["audio_out"]["sample_rate"] == 16000

        ws.send_text(json.dumps({"type": "stream.start"}))
        started = ws.receive_text()
        assert json.loads(started)["type"] == "stream.started"

        # stream ~1.2 s of audio in 20 ms frames (44-byte WAV header stripped)
        frame = synth_wav_bytes(seconds=0.02)[44:]
        for _ in range(60):
            ws.send_bytes(frame)
        ws.send_text(json.dumps({"type": "metrics.poll"}))

        audio_bytes = 0
        snapshot = None
        while snapshot is None:
            message = ws.receive()  # raw ASGI message: {'text': ...} or {'bytes': ...}
            if "bytes" in message:
                audio_bytes += len(message["bytes"])
            elif "text" in message:
                payload = json.loads(message["text"])
                assert payload["type"] != "error", payload
                if payload["type"] == "metrics.snapshot":
                    snapshot = payload["metrics"]
        assert snapshot["blocks_in"] >= 1
        assert audio_bytes >= snapshot["blocks_in"] * 320 * 2

        ws.send_text(json.dumps({"type": "stream.stop"}))
        stopped = False
        while not stopped:
            message = ws.receive()
            if "text" in message and json.loads(message["text"])["type"] == "stream.stopped":
                stopped = True

    # session metrics survive the connection for inspection
    detail = client.get(f"/api/v1/sessions/{session_id}").json()
    assert detail["state"] == "closed"
    assert detail["metrics"]["blocks_in"] >= 1


def test_unknown_session_websocket(client):
    import pytest
    from starlette.websockets import WebSocketDisconnect

    with (
        pytest.raises(WebSocketDisconnect) as excinfo,
        client.websocket_connect("/api/v1/stream/does-not-exist") as ws,
    ):
        ws.receive_text()  # server rejects with close code 4404
    assert excinfo.value.code == 4404


def test_audio_before_start_rejected(client, session_info):
    session_id = session_info["session_id"]
    with client.websocket_connect(f"/api/v1/stream/{session_id}") as ws:
        ws.receive_text()  # session.ready
        ws.send_bytes(b"\x00" * 640)  # audio before stream.start
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error"
        assert msg["code"] == "not_streaming"


def test_api_key_auth(tmp_path):
    from fastapi.testclient import TestClient

    from tests.conftest import make_settings

    settings = make_settings(tmp_path, api_keys=["sekrit"])
    from rafidain.main import create_app

    app = create_app(settings)
    with TestClient(app) as c:
        assert c.get("/api/v1/voices").status_code == 401
        assert c.get("/api/v1/voices", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/api/v1/voices", headers={"X-API-Key": "sekrit"}).status_code == 200
        # health/metrics stay public
        assert c.get("/healthz").status_code == 200
