"""Web UI plumbing tests: import safety, streaming helpers, live page route.

Everything runs against the mock engine — no GPU, no gradio dependency
needed for the core service (the panel import is exercised only when the
`webui` extra is installed, e.g. in the full sandbox/CI environment).
"""

from __future__ import annotations

import numpy as np
import pytest

from rafidain.api.deps import build_components
from rafidain.core.config import EngineSettings, Settings
from rafidain.webui import available
from rafidain.webui.convert import (
    MicStreamState,
    close_mic_stream,
    convert_file,
    open_stream,
    resample,
    stream_mic_chunk,
    to_mono_float,
)


@pytest.fixture()
def components(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = Settings(engine=EngineSettings(impl="mock"))
    return build_components(settings)


def _voice(components) -> str:
    import io
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        for i in range(16_000 * 2):  # 2 s of tone: long enough as a reference
            w.writeframes(struct.pack("<h", int(8000 * np.sin(2 * np.pi * 220 * i / 16000))))
    voice = __import__("asyncio").run(
        components.voice_service.register(name="t", audio=buf.getvalue())
    )
    return voice.voice_id


class TestAudioHelpers:
    def test_to_mono_float_flattens(self):
        stereo = np.ones((100, 2), dtype=np.float32)
        mono = to_mono_float(stereo)
        assert mono.shape == (100,) and mono[0] == 1.0

    def test_resample_identity(self):
        pcm = np.linspace(-1, 1, 1000, dtype=np.float32)
        assert resample(pcm, 16_000, 16_000).shape == (1000,)

    def test_resample_48k_to_16k(self):
        pcm = np.zeros(4800, dtype=np.float32)
        out = resample(pcm, 48_000, 16_000)
        assert abs(out.size - 1600) <= 8  # soxr length tolerance


class TestMicStreaming:
    def test_requires_voice(self, components):
        import asyncio

        with pytest.raises(ValueError):
            asyncio.run(open_stream(components, None, "balanced"))

    def test_stream_and_close_roundtrip(self, components):
        import asyncio

        voice = _voice(components)

        async def scenario():
            state = MicStreamState()
            chunk = np.zeros(4800, dtype=np.float32)
            state, out1 = await stream_mic_chunk(
                components, state, chunk, 48_000, voice, "balanced"
            )
            state, out2 = await stream_mic_chunk(
                components, state, chunk, 48_000, voice, "balanced"
            )
            tail = await close_mic_stream(state)
            return out1, out2, tail

        out1, out2, tail = asyncio.run(scenario())
        # mock engine passes through: 4800@48k == 1600@16k per chunk
        assert out1.size == 1600 and out2.size == 1600
        assert tail.size == 0

    def test_empty_chunk_is_noop(self, components):
        import asyncio

        async def scenario():
            return await stream_mic_chunk(
                components, None, np.zeros(0), 48_000, "v", "balanced"
            )

        state, out = asyncio.run(scenario())
        assert state is None and out.size == 0


class TestConvertFile:
    def test_mock_passthrough(self, components, tmp_path):
        import asyncio
        import io
        import struct
        import wave

        voice = _voice(components)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16_000)
            for i in range(16_000):  # 1 s
                w.writeframes(struct.pack("<h", int(6000 * np.sin(2 * np.pi * 440 * i / 16000))))
        out_path, stats = asyncio.run(
            convert_file(components, buf.getvalue(), voice, "balanced")
        )
        assert out_path.exists() and out_path.stat().st_size > 44
        assert stats["in_seconds"] == 1.0
        assert stats["out_seconds"] == 1.0  # mock keeps duration
        assert stats["sample_rate_out"] == 16_000


class TestWebUIAvailability:
    def test_available_is_bool(self):
        assert isinstance(available(), bool)

    def test_webui_disabled_by_setting(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        settings = Settings(webui=False)
        assert settings.webui is False
        assert Settings(webui=True).webui is True

    def test_live_page_registered(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from rafidain.webui.live import register_live_page

        app = FastAPI()
        register_live_page(app)
        client = TestClient(app)
        root = client.get("/")
        live = client.get("/live")
        assert root.status_code == 200 and "rtl" in root.text
        assert live.status_code == 200 and "البث الحي" in live.text
        assert "WebSocket" in live.text  # the page speaks the native protocol
