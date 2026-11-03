"""StreamPipeline behaviour with the mock engine (no ML dependencies)."""

from __future__ import annotations

import numpy as np

from rafidain.application.stream_pipeline import PipelineDependencies, build_pipeline
from rafidain.domain.entities import Session
from rafidain.domain.value_objects import DEFAULT_PROFILES
from rafidain.infrastructure.audio.resampler import SoxrResampler
from rafidain.infrastructure.audio.vad import VoiceActivityGate
from rafidain.infrastructure.mock_engine import MockConversionEngine

SAMPLE_RATE = 16_000


def make_pipeline(profile_key: str = "balanced"):
    profile = DEFAULT_PROFILES[profile_key]
    engine = MockConversionEngine()
    session = Session(voice_id="voice", profile=profile)
    block_samples = engine.preferred_block_samples(profile)
    deps = PipelineDependencies(
        converter=None,  # replaced below after new_session
        engine_sr_out=engine.sr_out,
        resampler=SoxrResampler(),
        vad=VoiceActivityGate(mode="off"),
    )
    deps.converter = engine.new_session(np.zeros(16000, dtype=np.float32), None, profile)
    pipeline, queue = build_pipeline(
        session, deps, block_samples=block_samples, send_queue_size=64
    )
    return pipeline, queue, session, profile


def tone(seconds: float, f0: float = 220.0) -> bytes:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    x = (np.sin(2 * np.pi * f0 * t) * 0.5).astype(np.float32)
    return (x * 32768).astype("<i2").tobytes()


def drain_audio(queue) -> bytes:
    data = bytearray()
    while not queue.empty():
        item = queue.get_nowait()
        if item.kind == "audio":
            data.extend(item.payload)
    return bytes(data)


async def test_passthrough_emits_audio_per_block():
    pipeline, queue, session, profile = make_pipeline()
    frame = tone(0.02)  # 20 ms
    n_frames = 100  # 2 seconds
    for _ in range(n_frames):
        await pipeline.feed(frame)

    audio = drain_audio(queue)
    block_bytes = int(profile.block_ms / 1000 * SAMPLE_RATE) * 2
    assert len(audio) > 0
    assert len(audio) % block_bytes == 0
    assert session.metrics.blocks_in == 11  # 2 s / 180 ms = 11 full blocks
    assert session.metrics.blocks_out == session.metrics.blocks_in
    assert session.metrics.bytes_in == n_frames * len(frame)


async def test_pause_discards_input():
    pipeline, queue, session, _ = make_pipeline()
    pipeline.pause()
    for _ in range(50):
        await pipeline.feed(tone(0.02))
    assert drain_audio(queue) == b""
    assert session.metrics.blocks_in == 0
    pipeline.resume()
    for _ in range(50):
        await pipeline.feed(tone(0.02))
    assert len(drain_audio(queue)) > 0


async def test_finish_flushes_remainder():
    pipeline, queue, session, _ = make_pipeline()
    for _ in range(53):  # 1.06 s -> 5 blocks + remainder
        await pipeline.feed(tone(0.02))
    _ = drain_audio(queue)
    await pipeline.finish()
    tail = drain_audio(queue)
    assert len(tail) > 0
    pipeline.close()


async def test_rtf_and_latency_metrics_recorded():
    pipeline, queue, session, _ = make_pipeline()
    for _ in range(100):
        await pipeline.feed(tone(0.02))
    _ = drain_audio(queue)
    snapshot = await pipeline.snapshot()
    assert snapshot["blocks_in"] == 11
    assert snapshot["chunk_p50_ms"] >= 0.0
    assert snapshot["rtf_ewma"] < 1.0  # mock engine is instant


async def test_backpressure_drops_oldest_audio():
    pipeline, queue, session, profile = make_pipeline()
    # 16 s of audio in one burst -> 88 blocks > queue limit (64): overflow path.
    for _ in range(800):
        await pipeline.feed(tone(0.02))
    dropped = session.metrics.overflows
    assert dropped > 0, "expected overflow counter to increment under burst load"
    # Control messages must never be dropped by the overflow path.
    assert session.metrics.blocks_out >= 1
    _ = profile
