"""The real-time conversion pipeline for one streaming session.

Data flow (all per 16 kHz input block of ``block_ms``)::

    WS binary frame ─▶ feed() ─▶ BlockAccumulator ─▶ VAD gate ─┐
                                                               ▸ asyncio.to_thread
                                                        Seed-VC converter
                                                               │
        WS binary frame ◀─ send_queue ◀─ resample+encode ◀──────┘

Design notes
------------
* The engine call runs in a worker thread (``asyncio.to_thread``) because
  it is synchronous torch code; the event loop stays free for other sessions.
* The send queue is bounded. On overflow the *oldest* audio message is
  dropped (never control messages) — a deliberate real-time trade-off that
  protects the pipeline from slow consumers, metered as ``overflows``.
* When the VAD gate classifies a block as silence the engine is skipped and
  digital silence is emitted instead (constant-length output stream).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from rafidain.core.logging import get_logger
from rafidain.domain.entities import Session
from rafidain.domain.ports import ConverterSession
from rafidain.infrastructure.audio.codecs import (
    BlockAccumulator,
    float_to_pcm_bytes,
    pcm_bytes_to_float,
)
from rafidain.infrastructure.audio.resampler import SoxrResampler
from rafidain.infrastructure.audio.vad import VoiceActivityGate
from rafidain.observability.metrics import metrics as prom

logger = get_logger(__name__)

SAMPLE_RATE_IN = 16_000
_WARNING_THROTTLE_S = 2.0


class OutMessage(NamedTuple):
    """Message the WebSocket writer drains: audio bytes or a control dict."""

    kind: str  # "audio" | "json"
    payload: Any


@dataclass(slots=True)
class PipelineDependencies:
    """Everything the pipeline needs, injected (all testable/moackable)."""

    converter: ConverterSession
    engine_sr_out: int
    resampler: SoxrResampler
    vad: VoiceActivityGate


class StreamPipeline:
    def __init__(
        self,
        session: Session,
        deps: PipelineDependencies,
        *,
        block_samples: int,
        send_queue: asyncio.Queue[OutMessage],
        send_queue_limit: int = 64,
    ) -> None:
        assert session.profile is not None
        self._session = session
        self._converter = deps.converter
        self._resampler = deps.resampler
        self._vad = deps.vad
        self._out_sr = session.profile.out_sample_rate
        self._engine_sr_out = deps.engine_sr_out
        self._block_samples = block_samples
        self._block_seconds = block_samples / SAMPLE_RATE_IN
        self._queue = send_queue
        self._queue_limit = send_queue_limit
        self._accumulator = BlockAccumulator(block_samples)
        self._paused = False
        self._last_warning_ts = 0.0
        self._prom = prom()

    # ------------------------------------------------------------------ public

    @property
    def out_sample_rate(self) -> int:
        return self._out_sr

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    async def feed(self, pcm_bytes: bytes) -> None:
        """Accept one WebSocket binary frame of PCM s16le mono 16 kHz."""
        if self._paused:
            return
        m = self._session.metrics
        m.bytes_in += len(pcm_bytes)
        try:
            pcm = pcm_bytes_to_float(pcm_bytes)
        except ValueError as exc:
            await self._warn("bad_pcm", f"invalid PCM payload: {exc}")
            return
        for block in self._accumulator.push(pcm):
            await self._convert(block)

    async def finish(self) -> None:
        """Flush pending audio; called on stream.stop / disconnect."""
        remainder = self._accumulator.flush(pad=True)
        if remainder.size and not self._paused:
            await self._convert(remainder, pad_warned=True)
        tail = await asyncio.to_thread(self._converter.finish)
        await self._emit_audio(tail)

    def close(self) -> None:
        """Release engine resources (synchronous; idempotent)."""
        try:
            self._converter.close()
        except Exception:  # noqa: BLE001 — teardown must not raise
            logger.exception("converter close failed")

    async def snapshot(self) -> dict:
        return self._session.metrics.snapshot()

    # ---------------------------------------------------------------- internals

    async def _convert(self, block: np.ndarray, *, pad_warned: bool = False) -> None:
        del pad_warned
        m = self._session.metrics
        m.blocks_in += 1
        profile_key = self._session.profile.key if self._session.profile else "?"
        self._prom["blocks_processed"].labels(profile=profile_key).inc()

        is_speech, gated = self._vad.process(block)
        if not is_speech:
            m.vad_silence_blocks += 1
            out = np.zeros(
                int(self._block_seconds * self._out_sr), dtype=np.float32
            )
        else:
            t0 = time.perf_counter()
            out = await asyncio.to_thread(self._converter.convert_block, gated)
            dt = time.perf_counter() - t0
            m.observe_block(audio_seconds=self._block_seconds, processing_seconds=dt)
            self._prom["block_latency"].labels(profile=profile_key).observe(dt)
            self._prom["block_rtf"].labels(session_id=self._session.session_id).set(
                m.rtf_ewma
            )
            out = self._resampler.resample(out, self._engine_sr_out, self._out_sr)

        await self._emit_audio(out)

    async def _emit_audio(self, pcm: np.ndarray) -> None:
        if pcm is None or pcm.size == 0:
            return
        data = float_to_pcm_bytes(pcm)
        m = self._session.metrics
        m.blocks_out += 1
        m.bytes_out += len(data)
        msg = OutMessage(kind="audio", payload=data)
        try:
            self._queue.put_nowait(msg)
        except asyncio.QueueFull:
            dropped = self._drop_oldest_audio()
            if dropped:
                m.overflows += dropped
                self._prom["overflows"].inc(dropped)
            # The freshest audio takes priority over the warning message.
            self._queue.put_nowait(msg)
            if dropped:
                await self._warn(
                    "output_backpressure",
                    f"slow consumer: dropped {dropped} buffered audio block(s)",
                )

    def _drop_oldest_audio(self) -> int:
        """Pop the oldest audio-only messages to free queue space.

        Pops until at least one *audio* message is dropped (control messages
        are stashed and re-queued), guaranteeing one free slot afterwards.
        """
        dropped = 0
        stash: list[OutMessage] = []
        while dropped == 0:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item.kind == "audio":
                dropped += 1
            else:
                stash.append(item)
        for item in stash:
            self._queue.put_nowait(item)
        return dropped

    async def _warn(self, code: str, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warning_ts < _WARNING_THROTTLE_S:
            return
        self._last_warning_ts = now
        logger.warning("%s: %s", code, message, extra={"session_id": self._session.session_id})
        await self._send_json({"type": "warning", "code": code, "message": message})

    async def _send_json(self, payload: dict) -> None:
        if self._queue.qsize() < self._queue_limit:
            self._queue.put_nowait(OutMessage(kind="json", payload=payload))


def build_pipeline(
    session: Session,
    deps: PipelineDependencies,
    *,
    block_samples: int,
    send_queue_size: int,
) -> tuple[StreamPipeline, asyncio.Queue[OutMessage]]:
    """Factory keeping construction in one place (used by the WS handler)."""
    queue: asyncio.Queue[OutMessage] = asyncio.Queue(maxsize=send_queue_size)
    pipeline = StreamPipeline(
        session,
        deps,
        block_samples=block_samples,
        send_queue=queue,
        send_queue_limit=send_queue_size,
    )
    pipeline._engine_sr_out = deps.engine_sr_out
    return pipeline, queue
