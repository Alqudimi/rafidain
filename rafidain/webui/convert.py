"""Conversion helpers shared by the Gradio panel and the web services.

Everything here is engine-agnostic: it talks to the ``ConversionEngine``
port only, so the mock engine exercises the exact same code path as the
real Seed-VC engine (no GPU needed to develop / test the UI).
"""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rafidain.core.config import Settings
from rafidain.core.logging import get_logger
from rafidain.domain.value_objects import SAMPLE_RATE_IN, LatencyProfile
from rafidain.infrastructure.audio.decode import decode_to_pcm, encode_wav_pcm16

logger = get_logger(__name__)


def to_mono_float(pcm: np.ndarray) -> np.ndarray:
    """Flatten whatever the UI handed us into float32 mono."""
    data = np.asarray(pcm, dtype=np.float32)
    if data.ndim > 1:  # (n, channels) -> average channels
        data = data.mean(axis=tuple(range(1, data.ndim)))
    return np.nan_to_num(data.reshape(-1), nan=0.0)


def resample(pcm: np.ndarray, sr_from: int, sr_to: int = SAMPLE_RATE_IN) -> np.ndarray:
    """High-quality resample via soxr (same library the pipeline uses)."""
    if sr_from == sr_to:
        return to_mono_float(pcm)
    import soxr

    return soxr.resample(to_mono_float(pcm), sr_from, sr_to).astype(np.float32)


@dataclass(slots=True)
class MicStreamState:
    """Per-browser-stream converter state (kept in a gradio.State)."""

    voice_id: str | None = None
    profile_key: str | None = None
    session: Any | None = None
    blocks_done: int = 0
    samples_in: int = 0
    samples_out: int = 0
    errors: list[str] = field(default_factory=list)

    def matches(self, voice_id: str | None, profile_key: str) -> bool:
        return (
            self.session is not None
            and self.voice_id == (voice_id or "")
            and self.profile_key == profile_key
        )


async def open_stream(components, voice_id: str | None, profile_key: str) -> Any:
    """Condition a converter on the target voice (engine.new_session)."""
    if not voice_id:
        raise ValueError(
            "لا يوجد صوت هدف — سجّل صوتاً من تبويب «الأصوات» ثم اختره"
        )
    profile: LatencyProfile = components.settings.profile(profile_key)
    ref_16k, ref_model_sr = await components.voice_service.load_reference(voice_id)
    return await asyncio.to_thread(
        components.engine.new_session, ref_16k, ref_model_sr, profile
    )


async def stream_mic_chunk(
    components,
    state: MicStreamState | None,
    chunk: np.ndarray,
    sample_rate: int,
    voice_id: str | None,
    profile_key: str,
) -> tuple[MicStreamState | None, np.ndarray]:
    """Feed one mic chunk through the converter; returns (state, out_pcm).

    The returned PCM is at ``components.engine.sr_out`` and may be empty
    while the engine buffers look-ahead context.
    """
    if chunk is None or getattr(chunk, "size", 0) == 0:
        return state, np.zeros(0, dtype=np.float32)

    if state is None:
        state = MicStreamState(voice_id=voice_id, profile_key=profile_key)
    elif not state.matches(voice_id, profile_key):
        _close_session(state)
        state = MicStreamState(voice_id=voice_id, profile_key=profile_key)

    if state.session is None:
        try:
            state.session = await open_stream(components, voice_id, profile_key)
        except Exception as exc:  # noqa: BLE001 — surface in UI, keep streaming
            logger.warning("webui: converter open failed: %s", exc)
            state.errors.append(str(exc))
            return state, np.zeros(0, dtype=np.float32)

    pcm16 = resample(chunk, sample_rate)
    state.samples_in += pcm16.size
    out = await asyncio.to_thread(state.session.convert_block, pcm16)
    out = np.asarray(out, dtype=np.float32).reshape(-1)
    state.blocks_done += 1
    state.samples_out += out.size
    return state, out


async def close_mic_stream(state: MicStreamState | None) -> np.ndarray:
    """Flush + close a mic stream; returns the final tail PCM."""
    if state is None or state.session is None:
        return np.zeros(0, dtype=np.float32)
    try:
        tail = await asyncio.to_thread(state.session.finish)
    except Exception:  # noqa: BLE001
        tail = np.zeros(0, dtype=np.float32)
    _close_session(state)
    return np.asarray(tail, dtype=np.float32).reshape(-1)


def _close_session(state: MicStreamState) -> None:
    try:
        if state.session is not None:
            state.session.close()
    except Exception:  # noqa: BLE001
        pass


async def convert_file(
    components,
    audio_bytes: bytes,
    voice_id: str | None,
    profile_key: str,
) -> tuple[Path, dict]:
    """Convert a whole audio file; returns (wav_path, stats).

    Uses the same block/crossfade machinery as the live path (chunked via
    ``convert_block`` + ``finish``), so quality matches realtime behaviour.
    """
    pcm16 = decode_to_pcm(audio_bytes, sample_rate=SAMPLE_RATE_IN)
    session = await open_stream(components, voice_id, profile_key)
    profile: LatencyProfile = components.settings.profile(profile_key)
    sr_out = components.engine.sr_out

    outs: list[np.ndarray] = []
    # Feed in profile-sized blocks so crossfades happen exactly like live.
    block = max(components.engine.preferred_block_samples(profile), SAMPLE_RATE_IN // 10)
    try:
        for start in range(0, pcm16.size, block):
            out = await asyncio.to_thread(
                session.convert_block, pcm16[start : start + block]
            )
            if getattr(out, "size", 0):
                outs.append(np.asarray(out, dtype=np.float32).reshape(-1))
        tail = await asyncio.to_thread(session.finish)
        if getattr(tail, "size", 0):
            outs.append(np.asarray(tail, dtype=np.float32).reshape(-1))
    finally:
        close = getattr(session, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):  # noqa: S110
                close()

    converted = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
    out_dir = Path(tempfile.mkdtemp(prefix="rafidain_convert_"))
    out_path = out_dir / "converted.wav"
    encode_wav_pcm16(converted, sr_out, out_path)
    stats = {
        "in_seconds": round(pcm16.size / SAMPLE_RATE_IN, 2),
        "out_seconds": round(converted.size / sr_out, 2),
        "sample_rate_out": sr_out,
        "profile": profile_key,
    }
    return out_path, stats


def settings_summary(settings: Settings) -> dict:
    """Small snapshot for the UI 'system' tab."""
    return {
        "engine_impl": settings.engine.impl,
        "engine_device": settings.engine.device,
        "profile": settings.default_profile,
        "auth_enabled": settings.auth_enabled,
        "data_dir": str(settings.data_dir),
        "checkpoints_dir": str(settings.engine.cache_dir),
    }
