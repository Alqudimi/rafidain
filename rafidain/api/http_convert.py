"""File-based audio-to-audio conversion endpoint.

The complement of the WebSocket live path: the client uploads a source
recording (any format ffmpeg understands) plus a target voice (a registered
``voice_id`` or an inline reference file) and receives a converted WAV.
Internally it drives the exact same engine contract the streaming pipeline
uses (``new_session`` → ``convert_block`` → ``finish``), block by block, so
both paths stay behaviourally identical.
"""

from __future__ import annotations

import asyncio
import io
import time
import wave
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import Response

from rafidain.api.deps import AppComponents, get_components, require_api_key
from rafidain.core.errors import ReferenceAudioError, VoiceValidationError
from rafidain.core.logging import get_logger
from rafidain.infrastructure.audio.decode import decode_to_pcm

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/convert", tags=["conversion"], dependencies=[Depends(require_api_key)]
)

#: Guard rails for uploaded source audio (converted block-by-block in RAM).
MIN_SOURCE_SECONDS = 0.25
MAX_SOURCE_SECONDS = 300.0
#: Model rate used for the reference conditioning path (Seed-VC contract).
MODEL_SR = 22_050


def _wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Serialise mono float PCM to a 16-bit WAV in memory."""
    clipped = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0 - 1e-6)
    raw = (clipped * 32768.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(raw)
    return buf.getvalue()


def _run_conversion(
    *,
    engine: Any,
    resampler: Any,
    source_16k: np.ndarray,
    ref_16k: np.ndarray,
    ref_model_sr: np.ndarray,
    profile: Any,
) -> tuple[np.ndarray, int]:
    """Blocking worker: feed the source through one engine session."""
    converter = engine.new_session(ref_16k, ref_model_sr, profile)
    try:
        block = engine.preferred_block_samples(profile)
        out: list[np.ndarray] = []
        for start in range(0, source_16k.size, block):
            chunk = converter.convert_block(source_16k[start : start + block])
            if chunk is not None and chunk.size:
                out.append(np.asarray(chunk, dtype=np.float32))
        tail = converter.finish()
        if tail is not None and tail.size:
            out.append(np.asarray(tail, dtype=np.float32))
    finally:
        converter.close()
    converted = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
    converted = resampler.resample(converted, engine.sr_out, profile.out_sample_rate)
    return converted, profile.out_sample_rate


@router.post("", summary="Convert an uploaded audio file to a target voice")
async def convert_file(
    source: UploadFile = File(..., description="source audio (wav/mp3/flac/ogg/…)"),
    voice_id: str | None = Form(default=None, description="registered target voice"),
    reference: UploadFile | None = File(
        default=None, description="inline reference audio (instead of voice_id)"
    ),
    latency_profile: str | None = Form(
        default=None, description="ultra_low | balanced | quality"
    ),
    components: AppComponents = Depends(get_components),
) -> Response:
    started = time.perf_counter()
    settings = components.settings
    profile = settings.profile(latency_profile or settings.default_profile)

    if bool(voice_id) == bool(reference):
        raise VoiceValidationError(
            "provide exactly one target: registered 'voice_id' OR an uploaded "
            "'reference' file (not both, not neither)"
        )

    # --- source audio -------------------------------------------------------
    source_bytes = await source.read()
    src_16k = decode_to_pcm(source_bytes, sample_rate=16_000)
    src_seconds = src_16k.size / 16_000
    if src_seconds < MIN_SOURCE_SECONDS:
        raise VoiceValidationError(f"source too short: {src_seconds:.2f}s")
    if src_seconds > MAX_SOURCE_SECONDS:
        raise VoiceValidationError(
            f"source too long: {src_seconds:.1f}s > {MAX_SOURCE_SECONDS:.0f}s cap"
        )

    # --- target voice -------------------------------------------------------
    if voice_id:
        ref_16k, ref_model_sr = await components.voice_service.load_reference(voice_id)
    else:
        ref_bytes = await reference.read()  # type: ignore[union-attr]
        ref_16k = decode_to_pcm(ref_bytes, sample_rate=16_000)
        ref_model_sr = decode_to_pcm(ref_bytes, sample_rate=MODEL_SR)

    if not components.engine.is_ready():
        from rafidain.core.errors import EngineNotReadyError

        raise EngineNotReadyError("engine not loaded yet — retry shortly")

    # --- convert (blocking work off the event loop) --------------------------
    try:
        converted, out_rate = await asyncio.to_thread(
            _run_conversion,
            engine=components.engine,
            resampler=components.resampler,
            source_16k=src_16k,
            ref_16k=ref_16k,
            ref_model_sr=ref_model_sr,
            profile=profile,
        )
    except ReferenceAudioError:
        raise
    except VoiceValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 — engine adapter failures
        logger.exception("file conversion failed")
        raise ReferenceAudioError(f"conversion failed: {exc}") from exc
    finally:
        await source.close()
        if reference is not None:
            await reference.close()

    if converted.size == 0:
        raise ReferenceAudioError("engine produced no audio for this input")

    wav = _wav_bytes(converted, out_rate)
    elapsed = time.perf_counter() - started
    out_seconds = converted.size / out_rate
    rtf = elapsed / out_seconds if out_seconds > 0 else 0.0
    logger.info(
        "file conversion done",
        extra={
            "voice_id": voice_id or "inline-reference",
            "profile": profile.key,
            "src_seconds": round(src_seconds, 2),
            "out_seconds": round(out_seconds, 2),
            "rtf": round(rtf, 3),
        },
    )
    from rafidain.observability.metrics import metrics as prom

    prom()["conversions_total"].labels(profile.key).inc()
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="rafidain_converted.wav"',
            "X-Conversion-Profile": profile.key,
            "X-Conversion-Seconds": f"{out_seconds:.2f}",
            "X-Conversion-Elapsed": f"{elapsed:.2f}",
            "X-Conversion-RTF": f"{rtf:.3f}",
            "X-Conversion-Sample-Rate": str(out_rate),
        },
    )
