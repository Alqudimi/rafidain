"""The streaming WebSocket endpoint — the heart of the live-call path."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from rafidain.api.ws.protocol import (
    CLOSE_INTERNAL,
    CLOSE_PROTOCOL_ERROR,
    CLOSE_SESSION_BUSY,
    CLOSE_SESSION_EXPIRED,
    CLOSE_SESSION_NOT_FOUND,
    CLOSE_UNAUTHORIZED,
    parse_client_message,
    server_message,
)
from rafidain.application.stream_pipeline import (
    OutMessage,
    PipelineDependencies,
    build_pipeline,
)
from rafidain.core.errors import (
    AuthenticationError,
    InvalidSessionStateError,
    ProtocolError,
    RafidainError,
    SessionLimitError,
    SessionNotFoundError,
)
from rafidain.core.logging import get_logger, set_request_id
from rafidain.domain.value_objects import SessionState
from rafidain.observability.metrics import metrics as prom

logger = get_logger(__name__)

router = APIRouter()

_WS_API_KEY_HEADER = "x-api-key"


def _authenticate(ws: WebSocket, api_keys: list[str]) -> None:
    if not api_keys:
        return
    key = ws.query_params.get("token") or ws.headers.get(_WS_API_KEY_HEADER)
    if not key or key not in api_keys:
        raise AuthenticationError("missing or invalid API key (token= or X-API-Key)")


class _GaugeGuard:
    """Inc/dec the active-session gauge exactly once per streaming period."""

    def __init__(self) -> None:
        self._held = False
        self._gauge = prom()["ws_sessions_active"]

    def acquire(self) -> None:
        if not self._held:
            self._gauge.inc()
            self._held = True

    def release(self) -> None:
        if self._held:
            self._gauge.dec()
            self._held = False


@router.websocket("/api/v1/stream/{session_id}")
async def stream_session(ws: WebSocket, session_id: str) -> None:
    """Serve one live conversion session over binary PCM frames."""
    set_request_id(f"ws-{uuid.uuid4().hex[:12]}")
    await ws.accept()
    components = ws.app.state.components
    settings = components.settings

    pipeline = None
    writer_task: asyncio.Task | None = None
    session = None
    guard = _GaugeGuard()
    ctx: dict[str, Any] = {"guard": guard}

    try:
        _authenticate(ws, settings.api_keys)
    except AuthenticationError as exc:
        await ws.close(code=CLOSE_UNAUTHORIZED, reason=exc.message[:100])
        return

    try:
        session = components.session_manager.attach(session_id)
    except SessionNotFoundError:
        await ws.close(code=CLOSE_SESSION_NOT_FOUND, reason="unknown session")
        return
    except SessionLimitError:
        await ws.close(code=CLOSE_SESSION_BUSY, reason="session already connected")
        return
    except RafidainError as exc:
        code = CLOSE_SESSION_EXPIRED if exc.code == "session_expired" else CLOSE_INTERNAL
        await ws.close(code=code, reason=exc.message[:100])
        return

    logger.info("ws attached", extra={"session_id": session.session_id})
    try:
        if session.state == SessionState.CREATED:
            # Condition the engine on the target voice (thread — heavy for seedvc).
            ref_16k, ref_model_sr = await components.voice_service.load_reference(
                session.voice_id
            )
            converter = await asyncio.to_thread(
                components.engine.new_session,
                ref_16k,
                ref_model_sr,
                session.profile,
            )
            components.converters[session.session_id] = converter
            session.transition(SessionState.READY)
        elif session.state != SessionState.PAUSED:
            raise InvalidSessionStateError(
                f"cannot resume session in state {session.state.value}"
            )
    except RafidainError as exc:
        await _safe_send_text(
            ws, server_message("error", code=exc.code, message=exc.message)
        )
        await ws.close(code=CLOSE_PROTOCOL_ERROR, reason=exc.message[:100])
        return

    block_samples = components.engine.preferred_block_samples(session.profile)
    pipeline, send_queue = build_pipeline(
        session,
        PipelineDependencies(
            converter=components.converters[session.session_id],
            engine_sr_out=components.engine.sr_out,
            resampler=components.resampler,
            vad=components.build_vad(),
        ),
        block_samples=block_samples,
        send_queue_size=settings.send_queue_size,
    )

    writer_task = asyncio.create_task(
        _writer(ws, send_queue), name=f"ws-writer-{session.session_id[:8]}"
    )
    await _safe_send_text(
        ws,
        server_message(
            "session.ready",
            session_id=session.session_id,
            voice_id=session.voice_id,
            profile=session.profile.as_dict(),
            audio_in={"sample_rate": 16_000, "encoding": "pcm_s16le", "channels": 1},
            audio_out={
                "sample_rate": pipeline.out_sample_rate,
                "encoding": "pcm_s16le",
                "channels": 1,
            },
        ),
    )

    # ----------------------------------------------------------------- reader
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if (data := message.get("text")) is not None:
                await _handle_control(ws, pipeline, session, data, ctx)
            elif (data := message.get("bytes")) is not None:
                if session.state != SessionState.STREAMING:
                    await _safe_send_text(
                        ws,
                        server_message(
                            "error",
                            code="not_streaming",
                            message="send stream.start before audio frames",
                        ),
                    )
                    continue
                await pipeline.feed(data)
    except WebSocketDisconnect:
        pass
    except RafidainError as exc:
        logger.warning("ws session error: %s", exc.message)
        await _safe_send_text(
            ws, server_message("error", code=exc.code, message=exc.message)
        )
    except Exception:  # noqa: BLE001 — never leak a stack trace to the socket
        logger.exception("ws handler crashed")
    finally:
        if pipeline is not None and session is not None:
            try:
                if session.state == SessionState.STREAMING:
                    await pipeline.finish()
            except Exception:  # noqa: BLE001
                logger.exception("pipeline finish failed")
            pipeline.close()
        if writer_task is not None:
            writer_task.cancel()
            try:
                await writer_task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                pass
        if session is not None:
            components.converters.pop(session.session_id, None)
            await components.session_manager.close(session, "client disconnected")
            components.session_manager.detach(session)
        guard.release()
        logger.info(
            "ws detached", extra={"session_id": session.session_id if session else "-"}
        )


async def _handle_control(
    ws: WebSocket, pipeline, session, text: str, ctx: dict[str, Any]
) -> None:
    try:
        msg = parse_client_message(text)
    except ProtocolError as exc:
        await _safe_send_text(
            ws, server_message("error", code=exc.code, message=exc.message)
        )
        return

    guard: _GaugeGuard = ctx["guard"]

    if msg.type == "stream.start":
        try:
            session.transition(SessionState.STREAMING)
        except InvalidSessionStateError as exc:
            await _safe_send_text(
                ws, server_message("error", code=exc.code, message=exc.message)
            )
            return
        pipeline.resume()
        guard.acquire()
        await _safe_send_text(ws, server_message("stream.started"))
    elif msg.type == "stream.pause":
        pipeline.pause()
        if session.state == SessionState.STREAMING:
            session.state = SessionState.PAUSED
        await _safe_send_text(ws, server_message("stream.paused"))
    elif msg.type == "stream.resume":
        pipeline.resume()
        if session.state == SessionState.PAUSED:
            session.state = SessionState.STREAMING
        await _safe_send_text(ws, server_message("stream.resumed"))
    elif msg.type == "stream.stop":
        await pipeline.finish()
        if session.state in (SessionState.STREAMING, SessionState.PAUSED):
            session.state = SessionState.READY
        guard.release()
        await _safe_send_text(ws, server_message("stream.stopped"))
    elif msg.type == "metrics.poll":
        await _safe_send_text(
            ws, server_message("metrics.snapshot", metrics=await pipeline.snapshot())
        )


async def _writer(ws: WebSocket, queue: asyncio.Queue[OutMessage]) -> None:
    """Drain the send queue; exits quietly when the client disconnects."""
    while True:
        item: OutMessage = await queue.get()
        try:
            if item.kind == "audio":
                await ws.send_bytes(item.payload)
            else:
                await ws.send_text(server_message(item.payload["type"], **item.payload))
        except Exception:  # noqa: BLE001 — socket closed under us: stop writing
            break


async def _safe_send_text(ws: WebSocket, text: str) -> None:
    with contextlib.suppress(Exception):  # socket may already be gone
        await ws.send_text(text)
