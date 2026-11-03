"""Domain entities: stateful objects with identity and lifecycle rules.

Entities may be mutated (sessions track their own state machine), but they
never perform I/O — persistence and streaming are port responsibilities.
"""

from __future__ import annotations

import statistics
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from rafidain.domain.value_objects import LatencyProfile, SessionState

_ALLOWED: dict[SessionState, set[SessionState]] = {
    SessionState.CREATED: {SessionState.READY, SessionState.CLOSED, SessionState.ERROR},
    SessionState.READY: {SessionState.STREAMING, SessionState.CLOSED, SessionState.ERROR},
    # STREAMING/PAUSED may stop back to READY (re-arm for another call segment).
    SessionState.STREAMING: {
        SessionState.PAUSED,
        SessionState.READY,
        SessionState.CLOSED,
        SessionState.ERROR,
    },
    SessionState.PAUSED: {
        SessionState.STREAMING,
        SessionState.READY,
        SessionState.CLOSED,
        SessionState.ERROR,
    },
    SessionState.CLOSED: set(),
    SessionState.ERROR: set(),
}


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


@dataclass(slots=True)
class StreamMetrics:
    """Rolling performance counters for one streaming session."""

    blocks_in: int = 0
    blocks_out: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    overflows: int = 0
    vad_silence_blocks: int = 0
    rtf_ewma: float = 0.0
    _block_seconds: deque = field(default_factory=lambda: deque(maxlen=128))

    def observe_block(self, *, audio_seconds: float, processing_seconds: float) -> None:
        self._block_seconds.append(processing_seconds)
        rtf = processing_seconds / audio_seconds if audio_seconds > 0 else 0.0
        # Exponential moving average, alpha=0.1 — stable for dashboards.
        self.rtf_ewma = 0.9 * self.rtf_ewma + 0.1 * rtf

    @property
    def chunk_p50_ms(self) -> float:
        return statistics.median(self._block_seconds) * 1000 if self._block_seconds else 0.0

    @property
    def chunk_p95_ms(self) -> float:
        if not self._block_seconds:
            return 0.0
        ordered = sorted(self._block_seconds)
        idx = max(0, round(0.95 * (len(ordered) - 1)))
        return ordered[idx] * 1000

    def snapshot(self) -> dict:
        return {
            "blocks_in": self.blocks_in,
            "blocks_out": self.blocks_out,
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "overflows": self.overflows,
            "vad_silence_blocks": self.vad_silence_blocks,
            "rtf_ewma": round(self.rtf_ewma, 4),
            "chunk_p50_ms": round(self.chunk_p50_ms, 1),
            "chunk_p95_ms": round(self.chunk_p95_ms, 1),
        }


@dataclass(slots=True)
class Session:
    """A live conversion session bound to one target voice."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    voice_id: str = ""
    profile: LatencyProfile | None = None
    state: SessionState = SessionState.CREATED
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 300.0)  # noqa: E501
    created_at_iso: str = field(default_factory=_utcnow_iso)
    ws_connected: bool = False
    close_reason: str | None = None
    metrics: StreamMetrics = field(default_factory=StreamMetrics)

    def transition(self, new_state: SessionState) -> None:
        """Guarded state-machine transition (raises on illegal moves)."""
        from rafidain.core.errors import InvalidSessionStateError

        if new_state not in _ALLOWED[self.state]:
            raise InvalidSessionStateError(
                f"illegal transition {self.state.value} -> {new_state.value}",
                detail={
                    "session_id": self.session_id,
                    "from": self.state.value,
                    "to": new_state.value,
                },
            )
        self.state = new_state

    def touch(self, ttl_seconds: float) -> None:
        self.expires_at = time.time() + ttl_seconds

    def is_expired(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        if self.state in (SessionState.CLOSED, SessionState.ERROR):
            return False
        return not self.ws_connected and now > self.expires_at

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "voice_id": self.voice_id,
            "profile": self.profile.key if self.profile else None,
            "state": self.state.value,
            "created_at": self.created_at_iso,
            "expires_at": self.expires_at,
            "ws_connected": self.ws_connected,
            "close_reason": self.close_reason,
            "metrics": self.metrics.snapshot(),
        }


@dataclass(slots=True)
class VoiceProfile:
    """A registered target voice (reference audio + metadata)."""

    voice_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    duration_s: float = 0.0
    sample_rate: int = 22_050
    created_at_iso: str = field(default_factory=_utcnow_iso)
    notes: str = ""
    sha256: str = ""

    def as_dict(self) -> dict:
        return {
            "voice_id": self.voice_id,
            "name": self.name,
            "duration_s": round(self.duration_s, 2),
            "sample_rate": self.sample_rate,
            "created_at": self.created_at_iso,
            "notes": self.notes,
        }


@dataclass(slots=True)
class JobRecord:
    """Status of an asynchronous job (fine-tuning runs, etc.)."""

    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    kind: str = "finetune"
    state: str = "queued"
    progress: float = 0.0
    message: str = ""
    created_at_iso: str = field(default_factory=_utcnow_iso)
    params: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    logs: deque = field(default_factory=lambda: deque(maxlen=80))

    def append_log(self, line: str) -> None:
        self.logs.append(line.rstrip())

    def as_dict(self, *, include_logs: bool = True) -> dict:
        data = {
            "job_id": self.job_id,
            "kind": self.kind,
            "state": self.state,
            "progress": round(self.progress, 3),
            "message": self.message,
            "created_at": self.created_at_iso,
            "params": self.params,
            "result": self.result,
        }
        if include_logs:
            data["logs"] = list(self.logs)
        return data
