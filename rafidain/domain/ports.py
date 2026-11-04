"""Ports (interfaces) the application layer depends on.

Dependency-Inversion in practice: the application orchestrates *these*
protocols, never concrete implementations. Infrastructure adapters
(``rafidain.infrastructure``) implement them.

Each protocol is intentionally small — Interface Segregation. A component
that needs only to push audio implements nothing else.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from rafidain.domain.entities import JobRecord, Session, VoiceProfile
from rafidain.domain.value_objects import LatencyProfile

# ---------------------------------------------------------------------------
# Voice conversion engine
# ---------------------------------------------------------------------------


@runtime_checkable
class ConverterSession(Protocol):
    """Per-stream conversion state machine owned by an engine session."""

    def convert_block(self, pcm: np.ndarray) -> np.ndarray:
        """Convert one block of 16 kHz mono float32 PCM.

        Returns converted audio (float32, engine output rate); may be empty
        while the engine buffers look-ahead context.
        """
        ...

    def finish(self) -> np.ndarray:
        """Flush remaining buffered audio at stream end."""
        ...

    def close(self) -> None:
        """Release per-session resources."""
        ...


@runtime_checkable
class ConversionEngine(Protocol):
    """A voice-conversion engine (Seed-VC adapter, mock, future models)."""

    impl: str

    @property
    def sr_in(self) -> int: ...

    @property
    def sr_out(self) -> int: ...

    def is_ready(self) -> bool: ...

    def load(self) -> None:
        """Eagerly initialise heavyweight resources (no-op if already)."""
        ...

    def new_session(
        self,
        ref_pcm_16k: np.ndarray,
        ref_pcm_model_sr: np.ndarray | None,
        profile: LatencyProfile,
    ) -> ConverterSession:
        """Condition a fresh converter on the target voice reference."""
        ...

    def preferred_block_samples(self, profile: LatencyProfile) -> int:
        """How many input samples the pipeline should accumulate per block."""
        ...

    def warmup(self, profile: LatencyProfile) -> dict[str, Any]:
        """Run a dummy conversion; returns timing stats (rtf, seconds)."""
        ...


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


@runtime_checkable
class VoiceRepository(Protocol):
    """Storage for voice profiles and their reference audio."""

    async def save(
        self, voice: VoiceProfile, ref_22k: np.ndarray, ref_16k: np.ndarray
    ) -> None: ...

    async def get(self, voice_id: str) -> VoiceProfile | None: ...

    async def list(self) -> list[VoiceProfile]: ...

    async def delete(self, voice_id: str) -> bool: ...

    async def load_audio(self, voice_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (pcm_16k, pcm_model_sr) for a stored voice."""
        ...


@runtime_checkable
class TaggedCache(Protocol):
    """Small key/value cache (embeddings, hot metadata)."""

    async def get(self, key: str) -> Any | None: ...

    async def put(self, key: str, value: Any, ttl_s: float | None = None) -> None: ...

    async def delete(self, key: str) -> None: ...


# ---------------------------------------------------------------------------
# Async jobs (fine-tuning)
# ---------------------------------------------------------------------------


@runtime_checkable
class JobQueue(Protocol):
    """Queue that executes long-running jobs and tracks their status."""

    async def submit(self, kind: str, params: dict) -> str: ...

    async def get(self, job_id: str) -> JobRecord | None: ...

    async def cancel(self, job_id: str) -> bool: ...


# ---------------------------------------------------------------------------
# Time (testability)
# ---------------------------------------------------------------------------


@runtime_checkable
class Clock(Protocol):
    def now(self) -> float: ...


class SystemClock:
    def now(self) -> float:
        import time

        return time.time()


__all__ = [
    "Clock",
    "ConversionEngine",
    "ConverterSession",
    "JobQueue",
    "Session",
    "SystemClock",
    "TaggedCache",
    "VoiceProfile",
    "VoiceRepository",
]
