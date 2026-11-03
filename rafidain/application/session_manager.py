"""Session lifecycle management (creation, WS attachment, expiry sweeping)."""

from __future__ import annotations

import asyncio
import contextlib
import time

from rafidain.core.config import Settings
from rafidain.core.errors import SessionLimitError, SessionNotFoundError
from rafidain.core.logging import get_logger
from rafidain.domain.entities import Session
from rafidain.domain.value_objects import SessionState

logger = get_logger(__name__)


class SessionManager:
    """In-memory registry of live sessions + background expiry sweeper.

    Session state lives in-process by design: a session is bound to one
    WebSocket and one engine converter. Horizontal scaling therefore pins a
    session to one node (see ARCHITECTURE.md — Scaling).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._sweeper_task: asyncio.Task | None = None

    # --- lifecycle -------------------------------------------------------------

    async def create(self, voice_id: str, profile_key: str) -> Session:
        async with self._lock:
            if len(self._sessions) >= self._settings.max_sessions:
                raise SessionLimitError(
                    f"server at max_sessions={self._settings.max_sessions}"
                )
            session = Session(
                voice_id=voice_id,
                profile=self._settings.profile(profile_key),
                expires_at=time.time() + self._settings.session_ttl_s,
            )
            self._sessions[session.session_id] = session
        logger.info(
            "session created",
            extra={"session_id": session.session_id, "voice_id": voice_id},
        )
        return session

    def get(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"no session {session_id!r}")
        return session

    def attach(self, session_id: str) -> Session:
        """Bind the WebSocket connection to a session (single connection)."""
        session = self.get(session_id)
        if session.ws_connected:
            raise SessionLimitError("session already has an active connection")
        if session.is_expired():
            self._sessions.pop(session_id, None)
            from rafidain.core.errors import SessionExpiredError

            raise SessionExpiredError(f"session {session_id} expired")
        session.ws_connected = True
        session.touch(self._settings.session_ttl_s)
        return session

    def detach(self, session: Session) -> None:
        session.ws_connected = False
        session.touch(self._settings.session_ttl_s)

    async def close(self, session: Session, reason: str) -> None:
        if session.state not in (SessionState.CLOSED, SessionState.ERROR):
            session.close_reason = reason
            try:
                session.transition(SessionState.CLOSED)
            except Exception:  # noqa: BLE001 — closing must never fail
                session.state = SessionState.CLOSED
        session.ws_connected = False
        logger.info("session closed (%s)", reason, extra={"session_id": session.session_id})

    async def remove(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    # --- queries ------------------------------------------------------------------

    def list(self) -> list[Session]:
        return list(self._sessions.values())

    @property
    def active_count(self) -> int:
        return sum(1 for s in self._sessions.values() if s.ws_connected)

    # --- background sweeper ------------------------------------------------------

    async def start_background(self) -> None:
        if self._sweeper_task is None:
            self._sweeper_task = asyncio.create_task(
                self._sweep_loop(), name="session-sweeper"
            )

    async def shutdown(self) -> None:
        if self._sweeper_task is not None:
            self._sweeper_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sweeper_task
            self._sweeper_task = None

    async def _sweep_loop(self) -> None:
        interval = self._settings.session_cleanup_interval_s
        while True:
            await asyncio.sleep(interval)
            try:
                await self.sweep_expired()
            except Exception:  # noqa: BLE001 — sweeper resilience
                logger.exception("session sweep failed")

    async def sweep_expired(self) -> int:
        """Drop sessions past their TTL without a connection. Returns count."""
        now = time.time()
        expired = [
            sid
            for sid, session in self._sessions.items()
            if session.is_expired(now) and not session.ws_connected
        ]
        async with self._lock:
            for sid in expired:
                session = self._sessions.pop(sid, None)
                if session is not None:
                    logger.info("session expired", extra={"session_id": sid})
        return len(expired)
