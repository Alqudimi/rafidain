"""SessionManager lifecycle tests."""

from __future__ import annotations

import pytest

from rafidain.core.config import Settings
from rafidain.core.errors import SessionLimitError, SessionNotFoundError
from rafidain.domain.value_objects import DEFAULT_PROFILES


def make_manager(tmp_path, **kw) -> object:
    from rafidain.application.session_manager import SessionManager

    settings = Settings(
        _env_file=None,
        engine={"impl": "mock"},
        data_dir=tmp_path / "data",
        max_sessions=kw.get("max_sessions", 8),
        session_ttl_s=kw.get("session_ttl_s", 60.0),
    )
    return SessionManager(settings)


async def test_create_and_get(tmp_path):
    mgr = make_manager(tmp_path)
    session = await mgr.create("voice1", "balanced")
    assert session.profile is DEFAULT_PROFILES["balanced"]
    assert mgr.get(session.session_id) is session


async def test_unknown_session_raises(tmp_path):
    mgr = make_manager(tmp_path)
    with pytest.raises(SessionNotFoundError):
        mgr.get("nope")


async def test_max_sessions_enforced(tmp_path):
    mgr = make_manager(tmp_path, max_sessions=2)
    await mgr.create("v", "balanced")
    await mgr.create("v", "balanced")
    with pytest.raises(SessionLimitError):
        await mgr.create("v", "balanced")


async def test_attach_guard(tmp_path):
    mgr = make_manager(tmp_path)
    session = await mgr.create("v", "balanced")
    attached = mgr.attach(session.session_id)
    assert attached.ws_connected
    with pytest.raises(SessionLimitError):  # second connection refused
        mgr.attach(session.session_id)
    mgr.detach(attached)
    mgr.attach(session.session_id)  # can re-attach after detach


async def test_sweep_expired(tmp_path):
    mgr = make_manager(tmp_path, session_ttl_s=0.05)
    session = await mgr.create("v", "balanced")
    assert await mgr.sweep_expired() == 0  # not yet expired
    import asyncio
    import time

    session.expires_at = time.time() - 1
    assert await mgr.sweep_expired() == 1
    with pytest.raises(SessionNotFoundError):
        mgr.get(session.session_id)
    _ = asyncio
