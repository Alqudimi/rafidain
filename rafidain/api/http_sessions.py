"""Session lifecycle endpoints (create/inspect/close; streaming happens over WS)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from rafidain.api.deps import AppComponents, get_components, require_api_key

router = APIRouter(
    prefix="/api/v1/sessions", tags=["sessions"], dependencies=[Depends(require_api_key)]
)


class CreateSessionRequest(BaseModel):
    voice_id: str = Field(..., description="registered target voice")
    latency_profile: str | None = Field(
        default=None, description="ultra_low | balanced | quality"
    )


@router.post("", status_code=201, summary="Create a conversion session")
async def create_session(
    payload: CreateSessionRequest,
    request: Request,
    components: AppComponents = Depends(get_components),
) -> dict:
    # Fail fast if the voice is unknown.
    await components.voice_service.get(payload.voice_id)
    profile_key = payload.latency_profile or components.settings.default_profile
    profile = components.settings.profile(profile_key)
    session = await components.session_manager.create(payload.voice_id, profile.key)
    base = str(request.base_url).rstrip("/")
    return {
        "session_id": session.session_id,
        "voice_id": session.voice_id,
        "profile": profile.as_dict(),
        "ws_url": f"{base}/api/v1/stream/{session.session_id}",
        "expires_at": session.expires_at,
    }


@router.get("", summary="List sessions")
async def list_sessions(components: AppComponents = Depends(get_components)) -> dict:
    return {
        "sessions": [s.as_dict() for s in components.session_manager.list()],
        "active": components.session_manager.active_count,
    }


@router.get("/{session_id}", summary="Session details & live metrics")
async def get_session(
    session_id: str, components: AppComponents = Depends(get_components)
) -> dict:
    session = components.session_manager.get(session_id)
    return session.as_dict()


@router.delete("/{session_id}", summary="Close a session")
async def close_session(
    session_id: str, components: AppComponents = Depends(get_components)
) -> dict:
    session = components.session_manager.get(session_id)
    await components.session_manager.close(session, "closed by client")
    return {"closed": session_id}
