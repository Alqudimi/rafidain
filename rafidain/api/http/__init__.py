"""HTTP routes: health, voices, sessions, jobs."""

from fastapi import APIRouter

from rafidain.api.http_health import router as health_router
from rafidain.api.http_jobs import router as jobs_router
from rafidain.api.http_sessions import router as sessions_router
from rafidain.api.http_voices import router as voices_router


def register_routes(app) -> None:
    api = APIRouter()
    api.include_router(voices_router)
    api.include_router(sessions_router)
    api.include_router(jobs_router)
    app.include_router(health_router)
    app.include_router(api)
