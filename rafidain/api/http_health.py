"""Liveness, readiness and Prometheus endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

import rafidain
from rafidain.api.deps import get_components
from rafidain.observability.metrics import render_metrics

router = APIRouter(tags=["ops"])


@router.get("/healthz", summary="Liveness probe")
async def healthz() -> dict:
    return {"status": "ok", "service": "rafidain", "version": rafidain.__version__}


@router.get("/readyz", summary="Readiness probe (engine + storage)")
async def readyz(request: Request) -> dict:
    components = get_components(request)
    engine_ready = components.engine.is_ready()
    storage_ok = True
    try:
        components.settings.voices_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        storage_ok = False
    ready = engine_ready and storage_ok
    return {
        "ready": ready,
        "engine": {
            "impl": components.engine.impl,
            "loaded": engine_ready,
            "ready": engine_ready,
        },
        "storage": {"ok": storage_ok},
        "sessions": {
            "registered": len(components.session_manager.list()),
            "active": components.session_manager.active_count,
            "max": components.settings.max_sessions,
        },
        "warmup": components.warmup_stats,
    }


@router.get("/metrics", summary="Prometheus metrics", response_class=Response)
async def prometheus_metrics() -> Response:
    return Response(content=render_metrics(), media_type="text/plain; version=0.0.4")
