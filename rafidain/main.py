"""Application factory & lifespan.

Run (dev):  RAFIDAIN__ENGINE__IMPL=mock uvicorn rafidain.main:app --reload
Run (prod): uvicorn rafidain.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rafidain import __version__
from rafidain.api.deps import build_components
from rafidain.api.http import register_routes
from rafidain.api.middleware import register_exception_handlers, request_context_middleware
from rafidain.api.ws.stream import router as ws_router
from rafidain.core.config import Settings, get_settings
from rafidain.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.settings
    configure_logging(settings.log_level, as_json=settings.log_json)
    components = build_components(settings)
    app.state.components = components

    if settings.engine.eager_load:
        try:
            await asyncio.to_thread(components.engine.load)
            if settings.engine.warmup:
                profile = settings.profile()
                components.warmup_stats = await asyncio.to_thread(
                    components.engine.warmup, profile
                )
        except Exception as exc:  # noqa: BLE001 — degrade, don't crash the API
            logger.error("engine load/warmup failed: %s — /readyz will report not-ready", exc)

    await components.session_manager.start_background()
    logger.info("rafidain %s up (engine=%s)", __version__, components.engine.impl)
    yield

    await components.session_manager.shutdown()
    logger.info("rafidain shut down cleanly")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="Rafidain — Realtime Iraqi Voice Conversion",
        description=(
            "Real-time voice conversion backend for live calls, built on "
            "Seed-VC (seed-uvit-tat-xlsr-tiny) and tuned for the Iraqi dialect. "
            "REST controls lifecycle; WebSocket carries the audio; /web is the "
            "Arabic RTL control panel and /live the realtime streaming page."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.middleware("http")(request_context_middleware)
    register_exception_handlers(app)
    register_routes(app)
    app.include_router(ws_router)

    # Web surfaces: landing + realtime live page (always) and the Gradio
    # control panel (needs the `webui` extra). Both degrade gracefully.
    try:
        from rafidain.webui.live import register_live_page

        register_live_page(app)
    except Exception as exc:  # noqa: BLE001 — the API must boot regardless
        logger.warning("live page unavailable: %s", exc)
    try:
        from rafidain.webui import mount_webui

        mount_webui(app)
    except Exception as exc:  # noqa: BLE001
        logger.warning("web control panel unavailable: %s", exc)
    return app


app = create_app()


def main() -> None:
    """``python -m rafidain.main`` — dev entrypoint."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "rafidain.main:app",
        host=settings.host,
        port=settings.port,
        log_config=None,
    )


if __name__ == "__main__":
    main()
