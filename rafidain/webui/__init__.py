# rafidain.webui — Arabic RTL web control panel + realtime live page.
#
# Gradio (extra `webui`) is imported lazily by ``mount_webui`` so the core
# service — and the whole test suite — runs without it.
from rafidain.webui.convert import (
    MicStreamState,
    close_mic_stream,
    convert_file,
    open_stream,
    resample,
    stream_mic_chunk,
)

__all__ = [
    "MicStreamState",
    "available",
    "close_mic_stream",
    "convert_file",
    "mount_webui",
    "open_stream",
    "resample",
    "stream_mic_chunk",
]


def available() -> bool:
    """True when the Gradio dependency is importable in this environment."""
    try:
        import gradio  # noqa: F401
    except ImportError:
        return False
    return True


def mount_webui(app):  # pragma: no cover - exercised via /web requests
    """Mount the Gradio panel at ``/web`` on the FastAPI application.

    Returns True when mounted; False (with a logged hint) when the `webui`
    extra is missing or ``RAFIDAIN__WEBUI=false``.
    """
    from rafidain.core.config import get_settings
    from rafidain.core.logging import get_logger

    logger = get_logger(__name__)
    settings = app.state.settings or get_settings()
    if not settings.webui:
        logger.info("web UI disabled (RAFIDAIN__WEBUI=false)")
        return False
    try:
        import gradio as gr
    except ImportError:
        logger.warning(
            "web UI requested but gradio is not installed — "
            "run `uv sync --extra webui` (or make install) and restart"
        )
        return False

    from rafidain.webui.app import build_webui

    demo = build_webui(app)
    gr.mount_gradio_app(app, demo, path="/web")
    logger.info("web control panel mounted at /web (gradio %s)", gr.__version__)
    return True
