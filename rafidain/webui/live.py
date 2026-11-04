"""The lightweight realtime page served at ``/live``.

Pure vanilla HTML/JS (no build step, no gradio): getUserMedia → 16 kHz PCM
frames over the project's existing WebSocket protocol, playback of the
converted stream as it arrives, plus a file-conversion mode. Designed to be
served behind the Cloudflare tunnel where HTTPS enables microphone access.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from rafidain.core.logging import get_logger

logger = get_logger(__name__)

_LIVE_HTML = Path(__file__).parent / "static" / "live.html"


def register_live_page(app: FastAPI) -> None:
    """Serve the realtime page at /live (idempotent)."""

    @app.get("/live", response_class=HTMLResponse, include_in_schema=False)
    async def live_page() -> HTMLResponse:
        try:
            html = _LIVE_HTML.read_text(encoding="utf-8")
        except OSError:
            logger.warning("live page asset missing: %s", _LIVE_HTML)
            return HTMLResponse(
                "<h1 dir='rtl'>صفحة البث الحي غير مثبتة — webui/static/live.html مفقودة</h1>",
                status_code=500,
            )
        return HTMLResponse(html)

    @app.get("/", include_in_schema=False)
    async def index() -> HTMLResponse:
        """Landing: point humans at the two UIs, machines at /docs."""
        html = """
        <!doctype html><html dir="rtl" lang="ar"><head><meta charset="utf-8">
        <title>رافدَين — تحويل الصوت اللحظي</title>
        <style>
          body{font-family:system-ui,'Segoe UI',Tahoma;background:#0f172a;color:#e2e8f0;
               display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
          .card{background:#1e293b;padding:42px 56px;border-radius:18px;text-align:center;
                box-shadow:0 20px 60px rgba(0,0,0,.45)}
          h1{margin:0 0 6px;font-size:2em}p{color:#94a3b8;margin:4px 0 22px}
          a{display:inline-block;margin:6px;padding:12px 26px;border-radius:12px;
            text-decoration:none;font-weight:600}
          .live{background:#14b8a6;color:#042f2e}.panel{background:#38bdf8;color:#082f49}
          .docs{background:#334155;color:#e2e8f0}
        </style></head><body><div class="card">
          <h1>🏛️ رافدَين</h1>
          <p>تحويل الصوت اللحظي — Seed-VC · اللهجة العراقية</p>
          <a class="live" href="/live">🔴 صفحة البث الحي</a>
          <a class="panel" href="/web">🎛️ لوحة التحكم الكاملة</a>
          <a class="docs" href="/docs">📜 وثائق REST/WS API</a>
        </div></body></html>
        """
        return HTMLResponse(html)
