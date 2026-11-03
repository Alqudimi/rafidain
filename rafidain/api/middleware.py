"""ASGI middleware: request IDs, access metrics, error taxonomy -> HTTP."""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rafidain.core.errors import RafidainError
from rafidain.core.logging import get_logger, set_request_id
from rafidain.observability.metrics import metrics as prom

logger = get_logger(__name__)


async def request_context_middleware(request: Request, call_next):
    """Attach a request id + observe latency/counts for every HTTP call."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    token = set_request_id(request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001 — let the error handler below format it
        logger.exception("unhandled error")
        response = JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "internal server error"}},
        )
    response.headers["X-Request-ID"] = request_id
    duration = time.perf_counter() - start
    route = request.scope.get("route")
    path = route.path if route else request.url.path
    m = prom()
    m["http_requests"].labels(
        method=request.method, path=path, status=str(response.status_code)
    ).inc()
    m["http_latency"].labels(method=request.method, path=path).observe(duration)
    set_request_id(token)  # restore (context isolation for tests)
    return response


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RafidainError)
    async def rafidain_error_handler(_: Request, exc: RafidainError):
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())
