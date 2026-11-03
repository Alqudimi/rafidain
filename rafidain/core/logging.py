"""Logging setup: stdlib-only, optional JSON lines, request-id contextvars.

We deliberately avoid extra dependencies (structlog & co.): the standard
library covers the needs and keeps the dependency tree minimal.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def set_request_id(request_id: str) -> contextvars.Token:
    return _request_id_var.set(request_id)


def current_request_id() -> str:
    return _request_id_var.get()


class JsonFormatter(logging.Formatter):
    """Minimal JSON-lines formatter for production log collectors."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": _request_id_var.get(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key in ("session_id", "voice_id", "job_id", "impl", "rtf", "block_ms"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", *, as_json: bool = True) -> None:
    """Configure the root logger once per process."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    # Idempotency: clear previous handlers to avoid duplicates on reload.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if as_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
    root.addHandler(handler)

    # Third-party noise control.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    for noisy in ("httpx", "matplotlib", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
