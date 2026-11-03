"""WebSocket protocol: message schemas and (de)serialization.

Text frames carry JSON control messages; binary frames carry raw PCM
s16le mono audio (16 kHz in, session-rate out).

Client -> Server::

    {"type": "stream.start"}
    {"type": "stream.pause"}
    {"type": "stream.resume"}
    {"type": "stream.stop"}
    {"type": "metrics.poll"}

Server -> Client::

    {"type": "session.ready", ...}
    {"type": "stream.started" | "stream.paused" | "stream.resumed" | "stream.stopped"}
    {"type": "metrics.snapshot", "metrics": {...}}
    {"type": "warning", "code": "...", "message": "..."}
    {"type": "error", "code": "...", "message": "..."}
    {"type": "session.closed", "reason": "..."}

Close codes: 4401 unauthorized, 4404 unknown session, 4410 session busy,
4429 expired, 4500 internal error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from rafidain.core.errors import ProtocolError

VALID_CLIENT_TYPES = {
    "stream.start",
    "stream.pause",
    "stream.resume",
    "stream.stop",
    "metrics.poll",
}


@dataclass(slots=True)
class ClientMessage:
    type: str
    raw: dict


def parse_client_message(text: str) -> ClientMessage:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"control message is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("control message must be a JSON object")
    msg_type = payload.get("type")
    if msg_type not in VALID_CLIENT_TYPES:
        raise ProtocolError(
            f"unknown message type {msg_type!r} (valid: {sorted(VALID_CLIENT_TYPES)})"
        )
    return ClientMessage(type=msg_type, raw=payload)


def server_message(msg_type: str, **fields) -> str:
    return json.dumps({"type": msg_type, **fields}, ensure_ascii=False)


# WebSocket close codes (4xxx: app-defined, no retry)
CLOSE_UNAUTHORIZED = 4401
CLOSE_SESSION_NOT_FOUND = 4404
CLOSE_SESSION_BUSY = 4410
CLOSE_SESSION_EXPIRED = 4429
CLOSE_PROTOCOL_ERROR = 4403
CLOSE_INTERNAL = 4500
