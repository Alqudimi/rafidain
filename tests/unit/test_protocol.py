"""WebSocket control-protocol tests."""

from __future__ import annotations

import json

import pytest

from rafidain.api.ws.protocol import parse_client_message, server_message
from rafidain.core.errors import ProtocolError


def test_valid_messages_parse():
    msg = parse_client_message(json.dumps({"type": "stream.start"}))
    assert msg.type == "stream.start"
    msg = parse_client_message('{"type": "metrics.poll"}')
    assert msg.type == "metrics.poll"


def test_invalid_json_rejected():
    with pytest.raises(ProtocolError):
        parse_client_message("{not json")


def test_non_object_rejected():
    with pytest.raises(ProtocolError):
        parse_client_message("[1, 2, 3]")


def test_unknown_type_rejected():
    with pytest.raises(ProtocolError):
        parse_client_message('{"type": "make.coffee"}')


def test_server_message_roundtrip():
    text = server_message("metrics.snapshot", metrics={"rtf": 0.2})
    payload = json.loads(text)
    assert payload["type"] == "metrics.snapshot"
    assert payload["metrics"]["rtf"] == 0.2
