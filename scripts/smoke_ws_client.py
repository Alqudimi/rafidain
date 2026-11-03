#!/usr/bin/env python3
"""End-to-end smoke test against a running Rafidain instance.

    python scripts/smoke_ws_client.py [--url http://127.0.0.1:8000] [--seconds 3]

Registers a synthetic voice, opens a session, streams PCM over the
WebSocket, prints the final metrics and saves the converted audio to
``smoke_output.wav``. Works against the mock engine (no GPU) — perfect CI
companion for deployments.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def synth_voice(seconds: float = 3.0, sr: int = 16_000) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    f0 = 150 + 30 * np.sin(2 * np.pi * 2 * t)
    x = np.sin(2 * np.pi * np.cumsum(f0) / sr).astype(np.float32) * 0.6
    return (x * 32768).astype("<i2").tobytes()


def http_json(url: str, payload: dict, api_key: str | None = None) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    if api_key:
        req.add_header("X-API-Key", api_key)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def register_voice(base: str, api_key: str | None) -> str:
    boundary = "rafidainsmokeboundary"
    pcm = synth_voice()
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(
        b'Content-Disposition: form-data; name="file"; filename="smoke.wav"\r\n'
        b"Content-Type: audio/wav\r\n\r\n"
    )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(pcm)
    body.write(buf.getvalue())
    body.write(f"\r\n--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="name"\r\n\r\nsmoke-voice\r\n')
    body.write(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        f"{base}/api/v1/voices",
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    if api_key:
        req.add_header("X-API-Key", api_key)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["voice_id"]


async def stream(base: str, session: dict, seconds: float, api_key: str | None) -> None:
    import websockets

    ws_url = session["ws_url"].replace("http://", "ws://").replace("https://", "wss://")
    if api_key:
        ws_url += f"?token={api_key}"
    async with websockets.connect(ws_url, max_size=2**22) as ws:
        print("->", await ws.recv())  # session.ready
        await ws.send(json.dumps({"type": "stream.start"}))
        print("->", await ws.recv())  # stream.started

        async def reader():
            received = bytearray()
            async for message in ws:
                if isinstance(message, bytes):
                    received.extend(message)
                else:
                    data = json.loads(message)
                    if data["type"] == "metrics.snapshot":
                        print("metrics:", json.dumps(data["metrics"], ensure_ascii=False))
                    else:
                        print("->", message)
            Path("smoke_output.pcm").write_bytes(received)
            print(f"received {len(received)} PCM bytes -> smoke_output.pcm")

        reader_task = asyncio.get_running_loop().create_task(reader())
        pcm = synth_voice(seconds)
        chunk = 16000 * 20 // 1000 * 2  # 20 ms frames
        for start in range(0, len(pcm) - chunk, chunk):
            await ws.send(pcm[start : start + chunk])
            await asyncio.sleep(0.02)  # real-time pacing
        await asyncio.sleep(1.0)
        await ws.send(json.dumps({"type": "metrics.poll"}))
        await asyncio.sleep(0.5)
        await ws.send(json.dumps({"type": "stream.stop"}))
        await asyncio.sleep(0.5)
        await ws.close()
        await asyncio.wait_for(reader_task, timeout=3)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    try:
        voice_id = register_voice(args.url, args.api_key)
        print(f"voice registered: {voice_id}")
        session = http_json(
            f"{args.url}/api/v1/sessions",
            {"voice_id": voice_id, "latency_profile": "balanced"},
            args.api_key,
        )
        print(f"session: {session['session_id']}  ws: {session['ws_url']}")
        asyncio.run(stream(args.url, session, args.seconds, args.api_key))
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}", file=sys.stderr)
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
