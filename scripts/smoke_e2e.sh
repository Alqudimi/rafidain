#!/usr/bin/env bash
# End-to-end smoke test of the rafidain server with the mock engine,
# including the new web surfaces (/web, /live) and the WS streaming path.
# Usage: bash scripts/smoke_e2e.sh    (from the repository root, uv-managed)
set -uo pipefail
cd "$(dirname "$0")/.."

PY="uv run --python 3.11 python"

# fixture: a 6 s reference and a 4 s source tone
python3 - <<'EOF'
import wave, math, struct, os
os.makedirs("/tmp/rafidain_smoke", exist_ok=True)
for name, dur in (("ref.wav", 6.0), ("src.wav", 4.0)):
    with wave.open(f"/tmp/rafidain_smoke/{name}", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        for i in range(int(16000 * dur)):
            f = 200 + 400 * (i / (16000 * dur))
            w.writeframes(struct.pack("<h", int(20000 * math.sin(2 * math.pi * f * i / 16000))))
print("fixtures ready")
EOF
REF=/tmp/rafidain_smoke/ref.wav

export RAFIDAIN__ENGINE__IMPL=mock RAFIDAIN__LOG_JSON=false RAFIDAIN__API_KEYS='["testkey123"]'
$PY -m rafidain.main > server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

for _ in $(seq 1 40); do
  curl -sf http://127.0.0.1:8000/healthz >/dev/null 2>&1 && break
  sleep 1
done
echo "== healthz: $(curl -s http://127.0.0.1:8000/healthz)"
echo "== /       -> $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/)"
echo "== /live   -> $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/live)"
echo "== /web    -> $(curl -s -o /dev/null -w '%{http_code}' -L http://127.0.0.1:8000/web)"
echo "== /docs   -> $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/docs)"

echo "== register voice (authed):"
curl -s -X POST http://127.0.0.1:8000/api/v1/voices \
  -H "X-API-Key: testkey123" -F "file=@$REF" -F "name=smoke-voice" | head -c 300
echo

VOICE_ID=$(curl -s -H "X-API-Key: testkey123" http://127.0.0.1:8000/api/v1/voices \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['voices'][0]['voice_id'])")
echo "== voice_id: $VOICE_ID"
echo "== unauthorized list (expect 401): $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/api/v1/voices)"

echo "== ws smoke client (3 s):"
$PY scripts/smoke_ws_client.py --url http://127.0.0.1:8000 --seconds 3 --api-key testkey123 2>&1 | tail -6

echo "== webui conversion helpers (mock):"
$PY - <<'EOF'
import asyncio
from pathlib import Path
import numpy as np
from rafidain.core.config import Settings
from rafidain.api.deps import build_components
from rafidain.webui.convert import convert_file, stream_mic_chunk, MicStreamState

async def main():
    settings = Settings(RAFIDAIN__ENGINE__IMPL="mock")
    comp = build_components(settings)
    # register a voice through the application layer (same path as the UI)
    ref = Path("/tmp/rafidain_smoke/ref.wav").read_bytes()
    voice = await comp.voice_service.register(name="smoke", audio=ref)
    data = Path("/tmp/rafidain_smoke/src.wav").read_bytes()
    out, stats = await convert_file(comp, data, voice.voice_id, "balanced")
    print("convert_file ->", out.name, stats)
    rng = np.random.default_rng(0)
    chunk = (rng.standard_normal(4800) * 0.1).astype(np.float32)
    state = MicStreamState()
    state, out1 = await stream_mic_chunk(comp, state, chunk, 48000, voice.voice_id, "balanced")
    state, out2 = await stream_mic_chunk(comp, state, chunk, 48000, voice.voice_id, "balanced")
    print("mic chunks ->", out1.size, out2.size, "samples")
    assert out1.size == 1600, f"expected 1600, got {out1.size}"

asyncio.run(main())
print("webui convert helpers OK")
EOF
echo "== done"
