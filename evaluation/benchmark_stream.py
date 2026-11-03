"""Latency / RTF benchmark of the streaming engine (no HTTP involved).

Runs the configured engine (mock or seed-vc) against one or more latency
profiles with synthetic speech-like input and reports per-block timings
against the profile's latency budget. Use on the production GPU node to
validate the deployment before taking traffic.

    python evaluation/benchmark_stream.py --profile balanced --seconds 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rafidain.core.config import get_settings  # noqa: E402
from rafidain.domain.value_objects import DEFAULT_PROFILES, get_profile  # noqa: E402
from rafidain.infrastructure.engine_registry import build_engine  # noqa: E402


def synth_speech(seconds: float, sr: int, seed: int = 7) -> np.ndarray:
    """Speech-like excitation: voiced bursts + noise + envelope (not for MOS!)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 120 + 40 * np.sin(2 * np.pi * 3.0 * t)
    voiced = np.sin(2 * np.pi * np.cumsum(f0) / sr).astype(np.float32)
    noise = rng.standard_normal(t.size).astype(np.float32) * 0.05
    env = (0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t)).astype(np.float32)
    x = (voiced * 0.6 + noise) * env
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.7).astype(np.float32)


def bench(engine, profile, seconds: float) -> dict:
    engine.load()
    ref = synth_speech(2.0, 16_000)
    session = engine.new_session(ref, None, profile)
    block = engine.preferred_block_samples(profile)
    audio = synth_speech(seconds + 1, 16_000)
    times: list[float] = []
    emitted = 0.0
    n_blocks = 0
    t_stream = time.perf_counter()
    for start in range(0, audio.size - block, block):
        chunk = audio[start : start + block]
        t0 = time.perf_counter()
        out = session.convert_block(chunk)
        times.append(time.perf_counter() - t0)
        emitted += out.size / engine.sr_out
        n_blocks += 1
    tail = session.finish()
    emitted += tail.size / engine.sr_out
    wall = time.perf_counter() - t_stream
    session.close()

    audio_sec = n_blocks * block / 16_000
    rtf = wall / audio_sec if audio_sec else float("nan")
    return {
        "profile": profile.key,
        "engine": engine.impl,
        "blocks": n_blocks,
        "block_ms": profile.block_ms,
        "audio_seconds": round(audio_sec, 2),
        "converted_seconds": round(emitted, 2),
        "rtf_overall": round(rtf, 3),
        "block_p50_ms": round(statistics.median(times) * 1000, 1),
        "block_p95_ms": round(sorted(times)[max(0, round(0.95 * (len(times) - 1)))] * 1000, 1),
        "algorithmic_delay_ms": profile.algorithmic_delay_ms,
        "estimated_e2e_ms": round(
            statistics.median(times) * 1000 + profile.algorithmic_delay_ms, 1
        ),
        "realtime_ok": rtf < 1.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", default=None, help="single profile key (default: all)")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    settings = get_settings()
    engine = build_engine(settings)
    profiles = (
        [get_profile(args.profile)] if args.profile else list(DEFAULT_PROFILES.values())
    )
    results = []
    for profile in profiles:
        print(f"benchmarking {profile.key} ({engine.impl}) ...", file=sys.stderr)
        results.append(bench(engine, profile, args.seconds))
    print(json.dumps(results, indent=2, ensure_ascii=False))
    bad = [r["profile"] for r in results if not r["realtime_ok"]]
    if bad:
        print(f"WARNING: RTF >= 1.0 for profiles: {bad}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
