"""In-memory mock engine: passthrough conversion for development & tests.

Lets the entire service (sessions, WebSocket pipeline, backpressure,
metrics, CI) run with zero ML dependencies. Implements the same
``ConversionEngine`` contract as the Seed-VC adapter — a direct proof of
Liskov substitutability at the architecture level.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from rafidain.domain.value_objects import SAMPLE_RATE_IN, LatencyProfile


class MockConverterSession:
    """Passthrough with optional simulated processing delay and gain."""

    def __init__(self, profile: LatencyProfile, *, delay_ms: float = 0.0, gain: float = 1.0):
        self._profile = profile
        self._delay = delay_ms / 1000.0
        self._gain = gain
        self.block_frame_16k = int(profile.block_ms / 1000 * SAMPLE_RATE_IN)

    def convert_block(self, pcm: np.ndarray) -> np.ndarray:
        if self._delay:
            time.sleep(self._delay)
        return np.asarray(pcm, dtype=np.float32) * self._gain

    def finish(self) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)

    def close(self) -> None:
        return None


class MockConversionEngine:
    """Deterministic engine used when ``RAFIDAIN__ENGINE__IMPL=mock``."""

    impl = "mock"
    sr_in = SAMPLE_RATE_IN
    sr_out = SAMPLE_RATE_IN

    def __init__(self, *, delay_ms: float = 0.0, gain: float = 1.0) -> None:
        self.delay_ms = delay_ms
        self.gain = gain

    def is_ready(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def new_session(
        self,
        ref_pcm_16k: np.ndarray,
        ref_pcm_model_sr: np.ndarray | None,
        profile: LatencyProfile,
    ) -> MockConverterSession:
        del ref_pcm_16k, ref_pcm_model_sr
        return MockConverterSession(profile, delay_ms=self.delay_ms, gain=self.gain)

    def preferred_block_samples(self, profile: LatencyProfile) -> int:
        return int(profile.block_ms / 1000 * SAMPLE_RATE_IN)

    def warmup(self, profile: LatencyProfile) -> dict[str, Any]:
        return {"impl": "mock", "warmup_rtf": 0.0, "device": "cpu"}

    def describe(self) -> dict[str, Any]:
        return {"impl": "mock", "loaded": True, "device": "cpu"}
