"""Audio infrastructure: PCM codecs, block accumulation, resampling, VAD, decode."""

from __future__ import annotations

import numpy as np

_INT16_MAX = 32768.0


def pcm_bytes_to_float(pcm: bytes) -> np.ndarray:
    """Little-endian signed 16-bit mono PCM bytes -> float32 in [-1, 1)."""
    if len(pcm) % 2:
        raise ValueError(f"pcm byte length must be even, got {len(pcm)}")
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / _INT16_MAX


def float_to_pcm_bytes(x: np.ndarray) -> bytes:
    """float32 [-1, 1) -> little-endian signed 16-bit mono PCM bytes."""
    clipped = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0 - 1e-6)
    return (clipped * _INT16_MAX).astype("<i2").tobytes()


class BlockAccumulator:
    """Accumulates an unbounded stream of PCM into fixed-size blocks.

    The WebSocket handler receives arbitrary packet sizes; the engine wants
    exact blocks. This tiny, dependency-free utility bridges the two.
    """

    def __init__(self, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self._buf = np.zeros(0, dtype=np.float32)
        self.blocks_emitted = 0

    def push(self, chunk: np.ndarray) -> list[np.ndarray]:
        """Append a chunk and return every complete block that became ready."""
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        self._buf = np.concatenate([self._buf, chunk])
        blocks: list[np.ndarray] = []
        while self._buf.size >= self.block_size:
            blocks.append(self._buf[: self.block_size].copy())
            self._buf = self._buf[self.block_size :]
            self.blocks_emitted += 1
        return blocks

    def flush(self, *, pad: bool = True) -> np.ndarray:
        """Return the remainder, zero-padded to a full block (if requested)."""
        remainder = self._buf
        self._buf = np.zeros(0, dtype=np.float32)
        if not pad or remainder.size == 0:
            return remainder
        if remainder.size >= self.block_size:
            return remainder
        return np.pad(remainder, (0, self.block_size - remainder.size))

    @property
    def pending(self) -> int:
        return self._buf.size
