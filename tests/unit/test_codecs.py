"""PCM codec & block accumulation tests."""

from __future__ import annotations

import numpy as np
import pytest

from rafidain.infrastructure.audio.codecs import (
    BlockAccumulator,
    float_to_pcm_bytes,
    pcm_bytes_to_float,
)


def test_pcm_roundtrip_is_lossless():
    x = np.linspace(-0.99, 0.99, 1001, dtype=np.float32)
    y = pcm_bytes_to_float(float_to_pcm_bytes(x))
    np.testing.assert_allclose(x, y, atol=1 / 32768)


def test_odd_byte_length_rejected():
    with pytest.raises(ValueError):
        pcm_bytes_to_float(b"\x01")


def test_accumulator_emits_exact_blocks():
    acc = BlockAccumulator(block_size=100)
    blocks = acc.push(np.zeros(250, dtype=np.float32))
    assert [b.size for b in blocks] == [100, 100]
    assert acc.pending == 50
    blocks = acc.push(np.zeros(10, dtype=np.float32))
    assert blocks == []
    assert acc.pending == 60
    rest = acc.flush(pad=True)
    assert rest.size == 100  # zero-padded to a full block
    assert acc.pending == 0


def test_accumulator_flush_without_pad():
    acc = BlockAccumulator(block_size=100)
    acc.push(np.zeros(30, dtype=np.float32))
    rest = acc.flush(pad=False)
    assert rest.size == 30
