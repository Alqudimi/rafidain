"""High-quality, dependency-light resampling (soxr)."""

from __future__ import annotations

import numpy as np
import soxr


class SoxrResampler:
    """One-shot block resampling.

    Used for output packetization (model rate -> session rate). The Seed-VC
    engine itself maintains torchaudio streaming resamplers internally, so
    this class is only needed at the transport boundary.
    """

    def resample(self, x: np.ndarray, orig_sr: int, new_sr: int) -> np.ndarray:
        if orig_sr == new_sr or x.size == 0:
            return np.asarray(x, dtype=np.float32)
        y = soxr.resample(np.asarray(x, dtype=np.float32), orig_sr, new_sr, quality="HQ")
        return y.astype(np.float32)
