"""Optional voice-activity gate in front of the engine.

When enabled, silent blocks bypass the neural pipeline entirely and are
returned as digital silence: this saves GPU time and avoids conversion
artifacts during microphone silence.
"""

from __future__ import annotations

import numpy as np

from rafidain.core.errors import DependencyMissingError

_FRAME_MS = 30  # webrtcvad supports 10/20/30 ms frames only


class VoiceActivityGate:
    """Classify blocks as speech/silence; replace silence with zeros."""

    def __init__(
        self,
        mode: str = "off",
        sample_rate: int = 16_000,
        speech_ratio_threshold: float = 0.5,
        attack_frames: int = 3,
    ) -> None:
        if mode not in ("off", "webrtc"):
            raise ValueError(f"unsupported VAD mode: {mode!r}")
        self.mode = mode
        self.sample_rate = sample_rate
        self.threshold = speech_ratio_threshold
        self.frame_len = sample_rate * _FRAME_MS // 1000
        self._frames_after_silence = attack_frames
        if mode == "webrtc":
            try:
                import webrtcvad
            except ImportError as exc:  # pragma: no cover - env specific
                raise DependencyMissingError(
                    "VAD mode 'webrtc' requires the webrtcvad package: "
                    "uv sync --extra vad"
                ) from exc
            self._vad = webrtcvad.Vad(2)  # aggressiveness 0..3

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def process(self, block: np.ndarray) -> tuple[bool, np.ndarray]:
        """Return ``(is_speech, out_block)``.

        ``out_block`` equals the input for speech blocks and zeros for
        silence blocks, so the pipeline always emits constant-length audio.
        """
        if not self.enabled:
            return True, block
        is_speech = self._classify(block)
        if is_speech:
            self._frames_after_silence = 0
            return True, block
        # Keep a short attack margin so trailing speech isn't clipped.
        if self._frames_after_silence > 0:
            self._frames_after_silence -= 1
            return True, block
        return False, np.zeros_like(block)

    def _classify(self, block: np.ndarray) -> bool:
        block = np.asarray(block, dtype=np.float32)
        total = 0
        speech = 0
        for start in range(0, block.size - self.frame_len + 1, self.frame_len):
            frame = block[start : start + self.frame_len]
            pcm = (np.clip(frame, -1.0, 1.0 - 1e-6) * 32768.0).astype("<i2").tobytes()
            total += 1
            if self._vad.is_speech(pcm, self.sample_rate):
                speech += 1
        if total == 0:
            return False
        return (speech / total) >= self.threshold
