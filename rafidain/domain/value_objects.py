"""Immutable value objects of the domain.

Everything here is pure Python (dataclasses / enums, numpy types allowed for
audio payloads). No framework imports: the domain must stay portable and
unit-testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Audio constants (contract of the whole system)
# ---------------------------------------------------------------------------

#: Rate at which the XLSR content encoder consumes audio (Seed-VC requirement).
SAMPLE_RATE_IN = 16_000
#: Seed-VC works on 20 ms frames => 50 frames per second at the model rate.
MODEL_FRAMES_PER_SECOND = 50
#: Default output packetization rate (telephony-friendly).
SAMPLE_RATE_OUT_DEFAULT = 16_000


@dataclass(frozen=True, slots=True)
class AudioFormat:
    """A concrete PCM audio format used on the wire."""

    sample_rate: int = SAMPLE_RATE_IN
    encoding: str = "pcm_s16le"
    channels: int = 1

    @property
    def bytes_per_sample(self) -> int:
        if self.encoding != "pcm_s16le":
            raise ValueError(f"unsupported encoding: {self.encoding}")
        return 2 * self.channels

    def bytes_for_duration(self, seconds: float) -> int:
        return int(self.sample_rate * seconds) * self.bytes_per_sample

    def duration_of(self, num_bytes: int) -> float:
        return num_bytes / self.sample_rate / self.bytes_per_sample


class SessionState(str, Enum):
    """Lifecycle of a conversion session (a finite state machine)."""

    CREATED = "created"
    READY = "ready"          # engine conditioned on the target voice
    STREAMING = "streaming"  # audio flowing in both directions
    PAUSED = "paused"        # connected, input discarded
    CLOSED = "closed"
    ERROR = "error"


class JobState(str, Enum):
    """Lifecycle of an asynchronous job (e.g. fine-tuning)."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Latency profiles
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LatencyProfile:
    """Tunable streaming parameters exposed to clients.

    The values mirror the knobs validated upstream by Seed-VC's real-time GUI
    (``real-time-gui.py``): block size, crossfade, left/right look-ahead,
    flow-matching steps and CFG rate.
    """

    key: str
    block_ms: int
    crossfade_ms: int
    context_left_ms: int
    context_right_ms: int
    diffusion_steps: int
    inference_cfg_rate: float
    max_prompt_s: float
    out_sample_rate: int = SAMPLE_RATE_OUT_DEFAULT
    description: str = ""

    @property
    def algorithmic_delay_ms(self) -> int:
        """Worst-case algorithmic added latency, excluding inference time.

        block (granularity) + crossfade (held overlap) + right context
        (look-ahead required before a block can be converted).
        """
        return self.block_ms + self.crossfade_ms + self.context_right_ms

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "block_ms": self.block_ms,
            "crossfade_ms": self.crossfade_ms,
            "context_left_ms": self.context_left_ms,
            "context_right_ms": self.context_right_ms,
            "diffusion_steps": self.diffusion_steps,
            "inference_cfg_rate": self.inference_cfg_rate,
            "max_prompt_s": self.max_prompt_s,
            "out_sample_rate": self.out_sample_rate,
            "algorithmic_delay_ms": self.algorithmic_delay_ms,
            "description": self.description,
        }

    def validate(self) -> None:
        if self.block_ms < 40:
            raise ValueError(f"{self.key}: block_ms must be >= 40 (50ms-frame engine)")
        if not 20 <= self.crossfade_ms <= 500:
            raise ValueError(f"{self.key}: crossfade_ms out of range [20, 500]")
        if self.diffusion_steps < 1:
            raise ValueError(f"{self.key}: diffusion_steps must be >= 1")
        if not 0.0 <= self.inference_cfg_rate <= 1.0:
            raise ValueError(f"{self.key}: inference_cfg_rate out of range [0, 1]")


#: Presets. ``balanced`` reproduces the upstream-published configuration of
#: seed-uvit-tat-xlsr-tiny: ~430 ms total latency with 10 diffusion steps.
DEFAULT_PROFILES: dict[str, LatencyProfile] = {
    "ultra_low": LatencyProfile(
        key="ultra_low",
        block_ms=120,
        crossfade_ms=40,
        context_left_ms=2000,
        context_right_ms=20,
        diffusion_steps=4,
        inference_cfg_rate=0.0,
        max_prompt_s=3.0,
        description="أدنى زمن وصول — خطوات تدفق 4 وبدون CFG",
    ),
    "balanced": LatencyProfile(
        key="balanced",
        block_ms=180,
        crossfade_ms=40,
        context_left_ms=2500,
        context_right_ms=20,
        diffusion_steps=10,
        inference_cfg_rate=0.7,
        max_prompt_s=3.0,
        description="توازن موثّق رسمياً: ~430ms مع 10 خطوات (مرجع Seed-VC)",
    ),
    "quality": LatencyProfile(
        key="quality",
        block_ms=500,
        crossfade_ms=100,
        context_left_ms=2500,
        context_right_ms=60,
        diffusion_steps=25,
        inference_cfg_rate=0.7,
        max_prompt_s=6.0,
        out_sample_rate=22_050,
        description="جودة أعلى للاستخدام غير التفاعلي — 25 خطوة تدفق",
    ),
}
for _profile in DEFAULT_PROFILES.values():
    _profile.validate()


def get_profile(key: str, profiles: dict[str, LatencyProfile] | None = None) -> LatencyProfile:
    table = profiles if profiles is not None else DEFAULT_PROFILES
    try:
        return table[key]
    except KeyError:
        raise ValueError(f"unknown latency profile: {key!r} (available: {sorted(table)})") from None
