"""Domain value-object unit tests."""

from __future__ import annotations

import pytest

from rafidain.domain.entities import Session
from rafidain.domain.value_objects import (
    DEFAULT_PROFILES,
    AudioFormat,
    LatencyProfile,
    SessionState,
    get_profile,
)


class TestLatencyProfile:
    def test_defaults_are_valid(self):
        for profile in DEFAULT_PROFILES.values():
            profile.validate()
            assert profile.algorithmic_delay_ms >= profile.block_ms

    def test_balanced_matches_upstream_reference(self):
        """The upstream-published tiny-model benchmark: 430 ms total latency."""
        p = DEFAULT_PROFILES["balanced"]
        assert p.block_ms == 180
        assert p.crossfade_ms == 40
        assert p.context_right_ms == 20
        assert p.diffusion_steps == 10
        # 180 + 40 + 20 = 240 ms algorithmic + ~150-190 ms inference ≈ 430 ms.
        assert p.algorithmic_delay_ms == 240

    def test_validation_rejects_bad_values(self):
        with pytest.raises(ValueError):
            LatencyProfile(
                key="bad", block_ms=10, crossfade_ms=40, context_left_ms=2000,
                context_right_ms=20, diffusion_steps=10, inference_cfg_rate=0.7,
                max_prompt_s=3.0,
            ).validate()
        with pytest.raises(ValueError):
            LatencyProfile(
                key="bad", block_ms=100, crossfade_ms=40, context_left_ms=2000,
                context_right_ms=20, diffusion_steps=10, inference_cfg_rate=1.5,
                max_prompt_s=3.0,
            ).validate()

    def test_get_profile_unknown_raises(self):
        with pytest.raises(ValueError):
            get_profile("does_not_exist")

    def test_as_dict_contains_contract_fields(self):
        data = DEFAULT_PROFILES["ultra_low"].as_dict()
        assert data["key"] == "ultra_low"
        assert {"block_ms", "crossfade_ms", "diffusion_steps"} <= set(data)


class TestAudioFormat:
    def test_byte_math(self):
        fmt = AudioFormat(sample_rate=16_000)
        assert fmt.bytes_for_duration(0.02) == 640  # 20 ms frame
        assert fmt.duration_of(640) == pytest.approx(0.02)
        with pytest.raises(ValueError):
            _ = AudioFormat(encoding="mp3").bytes_per_sample


class TestSessionStateMachine:
    def _session(self) -> Session:
        return Session(voice_id="v", profile=DEFAULT_PROFILES["balanced"])

    def test_happy_path(self):
        s = self._session()
        s.transition(SessionState.READY)
        s.transition(SessionState.STREAMING)
        s.transition(SessionState.PAUSED)
        s.transition(SessionState.STREAMING)
        s.transition(SessionState.READY)  # stop -> re-arm
        s.transition(SessionState.STREAMING)
        s.transition(SessionState.CLOSED)

    def test_illegal_transitions_raise(self):
        from rafidain.core.errors import InvalidSessionStateError

        s = self._session()
        with pytest.raises(InvalidSessionStateError):
            s.transition(SessionState.STREAMING)  # CREATED -> STREAMING illegal
        s.transition(SessionState.READY)
        with pytest.raises(InvalidSessionStateError):
            s.transition(SessionState.READY)  # READY -> READY illegal
        s.transition(SessionState.CLOSED)
        with pytest.raises(InvalidSessionStateError):
            s.transition(SessionState.READY)  # terminal

    def test_expiry(self):
        s = self._session()
        assert not s.is_expired()
        s.expires_at = 0.0
        assert s.is_expired()
        s.ws_connected = True
        assert not s.is_expired()  # live connection keeps it alive
