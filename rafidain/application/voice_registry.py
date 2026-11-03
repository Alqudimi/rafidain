"""Voice registration and retrieval use cases."""

from __future__ import annotations

import hashlib

import numpy as np

from rafidain.core.config import Settings
from rafidain.core.errors import VoiceNotFoundError, VoiceValidationError
from rafidain.core.logging import get_logger
from rafidain.domain.entities import VoiceProfile
from rafidain.domain.ports import VoiceRepository
from rafidain.infrastructure.audio.decode import decode_to_pcm

logger = get_logger(__name__)

#: Seed-VC references: 1..30 s (upstream recommendation; clones with 1 s).
MIN_REF_SECONDS = 1.0
MAX_REF_SECONDS = 30.0
MODEL_SR = 22_050


class VoiceService:
    """Registers target voices (reference audio) and serves them to sessions."""

    def __init__(self, repo: VoiceRepository, settings: Settings) -> None:
        self._repo = repo
        self._settings = settings

    async def register(
        self, *, name: str, audio: bytes, notes: str = ""
    ) -> VoiceProfile:
        if not audio:
            raise VoiceValidationError("reference audio file is required")
        pcm_16k = decode_to_pcm(audio, sample_rate=16_000)
        pcm_22k = decode_to_pcm(audio, sample_rate=MODEL_SR)
        duration = pcm_16k.size / 16_000
        if duration < MIN_REF_SECONDS:
            raise VoiceValidationError(
                f"reference too short: {duration:.2f}s < {MIN_REF_SECONDS}s"
            )
        if duration > MAX_REF_SECONDS:
            # Keep the loudest-centred MAX window rather than rejecting.
            center = pcm_16k.size // 2
            half = int(MAX_REF_SECONDS / 2 * 16_000)
            pcm_16k = pcm_16k[center - half : center + half]
            pcm_22k = pcm_22k[: int(MAX_REF_SECONDS * MODEL_SR)]
        if float(np.max(np.abs(pcm_16k))) < 1e-4:
            raise VoiceValidationError("reference audio is silent")

        voice = VoiceProfile(
            name=(name or "").strip()[:80],
            duration_s=min(duration, MAX_REF_SECONDS),
            sample_rate=MODEL_SR,
            notes=notes[:300],
            sha256=hashlib.sha256(audio).hexdigest()[:16],
        )
        await self._repo.save(voice, pcm_22k, pcm_16k)
        logger.info("voice registered", extra={"voice_id": voice.voice_id})
        return voice

    async def get(self, voice_id: str) -> VoiceProfile:
        voice = await self._repo.get(voice_id)
        if voice is None:
            raise VoiceNotFoundError(f"no voice {voice_id!r}")
        return voice

    async def list(self) -> list[VoiceProfile]:
        return await self._repo.list()

    async def delete(self, voice_id: str) -> None:
        await self.get(voice_id)  # 404 if unknown
        await self._repo.delete(voice_id)

    async def load_reference(self, voice_id: str) -> tuple[np.ndarray, np.ndarray]:
        """(pcm_16k, pcm_model_sr) — raises if the voice is missing."""
        await self.get(voice_id)
        audio = await self._repo.load_audio(voice_id)
        if audio is None:
            raise VoiceValidationError(
                f"voice {voice_id!r} has no stored audio — re-register it"
            )
        return audio
