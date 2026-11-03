"""Persistence adapters: voice repository (filesystem) and caches."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from rafidain.core.logging import get_logger
from rafidain.domain.entities import VoiceProfile
from rafidain.infrastructure.audio.decode import encode_wav_pcm16, read_wav_pcm16

logger = get_logger(__name__)

_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".opus", ".ogg"}


class FileSystemVoiceRepository:
    """Voice profiles stored as plain files — inspectable, backup-friendly.

    Layout::

        <root>/<voice_id>/meta.json
        <root>/<voice_id>/ref_16000.wav     # content/speaker encoder input
        <root>/<voice_id>/ref_22050.wav     # mel prompt (model rate)
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _dir(self, voice_id: str) -> Path:
        return self._root / voice_id

    async def save(
        self, voice: VoiceProfile, ref_22k: np.ndarray, ref_16k: np.ndarray
    ) -> None:
        import anyio

        def _write() -> None:
            vdir = self._dir(voice.voice_id)
            vdir.mkdir(parents=True, exist_ok=True)
            encode_wav_pcm16(ref_22k, voice.sample_rate, vdir / "ref_22050.wav")
            encode_wav_pcm16(ref_16k, 16_000, vdir / "ref_16000.wav")
            (vdir / "meta.json").write_text(
                json.dumps(voice.as_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        await anyio.to_thread.run_sync(_write)

    async def get(self, voice_id: str) -> VoiceProfile | None:
        import anyio

        meta_path = self._dir(voice_id) / "meta.json"

        def _read() -> VoiceProfile | None:
            if not meta_path.exists():
                return None
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            return VoiceProfile(
                voice_id=data["voice_id"],
                name=data.get("name", ""),
                duration_s=data.get("duration_s", 0.0),
                sample_rate=data.get("sample_rate", 22_050),
                created_at_iso=data.get("created_at", ""),
                notes=data.get("notes", ""),
                sha256=data.get("sha256", ""),
            )

        return await anyio.to_thread.run_sync(_read)

    async def list(self) -> list[VoiceProfile]:

        async def _all() -> list[VoiceProfile]:
            out: list[VoiceProfile] = []
            for child in sorted(self._root.iterdir()):
                if child.is_dir() and (child / "meta.json").exists():
                    voice = await self.get(child.name)
                    if voice is not None:
                        out.append(voice)
            return out

        return await _all()

    async def delete(self, voice_id: str) -> bool:
        import shutil

        import anyio

        vdir = self._dir(voice_id)
        if not vdir.exists():
            return False

        def _rm() -> None:
            shutil.rmtree(vdir)

        await anyio.to_thread.run_sync(_rm)
        return True

    async def load_audio(self, voice_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (pcm_16k, pcm_model_sr) for a stored voice."""
        import anyio

        vdir = self._dir(voice_id)
        if not (vdir / "ref_16000.wav").exists():
            return None

        def _read() -> tuple[np.ndarray, np.ndarray]:
            return (
                read_wav_pcm16(vdir / "ref_16000.wav"),
                read_wav_pcm16(vdir / "ref_22050.wav"),
            )

        return await anyio.to_thread.run_sync(_read)


class InMemoryCache:
    """Process-local TTL cache (single-node deployments, tests)."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[Any, float]] = {}

    async def get(self, key: str) -> Any | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at and time.time() > expires_at:
            self._data.pop(key, None)
            return None
        return value

    async def put(self, key: str, value: Any, ttl_s: float | None = None) -> None:
        expires_at = time.time() + ttl_s if ttl_s else None
        self._data[key] = (value, expires_at)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)


class RedisCache:
    """Distributed cache. Degrades gracefully to no-ops on outage."""

    def __init__(self, url: str, prefix: str = "rafidain:cache:") -> None:
        self._url = url
        self._prefix = prefix
        self._client: Any = None

    def _conn(self):
        if self._client is None:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(self._url, decode_responses=True)
        return self._client

    async def _safe(self, coro_factory, default=None):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001 — cache must never fail requests
            logger.warning("redis cache unavailable: %s", exc)
            return default

    async def get(self, key: str) -> Any | None:
        raw = await self._safe(lambda: self._conn().get(self._prefix + key))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return raw

    async def put(self, key: str, value: Any, ttl_s: float | None = None) -> None:
        payload = json.dumps(value, ensure_ascii=False, default=str)
        await self._safe(
            lambda: (
                self._conn().set(self._prefix + key, payload, ex=int(ttl_s))
                if ttl_s
                else self._conn().set(self._prefix + key, payload)
            )
        )

    async def delete(self, key: str) -> None:
        await self._safe(lambda: self._conn().delete(self._prefix + key))
