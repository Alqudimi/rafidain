"""Composition root: build every component once and wire the object graph.

This is the ONLY place that knows how concrete implementations fit together
(the rest of the code depends on ports). See ARCHITECTURE.md — the manual DI
keeps the dependency graph explicit and test-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi import Header, HTTPException, Request

from rafidain.application.finetune_service import FinetuneService
from rafidain.application.session_manager import SessionManager
from rafidain.application.stream_pipeline import ConverterSession
from rafidain.application.voice_registry import VoiceService
from rafidain.core.config import Settings
from rafidain.core.errors import AuthenticationError
from rafidain.infrastructure.audio.resampler import SoxrResampler
from rafidain.infrastructure.audio.vad import VoiceActivityGate
from rafidain.infrastructure.engine_registry import build_engine
from rafidain.infrastructure.stores import FileSystemVoiceRepository, InMemoryCache, RedisCache


@dataclass
class AppComponents:
    """The application's wired object graph (single instance per process)."""

    settings: Settings
    engine: Any
    voice_service: VoiceService
    session_manager: SessionManager
    finetune_service: FinetuneService
    resampler: SoxrResampler
    warmup_stats: dict = field(default_factory=dict)
    converters: dict[str, ConverterSession] = field(default_factory=dict)

    def build_vad(self) -> VoiceActivityGate:
        return VoiceActivityGate(mode=self.settings.vad)

    def cache(self) -> Any:
        return self._cache

    _cache: Any = None


def build_components(settings: Settings) -> AppComponents:
    """Instantiate the whole graph. Called once at startup (lifespan)."""
    engine = build_engine(settings)
    voice_repo = FileSystemVoiceRepository(settings.voices_dir)
    cache = (
        RedisCache(settings.redis_url)
        if settings.redis_url
        else InMemoryCache()
    )
    components = AppComponents(
        settings=settings,
        engine=engine,
        voice_service=VoiceService(voice_repo, settings),
        session_manager=SessionManager(settings),
        finetune_service=FinetuneService(settings),
        resampler=SoxrResampler(),
    )
    components._cache = cache
    return components


def get_components(request: Request) -> AppComponents:
    return request.app.state.components


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """FastAPI dependency enforcing the static API key (when configured)."""
    components = get_components(request)
    keys = components.settings.api_keys
    if not keys:
        return
    if not x_api_key or x_api_key not in keys:
        raise HTTPException(
            status_code=AuthenticationError.http_status,
            detail="missing or invalid X-API-Key",
        )
