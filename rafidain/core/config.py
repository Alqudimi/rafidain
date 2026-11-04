"""Typed, layered configuration (pydantic-settings v2).

Resolution order: environment variables (RAFIDAIN__*) > .env file > defaults.
Nested fields are addressed with ``__`` (e.g. ``RAFIDAIN__ENGINE__IMPL``).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rafidain.domain.value_objects import LatencyProfile


class EngineSettings(BaseModel):
    """Seed-VC engine wiring. Mirrors the upstream checkpoint layout."""

    impl: str = "seedvc"  # seedvc | mock
    device: str = "auto"  # auto | cuda | cpu | mps
    fp16: bool = True
    # HuggingFace source of seed-uvit-tat-xlsr-tiny:
    hf_repo: str = "Plachta/Seed-VC"
    checkpoint: str = "DiT_uvit_tat_xlsr_ema.pth"
    config: str = "config_dit_mel_seed_uvit_xlsr_tiny.yml"
    # Aux modules (same ones the upstream real-time GUI loads):
    campplus_repo: str = "funasr/campplus"
    campplus_file: str = "campplus_cn_common.bin"
    hift_repo: str = "FunAudioLLM/CosyVoice-300M"
    hift_file: str = "hift.pt"
    # Local fine-tuned checkpoint (Iraqi-dialect model) overrides HF download:
    local_checkpoint: str | None = None
    local_config: str | None = None
    cache_dir: Path = Path("./checkpoints")
    # Lifecycle:
    eager_load: bool = True
    warmup: bool = True
    max_concurrency: int = 1  # parallel model inferences across sessions

    @field_validator("impl")
    @classmethod
    def _known_impl(cls, v: str) -> str:
        if v not in ("seedvc", "mock"):
            raise ValueError("impl must be 'seedvc' or 'mock'")
        return v


class Settings(BaseSettings):
    """Environment contract: ``RAFIDAIN__FIELD`` for top-level and
    ``RAFIDAIN__NESTED__FIELD`` for nested settings (e.g. the engine).
    """

    model_config = SettingsConfigDict(
        env_prefix="RAFIDAIN__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- app ---------------------------------------------------------------
    app_name: str = "rafidain"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_json: bool = True
    api_keys: list[str] = Field(default_factory=list)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # --- engine --------------------------------------------------------------
    engine: EngineSettings = Field(default_factory=EngineSettings)

    # --- streaming -----------------------------------------------------------
    default_profile: str = "balanced"
    session_ttl_s: float = 300.0
    session_cleanup_interval_s: float = 30.0
    max_sessions: int = 64
    send_queue_size: int = 64
    vad: str = "off"  # off | webrtc

    # --- storage ---------------------------------------------------------------
    data_dir: Path = Path("./data")
    redis_url: str | None = None

    @field_validator("api_keys", "cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v):
        """CSV helper for direct Settings(...) construction.

        Note: when the value arrives from an environment variable or a .env
        file, pydantic-settings parses list fields as JSON — use
        ``RAFIDAIN__API_KEYS=["key"]`` / ``RAFIDAIN__CORS_ORIGINS=["*"]``.
        """
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    def profile(self, key: str | None = None) -> LatencyProfile:
        from rafidain.domain.value_objects import get_profile

        return get_profile(key or self.default_profile)

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)

    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "voices"


def seed_vc_root() -> Path:
    """Location of the cloned https://github.com/Plachtaa/seed-vc repo.

    Resolution: $SEED_VC_ROOT > ./third_party/seed-vc.
    """
    env = os.environ.get("SEED_VC_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path.cwd() / "third_party" / "seed-vc").resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
