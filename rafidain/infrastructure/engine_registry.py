"""Engine factory — Open/Closed at the composition level.

New engine implementations register themselves here instead of touching
existing call sites: ``register_engine("myengine", factory)``.
"""

from __future__ import annotations

from collections.abc import Callable

from rafidain.core.config import EngineSettings, Settings
from rafidain.core.errors import ConfigError
from rafidain.core.logging import get_logger
from rafidain.domain.ports import ConversionEngine
from rafidain.infrastructure.mock_engine import MockConversionEngine

logger = get_logger(__name__)

EngineFactory = Callable[[EngineSettings], ConversionEngine]

_FACTORIES: dict[str, EngineFactory] = {
    "mock": lambda _cfg: MockConversionEngine(),
}


def register_engine(name: str, factory: EngineFactory) -> None:
    """Plug in a custom engine implementation (extension without modification)."""
    _FACTORIES[name] = factory
    logger.info("engine implementation registered: %s", name)


def build_engine(settings: Settings) -> ConversionEngine:
    try:
        if settings.engine.impl == "seedvc":
            # Imported lazily: pulls the whole ML stack.
            from rafidain.infrastructure.seedvc.engine import SeedVCStreamingEngine

            return SeedVCStreamingEngine(settings.engine)
        factory = _FACTORIES[settings.engine.impl]
    except KeyError:
        raise ConfigError(
            f"unknown engine implementation: {settings.engine.impl!r} "
            f"(available: {sorted(_FACTORIES)})"
        ) from None
    return factory(settings.engine)
