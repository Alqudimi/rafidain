"""Standalone Redis job worker: ``python -m rafidain.workers.finetune_worker``.

Deployed as a separate container/process when RAFIDAIN__REDIS_URL is set,
so training never competes with inference for GPU memory on the API node.
"""

from __future__ import annotations

import asyncio

from rafidain.application.finetune_service import FinetuneService
from rafidain.core.config import get_settings
from rafidain.core.logging import configure_logging, get_logger
from rafidain.infrastructure.jobs import RedisJobQueue

logger = get_logger(__name__)


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, as_json=settings.log_json)
    if not settings.redis_url:
        raise SystemExit(
            "RAFIDAIN__REDIS_URL is required for the worker "
            "(otherwise jobs run in-process inside the API)"
        )
    service = FinetuneService(settings)
    queue = service.queue
    if not isinstance(queue, RedisJobQueue):
        raise SystemExit("job queue is not Redis-backed — check RAFIDAIN__REDIS_URL")
    logger.info("finetune worker starting (redis=%s)", settings.redis_url)
    await queue.run_worker_forever()


if __name__ == "__main__":
    asyncio.run(main())
