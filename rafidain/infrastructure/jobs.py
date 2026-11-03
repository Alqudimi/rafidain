"""Async job queues for long-running work (fine-tuning).

Two interchangeable implementations of the same port:

* :class:`InProcessJobQueue` — jobs run as asyncio tasks in the API process
  (simple, single-node; the runner executes in a worker thread).
* :class:`RedisJobQueue` — jobs are pushed to a Redis list consumed by
  ``python -m rafidain.workers.finetune_worker`` on any node.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from typing import Any

from rafidain.core.logging import get_logger
from rafidain.domain.entities import JobRecord
from rafidain.domain.value_objects import JobState

logger = get_logger(__name__)

#: Contract of the runner executed for each job. ``ctx`` lets the runner
#: report progress, append logs and observe cancellation.
JobRunner = Callable[[JobRecord, "JobContext"], None]


class JobContext:
    """Handle given to a job runner (thread-safe)."""

    def __init__(self, record: JobRecord, cancel_event: threading.Event) -> None:
        self._record = record
        self._cancel = cancel_event
        self._lock = threading.Lock()

    def set_progress(self, fraction: float, message: str = "") -> None:
        with self._lock:
            self._record.progress = max(0.0, min(1.0, float(fraction)))
            if message:
                self._record.message = message

    def log(self, line: str) -> None:
        with self._lock:
            self._record.append_log(line)

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise JobCancelledError()


class JobCancelledError(Exception):
    """Raised inside a runner when cancellation was requested."""


class InProcessJobQueue:
    """Executes jobs inside the API process (each in a worker thread)."""

    def __init__(self, runner: JobRunner) -> None:
        self._runner = runner
        self._jobs: dict[str, JobRecord] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = threading.Lock()

    async def submit(self, kind: str, params: dict) -> str:
        record = JobRecord(kind=kind, params=params, state=JobState.QUEUED.value)
        event = threading.Event()
        with self._lock:
            self._jobs[record.job_id] = record
            self._cancel_events[record.job_id] = event
        self._tasks[record.job_id] = asyncio.create_task(
            self._execute(record, event), name=f"job-{record.job_id}"
        )
        return record.job_id

    async def _execute(self, record: JobRecord, cancel_event: threading.Event) -> None:
        ctx = JobContext(record, cancel_event)
        record.state = JobState.RUNNING.value
        logger.info(
            "job started: %s %s", record.kind, record.job_id,
            extra={"job_id": record.job_id},
        )
        try:
            await asyncio.to_thread(self._runner, record, ctx)
        except JobCancelledError:
            record.state = JobState.CANCELLED.value
            record.message = record.message or "cancelled"
            logger.info("job cancelled: %s", record.job_id, extra={"job_id": record.job_id})
        except Exception as exc:  # noqa: BLE001 — job failures are data, not crashes
            record.state = JobState.FAILED.value
            record.message = f"{type(exc).__name__}: {exc}"
            record.append_log(record.message)
            logger.exception("job failed: %s", record.job_id)
        else:
            record.state = JobState.DONE.value
            record.progress = 1.0
            logger.info("job done: %s", record.job_id, extra={"job_id": record.job_id})
        finally:
            self._tasks.pop(record.job_id, None)

    async def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    async def cancel(self, job_id: str) -> bool:
        event = self._cancel_events.get(job_id)
        record = self._jobs.get(job_id)
        if event is None or record is None:
            return False
        if record.state in (JobState.DONE.value, JobState.FAILED.value, JobState.CANCELLED.value):
            return False
        event.set()
        return True


class RedisJobQueue:
    """Multi-node queue: LPUSH for jobs, a worker consumes via BRPOP.

    State is kept in a Redis hash per job so any API instance can serve
    ``GET /jobs/{id}`` while the worker updates it.
    """

    _QUEUE_KEY = "rafidain:jobs:queue"
    _HASH_PREFIX = "rafidain:jobs:job:"
    _CANCEL_PREFIX = "rafidain:jobs:cancel:"

    def __init__(self, url: str, runner: JobRunner) -> None:
        self._url = url
        self._runner = runner
        self._client: Any = None

    def _conn(self):
        if self._client is None:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(self._url, decode_responses=True)
        return self._client

    @staticmethod
    def _serialize(record: JobRecord) -> str:
        data = record.as_dict(include_logs=True)
        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def _deserialize(raw: str) -> JobRecord:
        data = json.loads(raw)
        record = JobRecord(
            job_id=data["job_id"],
            kind=data.get("kind", "finetune"),
            state=data.get("state", JobState.QUEUED.value),
            progress=float(data.get("progress", 0.0)),
            message=data.get("message", ""),
            created_at_iso=data.get("created_at", ""),
            params=data.get("params", {}),
            result=data.get("result", {}),
        )
        for line in data.get("logs", [])[-80:]:
            record.append_log(line)
        return record

    async def submit(self, kind: str, params: dict) -> str:
        record = JobRecord(kind=kind, params=params, state=JobState.QUEUED.value)
        conn = self._conn()
        await conn.hset(self._HASH_PREFIX + record.job_id, "data", self._serialize(record))
        await conn.rpush(
            self._QUEUE_KEY,
            json.dumps({"job_id": record.job_id, "kind": kind, "params": params}),
        )
        return record.job_id

    async def get(self, job_id: str) -> JobRecord | None:
        raw = await self._conn().hget(self._HASH_PREFIX + job_id, "data")
        if raw is None:
            return None
        return self._deserialize(raw)

    async def cancel(self, job_id: str) -> bool:
        conn = self._conn()
        raw = await conn.hget(self._HASH_PREFIX + job_id, "data")
        if raw is None:
            return False
        record = self._deserialize(raw)
        if record.state not in (JobState.QUEUED.value, JobState.RUNNING.value):
            return False
        await conn.set(self._CANCEL_PREFIX + job_id, "1", ex=86_400)
        return True

    # --- worker side -----------------------------------------------------------

    async def consume_next(
        self, poll_timeout_s: float = 5.0
    ) -> tuple[JobRecord, JobContext] | None:
        """Pop one queued job (worker coroutine)."""
        result = await self._conn().blpop([self._QUEUE_KEY], timeout=int(poll_timeout_s))
        if result is None:
            return None
        _queue, payload = result
        spec = json.loads(payload)
        raw = await self._conn().hget(self._HASH_PREFIX + spec["job_id"], "data")
        if raw is None:
            return None
        record = self._deserialize(raw)
        record.state = JobState.RUNNING.value
        await self._write(record)

        cancel_event = threading.Event()

        async def _watch_cancel() -> None:
            while not cancel_event.is_set():
                if await self._conn().get(self._CANCEL_PREFIX + record.job_id):
                    cancel_event.set()
                    return
                await asyncio.sleep(1.0)

        asyncio.get_running_loop().create_task(_watch_cancel())
        return record, JobContext(record, cancel_event)

    async def _write(self, record: JobRecord) -> None:
        await self._conn().hset(
            self._HASH_PREFIX + record.job_id, "data", self._serialize(record)
        )

    async def run_worker_forever(self) -> None:
        """Worker loop: pop jobs, execute the runner, persist status."""
        logger.info("redis job worker started")
        while True:
            item = await self.consume_next()
            if item is None:
                continue
            record, ctx = item
            try:
                await asyncio.to_thread(self._runner, record, ctx)
            except JobCancelledError:
                record.state = JobState.CANCELLED.value
            except Exception as exc:  # noqa: BLE001
                record.state = JobState.FAILED.value
                record.message = f"{type(exc).__name__}: {exc}"
                record.append_log(record.message)
            else:
                record.state = JobState.DONE.value
                record.progress = 1.0
            await self._write(record)
