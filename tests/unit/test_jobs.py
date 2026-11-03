"""Job queue & fine-tune service behaviour (no GPU, no seed-vc required)."""

from __future__ import annotations

import threading

import pytest

from rafidain.core.config import Settings
from rafidain.core.errors import DependencyMissingError, JobError
from rafidain.domain.value_objects import JobState
from rafidain.infrastructure.jobs import InProcessJobQueue, JobContext


def test_job_context_reports_progress():
    record = type("R", (), {})()
    from rafidain.domain.entities import JobRecord

    record = JobRecord(kind="test")
    ctx = JobContext(record, threading.Event())
    ctx.set_progress(0.5, "halfway")
    assert record.progress == 0.5
    assert record.message == "halfway"
    ctx.log("line one")
    assert list(record.logs) == ["line one"]


async def test_inprocess_queue_success_path():
    def runner(record, ctx):
        ctx.log("starting")
        for step in range(5):
            ctx.set_progress(step / 5)
        record.result = {"ok": True}

    queue = InProcessJobQueue(runner)
    job_id = await queue.submit("test", {"x": 1})
    # wait for the asyncio task to complete
    for _ in range(100):
        record = await queue.get(job_id)
        if record.state == JobState.DONE.value:
            break
        await __import__("asyncio").sleep(0.01)
    assert record.state == JobState.DONE.value
    assert record.progress == 1.0
    assert record.result == {"ok": True}
    assert "starting" in list(record.logs)


async def test_inprocess_queue_failure_is_recorded():
    def runner(record, ctx):
        raise RuntimeError("boom")

    queue = InProcessJobQueue(runner)
    job_id = await queue.submit("test", {})
    for _ in range(100):
        record = await queue.get(job_id)
        if record.state == JobState.FAILED.value:
            break
        await __import__("asyncio").sleep(0.01)
    assert record.state == JobState.FAILED.value
    assert "RuntimeError" in record.message


async def test_inprocess_queue_cancellation(tmp_path):
    from rafidain.infrastructure.jobs import JobCancelledError

    started = threading.Event()

    def runner(record, ctx):
        started.set()
        import time as _time

        deadline = _time.time() + 5
        while _time.time() < deadline:
            ctx.check_cancelled()  # raises JobCancelledError when cancelled
            _time.sleep(0.01)

    queue = InProcessJobQueue(runner)
    job_id = await queue.submit("test", {})
    while not started.is_set():
        await __import__("asyncio").sleep(0.01)
    assert await queue.cancel(job_id) is True
    for _ in range(200):
        record = await queue.get(job_id)
        if record.state == JobState.CANCELLED.value:
            break
        await __import__("asyncio").sleep(0.01)
    assert record.state == JobState.CANCELLED.value
    _ = JobCancelledError


async def test_finetune_submit_requires_dataset(tmp_path):
    from rafidain.application.finetune_service import FinetuneParams, FinetuneService

    settings = Settings(_env_file=None, engine={"impl": "mock"}, data_dir=tmp_path / "d")
    service = FinetuneService(settings)
    with pytest.raises(JobError):
        await service.submit(FinetuneParams(dataset_dir=str(tmp_path / "nope")))


async def test_finetune_submit_requires_seedvc_repo(tmp_path, monkeypatch):
    from rafidain.application.finetune_service import FinetuneParams, FinetuneService

    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "a.wav").write_bytes(b"x" * 100)
    # hermetic: point SEED_VC_ROOT at a missing dir even when third_party/seed-vc
    # really exists (e.g. right after running the install scripts, which clone it)
    monkeypatch.setenv("SEED_VC_ROOT", str(tmp_path / "no-seed-vc-here"))
    settings = Settings(_env_file=None, engine={"impl": "mock"}, data_dir=tmp_path / "d")
    service = FinetuneService(settings)
    # no seed-vc repo at the resolved root -> clear dependency error, not a crash
    with pytest.raises(DependencyMissingError):
        await service.submit(FinetuneParams(dataset_dir=str(dataset)))
