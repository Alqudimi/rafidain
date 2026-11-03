"""Fine-tuning use case: submit, track and cancel Seed-VC training jobs.

The heavy lifting is delegated to ``finetune/run_finetune.py`` executed as a
subprocess — isolating CUDA memory, logs and crashes from the API process.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from rafidain.core.config import Settings, seed_vc_root
from rafidain.core.errors import DependencyMissingError, JobError
from rafidain.core.logging import get_logger
from rafidain.domain.ports import JobQueue
from rafidain.infrastructure.jobs import (
    InProcessJobQueue,
    JobCancelledError,
    JobContext,
    JobRunner,
    RedisJobQueue,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class FinetuneParams:
    """Parameters accepted by the fine-tuning pipeline."""

    dataset_dir: str
    run_name: str = "iraqi_run"
    max_steps: int = 1000
    max_epochs: int = 1000
    save_every: int = 500
    batch_size: int = 2
    num_workers: int = 0
    gpu: int = 0
    prepare: bool = True  # run dataset preparation first
    base_checkpoint: str | None = None  # default: current engine checkpoint
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "dataset_dir": self.dataset_dir,
            "run_name": self.run_name,
            "max_steps": self.max_steps,
            "max_epochs": self.max_epochs,
            "save_every": self.save_every,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "gpu": self.gpu,
            "prepare": self.prepare,
            "base_checkpoint": self.base_checkpoint,
            **self.extra,
        }


class FinetuneService:
    """Application service around the fine-tuning job queue."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._queue = build_job_queue(self._runner, settings)
        self._procs: dict[str, subprocess.Popen] = {}
        self._proc_lock = asyncio.Lock()

    @property
    def queue(self) -> JobQueue:
        return self._queue

    async def submit(self, params: FinetuneParams) -> str:
        dataset = Path(params.dataset_dir).expanduser()
        if not dataset.is_dir():
            raise JobError(f"dataset_dir does not exist: {dataset}")
        if not seed_vc_root().exists():
            raise DependencyMissingError(
                "Seed-VC repository is required for fine-tuning: run `make seedvc`"
            )
        if params.max_steps < 100:
            # Upstream documents 100 steps as the practical minimum.
            raise JobError("max_steps must be >= 100")
        job_id = await self._queue.submit("finetune", params.as_dict())
        logger.info("finetune job submitted", extra={"job_id": job_id})
        return job_id

    async def status(self, job_id: str):
        return await self._queue.get(job_id)

    async def cancel(self, job_id: str) -> bool:
        ok = await self._queue.cancel(job_id)
        proc = self._procs.get(job_id)
        if proc is not None and proc.poll() is None:
            proc.terminate()
        return ok

    # ------------------------------------------------------------------ runner

    def _runner(self, record, ctx: JobContext) -> None:
        """JobRunner: run finetune/run_finetune.py and stream its progress."""
        params = record.params
        script = Path(__file__).resolve().parents[2] / "finetune" / "run_finetune.py"
        if not script.exists():
            raise JobError(f"fine-tuning wrapper not found: {script}")

        cmd = [
            sys.executable,
            str(script),
            "--dataset-dir",
            str(params["dataset_dir"]),
            "--run-name",
            str(params.get("run_name", "iraqi_run")),
            "--max-steps",
            str(params.get("max_steps", 1000)),
            "--max-epochs",
            str(params.get("max_epochs", 1000)),
            "--save-every",
            str(params.get("save_every", 500)),
            "--batch-size",
            str(params.get("batch_size", 2)),
            "--num-workers",
            str(params.get("num_workers", 0)),
            "--gpu",
            str(params.get("gpu", 0)),
        ]
        if params.get("prepare", True):
            cmd.append("--prepare")
        if params.get("base_checkpoint"):
            cmd += ["--pretrained-ckpt", str(params["base_checkpoint"])]

        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")

        ctx.log(f"$ {' '.join(cmd)}")
        record.state = "running"
        proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(script.parent),
            text=True,
            bufsize=1,
        )
        self._procs[record.job_id] = proc
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                ctx.log(line)
                if line.startswith("[progress]"):
                    with contextlib.suppress(IndexError, ValueError):
                        ctx.set_progress(float(line.split()[1]))
                if ctx.cancel_requested:
                    proc.terminate()
                    raise JobCancelledError()
            code = proc.wait()
            if code != 0:
                raise JobError(f"fine-tuning exited with code {code}")
            ctx.set_progress(1.0, "training finished")
            record.result = {
                "runs_dir": str(Path("runs") / str(params.get("run_name", "iraqi_run"))),
                "final_model": str(
                    Path("runs") / str(params.get("run_name", "iraqi_run")) / "ft_model.pth"
                ),
            }
        finally:
            self._procs.pop(record.job_id, None)

    # executed in a worker thread; cancellation path mirrors the queue contract



def build_job_queue(runner: JobRunner, settings: Settings) -> JobQueue:
    """Pick the queue implementation from settings (composition root helper)."""
    if settings.redis_url:
        return RedisJobQueue(settings.redis_url, runner)
    return InProcessJobQueue(runner)
