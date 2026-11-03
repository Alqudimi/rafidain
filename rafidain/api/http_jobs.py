"""Fine-tuning job endpoints (submit, poll, cancel)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from rafidain.api.deps import AppComponents, get_components, require_api_key
from rafidain.application.finetune_service import FinetuneParams
from rafidain.core.errors import JobError, JobNotFoundError

router = APIRouter(
    prefix="/api/v1/finetune",
    tags=["fine-tuning"],
    dependencies=[Depends(require_api_key)],
)


class FinetuneRequest(BaseModel):
    dataset_dir: str = Field(..., description="directory of 1-30 s utterances")
    run_name: str = Field(default="iraqi_run", description="runs/<run_name> output dir")
    max_steps: int = Field(default=1000, ge=100)
    max_epochs: int = Field(default=1000, ge=1)
    save_every: int = Field(default=500, ge=50)
    batch_size: int = Field(default=2, ge=1)
    num_workers: int = Field(default=0, ge=0)
    gpu: int = Field(default=0, ge=0)
    prepare: bool = Field(default=True, description="run dataset preparation first")
    base_checkpoint: str | None = Field(
        default=None, description="start from this checkpoint (default: engine ckpt)"
    )


@router.post("/jobs", status_code=202, summary="Submit a fine-tuning job")
async def submit_finetune(
    payload: FinetuneRequest, components: AppComponents = Depends(get_components)
) -> dict:
    params = FinetuneParams(**payload.model_dump())
    job_id = await components.finetune_service.submit(params)
    return {"job_id": job_id, "status_url": f"/api/v1/finetune/jobs/{job_id}"}


@router.get("/jobs/{job_id}", summary="Job status, progress & logs")
async def job_status(
    job_id: str, components: AppComponents = Depends(get_components)
) -> dict:
    record = await components.finetune_service.status(job_id)
    if record is None:
        raise JobNotFoundError(f"no job {job_id!r}")
    return record.as_dict()


@router.post("/jobs/{job_id}/cancel", summary="Cancel a running job")
async def cancel_job(
    job_id: str, components: AppComponents = Depends(get_components)
) -> dict:
    record = await components.finetune_service.status(job_id)
    if record is None:
        raise JobNotFoundError(f"no job {job_id!r}")
    if record.state in ("done", "failed", "cancelled"):
        raise JobError(f"job already {record.state}")
    ok = await components.finetune_service.cancel(job_id)
    return {"cancelled": ok, "job_id": job_id}
