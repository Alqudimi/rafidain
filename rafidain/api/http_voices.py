"""Voice profile management (registration of target voices)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse

from rafidain.api.deps import AppComponents, get_components, require_api_key

router = APIRouter(
    prefix="/api/v1/voices", tags=["voices"], dependencies=[Depends(require_api_key)]
)


@router.post("", status_code=201, summary="Register a target voice from reference audio")
async def register_voice(
    file: UploadFile = File(..., description="1-30 s of clean reference audio"),
    name: str = Form(default=""),
    notes: str = Form(default=""),
    components: AppComponents = Depends(get_components),
) -> JSONResponse:
    audio = await file.read()
    voice = await components.voice_service.register(
        name=name or (file.filename or "voice"), audio=audio, notes=notes
    )
    return JSONResponse(status_code=201, content=voice.as_dict())


@router.get("", summary="List registered voices")
async def list_voices(components: AppComponents = Depends(get_components)) -> dict:
    return {"voices": [v.as_dict() for v in await components.voice_service.list()]}


@router.get("/{voice_id}", summary="Fetch one voice")
async def get_voice(
    voice_id: str, components: AppComponents = Depends(get_components)
) -> dict:
    voice = await components.voice_service.get(voice_id)
    return voice.as_dict()


@router.delete("/{voice_id}", summary="Delete a voice")
async def delete_voice(
    voice_id: str, components: AppComponents = Depends(get_components)
) -> dict:
    await components.voice_service.delete(voice_id)
    return {"deleted": voice_id}
