from datetime import timedelta
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.async_utils import maybe_await
from app.errors import ErrorCode
from app.schemas import CreateUploadRequest, CreateUploadResponse
from app.settings import Settings, get_settings

router = APIRouter(prefix="/v1/uploads", tags=["uploads"])
SettingsDep = Depends(get_settings)


def new_id() -> str:
    return uuid4().hex


@router.post("", response_model=CreateUploadResponse)
async def create_upload(
    payload: CreateUploadRequest,
    request: Request,
    settings: Settings = SettingsDep,
) -> CreateUploadResponse:
    if payload.size_bytes > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={"code": ErrorCode.UPLOAD_TOO_LARGE, "message": "Upload is too large"},
        )

    upload_id = request.app.state.new_id() if hasattr(request.app.state, "new_id") else new_id()
    object_key = request.app.state.storage.build_object_key(upload_id, payload.filename)
    expires_at = request.app.state.now() + timedelta(seconds=settings.presigned_upload_ttl_sec)
    await maybe_await(request.app.state.upload_repo.create_upload(
        upload_id=upload_id,
        object_key=object_key,
        filename=payload.filename,
        content_type=payload.content_type,
        size_bytes=payload.size_bytes,
        expires_at=expires_at,
    ))
    upload_url = await maybe_await(request.app.state.storage.create_presigned_put_url(object_key))
    return CreateUploadResponse(
        upload_id=upload_id,
        upload_url=upload_url,
        expires_in_sec=settings.presigned_upload_ttl_sec,
    )
