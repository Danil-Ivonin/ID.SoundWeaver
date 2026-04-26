from datetime import timedelta

from fastapi import APIRouter

from app.async_utils import maybe_await
from app.dependencies import AppDependencies
from app.schemas import CreateUploadRequest, CreateUploadResponse


def build_uploads_router(deps: AppDependencies) -> APIRouter:
    router = APIRouter(prefix="/v1/uploads", tags=["uploads"])

    @router.post("", response_model=CreateUploadResponse)
    async def create_upload(payload: CreateUploadRequest) -> CreateUploadResponse:
        upload_id = deps.new_id()
        object_key = deps.storage.build_object_key(upload_id, payload.filename)
        expires_at = deps.now() + timedelta(seconds=deps.settings.presigned_upload_ttl_sec)
        await maybe_await(deps.upload_repo.create_upload(
            upload_id=upload_id,
            object_key=object_key,
            filename=payload.filename,
            content_type=payload.content_type,
            size_bytes=None,
            expires_at=expires_at,
        ))
        upload_url = await maybe_await(deps.storage.create_presigned_put_url(object_key))
        return CreateUploadResponse(
            upload_id=upload_id,
            upload_url=upload_url,
            expires_in_sec=deps.settings.presigned_upload_ttl_sec,
        )

    return router
