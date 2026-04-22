from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status

from app.errors import ErrorCode
from app.schemas import (
    CreateTranscriptionRequest,
    CreateTranscriptionResponse,
    ErrorBody,
    TranscriptionStatusResponse,
)

router = APIRouter(prefix="/v1/transcriptions", tags=["transcriptions"])


def new_id() -> str:
    return uuid4().hex


@router.post("", response_model=CreateTranscriptionResponse)
def create_transcription(payload: CreateTranscriptionRequest, request: Request):
    upload = request.app.state.job_repo.get_upload(payload.upload_id)
    if upload is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": ErrorCode.UPLOAD_NOT_FOUND, "message": "Upload not found"},
        )

    job_id = request.app.state.new_id() if hasattr(request.app.state, "new_id") else new_id()
    job = request.app.state.job_repo.create_job(
        job_id=job_id,
        upload_id=payload.upload_id,
        diarization=payload.diarization,
        num_speakers=payload.num_speakers,
        min_speakers=payload.min_speakers,
        max_speakers=payload.max_speakers,
    )
    request.app.state.queue.enqueue(job.id)
    return CreateTranscriptionResponse(job_id=job.id, status=job.status)


@router.get("/{job_id}", response_model=TranscriptionStatusResponse)
def get_transcription(job_id: str, request: Request):
    job = request.app.state.job_repo.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": ErrorCode.JOB_NOT_FOUND, "message": "Job not found"},
        )

    if job.status == "failed":
        return TranscriptionStatusResponse(
            job_id=job.id,
            status=job.status,
            created_at=job.created_at,
            updated_at=job.updated_at,
            error=ErrorBody(
                code=job.error_code or ErrorCode.INTERNAL_ERROR,
                message=job.error_message or "",
            ),
        )

    if job.status == "completed" and job.result is not None:
        return TranscriptionStatusResponse(
            job_id=job.id,
            status=job.status,
            created_at=job.created_at,
            updated_at=job.updated_at,
            duration_sec=job.result.duration_sec,
            text=job.result.text,
            utterances=job.result.utterances,
            diagnostics=job.result.diagnostics,
        )

    return TranscriptionStatusResponse(
        job_id=job.id,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
