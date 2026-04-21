from dataclasses import dataclass
from enum import StrEnum


class ErrorCode(StrEnum):
    UPLOAD_TOO_LARGE = "upload_too_large"
    UNSUPPORTED_CONTENT_TYPE = "unsupported_content_type"
    UPLOAD_NOT_FOUND = "upload_not_found"
    JOB_NOT_FOUND = "job_not_found"
    UPLOAD_NOT_COMPLETED = "upload_not_completed"
    INVALID_SPEAKER_PARAMS = "invalid_speaker_params"
    AUDIO_DECODE_FAILED = "audio_decode_failed"
    AUDIO_DURATION_EXCEEDED = "audio_duration_exceeded"
    CUDA_UNAVAILABLE = "cuda_unavailable"
    ASR_FAILED = "asr_failed"
    DIARIZATION_FAILED = "diarization_failed"
    ALIGNMENT_FAILED = "alignment_failed"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class AppError(Exception):
    code: ErrorCode
    message: str
