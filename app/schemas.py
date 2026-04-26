from datetime import datetime

from pydantic import BaseModel, Field, model_validator

SUPPORTED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/flac",
    "audio/mp4",
    "audio/x-m4a",
}


class CreateUploadRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_type: str

    @model_validator(mode="after")
    def validate_content_type(self) -> "CreateUploadRequest":
        if self.content_type not in SUPPORTED_CONTENT_TYPES:
            raise ValueError("Unsupported audio content type. Expected one of: " + ", ".join(SUPPORTED_CONTENT_TYPES))
        return self


class CreateUploadResponse(BaseModel):
    upload_id: str
    upload_url: str
    method: str = "PUT"
    expires_in_sec: int


class CreateTranscriptionRequest(BaseModel):
    upload_id: str
    diarization: bool = False
    num_speakers: int | None = Field(default=None, ge=1)
    min_speakers: int | None = Field(default=None, ge=1)
    max_speakers: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_speaker_params(self) -> "CreateTranscriptionRequest":
        if self.num_speakers is not None and (
            self.min_speakers is not None or self.max_speakers is not None
        ):
            raise ValueError("num_speakers cannot be combined with min_speakers or max_speakers")
        if self.min_speakers is not None and self.max_speakers is not None:
            if self.min_speakers > self.max_speakers:
                raise ValueError("min_speakers cannot be greater than max_speakers")
        return self


class CreateTranscriptionResponse(BaseModel):
    job_id: str
    status: str


class ErrorBody(BaseModel):
    code: str
    message: str


class Utterance(BaseModel):
    speaker: str
    start: float
    end: float
    text: str


class TranscriptionStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    duration_sec: float | None = None
    text: str | None = None
    utterances: list[Utterance] | None = None
    diagnostics: dict | None = None
    error: ErrorBody | None = None
