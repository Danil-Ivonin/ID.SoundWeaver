import pytest
from pydantic import ValidationError

from app.schemas import CreateTranscriptionRequest, CreateUploadRequest


def test_create_upload_accepts_supported_audio_type():
    request = CreateUploadRequest(
        filename="meeting.wav",
        content_type="audio/wav",
    )

    assert request.filename == "meeting.wav"


def test_create_upload_rejects_unsupported_audio_type():
    with pytest.raises(ValidationError):
        CreateUploadRequest(filename="notes.txt", content_type="text/plain")


def test_transcription_rejects_conflicting_speaker_params():
    with pytest.raises(ValidationError):
        CreateTranscriptionRequest(
            upload_id="upload_1",
            diarization=True,
            num_speakers=2,
            min_speakers=1,
            max_speakers=3,
        )


def test_transcription_allows_no_diarization_without_speaker_params():
    request = CreateTranscriptionRequest(upload_id="upload_1", diarization=False)

    assert request.diarization is False
