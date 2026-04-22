from pathlib import Path

from app.alignment import WordTimestamp
from app.asr.gigaam_service import GigaAMService


class FakeWord:
    text = "hello"
    start = 1
    end = 2


class FakeTranscriptionResult:
    text = "hello"
    words = [FakeWord()]


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path, *, word_timestamps=False):
        self.calls.append(("transcribe", audio_path, word_timestamps))
        return FakeTranscriptionResult()

    def transcribe_longform(self, audio_path, *, word_timestamps=False):
        self.calls.append(("transcribe_longform", audio_path, word_timestamps))
        return FakeTranscriptionResult()


def make_service(model: FakeModel) -> GigaAMService:
    service = GigaAMService.__new__(GigaAMService)
    service.model_name = "fake"
    service.model = model
    return service


def test_transcribe_uses_regular_model_call_for_short_audio():
    model = FakeModel()
    service = make_service(model)

    text, words = service.transcribe(Path("short.wav"), word_timestamps=True, duration_sec=30.0)

    assert text == "hello"
    assert words == [WordTimestamp(text="hello", start=1.0, end=2.0)]
    assert model.calls == [("transcribe", "short.wav", True)]


def test_transcribe_uses_longform_model_call_for_audio_over_30_seconds():
    model = FakeModel()
    service = make_service(model)

    text, words = service.transcribe(Path("long.wav"), word_timestamps=True, duration_sec=30.1)

    assert text == "hello"
    assert words == [WordTimestamp(text="hello", start=1.0, end=2.0)]
    assert model.calls == [("transcribe_longform", "long.wav", True)]
