from pathlib import Path

import pytest
import torch

from app.asr import gigaam_service
from app.alignment import WordTimestamp
from app.asr.gigaam_service import GigaAMEmotionService, GigaAMService


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


def test_get_service_reuses_cached_model_instance(monkeypatch):
    loaded = []

    class FakeLoadedModel(FakeModel):
        pass

    def fake_load_model(model_name):
        loaded.append(model_name)
        return FakeLoadedModel()

    monkeypatch.setattr("app.asr.gigaam_service.gigaam.load_model", fake_load_model)
    GigaAMService.clear_cache()

    first = GigaAMService.get_cached("model_a")
    second = GigaAMService.get_cached("model_a")

    assert first is second
    assert loaded == ["model_a"]


def test_get_emotion_service_reuses_cached_model_instance(monkeypatch):
    loaded = []

    class FakeEmotionModel:
        def get_probs(self, _audio_path):
            return {"neutral": 1.0}

    def fake_load_model(model_name):
        loaded.append(model_name)
        return FakeEmotionModel()

    monkeypatch.setattr("app.asr.gigaam_service.gigaam.load_model", fake_load_model)
    GigaAMEmotionService.clear_cache()

    first = GigaAMEmotionService.get_cached()
    second = GigaAMEmotionService.get_cached()

    assert first is second
    assert loaded == ["emo"]


def test_get_emotion_probs_chunks_long_audio_and_averages_results(monkeypatch, tmp_path):
    saved_chunks = []

    class FakeEmotionModel:
        def get_probs(self, audio_path):
            chunk_name = Path(audio_path).stem
            if chunk_name == "chunk_0":
                return {"neutral": 0.8, "happy": 0.2}
            if chunk_name == "chunk_1":
                return {"neutral": 0.1, "happy": 0.9}
            return {"neutral": 0.6, "happy": 0.4}

    service = GigaAMEmotionService.__new__(GigaAMEmotionService)
    service.model = FakeEmotionModel()

    waveform = torch.ones(1, 61 * 16_000)
    monkeypatch.setattr(gigaam_service.torchaudio, "load", lambda _path: (waveform, 16_000))

    def fake_save(path, chunk_waveform, sample_rate):
        saved_chunks.append((Path(path).name, chunk_waveform.shape[1], sample_rate))
        Path(path).write_bytes(b"chunk")

    monkeypatch.setattr(gigaam_service.torchaudio, "save", fake_save)
    monkeypatch.setattr(gigaam_service, "clear_cuda_state", lambda: None)

    probs = service.get_probs(tmp_path / "long.wav", duration_sec=61.0)

    assert saved_chunks == [
        ("chunk_0.wav", 30 * 16_000, 16_000),
        ("chunk_1.wav", 30 * 16_000, 16_000),
        ("chunk_2.wav", 1 * 16_000, 16_000),
    ]
    assert probs == pytest.approx(
        {"neutral": 27.6 / 61.0, "happy": 33.4 / 61.0},
        abs=1e-6,
    )


def test_clear_cuda_state_tolerates_missing_cuda(monkeypatch):
    monkeypatch.setattr("app.asr.gigaam_service.gc.collect", lambda: None)
    monkeypatch.setattr("app.asr.gigaam_service.torch.cuda.is_available", lambda: False)

    GigaAMService.clear_cuda_state()
