from pathlib import Path

from app.alignment import SpeakerSegment, WordTimestamp
from app.tasks.transcription import TranscriptionProcessor


class FakeStorage:
    def download_to_file(self, object_key, path):
        Path(path).write_bytes(b"fake")


class FakeAudio:
    duration_sec = 1.0
    sample_rate = 16000
    waveform = "waveform"
    path = Path("normalized.wav")


class FakeAudioProcessor:
    def normalize(self, input_path, output_path):
        return FakeAudio()


class FakeAsr:
    def transcribe(self, audio_path, *, word_timestamps):
        return "hello world", [
            WordTimestamp(text="hello", start=0.0, end=0.5),
            WordTimestamp(text="world", start=0.5, end=1.0),
        ]


class FakeDiarization:
    def diarize(self, **kwargs):
        return [SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=1.0)]


class FakeEmotionModel:
    def __init__(self):
        self.audio_path = None

    def get_probs(self, audio_path):
        self.audio_path = audio_path
        return {"neutral": 0.7, "happy": 0.3}


class FakeClock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


def test_processor_returns_empty_utterances_without_diarization(tmp_path):
    emotion_model = FakeEmotionModel()
    processor = TranscriptionProcessor(
        storage=FakeStorage(),
        audio_processor=FakeAudioProcessor(),
        asr=FakeAsr(),
        diarization=FakeDiarization(),
        emotion_model=emotion_model,
        work_dir=tmp_path,
        clock=FakeClock([0.0, 1.0, 3.4, 4.5]),
    )

    result = processor.process(
        object_key="key",
        diarization=False,
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
    )

    assert result["text"] == "hello world"
    assert result["utterances"] == []
    assert result["diagnostics"]["asr_duration_sec"] == 2.4
    assert result["diagnostics"]["diarization_duration_sec"] == 0.0
    assert result["diagnostics"]["total_processing_sec"] == 4.5
    assert result["diagnostics"]["emotions"] == {"neutral": 0.7, "happy": 0.3}
    assert emotion_model.audio_path == FakeAudio.path


def test_processor_returns_speaker_utterances_with_diarization(tmp_path):
    processor = TranscriptionProcessor(
        storage=FakeStorage(),
        audio_processor=FakeAudioProcessor(),
        asr=FakeAsr(),
        diarization=FakeDiarization(),
        emotion_model=FakeEmotionModel(),
        work_dir=tmp_path,
        clock=FakeClock([0.0, 1.0, 3.4, 3.5, 5.3, 5.5]),
    )

    result = processor.process(
        object_key="key",
        diarization=True,
        num_speakers=None,
        min_speakers=1,
        max_speakers=5,
    )

    assert result["utterances"][0]["speaker"] == "SPEAKER_00"
    assert result["utterances"][0]["text"] == "hello world"
    assert result["diagnostics"]["asr_duration_sec"] == 2.4
    assert result["diagnostics"]["diarization_duration_sec"] == 1.8
    assert result["diagnostics"]["total_processing_sec"] == 5.5
