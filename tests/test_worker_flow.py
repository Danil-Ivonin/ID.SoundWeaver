import asyncio
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.alignment import SpeakerSegment, WordTimestamp
from app.db.base import Base
from app.db.models import utc_now
from app.db.repositories import create_job, create_upload, get_job, upsert_task_result
from app.tasks import transcription
from app.tasks.transcription import TranscriptionProcessor, build_transcription_child_signatures, merge_asr_chunk_payloads


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
    def __init__(self):
        self.duration_sec = None

    def transcribe(self, audio_path, *, word_timestamps, duration_sec=None):
        self.duration_sec = duration_sec
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
        self.duration_sec = None

    def get_probs(self, audio_path, *, duration_sec=None):
        self.audio_path = audio_path
        self.duration_sec = duration_sec
        return {"neutral": 0.7, "happy": 0.3}


class FakeClock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


def test_processor_returns_empty_utterances_without_diarization(tmp_path):
    emotion_model = FakeEmotionModel()
    asr = FakeAsr()
    processor = TranscriptionProcessor(
        storage=FakeStorage(),
        audio_processor=FakeAudioProcessor(),
        asr=asr,
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
    assert emotion_model.duration_sec == FakeAudio.duration_sec
    assert asr.duration_sec == FakeAudio.duration_sec


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


def test_run_asr_records_model_execution_duration(monkeypatch, tmp_path):
    recorded = {}

    class FakeStorage:
        async def download_to_file(self, object_key, path):
            Path(path).write_bytes(b"fake")

    class FakeAsrService:
        def __init__(self, _model_name):
            pass

        @classmethod
        def get_cached(cls, model_name):
            return cls(model_name)

        def transcribe(self, audio_path, *, word_timestamps, duration_sec):
            return "hello world", [
                WordTimestamp(text="hello", start=0.0, end=0.5),
                WordTimestamp(text="world", start=0.5, end=1.0),
            ]

    class FakePerfCounter:
        def __init__(self, values):
            self.values = iter(values)

        def __call__(self):
            return next(self.values)

    async def fake_record_task_success(job_id, task_type, payload, exec_duration=None):
        recorded["job_id"] = job_id
        recorded["task_type"] = task_type
        recorded["payload"] = payload
        recorded["exec_duration"] = exec_duration

    monkeypatch.setattr(transcription, "assert_cuda_available", lambda: None)
    monkeypatch.setattr(transcription, "AsyncS3Storage", lambda settings: FakeStorage())
    monkeypatch.setattr(transcription, "GigaAMService", FakeAsrService)
    monkeypatch.setattr(transcription, "_record_task_success", fake_record_task_success)
    monkeypatch.setattr(transcription, "perf_counter", FakePerfCounter([10.0, 10.4]))

    result = transcription.run_asr("job_1", "artifacts/job_1/normalized.wav", True, 1.0)

    assert result == {"task_type": "asr", "status": "completed"}
    assert recorded["job_id"] == "job_1"
    assert recorded["task_type"] == "asr"
    assert recorded["exec_duration"] == 0.4
    assert recorded["payload"]["asr_duration_sec"] == 0.4


def test_run_asr_records_chunk_metadata_and_absolute_word_times(monkeypatch):
    recorded = {}

    class FakeStorage:
        async def download_to_file(self, object_key, path):
            Path(path).write_bytes(b"fake")

    class FakeAsrService:
        @classmethod
        def get_cached(cls, model_name):
            return cls()

        def transcribe(self, audio_path, *, word_timestamps, duration_sec):
            return "middle word", [
                WordTimestamp(text="middle", start=1.0, end=1.5),
                WordTimestamp(text="word", start=2.0, end=2.5),
            ]

    async def fake_record_task_success(job_id, task_type, payload, exec_duration=None):
        recorded["job_id"] = job_id
        recorded["task_type"] = task_type
        recorded["payload"] = payload
        recorded["exec_duration"] = exec_duration

    monkeypatch.setattr(transcription, "assert_cuda_available", lambda: None)
    monkeypatch.setattr(transcription, "AsyncS3Storage", lambda settings: FakeStorage())
    monkeypatch.setattr(transcription, "GigaAMService", FakeAsrService)
    monkeypatch.setattr(transcription, "_record_task_success", fake_record_task_success)
    monkeypatch.setattr(transcription, "perf_counter", FakeClock([10.0, 10.4]))

    result = transcription.run_asr(
        "job_1",
        "artifacts/job_1/chunks/000001.wav",
        True,
        30.0,
        chunk_index=1,
        chunk_start_sec=25.0,
        chunk_end_sec=55.0,
    )

    assert result == {"task_type": "asr:000001", "status": "completed"}
    assert recorded["task_type"] == "asr:000001"
    assert recorded["payload"]["chunk_index"] == 1
    assert recorded["payload"]["chunk_start_sec"] == 25.0
    assert recorded["payload"]["chunk_end_sec"] == 55.0
    assert recorded["payload"]["words"] == [
        {"text": "middle", "start": 26.0, "end": 26.5},
        {"text": "word", "start": 27.0, "end": 27.5},
    ]


def test_run_pipeline_records_stage_results_sequentially(monkeypatch, tmp_path):
    stage_calls = []
    recorded = []

    class FakeAudio:
        duration_sec = 61.0
        sample_rate = 16000
        waveform = "waveform"
        path = tmp_path / "normalized.wav"

    class FakeAsr:
        def transcribe(self, audio_path, *, word_timestamps, duration_sec=None):
            stage_calls.append(("asr", audio_path, word_timestamps, duration_sec))
            return "hello world", [
                WordTimestamp(text="hello", start=0.0, end=0.5),
                WordTimestamp(text="world", start=0.5, end=1.0),
            ]

    class FakeEmotionModel:
        def get_probs(self, audio_path, *, duration_sec=None):
            stage_calls.append(("emotions", audio_path, duration_sec))
            return {"neutral": 0.7}

    class FakeDiarization:
        def diarize(self, **kwargs):
            stage_calls.append(("diarization", kwargs["waveform"], kwargs["sample_rate"]))
            return [SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=1.0)]

    async def fake_record_task_success(job_id, task_type, payload, exec_duration=None):
        recorded.append((job_id, task_type, payload, exec_duration))

    monkeypatch.setattr(transcription, "_record_task_success", fake_record_task_success)
    monkeypatch.setattr(transcription, "perf_counter", FakeClock([1.0, 1.4, 2.0, 2.3, 3.0, 3.6]))

    result = transcription.run_transcription_pipeline(
        "job_1",
        audio=FakeAudio(),
        diarization_enabled=True,
        num_speakers=None,
        min_speakers=1,
        max_speakers=3,
        asr=FakeAsr(),
        emotion_model=FakeEmotionModel(),
        diarization_service=FakeDiarization(),
    )

    assert [call[0] for call in stage_calls] == ["asr", "emotions", "diarization"]
    assert stage_calls[1] == ("emotions", tmp_path / "normalized.wav", 61.0)
    assert [item[1] for item in recorded] == ["asr", "emotions", "diarization"]
    assert result["text"] == "hello world"
    assert result["utterances"][0]["speaker"] == "SPEAKER_00"
    assert result["diagnostics"]["asr_duration_sec"] == 0.4
    assert result["diagnostics"]["emotions_duration_sec"] == 0.3
    assert result["diagnostics"]["diarization_duration_sec"] == 0.6


def test_run_pipeline_clears_cuda_after_oom(monkeypatch, tmp_path):
    cleared = []
    failures = []

    class FakeAudio:
        duration_sec = 61.0
        sample_rate = 16000
        waveform = "waveform"
        path = tmp_path / "normalized.wav"

    class ExplodingAsr:
        def transcribe(self, audio_path, *, word_timestamps, duration_sec=None):
            raise RuntimeError("CUDA out of memory while processing audio")

    async def fake_record_task_failure(job_id, task_type, error_code, error_message, exec_duration=None):
        failures.append((job_id, task_type, error_code, error_message, exec_duration))

    monkeypatch.setattr(transcription, "_record_task_failure", fake_record_task_failure)
    monkeypatch.setattr(transcription, "clear_cuda_state", lambda: cleared.append("cleared"))

    with pytest.raises(RuntimeError, match="out of memory"):
        transcription.run_transcription_pipeline(
            "job_oom",
            audio=FakeAudio(),
            diarization_enabled=False,
            num_speakers=None,
            min_speakers=None,
            max_speakers=None,
            asr=ExplodingAsr(),
            emotion_model=FakeEmotionModel(),
            diarization_service=FakeDiarization(),
        )

    assert cleared == ["cleared", "cleared"]
    assert failures == [
        ("job_oom", "asr", "asr_failed", "CUDA out of memory while processing audio", None)
    ]


def test_async_pipeline_records_results_without_nested_run_sync(monkeypatch, tmp_path):
    recorded = []

    class FakeAudio:
        duration_sec = 61.0
        sample_rate = 16000
        waveform = "waveform"
        path = tmp_path / "normalized.wav"

    class FakeAsr:
        def transcribe(self, audio_path, *, word_timestamps, duration_sec=None):
            return "hello world", [
                WordTimestamp(text="hello", start=0.0, end=0.5),
                WordTimestamp(text="world", start=0.5, end=1.0),
            ]

    class FakeEmotionModel:
        def get_probs(self, audio_path, *, duration_sec=None):
            return {"neutral": 0.7}

    class FakeDiarization:
        def diarize(self, **kwargs):
            return [SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=1.0)]

    async def fake_record_task_success(job_id, task_type, payload, exec_duration=None):
        recorded.append((job_id, task_type, payload, exec_duration))

    monkeypatch.setattr(transcription, "_record_task_success", fake_record_task_success)
    monkeypatch.setattr(transcription, "perf_counter", FakeClock([1.0, 1.4, 2.0, 2.3, 3.0, 3.6]))

    result = transcription.run_sync(
        transcription.run_transcription_pipeline_async(
            "job_1",
            audio=FakeAudio(),
            diarization_enabled=True,
            num_speakers=None,
            min_speakers=1,
            max_speakers=3,
            asr=FakeAsr(),
            emotion_model=FakeEmotionModel(),
            diarization_service=FakeDiarization(),
        )
    )

    assert [item[1] for item in recorded] == ["asr", "emotions", "diarization"]
    assert result["text"] == "hello world"


def test_merge_asr_chunk_payloads_deduplicates_overlap_by_midpoint():
    payloads = [
        {
            "chunk_index": 0,
            "chunk_start_sec": 0.0,
            "chunk_end_sec": 30.0,
            "duration_sec": 30.0,
            "asr_duration_sec": 0.4,
            "words": [
                {"text": "first", "start": 24.0, "end": 24.4},
                {"text": "left", "start": 26.9, "end": 27.1},
                {"text": "duplicate", "start": 28.0, "end": 28.2},
            ],
        },
        {
            "chunk_index": 1,
            "chunk_start_sec": 25.0,
            "chunk_end_sec": 55.0,
            "duration_sec": 30.0,
            "asr_duration_sec": 0.5,
            "words": [
                {"text": "duplicate", "start": 26.0, "end": 26.2},
                {"text": "right", "start": 28.0, "end": 28.3},
                {"text": "last", "start": 40.0, "end": 40.5},
            ],
        },
    ]

    merged = merge_asr_chunk_payloads(payloads)

    assert merged["text"] == "first left right last"
    assert merged["duration_sec"] == 55.0
    assert merged["asr_duration_sec"] == 0.9
    assert [word["text"] for word in merged["words"]] == ["first", "left", "right", "last"]


def test_build_transcription_child_signatures_uses_chunked_asr_and_full_file_optional_stages():
    signatures = build_transcription_child_signatures(
        job_id="job_1",
        normalized_object_key="artifacts/job_1/normalized.wav",
        chunks=[
            {
                "index": 0,
                "object_key": "artifacts/job_1/chunks/000000.wav",
                "start_sec": 0.0,
                "end_sec": 30.0,
                "duration_sec": 30.0,
            },
            {
                "index": 1,
                "object_key": "artifacts/job_1/chunks/000001.wav",
                "start_sec": 25.0,
                "end_sec": 55.0,
                "duration_sec": 30.0,
            },
        ],
        diarization=True,
        num_speakers=None,
        min_speakers=1,
        max_speakers=3,
        duration_sec=55.0,
    )

    assert [signature.task for signature in signatures] == [
        "run_asr",
        "run_asr",
        "run_emotions",
        "run_diarization",
    ]
    assert signatures[0].args == ("job_1", "artifacts/job_1/chunks/000000.wav", True, 30.0)
    assert signatures[0].kwargs == {"chunk_index": 0, "chunk_start_sec": 0.0, "chunk_end_sec": 30.0}
    assert signatures[1].kwargs == {"chunk_index": 1, "chunk_start_sec": 25.0, "chunk_end_sec": 55.0}
    assert signatures[2].args == ("job_1", "artifacts/job_1/normalized.wav", 55.0)
    assert signatures[3].args == ("job_1", "artifacts/job_1/normalized.wav", None, 1, 3)


def test_aggregate_transcription_job_merges_chunked_asr_results(monkeypatch):
    async def run():
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        Session = async_sessionmaker(bind=engine, expire_on_commit=False)
        monkeypatch.setattr(transcription, "SessionLocal", Session)

        try:
            async with Session() as session:
                upload = await create_upload(
                    session,
                    upload_id="upload_chunked",
                    object_key="uploads/upload_chunked/audio.wav",
                    filename="audio.wav",
                    content_type="audio/wav",
                    size_bytes=2048,
                    expires_at=utc_now(),
                )
                job = await create_job(
                    session,
                    job_id="job_chunked",
                    upload_id=upload.id,
                    diarization=False,
                    num_speakers=None,
                    min_speakers=None,
                    max_speakers=None,
                )
                await upsert_task_result(
                    session,
                    job_id=job.id,
                    task_type="asr_manifest",
                    status="completed",
                    payload={
                        "duration_sec": 55.0,
                        "chunks": [
                            {"index": 0, "start_sec": 0.0, "end_sec": 30.0},
                            {"index": 1, "start_sec": 25.0, "end_sec": 55.0},
                        ],
                    },
                )
                await upsert_task_result(
                    session,
                    job_id=job.id,
                    task_type="asr:000000",
                    status="completed",
                    payload={
                        "chunk_index": 0,
                        "chunk_start_sec": 0.0,
                        "chunk_end_sec": 30.0,
                        "duration_sec": 30.0,
                        "asr_duration_sec": 0.4,
                        "words": [
                            {"text": "hello", "start": 1.0, "end": 1.2},
                            {"text": "old", "start": 28.0, "end": 28.3},
                        ],
                    },
                    exec_duration=0.4,
                )
                await upsert_task_result(
                    session,
                    job_id=job.id,
                    task_type="asr:000001",
                    status="completed",
                    payload={
                        "chunk_index": 1,
                        "chunk_start_sec": 25.0,
                        "chunk_end_sec": 55.0,
                        "duration_sec": 30.0,
                        "asr_duration_sec": 0.5,
                        "words": [
                            {"text": "new", "start": 28.0, "end": 28.3},
                            {"text": "world", "start": 40.0, "end": 40.2},
                        ],
                    },
                    exec_duration=0.5,
                )
                await upsert_task_result(
                    session,
                    job_id=job.id,
                    task_type="emotions",
                    status="completed",
                    payload={"emotions": {"neutral": 0.8}, "emotions_duration_sec": 0.2},
                    exec_duration=0.2,
                )

            complete = await transcription._aggregate_transcription_job("job_chunked")
            async with Session() as session:
                loaded = await get_job(session, "job_chunked")
        finally:
            await engine.dispose()

        assert complete is True
        assert loaded.status == "completed"
        assert loaded.result.text == "hello new world"
        assert loaded.result.duration_sec == 55.0
        assert loaded.result.diagnostics["asr_duration_sec"] == 0.9
        assert loaded.result.diagnostics["emotions"] == {"neutral": 0.8}

    asyncio.run(run())
