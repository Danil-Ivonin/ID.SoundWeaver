from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch

from app.alignment import build_utterances
from app.asr.gigaam_service import GigaAMEmotionService, GigaAMService
from app.audio.processing import normalize_audio
from app.db.repositories import get_job, mark_job_completed, mark_job_failed, mark_job_processing
from app.db.session import SessionLocal
from app.diarization.pyannote_service import PyannoteDiarizationService
from app.errors import AppError, ErrorCode
from app.settings import get_settings
from app.tasks.celery_app import celery_app


class TorchaudioProcessor:
    def __init__(self, max_duration_sec: int) -> None:
        self.max_duration_sec = max_duration_sec

    def normalize(self, input_path: Path, output_path: Path):
        return normalize_audio(input_path, output_path, self.max_duration_sec)


class TranscriptionProcessor:
    def __init__(
        self,
        *,
        storage,
        audio_processor,
        asr,
        diarization,
        emotion_model,
        work_dir: Path,
        clock=perf_counter,
    ) -> None:
        self.storage = storage
        self.audio_processor = audio_processor
        self.asr = asr
        self.diarization = diarization
        self.emotion_model = emotion_model
        self.work_dir = work_dir
        self.clock = clock

    def process(
        self,
        *,
        object_key: str,
        diarization: bool,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> dict:
        total_started_at = self.clock()
        diarization_duration_sec = 0.0
        input_path = self.work_dir / "input_audio"
        normalized_path = self.work_dir / "normalized.wav"
        self.storage.download_to_file(object_key, input_path)
        audio = self.audio_processor.normalize(input_path, normalized_path)

        asr_started_at = self.clock()
        text, words = self.asr.transcribe(audio.path, word_timestamps=diarization)
        asr_duration_sec = self.clock() - asr_started_at
        emotions = self.emotion_model.get_probs(audio.path)

        utterances = []
        if diarization:
            diarization_started_at = self.clock()
            segments = self.diarization.diarize(
                waveform=audio.waveform,
                sample_rate=audio.sample_rate,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            diarization_duration_sec = self.clock() - diarization_started_at
            utterances = build_utterances(words, segments)

        total_processing_sec = self.clock() - total_started_at
        return {
            "duration_sec": audio.duration_sec,
            "text": text,
            "utterances": utterances,
            "diagnostics": {
                "device": get_settings().device,
                "asr_duration_sec": round(asr_duration_sec, 3),
                "diarization_duration_sec": round(diarization_duration_sec, 3),
                "total_processing_sec": round(total_processing_sec, 3),
                "emotions": emotions,
            },
        }


def assert_cuda_available() -> None:
    if not torch.cuda.is_available():
        raise AppError(ErrorCode.CUDA_UNAVAILABLE, "CUDA is not available")


@celery_app.task(name="transcribe_audio")
def transcribe_audio(job_id: str) -> None:
    settings = get_settings()
    assert_cuda_available()
    with SessionLocal() as session:
        job = get_job(session, job_id)
        if job is None:
            return
        mark_job_processing(session, job_id)
        try:
            with TemporaryDirectory() as tmp:
                from app.main import RuntimeStorage

                processor = TranscriptionProcessor(
                    storage=RuntimeStorage(settings),
                    audio_processor=TorchaudioProcessor(settings.max_audio_duration_sec),
                    asr=GigaAMService(settings.gigaam_model),
                    diarization=PyannoteDiarizationService(
                        settings.pyannote_model,
                        settings.hf_token,
                        settings.device,
                    ),
                    emotion_model=GigaAMEmotionService(),
                    work_dir=Path(tmp),
                )
                result = processor.process(
                    object_key=job.upload.object_key,
                    diarization=job.diarization,
                    num_speakers=job.num_speakers,
                    min_speakers=job.min_speakers,
                    max_speakers=job.max_speakers,
                )
            mark_job_completed(session, job_id=job_id, **result)
        except AppError as exc:
            mark_job_failed(session, job_id, exc.code.value, exc.message)
        except Exception as exc:
            mark_job_failed(session, job_id, ErrorCode.INTERNAL_ERROR.value, str(exc))
