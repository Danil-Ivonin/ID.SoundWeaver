from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
import torchaudio
from celery.exceptions import Retry

from app.alignment import SpeakerSegment, WordTimestamp, build_utterances
from app.asr.gigaam_service import GigaAMEmotionService, GigaAMService, clear_cuda_state
from app.async_utils import run_sync
from app.audio.processing import normalize_audio
from app.db.repositories import (
    claim_job_processing,
    get_job,
    get_task_results,
    mark_job_completed,
    mark_job_failed,
    upsert_task_result,
)
from app.db.session import SessionLocal
from app.diarization.pyannote_service import PyannoteDiarizationService
from app.errors import AppError, ErrorCode
from app.settings import get_settings
from app.storage.minio import AsyncS3Storage
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
        text, words = self.asr.transcribe(
            audio.path,
            word_timestamps=diarization,
            duration_sec=audio.duration_sec,
        )
        asr_duration_sec = self.clock() - asr_started_at
        emotions = self.emotion_model.get_probs(audio.path, duration_sec=audio.duration_sec)

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


async def _mark_failed(job_id: str, error_code: str, error_message: str) -> None:
    async with SessionLocal() as session:
        await mark_job_failed(session, job_id, error_code, error_message)


async def _record_task_success(
    job_id: str,
    task_type: str,
    payload: dict,
    exec_duration: float | None = None,
) -> None:
    async with SessionLocal() as session:
        await upsert_task_result(
            session,
            job_id=job_id,
            task_type=task_type,
            status="completed",
            payload=payload,
            exec_duration=exec_duration,
        )


async def _record_task_failure(
    job_id: str,
    task_type: str,
    error_code: str,
    error_message: str,
    exec_duration: float | None = None,
) -> None:
    async with SessionLocal() as session:
        await upsert_task_result(
            session,
            job_id=job_id,
            task_type=task_type,
            status="failed",
            payload={},
            exec_duration=exec_duration,
            error_code=error_code,
            error_message=error_message,
        )
        await mark_job_failed(session, job_id, error_code, error_message)


def _word_to_payload(word: WordTimestamp) -> dict:
    return {"text": word.text, "start": word.start, "end": word.end}


def _word_from_payload(payload: dict) -> WordTimestamp:
    return WordTimestamp(
        text=payload["text"],
        start=float(payload["start"]),
        end=float(payload["end"]),
    )


def _segment_to_payload(segment: SpeakerSegment) -> dict:
    return {"speaker": segment.speaker, "start": segment.start, "end": segment.end}


def _segment_from_payload(payload: dict) -> SpeakerSegment:
    return SpeakerSegment(
        speaker=payload["speaker"],
        start=float(payload["start"]),
        end=float(payload["end"]),
    )


def reset_model_caches() -> None:
    GigaAMService.clear_cache()
    GigaAMEmotionService.clear_cache()
    PyannoteDiarizationService.clear_cache()


def _handle_stage_failure(
    job_id: str,
    task_type: str,
    error_code: str,
    error_message: str,
    exec_duration: float | None = None,
) -> None:
    run_sync(_record_task_failure(job_id, task_type, error_code, error_message, exec_duration))
    if "out of memory" in error_message.lower():
        reset_model_caches()
    clear_cuda_state()


async def _handle_stage_failure_async(
    job_id: str,
    task_type: str,
    error_code: str,
    error_message: str,
    exec_duration: float | None = None,
) -> None:
    await _record_task_failure(job_id, task_type, error_code, error_message, exec_duration)
    if "out of memory" in error_message.lower():
        reset_model_caches()
    clear_cuda_state()


def run_transcription_pipeline(
    job_id: str,
    *,
    audio,
    diarization_enabled: bool,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    asr,
    emotion_model,
    diarization_service,
) -> dict:
    task_results: dict[str, dict] = {}

    asr_duration = None
    try:
        started_at = perf_counter()
        text, words = asr.transcribe(
            audio.path,
            word_timestamps=diarization_enabled,
            duration_sec=audio.duration_sec,
        )
        asr_duration = round(perf_counter() - started_at, 3)
        task_results["asr"] = {
            "text": text,
            "words": words,
            "duration_sec": audio.duration_sec,
            "asr_duration_sec": asr_duration,
        }
        run_sync(
            _record_task_success(
                job_id,
                "asr",
                {
                    "text": text,
                    "words": [_word_to_payload(word) for word in words],
                    "duration_sec": audio.duration_sec,
                    "asr_duration_sec": asr_duration,
                },
                asr_duration,
            )
        )
    except AppError as exc:
        _handle_stage_failure(job_id, "asr", exc.code.value, exc.message, asr_duration)
        raise
    except Exception as exc:
        _handle_stage_failure(job_id, "asr", ErrorCode.ASR_FAILED.value, str(exc), asr_duration)
        raise
    finally:
        clear_cuda_state()

    emotions_duration = None
    try:
        started_at = perf_counter()
        emotions = emotion_model.get_probs(audio.path, duration_sec=audio.duration_sec)
        emotions_duration = round(perf_counter() - started_at, 3)
        task_results["emotions"] = {
            "emotions": emotions,
            "emotions_duration_sec": emotions_duration,
        }
        run_sync(
            _record_task_success(
                job_id,
                "emotions",
                {
                    "emotions": emotions,
                    "emotions_duration_sec": emotions_duration,
                },
                emotions_duration,
            )
        )
    except AppError as exc:
        _handle_stage_failure(job_id, "emotions", exc.code.value, exc.message, emotions_duration)
        raise
    except Exception as exc:
        _handle_stage_failure(job_id, "emotions", ErrorCode.INTERNAL_ERROR.value, str(exc), emotions_duration)
        raise
    finally:
        clear_cuda_state()

    segments: list[SpeakerSegment] = []
    diarization_duration = 0.0
    if diarization_enabled:
        try:
            started_at = perf_counter()
            segments = diarization_service.diarize(
                waveform=audio.waveform,
                sample_rate=audio.sample_rate,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            diarization_duration = round(perf_counter() - started_at, 3)
            task_results["diarization"] = {
                "segments": segments,
                "diarization_duration_sec": diarization_duration,
            }
            run_sync(
                _record_task_success(
                    job_id,
                    "diarization",
                    {
                        "segments": [_segment_to_payload(segment) for segment in segments],
                        "diarization_duration_sec": diarization_duration,
                    },
                    diarization_duration,
                )
            )
        except AppError as exc:
            _handle_stage_failure(job_id, "diarization", exc.code.value, exc.message, diarization_duration or None)
            raise
        except Exception as exc:
            _handle_stage_failure(
                job_id,
                "diarization",
                ErrorCode.DIARIZATION_FAILED.value,
                str(exc),
                diarization_duration or None,
            )
            raise
        finally:
            clear_cuda_state()

    utterances = build_utterances(task_results["asr"]["words"], segments) if segments else []
    return {
        "duration_sec": task_results["asr"]["duration_sec"],
        "text": task_results["asr"]["text"],
        "utterances": utterances,
        "diagnostics": {
            "device": get_settings().device,
            "asr_duration_sec": task_results["asr"]["asr_duration_sec"],
            "diarization_duration_sec": diarization_duration,
            "emotions_duration_sec": task_results["emotions"]["emotions_duration_sec"],
            "emotions": task_results["emotions"]["emotions"],
        },
    }


async def run_transcription_pipeline_async(
    job_id: str,
    *,
    audio,
    diarization_enabled: bool,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    asr,
    emotion_model,
    diarization_service,
) -> dict:
    task_results: dict[str, dict] = {}

    asr_duration = None
    try:
        started_at = perf_counter()
        text, words = asr.transcribe(
            audio.path,
            word_timestamps=diarization_enabled,
            duration_sec=audio.duration_sec,
        )
        asr_duration = round(perf_counter() - started_at, 3)
        task_results["asr"] = {
            "text": text,
            "words": words,
            "duration_sec": audio.duration_sec,
            "asr_duration_sec": asr_duration,
        }
        await _record_task_success(
            job_id,
            "asr",
            {
                "text": text,
                "words": [_word_to_payload(word) for word in words],
                "duration_sec": audio.duration_sec,
                "asr_duration_sec": asr_duration,
            },
            asr_duration,
        )
    except AppError as exc:
        await _handle_stage_failure_async(job_id, "asr", exc.code.value, exc.message, asr_duration)
        raise
    except Exception as exc:
        await _handle_stage_failure_async(job_id, "asr", ErrorCode.ASR_FAILED.value, str(exc), asr_duration)
        raise
    finally:
        clear_cuda_state()

    emotions_duration = None
    try:
        started_at = perf_counter()
        emotions = emotion_model.get_probs(audio.path, duration_sec=audio.duration_sec)
        emotions_duration = round(perf_counter() - started_at, 3)
        task_results["emotions"] = {
            "emotions": emotions,
            "emotions_duration_sec": emotions_duration,
        }
        await _record_task_success(
            job_id,
            "emotions",
            {
                "emotions": emotions,
                "emotions_duration_sec": emotions_duration,
            },
            emotions_duration,
        )
    except AppError as exc:
        await _handle_stage_failure_async(job_id, "emotions", exc.code.value, exc.message, emotions_duration)
        raise
    except Exception as exc:
        await _handle_stage_failure_async(
            job_id,
            "emotions",
            ErrorCode.INTERNAL_ERROR.value,
            str(exc),
            emotions_duration,
        )
        raise
    finally:
        clear_cuda_state()

    segments: list[SpeakerSegment] = []
    diarization_duration = 0.0
    if diarization_enabled:
        try:
            started_at = perf_counter()
            segments = diarization_service.diarize(
                waveform=audio.waveform,
                sample_rate=audio.sample_rate,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            diarization_duration = round(perf_counter() - started_at, 3)
            task_results["diarization"] = {
                "segments": segments,
                "diarization_duration_sec": diarization_duration,
            }
            await _record_task_success(
                job_id,
                "diarization",
                {
                    "segments": [_segment_to_payload(segment) for segment in segments],
                    "diarization_duration_sec": diarization_duration,
                },
                diarization_duration,
            )
        except AppError as exc:
            await _handle_stage_failure_async(
                job_id,
                "diarization",
                exc.code.value,
                exc.message,
                diarization_duration or None,
            )
            raise
        except Exception as exc:
            await _handle_stage_failure_async(
                job_id,
                "diarization",
                ErrorCode.DIARIZATION_FAILED.value,
                str(exc),
                diarization_duration or None,
            )
            raise
        finally:
            clear_cuda_state()

    utterances = build_utterances(task_results["asr"]["words"], segments) if segments else []
    return {
        "duration_sec": task_results["asr"]["duration_sec"],
        "text": task_results["asr"]["text"],
        "utterances": utterances,
        "diagnostics": {
            "device": get_settings().device,
            "asr_duration_sec": task_results["asr"]["asr_duration_sec"],
            "diarization_duration_sec": diarization_duration,
            "emotions_duration_sec": task_results["emotions"]["emotions_duration_sec"],
            "emotions": task_results["emotions"]["emotions"],
        },
    }


async def _prepare_transcription_job(job_id: str) -> None:
    settings = get_settings()
    async with SessionLocal() as session:
        claimed = await claim_job_processing(session, job_id)
        if not claimed:
            return
        job = await get_job(session, job_id)
        if job is None:
            return

        object_key = job.upload.object_key
        diarization = job.diarization
        num_speakers = job.num_speakers
        min_speakers = job.min_speakers
        max_speakers = job.max_speakers

    storage = AsyncS3Storage(settings)
    with TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        input_path = work_dir / "input_audio"
        normalized_path = work_dir / "normalized.wav"
        await storage.download_to_file(object_key, input_path)
        audio = normalize_audio(input_path, normalized_path, settings.max_audio_duration_sec)
        result = await run_transcription_pipeline_async(
            job_id,
            audio=audio,
            diarization_enabled=diarization,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            asr=GigaAMService.get_cached(settings.gigaam_model),
            emotion_model=GigaAMEmotionService.get_cached(),
            diarization_service=(
                PyannoteDiarizationService.get_cached(
                    settings.pyannote_model,
                    settings.hf_token,
                    settings.device,
                )
                if diarization
                else None
            ),
        )
    async with SessionLocal() as session:
        await mark_job_completed(
            session,
            job_id=job_id,
            duration_sec=result["duration_sec"],
            text=result["text"],
            utterances=result["utterances"],
            diagnostics=result["diagnostics"],
        )


@celery_app.task(name="prepare_transcription_job")
def prepare_transcription_job(job_id: str) -> None:
    try:
        run_sync(_prepare_transcription_job(job_id))
    except AppError as exc:
        run_sync(_mark_failed(job_id, exc.code.value, exc.message))
    except Exception as exc:
        run_sync(_mark_failed(job_id, ErrorCode.INTERNAL_ERROR.value, str(exc)))


@celery_app.task(name="run_asr")
def run_asr(job_id: str, normalized_object_key: str, word_timestamps: bool, duration_sec: float) -> dict:
    task_type = "asr"
    exec_duration = None
    try:
        assert_cuda_available()
        settings = get_settings()
        storage = AsyncS3Storage(settings)
        with TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "normalized.wav"
            run_sync(storage.download_to_file(normalized_object_key, audio_path))
            started_at = perf_counter()
            text, words = GigaAMService.get_cached(settings.gigaam_model).transcribe(
                audio_path,
                word_timestamps=word_timestamps,
                duration_sec=duration_sec,
            )
            exec_duration = round(perf_counter() - started_at, 3)
        payload = {
            "text": text,
            "words": [_word_to_payload(word) for word in words],
            "duration_sec": duration_sec,
            "asr_duration_sec": exec_duration,
        }
        run_sync(_record_task_success(job_id, task_type, payload, exec_duration))
        return {"task_type": task_type, "status": "completed"}
    except AppError as exc:
        _handle_stage_failure(job_id, task_type, exc.code.value, exc.message, exec_duration)
        raise
    except Exception as exc:
        _handle_stage_failure(job_id, task_type, ErrorCode.ASR_FAILED.value, str(exc), exec_duration)
        raise


@celery_app.task(name="run_diarization")
def run_diarization(
    job_id: str,
    normalized_object_key: str,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> dict:
    task_type = "diarization"
    exec_duration = None
    try:
        assert_cuda_available()
        settings = get_settings()
        storage = AsyncS3Storage(settings)
        with TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "normalized.wav"
            run_sync(storage.download_to_file(normalized_object_key, audio_path))
            waveform, sample_rate = torchaudio.load(str(audio_path))
            started_at = perf_counter()
            segments = PyannoteDiarizationService.get_cached(
                settings.pyannote_model,
                settings.hf_token,
                settings.device,
            ).diarize(
                waveform=waveform,
                sample_rate=sample_rate,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            exec_duration = round(perf_counter() - started_at, 3)
        payload = {
            "segments": [_segment_to_payload(segment) for segment in segments],
            "diarization_duration_sec": exec_duration,
        }
        run_sync(_record_task_success(job_id, task_type, payload, exec_duration))
        return {"task_type": task_type, "status": "completed"}
    except AppError as exc:
        _handle_stage_failure(job_id, task_type, exc.code.value, exc.message, exec_duration)
        raise
    except Exception as exc:
        _handle_stage_failure(job_id, task_type, ErrorCode.DIARIZATION_FAILED.value, str(exc), exec_duration)
        raise


@celery_app.task(name="run_emotions")
def run_emotions(job_id: str, normalized_object_key: str) -> dict:
    task_type = "emotions"
    exec_duration = None
    try:
        assert_cuda_available()
        settings = get_settings()
        storage = AsyncS3Storage(settings)
        with TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "normalized.wav"
            run_sync(storage.download_to_file(normalized_object_key, audio_path))
            started_at = perf_counter()
            emotions = GigaAMEmotionService.get_cached().get_probs(audio_path, duration_sec=duration_sec)
            exec_duration = round(perf_counter() - started_at, 3)
        payload = {
            "emotions": emotions,
            "emotions_duration_sec": exec_duration,
        }
        run_sync(_record_task_success(job_id, task_type, payload, exec_duration))
        return {"task_type": task_type, "status": "completed"}
    except AppError as exc:
        _handle_stage_failure(job_id, task_type, exc.code.value, exc.message, exec_duration)
        raise
    except Exception as exc:
        _handle_stage_failure(job_id, task_type, ErrorCode.INTERNAL_ERROR.value, str(exc), exec_duration)
        raise


async def _aggregate_transcription_job(job_id: str) -> bool:
    async with SessionLocal() as session:
        job = await get_job(session, job_id)
        if job is None or job.status == "failed":
            return True

        results = await get_task_results(session, job_id)
        required = {"asr", "emotions"}
        if job.diarization:
            required.add("diarization")
        if not required.issubset(results):
            return False

        failed = [result for result in results.values() if result.status == "failed"]
        if failed:
            first = failed[0]
            await mark_job_failed(
                session,
                job_id,
                first.error_code or ErrorCode.INTERNAL_ERROR.value,
                first.error_message or "Transcription child task failed",
            )
            return True

        asr_payload = results["asr"].payload
        emotion_payload = results["emotions"].payload
        words = [_word_from_payload(item) for item in asr_payload.get("words", [])]
        segments = []
        diarization_duration_sec = 0.0
        if job.diarization:
            diarization_payload = results["diarization"].payload
            segments = [_segment_from_payload(item) for item in diarization_payload.get("segments", [])]
            diarization_duration_sec = diarization_payload.get("diarization_duration_sec", 0.0)

        utterances = build_utterances(words, segments) if segments else []
        diagnostics = {
            "device": get_settings().device,
            "asr_duration_sec": results["asr"].exec_duration or asr_payload.get("asr_duration_sec", 0.0),
            "diarization_duration_sec": (
                results["diarization"].exec_duration or diarization_duration_sec
                if job.diarization
                else diarization_duration_sec
            ),
            "emotions_duration_sec": (
                results["emotions"].exec_duration or emotion_payload.get("emotions_duration_sec", 0.0)
            ),
            "emotions": emotion_payload.get("emotions", {}),
        }
        await mark_job_completed(
            session,
            job_id=job_id,
            duration_sec=asr_payload["duration_sec"],
            text=asr_payload["text"],
            utterances=utterances,
            diagnostics=diagnostics,
        )
        return True


@celery_app.task(bind=True, name="aggregate_transcription_job", max_retries=12, default_retry_delay=5)
def aggregate_transcription_job(self, _child_results, job_id: str) -> None:
    try:
        complete = run_sync(_aggregate_transcription_job(job_id))
        if not complete:
            raise self.retry(countdown=5)
    except Retry:
        raise
    except AppError as exc:
        run_sync(_mark_failed(job_id, exc.code.value, exc.message))
    except Exception as exc:
        run_sync(_mark_failed(job_id, ErrorCode.INTERNAL_ERROR.value, str(exc)))


@celery_app.task(name="transcribe_audio")
def transcribe_audio(job_id: str) -> None:
    prepare_transcription_job.delay(job_id)
