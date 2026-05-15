from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
import torchaudio
from celery import chord, group
from celery.exceptions import Retry

from app.alignment import SpeakerSegment, WordTimestamp, build_utterances
from app.asr.gigaam_service import GigaAMEmotionService, GigaAMService, clear_cuda_state
from app.async_utils import run_sync
from app.audio.processing import build_chunk_windows, normalize_audio, save_audio_chunk
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

ASR_TASK_PREFIX = "asr:"
ASR_MANIFEST_TASK_TYPE = "asr_manifest"


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


def _offset_word(word: WordTimestamp, offset_sec: float) -> WordTimestamp:
    return WordTimestamp(
        text=word.text,
        start=round(word.start + offset_sec, 6),
        end=round(word.end + offset_sec, 6),
    )


def _asr_chunk_task_type(chunk_index: int) -> str:
    return f"{ASR_TASK_PREFIX}{chunk_index:06d}"


def merge_asr_chunk_payloads(payloads: list[dict]) -> dict:
    chunks = sorted(payloads, key=lambda payload: payload["chunk_index"])
    if not chunks:
        return {"text": "", "words": [], "duration_sec": 0.0, "asr_duration_sec": 0.0}

    boundaries = []
    for left, right in zip(chunks, chunks[1:], strict=False):
        left_end = float(left["chunk_end_sec"])
        right_start = float(right["chunk_start_sec"])
        boundaries.append((left_end + right_start) / 2 if left_end > right_start else right_start)

    merged_words = []
    for index, chunk in enumerate(chunks):
        lower_bound = boundaries[index - 1] if index > 0 else float("-inf")
        upper_bound = boundaries[index] if index < len(boundaries) else float("inf")
        for word in chunk.get("words", []):
            start = float(word["start"])
            end = float(word["end"])
            center = (start + end) / 2
            if lower_bound <= center < upper_bound:
                merged_words.append({"text": word["text"], "start": start, "end": end})

    merged_words.sort(key=lambda word: (word["start"], word["end"]))
    return {
        "text": " ".join(word["text"] for word in merged_words).strip(),
        "words": merged_words,
        "duration_sec": max(float(chunk["chunk_end_sec"]) for chunk in chunks),
        "asr_duration_sec": round(sum(float(chunk.get("asr_duration_sec", 0.0)) for chunk in chunks), 3),
    }


def build_transcription_child_signatures(
    *,
    job_id: str,
    normalized_object_key: str,
    chunks: list[dict],
    diarization: bool,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    duration_sec: float,
) -> list:
    signatures = [
        run_asr.s(
            job_id,
            chunk["object_key"],
            True,
            chunk["duration_sec"],
            chunk_index=chunk["index"],
            chunk_start_sec=chunk["start_sec"],
            chunk_end_sec=chunk["end_sec"],
        )
        for chunk in chunks
    ]
    signatures.append(run_emotions.s(job_id, normalized_object_key, duration_sec))
    if diarization:
        signatures.append(
            run_diarization.s(
                job_id,
                normalized_object_key,
                num_speakers,
                min_speakers,
                max_speakers,
            )
        )
    return signatures


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
        normalized_object_key = storage.build_normalized_artifact_key(job_id)
        await storage.upload_file(normalized_path, normalized_object_key)

        chunks = []
        for window in build_chunk_windows(
            duration_sec=audio.duration_sec,
            chunk_duration_sec=settings.transcription_chunk_duration_sec,
            chunk_stride_sec=settings.transcription_chunk_stride_sec,
        ):
            chunk_path = work_dir / f"chunk_{window['index']:06d}.wav"
            chunk_audio = save_audio_chunk(
                audio,
                chunk_path,
                start_sec=window["start_sec"],
                end_sec=window["end_sec"],
            )
            chunk_object_key = storage.build_audio_chunk_artifact_key(job_id, window["index"])
            await storage.upload_file(chunk_path, chunk_object_key)
            chunks.append(
                {
                    "index": window["index"],
                    "object_key": chunk_object_key,
                    "start_sec": window["start_sec"],
                    "end_sec": window["end_sec"],
                    "duration_sec": chunk_audio.duration_sec,
                }
            )

        await _record_task_success(
            job_id,
            ASR_MANIFEST_TASK_TYPE,
            {
                "chunks": chunks,
                "duration_sec": audio.duration_sec,
                "chunk_duration_sec": settings.transcription_chunk_duration_sec,
                "chunk_stride_sec": settings.transcription_chunk_stride_sec,
            },
        )
        child_signatures = build_transcription_child_signatures(
            job_id=job_id,
            normalized_object_key=normalized_object_key,
            chunks=chunks,
            diarization=diarization,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            duration_sec=audio.duration_sec,
        )
        chord(group(child_signatures), aggregate_transcription_job.s(job_id)).apply_async()


@celery_app.task(name="prepare_transcription_job")
def prepare_transcription_job(job_id: str) -> None:
    try:
        run_sync(_prepare_transcription_job(job_id))
    except AppError as exc:
        run_sync(_mark_failed(job_id, exc.code.value, exc.message))
    except Exception as exc:
        run_sync(_mark_failed(job_id, ErrorCode.INTERNAL_ERROR.value, str(exc)))


@celery_app.task(name="run_asr")
def run_asr(
    job_id: str,
    normalized_object_key: str,
    word_timestamps: bool,
    duration_sec: float,
    *,
    chunk_index: int | None = None,
    chunk_start_sec: float = 0.0,
    chunk_end_sec: float | None = None,
) -> dict:
    task_type = _asr_chunk_task_type(chunk_index) if chunk_index is not None else "asr"
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
        payload_words = [_word_to_payload(_offset_word(word, chunk_start_sec)) for word in words]
        payload = {
            "text": text,
            "words": payload_words,
            "duration_sec": duration_sec,
            "asr_duration_sec": exec_duration,
        }
        if chunk_index is not None:
            payload.update(
                {
                    "chunk_index": chunk_index,
                    "chunk_start_sec": chunk_start_sec,
                    "chunk_end_sec": chunk_end_sec if chunk_end_sec is not None else chunk_start_sec + duration_sec,
                }
            )
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
def run_emotions(job_id: str, normalized_object_key: str, duration_sec: float) -> dict:
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

        required = {"emotions"}
        if job.diarization:
            required.add("diarization")
        if ASR_MANIFEST_TASK_TYPE in results:
            manifest_payload = results[ASR_MANIFEST_TASK_TYPE].payload
            chunk_count = len(manifest_payload.get("chunks", []))
            required.update(_asr_chunk_task_type(index) for index in range(chunk_count))
        else:
            required.add("asr")
        if not required.issubset(results):
            return False

        if ASR_MANIFEST_TASK_TYPE in results:
            asr_payload = merge_asr_chunk_payloads(
                [
                    results[_asr_chunk_task_type(index)].payload
                    for index in range(len(results[ASR_MANIFEST_TASK_TYPE].payload.get("chunks", [])))
                ]
            )
            asr_duration_sec = asr_payload.get("asr_duration_sec", 0.0)
        else:
            asr_payload = results["asr"].payload
            asr_duration_sec = results["asr"].exec_duration or asr_payload.get("asr_duration_sec", 0.0)
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
            "asr_duration_sec": asr_duration_sec,
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
