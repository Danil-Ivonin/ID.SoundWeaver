from dataclasses import dataclass
from pathlib import Path

import torchaudio

from app.errors import AppError, ErrorCode


@dataclass(frozen=True)
class NormalizedAudio:
    waveform: object
    sample_rate: int
    duration_sec: float
    path: Path


def build_chunk_windows(
    *,
    duration_sec: float,
    chunk_duration_sec: int,
    chunk_stride_sec: int,
) -> list[dict]:
    if chunk_duration_sec <= 0:
        raise ValueError("chunk_duration_sec must be positive")
    if chunk_stride_sec <= 0:
        raise ValueError("chunk_stride_sec must be positive")
    if chunk_stride_sec > chunk_duration_sec:
        raise ValueError("chunk_stride_sec cannot exceed chunk_duration_sec")
    if duration_sec <= 0:
        return []

    windows = []
    start_sec = 0.0
    index = 0
    while start_sec < duration_sec:
        end_sec = min(start_sec + chunk_duration_sec, duration_sec)
        windows.append({"index": index, "start_sec": round(start_sec, 6), "end_sec": round(end_sec, 6)})
        if end_sec >= duration_sec:
            break
        start_sec += chunk_stride_sec
        index += 1
    return windows


def save_audio_chunk(audio: NormalizedAudio, output_path: Path, *, start_sec: float, end_sec: float) -> NormalizedAudio:
    start_frame = max(0, int(round(start_sec * audio.sample_rate)))
    end_frame = min(audio.waveform.shape[1], int(round(end_sec * audio.sample_rate)))
    waveform = audio.waveform[:, start_frame:end_frame]
    torchaudio.save(str(output_path), waveform, audio.sample_rate)
    return NormalizedAudio(
        waveform=waveform,
        sample_rate=audio.sample_rate,
        duration_sec=waveform.shape[1] / audio.sample_rate,
        path=output_path,
    )


def normalize_audio(input_path: Path, output_path: Path, max_duration_sec: int) -> NormalizedAudio:
    try:
        waveform, sample_rate = torchaudio.load(str(input_path))
    except Exception as exc:
        raise AppError(ErrorCode.AUDIO_DECODE_FAILED, "Could not decode audio file") from exc

    if waveform.ndim != 2:
        raise AppError(ErrorCode.AUDIO_DECODE_FAILED, "Decoded audio has invalid shape")

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != 16_000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
        sample_rate = 16_000

    duration_sec = waveform.shape[1] / sample_rate
    if duration_sec > max_duration_sec:
        raise AppError(ErrorCode.AUDIO_DURATION_EXCEEDED, "Audio duration exceeds the limit")

    torchaudio.save(str(output_path), waveform, sample_rate)
    return NormalizedAudio(
        waveform=waveform,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        path=output_path,
    )
