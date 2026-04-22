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
