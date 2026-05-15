from pathlib import Path

import pytest

from app.audio.processing import build_chunk_windows, normalize_audio
from app.errors import AppError, ErrorCode

torch = pytest.importorskip("torch")
torchaudio = pytest.importorskip("torchaudio")


def test_normalize_audio_converts_to_mono_16khz(tmp_path: Path):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "normalized.wav"
    waveform = torch.ones(2, 48_000)
    torchaudio.save(str(input_path), waveform, 48_000)

    result = normalize_audio(input_path, output_path, max_duration_sec=5)

    assert result.sample_rate == 16_000
    assert result.waveform.shape[0] == 1
    assert result.duration_sec == pytest.approx(1.0, abs=0.01)
    assert output_path.exists()


def test_normalize_audio_rejects_duration_above_limit(tmp_path: Path):
    input_path = tmp_path / "long.wav"
    output_path = tmp_path / "normalized.wav"
    waveform = torch.ones(1, 32_000)
    torchaudio.save(str(input_path), waveform, 16_000)

    with pytest.raises(AppError) as exc_info:
        normalize_audio(input_path, output_path, max_duration_sec=1)

    assert exc_info.value.code == ErrorCode.AUDIO_DURATION_EXCEEDED


def test_build_chunk_windows_uses_overlap_stride():
    windows = build_chunk_windows(duration_sec=80.0, chunk_duration_sec=30, chunk_stride_sec=25)

    assert windows == [
        {"index": 0, "start_sec": 0.0, "end_sec": 30.0},
        {"index": 1, "start_sec": 25.0, "end_sec": 55.0},
        {"index": 2, "start_sec": 50.0, "end_sec": 80.0},
    ]


def test_build_chunk_windows_keeps_short_audio_as_single_chunk():
    windows = build_chunk_windows(duration_sec=12.5, chunk_duration_sec=30, chunk_stride_sec=25)

    assert windows == [{"index": 0, "start_sec": 0.0, "end_sec": 12.5}]
