from pathlib import Path

import pytest

from app.audio.processing import normalize_audio
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
