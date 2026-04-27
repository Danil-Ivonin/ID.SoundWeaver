import gc
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import gigaam
import torch
import torchaudio

from app.alignment import WordTimestamp


def clear_cuda_state() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


class GigaAMService:
    LONGFORM_THRESHOLD_SEC = 30.0
    _instances: dict[str, "GigaAMService"] = {}
    _lock = Lock()

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = gigaam.load_model(model_name)

    @classmethod
    def get_cached(cls, model_name: str) -> "GigaAMService":
        with cls._lock:
            service = cls._instances.get(model_name)
            if service is None:
                service = cls(model_name)
                cls._instances[model_name] = service
            return service

    @classmethod
    def clear_cache(cls) -> None:
        with cls._lock:
            cls._instances.clear()
        clear_cuda_state()

    @staticmethod
    def clear_cuda_state() -> None:
        clear_cuda_state()

    def transcribe(
        self,
        audio_path: Path,
        *,
        word_timestamps: bool,
        duration_sec: float | None = None,
    ) -> tuple[str, list[WordTimestamp]]:
        method = self.model.transcribe
        if duration_sec is not None and duration_sec > self.LONGFORM_THRESHOLD_SEC:
            method = self.model.transcribe_longform

        with torch.inference_mode():
            result = method(str(audio_path), word_timestamps=word_timestamps)
        if not word_timestamps:
            return result.text, []

        words = [
            WordTimestamp(text=word.text, start=float(word.start), end=float(word.end))
            for word in result.words
        ]
        clear_cuda_state()
        return result.text, words


class GigaAMEmotionService:
    CHUNK_DURATION_SEC = 30.0
    _instance: "GigaAMEmotionService | None" = None
    _lock = Lock()

    def __init__(self) -> None:
        self.model = gigaam.load_model("emo")

    @classmethod
    def get_cached(cls) -> "GigaAMEmotionService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def clear_cache(cls) -> None:
        with cls._lock:
            cls._instance = None
        clear_cuda_state()

    def get_probs(self, audio_path: Path, *, duration_sec: float | None = None) -> dict:
        if duration_sec is not None and duration_sec > self.CHUNK_DURATION_SEC:
            return self._get_chunked_probs(audio_path)

        with torch.inference_mode():
            res = self.model.get_probs(str(audio_path))
        clear_cuda_state()
        return res

    def _get_chunked_probs(self, audio_path: Path) -> dict:
        waveform, sample_rate = torchaudio.load(str(audio_path))
        chunk_size = int(self.CHUNK_DURATION_SEC * sample_rate)
        total_duration = 0.0
        weighted: dict[str, float] = {}

        with TemporaryDirectory() as tmp:
            chunk_dir = Path(tmp)
            for index, start in enumerate(range(0, waveform.shape[1], chunk_size)):
                chunk_waveform = waveform[:, start : start + chunk_size]
                chunk_duration = chunk_waveform.shape[1] / sample_rate
                if chunk_duration <= 0:
                    continue

                chunk_path = chunk_dir / f"chunk_{index}.wav"
                torchaudio.save(str(chunk_path), chunk_waveform, sample_rate)
                with torch.inference_mode():
                    probs = self.model.get_probs(str(chunk_path))

                total_duration += chunk_duration
                for label, value in probs.items():
                    weighted[label] = weighted.get(label, 0.0) + float(value) * chunk_duration

        clear_cuda_state()
        if total_duration == 0:
            return {}
        return {
            label: value / total_duration
            for label, value in weighted.items()
        }
