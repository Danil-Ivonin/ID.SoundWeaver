import gc
from threading import Lock

import torch
from pyannote.audio import Pipeline

from app.alignment import SpeakerSegment
from app.asr.gigaam_service import clear_cuda_state


class PyannoteDiarizationService:
    _instances: dict[tuple[str, str, str], "PyannoteDiarizationService"] = {}
    _lock = Lock()

    def __init__(self, model_name: str, token: str, device: str) -> None:
        self.pipeline = Pipeline.from_pretrained(model_name, token=token)
        self.pipeline.to(torch.device(device))

    @classmethod
    def get_cached(cls, model_name: str, token: str, device: str) -> "PyannoteDiarizationService":
        cache_key = (model_name, token, device)
        with cls._lock:
            service = cls._instances.get(cache_key)
            if service is None:
                service = cls(model_name, token, device)
                cls._instances[cache_key] = service
            return service

    @classmethod
    def clear_cache(cls) -> None:
        with cls._lock:
            cls._instances.clear()
        clear_cuda_state()

    def diarize(
        self,
        *,
        waveform,
        sample_rate: int,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> list[SpeakerSegment]:
        kwargs = {
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        with torch.inference_mode():
            output = self.pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)
        gc.collect()
        clear_cuda_state()
        return [
            SpeakerSegment(speaker=speaker, start=float(turn.start), end=float(turn.end))
            for turn, speaker in output.speaker_diarization
        ]
