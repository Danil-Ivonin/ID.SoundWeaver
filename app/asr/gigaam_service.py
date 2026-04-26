import gc
from pathlib import Path

import gigaam
import torch

from app.alignment import WordTimestamp


class GigaAMService:
    LONGFORM_THRESHOLD_SEC = 30.0

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = gigaam.load_model(model_name)

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

        result = method(str(audio_path), word_timestamps=word_timestamps)
        if not word_timestamps:
            return result.text, []

        words = [
            WordTimestamp(text=word.text, start=float(word.start), end=float(word.end))
            for word in result.words
        ]
        gc.collect()
        torch.cuda.empty_cache()
        return result.text, words


class GigaAMEmotionService:
    def __init__(self) -> None:
        self.model = gigaam.load_model("emo")

    def get_probs(self, audio_path: Path) -> dict:
        res = self.model.get_probs(str(audio_path))
        gc.collect()
        torch.cuda.empty_cache()
        return res
