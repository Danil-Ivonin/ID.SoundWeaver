from pathlib import Path

import gigaam

from app.alignment import WordTimestamp


class GigaAMService:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = gigaam.load_model(model_name)

    def transcribe(
        self,
        audio_path: Path,
        *,
        word_timestamps: bool,
    ) -> tuple[str, list[WordTimestamp]]:
        result = self.model.transcribe(str(audio_path), word_timestamps=word_timestamps)
        if isinstance(result, str):
            return result, []

        text = getattr(result, "text", str(result))
        words = [
            WordTimestamp(text=word.text, start=float(word.start), end=float(word.end))
            for word in getattr(result, "words", [])
        ]
        return text, words


class GigaAMEmotionService:
    def __init__(self) -> None:
        self.model = gigaam.load_model("emo")

    def get_probs(self, audio_path: Path) -> dict:
        return self.model.get_probs(str(audio_path))
