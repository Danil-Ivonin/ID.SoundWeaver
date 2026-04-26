from app.asr.gigaam_service import GigaAMEmotionService, GigaAMService
from app.diarization.pyannote_service import PyannoteDiarizationService
from app.settings import get_settings


def preload_models() -> None:
    settings = get_settings()
    GigaAMService(settings.gigaam_model)
    GigaAMEmotionService()
    PyannoteDiarizationService(settings.pyannote_model, settings.hf_token, settings.device)


if __name__ == "__main__":
    preload_models()
