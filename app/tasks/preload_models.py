from celery.signals import worker_process_init

from app.settings import get_settings


def preload_models() -> None:
    from app.asr.gigaam_service import GigaAMEmotionService, GigaAMService
    from app.diarization.pyannote_service import PyannoteDiarizationService

    settings = get_settings()
    GigaAMService.get_cached(settings.gigaam_model)
    GigaAMEmotionService.get_cached()
    if settings.hf_token:
        PyannoteDiarizationService.get_cached(settings.pyannote_model, settings.hf_token, settings.device)


@worker_process_init.connect
def preload_models_for_worker_process(**_kwargs) -> None:
    preload_models()


if __name__ == "__main__":
    preload_models()
