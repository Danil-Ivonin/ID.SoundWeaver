from celery import Celery

from app.settings import get_settings


def create_celery_app() -> Celery:
    settings = get_settings()
    app = Celery(
        "soundweaver",
        broker=settings.redis_url,
        backend=settings.redis_url,
        include=["app.tasks.transcription"],
    )
    app.conf.task_track_started = True
    app.conf.result_extended = True
    return app


celery_app = create_celery_app()
