# ID.SoundWeaver

ID.SoundWeaver is a Python microservice for speech-to-text processing of audio recordings.

The service is designed to recognize Russian speech with GigaAM v3 `e2e_rnnt` and, when requested, split the recognized text by speakers using `pyannote.audio`. Audio files are uploaded through MinIO with presigned URLs, then processed asynchronously by Celery workers.

Planned production stack:

- FastAPI for the HTTP API.
- MinIO for S3-compatible audio storage.
- PostgreSQL and SQLAlchemy for jobs and results.
- Redis and Celery for background processing.
- NVIDIA GPU inference for GigaAM and pyannote.

The service targets recordings up to 1 hour long and returns either plain transcription text or speaker-labeled utterances. Long ASR inputs use GigaAM `transcribe_longform`.

This service requires a minimum of `880 MB` of video memory for stable operation.

## Development

Copy `.env.example` to `.env`, fill `HF_TOKEN`, then start services:

```bash
docker compose up --build
```

The API listens on `http://localhost:8000`. MinIO console listens on `http://localhost:9001`.

The default development stack starts API, PostgreSQL, Redis, and MinIO. The Celery worker is placed behind the `gpu` profile because it requires an NVIDIA-enabled Docker runtime:

```bash
docker compose --profile gpu up --build
```

Database connection settings are configured through `DB_*` variables in `.env`:

```env
DB_DRIVER=postgresql+asyncpg
DB_USER=soundweaver
DB_PASSWORD=soundweaver
DB_HOST=postgres
DB_PORT=5432
DB_NAME=soundweaver
```
