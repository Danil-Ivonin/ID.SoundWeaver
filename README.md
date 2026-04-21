# ID.SoundWeaver

ID.SoundWeaver is a Python microservice for speech-to-text processing of short audio recordings.

The service is designed to recognize Russian speech with GigaAM v3 `e2e_rnnt` and, when requested, split the recognized text by speakers using `pyannote.audio`. Audio files are uploaded through MinIO with presigned URLs, then processed asynchronously by a GPU Celery worker.

Planned production stack:

- FastAPI for the HTTP API.
- MinIO for S3-compatible audio storage.
- PostgreSQL and SQLAlchemy for jobs and results.
- Redis and Celery for background processing.
- NVIDIA GPU inference for GigaAM and pyannote.

The first version targets recordings up to 5 minutes long and returns either plain transcription text or speaker-labeled utterances.

This service requires a minimum of `880 MB` of video memory for stable operation.

## Development

Copy `.env.example` to `.env`, fill `HF_TOKEN`, then start services:

```bash
docker compose up --build
```

The API listens on `http://localhost:8000`. MinIO console listens on `http://localhost:9001`.
