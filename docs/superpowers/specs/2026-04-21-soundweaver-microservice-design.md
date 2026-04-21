# SoundWeaver ASR Microservice Design

## Purpose

SoundWeaver is a Python microservice for Russian speech recognition with optional speaker diarization. The first version processes uploaded audio recordings up to 5 minutes long, runs GigaAM v3 `e2e_rnnt` for speech-to-text, and optionally runs `pyannote.audio` to split recognized text into speaker utterances.

The production target is a Linux server with an NVIDIA GPU. CPU fallback is intentionally out of scope for the first implementation.

## Scope

In scope:

- Reliable audio upload through S3-compatible object storage.
- MinIO deployed in Docker on the same server.
- Async transcription jobs.
- GigaAM v3 `e2e_rnnt` transcription.
- Optional pyannote speaker diarization.
- Speaker-labeled utterances when diarization is enabled.
- SQLAlchemy-based persistence.
- Redis/Celery-based background processing.
- Docker Compose deployment.

Out of scope:

- Streaming recognition.
- CPU production mode.
- Multi-server object storage.
- Audio longer than 5 minutes.
- User-facing authentication and authorization.
- Manual speaker identity enrollment.

## Architecture

The service uses FastAPI for HTTP API, PostgreSQL for persistent metadata and results, Redis as Celery broker/result backend, MinIO for audio object storage, and a dedicated Celery worker for GPU inference.

```text
Client
  -> FastAPI
    -> PostgreSQL via SQLAlchemy
    -> MinIO presigned upload URL
    -> Redis/Celery enqueue job_id
      -> Celery GPU worker
        -> MinIO read audio
        -> normalize audio
        -> GigaAM e2e_rnnt
        -> optional pyannote diarization
        -> speaker alignment
        -> PostgreSQL save result
```

Only the Celery worker requires GPU access. FastAPI, PostgreSQL, Redis, and MinIO run without GPU.

## Upload Flow

Production upload uses presigned MinIO URLs so the API service does not proxy large audio bodies.

1. Client requests an upload URL from `POST /v1/uploads`.
2. API validates filename, content type, and declared size.
3. API creates an `uploads` row and returns a presigned `PUT` URL.
4. Client uploads the file directly to MinIO with `PUT`.
5. Client creates a transcription job with `POST /v1/transcriptions` and the returned `upload_id`.
6. Worker reads the object from MinIO during processing.

This is more reliable than sending the whole audio file through a regular API request because upload bandwidth and retry behavior are isolated from the API process. It also keeps the architecture ready for later horizontal scaling.

## API Contract

### Create Upload

```http
POST /v1/uploads
Content-Type: application/json
```

Request:

```json
{
  "filename": "meeting.wav",
  "content_type": "audio/wav",
  "size_bytes": 52428800
}
```

Response:

```json
{
  "upload_id": "01J...",
  "upload_url": "http://minio:9000/...",
  "method": "PUT",
  "expires_in_sec": 900
}
```

### Create Transcription Job

```http
POST /v1/transcriptions
Content-Type: application/json
```

Request:

```json
{
  "upload_id": "01J...",
  "diarization": true,
  "num_speakers": null,
  "min_speakers": 1,
  "max_speakers": 5
}
```

Response:

```json
{
  "job_id": "01J...",
  "status": "queued"
}
```

### Get Transcription

```http
GET /v1/transcriptions/{job_id}
```

Queued or processing response:

```json
{
  "job_id": "01J...",
  "status": "processing",
  "created_at": "2026-04-21T09:00:00Z",
  "updated_at": "2026-04-21T09:00:04Z"
}
```

Completed response without diarization:

```json
{
  "job_id": "01J...",
  "status": "completed",
  "duration_sec": 42.1,
  "text": "recognized text",
  "utterances": []
}
```

Completed response with diarization:

```json
{
  "job_id": "01J...",
  "status": "completed",
  "duration_sec": 42.1,
  "text": "recognized text",
  "utterances": [
    {
      "speaker": "SPEAKER_00",
      "start": 1.12,
      "end": 4.8,
      "text": "first phrase"
    },
    {
      "speaker": "UNKNOWN",
      "start": 4.81,
      "end": 5.2,
      "text": "word without a confident speaker segment"
    }
  ]
}
```

Failed response:

```json
{
  "job_id": "01J...",
  "status": "failed",
  "error": {
    "code": "audio_decode_failed",
    "message": "Could not decode audio file"
  }
}
```

## Limits

- Maximum declared upload size: 100 MB.
- Maximum decoded audio duration: 5 minutes.
- Supported input formats: WAV, MP3, OGG, FLAC, M4A.
- Internal processing format: mono, 16 kHz.
- GPU worker concurrency: 1.
- Presigned upload URL TTL: 15 minutes.

The size limit is intentionally larger than typical 5-minute compressed audio and also covers high-quality WAV files. The duration limit is enforced after decoding because file size alone is not a reliable proxy for audio length.

## Components

### `app/api`

FastAPI routers for uploads, transcription jobs, health checks, and result retrieval. The API validates request shape and stores metadata, but it does not perform inference.

### `app/db`

SQLAlchemy models, sessions, migrations, and repository helpers. PostgreSQL is the production database.

### `app/storage`

MinIO client wrapper for object keys, presigned upload URLs, object existence checks, download streams, and retention cleanup. The storage boundary should allow a local backend later for development if needed, but MinIO is the production path.

### `app/audio`

Audio decoding and validation. This module normalizes inputs to mono 16 kHz and checks duration. It should prefer explicit decoding through ffmpeg or torchaudio and pass preloaded waveform tensors to pyannote to avoid relying on pyannote's internal file decoder.

### `app/asr`

GigaAM v3 `e2e_rnnt` model wrapper. The model is loaded once per worker process. When diarization is enabled, transcription must request word timestamps for alignment.

### `app/diarization`

pyannote pipeline wrapper. The pipeline is loaded once per worker process and moved to CUDA. Hugging Face token and model name come from environment settings.

### `app/alignment`

Assigns recognized words to speakers by maximum time overlap with pyannote speaker segments. Words that do not overlap any speaker segment are kept and assigned to speaker `UNKNOWN`. Neighboring words with the same speaker are merged into utterances.

### `app/tasks`

Celery application and transcription task. The worker processes jobs sequentially on GPU and writes final state to PostgreSQL.

### `app/settings`

Typed environment configuration for database URL, Redis URL, MinIO credentials, bucket names, model names, GPU device, upload limits, and retention settings.

## Database Model

### `uploads`

- `id`: string ULID/UUID primary key.
- `object_key`: MinIO object key.
- `filename`: original filename.
- `content_type`: declared content type.
- `size_bytes`: declared size.
- `status`: `created`, `uploaded`, `expired`, `failed`.
- `created_at`: timestamp.
- `expires_at`: timestamp.

### `transcription_jobs`

- `id`: string ULID/UUID primary key.
- `upload_id`: foreign key to `uploads`.
- `status`: `queued`, `processing`, `completed`, `failed`.
- `diarization`: boolean.
- `num_speakers`: nullable integer.
- `min_speakers`: nullable integer.
- `max_speakers`: nullable integer.
- `error_code`: nullable string.
- `error_message`: nullable string.
- `created_at`: timestamp.
- `updated_at`: timestamp.
- `started_at`: nullable timestamp.
- `finished_at`: nullable timestamp.

### `transcription_results`

- `job_id`: primary key and foreign key to `transcription_jobs`.
- `duration_sec`: float.
- `text`: full recognized text.
- `utterances`: JSONB array.
- `diagnostics`: JSONB object for model versions, processing timings, and warnings.
- `created_at`: timestamp.

## Processing Flow

For every transcription job:

1. Worker marks job as `processing`.
2. Worker verifies the MinIO object exists.
3. Worker downloads or streams the object to a temporary local file.
4. Audio is decoded, duration-checked, resampled to mono 16 kHz.
5. GigaAM runs transcription.
6. If `diarization` is false, result is saved with full text and empty `utterances`.
7. If `diarization` is true, GigaAM word timestamps and pyannote speaker segments are aligned.
8. Aligned words are merged into speaker utterances.
9. Result is saved and job is marked `completed`.
10. On failure, job is marked `failed` with a stable error code and message.

## Deployment

Docker Compose services:

- `api`: FastAPI app.
- `worker`: Celery worker with NVIDIA GPU access.
- `postgres`: PostgreSQL database.
- `redis`: Celery broker/result backend.
- `minio`: S3-compatible object storage.
- `nginx`: optional reverse proxy.

The worker should run with Celery concurrency set to 1. The container image must install a CUDA-enabled PyTorch and matching torchaudio before installing higher-level model dependencies.

Required environment variables:

- `DATABASE_URL`
- `REDIS_URL`
- `MINIO_ENDPOINT`
- `MINIO_ACCESS_KEY`
- `MINIO_SECRET_KEY`
- `MINIO_BUCKET`
- `HF_TOKEN`
- `PYANNOTE_MODEL`
- `GIGAAM_MODEL`
- `DEVICE=cuda`

## Dependency Strategy

The Docker image should pin the CUDA, PyTorch, and torchaudio versions together. The install order should prevent transitive dependencies from replacing the CUDA-enabled torch build with a CPU build:

1. Start from an NVIDIA CUDA runtime image.
2. Install system ffmpeg.
3. Install Python dependencies with pinned versions.
4. Install CUDA-enabled `torch` and matching `torchaudio`.
5. Install `gigaam`, `pyannote.audio`, and service dependencies.
6. Run a startup check that fails if `torch.cuda.is_available()` is false.

Model caches should be mounted as persistent Docker volumes so restarts do not re-download GigaAM and pyannote artifacts.

## Error Handling

The API should use stable error codes:

- `upload_too_large`
- `unsupported_content_type`
- `upload_not_found`
- `upload_not_completed`
- `invalid_speaker_params`
- `audio_decode_failed`
- `audio_duration_exceeded`
- `cuda_unavailable`
- `asr_failed`
- `diarization_failed`
- `alignment_failed`
- `internal_error`

HTTP status codes:

- `400` for invalid parameters.
- `404` for missing upload or job.
- `409` for upload state conflicts.
- `413` for declared size above the limit.
- `422` for semantically invalid speaker parameters.
- `429` or `503` when the queue is full or worker capacity is unavailable.

## First Implementation Scope

The first implementation should include:

- FastAPI endpoints for uploads and transcription jobs.
- MinIO presigned upload integration.
- SQLAlchemy models and migrations.
- Redis/Celery worker wiring.
- GPU-only startup validation.
- Audio normalization and duration validation.
- GigaAM transcription.
- Optional pyannote diarization.
- Word-to-speaker alignment with `UNKNOWN` fallback.
- Docker Compose for local single-server deployment.
- Focused tests for API validation, SQLAlchemy repositories, storage key generation, and alignment logic.

