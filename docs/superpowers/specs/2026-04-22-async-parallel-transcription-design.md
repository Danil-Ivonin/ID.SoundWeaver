# Async Parallel Transcription Design

## Goal

Make ID.SoundWeaver asynchronous across the HTTP API, database access, S3 storage access, and transcription workflow, while splitting speech recognition, diarization, and emotion detection into separate Celery tasks that run in parallel for a single transcription job.

## Scope

This change covers:

- async FastAPI endpoints for uploads and transcriptions;
- async SQLAlchemy sessions and repositories;
- async S3-compatible storage operations;
- a default maximum normalized audio duration of 1 hour;
- a Celery workflow with separate ASR, diarization, and emotion tasks;
- safe aggregation of parallel task outputs into one public transcription result;
- idempotent handling for duplicate HTTP requests and repeated Celery task delivery.

This change does not add a runtime switch to disable parallel inference. ASR, diarization, and emotion detection are designed to run in parallel whenever their corresponding work is required.

## Architecture

The API remains responsible for request validation, upload registration, transcription job creation, and status reads. It no longer performs synchronous database or storage calls. Runtime repositories use `AsyncSession`, and storage uses an async S3 client compatible with MinIO.

The worker layer changes from one monolithic `transcribe_audio` task to an orchestration workflow:

1. The API creates or reuses a transcription job for the same upload and parameters.
2. The API enqueues an orchestration task for that job only when a new job is created.
3. The orchestration task atomically claims the job by changing it from `queued` to `processing`.
4. It downloads and normalizes the audio once.
5. It uploads the normalized WAV as a worker artifact in S3.
6. It launches separate Celery tasks for ASR, diarization, and emotion detection. Each child task downloads the normalized artifact independently, so tasks do not require a shared filesystem.
7. An aggregation task reads the child task outputs, builds utterances, writes `transcription_results`, and marks the job `completed`.

If diarization is disabled, no diarization model task is launched. Aggregation uses an empty segment list and returns empty utterances.

## Async API And Database

The database layer will use SQLAlchemy async APIs:

- `create_async_engine`;
- `async_sessionmaker`;
- `AsyncSession`;
- `await session.execute(...)`, `await session.commit()`, and `await session.refresh(...)`.

Application database URLs should use async drivers:

- PostgreSQL: `postgresql+asyncpg://soundweaver:soundweaver@postgres:5432/soundweaver`;
- SQLite tests: `sqlite+aiosqlite:///:memory:`.

The public repository functions keep the same intent as today but become async. Runtime repository wrappers in `app/main.py` will await those methods from async endpoints.

## Async S3 Storage

Storage will move from the synchronous `minio` Python client to `aioboto3`, using MinIO through its S3-compatible API.

The storage abstraction will expose:

- `build_object_key(upload_id, filename)`;
- `async create_presigned_put_url(object_key)`;
- `async download_to_file(object_key, path)`;
- `async upload_file(path, object_key)`;
- `async object_exists(object_key)`.

`object_exists` is used before transcription job creation so a client cannot enqueue processing for an upload record whose file was never uploaded to MinIO.

Normalized worker artifacts are stored under `artifacts/{job_id}/normalized.wav`. They are internal implementation objects, not part of the public API.

## Celery Task Model

The workflow uses these task roles:

- `prepare_transcription_job(job_id)`: claims the job, downloads audio, normalizes it, uploads the normalized artifact, and dispatches child tasks.
- `run_asr(job_id, normalized_object_key, needs_word_timestamps)`: downloads the normalized artifact and produces text and word timestamps.
- `run_diarization(job_id, normalized_object_key, diarization params)`: downloads the normalized artifact and produces speaker segments.
- `run_emotions(job_id, normalized_object_key)`: downloads the normalized artifact and produces emotion probabilities.
- `aggregate_transcription_job(job_id)`: combines ASR, diarization, and emotion outputs into the final result.

Celery tasks are still synchronous function entrypoints because Celery workers execute task functions, but each task may call async DB/storage helpers through a small `asyncio.run(...)` boundary when needed. Heavy ML calls remain normal blocking model calls inside their own Celery tasks. Parallelism comes from Celery executing separate tasks concurrently, not from `asyncio.gather` inside one process.

ASR uses `model.transcribe(...)` for normalized audio up to 30 seconds and `model.transcribe_longform(...)` for normalized audio longer than 30 seconds. The same `word_timestamps` flag is passed to both calls so diarization alignment can consume word timestamps for long audio too.

The compose worker command must allow actual parallel task execution. The current `--concurrency=1 --pool=solo` is incompatible with running child tasks in parallel in a single worker process. The worker configuration will be changed so multiple task executions can run at the same time. For GPU deployments, the recommended production shape is separate workers or queues per model family when one GPU cannot safely host all three models concurrently.

## Data Model

Existing tables stay:

- `uploads`;
- `transcription_jobs`;
- `transcription_results`.

A new table stores child task outputs:

`transcription_task_results`

- `job_id`;
- `task_type`: `asr`, `diarization`, or `emotions`;
- `status`: `completed` or `failed`;
- `payload`: JSON result payload;
- `error_code`;
- `error_message`;
- timestamps.

The primary key is `(job_id, task_type)`. Writes use upsert semantics so repeated task delivery replaces the same child result instead of creating duplicates.

## Duplicate And Retry Safety

Duplicate HTTP requests are handled by a deterministic job uniqueness rule. For a given upload and transcription parameters, the system returns the existing job when one already exists instead of creating a second job.

The unique transcription identity includes:

- `upload_id`;
- `diarization`;
- `num_speakers`;
- `min_speakers`;
- `max_speakers`.

Repeated Celery delivery is handled through state transitions and idempotent writes:

- job claiming updates only rows where `status = 'queued'`;
- child task result writes are upserts on `(job_id, task_type)`;
- aggregation checks required child outputs before finalizing;
- final result writing is idempotent for `job_id`;
- failed jobs are not overwritten by late successful child tasks.

## Error Handling

If upload object verification fails, the API returns a client error before queuing the job.

If preparation fails, the job is marked `failed`.

If any child task fails, it writes a failed child result and marks the parent job failed if the parent is still active.

If aggregation runs before all required child results exist, it raises a controlled retryable error and Celery retries the aggregation task later. It does not mark the job completed until all required child results are present.

## Testing

Tests should cover:

- async upload and transcription API behavior;
- async repository round trips;
- async storage wrapper behavior with fake clients;
- duplicate transcription requests returning the same job;
- job claim compare-and-set behavior;
- child task result upsert behavior;
- aggregation with ASR + emotions only;
- aggregation with ASR + diarization + emotions;
- failure propagation from a child task to the parent job.

The existing worker processor tests should be split so ASR, diarization, emotions, and aggregation can be tested independently without loading real ML models.
