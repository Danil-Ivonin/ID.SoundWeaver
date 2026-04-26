# ID.SoundWeaver API

HTTP API для загрузки аудио и асинхронного получения результатов распознавания речи.

Базовый URL в локальной среде:

```text
http://localhost:8000
```

Если сервис запущен с дефолтной конфигурацией FastAPI, интерактивная схема также доступна по:

- `/docs`
- `/redoc`

## Workflow

Полный сценарий работы выглядит так:

1. Клиент вызывает `POST /v1/uploads` и получает `upload_id` и presigned `PUT` URL.
2. Клиент загружает аудиофайл напрямую в MinIO по `upload_url`.
3. Клиент вызывает `POST /v1/transcriptions` с `upload_id`.
4. Сервис ставит задачу в очередь и возвращает `job_id`.
5. Клиент опрашивает `GET /v1/transcriptions/{job_id}` до статуса `completed` или `failed`.

## Supported Audio Content Types

Поддерживаются следующие `content_type`:

- `audio/wav`
- `audio/x-wav`
- `audio/mpeg`
- `audio/mp3`
- `audio/ogg`
- `audio/flac`
- `audio/mp4`
- `audio/x-m4a`

## Endpoints

### `GET /health`

Проверка доступности сервиса.

**Response `200 OK`**

```json
{
  "status": "ok"
}
```

### `POST /v1/uploads`

Создаёт запись о загрузке и возвращает presigned URL для прямой загрузки аудиофайла в MinIO.

**Request body**

```json
{
  "filename": "call.wav",
  "content_type": "audio/wav"
}
```

**Fields**

- `filename`: имя файла, от `1` до `255` символов.
- `content_type`: MIME-тип аудиофайла.

**Response `200 OK`**

```json
{
  "upload_id": "5a1abce661a34bd8bf6e5247f73dabf3",
  "upload_url": "http://localhost:9000/soundweaver-audio/uploads/5a1abce661a34bd8bf6e5247f73dabf3/call.wav?...",
  "method": "PUT",
  "expires_in_sec": 900
}
```

**Example**

```bash
curl -X POST http://localhost:8000/v1/uploads \
  -H 'Content-Type: application/json' \
  -d '{
    "filename": "call.wav",
    "content_type": "audio/wav"
  }'
```

После получения ответа файл нужно загрузить по `upload_url`:

```bash
curl -X PUT \
  -H 'Content-Type: audio/wav' \
  --upload-file ./call.wav \
  'http://localhost:9000/soundweaver-audio/uploads/5a1abce661a34bd8bf6e5247f73dabf3/call.wav?...'
```

**Validation errors**

FastAPI вернёт `422 Unprocessable Entity`, если:

- отсутствует обязательное поле;
- `filename` пустой;
- передан неподдерживаемый `content_type`.

Пример ответа:

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body"],
      "msg": "Value error, Unsupported audio content type. Expected one of: audio/wav, ...",
      "input": {
        "filename": "call.txt",
        "content_type": "text/plain"
      }
    }
  ]
}
```

### `POST /v1/transcriptions`

Создаёт задачу транскрибации для уже загруженного файла.

**Request body**

Минимальный запрос:

```json
{
  "upload_id": "5a1abce661a34bd8bf6e5247f73dabf3"
}
```

Запрос с diarization:

```json
{
  "upload_id": "5a1abce661a34bd8bf6e5247f73dabf3",
  "diarization": true,
  "min_speakers": 1,
  "max_speakers": 3
}
```

**Fields**

- `upload_id`: идентификатор upload, полученный из `POST /v1/uploads`.
- `diarization`: если `true`, сервис попытается разделить речь по спикерам.
- `num_speakers`: точное количество спикеров.
- `min_speakers`: нижняя граница количества спикеров.
- `max_speakers`: верхняя граница количества спикеров.

**Ограничения speaker params**

- `num_speakers` нельзя передавать вместе с `min_speakers` или `max_speakers`.
- `min_speakers` не может быть больше `max_speakers`.
- все speaker-поля должны быть `>= 1`.

**Response `200 OK`**

```json
{
  "job_id": "380afc8cb8c64ced96d7a54e336ad739",
  "status": "queued"
}
```

Если такой же запрос уже был создан ранее для того же `upload_id` и тех же speaker-параметров, сервис может вернуть уже существующий `job_id`.

**Example**

```bash
curl -X POST http://localhost:8000/v1/transcriptions \
  -H 'Content-Type: application/json' \
  -d '{
    "upload_id": "5a1abce661a34bd8bf6e5247f73dabf3",
    "diarization": true,
    "min_speakers": 1,
    "max_speakers": 3
  }'
```

**Error `404 Not Found`**

Если `upload_id` не найден:

```json
{
  "detail": {
    "code": "upload_not_found",
    "message": "Upload not found"
  }
}
```

**Error `409 Conflict`**

Если запись upload есть, но объект ещё не был загружен в MinIO:

```json
{
  "detail": {
    "code": "upload_not_completed",
    "message": "Upload object was not found in storage"
  }
}
```

**Validation errors**

`422 Unprocessable Entity` возвращается для неверных speaker-параметров.

Пример:

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body"],
      "msg": "Value error, num_speakers cannot be combined with min_speakers or max_speakers",
      "input": {
        "upload_id": "5a1abce661a34bd8bf6e5247f73dabf3",
        "num_speakers": 2,
        "min_speakers": 1
      }
    }
  ]
}
```

### `GET /v1/transcriptions/{job_id}`

Возвращает текущий статус задачи транскрибации и, если задача завершена, итоговый результат.

**Example**

```bash
curl http://localhost:8000/v1/transcriptions/380afc8cb8c64ced96d7a54e336ad739
```

#### Response `200 OK`, job in progress

```json
{
  "job_id": "380afc8cb8c64ced96d7a54e336ad739",
  "status": "processing",
  "created_at": "2026-04-26T18:47:35.447443Z",
  "updated_at": "2026-04-26T18:47:35.447443Z",
  "duration_sec": null,
  "text": null,
  "utterances": null,
  "diagnostics": null,
  "error": null
}
```

#### Response `200 OK`, job completed without diarization

```json
{
  "job_id": "380afc8cb8c64ced96d7a54e336ad739",
  "status": "completed",
  "created_at": "2026-04-26T18:47:35.447443Z",
  "updated_at": "2026-04-26T18:47:40.102311Z",
  "duration_sec": 2.0,
  "text": "пример распознанного текста",
  "utterances": [],
  "diagnostics": {
    "device": "cuda",
    "asr_duration_sec": 0.412,
    "diarization_duration_sec": 0.0,
    "emotions_duration_sec": 0.083,
    "emotions": {
      "neutral": 0.82,
      "happy": 0.11,
      "sad": 0.07
    }
  },
  "error": null
}
```

#### Response `200 OK`, job completed with diarization

```json
{
  "job_id": "380afc8cb8c64ced96d7a54e336ad739",
  "status": "completed",
  "created_at": "2026-04-26T18:47:35.447443Z",
  "updated_at": "2026-04-26T18:47:42.778120Z",
  "duration_sec": 7.3,
  "text": "полный текст записи",
  "utterances": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 3.1,
      "text": "добрый день"
    },
    {
      "speaker": "SPEAKER_01",
      "start": 3.1,
      "end": 7.3,
      "text": "здравствуйте"
    }
  ],
  "diagnostics": {
    "device": "cuda",
    "asr_duration_sec": 0.551,
    "diarization_duration_sec": 0.913,
    "emotions_duration_sec": 0.092,
    "emotions": {
      "neutral": 0.74,
      "happy": 0.18,
      "sad": 0.08
    }
  },
  "error": null
}
```

#### Response `200 OK`, job failed

```json
{
  "job_id": "380afc8cb8c64ced96d7a54e336ad739",
  "status": "failed",
  "created_at": "2026-04-26T18:47:35.447443Z",
  "updated_at": "2026-04-26T18:51:35.411419Z",
  "duration_sec": null,
  "text": null,
  "utterances": null,
  "diagnostics": null,
  "error": {
    "code": "internal_error",
    "message": "Task failed"
  }
}
```

**Error `404 Not Found`**

Если `job_id` не найден:

```json
{
  "detail": {
    "code": "job_not_found",
    "message": "Job not found"
  }
}
```

## Status Values

На практике сервис использует следующие значения статуса задачи:

- `queued`: задача создана и ожидает обработки.
- `processing`: задача взята в работу воркером.
- `completed`: задача успешно завершена.
- `failed`: задача завершилась ошибкой.

## Error Codes

В ответах сервиса и внутренних результатах могут встречаться следующие коды ошибок:

- `upload_too_large`
- `unsupported_content_type`
- `upload_not_found`
- `job_not_found`
- `upload_not_completed`
- `invalid_speaker_params`
- `audio_decode_failed`
- `audio_duration_exceeded`
- `cuda_unavailable`
- `asr_failed`
- `diarization_failed`
- `alignment_failed`
- `internal_error`

Фактически в HTTP API напрямую сейчас используются прежде всего:

- `upload_not_found`
- `upload_not_completed`
- `job_not_found`
- `internal_error`

Остальные коды обычно приходят в поле `error.code` у уже завершившейся со статусом `failed` задачи.

## Example End-to-End

```bash
# 1. Request upload URL
curl -X POST http://localhost:8000/v1/uploads \
  -H 'Content-Type: application/json' \
  -d '{
    "filename": "call.wav",
    "content_type": "audio/wav"
  }'

# 2. Upload file to MinIO with the returned upload_url
curl -X PUT \
  -H 'Content-Type: audio/wav' \
  --upload-file ./call.wav \
  'http://localhost:9000/soundweaver-audio/uploads/...'

# 3. Create transcription job
curl -X POST http://localhost:8000/v1/transcriptions \
  -H 'Content-Type: application/json' \
  -d '{
    "upload_id": "5a1abce661a34bd8bf6e5247f73dabf3",
    "diarization": true,
    "min_speakers": 1,
    "max_speakers": 3
  }'

# 4. Poll status
curl http://localhost:8000/v1/transcriptions/380afc8cb8c64ced96d7a54e336ad739
```
