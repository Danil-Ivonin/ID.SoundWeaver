from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import create_app


class FakeJobRepo:
    def __init__(self):
        self.jobs = {}

    def get_upload(self, upload_id):
        if upload_id == "missing":
            return None
        return type("Upload", (), {"id": upload_id, "status": "created", "object_key": "key"})

    def create_job(self, **kwargs):
        job = type("Job", (), {"id": kwargs["job_id"], "status": "queued"})
        self.jobs[job.id] = job
        return job

    def get_job(self, job_id):
        if job_id == "completed":
            result = type(
                "Result",
                (),
                {
                    "duration_sec": 1.0,
                    "text": "hello",
                    "utterances": [],
                    "diagnostics": {"emotions": {"neutral": 0.7}},
                },
            )
            return type(
                "Job",
                (),
                {
                    "id": job_id,
                    "status": "completed",
                    "created_at": datetime(2026, 4, 21, tzinfo=timezone.utc),
                    "updated_at": datetime(2026, 4, 21, tzinfo=timezone.utc),
                    "result": result,
                    "error_code": None,
                    "error_message": None,
                },
            )
        return None


class FakeQueue:
    def __init__(self):
        self.enqueued = []

    def enqueue(self, job_id):
        self.enqueued.append(job_id)


def test_create_transcription_enqueues_job():
    app = create_app()
    app.state.job_repo = FakeJobRepo()
    app.state.queue = FakeQueue()
    app.state.new_id = lambda: "job_1"
    client = TestClient(app)

    response = client.post(
        "/v1/transcriptions",
        json={"upload_id": "upload_1", "diarization": True, "min_speakers": 1, "max_speakers": 5},
    )

    assert response.status_code == 200
    assert response.json() == {"job_id": "job_1", "status": "queued"}
    assert app.state.queue.enqueued == ["job_1"]


def test_get_completed_transcription_returns_empty_utterances_without_diarization():
    app = create_app()
    app.state.job_repo = FakeJobRepo()
    client = TestClient(app)

    response = client.get("/v1/transcriptions/completed")

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["utterances"] == []
    assert response.json()["diagnostics"] == {"emotions": {"neutral": 0.7}}
