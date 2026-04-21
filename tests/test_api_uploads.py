from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import get_settings


class FakeUploadRepo:
    def create_upload(self, **kwargs):
        return type("Upload", (), kwargs)


class FakeStorage:
    def build_object_key(self, upload_id, filename):
        return f"uploads/{upload_id}/{filename}"

    def create_presigned_put_url(self, object_key):
        return f"http://minio/audio/{object_key}?signature=test"


def make_test_app(max_upload_size_bytes=104_857_600):
    app = create_app()
    app.state.upload_repo = FakeUploadRepo()
    app.state.storage = FakeStorage()
    app.state.now = lambda: datetime(2026, 4, 21, tzinfo=timezone.utc)
    app.state.new_id = lambda: "upload_1"
    app.dependency_overrides[get_settings] = lambda: get_settings().model_copy(
        update={"max_upload_size_bytes": max_upload_size_bytes}
    )
    return app


def test_create_upload_returns_presigned_url():
    client = TestClient(make_test_app())

    response = client.post(
        "/v1/uploads",
        json={"filename": "audio.wav", "content_type": "audio/wav", "size_bytes": 1024},
    )

    assert response.status_code == 200
    assert response.json()["upload_id"] == "upload_1"
    assert response.json()["method"] == "PUT"


def test_create_upload_rejects_too_large_file():
    client = TestClient(make_test_app(max_upload_size_bytes=100))

    response = client.post(
        "/v1/uploads",
        json={"filename": "audio.wav", "content_type": "audio/wav", "size_bytes": 101},
    )

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "upload_too_large"
