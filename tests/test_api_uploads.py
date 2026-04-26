from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.settings import Settings


class FakeUploadRepo:
    def create_upload(self, **kwargs):
        return type("Upload", (), kwargs)


class FakeStorage:
    def __init__(self):
        self.ensure_bucket_called = False

    def build_object_key(self, upload_id, filename):
        return f"uploads/{upload_id}/{filename}"

    def create_presigned_put_url(self, object_key):
        return f"http://minio/audio/{object_key}?signature=test"

    async def ensure_bucket(self):
        self.ensure_bucket_called = True


def make_test_app(storage=None):
    return create_app(
        settings=Settings(),
        upload_repo=FakeUploadRepo(),
        storage=storage or FakeStorage(),
        now=lambda: datetime(2026, 4, 21, tzinfo=timezone.utc),
        new_id=lambda: "upload_1",
    )


@pytest.mark.anyio
async def test_create_upload_returns_presigned_url():
    async with AsyncClient(transport=ASGITransport(app=make_test_app()), base_url="http://test") as client:
        response = await client.post(
            "/v1/uploads",
            json={"filename": "audio.wav", "content_type": "audio/wav"},
        )

    assert response.status_code == 200
    assert response.json()["upload_id"] == "upload_1"
    assert response.json()["method"] == "PUT"


@pytest.mark.anyio
async def test_create_upload_accepts_request_without_size_bytes():
    async with AsyncClient(transport=ASGITransport(app=make_test_app()), base_url="http://test") as client:
        response = await client.post(
            "/v1/uploads",
            json={"filename": "audio.wav", "content_type": "audio/wav"},
        )

    assert response.status_code == 200


@pytest.mark.anyio
async def test_app_ensures_bucket_on_startup():
    storage = FakeStorage()
    app = make_test_app(storage=storage)

    async with app.router.lifespan_context(app):
        assert storage.ensure_bucket_called is True
