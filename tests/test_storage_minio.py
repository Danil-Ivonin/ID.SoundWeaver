from botocore.exceptions import ClientError
import pytest

from app.settings import Settings
from app.storage.minio import AsyncS3Storage, build_object_key


class FakeS3Client:
    def __init__(self, *, presigned_url: str = "http://example.test/upload", head_bucket_error=None):
        self.presigned_url = presigned_url
        self.head_bucket_error = head_bucket_error
        self.generate_calls = []
        self.head_bucket_calls = []
        self.create_bucket_calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def generate_presigned_url(self, *, ClientMethod, Params, ExpiresIn):
        self.generate_calls.append((ClientMethod, Params, ExpiresIn))
        return self.presigned_url

    async def head_bucket(self, *, Bucket):
        self.head_bucket_calls.append(Bucket)
        if self.head_bucket_error is not None:
            raise self.head_bucket_error

    async def create_bucket(self, *, Bucket):
        self.create_bucket_calls.append(Bucket)


class FakeSession:
    def __init__(self, client):
        self.client_instance = client
        self.client_calls = []

    def client(self, service_name, **kwargs):
        self.client_calls.append((service_name, kwargs))
        return self.client_instance


def test_build_object_key_sanitizes_filename():
    key = build_object_key("upload_1", "../my meeting.wav")

    assert key == "uploads/upload_1/my_meeting.wav"


@pytest.mark.anyio
async def test_async_storage_uses_public_endpoint_and_s3v4_for_presigned_put_url():
    client = FakeS3Client()
    session = FakeSession(client)
    storage = AsyncS3Storage(
        Settings(
            _env_file=None,
            minio_endpoint="minio:9000",
            minio_public_endpoint="http://localhost:9000",
            minio_bucket="audio",
        ),
        session=session,
    )

    url = await storage.create_presigned_put_url("uploads/upload_1/audio.wav")

    assert url == "http://example.test/upload"
    assert session.client_calls[0][0] == "s3"
    assert session.client_calls[0][1]["endpoint_url"] == "http://localhost:9000"
    assert session.client_calls[0][1]["config"].signature_version == "s3v4"
    assert client.generate_calls == [
        (
            "put_object",
            {"Bucket": "audio", "Key": "uploads/upload_1/audio.wav"},
            900,
        )
    ]


@pytest.mark.anyio
async def test_async_storage_creates_missing_bucket():
    error = ClientError(
        {
            "Error": {"Code": "404", "Message": "missing"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        },
        "HeadBucket",
    )
    client = FakeS3Client(head_bucket_error=error)
    session = FakeSession(client)
    storage = AsyncS3Storage(
        Settings(_env_file=None, minio_endpoint="minio:9000", minio_bucket="audio"),
        session=session,
    )

    await storage.ensure_bucket()

    assert client.head_bucket_calls == ["audio"]
    assert client.create_bucket_calls == ["audio"]
