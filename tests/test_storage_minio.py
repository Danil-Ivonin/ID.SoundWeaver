from app.storage.minio import build_object_key, create_presigned_put_url


class FakeMinio:
    def __init__(self):
        self.calls = []

    def presigned_put_object(self, bucket_name, object_name, expires):
        self.calls.append((bucket_name, object_name, expires.total_seconds()))
        return f"http://minio/{bucket_name}/{object_name}?signature=test"


def test_build_object_key_sanitizes_filename():
    key = build_object_key("upload_1", "../my meeting.wav")

    assert key == "uploads/upload_1/my_meeting.wav"


def test_create_presigned_put_url_uses_bucket_and_ttl():
    client = FakeMinio()

    url = create_presigned_put_url(
        client=client,
        bucket="audio",
        object_key="uploads/upload_1/audio.wav",
        ttl_sec=900,
    )

    assert url.startswith("http://minio/audio/uploads/upload_1/audio.wav")
    assert client.calls[0][0] == "audio"
    assert client.calls[0][2] == 900
