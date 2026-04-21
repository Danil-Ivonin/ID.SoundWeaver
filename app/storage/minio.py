import re
from datetime import timedelta
from pathlib import PurePath

from minio import Minio

from app.settings import Settings, get_settings

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str) -> str:
    name = PurePath(filename).name
    cleaned = _SAFE_FILENAME_RE.sub("_", name).strip("._")
    return cleaned or "audio"


def build_object_key(upload_id: str, filename: str) -> str:
    return f"uploads/{upload_id}/{sanitize_filename(filename)}"


def create_minio_client(settings: Settings | None = None) -> Minio:
    cfg = settings or get_settings()
    return Minio(
        endpoint=cfg.minio_endpoint,
        access_key=cfg.minio_access_key,
        secret_key=cfg.minio_secret_key,
        secure=cfg.minio_secure,
    )


def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)


def create_presigned_put_url(client, bucket: str, object_key: str, ttl_sec: int) -> str:
    return client.presigned_put_object(
        bucket_name=bucket,
        object_name=object_key,
        expires=timedelta(seconds=ttl_sec),
    )
