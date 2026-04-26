import re
from inspect import isawaitable
from pathlib import Path, PurePath
from typing import Any

from botocore.config import Config
from botocore.exceptions import ClientError

from app.settings import Settings, get_settings

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str) -> str:
    name = PurePath(filename).name
    cleaned = _SAFE_FILENAME_RE.sub("_", name).strip("._")
    return cleaned or "audio"


def build_object_key(upload_id: str, filename: str) -> str:
    return f"uploads/{upload_id}/{sanitize_filename(filename)}"


def build_normalized_artifact_key(job_id: str) -> str:
    return f"artifacts/{job_id}/normalized.wav"


def create_aioboto3_session():
    import aioboto3

    return aioboto3.Session()


class AsyncS3Storage:
    def __init__(self, settings: Settings | None = None, session: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self.session = session or create_aioboto3_session()

    def build_object_key(self, upload_id: str, filename: str) -> str:
        return build_object_key(upload_id, filename)

    def build_normalized_artifact_key(self, job_id: str) -> str:
        return build_normalized_artifact_key(job_id)

    def _resolve_endpoint_url(self, endpoint: str) -> str:
        scheme = "https" if self.settings.minio_secure else "http"
        endpoint_url = endpoint
        if not endpoint_url.startswith(("http://", "https://")):
            endpoint_url = f"{scheme}://{endpoint_url}"
        return endpoint_url

    def client(self, *, public: bool = False):
        endpoint = self.settings.minio_public_endpoint if public else self.settings.minio_endpoint
        endpoint_url = self._resolve_endpoint_url(endpoint)
        return self.session.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=self.settings.minio_access_key,
            aws_secret_access_key=self.settings.minio_secret_key,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )

    async def create_presigned_put_url(self, object_key: str) -> str:
        async with self.client(public=True) as client:
            result = client.generate_presigned_url(
                ClientMethod="put_object",
                Params={"Bucket": self.settings.minio_bucket, "Key": object_key},
                ExpiresIn=self.settings.presigned_upload_ttl_sec,
            )
            if isawaitable(result):
                return await result
            return result

    async def ensure_bucket(self) -> None:
        async with self.client() as client:
            try:
                await client.head_bucket(Bucket=self.settings.minio_bucket)
            except ClientError as exc:
                error_code = exc.response.get("Error", {}).get("Code")
                status_code = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                if error_code not in {"404", "NoSuchBucket", "NotFound"} and status_code != 404:
                    raise
                await client.create_bucket(Bucket=self.settings.minio_bucket)

    async def download_to_file(self, object_key: str, path: Path) -> None:
        import aiofiles

        async with self.client() as client:
            response = await client.get_object(Bucket=self.settings.minio_bucket, Key=object_key)
            body = response["Body"]
            async with aiofiles.open(path, "wb") as file:
                while chunk := await body.read(1024 * 1024):
                    await file.write(chunk)

    async def upload_file(self, path: Path, object_key: str) -> None:
        import aiofiles

        async with aiofiles.open(path, "rb") as file:
            data = await file.read()
        async with self.client() as client:
            await client.put_object(Bucket=self.settings.minio_bucket, Key=object_key, Body=data)

    async def object_exists(self, object_key: str) -> bool:
        from botocore.exceptions import ClientError

        async with self.client() as client:
            try:
                await client.head_object(Bucket=self.settings.minio_bucket, Key=object_key)
            except ClientError as exc:
                status_code = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
                if status_code == 404:
                    return False
                raise
        return True
