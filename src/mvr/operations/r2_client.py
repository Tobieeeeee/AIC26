"""R2/S3-compatible object storage client for Cloudflare R2.

Credentials come only from environment variables — nothing is ever committed.
"""
from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, BinaryIO

LOGGER = logging.getLogger(__name__)

# Required environment variables
R2_ACCOUNT_ID = "R2_ACCOUNT_ID"
R2_ACCESS_KEY_ID = "R2_ACCESS_KEY_ID"
R2_SECRET_ACCESS_KEY = "R2_SECRET_ACCESS_KEY"
R2_BUCKET = "R2_BUCKET"
R2_ENDPOINT = "R2_ENDPOINT"

_REQUIRED_ENV_VARS = (R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET, R2_ENDPOINT)

# Retry defaults
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_RETRY_BACKOFF = 2.0

# Multipart threshold (5 MiB — Cloudflare's minimum part size)
MULTIPART_THRESHOLD = 5 * 1024 * 1024
MULTIPART_CHUNK_SIZE = 8 * 1024 * 1024


class R2ConfigError(ValueError):
    """Missing or invalid R2 configuration."""


class R2OperationError(RuntimeError):
    """An R2 operation failed after retries."""


class R2ChecksumError(R2OperationError):
    """Downloaded content does not match the expected checksum."""


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise R2ConfigError(f"environment variable {name} is required but empty or unset")
    return value


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


class R2Client:
    """Reusable S3-compatible client for Cloudflare R2.

    Uses boto3 under the hood. The client is created lazily to avoid import
    errors when boto3 is not installed (e.g. in unit tests with mocks).
    """

    def __init__(
        self,
        *,
        bucket: str | None = None,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    ) -> None:
        self.bucket = bucket or _require_env(R2_BUCKET)
        self._endpoint_url = endpoint_url or _require_env(R2_ENDPOINT)
        self._access_key_id = access_key_id or _require_env(R2_ACCESS_KEY_ID)
        self._secret_access_key = secret_access_key or _require_env(R2_SECRET_ACCESS_KEY)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise R2ConfigError(
                    "boto3 is required for R2 operations; install with: "
                    'pip install boto3'
                ) from exc
            self._client = boto3.client(
                "s3",
                endpoint_url=self._endpoint_url,
                aws_access_key_id=self._access_key_id,
                aws_secret_access_key=self._secret_access_key,
                region_name="auto",
            )
        return self._client

    def _retry(self, operation: str, fn, *args, **kwargs) -> Any:
        """Execute an S3 operation with exponential backoff retry."""
        delay = self.retry_delay
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    break
                LOGGER.warning(
                    "%s attempt %d/%d failed: %s; retrying in %.1fs",
                    operation, attempt, self.max_retries, exc, delay,
                )
                time.sleep(delay)
                delay *= self.retry_backoff
        raise R2OperationError(
            f"{operation} failed after {self.max_retries} attempts: {last_exc}"
        ) from last_exc

    # ---- Core operations ----

    def head_object(self, key: str) -> dict:
        """Return object metadata (ContentLength, ETag, etc.)."""
        client = self._get_client()
        return self._retry(
            f"HeadObject({key})",
            client.head_object, Bucket=self.bucket, Key=key,
        )

    def object_exists(self, key: str) -> bool:
        """Check if an object exists without downloading it."""
        try:
            self.head_object(key)
            return True
        except R2OperationError:
            return False

    def list_objects(
        self, prefix: str, *, max_keys: int = 1000
    ) -> list[dict[str, Any]]:
        """List objects under a prefix. Returns list of {Key, Size, ...}."""
        client = self._get_client()
        result: list[dict[str, Any]] = []
        continuation_token: str | None = None
        while True:
            kwargs: dict[str, Any] = {
                "Bucket": self.bucket,
                "Prefix": prefix,
                "MaxKeys": max_keys,
            }
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            response = self._retry(
                f"ListObjects({prefix})", client.list_objects_v2, **kwargs
            )
            result.extend(response.get("Contents", []))
            if not response.get("IsTruncated"):
                break
            continuation_token = response.get("NextContinuationToken")
        return result

    def download_file(
        self,
        key: str,
        local_path: Path,
        *,
        expected_sha256: str | None = None,
    ) -> Path:
        """Download an object to a local path atomically.

        Uses a temporary file and rename to prevent partial downloads from
        being treated as valid.
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        temp_fd, temp_path_str = tempfile.mkstemp(
            dir=local_path.parent, prefix=f".{local_path.name}.", suffix=".tmp"
        )
        temp_path = Path(temp_path_str)
        try:
            os.close(temp_fd)
            client = self._get_client()
            self._retry(
                f"DownloadFile({key})",
                client.download_file, self.bucket, key, str(temp_path),
            )
            if expected_sha256:
                actual = _sha256_file(temp_path)
                if actual != expected_sha256:
                    raise R2ChecksumError(
                        f"SHA256 mismatch for {key}: "
                        f"expected {expected_sha256}, got {actual}"
                    )
            temp_path.replace(local_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        return local_path

    def upload_file(
        self,
        local_path: Path,
        key: str,
        *,
        content_type: str | None = None,
    ) -> dict:
        """Upload a local file to R2. Uses multipart for large files."""
        local_path = Path(local_path)
        if not local_path.is_file():
            raise FileNotFoundError(f"upload source not found: {local_path}")
        client = self._get_client()
        extra_args: dict[str, str] = {}
        if content_type:
            extra_args["ContentType"] = content_type
        file_size = local_path.stat().st_size
        if file_size > MULTIPART_THRESHOLD:
            from boto3.s3.transfer import TransferConfig
            config = TransferConfig(
                multipart_threshold=MULTIPART_THRESHOLD,
                multipart_chunksize=MULTIPART_CHUNK_SIZE,
            )
            self._retry(
                f"UploadFile({key})",
                client.upload_file,
                str(local_path), self.bucket, key,
                ExtraArgs=extra_args or None,
                Config=config,
            )
        else:
            self._retry(
                f"UploadFile({key})",
                client.upload_file,
                str(local_path), self.bucket, key,
                ExtraArgs=extra_args or None,
            )
        return {"key": key, "size": file_size, "sha256": _sha256_file(local_path)}

    def delete_object(self, key: str) -> None:
        """Delete a single object."""
        client = self._get_client()
        self._retry(
            f"DeleteObject({key})",
            client.delete_object, Bucket=self.bucket, Key=key,
        )

    def delete_prefix(self, prefix: str) -> int:
        """Delete all objects under a prefix. Returns count of deleted objects."""
        objects = self.list_objects(prefix)
        for obj in objects:
            self.delete_object(obj["Key"])
        return len(objects)
