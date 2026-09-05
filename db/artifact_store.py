"""Content-addressed S3-compatible artifact storage."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from core.runtime.contracts import ArtifactRef


class ArtifactStore(Protocol):
    async def put(self, content: bytes, content_type: str) -> ArtifactRef: ...
    async def ready(self) -> bool: ...


class S3ArtifactStore:
    def __init__(self, bucket: str, endpoint_url: str | None = None) -> None:
        self.bucket = bucket
        self.endpoint_url = endpoint_url

    async def put(self, content: bytes, content_type: str) -> ArtifactRef:
        digest = hashlib.sha256(content).hexdigest()
        key = f"sha256/{digest}"

        def upload() -> None:
            import boto3

            boto3.client("s3", endpoint_url=self.endpoint_url).put_object(
                Bucket=self.bucket,
                Key=key,
                Body=content,
                ContentType=content_type,
            )

        await asyncio.to_thread(upload)
        return ArtifactRef(
            uri=f"s3://{self.bucket}/{key}",
            content_hash=digest,
            content_type=content_type,
            size_bytes=len(content),
        )

    async def ready(self) -> bool:
        def check() -> None:
            import boto3

            boto3.client("s3", endpoint_url=self.endpoint_url).head_bucket(
                Bucket=self.bucket
            )

        try:
            await asyncio.to_thread(check)
        except Exception:
            return False
        return True


class LocalArtifactStore:
    """Content-addressed artifacts for the single-host SQLite runtime."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser()

    async def put(self, content: bytes, content_type: str) -> ArtifactRef:
        digest = hashlib.sha256(content).hexdigest()
        path = self.directory / "sha256" / digest

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                return
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            try:
                temporary.write_bytes(content)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

        await asyncio.to_thread(write)
        return ArtifactRef(
            uri=path.resolve().as_uri(),
            content_hash=digest,
            content_type=content_type,
            size_bytes=len(content),
        )

    async def ready(self) -> bool:
        def check() -> bool:
            self.directory.mkdir(parents=True, exist_ok=True)
            return self.directory.is_dir() and os.access(self.directory, os.W_OK)

        return await asyncio.to_thread(check)
