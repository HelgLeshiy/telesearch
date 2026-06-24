"""Blob storage for uploads with pluggable backends.

* ``LocalBlobStore`` (default): filesystem under ``data_dir/blobs``, with optional
  at-rest encryption (Fernet) and a signed "presigned" PUT URL served by the API.
* ``S3BlobStore``: S3-compatible object storage with real presigned PUT URLs
  (requires the ``s3`` extra / boto3).

``materialize`` returns a local directory the ingestion parsers can read,
decrypting and unpacking archives (zip/tar) safely first.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Protocol

from itsdangerous import BadSignature, URLSafeTimedSerializer

from .config import ServerSettings, get_server_settings

_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2")
_UPLOAD_SALT = "telesearch.upload"


def _is_archive(name: str) -> bool:
    low = name.lower()
    return any(low.endswith(s) for s in _ARCHIVE_SUFFIXES)


def _safe_extract_zip(path: Path, dest: Path) -> None:
    with zipfile.ZipFile(path) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise ValueError(f"unsafe path in archive: {member}")
        zf.extractall(dest)


def _safe_extract_tar(path: Path, dest: Path) -> None:
    with tarfile.open(path) as tf:
        for member in tf.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise ValueError(f"unsafe path in archive: {member.name}")
        tf.extractall(dest)


def _unpack_if_archive(base: Path, key_dir: Path) -> Path:
    entries = [p for p in base.iterdir() if p.is_file()] if base.exists() else []
    if len(entries) == 1 and _is_archive(entries[0].name):
        extracted = key_dir / "extracted"
        if extracted.exists():
            shutil.rmtree(extracted, ignore_errors=True)
        extracted.mkdir(parents=True, exist_ok=True)
        archive = entries[0]
        if archive.name.lower().endswith(".zip"):
            _safe_extract_zip(archive, extracted)
        else:
            _safe_extract_tar(archive, extracted)
        return extracted
    return base


class BlobStore(Protocol):
    def save(self, blob_key: str, filename: str, fileobj: BinaryIO) -> int: ...

    def save_bytes(self, blob_key: str, filename: str, data: bytes) -> int: ...

    def materialize(self, blob_key: str) -> Path: ...

    def content_hash(self, blob_key: str) -> str: ...

    def delete(self, blob_key: str) -> None: ...

    def presign_put(self, blob_key: str, filename: str) -> dict: ...


class LocalBlobStore:
    """Filesystem blobs at ``<root>/<blob_key>/raw/<filename>`` (optionally encrypted)."""

    def __init__(self, root: Path, *, encryption_key: str = "", secret_key: str = ""):
        self.root = Path(root)
        self._fernet = None
        if encryption_key:
            from cryptography.fernet import Fernet

            self._fernet = Fernet(encryption_key.encode())
        self._serializer = URLSafeTimedSerializer(secret_key or "dev", salt=_UPLOAD_SALT)

    def _raw_dir(self, blob_key: str) -> Path:
        return self.root / blob_key / "raw"

    def _store_name(self, filename: str) -> str:
        safe = Path(filename).name or "upload"
        return safe + ".enc" if self._fernet else safe

    def save_bytes(self, blob_key: str, filename: str, data: bytes) -> int:
        raw = self._raw_dir(blob_key)
        raw.mkdir(parents=True, exist_ok=True)
        payload = self._fernet.encrypt(data) if self._fernet else data
        (raw / self._store_name(filename)).write_bytes(payload)
        return len(data)

    def save(self, blob_key: str, filename: str, fileobj: BinaryIO) -> int:
        # Read fully so encryption/length are straightforward; uploads are bounded
        # by the size quota enforced upstream.
        data = fileobj.read()
        return self.save_bytes(blob_key, filename, data)

    def _decrypt_into(self, blob_key: str) -> Path:
        raw = self._raw_dir(blob_key)
        files = [p for p in raw.iterdir() if p.is_file()] if raw.exists() else []
        if not self._fernet:
            return raw
        work = self.root / blob_key / "work"
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        for p in files:
            data = self._fernet.decrypt(p.read_bytes())
            name = p.name[:-4] if p.name.endswith(".enc") else p.name
            (work / name).write_bytes(data)
        return work

    def materialize(self, blob_key: str) -> Path:
        base = self._decrypt_into(blob_key)
        return _unpack_if_archive(base, self.root / blob_key)

    def content_hash(self, blob_key: str) -> str:
        raw = self._raw_dir(blob_key)
        files = sorted(p for p in raw.iterdir() if p.is_file()) if raw.exists() else []
        if not files:
            return ""
        h = hashlib.sha256()
        for p in files:
            data = p.read_bytes()
            if self._fernet:
                data = self._fernet.decrypt(data)
            h.update(data)
        return h.hexdigest()

    def delete(self, blob_key: str) -> None:
        target = self.root / blob_key
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

    # --- presigned local upload (served by the uploads router) ---
    def presign_put(self, blob_key: str, filename: str) -> dict:
        token = self._serializer.dumps({"k": blob_key, "f": filename})
        return {"method": "PUT", "url": f"/api/uploads/{token}", "backend": "local"}

    def verify_upload_token(self, token: str, max_age: int = 3600) -> tuple[str, str]:
        try:
            data = self._serializer.loads(token, max_age=max_age)
        except BadSignature as exc:
            raise ValueError("invalid or expired upload token") from exc
        return data["k"], data["f"]


class S3BlobStore:  # pragma: no cover - requires boto3 + a live bucket
    """S3-compatible blob storage with real presigned PUT URLs."""

    def __init__(self, settings: ServerSettings):
        import boto3

        self.settings = settings
        self.bucket = settings.s3_bucket
        self._s3 = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url or None,
            region_name=settings.s3_region or None,
        )

    def _prefix(self, blob_key: str) -> str:
        return f"{blob_key}/raw"

    def save_bytes(self, blob_key: str, filename: str, data: bytes) -> int:
        name = Path(filename).name or "upload"
        self._s3.put_object(Bucket=self.bucket, Key=f"{self._prefix(blob_key)}/{name}", Body=data)
        return len(data)

    def save(self, blob_key: str, filename: str, fileobj) -> int:
        return self.save_bytes(blob_key, filename, fileobj.read())

    def materialize(self, blob_key: str) -> Path:
        import tempfile

        dest = Path(tempfile.mkdtemp(prefix="ts-s3-"))
        prefix = self._prefix(blob_key)
        resp = self._s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            local = dest / Path(key).name
            self._s3.download_file(self.bucket, key, str(local))
        return _unpack_if_archive(dest, dest.parent / (dest.name + "-x"))

    def content_hash(self, blob_key: str) -> str:
        return ""  # rely on ETag/skip dedup for S3 in this iteration

    def delete(self, blob_key: str) -> None:
        resp = self._s3.list_objects_v2(Bucket=self.bucket, Prefix=blob_key)
        for obj in resp.get("Contents", []):
            self._s3.delete_object(Bucket=self.bucket, Key=obj["Key"])

    def presign_put(self, blob_key: str, filename: str) -> dict:
        name = Path(filename).name or "upload"
        url = self._s3.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.bucket, "Key": f"{self._prefix(blob_key)}/{name}"},
            ExpiresIn=self.settings.s3_presign_expiry,
        )
        return {"method": "PUT", "url": url, "backend": "s3"}


def get_blob_store(settings: ServerSettings | None = None) -> BlobStore:
    settings = settings or get_server_settings()
    if settings.blob_backend == "s3":  # pragma: no cover - optional backend
        return S3BlobStore(settings)
    return LocalBlobStore(
        settings.blob_dir,
        encryption_key=settings.blob_encryption_key,
        secret_key=settings.secret_key,
    )


def bytes_to_fileobj(data: bytes) -> BinaryIO:
    return BytesIO(data)
