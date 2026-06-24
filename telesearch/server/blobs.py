"""Blob storage for uploads (raw files) with a pluggable backend.

Default is local filesystem under ``data_dir/blobs``; an S3 backend can be added
behind the same interface. ``materialize`` returns a local directory the
ingestion parsers can read, unpacking archives (zip/tar) safely first.
"""

from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import BinaryIO, Protocol

from .config import ServerSettings, get_server_settings

_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2")


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


class BlobStore(Protocol):
    def save(self, blob_key: str, filename: str, fileobj: BinaryIO) -> int: ...

    def materialize(self, blob_key: str) -> Path: ...

    def delete(self, blob_key: str) -> None: ...


class LocalBlobStore:
    """Stores each upload under ``<root>/<blob_key>/raw/<filename>``."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _raw_dir(self, blob_key: str) -> Path:
        return self.root / blob_key / "raw"

    def save(self, blob_key: str, filename: str, fileobj: BinaryIO) -> int:
        raw = self._raw_dir(blob_key)
        raw.mkdir(parents=True, exist_ok=True)
        # Strip any directory components from the client-supplied name.
        safe_name = Path(filename).name or "upload"
        dest = raw / safe_name
        written = 0
        with dest.open("wb") as out:
            while True:
                chunk = fileobj.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
        return written

    def materialize(self, blob_key: str) -> Path:
        """Return a directory the parsers can read (archives unpacked)."""
        raw = self._raw_dir(blob_key)
        entries = [p for p in raw.iterdir() if p.is_file()] if raw.exists() else []
        if len(entries) == 1 and _is_archive(entries[0].name):
            extracted = self.root / blob_key / "extracted"
            if not extracted.exists():
                extracted.mkdir(parents=True, exist_ok=True)
                archive = entries[0]
                if archive.name.lower().endswith(".zip"):
                    _safe_extract_zip(archive, extracted)
                else:
                    _safe_extract_tar(archive, extracted)
            return extracted
        return raw

    def delete(self, blob_key: str) -> None:
        target = self.root / blob_key
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)


def get_blob_store(settings: ServerSettings | None = None) -> BlobStore:
    settings = settings or get_server_settings()
    if settings.blob_backend == "s3":  # pragma: no cover - optional backend
        raise NotImplementedError(
            "S3 blob backend not yet implemented; use blob_backend=local"
        )
    return LocalBlobStore(settings.blob_dir)
