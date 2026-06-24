"""Tests for blob storage: at-rest encryption and archive materialization."""

import io
import zipfile

from telesearch.server.blobs import LocalBlobStore


def test_local_blob_roundtrip_plain(tmp_path):
    store = LocalBlobStore(tmp_path)
    store.save("ws/s1", "note.txt", io.BytesIO(b"hello world"))
    out = store.materialize("ws/s1")
    assert (out / "note.txt").read_bytes() == b"hello world"
    import hashlib

    assert store.content_hash("ws/s1") == hashlib.sha256(b"hello world").hexdigest()


def test_local_blob_encryption_at_rest(tmp_path):
    from cryptography.fernet import Fernet

    key = Fernet.generate_key().decode()
    store = LocalBlobStore(tmp_path, encryption_key=key)
    store.save_bytes("ws/s2", "secret.txt", b"top secret data")

    # On disk it is encrypted (no plaintext, .enc suffix).
    raw_files = list((tmp_path / "ws/s2/raw").iterdir())
    assert raw_files and raw_files[0].name.endswith(".enc")
    assert b"top secret data" not in raw_files[0].read_bytes()

    # Materialize decrypts to plaintext for the parsers.
    out = store.materialize("ws/s2")
    assert (out / "secret.txt").read_bytes() == b"top secret data"


def test_local_blob_unpacks_archive(tmp_path):
    store = LocalBlobStore(tmp_path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("result.json", '{"messages": []}')
    store.save_bytes("ws/s3", "export.zip", buf.getvalue())
    out = store.materialize("ws/s3")
    assert (out / "result.json").exists()


def test_presign_token_roundtrip(tmp_path):
    store = LocalBlobStore(tmp_path, secret_key="k")
    info = store.presign_put("ws/s4", "a.txt")
    assert info["url"].startswith("/api/uploads/")
    token = info["url"].rsplit("/", 1)[1]
    blob_key, filename = store.verify_upload_token(token)
    assert blob_key == "ws/s4" and filename == "a.txt"
