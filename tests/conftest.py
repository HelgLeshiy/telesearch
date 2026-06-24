"""Shared pytest fixtures for the HTTP service tests."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    """A TestClient bound to a fresh temp SQLite DB + local blobs, worker off."""
    monkeypatch.setenv("TELESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TELESEARCH_DATABASE_URL", f"sqlite:///{tmp_path/'t.db'}")
    monkeypatch.setenv("TELESEARCH_SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("TELESEARCH_WORKER_INLINE", "false")
    monkeypatch.setenv("TELESEARCH_DEVICE", "cpu")

    import telesearch.config as cfg
    import telesearch.server.config as scfg
    import telesearch.server.db as sdb

    cfg._settings = None
    scfg.get_server_settings.cache_clear()
    sdb.reset_engine()

    from telesearch.server.app import create_app

    app = create_app(scfg.get_server_settings())
    with TestClient(app) as c:
        yield c

    sdb.reset_engine()
    cfg._settings = None
    scfg.get_server_settings.cache_clear()


def register(client, email="a@example.com", password="password123", name="A"):
    r = client.post(
        "/api/auth/register",
        json={"email": email, "password": password, "name": name},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def auth(token):
    return {"Authorization": f"Bearer {token}"}
