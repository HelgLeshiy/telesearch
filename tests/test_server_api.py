"""API tests for the Phase 1 service backbone (no models/GPU required).

Uses a temp SQLite DB and local blob storage; the indexing job's heavy
``index_source`` is stubbed so the worker path is exercised without loading
embedding models.
"""

import io
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
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

    settings = scfg.get_server_settings()
    app = create_app(settings)
    with TestClient(app) as c:
        yield c

    sdb.reset_engine()
    cfg._settings = None
    scfg.get_server_settings.cache_clear()


def _register(client, email="a@example.com", password="password123", name="A"):
    r = client.post("/api/auth/register", json={"email": email, "password": password, "name": name})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_register_login_me(client):
    token = _register(client)
    me = client.get("/api/auth/me", headers=_auth(token))
    assert me.status_code == 200
    assert me.json()["email"] == "a@example.com"

    login = client.post(
        "/api/auth/login", json={"email": "a@example.com", "password": "password123"}
    )
    assert login.status_code == 200
    assert login.json()["access_token"]


def test_duplicate_registration_conflict(client):
    _register(client)
    r = client.post(
        "/api/auth/register", json={"email": "a@example.com", "password": "password123"}
    )
    assert r.status_code == 409


def test_auth_required(client):
    assert client.get("/api/workspaces").status_code == 401
    assert client.get("/api/auth/me", headers=_auth("garbage")).status_code == 401


def test_personal_workspace_created_on_register(client):
    token = _register(client)
    r = client.get("/api/workspaces", headers=_auth(token))
    assert r.status_code == 200
    wss = r.json()
    assert len(wss) == 1
    assert wss[0]["name"] == "Personal"
    assert wss[0]["role"] == "owner"


def test_workspace_isolation(client):
    ta = _register(client, email="a@example.com")
    tb = _register(client, email="b@example.com")
    ws_a = client.get("/api/workspaces", headers=_auth(ta)).json()[0]["id"]
    # B has no access to A's workspace.
    r = client.get(f"/api/workspaces/{ws_a}/sources", headers=_auth(tb))
    assert r.status_code == 403


def test_upload_creates_source_and_worker_indexes(client, monkeypatch):
    # Stub the heavy indexer so the worker runs without loading models.
    import telesearch.server.queue as q

    def fake_index_source(root, settings, **kwargs):
        return SimpleNamespace(parser="telegram", messages=3, chunks=4, collection_id="c", db_path="x")

    monkeypatch.setattr(q, "index_source", fake_index_source)

    token = _register(client)
    ws = client.get("/api/workspaces", headers=_auth(token)).json()[0]["id"]

    files = {"file": ("notes.txt", io.BytesIO(b"hello world notes"), "text/plain")}
    r = client.post(
        f"/api/workspaces/{ws}/sources",
        headers=_auth(token),
        files=files,
        data={"name": "My notes", "kind": ""},
    )
    assert r.status_code == 200, r.text
    source = r.json()
    assert source["status"] == "uploaded"
    assert source["collection_id"] == source["id"]

    # A pending ingest job exists; run it directly via the worker entrypoint.
    jobs = client.get(f"/api/workspaces/{ws}/jobs", headers=_auth(token)).json()
    assert len(jobs) == 1 and jobs[0]["state"] == "pending"

    q.run_job(sessionfactory(), jobs[0]["id"])

    job = client.get(f"/api/workspaces/{ws}/jobs/{jobs[0]['id']}", headers=_auth(token)).json()
    assert job["state"] == "completed", job
    src = client.get(f"/api/workspaces/{ws}/sources", headers=_auth(token)).json()[0]
    assert src["status"] == "ready"


def test_search_endpoint_serializes_results(client, monkeypatch):
    # Stub the scoped service so search routing is tested without models.
    import telesearch.server.routers.search as sr
    from telesearch.search.retriever import SearchResult

    class FakeService:
        def search(self, query):
            assert query.text == "rome"
            return [
                SearchResult(
                    chunk_id="c:1:text", message_id=1, sender="Alice",
                    date_str="2024-01-01", modality="text", content="Ritz in Rome",
                    media_path=None, score=0.9, chat="c",
                )
            ]

    monkeypatch.setattr(sr, "_scoped_service", lambda scope, settings: FakeService())

    token = _register(client)
    ws = client.get("/api/workspaces", headers=_auth(token)).json()[0]["id"]
    r = client.post(
        f"/api/workspaces/{ws}/search", headers=_auth(token), json={"query": "rome", "k": 5}
    )
    assert r.status_code == 200, r.text
    hits = r.json()
    assert len(hits) == 1
    assert hits[0]["content"] == "Ritz in Rome"
    assert hits[0]["score"] == pytest.approx(0.9)


def sessionfactory():
    from telesearch.server.db import get_session_factory

    return get_session_factory()
