"""API tests for Phase 4 hardening: presign, dedup, SSE, metrics, OIDC, queue."""

import io

from types import SimpleNamespace

from tests.conftest import auth, register


def _first_ws(client, token):
    return client.get("/api/workspaces", headers=auth(token)).json()[0]["id"]


def _session_factory():
    from telesearch.server.db import get_session_factory

    return get_session_factory()


# --------------------------------------------------------------------------- #
# Presigned upload flow
# --------------------------------------------------------------------------- #
def test_presigned_upload_flow(api_client, monkeypatch):
    token = register(api_client)
    ws = _first_ws(api_client, token)

    pres = api_client.post(
        f"/api/workspaces/{ws}/sources/presign",
        headers=auth(token),
        json={"filename": "notes.txt", "name": "Notes"},
    ).json()
    assert pres["upload"]["url"].startswith("/api/uploads/")

    put = api_client.put(pres["upload"]["url"], content=b"hello presigned content")
    assert put.status_code == 200, put.text

    done = api_client.post(
        f"/api/workspaces/{ws}/sources/{pres['source_id']}/complete",
        headers=auth(token),
        json={"index_media": False},
    )
    assert done.status_code == 200, done.text
    assert done.json()["status"] == "uploaded"

    jobs = api_client.get(f"/api/workspaces/{ws}/jobs", headers=auth(token)).json()
    assert any(j["type"] == "ingest" for j in jobs)


# --------------------------------------------------------------------------- #
# Dedup
# --------------------------------------------------------------------------- #
def test_duplicate_upload_is_detected(api_client):
    token = register(api_client)
    ws = _first_ws(api_client, token)
    payload = {"file": ("a.txt", io.BytesIO(b"identical bytes"), "text/plain")}
    first = api_client.post(f"/api/workspaces/{ws}/sources", headers=auth(token), files=payload).json()
    assert first["status"] == "uploaded"

    payload2 = {"file": ("b.txt", io.BytesIO(b"identical bytes"), "text/plain")}
    second = api_client.post(f"/api/workspaces/{ws}/sources", headers=auth(token), files=payload2).json()
    assert second["status"] == "duplicate"


# --------------------------------------------------------------------------- #
# SSE job progress
# --------------------------------------------------------------------------- #
def test_job_sse_stream(api_client, monkeypatch):
    import telesearch.server.queue as q

    monkeypatch.setattr(
        q, "index_source",
        lambda *a, **k: SimpleNamespace(parser="x", messages=1, chunks=1, collection_id="c", db_path="d"),
    )
    token = register(api_client)
    ws = _first_ws(api_client, token)
    files = {"file": ("n.txt", io.BytesIO(b"content"), "text/plain")}
    api_client.post(f"/api/workspaces/{ws}/sources", headers=auth(token), files=files)
    job = api_client.get(f"/api/workspaces/{ws}/jobs", headers=auth(token)).json()[0]
    q.run_job(_session_factory(), job["id"])

    r = api_client.get(f"/api/workspaces/{ws}/jobs/{job['id']}/events", headers=auth(token))
    assert r.status_code == 200
    assert "completed" in r.text and "data:" in r.text


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_metrics_endpoint(api_client):
    register(api_client)
    r = api_client.get("/api/metrics")
    assert r.status_code == 200
    assert "telesearch_users_total" in r.text
    assert "telesearch_jobs" in r.text


# --------------------------------------------------------------------------- #
# Queue lanes / priority / quota
# --------------------------------------------------------------------------- #
def test_queue_priority_and_lane(api_client):
    from telesearch.server.queue import Worker, enqueue

    token = register(api_client)
    ws = _first_ws(api_client, token)
    sf = _session_factory()
    db = sf()
    try:
        enqueue(db, workspace_id=ws, job_type="ingest", lane="cpu", priority=0)
        hi = enqueue(db, workspace_id=ws, job_type="ingest", lane="cpu", priority=10)
        gpu = enqueue(db, workspace_id=ws, job_type="ingest", lane="gpu", priority=99)
    finally:
        db.close()

    # CPU-only worker ignores the gpu job and takes the highest-priority cpu job.
    cpu_worker = Worker(sf)
    cpu_worker.server_settings.worker_lanes = "cpu"
    assert cpu_worker._claim_pending() == hi.id

    gpu_worker = Worker(sf)
    gpu_worker.server_settings.worker_lanes = "gpu"
    assert gpu_worker._claim_pending() == gpu.id


def test_pending_job_quota(api_client):
    from telesearch.server.config import get_server_settings

    get_server_settings().max_pending_jobs_per_workspace = 1
    try:
        token = register(api_client)
        ws = _first_ws(api_client, token)
        f1 = {"file": ("a.txt", io.BytesIO(b"one"), "text/plain")}
        assert api_client.post(f"/api/workspaces/{ws}/sources", headers=auth(token), files=f1).status_code == 200
        f2 = {"file": ("b.txt", io.BytesIO(b"two"), "text/plain")}
        assert api_client.post(f"/api/workspaces/{ws}/sources", headers=auth(token), files=f2).status_code == 429
    finally:
        get_server_settings().max_pending_jobs_per_workspace = 50


# --------------------------------------------------------------------------- #
# OIDC (stubbed IdP)
# --------------------------------------------------------------------------- #
def test_oidc_callback_creates_user(api_client, monkeypatch):
    import telesearch.server.routers.oidc as oidc
    from telesearch.server.config import get_server_settings

    s = get_server_settings()
    s.oidc_enabled = True
    s.oidc_issuer = "https://idp.example.com"
    s.oidc_client_id = "cid"
    s.oidc_redirect_uri = "https://app/cb"
    try:
        monkeypatch.setattr(oidc, "_discover", lambda issuer: {
            "authorization_endpoint": "https://idp/auth",
            "token_endpoint": "https://idp/token",
            "userinfo_endpoint": "https://idp/userinfo",
        })
        monkeypatch.setattr(oidc, "_exchange_code", lambda cfg, code, st: {"access_token": "AT"})
        monkeypatch.setattr(oidc, "_userinfo", lambda cfg, at: {
            "sub": "abc123", "email": "sso@example.com", "name": "SSO User",
        })

        login = api_client.get("/api/auth/oidc/login")
        assert login.status_code == 200 and "authorization_url" in login.json()

        cb = api_client.get("/api/auth/oidc/callback", params={"code": "xyz"})
        assert cb.status_code == 200, cb.text
        tok = cb.json()["access_token"]
        me = api_client.get("/api/auth/me", headers=auth(tok)).json()
        assert me["email"] == "sso@example.com"
    finally:
        s.oidc_enabled = False


def test_oidc_disabled_returns_404(api_client):
    assert api_client.get("/api/auth/oidc/login").status_code == 404


# --------------------------------------------------------------------------- #
# G3 gated endpoint
# --------------------------------------------------------------------------- #
def test_g3_disabled_by_default(api_client):
    token = register(api_client)
    ws = _first_ws(api_client, token)
    r = api_client.post(f"/api/workspaces/{ws}/graph/g3", headers=auth(token))
    assert r.status_code == 403
