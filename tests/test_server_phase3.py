"""API tests for Phase 3: knowledge graph, workspace deletion, audit, quotas."""

import io

import numpy as np

from tests.conftest import auth, register


def _first_ws(client, token):
    return client.get("/api/workspaces", headers=auth(token)).json()[0]["id"]


def _session_factory():
    from telesearch.server.db import get_session_factory

    return get_session_factory()


def _seed_vectors(workspace_id, n=10):
    """Write clustered chunks straight into the workspace's vector store."""
    from telesearch.config import get_settings
    from telesearch.index.store import VectorStore
    from telesearch.models import Chunk

    store = VectorStore(get_settings().workspace_db_path(workspace_id), dim=8)
    chunks, vecs = [], []
    rng = np.random.default_rng(0)
    for i in range(n):
        axis = 0 if i < n // 2 else 1
        v = rng.normal(0, 0.05, 8).astype("float32")
        v[axis] += 1.0
        words = "alpha apple" if axis == 0 else "beta banana"
        chunks.append(Chunk(
            chunk_id=f"col:{i}:text", message_id=i, chat="c", sender="s",
            timestamp=1700000000 + i * 60, date_str="", modality="text",
            content=f"{words} note {i}", collection_id="col",
        ))
        vecs.append(v)
    store.add([c.to_row() for c in chunks], np.array(vecs))
    store.build_fts()


def test_graph_refresh_and_fetch(api_client):
    token = register(api_client)
    ws = _first_ws(api_client, token)
    _seed_vectors(ws, n=10)

    # Empty until built.
    g0 = api_client.get(f"/api/workspaces/{ws}/graph", headers=auth(token)).json()
    assert g0["nodes"] == []

    job = api_client.post(f"/api/workspaces/{ws}/graph/refresh", headers=auth(token)).json()
    assert job["type"] == "graph_refresh"

    from telesearch.server.queue import run_job

    run_job(_session_factory(), job["id"])

    g = api_client.get(f"/api/workspaces/{ws}/graph", headers=auth(token)).json()
    assert g["meta"]["n_chunks"] == 10
    assert g["meta"]["n_topics"] >= 2
    node_id = g["nodes"][0]["id"]

    topic = api_client.get(
        f"/api/workspaces/{ws}/graph/topics/{node_id}", headers=auth(token)
    ).json()
    assert "sample_contents" in topic and topic["size"] >= 1


def test_workspace_deletion_removes_access_and_data(api_client):
    from telesearch.config import get_settings

    token = register(api_client)
    ws = _first_ws(api_client, token)
    _seed_vectors(ws, n=6)
    db_path = get_settings().workspace_db_path(ws)
    assert db_path.exists()

    r = api_client.delete(f"/api/workspaces/{ws}", headers=auth(token))
    assert r.status_code == 204

    # Workspace is gone for the user and its vector data removed from disk.
    assert ws not in {w["id"] for w in api_client.get("/api/workspaces", headers=auth(token)).json()}
    assert api_client.get(f"/api/workspaces/{ws}/sources", headers=auth(token)).status_code in (403, 404)
    assert not db_path.exists()


def test_audit_trail_records_upload(api_client):
    token = register(api_client)
    ws = _first_ws(api_client, token)
    files = {"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    api_client.post(f"/api/workspaces/{ws}/sources", headers=auth(token), files=files)

    rows = api_client.get(f"/api/workspaces/{ws}/audit", headers=auth(token)).json()
    actions = {r["action"] for r in rows}
    assert "source.upload" in actions


def test_upload_size_limit(api_client):
    from telesearch.server.config import get_server_settings

    get_server_settings().max_upload_bytes = 5  # tiny cap for the test
    try:
        token = register(api_client)
        ws = _first_ws(api_client, token)
        files = {"file": ("big.txt", io.BytesIO(b"way more than five bytes"), "text/plain")}
        r = api_client.post(f"/api/workspaces/{ws}/sources", headers=auth(token), files=files)
        assert r.status_code == 413
        assert api_client.get(f"/api/workspaces/{ws}/sources", headers=auth(token)).json() == []
    finally:
        get_server_settings().max_upload_bytes = 512 * 1024 * 1024


def test_source_quota_limit(api_client):
    from telesearch.server.config import get_server_settings

    get_server_settings().max_sources_per_workspace = 1
    try:
        token = register(api_client)
        ws = _first_ws(api_client, token)
        f1 = {"file": ("a.txt", io.BytesIO(b"one"), "text/plain")}
        assert api_client.post(f"/api/workspaces/{ws}/sources", headers=auth(token), files=f1).status_code == 200
        f2 = {"file": ("b.txt", io.BytesIO(b"two"), "text/plain")}
        assert api_client.post(f"/api/workspaces/{ws}/sources", headers=auth(token), files=f2).status_code == 409
    finally:
        get_server_settings().max_sources_per_workspace = 100
