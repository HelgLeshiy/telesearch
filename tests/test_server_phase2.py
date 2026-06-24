"""API tests for Phase 2: global fan-out search assembly, presets, guides."""

import io

from tests.conftest import auth, register


def _first_ws(client, token):
    return client.get("/api/workspaces", headers=auth(token)).json()[0]["id"]


def _fake_results():
    from telesearch.search.retriever import SearchResult

    return [
        SearchResult(
            chunk_id="c:1:text", message_id=1, sender="Alice", date_str="d",
            modality="text", content="hit", media_path=None, score=1.0, chat="c",
        )
    ]


def test_global_search_targets_all_member_workspaces(api_client, monkeypatch):
    import telesearch.server.routers.global_search as gs

    captured = {}

    def fake_fan_out(embedder, reranker, targets, query, **kw):
        captured["targets"] = targets
        captured["query"] = query
        return _fake_results()

    monkeypatch.setattr(gs, "fan_out_search", fake_fan_out)
    monkeypatch.setattr(gs, "get_shared_models", lambda settings: (None, None))

    token = register(api_client)
    # Create a second workspace -> user now belongs to two.
    api_client.post("/api/workspaces", headers=auth(token), json={"name": "Work"})

    r = api_client.post("/api/search", headers=auth(token), json={"query": "rome", "k": 5})
    assert r.status_code == 200, r.text
    assert len(captured["targets"]) == 2  # both member workspaces searched
    assert captured["query"] == "rome"
    assert r.json()[0]["content"] == "hit"


def test_global_search_includes_shared_source_from_other_workspace(api_client, monkeypatch):
    import telesearch.server.routers.global_search as gs

    captured = {}

    def fake_fan_out(embedder, reranker, targets, query, **kw):
        captured["targets"] = targets
        return _fake_results()

    monkeypatch.setattr(gs, "fan_out_search", fake_fan_out)
    monkeypatch.setattr(gs, "get_shared_models", lambda settings: (None, None))

    owner = register(api_client, email="owner@example.com")
    ws_owner = _first_ws(api_client, owner)

    # Owner uploads a source (no worker needed; the row + collection are created).
    files = {"file": ("notes.txt", io.BytesIO(b"shared content"), "text/plain")}
    src = api_client.post(
        f"/api/workspaces/{ws_owner}/sources", headers=auth(owner), files=files
    ).json()

    # A different user with their own workspace.
    other = register(api_client, email="other@example.com")

    # Before sharing: 'other' only searches their own 1 workspace.
    api_client.post("/api/search", headers=auth(other), json={"query": "x"})
    assert len(captured["targets"]) == 1

    # Owner shares the source to 'other'.
    r = api_client.post(
        f"/api/workspaces/{ws_owner}/sources/{src['id']}/share",
        headers=auth(owner),
        json={"user_email": "other@example.com", "role": "viewer"},
    )
    assert r.status_code == 204, r.text

    # After sharing: 'other' searches own workspace + owner's (for the shared collection).
    captured.clear()
    api_client.post("/api/search", headers=auth(other), json={"query": "x"})
    targets = captured["targets"]
    assert len(targets) == 2
    # One target is constrained to the shared collection id.
    assert any(t.where and src["collection_id"] in t.where for t in targets)


def test_presets_crud_and_isolation(api_client):
    a = register(api_client, email="a@example.com")
    b = register(api_client, email="b@example.com")

    created = api_client.post(
        "/api/presets",
        headers=auth(a),
        json={"name": "Rome trip", "params": {"query": "rome", "modalities": ["text"]}},
    )
    assert created.status_code == 200, created.text
    pid = created.json()["id"]

    lst = api_client.get("/api/presets", headers=auth(a)).json()
    assert len(lst) == 1 and lst[0]["name"] == "Rome trip"
    assert lst[0]["params"]["query"] == "rome"

    # Another user cannot see or delete it.
    assert api_client.get("/api/presets", headers=auth(b)).json() == []
    assert api_client.delete(f"/api/presets/{pid}", headers=auth(b)).status_code == 404

    assert api_client.delete(f"/api/presets/{pid}", headers=auth(a)).status_code == 204
    assert api_client.get("/api/presets", headers=auth(a)).json() == []


def test_guides_endpoint(api_client):
    token = register(api_client)
    r = api_client.get("/api/guides", headers=auth(token))
    assert r.status_code == 200
    kinds = {g["kind"] for g in r.json()}
    assert {"telegram", "whatsapp", "json_chat"} <= kinds
