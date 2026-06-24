"""Tests for the service layer: filter compilation, prefilter, workspace paths."""

import numpy as np

from telesearch.config import Settings
from telesearch.index.store import VectorStore
from telesearch.models import Chunk
from telesearch.service import RequestContext, SearchQuery, build_where


# --------------------------------------------------------------------------- #
# build_where
# --------------------------------------------------------------------------- #
def test_build_where_empty_is_none():
    assert build_where(SearchQuery(text="x")) is None


def test_build_where_dates_and_modalities():
    where = build_where(
        SearchQuery(text="x", date_from=100, date_to=200, modalities=["text", "image"])
    )
    assert "timestamp >= 100" in where
    assert "timestamp <= 200" in where
    assert "modality IN ('text', 'image')" in where


def test_build_where_escapes_quotes():
    where = build_where(SearchQuery(text="x", senders=["O'Brien"]))
    assert "sender IN ('O''Brien')" in where


def test_build_where_scope_intersection():
    ctx = RequestContext(collections=["a", "b"])
    # Query asks for a and c; only a is allowed -> scope is just a.
    where = build_where(SearchQuery(text="x", collections=["a", "c"]), ctx)
    assert "collection_id IN ('a')" in where


def test_build_where_scope_disjoint_matches_none():
    ctx = RequestContext(collections=["a"])
    where = build_where(SearchQuery(text="x", collections=["z"]), ctx)
    assert where == "collection_id IN ()"


# --------------------------------------------------------------------------- #
# Store prefilter end-to-end (no models)
# --------------------------------------------------------------------------- #
def _chunk(cid, mid, content, *, collection_id="", source_kind="telegram", ts=None):
    return Chunk(
        chunk_id=cid,
        message_id=mid,
        chat="c",
        sender="Alice",
        timestamp=ts if ts is not None else mid * 100,
        date_str="2024-01-01",
        modality="text",
        content=content,
        collection_id=collection_id,
        source_kind=source_kind,
    )


def test_hybrid_search_prefilters_by_collection(tmp_path):
    dim = 8
    store = VectorStore(tmp_path / "db", dim)
    rng = np.random.default_rng(0)
    chunks = [
        _chunk("1:text", 1, "sunset over the ocean beach", collection_id="A"),
        _chunk("2:text", 2, "sunset over the ocean beach", collection_id="B"),
    ]
    vecs = rng.standard_normal((2, dim)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    store.add([c.to_row() for c in chunks], vecs)
    store.build_fts()

    rows = store.hybrid_search("sunset beach", vecs[0], k=5, where="collection_id = 'A'")
    assert {r["chunk_id"] for r in rows} == {"1:text"}
    assert rows[0]["collection_id"] == "A"


def test_chunk_roundtrip_preserves_new_fields(tmp_path):
    store = VectorStore(tmp_path / "db", 8)
    c = _chunk("1:text", 1, "hello", collection_id="X", source_kind="whatsapp")
    vec = np.ones((1, 8), dtype="float32")
    store.add([c.to_row()], vec)
    row = store.table.to_arrow().to_pylist()[0]
    assert row["collection_id"] == "X"
    assert row["source_kind"] == "whatsapp"


# --------------------------------------------------------------------------- #
# Workspace isolation paths
# --------------------------------------------------------------------------- #
def test_workspace_db_path_default_is_legacy():
    s = Settings(data_dir="/tmp/data")
    assert s.workspace_db_path("default") == s.db_path
    assert s.workspace_db_path() == s.db_path


def test_workspace_db_path_isolated_per_workspace():
    s = Settings(data_dir="/tmp/data")
    p = s.workspace_db_path("alice")
    assert p != s.db_path
    assert "workspaces" in str(p) and "alice" in str(p)
